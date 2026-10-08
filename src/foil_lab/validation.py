"""Validation harness: compare Foil Lab against reference (XFOIL) polars.

Loads a reference polar (CSV with columns alpha,cl,cd,cm), runs Foil Lab across
the same angles, and reports per-point and aggregate errors. This is how we keep
the project honest: every claim about accuracy is reproducible from sourced data
and a script, not from memory.

The reference data shipped under ``tests/validation/data`` are XFOIL 6.96
predictions from airfoiltools.com (cited in each file header). XFOIL is itself a
viscous panel/BL code, so matching it is the most direct statement of "drop-in
alternative"; it is not wind-tunnel truth, but it is the de-facto reference the
target users already trust.

Usage
-----
    from foil_lab.validation import load_reference, validate_inviscid, ErrorReport
    ref = load_reference("naca2412_xfoil_re50k.csv")
    report = validate_inviscid(generate_naca4("2412", 200), ref)
    print(report.summary())
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .solver import solve_flow


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                        "tests", "validation", "data")


@dataclass
class ReferencePolar:
    name: str
    reynolds: float
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    source: str = ""


@dataclass
class ErrorReport:
    name: str
    alpha: np.ndarray
    cl_ref: np.ndarray
    cl_model: np.ndarray
    cd_ref: Optional[np.ndarray] = None
    cd_model: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    @property
    def cl_mae(self) -> float:
        return float(np.nanmean(np.abs(self.cl_model - self.cl_ref)))

    @property
    def cl_rmse(self) -> float:
        return float(np.sqrt(np.nanmean((self.cl_model - self.cl_ref) ** 2)))

    @property
    def cd_mae(self) -> Optional[float]:
        if self.cd_ref is None or self.cd_model is None:
            return None
        return float(np.nanmean(np.abs(self.cd_model - self.cd_ref)))

    def summary(self) -> str:
        lines = [f"Validation: {self.name}",
                 f"  Cl MAE  = {self.cl_mae:.4f}",
                 f"  Cl RMSE = {self.cl_rmse:.4f}"]
        if self.cd_mae is not None:
            # Relative drag error is the headline number for an XFOIL alternative.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.nanmean(np.abs(self.cd_model - self.cd_ref)
                                 / np.where(self.cd_ref > 0, self.cd_ref, np.nan))
            lines.append(f"  Cd MAE  = {self.cd_mae:.5f}")
            lines.append(f"  Cd mean rel. error = {rel*100:.1f}%")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def load_reference(filename: str) -> ReferencePolar:
    """Load a reference polar CSV from the validation data directory."""
    path = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
    alpha, cl, cd, cm = [], [], [], []
    source = ""
    reynolds = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                source += line.lstrip("# ") + " "
                if "Re=" in line:
                    try:
                        re_str = line.split("Re=")[1]
                        # Strip thousands separators and trailing tokens.
                        re_str = re_str.replace(",", "").split()[0].rstrip(",")
                        reynolds = float(re_str)
                    except Exception:
                        pass
                continue
            if line.lower().startswith("alpha"):
                continue
            if not line:
                continue
            parts = line.split(",")
            alpha.append(float(parts[0])); cl.append(float(parts[1]))
            cd.append(float(parts[2]))
            cm.append(float(parts[3]) if len(parts) > 3 else np.nan)
    return ReferencePolar(
        name=os.path.basename(path), reynolds=reynolds,
        alpha=np.array(alpha), cl=np.array(cl), cd=np.array(cd),
        cm=np.array(cm), source=source.strip())


def validate_inviscid(coords: np.ndarray, ref: ReferencePolar) -> ErrorReport:
    """Compare inviscid Cl to a reference polar (drag not modeled inviscidly)."""
    cl_model = np.array([solve_flow(coords, float(a)).cl for a in ref.alpha])
    return ErrorReport(
        name=f"{ref.name} [inviscid Cl]", alpha=ref.alpha,
        cl_ref=ref.cl, cl_model=cl_model,
        notes=["inviscid: lift only; expect over-prediction where the reference "
               "has strong viscous decambering (low Re / near stall)"])


def validate_vii(coords: np.ndarray, ref: ReferencePolar,
                 n_crit: float = 9.0, alpha_max: Optional[float] = None,
                 max_iter: int = 200) -> ErrorReport:
    """Compare VII Cl and Cd to a reference polar.

    Only angles up to ``alpha_max`` (if given) are evaluated; non-converged VII
    points are recorded as NaN so they don't silently corrupt the error metrics.
    """
    from .vii import solve_vii
    a_list, clm, cdm = [], [], []
    n_nonconv = 0
    for a in ref.alpha:
        if alpha_max is not None and a > alpha_max:
            continue
        a_list.append(a)
        try:
            r = solve_vii(coords, float(a), reynolds=ref.reynolds,
                          n_crit=n_crit, max_iter=max_iter, tol=1e-3)
            if r.converged:
                clm.append(r.solution.cl); cdm.append(r.cd_wake)
            else:
                clm.append(np.nan); cdm.append(np.nan); n_nonconv += 1
        except Exception:
            clm.append(np.nan); cdm.append(np.nan); n_nonconv += 1

    a_arr = np.array(a_list)
    mask = np.isin(ref.alpha, a_arr)
    notes = []
    if n_nonconv:
        notes.append(f"{n_nonconv}/{len(a_arr)} VII points did not converge "
                     f"(recorded as NaN, excluded from error)")
    return ErrorReport(
        name=f"{ref.name} [VII, Ncrit={n_crit}]", alpha=a_arr,
        cl_ref=ref.cl[mask], cl_model=np.array(clm),
        cd_ref=ref.cd[mask], cd_model=np.array(cdm), notes=notes)


def lift_curve_metrics(coords: np.ndarray, ref: ReferencePolar,
                       linear_alpha_max: float = 5.0) -> Dict[str, float]:
    """Compare inviscid lift-curve slope and zero-lift angle to a reference.

    Fits both over the (near-)linear range up to ``linear_alpha_max`` and
    returns slope (per deg) and alpha_L0 (deg) for each, plus the deltas. These
    are inviscid properties; large discrepancies at low Re reflect the
    reference's viscous decambering, not a core error.
    """
    mask = ref.alpha <= linear_alpha_max
    a = ref.alpha[mask]
    ref_fit = np.polyfit(a, ref.cl[mask], 1)
    ref_slope = ref_fit[0]
    ref_a0 = -ref_fit[1] / ref_slope

    model_cl = np.array([solve_flow(coords, float(ai)).cl for ai in a])
    m_fit = np.polyfit(a, model_cl, 1)
    m_slope = m_fit[0]
    m_a0 = -m_fit[1] / m_slope

    return {
        "ref_slope_per_deg": float(ref_slope),
        "model_slope_per_deg": float(m_slope),
        "ref_alpha_L0_deg": float(ref_a0),
        "model_alpha_L0_deg": float(m_a0),
        "slope_ratio": float(m_slope / ref_slope),
    }
