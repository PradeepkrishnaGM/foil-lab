"""Empirical post-stall corrections.

The inviscid panel solver has no stall: its lift grows linearly without bound.
Once the boundary-layer solver flags significant separation, real lift collapses
and drag rises sharply. These empirical corrections clamp that behaviour so a
polar swept past stall stays physically plausible.

These are *engineering approximations*, not a separated-flow model. They are
intended to keep post-stall polars qualitatively correct (rounded Cl peak,
drag bucket opening up), not to predict exact max-lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def estimate_separation_from_cp(
    cp_min: float,
    cp_onset: float = -9.0,
    cp_full: float = -22.0,
) -> float:
    """Estimate a separated fraction (0..1) from the leading-edge suction peak.

    Rationale
    ---------
    An integral boundary-layer method with a flat-plate turbulent closure
    cannot reliably locate the turbulent separation point, so we use the
    accurately-computed inviscid suction peak as a stall proxy. Trailing-edge
    turbulent separation grows as the adverse pressure recovery (set by the
    suction-peak depth) increases. The mapping below is an empirical ramp:
    attached until ``cp_onset``, fully separated by ``cp_full``. The defaults
    place stall onset near the angles where moderate-thickness sections (e.g.
    NACA 24xx) are observed to stall (~14-16 deg). Calibrate per family.

    This is an engineering approximation for keeping post-stall polars
    plausible, not a separated-flow prediction.
    """
    if cp_min >= cp_onset:
        return 0.0
    frac = (cp_onset - cp_min) / (cp_onset - cp_full)
    return float(np.clip(frac, 0.0, 1.0))


@dataclass(frozen=True)
class PostStallResult:
    cl: float
    cd: float
    stalled: bool


def apply_post_stall(
    cl_attached: float,
    cd_attached: float,
    sep_fraction: float,
    sep_threshold: float = 0.30,
    cl_collapse: float = 1.2,
    cd_increment: float = 1.5,
) -> PostStallResult:
    """Reduce Cl and add a drag increment based on the separated fraction.

    Parameters
    ----------
    cl_attached, cd_attached : coefficients from the attached-flow solve.
    sep_fraction : fraction of a surface that is separated (0..1), from the
        boundary-layer solver.
    sep_threshold : separation fraction below which flow is treated as attached
        (small trailing-edge separation is normal and not yet stall).
    cl_collapse : how aggressively lift is shed per unit separation beyond the
        threshold (linear knock-down factor).
    cd_increment : peak added drag coefficient at full separation.

    Returns
    -------
    PostStallResult with corrected cl, cd, and a stalled flag.
    """
    if sep_fraction <= sep_threshold:
        return PostStallResult(cl=cl_attached, cd=cd_attached, stalled=False)

    # Excess separation beyond the threshold, normalized to [0, 1].
    excess = (sep_fraction - sep_threshold) / max(1e-6, 1.0 - sep_threshold)
    excess = min(1.0, max(0.0, excess))

    # Linearly shed lift and add a large drag increment.
    cl = cl_attached * max(0.0, 1.0 - cl_collapse * excess)
    cd = cd_attached + cd_increment * excess

    return PostStallResult(cl=cl, cd=cd, stalled=True)
