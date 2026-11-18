"""Compressibility, sweep and finite-wing corrections, applied once each in a fixed order (M6).

Order (SPEC.md 4.7; DECISIONS.md M6), as :func:`foil_lab.analyze` applies it:

1. **Ground effect** is part of the two-dimensional flow problem (image method,
   :mod:`foil_lab.ground_effect`), not a correction of coefficients.
2. **Sweep** (simple sweep theory of the infinite swept wing; Busemann 1935, the independence
   principle; e.g. Schlichting & Truckenbrodt, *Aerodynamics of the Airplane*, 1979): the section
   is solved in the plane normal to the leading edge, with the normal components of the
   freestream, :func:`to_normal_plane`.
3. **Compressibility**: Kármán-Tsien on the surface pressure of that solution, at the normal Mach
   number (von Kármán, J. Aero. Sci. 8, 1941; Tsien, J. Aero. Sci. 6, 1939),
   :func:`karman_tsien_correction`. Inviscid only.
4. Back to the freestream: :func:`from_normal_plane`.
5. **Aspect ratio** (finite wing, elliptic loading): the lift of the infinite (swept) wing times
   Helmbold's factor, with the infinite wing's own lift slope (which already contains the
   compressibility and the sweep, so neither is applied twice); induced drag CL^2 / (pi e AR);
   :func:`finite_wing`. Helmbold, H. B., Jahrbuch der deutschen Luftfahrtforschung, 1942.

The critical Mach number (:func:`critical_mach`) is the freestream Mach number at which the
Kármán-Tsien-corrected minimum pressure coefficient reaches the critical (sonic) value; with
sweep it is that of the normal plane divided by cos(sweep).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .params import CORRECTIONS, CorrectionParams


# --------------------------------------------------------------------------------------- #
# Compressibility
# --------------------------------------------------------------------------------------- #
def karman_tsien_correction(cp_incompressible: np.ndarray, mach: float) -> np.ndarray:
    """Kármán-Tsien rule: Cp = Cp0 / (beta + M^2 / (1 + beta) * Cp0 / 2), beta = sqrt(1 - M^2).

    ``mach`` 0 returns the input unchanged. Raises ``ValueError`` for Mach < 0 or >= 1.
    """
    if mach == 0.0:
        return cp_incompressible
    if not 0.0 < mach < 1.0:
        raise ValueError(f"Mach number must be in [0, 1) for the Kármán-Tsien rule, got {mach!r}")
    m2 = mach ** 2
    beta = np.sqrt(1.0 - m2)
    cp0 = np.asarray(cp_incompressible)
    return cp0 / (beta + (m2 / (1.0 + beta)) * (cp0 / 2.0))


def critical_cp(mach: float, params: CorrectionParams = CORRECTIONS) -> float:
    """Pressure coefficient at which the local flow is sonic (isentropic):
    Cp* = 2 / (g M^2) [((2 + (g - 1) M^2) / (g + 1))^(g / (g - 1)) - 1]."""
    g = params.gamma
    m2 = mach ** 2
    return 2.0 / (g * m2) * (((2.0 + (g - 1.0) * m2) / (g + 1.0)) ** (g / (g - 1.0)) - 1.0)


def critical_mach(cp_min_incompressible: float, params: CorrectionParams = CORRECTIONS) -> float:
    """Freestream Mach number at which the Kármán-Tsien-corrected ``cp_min_incompressible`` equals
    the critical pressure coefficient. ``nan`` if the minimum pressure coefficient is not negative
    (no point of the surface is faster than the freestream)."""
    from scipy.optimize import brentq
    c0 = float(cp_min_incompressible)
    if not np.isfinite(c0) or c0 >= 0.0:
        return float("nan")

    def f(m):
        return float(karman_tsien_correction(np.array(c0), m)) - critical_cp(m, params)
    # f > 0 at low Mach (Cp* tends to minus infinity). The Kármán-Tsien denominator
    # beta + M^2/(1 + beta) Cp0/2 reaches zero at a higher Mach number for a suction peak (a pole of
    # the rule), so the root is the first sign change scanning up from low Mach, below that pole.
    ms = np.linspace(1e-4, 1.0 - 1e-9, 400)
    beta = np.sqrt(1.0 - ms ** 2)
    ok = beta + ms ** 2 / (1.0 + beta) * c0 / 2.0 > 0.0
    fs = np.array([f(m) if o else -1.0 for m, o in zip(ms, ok)])
    k = int(np.argmax(fs <= 0.0))
    if k == 0:
        return float(ms[0])
    return float(brentq(f, ms[k - 1], ms[k] if ok[k] else ms[k - 1] + (ms[k] - ms[k - 1]) * 0.999999,
                        xtol=1e-12))


# --------------------------------------------------------------------------------------- #
# Sweep (simple sweep theory)
# --------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NormalPlane:
    """Conditions in the plane normal to the leading edge of an infinite swept wing."""

    alpha: float          # deg
    mach: float
    reynolds: float       # based on the section chord and the normal velocity
    cos_sweep: float


def to_normal_plane(alpha: float, sweep: float, mach: float = 0.0, reynolds: float = 0.0,
                    params: CorrectionParams = CORRECTIONS) -> NormalPlane:
    """Simple sweep theory: the section (given normal to the leading edge, Reynolds number based on
    its chord and the freestream speed) sees the normal velocity V cos(sweep) at the angle
    alpha_n = atan(tan(alpha) / cos(sweep)), so M_n = M cos(sweep), Re_n = Re cos(sweep)."""
    if abs(sweep) > params.max_sweep:
        raise ValueError(f"sweep {sweep:g} deg is outside +-{params.max_sweep:g} deg "
                         "(params.CORRECTIONS.max_sweep)")
    c = float(np.cos(np.deg2rad(sweep)))
    a_n = float(np.rad2deg(np.arctan(np.tan(np.deg2rad(alpha)) / c)))
    return NormalPlane(a_n, mach * c, reynolds * c, c)


def from_normal_plane(cl_n: float, cd_n, cm_n: float, plane: NormalPlane) -> Tuple[float, object, float]:
    """Coefficients of the normal-plane solution referred to the freestream dynamic pressure: lift
    and moment act in the normal plane and scale with (V_n / V)^2 = cos^2(sweep); the normal-plane
    drag force has a streamwise component cos(sweep) of that, so Cd scales with cos^3(sweep) (the
    friction of the spanwise flow is not included). ``cd_n`` may be ``None`` (inviscid)."""
    c2 = plane.cos_sweep ** 2
    cd = None if cd_n is None else cd_n * c2 * plane.cos_sweep
    return cl_n * c2, cd, cm_n * c2


# --------------------------------------------------------------------------------------- #
# Finite wing
# --------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FiniteWing:
    cl: float
    cdi: float
    factor: float          # CL / Cl of the infinite wing
    lift_slope: float      # of the infinite wing, per radian


def helmbold_factor(lift_slope: float, aspect_ratio: float) -> float:
    """CL / Cl_infinite of an unswept elliptic wing (Helmbold 1942):
    1 / (sqrt(1 + k^2) + k), k = a / (pi AR), with ``lift_slope`` a per radian. Tends to the
    lifting-line value 1 / (1 + a / (pi AR)) at large aspect ratio."""
    k = lift_slope / (np.pi * aspect_ratio)
    return float(1.0 / (np.sqrt(1.0 + k * k) + k))


def finite_wing(cl_inf: float, lift_slope: float, aspect_ratio: float,
                params: CorrectionParams = CORRECTIONS) -> FiniteWing:
    """Finite-wing lift CL = F Cl_inf (F from :func:`helmbold_factor` with the infinite wing's lift
    slope) and induced drag CL^2 / (pi e AR). Lift and slope are those of the infinite wing after
    sweep and compressibility, so the factor does not contain the Mach number a second time. The
    profile drag and the moment about the quarter chord stay those of the section at the geometric
    angle (lifting-line approximation)."""
    if not aspect_ratio > 0.0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio!r}")
    f = helmbold_factor(lift_slope, aspect_ratio)
    cl = f * cl_inf
    return FiniteWing(cl, cl * cl / (np.pi * params.oswald * aspect_ratio), f, lift_slope)


def apply_3d_corrections(cl_2d: float, cm_2d: float, alpha_deg: float, aspect_ratio: float,
                         mach: float = 0.0, lift_slope: float = 2.0 * np.pi) -> tuple:
    """(CL, CDi, Cm) of :func:`finite_wing` for the v1.3.1-shaped interfaces (``solver.solve_flow``).

    ``aspect_ratio`` <= 0 means the section (2D) and returns the input. ``mach`` is not used: the
    section's own lift slope ``lift_slope`` (per radian) already contains the compressibility, which
    v1.3.1 applied a second time here (CODE_REPORT section 9 #5). ``alpha_deg`` is kept for the
    signature."""
    if aspect_ratio <= 0.0:
        return cl_2d, 0.0, cm_2d
    fw = finite_wing(cl_2d, lift_slope, aspect_ratio)
    return fw.cl, fw.cdi, cm_2d
