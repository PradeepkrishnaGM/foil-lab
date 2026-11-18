"""Advanced flight-condition analyses built on the core panel solver.

Two features:

1. Constant wing-loading polar -- evaluate a sweep at fixed wing loading (W/S)
   instead of fixed Reynolds number, iterating speed/Re to convergence so the
   polar reflects a real trimmed flight condition.

2. Simple sweep theory -- estimate the aerodynamics of an infinite swept wing by
   running the 2D solver on the freestream component normal to the leading edge and
   recombining (foil_lab.corrections).

Both are thin wrappers around :func:`solver.solve_flow`; they add no per-panel
cost beyond the solves they request. With a Reynolds number each solve is the coupled
viscous-inviscid solver (one body, Mach 0 in this version), started from its starting march.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .solver import SolutionData, solve_flow

# Sea-level standard air, SI. Override via the function arguments if needed.
RHO_DEFAULT = 1.225      # kg/m^3
MU_DEFAULT = 1.81e-5     # Pa.s (dynamic viscosity)


# --------------------------------------------------------------------------- #
# 1. Constant wing-loading polar
# --------------------------------------------------------------------------- #
@dataclass
class WingLoadingPoint:
    alpha_deg: float
    cl: float
    cd: float
    cm_qc: float
    speed: float        # converged flight speed (m/s)
    reynolds: float     # converged chord Reynolds number
    converged: bool     # Re fixed point reached and the final viscous solve converged


def solve_constant_wing_loading(
    coords,
    alpha_deg: float,
    wing_loading: float,   # W/S in N/m^2
    chord: float,          # reference chord in metres
    mach: float = 0.0,
    aspect_ratio: float = 0.0,
    rho: float = RHO_DEFAULT,
    mu: float = MU_DEFAULT,
    re_init: float = 1.0e6,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> WingLoadingPoint:
    """Solve one alpha at fixed wing loading, iterating speed/Re to consistency.

    Physics of the fixed-point iteration
    -------------------------------------
    Level flight requires lift = weight per unit span share, i.e.

        0.5 * rho * V^2 * Cl = W/S        ->        V = sqrt( 2 (W/S) / (rho Cl) )

    But Cl depends (through the viscous solve) on Re = rho V c / mu, which
    depends on V. We iterate: guess Re -> solve for Cl -> update V -> update Re,
    until Re converges. If Cl <= 0 (no level-flight solution at this alpha) the
    point is returned unconverged with the last estimates. The point is converged only if the
    fixed point is reached and the final viscous solve converged (``SolutionData.converged``).
    """
    reynolds = float(re_init)
    speed = np.nan
    converged = False
    sol = None

    for _ in range(max_iter):
        sol = solve_flow(coords, alpha_deg, mach=mach, reynolds=reynolds,
                         aspect_ratio=aspect_ratio)
        cl = sol.cl
        if cl <= 1e-6:
            # No positive lift => cannot support weight at this alpha.
            return WingLoadingPoint(alpha_deg, cl, sol.cd_total, sol.cm_qc,
                                    np.nan, reynolds, False)

        speed = np.sqrt(2.0 * wing_loading / (rho * cl))
        new_re = rho * speed * chord / mu

        if abs(new_re - reynolds) <= tol * max(new_re, 1.0):
            reynolds = new_re
            converged = True
            break
        reynolds = new_re

    # Final solve at the converged Reynolds number for self-consistent drag.
    sol = solve_flow(coords, alpha_deg, mach=mach, reynolds=reynolds,
                     aspect_ratio=aspect_ratio)
    return WingLoadingPoint(alpha_deg, sol.cl, sol.cd_total, sol.cm_qc,
                            speed, reynolds, bool(converged and sol.converged))


def constant_wing_loading_sweep(
    coords,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    wing_loading: float,
    chord: float,
    **kwargs,
) -> List[WingLoadingPoint]:
    """Run :func:`solve_constant_wing_loading` across a range of alpha."""
    n = int(round((alpha_end - alpha_start) / alpha_step)) + 1
    alphas = np.linspace(alpha_start, alpha_end, n)
    return [
        solve_constant_wing_loading(coords, float(a), wing_loading, chord, **kwargs)
        for a in alphas
    ]


# --------------------------------------------------------------------------- #
# 2. Busemann simple-sweep approximation
# --------------------------------------------------------------------------- #
@dataclass
class SweptResult:
    sweep_deg: float
    alpha_eff_deg: float    # effective angle of attack seen in the normal plane
    cl_3d: float            # lift coefficient referred to the full freestream
    cd_3d: float
    cm_qc_3d: float
    normal_solution: SolutionData


def solve_swept_wing(
    coords,
    alpha_deg: float,
    sweep_deg: float,
    mach: float = 0.0,
    reynolds: float = 0.0,
    aspect_ratio: float = 0.0,
    v_inf: float = 1.0,
) -> SweptResult:
    """Simple sweep theory (:func:`foil_lab.corrections.to_normal_plane`, the same correction
    ``fl.analyze(..., sweep=...)`` applies): the section is solved in the plane normal to the leading
    edge at alpha_n = atan(tan(alpha) / cos(sweep)), Mach and Reynolds number times cos(sweep); lift
    and moment are referred to the freestream with cos^2(sweep), drag with cos^3(sweep).

    v1.3.1 solved the normal plane at the geometric alpha, which made the lift slope scale with
    cos^2(sweep) instead of cos(sweep) (DECISIONS.md M6).
    """
    from .corrections import from_normal_plane, to_normal_plane
    plane = to_normal_plane(alpha_deg, sweep_deg, mach, reynolds)
    sol_n = solve_flow(coords, plane.alpha, v_inf=v_inf, mach=plane.mach,
                       reynolds=plane.reynolds, aspect_ratio=aspect_ratio)
    cl, cd, cm = from_normal_plane(sol_n.cl, sol_n.cd_total, sol_n.cm_qc, plane)
    return SweptResult(
        sweep_deg=sweep_deg,
        alpha_eff_deg=plane.alpha,
        cl_3d=cl,
        cd_3d=cd,
        cm_qc_3d=cm,
        normal_solution=sol_n,
    )
