"""Inverse design and shape optimization.

Couples the PARSEC airfoil parameterization to a numerical optimizer so you can
ask for an airfoil that meets a target instead of guessing parameters by hand.
Three problem types:

* ``design_for_cl``    -- find a shape that produces a target Cl at a given alpha.
* ``maximize_ld``      -- maximize lift-to-drag ratio (uses the viscous solver).
* ``match_cp``         -- inverse design: recover a shape whose pressure
                          distribution matches a target Cp (the classic inverse
                          problem, posed as least-squares over PARSEC params).

How the gradients work
----------------------
The expensive part of every evaluation is the panel solve, which we run through
the **pre-compiled JAX inviscid core**: because the panel count is fixed during
an optimization, the JAX function compiles once and is reused for every
candidate geometry (~3 ms/eval instead of ~55 ms with recompilation). PARSEC has
only 11 parameters, so we obtain objective gradients by finite-differencing the
cheap PARSEC->solve chain (12 fast solves per gradient) and hand them to a
gradient-based optimizer (SLSQP), with bounds and geometry constraints
(thickness, trailing-edge closure) enforced directly.

For purely inviscid objectives (target Cl, match Cp) the JAX core is used (the NumPy core of
:mod:`foil_lab.inviscid` without JAX). For L/D, which needs viscous drag, the coupled
viscous-inviscid solver (:func:`foil_lab.viscous.solve_single`) is used.

Loads in this module are per unit chord about the quarter-chord point (0.25, 0): PARSEC sections
have their leading edge at the origin and a unit chord.

Trailing edge
-------------
Within one optimisation the trailing-edge type is fixed (``trailing_edge``). A closed section
(gap ``dy_te`` = 0) stays closed: the gap is held at zero and is not a design variable. An open
section has the gap as a design variable with the positive lower bound
``params.DESIGN.min_te_gap``. Reason: the inviscid core treats a closed edge with Drela's eq. (9)
and an open edge with a trailing-edge panel; at zero gap the two differ by a discretisation
amount (PARSEC start section at 5 deg: Cl jump 1.1e-3 at 100 panels, 1.4e-4 at 800; NACA 2412:
5e-5 at 100, 1.6e-6 at 640; DECISIONS.md M2). A finite-difference step across zero gap would see
that jump as a gradient of order 1e5.

Honesty
-------
* PARSEC spans a broad but not universal airfoil family; some exotic targets are
  simply not representable, and the optimizer will return the closest PARSEC fit.
* match_cp recovers *a* shape with a similar Cp, not necessarily the unique
  original; inverse problems can be ill-conditioned, so we regularize lightly
  toward the starting shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from . import inviscid as core
from . import viscous
from .generators import generate_parsec
from .params import DESIGN


def _inviscid(coords: np.ndarray, alpha_deg: float):
    """(panel set, inviscid solution) of one body at ``alpha_deg``."""
    ps = core.PanelSet.from_bodies([coords])
    return ps, core.solve(ps, alpha_deg)


def _viscous(coords: np.ndarray, alpha_deg: float, reynolds: float, fallback: bool) -> viscous.ViscousResult:
    """Coupled viscous solution of one body (a closed trailing edge opened as by ``fl.analyze``);
    ``fallback=False``: the direct start only, no continuation."""
    body, _ = viscous.open_closed_te(coords)
    return viscous.solve_single(body, float(alpha_deg), float(reynolds), viscous.VISCOUS, None,
                                fallback=fallback)


# Canonical PARSEC parameter order used as the optimization vector.
PARSEC_KEYS = ["rle", "x_up", "y_up", "yxx_up", "x_lo", "y_lo", "yxx_lo",
               "y_te", "dy_te", "ang_te", "ang_wedge"]

# Sensible default bounds for a subsonic airfoil (keeps the optimizer in
# physically meaningful, representable territory).
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "rle": (0.002, 0.05),
    "x_up": (0.2, 0.6), "y_up": (0.02, 0.15), "yxx_up": (-2.0, -0.1),
    "x_lo": (0.2, 0.6), "y_lo": (-0.12, 0.02), "yxx_lo": (0.1, 2.0),
    "y_te": (-0.02, 0.02), "dy_te": (0.0, 0.01),
    "ang_te": (-25.0, -1.0), "ang_wedge": (1.0, 25.0),
}


def default_start() -> Dict[str, float]:
    """A reasonable starting PARSEC airfoil (~NACA 2412-ish)."""
    return dict(rle=0.012, x_up=0.40, y_up=0.062, yxx_up=-0.55,
                x_lo=0.40, y_lo=-0.042, yxx_lo=0.55,
                y_te=0.0, dy_te=0.0, ang_te=-9.0, ang_wedge=9.0)


@dataclass
class DesignResult:
    params: Dict[str, float]
    coords: np.ndarray
    success: bool
    objective: float
    n_evals: int
    message: str
    history: List[float] = field(default_factory=list)
    n_failed: int = 0          # maximize_ld: candidates whose viscous solve did not converge


def _vec_to_params(x: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(PARSEC_KEYS, x)}


def _params_to_vec(p: Dict[str, float]) -> np.ndarray:
    return np.array([p[k] for k in PARSEC_KEYS], dtype=float)


def _bounds_list(bounds: Dict[str, Tuple[float, float]]):
    return [bounds[k] for k in PARSEC_KEYS]


@dataclass(frozen=True)
class _DesignSpace:
    """Design variables of one optimisation, with the trailing-edge type fixed."""

    keys: Tuple[str, ...]
    fixed: Dict[str, float]

    def params(self, x: np.ndarray) -> Dict[str, float]:
        p = dict(self.fixed)
        p.update({k: float(v) for k, v in zip(self.keys, x)})
        return {k: p[k] for k in PARSEC_KEYS}


def _design_space(start: Dict[str, float], bounds: Dict[str, Tuple[float, float]],
                  trailing_edge: Optional[str]):
    """(space, x0, bounds list). ``trailing_edge``: None (closed if the start gap is zero, else
    open), ``"closed"`` (gap held at 0) or ``"open"`` (gap >= ``DESIGN.min_te_gap``)."""
    if trailing_edge is None:
        trailing_edge = "closed" if start["dy_te"] == 0.0 else "open"
    if trailing_edge == "closed":
        space = _DesignSpace(tuple(k for k in PARSEC_KEYS if k != "dy_te"), {"dy_te": 0.0})
        b = dict(bounds)
    elif trailing_edge == "open":
        space = _DesignSpace(tuple(PARSEC_KEYS), {})
        lo, hi = bounds["dy_te"]
        b = dict(bounds, dy_te=(max(lo, DESIGN.min_te_gap), max(hi, DESIGN.min_te_gap)))
        start = dict(start, dy_te=max(start["dy_te"], b["dy_te"][0]))
    else:
        raise ValueError(f"trailing_edge must be None, 'closed' or 'open', got {trailing_edge!r}")
    x0 = np.array([start[k] for k in space.keys], dtype=float)
    return space, x0, [b[k] for k in space.keys]


def _inviscid_cp(coords: np.ndarray, alpha_deg: float):
    """(panel set, surface Cp at the panel midpoints in solver order)."""
    ps, sol = _inviscid(coords, alpha_deg)
    return ps, sol.cp


def _make_jax_cl(num_panels: int):
    """Return a function coords->(cl,cm) using the pre-compiled JAX core.

    Falls back to the NumPy inviscid solver if JAX is unavailable.
    """
    from .jax_backend import HAS_JAX
    if HAS_JAX:
        from . import jax_core

        def f(coords, alpha_deg):
            return jax_core.solve_numpy_compatible(coords, alpha_deg)
        return f

    def f(coords, alpha_deg):
        _, sol = _inviscid(coords, alpha_deg)
        b = core.body_loads(sol, 0)
        return b.cl, b.cm
    return f


def _thickness(coords: np.ndarray) -> float:
    return float(coords[:, 1].max() - coords[:, 1].min())


# --------------------------------------------------------------------------- #
# 1. Design for a target Cl
# --------------------------------------------------------------------------- #
def design_for_cl(target_cl: float, alpha_deg: float = 5.0,
                  num_panels: int = 120, min_thickness: float = 0.08,
                  max_thickness: float = 0.18,
                  start: Optional[Dict[str, float]] = None,
                  bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                  max_iter: int = 60, trailing_edge: Optional[str] = None) -> DesignResult:
    """Find a PARSEC airfoil giving ``target_cl`` at ``alpha_deg`` (inviscid).

    ``trailing_edge``: see the module docstring (default: keep the start's type).
    """
    start = start or default_start()
    bounds = bounds or DEFAULT_BOUNDS
    space, x0, blist = _design_space(start, bounds, trailing_edge)
    solve = _make_jax_cl(num_panels)
    history: List[float] = []
    n_eval = [0]

    def objective(x):
        n_eval[0] += 1
        coords = generate_parsec(space.params(x), num_panels)
        cl, _ = solve(coords, alpha_deg)
        obj = (cl - target_cl) ** 2
        history.append(obj)
        return obj

    cons = [
        {"type": "ineq", "fun": lambda x: _thickness(
            generate_parsec(space.params(x), num_panels)) - min_thickness},
        {"type": "ineq", "fun": lambda x: max_thickness - _thickness(
            generate_parsec(space.params(x), num_panels))},
    ]

    res = minimize(objective, x0, method="SLSQP",
                   bounds=blist, constraints=cons,
                   options={"maxiter": max_iter, "ftol": 1e-8})

    params = space.params(res.x)
    coords = generate_parsec(params, num_panels)
    final_cl, _ = solve(coords, alpha_deg)
    return DesignResult(
        params=params, coords=coords,
        success=abs(final_cl - target_cl) < 0.02,
        objective=float(res.fun), n_evals=n_eval[0],
        message=f"Cl={final_cl:.4f} (target {target_cl:.4f})", history=history)


# --------------------------------------------------------------------------- #
# 2. Maximize L/D (viscous)
# --------------------------------------------------------------------------- #
def maximize_ld(alpha_deg: float = 5.0, reynolds: float = 3e6,
                num_panels: int = 120, min_thickness: float = 0.08,
                max_thickness: float = 0.18, target_cl: Optional[float] = None,
                start: Optional[Dict[str, float]] = None,
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                max_iter: int = 30, trailing_edge: Optional[str] = None) -> DesignResult:
    """Maximize lift-to-drag ratio at fixed alpha and Reynolds.

    Drag source
    -----------
    Lift and drag come from the coupled viscous-inviscid solver (:func:`_viscous`, at
    ``reynolds``). During the search each candidate is solved from the direct start only (no
    continuation, ``fallback=False``): a candidate that does not converge from it counts as a
    failed candidate, is penalised and is never used for L/D (user, after report 12). The result
    reports how many failed (``n_failed``). The final design is solved with the full single-point
    logic (:func:`foil_lab.viscous.solve_single`); ``success`` needs it to converge.

    Constraint
    ----------
    If ``target_cl`` is given, lift is constrained to it (within 0.03) so the
    optimizer improves L/D by reducing drag at fixed lift, rather than just
    piling on camber. Strongly recommended for a meaningful result.

    ``trailing_edge``: see the module docstring (default: keep the start's type).
    """
    start = start or default_start()
    bounds = bounds or DEFAULT_BOUNDS
    space, x0, blist = _design_space(start, bounds, trailing_edge)
    history: List[float] = []
    n_eval = [0]

    n_failed = [0]

    def _solve(coords, final=False):
        try:
            s = _viscous(coords, alpha_deg, reynolds, fallback=final)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            s = None
        if s is None or not s.converged:
            if not final:
                n_failed[0] += 1
            return (np.nan, np.nan, False) if s is None else (float(s.cl), float(s.cd), False)
        return float(s.cl), float(s.cd), True

    def neg_ld(x):
        n_eval[0] += 1
        coords = generate_parsec(space.params(x), num_panels)
        cl, cd, conv = _solve(coords)
        # Guard against the optimizer exploiting non-converged cases (where
        # cd can collapse to a meaningless tiny value, faking a huge L/D) and
        # against an unphysically small drag floor.
        if not conv or not cd >= 1e-4:
            history.append(0.0)
            return 1e3              # large penalty -> optimizer avoids these
        ld = cl / cd
        history.append(ld)
        return -ld

    cons = [
        {"type": "ineq", "fun": lambda x: _thickness(
            generate_parsec(space.params(x), num_panels)) - min_thickness},
        {"type": "ineq", "fun": lambda x: max_thickness - _thickness(
            generate_parsec(space.params(x), num_panels))},
    ]
    if target_cl is not None:
        def _cl_margin(x):
            cl_x = _solve(generate_parsec(space.params(x), num_panels))[0]
            return 0.03 - abs(cl_x - target_cl) if np.isfinite(cl_x) else -1.0   # failed: violated
        cons.append({"type": "ineq", "fun": _cl_margin})

    res = minimize(neg_ld, x0, method="SLSQP",
                   bounds=blist, constraints=cons,
                   options={"maxiter": max_iter, "ftol": 1e-4})

    params = space.params(res.x)
    coords = generate_parsec(params, num_panels)
    cl, cd, conv = _solve(coords, final=True)
    return DesignResult(
        params=params, coords=coords, success=bool(res.success and conv),
        objective=float(cl / cd), n_evals=n_eval[0],
        message=f"L/D={cl/cd:.1f} (Cl={cl:.3f}, Cd={cd:.5f}, "
                f"viscous{'' if conv else ', NOT converged'}; {n_failed[0]} candidates failed)",
        history=history, n_failed=n_failed[0])


# --------------------------------------------------------------------------- #
# 3. Inverse design: match a target Cp distribution
# --------------------------------------------------------------------------- #
def match_cp(target_coords: np.ndarray, alpha_deg: float = 5.0,
             num_panels: int = 120,
             start: Optional[Dict[str, float]] = None,
             bounds: Optional[Dict[str, Tuple[float, float]]] = None,
             regularization: float = 1e-3,
             max_iter: int = 80, trailing_edge: Optional[str] = None) -> DesignResult:
    """Recover a PARSEC airfoil whose Cp matches that of ``target_coords``.

    The target Cp is computed once from the supplied target airfoil at the same
    alpha. We then minimize the mean-squared Cp difference (sampled at matching
    x/c) over PARSEC parameters, with light Tikhonov regularization toward the
    start to keep the ill-posed inverse problem well-behaved.

    ``trailing_edge``: see the module docstring (default: keep the start's type).
    """
    start = start or default_start()
    bounds = bounds or DEFAULT_BOUNDS
    space, x0, blist = _design_space(start, bounds, trailing_edge)
    history: List[float] = []
    n_eval = [0]

    # Target Cp sampled vs x on each surface, from the NumPy solver.
    # Sample the Cp (panel midpoints, solver order) at fixed x stations on each surface.
    xs = np.linspace(0.02, 0.98, 40)

    def sample_cp(ps, cp):
        cp = np.asarray(cp)
        mx, my = ps.mid_x, ps.mid_y
        up = my >= 0
        cu = np.interp(xs, mx[up][np.argsort(mx[up])], cp[up][np.argsort(mx[up])])
        lo = ~up
        cl_ = np.interp(xs, mx[lo][np.argsort(mx[lo])], cp[lo][np.argsort(mx[lo])])
        return cu, cl_

    tgt_cu, tgt_cl = sample_cp(*_inviscid_cp(target_coords, alpha_deg))

    def objective(x):
        n_eval[0] += 1
        coords = generate_parsec(space.params(x), num_panels)
        cu, cl_ = sample_cp(*_inviscid_cp(coords, alpha_deg))
        err = np.mean((cu - tgt_cu) ** 2) + np.mean((cl_ - tgt_cl) ** 2)
        reg = regularization * np.mean(((x - x0) / (np.abs(x0) + 1e-3)) ** 2)
        val = err + reg
        history.append(err)
        return val

    res = minimize(objective, x0, method="SLSQP",
                   bounds=blist,
                   options={"maxiter": max_iter, "ftol": 1e-9})

    params = space.params(res.x)
    coords = generate_parsec(params, num_panels)
    return DesignResult(
        params=params, coords=coords, success=res.success,
        objective=float(res.fun), n_evals=n_eval[0],
        message=f"final Cp MSE={history[-1]:.2e}", history=history)
