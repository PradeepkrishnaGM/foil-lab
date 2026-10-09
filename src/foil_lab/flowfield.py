"""Off-body velocity field reconstruction for streamline visualization.

Rebuilds the velocity (U, V) on a grid from the solution's own singularity model: the core's
nodal vortex strengths (``solution.gamma``), any known panel sources (``solution.sources``,
the VII transpiration) and, for an open trailing edge, the trailing-edge panel, all on the panel
set the solver used (``solution.panels``; for ground effect that includes the image body).
The velocity is therefore the one the solver's boundary conditions hold.
"""

from __future__ import annotations

import numpy as np

from . import inviscid as core
from .solver import SolutionData

#: Grid points evaluated per block (bounds the (points x panels) work arrays).
_CHUNK = 20000


def inside_bodies(ps: core.PanelSet, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for points inside any body (even-odd ray casting on each closed node polygon)."""
    inside = np.zeros(x.shape, dtype=bool)
    for sl in ps.body_nodes:
        px, py = ps.node_x[sl], ps.node_y[sl]
        xa, ya, xb, yb = px, py, np.roll(px, -1), np.roll(py, -1)   # closes an open TE too
        crosses = ((ya[None, :] > y[:, None]) != (yb[None, :] > y[:, None]))
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = xa[None, :] + (y[:, None] - ya[None, :]) * (xb - xa)[None, :] / (yb - ya)[None, :]
        inside ^= (np.sum(crosses & (x[:, None] < xi), axis=1) % 2).astype(bool)
    return inside


def compute_velocity_field(
    solution: SolutionData,
    x_bounds: tuple = (-0.5, 1.5),
    y_bounds: tuple = (-0.5, 0.5),
    grid_res: int = 60,
):
    """Return (X, Y, U, V) on a grid around the solved geometry (zero inside bodies)."""
    if solution.panels is None:
        raise ValueError("solution has no panel set; solve it with foil_lab.solver.solve_flow")
    ps = solution.panels
    x_vals = np.linspace(x_bounds[0], x_bounds[1], grid_res)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    Xf, Yf = X.ravel(), Y.ravel()

    a = np.deg2rad(solution.alpha_deg)
    U = np.full(Xf.shape, np.cos(a))
    V = np.full(Xf.shape, np.sin(a))
    sig = solution.sources if np.any(solution.sources) else None
    for i in range(0, Xf.size, _CHUNK):
        sl = slice(i, i + _CHUNK)
        u, v = core.induced_velocity(ps, solution.gamma, Xf[sl], Yf[sl], source_strength=sig)
        U[sl] += u
        V[sl] += v

    inside = inside_bodies(ps, Xf, Yf)
    U[inside] = 0.0
    V[inside] = 0.0
    return X, Y, U.reshape(X.shape), V.reshape(Y.shape)
