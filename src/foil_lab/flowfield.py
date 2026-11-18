"""Off-body velocity field reconstruction for streamline visualization.

Rebuilds the velocity (U, V) on a grid from the solution's own singularity model: the core's
nodal vortex strengths (``solution.gamma``), any panel sources (``solution.sources``: the
displacement sources of a coupled viscous solution) and, for an open trailing edge, the
trailing-edge panel, all on the panel set the solver used (``solution.panels``; for ground effect
that includes the image body). For a viscous solution the wake sources are not included, so the
field differs from the solver's in the wake region; elsewhere the velocity is the one the
solver's boundary conditions hold.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class VelocityField:
    """Velocity on a grid (``u``, ``v`` per freestream speed; zero inside bodies) and the bodies."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    inside: np.ndarray
    bodies: list
    alpha: float              # freestream angle of the field (0 with ground: the section is pitched)


def velocity_field(geometry, alpha: float, mach: float = 0.0, sweep: float = 0.0,
                   ground_height=None, pivot=None, grid_res: int = 100, margin: float = 0.4,
                   num_panels=None) -> VelocityField:
    """Inviscid velocity field around ``geometry`` (``Airfoil``, ``Element`` or ``Configuration``)
    at ``alpha`` (normal-plane angle with ``sweep``; with ``ground_height`` the field of the pitched
    section and its image, freestream parallel to the ground at y = 0), from the same panel solution
    as ``fl.analyze`` (nodal vortex strengths and, for open trailing edges, the trailing-edge panel;
    the induced velocity of :func:`foil_lab.inviscid.induced_velocity`). The grid covers the real
    bodies with ``margin`` reference chords around them (above the ground with ground effect); the
    interior of every body is masked by ray casting. ``mach`` is not applied to the field (the
    Kármán-Tsien rule corrects surface pressure only). Viscous displacement is not included."""
    from .airfoil import Configuration
    from .api import _problem
    from .corrections import NormalPlane, to_normal_plane
    cfg = Configuration.of(geometry)
    plane = to_normal_plane(alpha, sweep) if sweep else NormalPlane(alpha, 0.0, 0.0, 1.0)
    pb = _problem(cfg, cfg, True, num_panels, plane.alpha, ground_height, pivot)
    ps = core.PanelSet.from_bodies(pb.bodies)
    sol = core.solve(ps, pb.alpha)
    real = pb.bodies[:pb.n_real]
    pts = np.vstack(real)
    c = pb.chord
    x0, x1 = pts[:, 0].min() - margin * c, pts[:, 0].max() + margin * c
    y0, y1 = pts[:, 1].min() - margin * c, pts[:, 1].max() + margin * c
    if ground_height is not None:
        y0 = 0.0
    nx = int(grid_res)
    ny = max(int(grid_res * (y1 - y0) / (x1 - x0)), 20)
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    Xf, Yf = X.ravel(), Y.ravel()
    a = np.deg2rad(pb.alpha)
    U = np.full(Xf.shape, np.cos(a))
    V = np.full(Xf.shape, np.sin(a))
    for i in range(0, Xf.size, _CHUNK):
        sl = slice(i, i + _CHUNK)
        u, v = core.induced_velocity(ps, np.asarray(sol.gamma), Xf[sl], Yf[sl])
        U[sl] += u
        V[sl] += v
    inside = inside_bodies(ps, Xf, Yf)
    U[inside] = 0.0
    V[inside] = 0.0
    return VelocityField(X, Y, U.reshape(X.shape), V.reshape(X.shape), inside.reshape(X.shape),
                         [np.asarray(b) for b in real], pb.alpha)
