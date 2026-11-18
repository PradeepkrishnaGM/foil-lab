"""Off-body velocity field for streamline plots and far-field checks.

:func:`velocity_field` solves the same panel problem as ``fl.analyze`` and evaluates the velocity
on a grid around the geometry. :func:`field_on_points` is the evaluation itself: the freestream
plus the velocity induced by a panel solution's nodal vortex strengths, any panel sources and, for
an open trailing edge, the trailing-edge panel (:func:`foil_lab.inviscid.induced_velocity`), so the
field is the one the solver's boundary conditions hold. Viscous displacement is not included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import inviscid as core

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


def field_on_points(ps: core.PanelSet, gamma, alpha: float, x, y, sources=None):
    """Velocity (u, v) per freestream speed at the points (``x``, ``y``): the freestream at
    ``alpha`` (deg) plus the velocity induced by the panel solution (``gamma``: nodal vortex
    strengths; ``sources``: optional panel source strengths) on ``ps``. Points inside bodies are
    not masked (see :func:`inside_bodies`)."""
    xf, yf = np.ravel(np.asarray(x, float)), np.ravel(np.asarray(y, float))
    a = np.deg2rad(alpha)
    u = np.full(xf.shape, np.cos(a))
    v = np.full(xf.shape, np.sin(a))
    g = np.asarray(gamma)
    for i in range(0, xf.size, _CHUNK):
        sl = slice(i, i + _CHUNK)
        du, dv = core.induced_velocity(ps, g, xf[sl], yf[sl], source_strength=sources)
        u[sl] += du
        v[sl] += dv
    shape = np.shape(x)
    return u.reshape(shape), v.reshape(shape)


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
    U, V = field_on_points(ps, sol.gamma, pb.alpha, X, Y)
    inside = inside_bodies(ps, X.ravel(), Y.ravel()).reshape(X.shape)
    U[inside] = 0.0
    V[inside] = 0.0
    return VelocityField(X, Y, U, V, inside, [np.asarray(b) for b in real], pb.alpha)
