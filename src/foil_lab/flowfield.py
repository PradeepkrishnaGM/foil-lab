"""Off-body velocity field reconstruction for streamline visualization.

Rebuilds the velocity (U, V) on a grid from the converged singularity strengths:
each panel carries its constant source strength plus the body's shared vortex.
Vectorized over all grid points per panel.

This is consistent with the Hess--Smith core: ``solution.sources`` holds the
per-panel source strengths and ``solution.gamma`` holds the shared vortex
strength (one entry per body). Strengths are in the solver's normalized units
(per 2*pi*V_inf), so the induced velocities are scaled accordingly.
"""

from __future__ import annotations

import numpy as np

from .panels import build_panels
from .solver import SolutionData

_TWO_PI = 2.0 * np.pi


def compute_velocity_field(
    solution: SolutionData,
    x_bounds: tuple = (-0.5, 1.5),
    y_bounds: tuple = (-0.5, 0.5),
    grid_res: int = 60,
):
    """Return (X, Y, U, V) on a grid around the solved geometry."""
    x_vals = np.linspace(x_bounds[0], x_bounds[1], grid_res)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    Xf, Yf = X.ravel(), Y.ravel()

    alpha_rad = np.deg2rad(solution.alpha_deg)
    U = np.full(Xf.shape, np.cos(alpha_rad))
    V = np.full(Xf.shape, np.sin(alpha_rad))

    geom = build_panels(solution.coords_list)
    sources = solution.sources                  # (N,)
    gammas = np.atleast_1d(solution.gamma)       # (num_bodies,) or (1,)

    # Vortex strength per panel: each body's panels share that body's gamma.
    gamma_per_panel = np.empty(geom.num_panels)
    for k in range(geom.num_bodies):
        gamma_per_panel[geom.body_slice(k)] = gammas[min(k, gammas.size - 1)]

    # Induced velocity: vectorize over grid points, loop panels (N small).
    for j in range(geom.num_panels):
        dx = Xf - geom.x1[j]
        dy = Yf - geom.y1[j]
        cos_t, sin_t = np.cos(geom.theta[j]), np.sin(geom.theta[j])

        x_loc = dx * cos_t + dy * sin_t
        y_loc = -dx * sin_t + dy * cos_t
        lj = geom.lengths[j]

        r1_sq = np.maximum(x_loc ** 2 + y_loc ** 2, 1e-10)
        r2_sq = np.maximum((x_loc - lj) ** 2 + y_loc ** 2, 1e-10)

        u_src = (0.5 * np.log(r1_sq / r2_sq)) / _TWO_PI
        v_src = np.arctan2(y_loc * lj, x_loc * (x_loc - lj) + y_loc ** 2) / _TWO_PI
        u_vtx, v_vtx = v_src, -u_src

        sj = sources[j]
        gj = gamma_per_panel[j]

        u_glob = (u_src * cos_t - v_src * sin_t) * sj + (u_vtx * cos_t - v_vtx * sin_t) * gj
        v_glob = (u_src * sin_t + v_src * cos_t) * sj + (u_vtx * sin_t + v_vtx * cos_t) * gj
        U += u_glob
        V += v_glob

    # Mask points inside any body (keep streamlines out of solids).
    for coords in solution.coords_list:
        xn, yn = coords[:, 0], coords[:, 1]
        inside = (Xf > xn.min()) & (Xf < xn.max()) & (Yf > yn.min()) & (Yf < yn.max())
        U[inside] = 0.0
        V[inside] = 0.0

    return X, Y, U.reshape(X.shape), V.reshape(Y.shape)
