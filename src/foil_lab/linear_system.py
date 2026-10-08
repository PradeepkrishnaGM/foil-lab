"""Boundary-condition enforcement and linear-system assembly.

Decoupled from geometry and from the singularity kernel. Supports TRUE
multi-element analysis: each body gets

  * its own bound-circulation unknown (one shared vortex strength per body), and
  * its own Kutta condition (smooth flow-off at that body's trailing edge),

so a wing + separate flap is solved consistently rather than approximated with a
single global circulation. The single-body case is the one-body special case.

Unknowns
--------
  n source strengths (one per panel, all bodies concatenated)
  + nb circulation strengths (one per body)
  = n + nb total.

System
------
  Tangency rows (n): S_n @ sigma + V_n_body @ gamma = sin(theta - alpha)
  Kutta rows  (nb): per body, (S_t[first]+S_t[last]) . sigma
                    + per-body vortex tangential closure
                    = -(freestream tangential at that body's first+last panel)

where V_n_body collapses the kernel's per-node/per-panel vortex columns to one
column per body (every node/panel on a body shares that body's gamma).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .influence import InfluenceCoeffs
from .panels import PanelGeometry


@dataclass(frozen=True)
class FlowSolution:
    sources: np.ndarray     # (n,)
    gammas: np.ndarray      # (nb,) one bound circulation per body
    vt_total: np.ndarray    # (n,) tangential velocity at each collocation point


def _per_body_vortex_columns(V: np.ndarray, geom: PanelGeometry) -> np.ndarray:
    """Collapse vortex influence (N, V) to (N, nb): one column per body.

    Every vortex DOF belonging to a body shares that body's single circulation,
    so the body column is the row-wise sum of that body's vortex-DOF columns.
    For both nodal (V=N) and per-panel (V=N) layouts the DOFs map 1:1 to panels'
    bodies, so we slice by body panel ranges.
    """
    n_field = V.shape[0]
    cols = np.zeros((n_field, geom.num_bodies))
    for k in range(geom.num_bodies):
        sl = geom.body_slice(k)
        cols[:, k] = np.sum(V[:, sl], axis=1)
    return cols


def assemble_and_solve(
    geom: PanelGeometry,
    coeffs: InfluenceCoeffs,
    alpha_rad: float,
    v_inf: float = 1.0,
) -> FlowSolution:
    """Assemble the multi-body panel system and solve."""
    n = geom.num_panels
    nb = geom.num_bodies
    size = n + nb
    theta = geom.theta

    Vn_body = _per_body_vortex_columns(coeffs.V_n, geom)   # (n, nb)
    Vt_body = _per_body_vortex_columns(coeffs.V_t, geom)   # (n, nb)

    A = np.zeros((size, size))
    rhs = np.zeros(size)

    # --- Tangency rows (one per panel) -------------------------------------
    A[:n, :n] = coeffs.S_n
    A[:n, n:] = Vn_body
    rhs[:n] = np.sin(theta - alpha_rad)

    # --- Kutta rows (one per body) -----------------------------------------
    for k in range(nb):
        sl = geom.body_slice(k)
        first, last = sl.start, sl.stop - 1
        row = n + k
        A[row, :n] = coeffs.S_t[first, :] + coeffs.S_t[last, :]
        A[row, n:] = Vt_body[first, :] + Vt_body[last, :]
        rhs[row] = -(np.cos(theta[first] - alpha_rad)
                     + np.cos(theta[last] - alpha_rad))

    strengths = np.linalg.solve(A, rhs)
    sources = strengths[:n]
    gammas = strengths[n:]

    # --- Reconstruct tangential velocity -----------------------------------
    vt = np.cos(theta - alpha_rad) + coeffs.S_t @ sources + Vt_body @ gammas
    vt *= v_inf

    return FlowSolution(sources=sources, gammas=gammas, vt_total=vt)
