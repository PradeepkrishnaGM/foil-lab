"""Pressure integration into lift and pitching moments.

Kept separate from the solve so alternative post-processing (per-element load
breakdown, near-field vs. circulation lift) can be added without touching the
solver.

Validation
----------
With the Hess--Smith core and clockwise panels, the near-field pressure-integral
lift matches both thin-airfoil theory (~2*pi slope, correct zero-lift angle) and
the Kutta--Joukowski circulation lift. See ``tests/validation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panels import PanelGeometry


@dataclass(frozen=True)
class IntegratedLoads:
    cl: float
    cm_le: float
    cm_qc: float


def integrate_loads(
    geom: PanelGeometry,
    cp: np.ndarray,
    alpha_rad: float,
    body_index: int | None = None,
) -> IntegratedLoads:
    """Integrate Cp into Cl and pitching moments.

    body_index : if given, integrate over only that body's panels. This is the
        correct way to extract loads on the *real* body in a Method-of-Images
        ground-effect run (a mirror body is also present); a global integral
        would cancel the real and image contributions.
    """
    if body_index is None:
        sl = slice(0, geom.num_panels)
    else:
        sl = geom.body_slice(body_index)

    nx, ny = geom.nx[sl], geom.ny[sl]
    lengths = geom.lengths[sl]
    mid_x, mid_y = geom.mid_x[sl], geom.mid_y[sl]
    cp_b = cp[sl]

    # Pressure force per unit chord: dF = -Cp * n_hat * dS (outward normal).
    dfx = -cp_b * nx * lengths
    dfy = -cp_b * ny * lengths
    fx = np.sum(dfx)
    fy = np.sum(dfy)

    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
    cl = fy * cos_a - fx * sin_a

    # Moment about LE (x=0), nose-up positive, then transfer to c/4.
    cm_le = -np.sum(mid_x * dfy - mid_y * dfx)
    cm_qc = cm_le + 0.25 * cl

    return IntegratedLoads(cl=float(cl), cm_le=float(cm_le), cm_qc=float(cm_qc))
