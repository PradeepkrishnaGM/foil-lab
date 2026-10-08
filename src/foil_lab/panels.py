"""Panel geometry: turn ordered airfoil node lists into a discretized panel set.

This is the closest analogue to a "mesh" layer in a panel method. It is kept
strictly separate from singularity influence (``influence.py``) and from the
linear solve (``linear_system.py``) so that either of those can be swapped
without touching geometry handling.

Orientation convention
----------------------
Nodes are forced to a **clockwise (CW)** traversal, which is the native Selig
`.dat` convention (TE -> upper -> LE -> lower -> TE) and the ordering the
Hess--Smith influence kernel is derived for. NACA generators emit
counter-clockwise points, so those are flipped here. Getting this orientation
right is what makes the Kutta condition produce physically correct circulation
(a symmetric airfoil gives Cl = 0 at zero incidence); the legacy lumped-vortex
solver was insensitive to it and under-predicted lift regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np

CoordsLike = Union[np.ndarray, Sequence[np.ndarray]]


def _signed_area(coords: np.ndarray) -> float:
    """Shoelace signed area; positive => counter-clockwise node ordering."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ensure_cw(coords: np.ndarray) -> np.ndarray:
    """Return coords ordered clockwise (reverse if needed).

    Selig `.dat` files are already clockwise; NACA generators emit
    counter-clockwise points and get flipped. Clockwise is the orientation the
    Hess--Smith kernel expects.
    """
    return coords if _signed_area(coords) < 0.0 else coords[::-1].copy()


@dataclass(frozen=True)
class PanelGeometry:
    """Vectorized panel geometry for one or more disjoint bodies.

    All arrays are concatenated across bodies in body order; ``body_counts``
    records how many panels belong to each body so callers can slice per-body
    (e.g. one Kutta condition per body, per-body force integration).
    """

    x1: np.ndarray          # panel start-node x
    y1: np.ndarray          # panel start-node y
    x2: np.ndarray          # panel end-node x
    y2: np.ndarray          # panel end-node y
    mid_x: np.ndarray       # collocation point x (panel midpoint)
    mid_y: np.ndarray       # collocation point y
    lengths: np.ndarray     # panel lengths
    theta: np.ndarray       # panel inclination angle (atan2 of the panel vector)
    nx: np.ndarray          # outward unit normal x
    ny: np.ndarray          # outward unit normal y
    tx: np.ndarray          # unit tangent x (direction of traversal)
    ty: np.ndarray          # unit tangent y
    body_counts: List[int]  # panels per body, in order

    @property
    def num_panels(self) -> int:
        return self.mid_x.size

    @property
    def num_bodies(self) -> int:
        return len(self.body_counts)

    def body_slice(self, k: int) -> slice:
        """Return the panel-index slice for body ``k``."""
        start = int(np.sum(self.body_counts[:k]))
        return slice(start, start + self.body_counts[k])


def as_body_list(coords: CoordsLike) -> List[np.ndarray]:
    """Normalize a single coordinate array or a list of them to a list."""
    if isinstance(coords, np.ndarray):
        return [coords]
    return list(coords)


def build_panels(coords: CoordsLike) -> PanelGeometry:
    """Construct :class:`PanelGeometry` from one body or a list of bodies.

    Vectorized end-to-end: no per-panel Python loop. Each body is forced CCW
    so that the resulting normals are outward.
    """
    bodies = [ensure_cw(np.asarray(c, dtype=float)) for c in as_body_list(coords)]

    x1 = np.concatenate([b[:-1, 0] for b in bodies])
    y1 = np.concatenate([b[:-1, 1] for b in bodies])
    x2 = np.concatenate([b[1:, 0] for b in bodies])
    y2 = np.concatenate([b[1:, 1] for b in bodies])
    body_counts = [b.shape[0] - 1 for b in bodies]

    dx = x2 - x1
    dy = y2 - y1
    lengths = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    # Outward normal for CW traversal (verified by point-in-polygon test:
    # rotating the tangent by +90 deg points away from the body), unit tangent
    # along traversal.
    nx, ny = -np.sin(theta), np.cos(theta)
    tx, ty = np.cos(theta), np.sin(theta)

    mid_x = x1 + 0.5 * dx
    mid_y = y1 + 0.5 * dy

    return PanelGeometry(
        x1=x1, y1=y1, x2=x2, y2=y2,
        mid_x=mid_x, mid_y=mid_y,
        lengths=lengths, theta=theta,
        nx=nx, ny=ny, tx=tx, ty=ty,
        body_counts=body_counts,
    )
