"""Singularity influence coefficients (the swappable numerical kernel).

Isolates *the physics of one panel inducing velocity at another* so the linear
solver and force integrator never change when the element model changes.

Operator layout for multi-element support
------------------------------------------
A kernel returns four matrices plus a vortex column-assembly map:

  S_n, S_t : (N, N)  constant-source normal/tangential influence (column j =
             source on panel j).
  V_n, V_t : (N, M)  vortex influence, where M is the number of vortex DOFs of a
             single body's representation: M = N for a per-node (linear) vortex
             distribution. Column m is the influence of vortex-DOF m.

The linear-system layer assigns an independent vortex distribution and Kutta
condition PER BODY by slicing these per-body blocks, which is what genuinely
disjoint multi-element cases require.

Models
------
* :class:`LinearVortexModel` (default) -- constant source + linearly varying
  (nodal, continuous) vorticity. Smooth surface-velocity gradients, no
  panel-to-panel vorticity jump. This is the validated v0.5 core.
* :class:`ConstantVortexModel` -- constant source + constant (per-panel) vortex.
  Lower order; provided for comparison and speed.

Both expect clockwise node ordering (native Selig `.dat`); :mod:`panels`
delivers that. All coefficients are built with NumPy broadcasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .panels import PanelGeometry


@dataclass(frozen=True)
class InfluenceCoeffs:
    """Source and vortex influence operators for the panel set.

    S_n, S_t : (N, N) constant-source normal/tangential influence.
    V_n, V_t : (N, V) vortex normal/tangential influence (V vortex DOFs).
    vortex_is_nodal : True if the vortex DOFs are nodal (linear distribution),
        meaning a single body of n panels has n independent nodal strengths with
        a wrap (closed-body) coupling; False for one constant vortex per panel.
    """

    S_n: np.ndarray
    S_t: np.ndarray
    V_n: np.ndarray
    V_t: np.ndarray
    vortex_is_nodal: bool


class InfluenceModel(Protocol):
    def compute(self, geom: PanelGeometry) -> InfluenceCoeffs:  # pragma: no cover
        ...


def _hess_smith_terms(geom: PanelGeometry):
    """Kuethe & Chow constant-source / linear-vortex sub-coefficients.

    Returns Cn1, Cn2, Ct1, Ct2, each (N, N).
    """
    x, y = geom.x1, geom.y1
    xc, yc = geom.mid_x, geom.mid_y
    theta = geom.theta
    S = geom.lengths
    n = geom.num_panels

    sin_j = np.sin(theta)[None, :]
    cos_j = np.cos(theta)[None, :]
    th_i = theta[:, None]
    th_j = theta[None, :]
    Sj = S[None, :]

    dxi = xc[:, None] - x[None, :]
    dyi = yc[:, None] - y[None, :]

    A = -dxi * cos_j - dyi * sin_j
    B = dxi ** 2 + dyi ** 2
    C = np.sin(th_i - th_j)
    D = np.cos(th_i - th_j)
    E = dxi * sin_j - dyi * cos_j

    eye = np.eye(n, dtype=bool)
    B_safe = np.where(eye, 1.0, B)
    Sj_safe = np.where(eye, 1.0, Sj)

    F = np.log(np.where(eye, 1.0, 1.0 + (Sj ** 2 + 2.0 * A * Sj) / B_safe))
    G = np.where(eye, 0.0, np.arctan2(E * Sj, B + A * Sj))

    P = dxi * np.sin(th_i - 2.0 * th_j) + dyi * np.cos(th_i - 2.0 * th_j)
    Qd = dxi * np.cos(th_i - 2.0 * th_j) - dyi * np.sin(th_i - 2.0 * th_j)

    Cn2 = D + 0.5 * Qd * F / Sj_safe - (A * C + D * E) * G / Sj_safe
    Cn1 = 0.5 * D * F + C * G - Cn2
    Ct2 = C + 0.5 * P * F / Sj_safe + (A * D - C * E) * G / Sj_safe
    Ct1 = 0.5 * C * F - D * G - Ct2

    Cn1[eye], Cn2[eye] = -1.0, 1.0
    Ct1[eye], Ct2[eye] = 0.5 * np.pi, 0.5 * np.pi
    return Cn1, Cn2, Ct1, Ct2


def _assemble_source(Cn1, Cn2, Ct1, Ct2):
    """Constant-source operator: column j is the source strength of panel j.

    Source normal influence S_n[i,j] = Cn1[i,j] (its own panel) accumulated with
    Cn2 from the shared-node neighbour, matching the validated assembly.
    """
    n = Cn1.shape[0]
    S_n = np.zeros((n, n))
    S_t = np.zeros((n, n))
    S_n[:, 0] = Cn1[:, 0]
    S_t[:, 0] = Ct1[:, 0]
    S_n[:, 1:n] = Cn1[:, 1:n] + Cn2[:, 0:n - 1]
    S_t[:, 1:n] = Ct1[:, 1:n] + Ct2[:, 0:n - 1]
    # wrap last source contribution onto node 0 column for a closed body
    S_n[:, 0] += Cn2[:, n - 1]
    S_t[:, 0] += Ct2[:, n - 1]
    return S_n, S_t


class LinearVortexModel:
    """Constant source + linearly varying (nodal, continuous) vortex."""

    def compute(self, geom: PanelGeometry) -> InfluenceCoeffs:
        Cn1, Cn2, Ct1, Ct2 = _hess_smith_terms(geom)
        n = geom.num_panels
        S_n, S_t = _assemble_source(Cn1, Cn2, Ct1, Ct2)

        # Nodal vortex: the vortex sign convention is the 90-degree rotation of
        # the source contributions. Node k accumulates from the panel starting
        # at k (Cn1/Ct1) and the panel ending at k (Cn2/Ct2), with wrap.
        V_n = np.zeros((n, n))
        V_t = np.zeros((n, n))
        VN1, VN2 = -Ct1, -Ct2     # vortex normal = -(source tangential parts)
        VT1, VT2 = Cn1, Cn2       # vortex tangential = +(source normal parts)
        for j in range(n):
            jp = (j + 1) % n
            V_n[:, j] += VN1[:, j]
            V_n[:, jp] += VN2[:, j]
            V_t[:, j] += VT1[:, j]
            V_t[:, jp] += VT2[:, j]
        return InfluenceCoeffs(S_n=S_n, S_t=S_t, V_n=V_n, V_t=V_t,
                               vortex_is_nodal=True)


class ConstantVortexModel:
    """Constant source + one constant vortex strength per panel (lower order)."""

    def compute(self, geom: PanelGeometry) -> InfluenceCoeffs:
        Cn1, Cn2, Ct1, Ct2 = _hess_smith_terms(geom)
        S_n, S_t = _assemble_source(Cn1, Cn2, Ct1, Ct2)
        # Per-panel constant vortex influence = 90-deg rotation of constant
        # source (sum of the two linear endpoint sub-coefficients).
        V_n = -(Ct1 + Ct2)
        V_t = (Cn1 + Cn2)
        return InfluenceCoeffs(S_n=S_n, S_t=S_t, V_n=V_n, V_t=V_t,
                               vortex_is_nodal=False)


DEFAULT_MODEL: InfluenceModel = LinearVortexModel()
