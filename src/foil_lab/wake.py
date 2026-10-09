"""Wake of a body for the coupled viscous solver: geometry and source operators (M3, C4).

Sources (DECISIONS.md M3, C4):

* Wake line: the inviscid streamline from the trailing-edge panel midpoint (LNE89 p. 8; FID
  p. 7). Its discretisation is FID's (p. 7): first point eps_w c = 1e-5 c downstream of the
  midpoint, N_w = floor(N/10 + 10 d_w) nodes (N airfoil nodes), spaced geometrically to a total
  length d_w c = 1 chord, traced with a predictor-corrector step. The first spacing equals the
  mean length of the two trailing-edge panels (our choice; FID does not state it).
* Wake sources (LNE89 pp. 7-8; FID pp. 3-4, eq. 2): each wake panel carries sigma = dm/ds and is
  split into two half-panels with a source strength varying linearly from the node values (mean
  of the neighbouring panel values) to the panel value at its midpoint, so the velocity at the
  wake nodes is finite. The node value at the first wake node is the sum of the first and last
  airfoil panel sources (FID p. 3); a half-panel of constant strength is added past the last node
  (FID p. 3).
* Linear-strength source velocities are the linear-strength vortex velocities of the core
  rotated by 90 degrees ((u, v)_source = (v, -u)_vortex for equal strength), checked against
  quadrature in ``tests/test_wake.py``.
* Edge velocity at a wake node: the component along the wake of freestream plus the velocity
  induced by all airfoil vortices and sources (including the trailing-edge panel) and the wake
  sources (LNE89 eq. 31, FID eq. 24).
* Dead-air thickness of an open trailing edge (FID eq. 15 = BTE89 eqs. 20-21):
  h_w = h_TE [1 + (2 + f_w dt/dxi) xi_w/L_w] (1 - xi_w/L_w)^2 for xi_w < L_w = 2.5 h_TE, with dt/dxi
  the trailing-edge thickness slope along the bisector from the two trailing-edge panels (our
  discrete evaluation), clipped as the closure set says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import inviscid

D_W = 1.0          # wake length in chords (FID Table 2)
EPS_W = 1e-5       # first wake point offset in chords (FID Table 2)
F_W = 2.5          # dead-air length / trailing-edge gap (FID Table 2, BTE89 eq. 21)


@dataclass
class Wake:
    body: int
    x: np.ndarray            # node coordinates (N_w)
    y: np.ndarray
    s: np.ndarray            # arc length from the trailing-edge midpoint
    tx: np.ndarray           # unit tangent at the nodes (downstream)
    ty: np.ndarray
    h_te: float              # trailing-edge gap
    dtdxi: float             # trailing-edge thickness slope along the bisector

    @property
    def n(self) -> int:
        return int(self.x.size)

    def dead_air(self, clip: float | None):
        """h_w at the nodes and h_w/h_TE (FID eq. 15, BTE89 eqs. 20-21)."""
        if self.h_te <= 0.0:
            return np.zeros(self.n), np.zeros(self.n)
        dt = self.dtdxi if clip is None else float(np.clip(self.dtdxi, -clip, clip))
        lw = F_W * self.h_te
        xb = np.minimum(self.s / lw, 1.0)
        hw = self.h_te * (1.0 + (2.0 + F_W * dt) * xb) * (1.0 - xb) ** 2
        return hw, hw / self.h_te


def _unit(u, v):
    n = np.hypot(u, v)
    return u / n, v / n


def build_wake(ps: inviscid.PanelSet, gamma_inv, alpha_deg: float, body: int = 0) -> Wake:
    """Wake line of one body: the inviscid streamline from the trailing-edge midpoint."""
    sn, sp = ps.body_nodes[body], ps.body_panels[body]
    f, last = sn.start, sn.stop - 1
    xa, ya = ps.node_x[last], ps.node_y[last]        # upper trailing-edge node (clockwise order)
    xb, yb = ps.node_x[f], ps.node_y[f]               # lower trailing-edge node
    xm, ym = 0.5 * (xa + xb), 0.5 * (ya + yb)
    chord = float(np.max(ps.node_x[sn]) - np.min(ps.node_x[sn]))
    lengths = np.asarray(ps.lengths)
    j0, j1 = sp.start, sp.stop - 1
    # bisector of the trailing-edge angle, downstream
    t0x, t0y = _unit(ps.x2[j0] - ps.x1[j0], ps.y2[j0] - ps.y1[j0])   # lower TE panel, upstream
    t1x, t1y = _unit(ps.x2[j1] - ps.x1[j1], ps.y2[j1] - ps.y1[j1])   # upper TE panel, downstream
    bx, by = _unit(t1x - t0x, t1y - t0y)
    # thickness slope along the bisector (n from lower to upper side)
    nx, ny = -by, bx
    dtdxi = float((t1x * nx + t1y * ny) / (t1x * bx + t1y * by)
                  - ((-t0x) * nx + (-t0y) * ny) / ((-t0x) * bx + (-t0y) * by))
    h_te = float(np.hypot(xa - xb, ya - yb))
    nw = int(np.floor(ps.num_nodes / 10 + 10 * D_W))
    d0 = 0.5 * (lengths[j0] + lengths[j1])
    total = D_W * chord
    r = brentq(lambda q: d0 * (q ** (nw - 1) - 1.0) / (q - 1.0) - total, 1.0 + 1e-9, 3.0)
    steps = d0 * r ** np.arange(nw - 1)
    a = np.deg2rad(alpha_deg)

    def vel(px, py):
        u, v = inviscid.induced_velocity(ps, gamma_inv, np.atleast_1d(px), np.atleast_1d(py))
        return _unit(np.cos(a) + float(u[0]), np.sin(a) + float(v[0]))

    x = [xm + EPS_W * chord * bx]
    y = [ym + EPS_W * chord * by]
    for dk in steps:                     # predictor-corrector (Heun) along the streamline
        u1, v1 = vel(x[-1], y[-1])
        u2, v2 = vel(x[-1] + dk * u1, y[-1] + dk * v1)
        ux, uy = _unit(u1 + u2, v1 + v2)
        x.append(x[-1] + dk * ux)
        y.append(y[-1] + dk * uy)
    x, y = np.array(x), np.array(y)
    s = EPS_W * chord + np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    tx, ty = np.empty(nw), np.empty(nw)
    tx[0], ty[0] = _unit(x[1] - x[0], y[1] - y[0])
    tx[-1], ty[-1] = _unit(x[-1] - x[-2], y[-1] - y[-2])
    tx[1:-1], ty[1:-1] = _unit(x[2:] - x[:-2], y[2:] - y[:-2])
    return Wake(body, x, y, s, tx, ty, h_te, dtdxi)


def linear_source_velocity(px, py, x1, y1, x2, y2):
    """Velocity at points of linear-strength source panels: (ua, va, ub, vb), each (P, S), the
    contributions per unit strength at the panel start (a) and end (b). From the core's linear
    vortex panel rotated by 90 degrees."""
    ua, va, ub, vb = inviscid.panel_velocity(np, px, py, x1, y1, x2, y2)
    return va, -ua, vb, -ub


@dataclass
class WakeOperators:
    """Linear maps of the source parameters q = [sigma_airfoil (N), sigma_wake panels (N_w - 1)]."""

    seg: tuple                 # half-panel segments x1, y1, x2, y2
    s_start: np.ndarray        # segment start strength = s_start @ q
    s_end: np.ndarray
    sn_air: np.ndarray         # outward normal velocity at airfoil panel midpoints per q (N, nq)
    ut_wake: np.ndarray        # velocity along the wake at the wake nodes per q, sources only (N_w, nq)
    ut_gamma: np.ndarray       # velocity along the wake at the wake nodes per nodal gamma (N_w, M)
    ut_inf: np.ndarray         # freestream component along the wake at the wake nodes (N_w)


def wake_operators(ps: inviscid.PanelSet, wk: Wake, alpha_deg: float) -> WakeOperators:
    n_air = ps.num_panels
    nw = wk.n
    npw = nw - 1
    nq = n_air + npw
    sp = ps.body_panels[wk.body]
    # node values of the wake source distribution, as rows over q
    node = np.zeros((nw, nq))
    node[0, sp.start] = 1.0                       # sum of the first and last airfoil panel sources
    node[0, sp.stop - 1] = 1.0
    for k in range(1, nw - 1):
        node[k, n_air + k - 1] = 0.5
        node[k, n_air + k] = 0.5
    node[nw - 1, n_air + npw - 1] = 1.0
    mid = np.zeros((npw, nq))
    mid[np.arange(npw), n_air + np.arange(npw)] = 1.0
    xm, ym = 0.5 * (wk.x[:-1] + wk.x[1:]), 0.5 * (wk.y[:-1] + wk.y[1:])
    x1 = np.concatenate([wk.x[:-1], xm, [wk.x[-1]]])
    y1 = np.concatenate([wk.y[:-1], ym, [wk.y[-1]]])
    lx, ly = wk.x[-1] - xm[-1], wk.y[-1] - ym[-1]          # extension past the last node
    x2 = np.concatenate([xm, wk.x[1:], [wk.x[-1] + lx]])
    y2 = np.concatenate([ym, wk.y[1:], [wk.y[-1] + ly]])
    s_start = np.vstack([node[:-1], mid, node[-1:]])
    s_end = np.vstack([mid, node[1:], node[-1:]])
    seg = (x1, y1, x2, y2)
    # normal velocity at the airfoil panel midpoints
    ua, va, ub, vb = linear_source_velocity(ps.mid_x, ps.mid_y, *seg)
    nx, ny = ps.normal
    sn_seg_a = ua * nx[:, None] + va * ny[:, None]
    sn_seg_b = ub * nx[:, None] + vb * ny[:, None]
    sn_air = sn_seg_a @ s_start + sn_seg_b @ s_end
    # tangential velocity at the wake nodes; evaluated a tiny distance off the sheet, where the
    # tangential component is continuous (the strength is continuous at the nodes)
    off = 1e-9
    qx, qy = wk.x - off * wk.ty, wk.y + off * wk.tx
    ua, va, ub, vb = linear_source_velocity(qx, qy, *seg)
    t = (wk.tx[:, None], wk.ty[:, None])
    ut_wake = (ua * t[0] + va * t[1]) @ s_start + (ub * t[0] + vb * t[1]) @ s_end
    su, sv = inviscid.source_panel_velocity(np, qx, qy, ps.x1, ps.y1, ps.x2, ps.y2)
    ut_wake[:, :n_air] += su * t[0] + sv * t[1]
    gu, gv = inviscid.induced_velocity(ps, np.eye(ps.num_nodes), qx, qy)
    ut_gamma = gu * t[0] + gv * t[1]
    a = np.deg2rad(alpha_deg)
    ut_inf = np.cos(a) * wk.tx + np.sin(a) * wk.ty
    return WakeOperators(seg, s_start, s_end, sn_air, ut_wake, ut_gamma, ut_inf)
