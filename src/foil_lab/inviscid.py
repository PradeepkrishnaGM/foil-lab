"""Inviscid panel method: linear-strength vortex panels, one assembly for every mode.

Formulation
-----------
Every body is a closed or nearly closed polygon of N panels with N + 1 nodes. The vortex sheet
strength varies linearly along each panel between nodal values gamma_j, so a body has N + 1
unknowns. Boundary conditions:

* zero normal velocity relative to the surface at the N panel midpoints (Neumann condition;
  a prescribed wall-normal velocity, for example a transpiration velocity, may be added);
* one Kutta condition per body, ``gamma_first + gamma_last = 0`` at the trailing edge.

There are no unknown sources. Each body's nodes and its Kutta condition belong to that body
only ("each body closes on itself"); bodies are coupled through their induced velocities.

Open trailing edge: a body whose first and last nodes do not coincide gets one straight
trailing-edge panel across the gap, carrying a uniform source sigma_TE and a uniform vortex
gamma_TE tied to the two trailing-edge node strengths, so it adds no unknowns and the Kutta
condition is unchanged. From M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds
Number Airfoils", in T. J. Mueller (ed.), *Low Reynolds Number Aerodynamics*, Lecture Notes in
Engineering 54, Springer (1989), pp. 1-12: eq. (2) and Figure 1 on page 2 (panel strengths
sigma_TE = 1/2 (gamma_1 - gamma_N) |s x t|, gamma_TE = 1/2 (gamma_1 - gamma_N) |s . t|, with s
the unit vector bisecting the trailing-edge angle and t the unit vector along the panel), the
last term of eq. (3) on page 2 (how the panel enters the influence sum), and eq. (8) on page 3
(Kutta condition). Only the trailing-edge panel is taken from that paper; the boundary condition
here is the midpoint Neumann condition, not the nodal streamfunction condition of its eq. (7).
In the clockwise solver order, 1/2 (gamma_first - gamma_last) is the mean speed of the flow
leaving the trailing edge along s (the same expression as Drela's); the panel is oriented from
the last node to the first, and with the flow inside the body at rest the jump conditions across
the panel give sigma_TE = q (s . n) and gamma_TE = -q (s . t) in this module's sign convention,
whose magnitudes are those of eq. (2). See DECISIONS.md, M2.

References: J. L. Kuethe and C.-Y. Chow, *Foundations of Aerodynamics*, 5th ed., Wiley (1998),
section 5.10 (vortex panel method with linearly varying strength); J. Katz and A. Plotkin,
*Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press (2001), sections 10.3-10.4 (linear
vortex element) and 11.4.2 (linear-strength vortex method). The velocity induced by a
linear-strength vortex panel is evaluated in closed form below and checked against numerical
quadrature of the Biot-Savart integral in ``tests/test_inviscid_core.py``.

Conventions
-----------
* Input bodies are in any orientation; internally every body is ordered clockwise (trailing
  edge, lower surface, leading edge, upper surface, trailing edge), so the outward normal is the
  panel tangent rotated by +90 degrees. Results are returned in Selig order.
* Circulation is positive counter-clockwise. A lifting section has negative circulation.
* Velocities are scaled by the freestream speed; lengths are in the units of the input.

Array namespace
---------------
The assembly and solve are written once against an array namespace ``xp`` (``numpy`` or
``jax.numpy``) without in-place updates, so the NumPy and JAX paths run the same arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

_TWO_PI = 2.0 * np.pi


def _signed_area(c: np.ndarray) -> float:
    x, y = c[:, 0], c[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass(frozen=True)
class PanelSet:
    """Geometry of one or more bodies, internally clockwise, with the node bookkeeping.

    Panel arrays have length N (all bodies concatenated); node arrays have length
    M = sum(N_k + 1). ``start_node[j]`` and ``end_node[j]`` are the node indices of panel j.
    ``body_panels[k]`` and ``body_nodes[k]`` are slices into the panel and node arrays.
    ``reversed_input[k]`` is True when body k was given counter-clockwise (Selig order) and was
    reversed for the solver.
    """

    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    start_node: np.ndarray
    end_node: np.ndarray
    body_panels: tuple[slice, ...]
    body_nodes: tuple[slice, ...]
    reversed_input: tuple[bool, ...]

    @property
    def num_panels(self) -> int:
        return int(self.x1.size)

    @property
    def num_nodes(self) -> int:
        return int(self.node_x.size)

    @property
    def num_bodies(self) -> int:
        return len(self.body_panels)

    @property
    def lengths(self) -> np.ndarray:
        return np.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def theta(self) -> np.ndarray:
        return np.arctan2(self.y2 - self.y1, self.x2 - self.x1)

    @property
    def mid_x(self) -> np.ndarray:
        return 0.5 * (self.x1 + self.x2)

    @property
    def mid_y(self) -> np.ndarray:
        return 0.5 * (self.y1 + self.y2)

    @property
    def normal(self) -> tuple[np.ndarray, np.ndarray]:
        """Outward unit normal of every panel."""
        t = self.theta
        return -np.sin(t), np.cos(t)

    @property
    def tangent(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit tangent in the (clockwise) solver direction."""
        t = self.theta
        return np.cos(t), np.sin(t)

    @classmethod
    def from_bodies(cls, bodies: np.ndarray | Sequence[np.ndarray]) -> "PanelSet":
        """Build from one ``(n, 2)`` node array or a list of them (any orientation)."""
        if isinstance(bodies, np.ndarray) and bodies.ndim == 2:
            bodies = [bodies]
        xs1, ys1, xs2, ys2, nxs, nys, sn, en = [], [], [], [], [], [], [], []
        bp, bn, rev = [], [], []
        p0 = n0 = 0
        for b in bodies:
            c = np.asarray(b, dtype=float)
            if c.ndim != 2 or c.shape[1] != 2 or len(c) < 4:
                raise ValueError(f"each body must be an (n, 2) array with n >= 4, got {c.shape}")
            r = _signed_area(c) > 0.0
            if r:
                c = c[::-1]
            n = len(c) - 1
            xs1.append(c[:-1, 0]); ys1.append(c[:-1, 1]); xs2.append(c[1:, 0]); ys2.append(c[1:, 1])
            nxs.append(c[:, 0]); nys.append(c[:, 1])
            sn.append(n0 + np.arange(n)); en.append(n0 + np.arange(1, n + 1))
            bp.append(slice(p0, p0 + n)); bn.append(slice(n0, n0 + n + 1)); rev.append(bool(r))
            p0 += n
            n0 += n + 1
        cat = np.concatenate
        return cls(cat(xs1), cat(ys1), cat(xs2), cat(ys2), cat(nxs), cat(nys), cat(sn), cat(en),
                   tuple(bp), tuple(bn), tuple(rev))


def panel_velocity(xp: Any, px, py, x1, y1, x2, y2, self_mask=None):
    """Velocity at points (px, py) induced by linear-strength vortex panels, per unit nodal strength.

    Returns ``(ua, va, ub, vb)``, each of shape (points, panels): the global-frame velocity
    induced at every point by panel j when its start-node strength is 1 and its end-node strength
    is 0 (``ua, va``), and vice versa (``ub, vb``). Circulation is counter-clockwise positive.

    Derivation (panel-local frame, panel from (0, 0) to (L, 0), point (x, z), strength
    gamma(t) = gamma_a (1 - t/L) + gamma_b t/L, Biot-Savart for a vortex sheet):
        u = -1/(2 pi) int gamma(t) z / r^2 dt,   w = 1/(2 pi) int gamma(t) (x - t) / r^2 dt,
    with r^2 = (x - t)^2 + z^2 and
        I0u = int z / r^2 dt          = theta2 - theta1,
        I0w = int (x - t) / r^2 dt    = ln(r1 / r2),
        I1u = int t z / r^2 dt        = x I0u - z I0w,
        I1w = int t (x - t) / r^2 dt  = x I0w - L + z I0u,
    where theta1 = atan2(z, x), theta2 = atan2(z, x - L), r1 = |(x, z)|, r2 = |(x - L, z)|.

    ``self_mask`` (points, panels), if given, marks points that lie on the panel's own midpoint;
    there the limit z -> 0 on the outward side (+z for clockwise bodies) is used:
    I0u = pi, I0w = 0, I1u = pi L / 2, I1w = -L.
    """
    dx = px[:, None] - x1[None, :]
    dy = py[:, None] - y1[None, :]
    lx = x2 - x1
    ly = y2 - y1
    L = xp.sqrt(lx * lx + ly * ly)[None, :]
    c = (lx / L[0])[None, :]
    s = (ly / L[0])[None, :]
    x = dx * c + dy * s
    z = -dx * s + dy * c
    r1sq = x * x + z * z
    r2sq = (x - L) ** 2 + z * z
    i0u = xp.arctan2(z, x - L) - xp.arctan2(z, x)
    if self_mask is not None:
        safe1 = xp.where(self_mask, 1.0, r1sq)
        safe2 = xp.where(self_mask, 1.0, r2sq)
        i0w = xp.where(self_mask, 0.0, 0.5 * xp.log(safe1 / safe2))
        i0u = xp.where(self_mask, np.pi, i0u)
        zz = xp.where(self_mask, 0.0, z)
    else:
        i0w = 0.5 * xp.log(r1sq / r2sq)
        zz = z
    i1u = x * i0u - zz * i0w
    i1w = x * i0w - L + zz * i0u
    ua_l = -(i0u - i1u / L) / _TWO_PI
    ub_l = -(i1u / L) / _TWO_PI
    wa_l = (i0w - i1w / L) / _TWO_PI
    wb_l = (i1w / L) / _TWO_PI
    return (ua_l * c - wa_l * s, ua_l * s + wa_l * c,
            ub_l * c - wb_l * s, ub_l * s + wb_l * c)


def source_panel_velocity(xp: Any, px, py, x1, y1, x2, y2, self_mask=None):
    """Velocity at points (px, py) induced by constant-strength source panels of unit strength.

    Returns ``(u, v)``, each (points, panels), in the global frame. Panel-local frame as in
    :func:`panel_velocity`; a point source of strength m gives (u, w) = m (x - t, z) / (2 pi r^2),
    so for a constant strength sigma
        u = sigma / (2 pi) ln(r1 / r2),   w = sigma / (2 pi) (theta2 - theta1).
    On the panel's own midpoint (``self_mask``) the outward-side limit is u = 0, w = sigma / 2.
    Reference: Katz and Plotkin (2001), section 10.2 (constant-strength source element).
    """
    dx = px[:, None] - x1[None, :]
    dy = py[:, None] - y1[None, :]
    lx = x2 - x1
    ly = y2 - y1
    L = xp.sqrt(lx * lx + ly * ly)[None, :]
    c = (lx / L[0])[None, :]
    s = (ly / L[0])[None, :]
    x = dx * c + dy * s
    z = -dx * s + dy * c
    r1sq = x * x + z * z
    r2sq = (x - L) ** 2 + z * z
    dth = xp.arctan2(z, x - L) - xp.arctan2(z, x)
    if self_mask is not None:
        lg = xp.where(self_mask, 0.0, 0.5 * xp.log(xp.where(self_mask, 1.0, r1sq) /
                                                  xp.where(self_mask, 1.0, r2sq)))
        dth = xp.where(self_mask, np.pi, dth)
    else:
        lg = 0.5 * xp.log(r1sq / r2sq)
    u_l = lg / _TWO_PI
    w_l = dth / _TWO_PI
    return u_l * c - w_l * s, u_l * s + w_l * c


def _node_maps(ps: PanelSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static 0/1 matrices: panel start node (N, M), panel end node (N, M), Kutta rows (nb, M)."""
    n, m = ps.num_panels, ps.num_nodes
    p_start = np.zeros((n, m))
    p_end = np.zeros((n, m))
    p_start[np.arange(n), ps.start_node] = 1.0
    p_end[np.arange(n), ps.end_node] = 1.0
    kutta = np.zeros((ps.num_bodies, m))
    for k, sl in enumerate(ps.body_nodes):
        kutta[k, sl.start] = 1.0
        kutta[k, sl.stop - 1] = 1.0
    return p_start, p_end, kutta


@dataclass(frozen=True)
class TrailingEdgePanels:
    """Trailing-edge panels of the bodies with an open trailing edge (Drela 1989, eq. 2).

    Arrays have one entry per open body. ``coupling`` (n_te, M) gives the mean trailing-edge
    speed q = 1/2 (gamma_first - gamma_last) from the nodal strengths; the panel source is
    ``s_dot_n * q`` and its vortex ``-s_dot_t * q``.
    """

    body: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    s_dot_n: np.ndarray
    s_dot_t: np.ndarray
    coupling: np.ndarray

    @property
    def size(self) -> int:
        return int(self.body.size)

    @property
    def lengths(self) -> np.ndarray:
        return np.hypot(self.x2 - self.x1, self.y2 - self.y1)


#: A trailing-edge gap shorter than this fraction of the body's size counts as closed.
CLOSED_GAP_FRACTION = 1e-10


def trailing_edge_panels(ps: PanelSet) -> TrailingEdgePanels:
    """Trailing-edge panel geometry for every body whose first and last nodes differ."""
    rows = []
    for k, (sp, sn) in enumerate(zip(ps.body_panels, ps.body_nodes)):
        first, last = sn.start, sn.stop - 1
        xa, ya = ps.node_x[last], ps.node_y[last]
        xb, yb = ps.node_x[first], ps.node_y[first]
        size = max(np.ptp(ps.node_x[sn]), np.ptp(ps.node_y[sn]))
        gap = np.hypot(xb - xa, yb - ya)
        if gap <= CLOSED_GAP_FRACTION * size:
            continue
        # s: bisector of the trailing-edge angle, pointing downstream (Drela 1989, Fig. 1)
        j0, j1 = sp.start, sp.stop - 1
        t0 = np.array([ps.x2[j0] - ps.x1[j0], ps.y2[j0] - ps.y1[j0]])
        t1 = np.array([ps.x2[j1] - ps.x1[j1], ps.y2[j1] - ps.y1[j1]])
        sv = t1 / np.hypot(*t1) - t0 / np.hypot(*t0)
        sv = sv / np.hypot(*sv)
        tt = np.array([xb - xa, yb - ya]) / gap
        nn = np.array([-tt[1], tt[0]])          # outward normal, same rule as the body panels
        c = np.zeros(ps.num_nodes)
        c[first], c[last] = 0.5, -0.5
        rows.append((k, xa, ya, xb, yb, float(sv @ nn), float(sv @ tt), c))
    if not rows:
        e = np.zeros(0)
        return TrailingEdgePanels(np.zeros(0, dtype=int), e, e, e, e, e, e, np.zeros((0, ps.num_nodes)))
    k, xa, ya, xb, yb, dn, dt, c = zip(*rows)
    return TrailingEdgePanels(np.array(k), np.array(xa), np.array(ya), np.array(xb), np.array(yb),
                              np.array(dn), np.array(dt), np.stack(c))


def _te_velocity(xp: Any, te: TrailingEdgePanels, px, py):
    """Velocity at points per unit mean trailing-edge speed q of each TE panel: (u, v), (P, n_te)."""
    x1, y1, x2, y2 = (xp.asarray(a) for a in (te.x1, te.y1, te.x2, te.y2))
    su, sv = source_panel_velocity(xp, px, py, x1, y1, x2, y2)
    ua, va, ub, vb = panel_velocity(xp, px, py, x1, y1, x2, y2)
    dn, dt = xp.asarray(te.s_dot_n), xp.asarray(te.s_dot_t)
    return su * dn - (ua + ub) * dt, sv * dn - (va + vb) * dt


def influence_matrices(ps: PanelSet, xp: Any = np):
    """Normal and tangential velocity at every panel midpoint per unit nodal strength.

    Returns ``(an, at)``, each (N, M). Normal is outward, tangential along the clockwise
    solver direction, both evaluated on the outward side of the surface.
    """
    x1, y1, x2, y2 = (xp.asarray(a) for a in (ps.x1, ps.y1, ps.x2, ps.y2))
    mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    n = ps.num_panels
    ua, va, ub, vb = panel_velocity(xp, mx, my, x1, y1, x2, y2, self_mask=xp.asarray(np.eye(n, dtype=bool)))
    nx, ny = (xp.asarray(a) for a in ps.normal)
    tx, ty = (xp.asarray(a) for a in ps.tangent)
    p_start, p_end, _ = (xp.asarray(a) for a in _node_maps(ps))
    an = (ua * nx[:, None] + va * ny[:, None]) @ p_start + (ub * nx[:, None] + vb * ny[:, None]) @ p_end
    at = (ua * tx[:, None] + va * ty[:, None]) @ p_start + (ub * tx[:, None] + vb * ty[:, None]) @ p_end
    te = trailing_edge_panels(ps)
    if te.size:
        tu, tv = _te_velocity(xp, te, mx, my)
        cpl = xp.asarray(te.coupling)
        an = an + (tu * nx[:, None] + tv * ny[:, None]) @ cpl
        at = at + (tu * tx[:, None] + tv * ty[:, None]) @ cpl
    return an, at


@dataclass
class InviscidSolution:
    """Solution of one operating point, in solver (clockwise) order.

    ``gamma``: nodal vortex strengths (M); ``vt``: tangential surface velocity at the panel
    midpoints on the outward side, along the solver direction (N), taken from the sheet strength
    (see :func:`solve`); ``cp``: pressure coefficient at the midpoints (N); ``circulation``: per
    body, counter-clockwise positive; ``vt_induced``: the same velocity evaluated from the
    induced-velocity sum at the midpoints, kept as a consistency diagnostic.
    """

    panels: PanelSet
    alpha_deg: float
    gamma: Any
    vt: Any
    cp: Any
    circulation: Any
    vt_induced: Any = field(repr=False, default=None)
    matrix: Any = field(repr=False, default=None)


def solve(ps: PanelSet, alpha_deg: float, source_strength=None, xp: Any = np,
          keep_matrix: bool = False) -> InviscidSolution:
    """Solve the panel system for freestream angle ``alpha_deg`` (degrees, unit speed).

    ``source_strength`` (N), if given, is a known constant source strength per panel, used to
    represent a wall transpiration (outward normal velocity) of the same value: the equivalent
    source of M. J. Lighthill, "On displacement thickness", J. Fluid Mech. 4 (1958) 383-392.
    The boundary condition is then "outward normal velocity on the outer side = source strength",
    so the velocity inside the body remains zero.

    Because the flow inside each closed body is at rest, the tangential velocity just outside the
    sheet equals the local sheet strength (Katz and Plotkin 2001, section 11.4.2); the surface
    velocity at a panel midpoint is therefore taken as the mean of its two nodal strengths. On the
    Kármán-Trefftz test this read-out converges at second order, whereas the induced-velocity sum
    at the midpoints (kept as ``vt_induced``) converges at first order (DECISIONS.md, M2).
    """
    an, at = influence_matrices(ps, xp)
    _, _, kutta = _node_maps(ps)
    a = xp.deg2rad(alpha_deg) if xp is not np else np.deg2rad(alpha_deg)
    nx, ny = (xp.asarray(v) for v in ps.normal)
    tx, ty = (xp.asarray(v) for v in ps.tangent)
    vinf_n = xp.cos(a) * nx + xp.sin(a) * ny
    vinf_t = xp.cos(a) * tx + xp.sin(a) * ty
    rhs_n = -vinf_n
    vt_src = 0.0
    if source_strength is not None:
        sig = xp.asarray(source_strength)
        x1, y1, x2, y2 = (xp.asarray(v) for v in (ps.x1, ps.y1, ps.x2, ps.y2))
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        su, sv = source_panel_velocity(xp, mx, my, x1, y1, x2, y2,
                                       self_mask=xp.asarray(np.eye(ps.num_panels, dtype=bool)))
        sn = su * nx[:, None] + sv * ny[:, None]
        st = su * tx[:, None] + sv * ty[:, None]
        # outer-side normal velocity must equal sigma: (vinf + vortex + sources) . n = sigma
        rhs_n = sig - vinf_n - sn @ sig
        vt_src = st @ sig
    mat = xp.concatenate([an, xp.asarray(kutta)], axis=0)
    rhs = xp.concatenate([rhs_n, xp.zeros(ps.num_bodies)])
    gamma = xp.linalg.solve(mat, rhs)
    gs, ge = xp.asarray(ps.start_node), xp.asarray(ps.end_node)
    g_mid = 0.5 * (gamma[gs] + gamma[ge])
    vt = -g_mid                       # counter-clockwise sheet, clockwise solver direction
    vt_ind = vinf_t + at @ gamma + vt_src
    cp = 1.0 - vt * vt
    per_panel = g_mid * xp.asarray(ps.lengths)
    circ = xp.stack([xp.sum(per_panel[sl]) for sl in ps.body_panels])
    te = trailing_edge_panels(ps)
    if te.size:
        # the trailing-edge panel's vortex -(s . t) q over its length adds to the body circulation
        q = xp.asarray(te.coupling) @ gamma
        add = -xp.asarray(te.s_dot_t) * q * xp.asarray(te.lengths)
        circ = circ + xp.asarray(np.eye(ps.num_bodies)[:, te.body]) @ add
    return InviscidSolution(ps, alpha_deg, gamma, vt, cp, circ, vt_ind, mat if keep_matrix else None)


def induced_velocity(ps: PanelSet, gamma, px, py, source_strength=None, xp: Any = np):
    """Velocity induced by the panels at arbitrary points (freestream not included)."""
    px, py = xp.asarray(px), xp.asarray(py)
    x1, y1, x2, y2 = (xp.asarray(a) for a in (ps.x1, ps.y1, ps.x2, ps.y2))
    ua, va, ub, vb = panel_velocity(xp, px, py, x1, y1, x2, y2)
    g = xp.asarray(gamma)
    gs, ge = g[xp.asarray(ps.start_node)], g[xp.asarray(ps.end_node)]
    u, v = ua @ gs + ub @ ge, va @ gs + vb @ ge
    te = trailing_edge_panels(ps)
    if te.size:
        tu, tv = _te_velocity(xp, te, px, py)
        q = xp.asarray(te.coupling) @ g
        u, v = u + tu @ q, v + tv @ q
    if source_strength is not None:
        su, sv = source_panel_velocity(xp, px, py, x1, y1, x2, y2)
        sig = xp.asarray(source_strength)
        u, v = u + su @ sig, v + sv @ sig
    return u, v


@dataclass(frozen=True)
class BodyLoads:
    """Pressure-integrated force and moment coefficients of one body (reference chord c_ref).

    ``cm`` is about the reference point, nose-up positive.
    """

    cl: float
    cd: float
    cm: float


def body_loads(sol: InviscidSolution, body: int, ref_chord: float = 1.0,
               ref_point: tuple[float, float] = (0.25, 0.0)) -> BodyLoads:
    """Integrate the surface pressure of one body into lift, pressure drag and moment."""
    ps = sol.panels
    sl = ps.body_panels[body]
    cp = np.asarray(sol.cp)[sl]
    nx, ny = (v[sl] for v in ps.normal)
    ln = ps.lengths[sl]
    fx = -cp * nx * ln
    fy = -cp * ny * ln
    a = np.deg2rad(sol.alpha_deg)
    cl = (np.sum(fy) * np.cos(a) - np.sum(fx) * np.sin(a)) / ref_chord
    cd = (np.sum(fx) * np.cos(a) + np.sum(fy) * np.sin(a)) / ref_chord
    rx, ry = ps.mid_x[sl] - ref_point[0], ps.mid_y[sl] - ref_point[1]
    cm = -np.sum(rx * fy - ry * fx) / ref_chord ** 2
    return BodyLoads(float(cl), float(cd), float(cm))


def to_selig_order(sol: InviscidSolution, values, body: int) -> np.ndarray:
    """Panel values of one body (solver order) returned in the order of the input nodes."""
    sl = sol.panels.body_panels[body]
    v = np.asarray(values)[sl]
    return v[::-1] if sol.panels.reversed_input[body] else v
