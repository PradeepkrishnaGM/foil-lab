"""Coupled viscous-inviscid solver: integral boundary layer and panel method, one Newton system.

Formulation (M3; DG87 = Drela & Giles 1987, LNE89 = Drela 1989, BTE89 = Drela AIAA-89-2200,
FID = Fidkowski; DECISIONS.md M3):

* Stations at the panel nodes, where the sheet strength is the surface speed: ue = -d gamma, with
  the direction factor d = +1 on the upper branch (flow along the clockwise solver direction) and
  -1 on the lower branch (LNE89 eq. 30; FID p. 9 with its sign convention); then the nodes of the
  wake of each body (:mod:`foil_lab.wake`; FID p. 7, LNE89 pp. 7-8), d = +1.
* Augmented state per station (FID p. 6 and eq. 27): theta, delta*, n (laminar) or C_tau
  (turbulent), and ue. Residuals per station: the three boundary-layer equations and the
  edge-velocity relation R_u = ue - ue_inv - D (ue delta*) over airfoil and wake (LNE89 eq. 33,
  FID eqs. 24-27); D is the exact response of our core to the airfoil and wake sources
  (:func:`build_coupling`), rebuilt for every angle of attack with the wake.
* Boundary-layer residuals: the interval equations of :mod:`foil_lab.bl.equations`, closures from
  the selected closure set (``ViscousParams.closure_set``). First interval of each branch: the
  stagnation-point equations (ue = K xi) applied to the state extrapolated linearly from the first
  two stations to xi = 0, K from a quadratic fit of ue through the stagnation point and those two
  stations (FID p. 7), and n = 0 at the first station.
* Wake: one layer (LNE89 p. 7; BTE89 p. 61). First wake station from the two trailing-edge
  states (FID eq. 14; BTE89 eqs. 16-18), written as ratios so that the residuals are
  dimensionless: theta_w/(theta_l + theta_u) - 1, delta*_w/(delta*_l + delta*_u + h_TE) - 1,
  (C_tau,w/C_tau,start)^1/2 - 1 with C_tau,start from the set (FID eq. 14 or BTE89 eq. 18); a
  laminar trailing-edge state enters with the set's C_tau at transition (FID p. 7). Wake intervals:
  the turbulent equations with the wake closures and the dead-air thickness (FID eq. 15; BTE89 eqs.
  20-33). In the wake xi continues the upper surface's (FID p. 3: the wake extends the upper
  surface): xi = xi_TE,upper + distance along the wake.
* The stagnation point lies between the two stations where the signed surface speed changes sign:
  s_stag = s_k + (s_k+1 - s_k) ue_k/(ue_k + ue_k+1) (FID p. 6); xi = |s - s_stag|; its
  dependence on the two ue enters the Jacobian (FID eq. 28). It is re-located after each update.
* Transition is re-identified after each update by marching the amplification equation from the
  stagnation point on the current state (FID p. 8); the laminar n are set from that march; new
  turbulent stations get C_tau interpolated from the transition value (FID p. 8).
* Jacobian, never hand-derived: complex step of the local boundary-layer residuals (stations
  coloured so that no residual depends on two stations of one colour; one evaluation per colour and
  variable gives every local derivative) with the chain rule through s_stag; R_u is bilinear and
  differentiated exactly. Or JAX forward-mode automatic differentiation of the same full residual.
* Newton steps are scaled down only to keep the state admissible, with the limits of FID p. 11
  plus Re_theta >= ``re_theta_min`` on turbulent stations; after the update negative C_tau and
  H_k below ``hk_min`` are clipped as FID p. 11 does. Every scaled step (with its binding limit)
  and every clipped station is counted in the result (``step_limits``).
* Drag: Squire & Young (ARC R&M 1838, eqs. 3, 13-14) at the last wake station (FID p. 12);
  friction drag from the surface shear (FID eq. 36); pressure drag = total - friction. The
  Squire-Young value from the two trailing-edge states (SY38 p. 4) is kept as a cross-check
  (``cd_te``). Lift and moment from the surface pressure of the viscous nodal strengths.

Mach 0. The structure allows several bodies (two branches and one wake each); M3 tests one body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import inviscid
from .bl import closure_sets
from .bl import equations as eq
from .bl.march import march_init, march_wake
from .params import VISCOUS, ViscousParams

NV = 4      # unknowns and residuals per station: theta, delta*, n or C_tau, ue
#: a branch's first node lies on the stagnation point "to machine precision" (FID p. 7) when its
#: sheet strength is below this fraction of the next node's
STAG_NODE_TOL = 1e-9


# --------------------------------------------------------------------------------------- #
# Geometry and the exact edge-velocity operator
# --------------------------------------------------------------------------------------- #
@dataclass
class Geometry:
    """Panel set, assembly and arc length of one set of bodies (independent of the angle)."""

    ps: inviscid.PanelSet
    assembly: inviscid.Assembly
    s: np.ndarray            # arc length at every node, per body from its first node (M)
    body_of_node: np.ndarray


def build_geometry(coords) -> Geometry:
    ps = inviscid.PanelSet.from_bodies(coords)
    asm = inviscid.assemble(ps, sources=True)
    m = ps.num_nodes
    lengths = np.asarray(ps.lengths)
    s = np.zeros(m)
    body_of_node = np.zeros(m, dtype=int)
    for k, (sp, sn) in enumerate(zip(ps.body_panels, ps.body_nodes)):
        s[sn.start + 1:sn.stop] = np.cumsum(lengths[sp])
        body_of_node[sn] = k
    return Geometry(ps, asm, s, body_of_node)


@dataclass
class Coupling:
    """Airfoil and wake stations of one angle of attack and the exact edge-velocity operator.

    Stations: the M airfoil nodes, then the N_w wake nodes of each body (:mod:`foil_lab.wake`).
    With m = ue delta* at every station and dd the direction factor (d on the airfoil, +1 in the
    wake), the panel sources are q = P (dd m) (FID eq. 23 on airfoil and wake panels); the airfoil
    nodal strengths are gamma = gamma_inv + G (dd m), G = B P, B the response of our core to unit
    airfoil and wake sources; the wake edge speed is ue_w = ue_w,inv + W (dd m), W = (C_gamma B +
    C_sigma) P (FID eqs. 24-25). On the airfoil ue = -d gamma, so D = [-diag(d) G; W] diag(dd)
    (FID eq. 26)."""

    alpha: float
    wakes: list
    m_air: int
    s: np.ndarray              # arc length per station; wake: from the body's first node, continuing
                               # past the upper trailing-edge node (xi = s - s_stag there)
    body_of_node: np.ndarray
    wake_nodes: list           # global station indices of each body's wake
    g: np.ndarray              # (M, Mt)
    w: np.ndarray              # (Nw_total, Mt)
    gamma_inv: np.ndarray      # (M)
    ue_inv_wake: np.ndarray    # (Nw_total)
    hw: np.ndarray             # dead-air thickness per station (0 on the airfoil)
    dead_frac: np.ndarray      # h_w / h_TE per station (0 on the airfoil)

    @property
    def m_total(self) -> int:
        return int(self.s.size)

    def dd(self, d_air: np.ndarray) -> np.ndarray:
        return np.concatenate([d_air, np.ones(self.m_total - self.m_air)])

    def dmat(self, d_air: np.ndarray) -> np.ndarray:
        return np.vstack([-d_air[:, None] * self.g, self.w]) * self.dd(d_air)[None, :]

    def ue_inv(self, d_air: np.ndarray) -> np.ndarray:
        return np.concatenate([-d_air * self.gamma_inv, self.ue_inv_wake])


def build_coupling(geo: Geometry, alpha: float, closure_set="fidkowski2022") -> Coupling:
    """Wake of every body along the inviscid streamline at ``alpha`` and the edge-velocity
    operator over airfoil and wake stations (rebuilt per angle; FID pp. 7, 9-10)."""
    from . import wake as wk_mod
    ps, asm = geo.ps, geo.assembly
    n, m = ps.num_panels, ps.num_nodes
    gamma_inv = np.asarray(inviscid.solve(ps, alpha, assembly=asm).gamma)
    cs = closure_sets.get(closure_set)
    wakes = [wk_mod.build_wake(ps, gamma_inv, alpha, k) for k in range(ps.num_bodies)]
    nw_tot = sum(w.n for w in wakes)
    mt = m + nw_tot
    nq = n + sum(w.n - 1 for w in wakes)
    # source parameters q = [airfoil panels, wake panels of body 0, ...] from dd m
    pq = np.zeros((nq, mt))
    lengths = np.asarray(ps.lengths)
    pq[np.arange(n), ps.start_node] = -1.0 / lengths
    pq[np.arange(n), ps.end_node] = 1.0 / lengths
    # normal velocity at the airfoil midpoints and wake tangential velocity, per q
    sn_air = np.zeros((n, nq))
    sn_air[:, :n] = np.asarray(asm.sn)
    ut_q = np.zeros((nw_tot, nq))
    ut_g = np.zeros((nw_tot, m))
    ue_w = np.zeros(nw_tot)
    s = np.concatenate([geo.s, np.zeros(nw_tot)])
    body = np.concatenate([geo.body_of_node, np.zeros(nw_tot, dtype=int)])
    hw = np.zeros(mt)
    dead = np.zeros(mt)
    wake_nodes = []
    r0, q0 = m, n
    for w in wakes:
        op = wk_mod.wake_operators(ps, w, alpha)
        npw = w.n - 1
        # this wake's operators act on the airfoil sources and its own wake sources; the velocity of
        # one body's wake sources at another body's wake is not included (single body in M3; M5)
        cols = np.r_[0:n, q0:q0 + npw]
        sn_air[:, cols] += op.sn_air
        rows = np.arange(r0, r0 + w.n)
        ut_q[np.ix_(rows - m, cols)] += op.ut_wake
        ut_g[rows - m] = op.ut_gamma
        ue_w[rows - m] = op.ut_inf + op.ut_gamma @ gamma_inv
        ds_w = np.diff(w.s)
        pq[q0 + np.arange(npw), rows[:-1]] = -1.0 / ds_w
        pq[q0 + np.arange(npw), rows[1:]] = 1.0 / ds_w
        last = ps.body_nodes[w.body].stop - 1
        s[rows] = geo.s[last] + w.s
        body[rows] = w.body
        hw[rows], dead[rows] = w.dead_air(cs.WAKE_DTDX_CLIP)
        wake_nodes.append(rows)
        r0 += w.n
        q0 += npw
    # airfoil Neumann rows: outer normal velocity = own panel source (inviscid.solve)
    rhs = -sn_air
    rhs[np.arange(n), np.arange(n)] += 1.0
    if asm.elimination is None:
        b = np.linalg.solve(np.asarray(asm.matrix), np.vstack([rhs, np.zeros((ps.num_bodies, nq))]))
    else:
        qm, rm = np.linalg.qr(np.asarray(asm.matrix))
        b = np.asarray(asm.elimination) @ np.linalg.solve(rm, qm.T @ rhs)
    g = b @ pq
    wmat = (ut_g @ b + ut_q) @ pq
    return Coupling(alpha, wakes, m, s, body, wake_nodes, g, wmat, gamma_inv, ue_w, hw, dead)


# --------------------------------------------------------------------------------------- #
# Branch structure (static during one Newton step)
# --------------------------------------------------------------------------------------- #
@dataclass
class Branch:
    body: int
    upper: bool
    nodes: np.ndarray                 # global node indices from the stagnation point downstream
    p_tr: int | None = None           # position (in nodes) of the 2nd node of the transition interval
    forced_w: float | None = None     # onset fraction if the transition is a forced trip
    s_trip: float | None = None       # arc length of the forced trip, if any
    trip_xi: float | None = None      # its xi for the current stagnation point
    k0: int = 0                       # 1 if the first node lies on the stagnation point


@dataclass
class Structure:
    cpl: Coupling
    k_stag: np.ndarray                # per body: global index of the last node of the lower branch
    d: np.ndarray                     # direction factor per station (Mt; +1 in the wake)
    branches: list
    turbulent: np.ndarray             # per station (always True in the wake)
    groups: dict = field(default_factory=dict)
    perm: np.ndarray | None = None
    dep: np.ndarray | None = None     # (3 Mt, 3) stations each boundary-layer row depends on
    colour: np.ndarray | None = None  # Jacobian colouring of the stations
    dmat: np.ndarray | None = None    # ue = ue_inv + dmat @ (ue delta*)
    ue_inv: np.ndarray | None = None


def _stagnation_panel(gamma_body: np.ndarray, x_body: np.ndarray, prev: int | None) -> int:
    """Body-local index k with gamma_k > 0 >= gamma_{k+1} (lower branch ends at k); among several,
    the one nearest the previous one, or nearest the leading edge (minimum x)."""
    cand = np.flatnonzero((gamma_body[:-1] > 0.0) & (gamma_body[1:] <= 0.0))
    if cand.size == 0:
        raise RuntimeError("no stagnation point found on a body")
    ref = prev if prev is not None else int(np.argmin(x_body))
    return int(cand[np.argmin(np.abs(cand - ref))])


def make_structure(geo: Geometry, cpl: Coupling, gamma: np.ndarray, prev: Structure | None = None,
                   trips: dict | None = None) -> Structure:
    """Stagnation points and branches for signed airfoil surface speeds ``gamma`` (sheet strengths).
    ``trips``: {(body, 'upper'|'lower'): x/c of a forced trip}. Transition data are copied from
    ``prev`` per branch."""
    ps = geo.ps
    m = ps.num_nodes
    d = np.ones(cpl.m_total)
    ks, branches = [], []
    for k, sn in enumerate(ps.body_nodes):
        prev_k = None if prev is None else int(prev.k_stag[k] - sn.start)
        kl = _stagnation_panel(np.asarray(gamma)[sn], ps.node_x[sn], prev_k)
        ks.append(sn.start + kl)
        lower = np.arange(sn.start + kl, sn.start - 1, -1)
        upper = np.arange(sn.start + kl + 1, sn.stop)
        d[lower], d[upper] = -1.0, 1.0
        for nodes, up in ((lower, False), (upper, True)):
            s_trip = None
            xt = (trips or {}).get((k, "upper" if up else "lower"))
            if xt is not None:
                xs, ss = ps.node_x[nodes], geo.s[nodes]
                j = np.flatnonzero(xs[1:] >= xt)
                if j.size:
                    j = int(j[0]) + 1
                    f = (xt - xs[j - 1]) / (xs[j] - xs[j - 1])
                    s_trip = float(ss[j - 1] + f * (ss[j] - ss[j - 1]))
            g_b = np.asarray(gamma)[nodes]
            k0 = int(abs(g_b[0]) <= STAG_NODE_TOL * abs(g_b[1]))
            branches.append(Branch(k, up, nodes, s_trip=s_trip, k0=k0))
    turb = np.zeros(cpl.m_total, dtype=bool)
    turb[m:] = True
    st = Structure(cpl, np.array(ks), d, branches, turb)
    if prev is not None:
        st.turbulent[:m] = prev.turbulent[:m]
        for b, b0 in zip(st.branches, prev.branches):
            b.p_tr, b.forced_w = b0.p_tr, b0.forced_w
    st.dmat = cpl.dmat(d[:m])
    st.ue_inv = cpl.ue_inv(d[:m])
    return st


def _colouring(dep: np.ndarray, n: int) -> np.ndarray:
    """Greedy colouring of the stations such that no residual row depends on two stations of the
    same colour (two colours on a chain; a third where a wake's first row joins both trailing
    edges)."""
    nbr = [set() for _ in range(n)]
    for row in dep:
        u = set(int(v) for v in row)
        for v in u:
            nbr[v] |= u - {v}
    col = -np.ones(n, dtype=int)
    for v in range(n):
        used = {col[u] for u in nbr[v] if col[u] >= 0}
        c = 0
        while c in used:
            c += 1
        col[v] = c
    return col


def finalize(st: Structure) -> Structure:
    """Static index arrays of the residual groups, the row permutation and the colouring."""
    cpl = st.cpl
    mt = cpl.m_total
    g = {"first": [], "lam": [], "turb": [], "trans": [], "forced": [], "wake_first": [], "wake": [],
         "stag_node": []}
    for b in st.branches:
        nd = b.nodes
        if nd.size < 3 + b.k0:
            raise RuntimeError("a branch has fewer than three stations")
        if b.k0:
            g["stag_node"].append((nd[0], nd[1]))
        g["first"].append((nd[b.k0], nd[b.k0 + 1]))
        for p in range(1 + b.k0, nd.size):
            if b.p_tr is None or p < b.p_tr:
                g["lam"].append((nd[p - 1], nd[p]))
            elif p == b.p_tr:
                if b.forced_w is not None:
                    g["forced"].append((nd[p - 1], nd[p], b.forced_w))
                else:
                    g["trans"].append((nd[p - 1], nd[p]))
            else:
                g["turb"].append((nd[p - 1], nd[p]))
    for k, wn in enumerate(cpl.wake_nodes):
        te = {b.upper: b.nodes[-1] for b in st.branches if b.body == k}
        g["wake_first"].append((te[False], te[True], wn[0], k))
        for p in range(1, wn.size):
            g["wake"].append((wn[p - 1], wn[p]))
    groups, target, dep = {}, [], []
    for key in ("first", "lam", "turb", "trans", "forced", "wake_first", "stag_node", "wake"):  # bl_residual order
        rows = g[key]
        if not rows:
            groups[key] = None
            continue
        a = np.array([r[0] for r in rows], dtype=int)
        bb = np.array([r[1] for r in rows], dtype=int)
        if key == "forced":
            groups[key] = (a, bb, np.array([r[2] for r in rows]))
        elif key == "wake_first":
            groups[key] = (a, bb, np.array([r[2] for r in rows], dtype=int),
                           np.array([r[3] for r in rows], dtype=int))
        else:
            groups[key] = (a, bb)
        if key in ("first", "stag_node"):
            row_node, dd_ = a, np.stack([a, bb, bb], axis=1)
        elif key == "wake_first":
            row_node = groups[key][2]
            dd_ = np.stack([a, bb, row_node], axis=1)
        else:
            row_node, dd_ = bb, np.stack([a, bb, bb], axis=1)
        for e in range(3):
            target.append(3 * row_node + e)
            dep.append(dd_)
    target = np.concatenate(target)
    if not np.array_equal(np.sort(target), np.arange(3 * mt)):
        raise RuntimeError("residual rows do not cover every station exactly once")
    perm = np.argsort(target)
    st.groups, st.perm, st.dep = groups, perm, np.concatenate(dep)[perm]
    st.colour = _colouring(st.dep, mt)
    return st


# --------------------------------------------------------------------------------------- #
# Residual (written once, array namespace xp)
# --------------------------------------------------------------------------------------- #
def stagnation_s(st: Structure, ue, xp=np):
    """Arc length of the stagnation point of every body from the two adjacent speeds (FID p. 6)."""
    k = st.k_stag
    s = xp.asarray(st.cpl.s)
    u0, u1 = ue[k], ue[k + 1]
    return s[k] + (s[k + 1] - s[k]) * u0 / (u0 + u1)


def _wake_station(theta, dstar, c, ue, idx, cpl, reynolds, xp, cs):
    return eq.station(theta, dstar, c, ue, reynolds, True, xp, cs, wake=True,
                      hw=xp.asarray(cpl.hw[idx]), dead_frac=xp.asarray(cpl.dead_frac[idx]))


def bl_residual(theta, dstar, c, ue, s_stag, st: Structure, reynolds, prm: ViscousParams, xp=np):
    """Boundary-layer residuals (3 Mt; station j: rows 3j..3j+2) from station values and the
    stagnation arc length per body (independent inputs here; see :func:`residual`)."""
    cs = prm.closure_set
    csm = closure_sets.get(cs)
    cpl = st.cpl
    bias, n_crit = prm.shape_eq_bias, prm.n_crit
    xi = xp.asarray(st.d) * (xp.asarray(cpl.s) - s_stag[xp.asarray(cpl.body_of_node)])
    blocks = []
    G = st.groups
    a, b = G["first"]
    # stagnation-point equations on the state extrapolated to xi = 0 (FID p. 7)
    x1, x2 = xi[a], xi[b]
    f = x1 / (x2 - x1)
    th0 = theta[a] - f * (theta[b] - theta[a])
    ds0 = dstar[a] - f * (dstar[b] - dstar[a])
    kk = (ue[a] * x2 ** 2 - ue[b] * x1 ** 2) / (x1 * x2 * (x2 - x1))
    r = eq.similarity_start(th0, ds0, kk, 1.0, 1.0, reynolds, xp, cs)
    blocks += [r[0], r[1], c[a]]
    if G["lam"] is not None:
        a, b = G["lam"]
        s1 = eq.station(theta[a], dstar[a], c[a], ue[a], reynolds, False, xp, cs)
        s2 = eq.station(theta[b], dstar[b], c[b], ue[b], reynolds, False, xp, cs)
        r = eq.laminar_interval(s1, s2, xi[a], xi[b], xp, bias, cs, n_crit)
        blocks += [r[0], r[1], r[2]]
    if G["turb"] is not None:
        a, b = G["turb"]
        s1 = eq.station(theta[a], dstar[a], c[a], ue[a], reynolds, True, xp, cs)
        s2 = eq.station(theta[b], dstar[b], c[b], ue[b], reynolds, True, xp, cs)
        r = eq.turbulent_interval(s1, s2, xi[a], xi[b], xp, bias, cs)
        blocks += [r[0], r[1], r[2]]
    for key in ("trans", "forced"):
        if G[key] is None:
            continue
        a, b = G[key][:2]
        w = None if key == "trans" else xp.asarray(G[key][2])
        r, _ = eq.transition_interval(theta[a], dstar[a], c[a], ue[a], theta[b], dstar[b], c[b], ue[b],
                                      xi[a], xi[b], reynolds, n_crit, xp, bias, w, cs)
        blocks += [r[0], r[1], r[2]]
    if G["wake_first"] is not None:
        # first wake station from the two trailing-edge states (FID eq. 14; BTE89 eqs. 16-18)
        lo, up, w0, body = G["wake_first"]
        h_te = xp.asarray(np.array([cpl.wakes[k].h_te for k in body]))

        def te_ctau(j):
            stt = eq.station(theta[j], dstar[j], c[j], ue[j], reynolds, True, xp, cs)
            c_tr = csm.ctau_transition(stt.hk, stt.ctau_eq, xp)
            return xp.where(xp.asarray(st.turbulent[j]), c[j], c_tr)
        c_start = csm.ctau_wake_start(theta[up], te_ctau(up), theta[lo], te_ctau(lo), xp)
        blocks += [theta[w0] / (theta[lo] + theta[up]) - 1.0,
                   dstar[w0] / (dstar[lo] + dstar[up] + h_te) - 1.0,
                   xp.sqrt(c[w0] / c_start) - 1.0]
    if G["stag_node"] is not None:
        # a node on the stagnation point to round-off (ue = 0 there, so no interval can start from
        # it): the stagnation equations apply at the branch's next node (group "first"), and this
        # node carries that node's theta and delta* (Hiemenz flow) with n = 0 (FID p. 7;
        # DECISIONS.md M3, C3)
        a, b = G["stag_node"]
        blocks += [theta[a] / theta[b] - 1.0, dstar[a] / dstar[b] - 1.0, c[a]]
    if G["wake"] is not None:
        a, b = G["wake"]
        s1 = _wake_station(theta[a], dstar[a], c[a], ue[a], a, cpl, reynolds, xp, cs)
        s2 = _wake_station(theta[b], dstar[b], c[b], ue[b], b, cpl, reynolds, xp, cs)
        r = eq.turbulent_interval(s1, s2, xi[a], xi[b], xp, bias, cs)
        blocks += [r[0], r[1], r[2]]
    return xp.concatenate(blocks)[xp.asarray(st.perm)]


def ue_residual(dstar, ue, st: Structure, xp=np):
    """R_u = ue - ue_inv - D (ue delta*) (LNE89 eq. 33, FID eq. 27), in units of the freestream."""
    return ue - xp.asarray(st.ue_inv) - xp.asarray(st.dmat) @ (ue * dstar)


def residual(x, st: Structure, reynolds, prm: ViscousParams, xp=np):
    """Full residual (4 Mt; station j: rows 4j..4j+3) for the state x (Mt, 4) or (4 Mt,)."""
    x = xp.reshape(x, (-1, NV))
    th, ds, c, ue = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    rb = bl_residual(th, ds, c, ue, stagnation_s(st, ue, xp), st, reynolds, prm, xp)
    ru = ue_residual(ds, ue, st, xp)
    return xp.concatenate([xp.reshape(rb, (-1, 3)), ru[:, None]], axis=1).reshape(-1)


def jacobian_complex_step(x, st: Structure, reynolds, prm, h=1e-30):
    """dR/dx (4 Mt x 4 Mt): complex step of the local residuals, chain rule through s_stag(ue)."""
    x = np.reshape(np.asarray(x, float), (-1, NV))
    mn = x.shape[0]
    ue = x[:, 3]
    ss = stagnation_s(st, ue)
    rows = 3 * mn
    colour, dep = st.colour, st.dep
    loc = np.zeros((NV, rows, mn))
    base = [x[:, v].astype(complex) for v in range(NV)]
    r_idx = np.arange(rows)
    dep_col = colour[dep]
    for v in range(NV):
        for c in range(int(colour.max()) + 1):
            args = [a.copy() for a in base]
            args[v] = args[v] + 1j * h * (colour == c)
            dr = np.imag(bl_residual(*args, ss.astype(complex), st, reynolds, prm)) / h
            hit = dep_col == c
            ok = hit.any(axis=1)
            col = dep[r_idx, np.argmax(hit, axis=1)]
            loc[v][r_idx[ok], col[ok]] += dr[ok]
    dr_ds = np.imag(bl_residual(*base, ss + 1j * h, st, reynolds, prm)) / h
    k = st.k_stag
    s = st.cpl.s
    u0, u1 = ue[k], ue[k + 1]
    den = (u0 + u1) ** 2
    ds_du0 = (s[k + 1] - s[k]) * u1 / den
    ds_du1 = -(s[k + 1] - s[k]) * u0 / den
    body_row = st.cpl.body_of_node[r_idx // 3]
    loc[3][r_idx, k[body_row]] += dr_ds * ds_du0[body_row]
    loc[3][r_idx, k[body_row] + 1] += dr_ds * ds_du1[body_row]
    j = np.zeros((NV * mn, NV * mn))
    bl_rows = (NV * (r_idx // 3) + r_idx % 3)
    for v in range(NV):
        j[np.ix_(bl_rows, np.arange(v, NV * mn, NV))] = loc[v]
    ru_rows = np.arange(3, NV * mn, NV)
    j[np.ix_(ru_rows, np.arange(1, NV * mn, NV))] = -st.dmat * ue[None, :]
    j[np.ix_(ru_rows, np.arange(3, NV * mn, NV))] = np.eye(mn) - st.dmat * x[:, 1][None, :]
    return j


def jacobian_jax(x, st: Structure, reynolds, prm):
    """dR/dx by JAX forward-mode automatic differentiation of :func:`residual`."""
    import jax
    import jax.numpy as jnp
    with jax.enable_x64(True):
        f = lambda v: residual(v, st, reynolds, prm, jnp)  # noqa: E731
        return np.asarray(jax.jacfwd(f)(jnp.asarray(np.ravel(x))))


# --------------------------------------------------------------------------------------- #
# Transition re-identification and step limiting
# --------------------------------------------------------------------------------------- #
def _march_amplification(x, xi, b: Branch, reynolds, prm, cs):
    """Laminar n along a branch on the current state (FID p. 8): (n per position, p_tr, forced_w)."""
    nd = b.nodes
    th, ds, ue = x[nd, 0], x[nd, 1], x[nd, 3]
    n = np.zeros(nd.size)
    hk = cs.hk(ds / th)
    ret = reynolds * ue * th
    xib = xi[nd]
    r_prev = cs.dn_dxi(hk[b.k0], th[b.k0], ret[b.k0], 0.0, prm.n_crit)
    for p in range(1 + b.k0, nd.size):
        dxi = xib[p] - xib[p - 1]
        if b.trip_xi is not None and xib[p - 1] < b.trip_xi <= xib[p]:
            return n, p, (b.trip_xi - xib[p - 1]) / dxi
        # implicit trapezoid (the FID rate depends on n), solved to round-off so that the n set
        # here satisfy the laminar residuals of the Newton system exactly
        n2 = n[p - 1]
        for _ in range(100):
            n_new = n[p - 1] + 0.5 * dxi * (r_prev + cs.dn_dxi(hk[p], th[p], ret[p], n2, prm.n_crit))
            done = abs(n_new - n2) <= 1e-15 * max(1.0, abs(n_new))
            n2 = n_new
            if done:
                break
        n[p] = n2
        if n2 >= prm.n_crit:
            return n, p, None
        r_prev = cs.dn_dxi(hk[p], th[p], ret[p], n2, prm.n_crit)
    return n, None, None


def update_transition(x, st: Structure, reynolds, prm) -> bool:
    """Re-identify the transition interval of every branch, set the laminar n, initialise C_tau on
    stations that became turbulent. Returns True if any transition interval moved."""
    cs = closure_sets.get(prm.closure_set)
    ss = stagnation_s(st, x[:, 3])
    xi = st.d * (st.cpl.s - ss[st.cpl.body_of_node])
    moved = False
    for b in st.branches:
        b.trip_xi = None if b.s_trip is None else float(st.d[b.nodes[0]] * (b.s_trip - ss[b.body]))
        n, p_tr, w = _march_amplification(x, xi, b, reynolds, prm, cs)
        nd = b.nodes
        if p_tr != b.p_tr or (w is None) != (b.forced_w is None):
            moved = True
        old_turb = st.turbulent[nd].copy()
        lam_end = nd.size if p_tr is None else p_tr
        x[nd[:lam_end], 2] = n[:lam_end]
        st.turbulent[nd[:lam_end]] = False
        if p_tr is not None:
            new = np.flatnonzero(~old_turb[p_tr:]) + p_tr
            if new.size:
                stn = eq.station(x[nd, 0], x[nd, 1], 1.0, x[nd, 3], reynolds, True, cs=cs)
                c_tr = np.real(cs.ctau_transition(stn.hk, stn.ctau_eq))
                later = np.flatnonzero(old_turb[p_tr:]) + p_tr
                c_end = x[nd[later[0]], 2] if later.size else c_tr[p_tr]
                for q in new:
                    f = 0.0 if not later.size else (q - p_tr) / max(later[0] - p_tr, 1)
                    x[nd[q], 2] = float((1 - f) * c_tr[p_tr] + f * c_end)
            st.turbulent[nd[p_tr:]] = True
        b.p_tr, b.forced_w = p_tr, w
    return moved


def step_scale(x, dx, st, reynolds, prm) -> tuple[float, str]:
    """Largest omega <= 1 keeping the state admissible, and the binding limit.

    Limits of FID p. 11: theta and delta* not below half; laminar n not up by more than 2 and,
    above 0.2, not down by more than 80 %; C_tau^1/2 not up by more than 0.05 nor down by more than
    80 % where C_tau^1/2 > 0.1 max C_tau^1/2; ue not by more than 20 %, except at the two stations
    next to each stagnation point, where
    ue -> 0 and a sign change only moves the stagnation point (FID p. 6). Then halving until theta,
    delta* > 0 and (turbulent) Re_theta >= re_theta_min wherever ue stays
    positive (DECISIONS.md M3, C2). H_k and C_tau are not limited by the step: they are clipped
    after the update, as FID p. 11 does (:func:`clip_state`; DECISIONS.md M3, C4)."""
    om, why = 1.0, ""
    th, ds, c, ue = x.T
    dth, dds, dc, due = dx.T
    hw = st.cpl.hw

    def lim(name, vals):
        nonlocal om, why
        v = float(np.min(vals)) if np.size(vals) else 1.0
        if v < om:
            om, why = v, name

    k = dth < 0
    lim("theta", 0.5 * th[k] / -dth[k])
    k = dds < 0
    lim("delta*", 0.5 * ds[k] / -dds[k])
    lam, tur = ~st.turbulent, st.turbulent
    k = lam & (dc > 0)
    lim("n up", 2.0 / dc[k])
    k = lam & (c > 0.2) & (dc < 0)
    lim("n down", 0.8 * c[k] / -dc[k])
    k = tur & (dc > 0)
    lim("C_tau up", ((np.sqrt(np.maximum(c[k], 0.0)) + 0.05) ** 2 - c[k]) / dc[k])
    big_c = np.sqrt(np.maximum(c, 0.0)) > 0.1 * np.sqrt(np.max(c[tur])) if tur.any() else tur
    k = tur & big_c & (dc < 0)
    lim("C_tau down", 0.96 * c[k] / -dc[k])
    near = np.zeros(ue.size, dtype=bool)
    near[st.k_stag] = near[st.k_stag + 1] = True
    k = (np.abs(due) > 0) & ~near
    lim("ue", 0.2 * np.abs(ue[k]) / np.abs(due[k]))
    for _ in range(60):
        xn = x + om * dx
        pos = xn[:, 3] > 0
        ret = reynolds * xn[:, 3] * xn[:, 0]
        if np.all(xn[:, 0] > 0) and np.all(xn[:, 1] > 0) and np.all(ret[tur & pos] >= prm.re_theta_min):
            break
        om *= 0.5
        why = "admissibility (Re_theta, positivity)"
    return float(om), why


def clip_state(x, st, prm) -> dict:
    """After the update (FID p. 11): negative C_tau^1/2 are set to 0.1 max C_tau^1/2, and delta* is
    raised where needed so that H_k >= ``hk_min`` (in the wake H_k of the closures, (delta* -
    h_w)/theta), except where ue is not positive (the two stations next to a stagnation point
    during a sign change). Returns the number of clipped stations per kind (DECISIONS.md M3, C4)."""
    out = {}
    tur = st.turbulent
    c = x[:, 2]
    neg = tur & (c <= 0)
    if neg.any():
        cmax = np.max(np.sqrt(np.maximum(c[tur], 0.0)))
        c[neg] = (0.1 * cmax) ** 2
        out["clip C_tau"] = int(neg.sum())
    hw = st.cpl.hw
    low = (x[:, 3] > 0) & ((x[:, 1] - hw) < prm.hk_min * x[:, 0])
    if low.any():
        x[low, 1] = prm.hk_min * x[low, 0] + hw[low]
        out["clip H_k"] = int(low.sum())
    return out


def relocate(x, geo, st: Structure, trips) -> tuple[Structure, bool]:
    """Re-locate the stagnation points from the signed speeds of the state. A station that changes
    branch keeps its speed (sign changed with the direction factor) and takes theta and delta* of
    its downstream neighbour on the new branch, with n = 0: next to a stagnation point theta and
    delta* are nearly uniform (Hiemenz flow), so this is the consistent value, whereas the values
    of the other branch are not (DECISIONS.md M3, C2)."""
    m = geo.ps.num_nodes
    gamma = -st.d[:m] * x[:m, 3]
    st_new = make_structure(geo, st.cpl, gamma, st, trips)
    flip = st_new.d != st.d
    if flip.any():
        x[flip, 3] = -x[flip, 3]
        for b in st_new.branches:
            nd = b.nodes
            moved = np.flatnonzero(flip[nd])
            if not moved.size:
                continue
            q = int(moved.max()) + 1                       # first station that did not move
            q = min(q, nd.size - 1)
            x[nd[moved], 0] = x[nd[q], 0]
            x[nd[moved], 1] = x[nd[q], 1]
            x[nd[moved], 2] = 0.0
            st_new.turbulent[nd[moved]] = False
    return st_new, bool(flip.any())


# --------------------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------------------- #
@dataclass
class ViscousResult:
    alpha: float
    reynolds: float
    converged: bool
    iterations: int                                   # Newton iterations of the final solve
    reason: str
    cl: float = np.nan
    cd: float = np.nan                                # Squire-Young at the end of the wake
    cdf: float = np.nan
    cdp: float = np.nan
    cd_te: float = np.nan                             # Squire-Young at the trailing edge (cross-check)
    cm: float = np.nan
    xtr_upper: float = np.nan
    xtr_lower: float = np.nan
    iterations_total: int = 0                         # including intermediate continuation solves
    continuation: int = 0
    residual_history: list = field(default_factory=list)
    scaled_steps: int = 0
    step_limits: dict = field(default_factory=dict)   # binding limit of each scaled step: count
    state: Any = field(default=None, repr=False)      # (x, structure) for a warm start
    surface: dict = field(default_factory=dict, repr=False)
    closure_set: str = ""


def _initial_state(geo, cpl: Coupling, reynolds, prm, trips):
    """Starting march on the inviscid edge velocity (FID section V.A): airfoil branches, then the
    wake from its first station (FID eq. 14; a laminar trailing edge with the set's C_tau at
    transition)."""
    cs = closure_sets.get(prm.closure_set)
    st = make_structure(geo, cpl, cpl.gamma_inv, None, trips)
    x = np.zeros((cpl.m_total, NV))
    ue_inv = st.ue_inv
    ss = stagnation_s(st, ue_inv)
    xi = st.d * (cpl.s - ss[cpl.body_of_node])
    te = {}
    for b in st.branches:
        nd = b.nodes
        trip_xi = None if b.s_trip is None else float(st.d[nd[0]] * (b.s_trip - ss[b.body]))
        mn = nd[b.k0:]
        r = march_init(xi[mn], np.maximum(ue_inv[mn], 1e-6), reynolds, prm.n_crit, trip_xi,
                       prm.shape_eq_bias, prm.closure_set, prm.init_hk_max_lam, prm.init_hk_max_turb)
        x[mn, 0], x[mn, 1], x[mn, 2], x[mn, 3] = r.theta, r.dstar, r.third, r.ue
        st.turbulent[mn] = r.turbulent
        if b.k0:                                   # the node on the stagnation point
            x[nd[0], :3] = x[nd[1], 0], x[nd[1], 1], 0.0
            x[nd[0], 3] = ue_inv[nd[0]]
            st.turbulent[nd[0]] = False
        j = nd[-1]
        c = x[j, 2]
        if not r.turbulent[-1]:
            stt = eq.station(x[j, 0], x[j, 1], 1.0, x[j, 3], reynolds, True, cs=cs)
            c = float(np.real(cs.ctau_transition(stt.hk, stt.ctau_eq)))
        te[(b.body, b.upper)] = (x[j, 0], x[j, 1], c)
    for k, wn in enumerate(cpl.wake_nodes):
        (tu, du, cu), (tl, dl, cl) = te[(k, True)], te[(k, False)]
        start = (tu + tl, du + dl + cpl.wakes[k].h_te, float(cs.ctau_wake_start(tu, cu, tl, cl)))
        r = march_wake(xi[wn], np.maximum(ue_inv[wn], 1e-6), cpl.hw[wn], cpl.dead_frac[wn], start,
                       reynolds, prm.shape_eq_bias, prm.closure_set, prm.init_hk_max_turb)
        x[wn, 0], x[wn, 1], x[wn, 2], x[wn, 3] = r.theta, r.dstar, r.third, r.ue
    return x, st


def solve_viscous(coords, alpha: float, reynolds: float, params: ViscousParams = VISCOUS,
                  trips: dict | None = None, start=None, geometry: Geometry | None = None
                  ) -> ViscousResult:
    """Coupled solution at one angle of attack (degrees), Reynolds number per unit chord, Mach 0.

    ``trips``: {(body, 'upper'|'lower'): x/c} forced transition locations. ``start``: the
    ``state`` of a previous result on the same geometry (warm start; DG87 p. 1353), else the
    starting march of :func:`foil_lab.bl.march.march_init` and :func:`march_wake`.
    """
    prm = params
    geo = geometry or build_geometry(coords)
    cpl = build_coupling(geo, alpha, prm.closure_set)
    if start is not None:
        x, st0 = start
        x = np.array(x, dtype=float)
        st0 = Structure(cpl, st0.k_stag, st0.d, st0.branches, st0.turbulent)
        st, _ = relocate(x, geo, st0, trips)
    else:
        x, st = _initial_state(geo, cpl, reynolds, prm, trips)
    hist, scaled, upd, limits = [], 0, np.inf, {}
    converged, reason, it, changed = False, "max_iter", 0, True
    for it in range(prm.max_iter + 1):
        update_transition(x, st, reynolds, prm) if it == 0 else None
        finalize(st)
        r = residual(x, st, reynolds, prm)
        rmax = float(np.max(np.abs(r)))
        hist.append(rmax)
        if not np.isfinite(rmax):
            reason = "non-finite residual"
            break
        if rmax <= prm.residual_tol and upd <= prm.update_tol and not changed:
            converged, reason = True, "converged"
            break
        if it == prm.max_iter:
            break
        if prm.jacobian == "jax":
            j = jacobian_jax(x, st, reynolds, prm)
        else:
            j = jacobian_complex_step(x, st, reynolds, prm)
        try:
            dx = np.linalg.solve(j, -r).reshape(-1, NV)
        except np.linalg.LinAlgError:
            reason = "singular Jacobian"
            break
        om, why = step_scale(x, dx, st, reynolds, prm)
        if om < 1.0:
            scaled += 1
            limits[why] = limits.get(why, 0) + 1
        x = x + om * dx
        upd = float(np.max(np.abs(om * dx[:, :2]) / np.abs(x[:, :2])))
        for k, v in clip_state(x, st, prm).items():
            limits[k] = limits.get(k, 0) + v
        st, moved_stag = relocate(x, geo, st, trips)
        moved_tr = update_transition(x, st, reynolds, prm)
        changed = moved_stag or moved_tr
    res = _loads(x, geo, st, alpha, reynolds, prm)
    return ViscousResult(alpha=alpha, reynolds=reynolds, converged=converged, iterations=it,
                         iterations_total=it, reason=reason, residual_history=hist,
                         scaled_steps=scaled, step_limits=limits, state=(x.copy(), st),
                         closure_set=prm.closure_set, **res)


def _loads(x, geo, st, alpha, reynolds, prm) -> dict:
    cs = closure_sets.get(prm.closure_set)
    ps = geo.ps
    cpl = st.cpl
    m = ps.num_nodes
    th, dsx, c, ue = x.T
    gamma = cpl.gamma_inv + cpl.g @ (cpl.dd(st.d[:m]) * ue * dsx)
    g_mid = 0.5 * (gamma[ps.start_node] + gamma[ps.end_node])
    cp = 1.0 - g_mid ** 2
    cl, _, cm = inviscid.pressure_loads(ps, cp, alpha)
    ss = stagnation_s(st, ue)
    a = np.deg2rad(alpha)
    dhat = np.array([np.cos(a), np.sin(a)])
    out = dict(cl=float(cl), cm=float(cm))
    h = dsx / th
    hk = cs.hk(h)
    ret = reynolds * ue * th
    with np.errstate(all="ignore"):
        cf = np.where(st.turbulent, cs.cf_turb(hk, np.maximum(ret, 1.0001)), cs.cf_lam(hk, ret))
    cf[m:] = 0.0
    tau = cf * ue ** 2                              # wall shear / (1/2 rho V^2)
    cdf = 0.0
    surface, te_states = {}, {}
    for b in st.branches:
        nd = b.nodes
        px, py = ps.node_x[nd], ps.node_y[nd]
        k = st.k_stag[b.body]
        f = (ss[b.body] - geo.s[k]) / (geo.s[k + 1] - geo.s[k])
        sx = ps.node_x[k] + f * (ps.node_x[k + 1] - ps.node_x[k])
        sy = ps.node_y[k] + f * (ps.node_y[k + 1] - ps.node_y[k])
        xs, ys = np.concatenate([[sx], px]), np.concatenate([[sy], py])
        t = np.concatenate([[0.0], tau[nd]])        # tau = 0 at the stagnation point
        cdf += float(np.sum(0.5 * (t[1:] + t[:-1]) * (np.diff(xs) * dhat[0] + np.diff(ys) * dhat[1])))
        key = "upper" if b.upper else "lower"
        xtr = np.nan
        if b.p_tr is not None:
            p = b.p_tr
            if b.forced_w is not None:
                w = b.forced_w
            else:
                xi_ = st.d[nd] * (geo.s[nd] - ss[b.body])
                s_ = eq.station(th[nd[p - 1]], dsx[nd[p - 1]], c[nd[p - 1]], ue[nd[p - 1]], reynolds, False, cs=cs)
                w = float(np.real(eq.transition_onset(s_, th[nd[p]], dsx[nd[p]], ue[nd[p]],
                                                      xi_[p] - xi_[p - 1], reynolds, prm.n_crit, cs=cs)))
            xtr = float(px[p - 1] + w * (px[p] - px[p - 1]))
        out["xtr_" + key] = xtr
        surface[key] = dict(x=px, y=py, xi=st.d[nd] * (geo.s[nd] - ss[b.body]), ue=ue[nd], theta=th[nd],
                            dstar=dsx[nd], h=h[nd], cf=cf[nd], third=c[nd], turbulent=st.turbulent[nd])
        te_states[key] = (th[nd[-1]], dsx[nd[-1]], ue[nd[-1]])
    wn = cpl.wake_nodes[0]
    wk = cpl.wakes[0]
    surface["wake"] = dict(x=wk.x, y=wk.y, xi=wk.s, ue=ue[wn], theta=th[wn], dstar=dsx[wn], h=h[wn],
                           hw=cpl.hw[wn], third=c[wn])
    # Squire & Young (ARC R&M 1838, eqs. 3, 13-14) at the last wake station (FID p. 12) ...
    j = wn[-1]
    cd = 2.0 * th[j] * ue[j] ** ((h[j] + 5.0) / 2.0)
    # ... and, as a cross-check, from both trailing-edge states combined (SY38 p. 4)
    (tu, du, uu), (tl, dl, ul) = te_states["upper"], te_states["lower"]
    th1 = tu + tl
    h1 = (du + dl) / th1
    u1 = (tu * uu + tl * ul) / th1
    cd_te = 2.0 * th1 * u1 ** ((h1 + 5.0) / 2.0)
    out.update(cd=float(cd), cd_te=float(cd_te), cdf=cdf, cdp=float(cd - cdf), surface=surface)
    return out


def solve_continued(coords, alpha: float, reynolds: float, params: ViscousParams = VISCOUS,
                    trips=None, start=None, alpha_start: float | None = None,
                    geometry: Geometry | None = None, depth: int = 3) -> ViscousResult:
    """Solve at ``alpha`` from the converged ``start`` obtained at ``alpha_start``; if that fails,
    reach ``alpha`` through intermediate angles (the increment halved, at most ``depth`` times):
    angle-of-attack continuation (FID section VIII.C). The converged answer does not depend on the
    path. ``iterations`` counts the final solve (the point is judged by it); ``iterations_total``
    adds the failed attempt and the intermediate solves (DECISIONS.md M3, C4)."""
    geo = geometry or build_geometry(coords)
    r = solve_viscous(coords, alpha, reynolds, params, trips, start, geo)
    if r.converged or start is None or alpha_start is None or depth == 0:
        return r
    mid = 0.5 * (alpha_start + alpha)
    rm = solve_continued(coords, mid, reynolds, params, trips, start, alpha_start, geo, depth - 1)
    if not rm.converged:
        r.iterations_total += rm.iterations_total
        return r
    r2 = solve_continued(coords, alpha, reynolds, params, trips, rm.state, mid, geo, depth - 1)
    r2.iterations_total += r.iterations_total + rm.iterations_total
    r2.continuation += 1
    if r2.converged:
        return r2
    r.iterations_total = r2.iterations_total
    return r


def polar(coords, alphas, reynolds: float, params: ViscousParams = VISCOUS, trips=None,
          geometry: Geometry | None = None, start=None, alpha_start: float | None = None) -> list:
    """Results along a sequence of angles, each started from the last converged one (DG87
    p. 1353), with angle-of-attack continuation when a point fails (:func:`solve_continued`;
    FID section VIII.C: a converged solution initialises the next, "using different starting
    angles of attack and fixed or adaptive increments"). After a point that still fails, the next
    angle again continues from the last converged state; only the first angle of a sweep without
    ``start`` is started from the march (DECISIONS.md M3, C4)."""
    geo = geometry or build_geometry(coords)
    out, a_prev = [], alpha_start
    for a in alphas:
        r = solve_continued(coords, a, reynolds, params, trips, start, a_prev, geo)
        out.append(r)
        if r.converged:
            start, a_prev = r.state, a
    return out
