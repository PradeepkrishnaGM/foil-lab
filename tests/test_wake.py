"""Wake of the coupled viscous solver (M3, checkpoint C4): geometry, source operators, coupling,
wake equations. Sources: FID (Fidkowski) pp. 3, 7, 9-10, eqs. 14-15, 22-27; BTE89 eqs. 20-21."""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from foil_lab import inviscid
from foil_lab import viscous as V
from foil_lab import wake as W
from foil_lab.benchmark.sections import canonical
from foil_lab.bl import closure_sets
from foil_lab.bl.march import march_wake, wake_hk_inverse
from tests.diagnostics.bl_compare import DATA, dump_name, read_dump



@pytest.fixture(scope="module")
def naca2412():
    coords = canonical("NACA2412").coords
    geo = V.build_geometry(coords)
    return coords, geo


def _quad_linear_source(px, py, x1, y1, x2, y2, sa, sb):
    """Velocity of a source sheet varying linearly from sa to sb, by quadrature of point sources."""
    L = np.hypot(x2 - x1, y2 - y1)

    def comp(t, i):
        sx, sy = x1 + (x2 - x1) * t / L, y1 + (y2 - y1) * t / L
        rx, ry = px - sx, py - sy
        g = sa + (sb - sa) * t / L
        return g * (rx if i == 0 else ry) / (2 * np.pi * (rx * rx + ry * ry))
    return tuple(quad(comp, 0, L, args=(i,), epsabs=1e-14, epsrel=1e-13, limit=400)[0] for i in (0, 1))


def test_linear_source_kernel_matches_quadrature():
    """The linear-strength source panel (the core's linear vortex rotated by 90 degrees) equals
    quadrature of point sources, for both end strengths, at points off the panel."""
    p = (np.array([0.2]), np.array([0.1]), np.array([0.9]), np.array([0.45]))
    pts = np.array([[0.3, 0.8], [1.5, -0.2], [0.55, 0.28], [-0.4, 0.05], [0.6, 0.33]])
    ua, va, ub, vb = W.linear_source_velocity(pts[:, 0], pts[:, 1], *p)
    for k, (px, py) in enumerate(pts):
        args = (px, py) + tuple(a[0] for a in p)
        assert np.allclose((ua[k, 0], va[k, 0]), _quad_linear_source(*args, 1.0, 0.0), atol=1e-12, rtol=0)
        assert np.allclose((ub[k, 0], vb[k, 0]), _quad_linear_source(*args, 0.0, 1.0), atol=1e-12, rtol=0)


def test_wake_geometry(naca2412):
    """FID p. 7: N_w = floor(N/10 + 10 d_w) nodes, first node eps_w c behind the trailing-edge
    midpoint, total length d_w c, geometric spacing; the nodes follow the inviscid streamline."""
    _, geo = naca2412
    ps = geo.ps
    g = np.asarray(inviscid.solve(ps, 5.0, assembly=geo.assembly).gamma)
    wk = W.build_wake(ps, g, 5.0)
    assert wk.n == int(np.floor(ps.num_nodes / 10 + 10 * W.D_W))
    xm = 0.5 * (ps.node_x[0] + ps.node_x[-1])
    ym = 0.5 * (ps.node_y[0] + ps.node_y[-1])
    chord = np.ptp(ps.node_x)
    assert np.hypot(wk.x[0] - xm, wk.y[0] - ym) == pytest.approx(W.EPS_W * chord, rel=1e-12)
    steps = np.hypot(np.diff(wk.x), np.diff(wk.y))
    assert np.sum(steps) == pytest.approx(W.D_W * chord, rel=1e-12)
    assert np.allclose(steps[1:] / steps[:-1], steps[1] / steps[0], rtol=1e-9)
    # streamline: the local flow direction at each segment midpoint is along the segment
    mx, my = 0.5 * (wk.x[1:] + wk.x[:-1]), 0.5 * (wk.y[1:] + wk.y[:-1])
    u, v = inviscid.induced_velocity(ps, g, mx, my)
    a = np.deg2rad(5.0)
    ang = np.arctan2(np.sin(a) + v, np.cos(a) + u) - np.arctan2(np.diff(wk.y), np.diff(wk.x))
    assert np.max(np.abs(ang)) < 2e-3


def test_dead_air_thickness(naca2412):
    """FID eq. 15 / BTE89 eqs. 20-21: h_w = h_TE at the trailing edge, zero with zero slope at
    L_w = 2.5 h_TE and beyond; the slope dt/dxi is clipped to +-3/f_w in the FID set."""
    _, geo = naca2412
    ps = geo.ps
    g = np.asarray(inviscid.solve(ps, 0.0, assembly=geo.assembly).gamma)
    wk = W.build_wake(ps, g, 0.0)
    assert wk.h_te == pytest.approx(np.hypot(ps.node_x[0] - ps.node_x[-1], ps.node_y[0] - ps.node_y[-1]))
    lw = W.F_W * wk.h_te
    probe = W.Wake(0, np.zeros(5), np.zeros(5), np.array([0.0, 0.5 * lw, lw - 1e-9, lw, 2 * lw]),
                   np.ones(5), np.zeros(5), wk.h_te, 0.3)
    hw, frac = probe.dead_air(None)
    assert hw[0] == pytest.approx(wk.h_te, rel=1e-14)
    assert hw[2] < 1e-12 and hw[3] == 0.0 and hw[4] == 0.0
    assert np.allclose(frac, hw / wk.h_te)
    xb = 0.5
    assert hw[1] == pytest.approx(wk.h_te * (1 + (2 + W.F_W * 0.3) * xb) * (1 - xb) ** 2, rel=1e-14)
    probe.dtdxi = 5.0
    clip = closure_sets.get("fidkowski2022").WAKE_DTDX_CLIP
    assert clip == pytest.approx(3.0 / W.F_W)
    hw_c, _ = probe.dead_air(clip)
    assert hw_c[1] == pytest.approx(wk.h_te * (1 + (2 + 3.0) * xb) * (1 - xb) ** 2, rel=1e-14)
    closed = W.Wake(0, np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), 0.0, 0.0)
    assert np.all(closed.dead_air(None)[0] == 0.0)


def test_wake_source_distribution(naca2412):
    """FID p. 3 and eq. 2: node strengths are the mean of the neighbouring wake panel values, the
    first node carries the sum of the first and last airfoil panel sources, the last half panel is
    repeated past the last node; the distribution is continuous along the sheet."""
    _, geo = naca2412
    ps = geo.ps
    g = np.asarray(inviscid.solve(ps, 3.0, assembly=geo.assembly).gamma)
    wk = W.build_wake(ps, g, 3.0)
    op = W.wake_operators(ps, wk, 3.0)
    n, nw = ps.num_panels, wk.n
    q = np.random.default_rng(3).standard_normal(n + nw - 1)
    sa, sb = op.s_start @ q, op.s_end @ q
    x1, y1, x2, y2 = op.seg
    npw = nw - 1
    # segments: first halves (node k -> midpoint k), second halves (midpoint k -> node k + 1),
    # then the repeated half panel past the last node
    first, second, ext = np.arange(npw), npw + np.arange(npw), 2 * npw
    assert np.allclose(x2[first], x1[second]) and np.allclose(y2[first], y1[second])
    assert np.allclose(x2[second[:-1]], x1[first[1:]]) and np.allclose(y2[second[:-1]], y1[first[1:]])
    assert np.allclose((x2[second[-1]], y2[second[-1]]), (x1[ext], y1[ext]))
    assert np.allclose(sb[first], sa[second], atol=1e-14)                  # continuous strength
    assert np.allclose(sb[second[:-1]], sa[first[1:]], atol=1e-14)
    assert sa[0] == pytest.approx(q[0] + q[n - 1])                          # first node
    assert np.allclose(sb[first], q[n:])                                     # panel value at its midpoint
    assert np.allclose(sa[first[1:]], 0.5 * (q[n:-1] + q[n + 1:]))          # interior nodes
    assert sb[second[-1]] == sa[ext] == sb[ext] == pytest.approx(q[-1])      # constant last half panel
    # the sheet starts at the trailing-edge midpoint (DECISIONS.md M3, C4)
    assert x1[0] == pytest.approx(0.5 * (ps.node_x[0] + ps.node_x[-1]))
    # normal velocity at airfoil midpoints against quadrature of the same distribution
    nx, ny = ps.normal
    for i in (0, 3, n // 2, n - 1):
        u = v = 0.0
        for k in range(x1.size):
            du, dv = _quad_linear_source(ps.mid_x[i], ps.mid_y[i], x1[k], y1[k], x2[k], y2[k], sa[k], sb[k])
            u, v = u + du, v + dv
        assert op.sn_air[i] @ q == pytest.approx(u * nx[i] + v * ny[i], abs=1e-10)


@pytest.mark.parametrize("alpha", [0.0, 6.0])
def test_coupling_matches_inviscid_solve(naca2412, alpha):
    """With a mass defect whose wake sheet strength is zero everywhere (constant wake m and
    sigma_first = -sigma_last), the coupling's airfoil strengths equal ``inviscid.solve`` with the
    same panel sources, and its wake edge speed equals freestream plus induced velocity along the
    wake (FID eqs. 24-27)."""
    _, geo = naca2412
    ps = geo.ps
    cpl = V.build_coupling(geo, alpha)
    m, n = ps.num_nodes, ps.num_panels
    d = np.where(np.arange(m) > np.argmin(ps.node_x), 1.0, -1.0)
    dm = np.zeros(cpl.m_total)
    rng = np.random.default_rng(4)
    dm[:m] = 1e-3 * (1.0 + rng.random(m))
    lengths = np.asarray(ps.lengths)
    sig = (dm[ps.end_node] - dm[ps.start_node]) / lengths
    # sigma_0 = -sigma_last through the first node's value
    dm[0] = dm[1] + sig[-1] * lengths[0]
    sig = (dm[ps.end_node] - dm[ps.start_node]) / lengths
    assert sig[0] + sig[-1] == pytest.approx(0.0, abs=1e-15)
    dm[m:] = 2.3e-3
    gamma = cpl.gamma_inv + cpl.g @ dm
    ref = inviscid.solve(ps, alpha, source_strength=sig, assembly=geo.assembly)
    assert np.max(np.abs(gamma - np.asarray(ref.gamma))) < 1e-12
    ue = cpl.ue_inv(d) + cpl.dmat(d) @ (dm * cpl.dd(d))
    assert np.allclose(ue[:m], -d * np.asarray(ref.gamma), atol=1e-12)
    wk = cpl.wakes[0]
    # evaluated where the operator evaluates it: 1e-9 beside the sheet (wake.wake_operators)
    qx, qy = wk.x - 1e-9 * wk.ty, wk.y + 1e-9 * wk.tx
    u, v = inviscid.induced_velocity(ps, ref.gamma, qx, qy, source_strength=sig)
    a = np.deg2rad(alpha)
    ut = (np.cos(a) + u) * wk.tx + (np.sin(a) + v) * wk.ty
    assert np.max(np.abs(ue[m:] - ut)) < 1e-10


def test_wake_hk_inverse_solves_eq_22():
    """FID eq. 22: six Newton iterations from H_k1 solve H_k2 + 0.03 dH (H_k2 - 1)^3 = H_k1."""
    for hk1, dh in ((2.2, 0.5), (1.6, 3.0), (3.0, 20.0)):
        hk2 = wake_hk_inverse(hk1, dh)
        assert hk2 + 0.03 * dh * (hk2 - 1.0) ** 3 == pytest.approx(hk1, abs=1e-10)
        assert 1.0 < hk2 <= hk1


@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_wake_equations_on_xfoil_wake(alpha):
    """Our wake equations (fidkowski2022), marched on XFOIL's wake stations and edge speed from
    XFOIL's first wake state, reproduce XFOIL's wake momentum thickness: theta and the
    Squire-Young drag at the wake end within 5 % (the C1 criterion for the trailing edge; measured
    0.6 %). XFOIL does not dump C_tau: the march starts from the equilibrium value."""
    _, wake = read_dump(DATA / dump_name(alpha))
    air, _ = read_dump(DATA / dump_name(alpha))
    h_te = float(np.hypot(*(air[0, 1:3] - air[-1, 1:3])))
    s_w = wake[:, 0] - wake[0, 0]
    ue = np.abs(wake[:, 3])
    xb = np.minimum(s_w / (W.F_W * h_te), 1.0)
    hw = h_te * (1 + 2 * xb) * (1 - xb) ** 2
    from foil_lab.bl import equations as eq
    st0 = eq.station(wake[0, 5], wake[0, 4], 1.0, ue[0], 1e6, True, wake=True, hw=hw[0], dead_frac=1.0)
    r = march_wake(1.0 + s_w, ue, hw, hw / h_te, (wake[0, 5], wake[0, 4], float(st0.ctau_eq)), 1e6,
                   hk_max=np.inf)
    assert not r.failed
    assert r.theta[-1] / wake[-1, 5] == pytest.approx(1.0, abs=0.05)
    cd = 2 * r.theta[-1] * ue[-1] ** ((r.h[-1] + 5) / 2)
    cd_x = 2 * wake[-1, 5] * ue[-1] ** ((wake[-1, 7] + 5) / 2)
    assert cd / cd_x == pytest.approx(1.0, abs=0.05)


@pytest.mark.slow
def test_coupled_solution_satisfies_wake_start_and_colouring(naca2412):
    """At a converged 2 deg solution (fidkowski2022): the first wake station is the sum of the
    trailing-edge states (FID eq. 14); the Jacobian colouring never gives two stations of one
    residual row the same colour; drag is Squire-Young at the last wake station."""
    coords, geo = naca2412
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = V.solve_viscous(coords, 2.0, 1e6, geometry=geo)
    assert r.converged
    x, st = r.state
    m = geo.ps.num_nodes
    w0, lo, up = m, 0, m - 1
    assert x[w0, 0] == pytest.approx(x[lo, 0] + x[up, 0], rel=1e-7)
    assert x[w0, 1] == pytest.approx(x[lo, 1] + x[up, 1] + st.cpl.wakes[0].h_te, rel=1e-7)
    V.finalize(st)
    col = st.colour[st.dep]
    for row, c in zip(st.dep, col):
        u = {}
        for node, cc in zip(row, c):
            assert u.get(cc, node) == node
            u[cc] = node
    j = st.cpl.wake_nodes[0][-1]
    assert r.cd == pytest.approx(2 * x[j, 0] * x[j, 3] ** ((x[j, 1] / x[j, 0] + 5) / 2), rel=1e-14)
