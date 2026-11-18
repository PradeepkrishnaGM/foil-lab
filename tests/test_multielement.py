"""M5: viscous analysis of several elements (SPEC.md 4.6).

The inviscid multi-element core is validated by the exact two-element solution in
``tests/test_williams.py`` (SPEC.md 4.2). Here: the coupling between the wakes of different bodies,
the per-element bookkeeping of the viscous result, and an end-to-end slotted-flap case.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import viscous, wake


def _two_bodies(dy):
    a = np.asarray(fl.Airfoil.naca("2412").repanelled(100).coords)
    return [a, a + np.array([0.0, dy])]


def test_cross_wake_velocity_against_quadrature():
    """Velocity along wake 1 from a unit source on one panel of wake 0 (linear half-panel strengths,
    :func:`wake.wake_operators`) equals Gauss quadrature of point sources, sigma / (2 pi r)."""
    geo = viscous.build_geometry(_two_bodies(0.6))
    cpl = viscous.build_coupling(geo, 3.0)
    w0, w1 = cpl.wakes
    op0 = wake.wake_operators(geo.ps, w0, 3.0)
    n = geo.ps.num_panels
    q = np.zeros(n + w0.n - 1)
    q[n + 3] = 1.0                                  # wake panel 3 of body 0
    got = wake.cross_wake_velocity(w1, op0) @ q
    xg, wg = np.polynomial.legendre.leggauss(40)
    t = 0.5 * (xg + 1.0)
    x1, y1, x2, y2 = op0.seg
    sa, sb = op0.s_start @ q, op0.s_end @ q
    want = np.zeros(w1.n)
    for i in np.flatnonzero((np.abs(sa) > 0) | (np.abs(sb) > 0)):
        px, py = x1[i] + t * (x2[i] - x1[i]), y1[i] + t * (y2[i] - y1[i])
        length = np.hypot(x2[i] - x1[i], y2[i] - y1[i])
        sig = sa[i] + t * (sb[i] - sa[i])
        rx = w1.x[:, None] - px[None]
        ry = w1.y[:, None] - py[None]
        r2 = rx ** 2 + ry ** 2
        u = (sig * rx / r2 * wg).sum(axis=1) * 0.5 * length / (2 * np.pi)
        v = (sig * ry / r2 * wg).sum(axis=1) * 0.5 * length / (2 * np.pi)
        want += u * w1.tx + v * w1.ty
    assert np.max(np.abs(want)) > 1e-5
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)


def test_wakes_of_different_bodies_are_coupled():
    """The edge-velocity operator along wake 1 depends on the displacement (m) at wake 0's stations,
    and the effect fades with distance."""
    near = viscous.build_coupling(viscous.build_geometry(_two_bodies(0.6)), 3.0)
    far = viscous.build_coupling(viscous.build_geometry(_two_bodies(20.0)), 3.0)

    def cross(c):
        return np.max(np.abs(c.w[np.ix_(c.wake_nodes[1] - c.m_air, c.wake_nodes[0])]))
    assert cross(near) > 10 * cross(far) > 0.0


def test_far_bodies_decouple():
    """With the second body 1000 chords away the first body's operators equal the single-body ones
    to the size of the far-field interaction."""
    geo2 = viscous.build_geometry(_two_bodies(1000.0))
    c2 = viscous.build_coupling(geo2, 3.0)
    a = _two_bodies(0.0)[0]
    c1 = viscous.build_coupling(viscous.build_geometry(a), 3.0)
    m1 = c1.m_air
    w1 = c1.wake_nodes[0]
    np.testing.assert_allclose(c2.gamma_inv[:m1], c1.gamma_inv, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(c2.g[:m1, :m1], c1.g[:, :m1], rtol=1e-3, atol=1e-2)
    nw = len(w1)
    np.testing.assert_allclose(c2.w[:nw, :m1], c1.w[:, :m1], rtol=1e-3, atol=1e-2)


@pytest.mark.slow
def test_far_apart_elements_reproduce_the_single_element():
    """Two NACA 2412 sections 200 chords apart at Re 1e6, 2 deg: each element's Cl, Cd and
    transition equal the single-section result to the interaction at that distance
    (Gamma / (2 pi 200) ~ 2e-4 of the freestream; measured dCl 3e-4)."""
    warnings.simplefilter("ignore")
    af = fl.Airfoil.naca("2412")
    one = fl.analyze(af, 2.0, re=1e6)
    two = fl.analyze(fl.Configuration((fl.Element(af), fl.Element(af, offset=(0.0, 200.0)))), 2.0, re=1e6)
    assert one.converged and two.converged
    for e in two.elements:
        assert e.cl == pytest.approx(one.cl, abs=1e-3)
        assert e.cd == pytest.approx(one.cd, rel=2e-3)
        assert e.xtr_upper == pytest.approx(one.xtr_upper, abs=1e-3)
    assert two.cd == pytest.approx(sum(e.cd for e in two.elements), rel=1e-12)
    assert two.cl == pytest.approx(sum(e.cl for e in two.elements), rel=1e-12)
    assert two.cd_friction == pytest.approx(sum(e.cd_friction for e in two.elements), rel=1e-12)
