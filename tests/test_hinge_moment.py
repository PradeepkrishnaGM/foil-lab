"""Hinge-moment coefficient (M4 step 2; SPEC.md 3 and 4.5, DECISIONS.md M4).

Sign convention (Q4): positive when the moment tends to rotate the flap trailing edge down (the
direction of positive deflection, clockwise, the same sense as nose-up Cm). Lift on the flap at
positive angle of attack gives a negative value. XFOIL's ``Chinge`` has the opposite sign.
"""

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import inviscid as core
from foil_lab.flap import ZONE_ARC

SYM = fl.Airfoil.naca("0012", 160)
CAM = fl.Airfoil.naca("2412", 160)


def _setup(af):
    ps = core.PanelSet.from_bodies([np.asarray(af.coords)])
    return ps, af.flap.flap_panels, af.flap.hinge


# --------------------------------------------------------------------------------------- #
# Sign
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("deflection", (0.0, 10.0, -10.0))
def test_lift_on_the_flap_gives_a_negative_hinge_moment(deflection):
    fa = SYM.with_flap(0.75, deflection)
    up = fl.analyze(fa, 5.0)
    assert up.cl > 0.5 or deflection < 0.0
    if deflection >= 0.0:
        assert up.ch < 0.0 and up.ch_pressure == up.ch and up.ch_friction is None
    # odd in alpha for the undeflected symmetric section
    if deflection == 0.0:
        assert fl.analyze(fa, -5.0).ch == pytest.approx(-up.ch, abs=1e-12)


def test_downward_deflection_gives_a_negative_hinge_moment_at_zero_angle():
    r = fl.analyze(SYM.with_flap(0.75, 10.0), 0.0)
    assert r.cl > 0 and r.ch < 0
    r = fl.analyze(SYM.with_flap(0.75, -10.0), 0.0)
    assert r.cl < 0 and r.ch > 0


def test_no_flap_gives_none():
    r = fl.analyze(SYM, 3.0)
    assert r.ch is None and r.ch_pressure is None and r.elements[0].ch is None
    assert r.to_dict()["elements"][0]["ch"] is None


# --------------------------------------------------------------------------------------- #
# Exact identities of the panel sums
# --------------------------------------------------------------------------------------- #
def _runs(mask):
    """Index ranges (first, last+1) of the runs of True in a cyclic-free boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


@pytest.mark.parametrize("deflection", (0.0, 15.0, -20.0))
def test_uniform_pressure_moment_is_a_difference_of_squared_distances(deflection):
    """Uniform Cp on a run of panels: the moment is the exact integral of d(|r|^2 / 2)."""
    fa = CAM.with_flap(0.7, deflection)
    ps, mask, hinge = _setup(fa)
    cp = np.full(ps.num_panels, 0.8)
    ch_p, ch_f = core.hinge_moment(ps, cp, 0, mask, hinge)
    assert ch_f == 0.0
    # solver order is the reverse of the input (Selig) order
    solver_mask = mask[::-1]
    expect = 0.0
    for a, b in _runs(solver_mask):
        A = np.array([ps.x1[a], ps.y1[a]]) - hinge
        B = np.array([ps.x2[b - 1], ps.y2[b - 1]]) - hinge
        expect += 0.5 * 0.8 * (B @ B - A @ A)
    assert ch_p == pytest.approx(expect, abs=1e-13)


@pytest.mark.parametrize("deflection", (10.0, -25.0))
def test_pressure_on_the_hinge_arc_has_no_moment(deflection):
    fa = CAM.with_flap(0.75, deflection)
    ps, _, hinge = _setup(fa)
    arc = fa.flap.zones == ZONE_ARC
    assert arc.any()
    rng = np.random.default_rng(3)
    cp = rng.normal(size=ps.num_panels)
    ch_p, _ = core.hinge_moment(ps, cp, 0, arc, hinge)
    assert abs(ch_p) < 1e-14


@pytest.mark.parametrize("deflection", (0.0, 12.0))
def test_moment_about_the_hinge_follows_cm_by_the_transfer_formula(deflection):
    """All panels: ch = Cm c^2 + (H - ref) x F, with F the pressure force (algebraic identity)."""
    fa = CAM.with_flap(0.75, deflection)
    ps, _, hinge = _setup(fa)
    alpha = 4.0
    sol = core.solve(ps, alpha)
    cp = np.asarray(sol.cp)
    ref = np.array([0.25, 0.0])
    cl, cd, cm = core.pressure_loads(ps, cp, alpha, 0, 1.0, tuple(ref))
    nx, ny = ps.normal
    fx, fy = float(np.sum(-cp * nx * ps.lengths)), float(np.sum(-cp * ny * ps.lengths))
    allmask = np.ones(ps.num_panels, dtype=bool)
    ch_p, _ = core.hinge_moment(ps, cp, 0, allmask, hinge)
    expect = cm + (hinge[0] - ref[0]) * fy - (hinge[1] - ref[1]) * fx
    assert ch_p == pytest.approx(expect, abs=1e-12)


def test_hinge_moment_scales_with_the_element_scale():
    a = fl.analyze(SYM.with_flap(0.75, 8.0), 3.0).ch
    b = fl.analyze(fl.Element(SYM.with_flap(0.75, 8.0), scale=2.0, offset=(1.0, 0.5)), 3.0).ch
    assert b == pytest.approx(a, abs=1e-10)


# --------------------------------------------------------------------------------------- #
# Published linear theory
# --------------------------------------------------------------------------------------- #
def _thin_ch(alpha_deg, delta_deg, xh, nterms=200, nth=40001):
    """Thin-airfoil theory for a flat plate with a plain flap: Fourier series of the vortex sheet
    (Glauert; Anderson, *Fundamentals of Aerodynamics*, thin-airfoil theory, A0 and An; section
    number unverified), pressure difference 4 [A0 (1 + cos t)/sin t + sum An sin(n t)], hinge
    moment of the load aft of the hinge, trailing edge down positive."""
    a, dl = np.radians(alpha_deg), np.radians(delta_deg)
    th_h = np.arccos(1.0 - 2.0 * xh)
    A0 = a + dl * (np.pi - th_h) / np.pi
    n = np.arange(1, nterms + 1)
    An = 2.0 * dl / (np.pi * n) * np.sin(n * th_h)
    th = np.linspace(th_h, np.pi, nth)
    s = np.zeros_like(th)
    for k in range(0, nterms, 50):
        nn = n[k:k + 50]
        s += (An[k:k + 50, None] * np.sin(nn[:, None] * th[None, :])).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = A0 * (1 + np.cos(th)) / np.sin(th) + s
    g[-1] = s[-1]
    x = 0.5 * (1 - np.cos(th))
    return -float(np.trapezoid(4.0 * g * (x - xh) * 0.5 * np.sin(th), th))


def test_hinge_moment_slopes_agree_with_thin_airfoil_theory():
    """NACA 0002, 25 % flap: dCh/dalpha within 5 %, dCh/ddelta within 2 % of thin-airfoil theory
    (both negative). Tolerances set from the expected O(thickness) difference; not an exact check."""
    xh, af = 0.75, fl.Airfoil.naca("0002", 160)
    eps = np.radians(1.0)

    def ch(a, d):
        return fl.analyze(af.with_flap(xh, d), a).ch
    ca = (ch(1.0, 0.0) - ch(-1.0, 0.0)) / (2 * eps)
    cd = (ch(0.0, 1.0) - ch(0.0, -1.0)) / (2 * eps)
    ta = (_thin_ch(1.0, 0.0, xh) - _thin_ch(-1.0, 0.0, xh)) / (2 * eps)
    td = (_thin_ch(0.0, 1.0, xh) - _thin_ch(0.0, -1.0, xh)) / (2 * eps)
    assert ta < 0 and td < 0 and ca < 0 and cd < 0
    assert ca == pytest.approx(ta, rel=0.05)
    assert cd == pytest.approx(td, rel=0.02)


# --------------------------------------------------------------------------------------- #
# Friction (viscous)
# --------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_shear_force_sums_to_the_friction_drag_and_enters_the_hinge_moment():
    from foil_lab import viscous
    fa = SYM.with_flap(0.75, 10.0)
    r = fl.analyze(fa, 3.0, re=1e6)
    assert r.converged
    body = np.asarray(fa.coords)
    vres = viscous.solve_single(body, 3.0, 1e6)
    a = np.radians(3.0)
    drag_dir = np.array([np.cos(a), np.sin(a)])
    assert float(np.sum(vres.shear @ drag_dir)) == pytest.approx(vres.cdf, rel=1e-12)
    assert r.ch_friction is not None and r.ch_friction != 0.0
    assert r.ch == pytest.approx(r.ch_pressure + r.ch_friction, abs=1e-15)
    assert r.ch < 0.0
    assert abs(r.ch_friction) < 0.05 * abs(r.ch_pressure)         # friction is a small part
