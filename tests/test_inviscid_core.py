"""Tests of the new inviscid core (``foil_lab.inviscid``), M2.

Tolerances are those fixed in DECISIONS.md (M1 Q11, M2 Q1). The tests in
``tests/test_exact_solutions.py`` move onto this core when the old assembly is routed through it.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from foil_lab import inviscid as core
from foil_lab.generators import generate_naca4, generate_naca5
from foil_lab.geometry import rotate, translate

from tests.test_exact_solutions import _kt


# --------------------------------------------------------------------------------------- #
# Panel kernels against numerical quadrature of the Biot-Savart integral
# --------------------------------------------------------------------------------------- #
_P = (np.array([0.2]), np.array([0.1]), np.array([0.9]), np.array([0.45]))
# Points far from, beside, beyond and close to (0.03 off) the panel; none lies on it.
_PTS = np.array([[0.3, 0.8], [1.5, -0.2], [0.55, 0.28], [-0.4, 0.05], [0.6, 0.33]])


def _quad_velocity(px, py, strength, kind):
    x1, y1, x2, y2 = (a[0] for a in _P)
    L = np.hypot(x2 - x1, y2 - y1)

    def comp(t, i):
        sx, sy = x1 + (x2 - x1) * t / L, y1 + (y2 - y1) * t / L
        rx, ry = px - sx, py - sy
        r2 = rx * rx + ry * ry
        g = strength(t / L)
        if kind == "vortex":       # counter-clockwise point vortex
            return g * (-ry if i == 0 else rx) / (2 * np.pi * r2)
        return g * (rx if i == 0 else ry) / (2 * np.pi * r2)   # point source
    return (quad(comp, 0, L, args=(0,), epsabs=1e-14, epsrel=1e-13, limit=400)[0],
            quad(comp, 0, L, args=(1,), epsabs=1e-14, epsrel=1e-13, limit=400)[0])


def test_linear_vortex_kernel_matches_quadrature():
    ua, va, ub, vb = core.panel_velocity(np, _PTS[:, 0], _PTS[:, 1], *_P)
    for k, (px, py) in enumerate(_PTS):
        assert np.allclose((ua[k, 0], va[k, 0]), _quad_velocity(px, py, lambda s: 1 - s, "vortex"),
                           atol=1e-12, rtol=0)
        assert np.allclose((ub[k, 0], vb[k, 0]), _quad_velocity(px, py, lambda s: s, "vortex"),
                           atol=1e-12, rtol=0)


def test_constant_source_kernel_matches_quadrature():
    u, v = core.source_panel_velocity(np, _PTS[:, 0], _PTS[:, 1], *_P)
    for k, (px, py) in enumerate(_PTS):
        assert np.allclose((u[k, 0], v[k, 0]), _quad_velocity(px, py, lambda s: 1.0, "source"),
                           atol=1e-12, rtol=0)


# --------------------------------------------------------------------------------------- #
# Single body
# --------------------------------------------------------------------------------------- #
def test_symmetric_section_has_no_lift_at_zero_incidence():
    s = core.solve(core.PanelSet.from_bodies(generate_naca4("0012", 160, trailing_edge="closed")), 0.0)
    assert abs(float(s.circulation[0])) < 1e-12
    assert abs(core.body_loads(s, 0).cl) < 1e-12


def test_pressure_lift_equals_circulation_lift_closed_te():
    """Within 0.5 % (M1 Q11), closed trailing edge (the open edge is pending, DECISIONS.md M2)."""
    s = core.solve(core.PanelSet.from_bodies(generate_naca4("2412", 200, trailing_edge="closed")), 4.0)
    assert abs(core.body_loads(s, 0).cl / (-2.0 * float(s.circulation[0])) - 1.0) <= 0.005


def _kt_errors(n, alpha):
    c, zm, cpe = _kt(n, alpha)
    sol = core.solve(core.PanelSet.from_bodies(c), alpha)
    cpn = core.to_selig_order(sol, sol.cp, 0)
    xc = (zm.real - c[:, 0].min()) / (c[0, 0] - c[:, 0].min())
    e = np.abs(cpn - cpe)[(xc >= 0.02) & (xc <= 0.98)]
    return float(e.max()), float(np.sqrt(np.mean(e ** 2)))


@pytest.mark.parametrize("alpha", [0.0, 4.0])
def test_karman_trefftz_new_core(alpha):
    """max |dCp| <= 0.02 at 200 panels and falling from 160 to 320 (M1 Q11); observed order of
    the same error measure >= 1.5 from 100, 200 and 400 panels (M2 Q1)."""
    e = {n: _kt_errors(n, alpha)[0] for n in (100, 160, 200, 320, 400)}
    assert e[200] <= 0.02
    assert e[320] < e[160]
    assert np.log2(e[100] / e[200]) >= 1.5
    assert np.log2(e[200] / e[400]) >= 1.5


# --------------------------------------------------------------------------------------- #
# Two bodies (data-free); the Williams exact solution follows
# --------------------------------------------------------------------------------------- #
def _cls(bodies, alpha):
    s = core.solve(core.PanelSet.from_bodies(bodies), alpha)
    return [core.body_loads(s, k).cl for k in range(len(bodies))]


def test_two_different_bodies_stacked_far_apart_new_core():
    """NACA 0012 and 4412, 50 chords apart across the stream: each Cl within 0.5 % of isolated."""
    a = generate_naca4("0012", 160, trailing_edge="closed")
    b = generate_naca4("4412", 160, trailing_edge="closed")
    ca, cb = _cls([a, translate(b, 0.0, 50.0)], 4.0)
    assert abs(ca / _cls([a], 4.0)[0] - 1.0) <= 0.005
    assert abs(cb / _cls([b], 4.0)[0] - 1.0) <= 0.005


def test_mirrored_pair_new_core():
    """A section and its mirror image about y = 0 in a horizontal stream: Cl1 = -Cl2 (1e-9)."""
    real = translate(rotate(generate_naca4("2412", 160, trailing_edge="closed"), 4.0,
                            pivot=(0.25, 0.0)), 0.0, 0.5)
    c1, c2 = _cls([real, real * np.array([1.0, -1.0])], 0.0)
    assert abs(c1 + c2) <= 1e-9


# --------------------------------------------------------------------------------------- #
# NumPy and JAX run the same assembly
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("trailing_edge", ["closed", "open"])
@pytest.mark.parametrize("name", ["0012", "2412", "4412", "23012"])
def test_numpy_and_jax_same_core(name, trailing_edge):
    """Inviscid Cl agrees to 1e-6 at -4, 4 and 10 deg; JAX's global x64 setting is unchanged."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    gen = generate_naca5 if len(name) == 5 else generate_naca4
    ps = core.PanelSet.from_bodies(gen(name, 200, trailing_edge=trailing_edge))
    before = jax.config.x64_enabled
    for alpha in (-4.0, 4.0, 10.0):
        cl_np = core.body_loads(core.solve(ps, alpha), 0).cl
        with jax.enable_x64(True):
            sj = core.solve(ps, alpha, xp=jnp)
            sj.cp = np.asarray(sj.cp)
        assert abs(cl_np - core.body_loads(sj, 0).cl) <= 1e-6
    assert jax.config.x64_enabled == before
