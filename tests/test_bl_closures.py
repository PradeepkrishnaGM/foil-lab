"""Checkpoint C0: closure sets and discrete equations of the two-equation boundary layer (M3).

Every check runs for each closure set of :mod:`foil_lab.bl.closure_sets`.
"""

import numpy as np
import pytest

from foil_lab.bl import closure_sets, drela1987, fidkowski2022
from foil_lab.bl import equations as eq
from foil_lab.bl.march import march

SETS = list(closure_sets.CLOSURE_SETS)


def test_registry_and_provenance():
    """Each set implements the whole interface and names a source for every closure."""
    for name, m in closure_sets.CLOSURE_SETS.items():
        assert m.NAME == name
        for key in ("hk", "hstar_lam", "cf_lam", "cd_lam", "cf_turb", "hstar_turb", "us", "delta",
                    "ctau_eq", "cd_turb", "lag", "dn_dxi", "ctau_transition"):
            assert m.PROVENANCE.get(key), (name, key)
    with pytest.raises(ValueError):
        closure_sets.get("no-such-set")


# --------------------------------------------------------------------------------------- #
# Closures: continuity at branch points, differentiability
# --------------------------------------------------------------------------------------- #
BRANCH_POINTS = {
    "drela1987": [("hstar_lam", 4.0, 500.0), ("cf_lam", 7.4, 500.0), ("cd_lam", 4.0, 500.0),
                  ("hstar_turb", 4.0, 300.0), ("hstar_turb", 3.2, 2000.0)],
    "fidkowski2022": [("hstar_lam", 4.35, 500.0), ("cf_lam", 5.5, 500.0), ("cd_lam", 4.0, 500.0),
                      ("hstar_turb", 4.0, 300.0), ("hstar_turb", 3.2, 2000.0)],
}


@pytest.mark.parametrize("cs", SETS)
def test_piecewise_closures_are_continuous(cs):
    m = closure_sets.get(cs)
    for fname, at, ret in BRANCH_POINTS[cs]:
        f = getattr(m, fname)
        e = 1e-9
        assert abs(f(at - e, ret) - f(at + e, ret)) <= 1e-7, (cs, fname, at)


def test_dg87_h0_continuous_at_400():
    assert abs(drela1987.h0_turb(400.0 - 1e-9) - drela1987.h0_turb(400.0 + 1e-9)) <= 1e-8


@pytest.mark.parametrize("cs", SETS)
@pytest.mark.parametrize("h", [1.4, 2.2, 2.6, 3.4, 4.5, 6.0, 8.0])
def test_closure_complex_step_matches_central_difference(cs, h):
    # Points avoid branch points (H_0 = 3 + 400/Re_theta = 3.5 at Re_theta = 800, where the
    # DG87 turbulent H* has an unbounded second derivative; FID's 4.35 and 5.5).
    m = closure_sets.get(cs)
    ret = 800.0
    funs = [lambda x: m.hstar_lam(x, ret), lambda x: m.cf_lam(x, ret), lambda x: m.cd_lam(x, ret),
            lambda x: m.hstar_turb(x, ret), lambda x: m.cf_turb(x, ret),
            lambda x: m.dn_dxi(x, 1e-3, ret, 5.0, 9.0)]
    for f in funs:
        cs_d = np.imag(f(h + 1e-30j)) / 1e-30
        fd = (f(h + 1e-6) - f(h - 1e-6)) / 2e-6
        assert abs(cs_d - fd) <= 1e-6 * max(1.0, abs(fd))


# --------------------------------------------------------------------------------------- #
# Interval residuals: JAX Jacobian == complex step == finite differences
# --------------------------------------------------------------------------------------- #
CASES = {
    "laminar": (2.4e-4, 6.2e-4, 3.0, 1.25, 2.6e-4, 6.9e-4, 3.4, 1.22, 0.02),
    "turbulent": (1.0e-3, 1.5e-3, 0.004, 1.05, 1.1e-3, 1.7e-3, 0.0042, 1.02, 0.02),
    "transition": (3.2e-4, 1.0e-3, 8.6, 1.20, 3.37e-4, 9.0e-4, 0.0037, 1.19, 0.0045),
}


def _residual(kind, v, re, cs, xp=np):
    t1, d1, c1, u1, t2, d2, c2, u2, dxi = v
    xi1 = 0.05
    if kind == "transition":
        r, _ = eq.transition_interval(t1, d1, c1, u1, t2, d2, c2, u2, xi1, xi1 + dxi, re, 9.0, xp=xp, cs=cs)
        return r
    turb = kind == "turbulent"
    s1 = eq.station(t1, d1, c1, u1, re, turb, xp, cs)
    s2 = eq.station(t2, d2, c2, u2, re, turb, xp, cs)
    if turb:
        return eq.turbulent_interval(s1, s2, xi1, xi1 + dxi, xp=xp, cs=cs)
    return eq.laminar_interval(s1, s2, xi1, xi1 + dxi, xp=xp, cs=cs)


@pytest.mark.parametrize("cs", SETS)
@pytest.mark.parametrize("kind", ["laminar", "turbulent", "transition"])
def test_interval_jacobians_agree(cs, kind):
    """JAX automatic differentiation and complex step agree to 1e-9 (relative) and both match
    central finite differences to 1e-5 (relative), with respect to all 8 station variables."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    re = 1e6
    v = np.array(CASES[kind][:8], dtype=float)
    dxi = CASES[kind][8]

    def f_np(x):
        return _residual(kind, list(x) + [dxi], re, cs)
    j_cs = np.empty((3, 8))
    for k in range(8):
        xc = v.astype(complex)
        xc[k] += 1e-30j
        j_cs[:, k] = np.imag(f_np(xc)) / 1e-30
    j_fd = np.empty((3, 8))
    for k in range(8):
        h = 1e-6 * abs(v[k])
        xp_, xm = v.copy(), v.copy()
        xp_[k] += h
        xm[k] -= h
        j_fd[:, k] = (f_np(xp_) - f_np(xm)) / (2 * h)
    with jax.enable_x64(True):
        j_ad = np.asarray(jax.jacfwd(lambda x: _residual(kind, [x[i] for i in range(8)] + [dxi], re,
                                                         cs, jnp))(jnp.asarray(v)))
    scale = np.maximum(np.abs(j_cs), 1e-8)
    assert np.max(np.abs(j_ad - j_cs) / scale) <= 1e-9
    assert np.max(np.abs(j_fd - j_cs) / np.maximum(np.abs(j_cs), 1e-3)) <= 1e-5


# --------------------------------------------------------------------------------------- #
# SPEC 4.2 flat-plate checks on the new boundary layer (tolerances of the v1.3.1 tests)
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("cs", SETS)
def test_blasius_laminar_flat_plate_new_bl(cs):
    re = 1e6
    s = np.linspace(1e-4, 1.0, 400)
    r = march(s, re, ue=np.ones_like(s), n_crit=1e9, cs=cs)
    i = int(np.argmin(np.abs(s - 0.1)))
    assert abs(r.theta[i] / (0.664 * np.sqrt(0.1 / re)) - 1.0) < 0.05
    assert abs(r.dstar[i] / (1.721 * np.sqrt(0.1 / re)) - 1.0) < 0.05
    assert 2.5 < r.h[i] < 2.7
    assert not r.failed


@pytest.mark.parametrize("cs", SETS)
def test_turbulent_flat_plate_cf_new_bl(cs):
    re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    r = march(s, re, ue=np.ones_like(s), xi_trip=0.1, cs=cs)
    for x in (0.3, 0.6, 0.9):
        i = int(np.argmin(np.abs(s - x)))
        assert abs(r.cf[i] / (0.0592 * (re * x) ** -0.2) - 1.0) < 0.10
    it = int(np.argmax(r.turbulent))
    assert 1.3 < np.mean(r.h[it:]) < 1.5
    assert abs(r.xi_transition - 0.1) <= 1e-12


# --------------------------------------------------------------------------------------- #
# Falkner-Skan reference and each set's laminar closures against it (errors as measured)
# --------------------------------------------------------------------------------------- #
def test_falkner_skan_reference_solver():
    """Shooting solver reproduces the classical values: Blasius f''(0) = 0.4696, H = 2.591;
    Hiemenz f''(0) = 1.2326, H = 2.216 (Hartree form)."""
    from tests.diagnostics.falkner_skan import solve
    b = solve(0.0)
    assert abs(b["fpp0"] - 0.4696) < 1e-4 and abs(b["h"] - 2.591) < 1e-3
    assert abs(b["ret_cf2"] - 0.664 * 0.332) < 1e-3
    s = solve(1.0)
    assert abs(s["fpp0"] - 1.2326) < 1e-4 and abs(s["h"] - 2.216) < 1e-3


#: max |error| against Falkner-Skan for H in [2.2, 2.6), [2.6, 3.0), [3.0, 3.5), [3.5, 4.0],
#: measured 2026-10-09 (DECISIONS.md M3, "Two closure sets"); recorded as they are, not targets
MEASURED_LAMINAR_ERRORS = {
    "drela1987": {"ret_cf2": (0.0106, 0.0022, 0.0060, 0.0072), "hstar": (0.0018, 0.0007, 0.0004, 0.0005),
                  "ret_2cd_hstar": (0.0001, 0.0007, 0.0010, 0.0009)},
    "fidkowski2022": {"ret_cf2": (0.0062, 0.0145, 0.0177, 0.0171), "hstar": (0.0029, 0.0068, 0.0112, 0.0130),
                      "ret_2cd_hstar": (0.0001, 0.0007, 0.0010, 0.0009)},
}


@pytest.mark.slow
def test_laminar_closures_against_falkner_skan_as_recorded():
    """Each set's laminar closure errors against the exact similarity relations equal the
    recorded values (to their rounding): a change in a closure or in the reference shows here."""
    from tests.diagnostics.laminar_closure_compare import errors
    _, res = errors()
    for cs, by_q in MEASURED_LAMINAR_ERRORS.items():
        for q, rec in by_q.items():
            got = [row[2] for row in res[q][cs]]
            assert np.allclose(got, rec, atol=5.1e-5), (cs, q, got)
