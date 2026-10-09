"""Checkpoint C0: closures and discrete equations of the two-equation boundary layer (M3)."""

import numpy as np
import pytest

from foil_lab.bl import closures as cl
from foil_lab.bl import equations as eq
from foil_lab.bl.march import march


# --------------------------------------------------------------------------------------- #
# Closures: continuity at branch points, Falkner-Skan consistency, differentiability
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("f,at,args", [
    (lambda h: cl.hstar_lam(h), 4.0, ()),
    (lambda h: cl.cf_lam(h, 500.0), 7.4, ()),
    (lambda h: cl.cd_lam(h, 500.0), 4.0, ()),
    (lambda h: cl.hstar_turb(h, 300.0), 4.0, ()),
    (lambda h: cl.hstar_turb(h, 2000.0), 3.2, ()),
])
def test_piecewise_closures_are_continuous(f, at, args):
    e = 1e-9
    assert abs(f(at - e) - f(at + e)) <= 1e-7


def test_h0_continuous_at_400():
    assert abs(cl.h0_turb(400.0 - 1e-9) - cl.h0_turb(400.0 + 1e-9)) <= 1e-8


def test_falkner_skan_flat_plate_and_stagnation():
    """m(H_k) = 0 gives the flat plate: theta sqrt(Re_x)/x within 1 % of Blasius 0.664 (the
    reference value of the SPEC 4.2 flat-plate test); m = 1 gives H_k between 2.2 and 2.25."""
    hk = cl.similarity_hk(0.0)
    assert abs(np.sqrt(cl.ell_fs(hk)) / 0.664 - 1.0) <= 0.01
    assert 2.2 < cl.stagnation_hk() < 2.25


@pytest.mark.parametrize("h", [1.4, 2.2, 2.6, 3.4, 4.5, 6.0, 8.0])
def test_closure_complex_step_matches_central_difference(h):
    # Points avoid H_0 = 3 + 400/Re_theta = 3.5 (Re_theta = 800), the branch point of the turbulent
    # H*: its slope is continuous there but the (H_0 - H)^1.6 branch has an unbounded second
    # derivative, so a central difference is not accurate at that point.
    ret = 800.0
    funs = [lambda x: cl.hstar_lam(x), lambda x: cl.cf_lam(x, ret), lambda x: cl.cd_lam(x, ret),
            lambda x: cl.hstar_turb(x, ret), lambda x: cl.cf_turb(x, ret),
            lambda x: cl.dn_dretheta(x), lambda x: cl.log10_retheta0(x)]
    for f in funs:
        cs = np.imag(f(h + 1e-30j)) / 1e-30
        fd = (f(h + 1e-6) - f(h - 1e-6)) / 2e-6
        assert abs(cs - fd) <= 1e-6 * max(1.0, abs(fd))


# --------------------------------------------------------------------------------------- #
# Interval residuals: JAX Jacobian == complex step == finite differences
# --------------------------------------------------------------------------------------- #
def _cases():
    re = 1e6
    lam = (2.4e-4, 6.2e-4, 3.0, 1.25, 2.6e-4, 6.9e-4, 3.4, 1.22, 0.02)
    tur = (1.0e-3, 1.5e-3, 0.004, 1.05, 1.1e-3, 1.7e-3, 0.0042, 1.02, 0.02)
    tra = (3.2e-4, 1.0e-3, 8.6, 1.20, 3.37e-4, 9.0e-4, 0.0037, 1.19, 0.0045)
    return re, {"laminar": lam, "turbulent": tur, "transition": tra}


def _residual(kind, v, re, xp=np):
    t1, d1, c1, u1, t2, d2, c2, u2, dxi = v
    if kind == "transition":
        r, _ = eq.transition_interval(t1, d1, c1, u1, t2, d2, c2, u2, dxi, re, 9.0, xp=xp)
        return r
    turb = kind == "turbulent"
    s1 = eq.station(t1, d1, c1, u1, re, turb, xp)
    s2 = eq.station(t2, d2, c2, u2, re, turb, xp)
    f = eq.turbulent_interval if turb else eq.laminar_interval
    return f(s1, s2, dxi, xp=xp)


@pytest.mark.parametrize("kind", ["laminar", "turbulent", "transition"])
def test_interval_jacobians_agree(kind):
    """JAX automatic differentiation and complex step agree to 1e-9 (relative) and both match
    central finite differences to 1e-5 (relative), with respect to all 8 station variables."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    re, cases = _cases()
    v = np.array(cases[kind][:8], dtype=float)
    dxi = cases[kind][8]

    def f_np(x):
        return _residual(kind, list(x) + [dxi], re)
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
                                                         jnp))(jnp.asarray(v)))
    scale = np.maximum(np.abs(j_cs), 1e-8)
    assert np.max(np.abs(j_ad - j_cs) / scale) <= 1e-9
    assert np.max(np.abs(j_fd - j_cs) / np.maximum(np.abs(j_cs), 1e-3)) <= 1e-5


# --------------------------------------------------------------------------------------- #
# SPEC 4.2 flat-plate checks on the new boundary layer (tolerances of the v1.3.1 tests)
# --------------------------------------------------------------------------------------- #
def test_blasius_laminar_flat_plate_new_bl():
    re = 1e6
    s = np.linspace(1e-4, 1.0, 400)
    r = march(s, re, ue=np.ones_like(s), n_crit=1e9)
    i = int(np.argmin(np.abs(s - 0.1)))
    assert abs(r.theta[i] / (0.664 * np.sqrt(0.1 / re)) - 1.0) < 0.05
    assert abs(r.dstar[i] / (1.721 * np.sqrt(0.1 / re)) - 1.0) < 0.05
    assert 2.5 < r.h[i] < 2.7
    assert not r.failed


def test_turbulent_flat_plate_cf_new_bl():
    re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    r = march(s, re, ue=np.ones_like(s), xi_trip=0.1)
    for x in (0.3, 0.6, 0.9):
        i = int(np.argmin(np.abs(s - x)))
        assert abs(r.cf[i] / (0.0592 * (re * x) ** -0.2) - 1.0) < 0.10
    it = int(np.argmax(r.turbulent))
    assert 1.3 < np.mean(r.h[it:]) < 1.5
    assert abs(r.xi_transition - 0.1) <= 1e-12
