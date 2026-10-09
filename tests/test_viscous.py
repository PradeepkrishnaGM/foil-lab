"""Checkpoint C2: the coupled viscous-inviscid Newton solver (M3)."""

import dataclasses
import warnings

import numpy as np
import pytest

from foil_lab import inviscid
from foil_lab import viscous as V
from foil_lab.benchmark.sections import canonical
from foil_lab.params import VISCOUS

SETS = ("fidkowski2022", "drela1987")


@pytest.fixture(scope="module")
def naca2412():
    coords = canonical("NACA2412").coords
    return coords, V.build_geometry(coords)


def _off_solution_state(coords, geo, alpha, cs):
    """A converged state, perturbed by 0.1 % so the Jacobian is checked off the solution."""
    prm = dataclasses.replace(VISCOUS, closure_set=cs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = V.solve_viscous(coords, alpha, 1e6, prm, geometry=geo)
    assert r.converged
    x, st = r.state
    g = np.asarray(inviscid.solve(geo.ps, alpha, assembly=geo.assembly).gamma)
    V.finalize(st, geo.ps.num_nodes)
    x = x * (1.0 + 1e-3 * np.random.default_rng(0).standard_normal(x.shape))
    return x, st, -st.d * g, prm


@pytest.mark.slow
@pytest.mark.parametrize("cs", SETS)
def test_full_jacobian_complex_step_matches_finite_differences(naca2412, cs):
    """Complex step of the local residuals with the chain rule through s_stag(ue) equals central
    finite differences of the full residual (all 4M columns of a sample), to 1e-5 of the column."""
    coords, geo = naca2412
    x, st, ue_inv, prm = _off_solution_state(coords, geo, 5.0, cs)
    j = V.jacobian_complex_step(x, geo, st, ue_inv, 1e6, prm)
    xf = x.ravel()
    for k in np.random.default_rng(1).choice(xf.size, 60, replace=False):
        h = 1e-6 * max(abs(xf[k]), 1e-8)
        xp_, xm = xf.copy(), xf.copy()
        xp_[k] += h
        xm[k] -= h
        fd = (V.residual(xp_, geo, st, ue_inv, 1e6, prm) - V.residual(xm, geo, st, ue_inv, 1e6, prm)) / (2 * h)
        assert np.max(np.abs(fd - j[:, k])) <= 1e-5 * max(np.max(np.abs(fd)), 1e-12), k


@pytest.mark.slow
@pytest.mark.parametrize("cs", SETS)
def test_full_jacobian_jax_matches_complex_step(naca2412, cs):
    """JAX forward-mode differentiation of the same full residual agrees with the complex-step
    Jacobian to 1e-8 of each column."""
    pytest.importorskip("jax")
    coords, geo = naca2412
    x, st, ue_inv, prm = _off_solution_state(coords, geo, 5.0, cs)
    jc = V.jacobian_complex_step(x, geo, st, ue_inv, 1e6, prm)
    jj = V.jacobian_jax(x, geo, st, ue_inv, 1e6, prm)
    assert np.max(np.abs(jj - jc) / (np.max(np.abs(jc), axis=0) + 1e-300)) <= 1e-8


def test_converges_quadratically_at_5deg(naca2412):
    """Cold start at 5 deg converges to the tolerances within 20 iterations, the last steps with
    full Newton steps and quadratic reduction of the residual."""
    coords, geo = naca2412
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = V.solve_viscous(coords, 5.0, 1e6, geometry=geo)
    assert r.converged and r.iterations < 20
    h = r.residual_history
    assert h[-1] <= VISCOUS.residual_tol
    assert h[-2] <= 1e-4 and h[-1] <= 10 * h[-2] ** 1.8


@pytest.mark.slow
def test_solution_does_not_depend_on_the_start(naca2412):
    """4 deg from the starting march and from the 2 deg solution: same Cl, Cd, Cm, transition."""
    coords, geo = naca2412
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold = V.solve_viscous(coords, 4.0, 1e6, geometry=geo)
        r2 = V.solve_viscous(coords, 2.0, 1e6, geometry=geo)
        warm = V.solve_continued(coords, 4.0, 1e6, start=r2.state, alpha_start=2.0, geometry=geo)
    assert cold.converged and warm.converged
    for q in ("cl", "cd", "cm", "xtr_upper"):
        assert abs(getattr(cold, q) - getattr(warm, q)) <= 1e-7, q
