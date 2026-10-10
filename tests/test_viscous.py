"""Checkpoints C2 and C4: the coupled viscous-inviscid Newton solver with the wake (M3)."""

import dataclasses
import warnings

import numpy as np
import pytest

from foil_lab import viscous as V
from foil_lab.benchmark.sections import canonical
from foil_lab.params import VISCOUS

SETS = ("fidkowski2022", "drela1987")


@pytest.fixture(scope="module")
def naca2412():
    coords = canonical("NACA2412").coords
    return coords, V.build_geometry(coords)


def _off_solution_state(coords, geo, alpha, cs):
    """A converged state (fidkowski2022, also used as the test point of the drela1987 residual,
    whose Jacobian is checked the same way), perturbed by 0.1 % so the Jacobian is checked off the
    solution."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = V.solve_viscous(coords, alpha, 1e6, VISCOUS, geometry=geo)
    assert r.converged
    x, st = r.state
    V.finalize(st)
    x = x * (1.0 + 1e-3 * np.random.default_rng(0).standard_normal(x.shape))
    return x, st, dataclasses.replace(VISCOUS, closure_set=cs)


@pytest.mark.slow
@pytest.mark.parametrize("cs", SETS)
def test_full_jacobian_complex_step_matches_finite_differences(naca2412, cs):
    """Complex step of the local residuals with the chain rule through s_stag(ue) equals central
    finite differences of the full residual (a sample of the 4 Mt columns, airfoil and wake), to 1e-5 of the column."""
    coords, geo = naca2412
    x, st, prm = _off_solution_state(coords, geo, 5.0, cs)
    j = V.jacobian_complex_step(x, st, 1e6, prm)
    xf = x.ravel()
    m = geo.ps.num_nodes
    # a random sample plus every variable of both trailing-edge stations and the first two wake
    # stations (FID eq. 14 couples them)
    must = np.concatenate([4 * j + np.arange(4) for j in (0, m - 1, m, m + 1)])
    for k in np.concatenate([must, np.random.default_rng(1).choice(xf.size, 60, replace=False)]):
        h = 1e-6 * max(abs(xf[k]), 1e-8)
        xp_, xm = xf.copy(), xf.copy()
        xp_[k] += h
        xm[k] -= h
        fd = (V.residual(xp_, st, 1e6, prm) - V.residual(xm, st, 1e6, prm)) / (2 * h)
        assert np.max(np.abs(fd - j[:, k])) <= 1e-5 * max(np.max(np.abs(fd)), 1e-12), k


@pytest.mark.slow
@pytest.mark.parametrize("cs", SETS)
def test_full_jacobian_jax_matches_complex_step(naca2412, cs):
    """JAX forward-mode differentiation of the same full residual agrees with the complex-step
    Jacobian to 1e-8 of each column."""
    pytest.importorskip("jax")
    coords, geo = naca2412
    x, st, prm = _off_solution_state(coords, geo, 5.0, cs)
    jc = V.jacobian_complex_step(x, st, 1e6, prm)
    jj = V.jacobian_jax(x, st, 1e6, prm)
    assert np.max(np.abs(jj - jc) / (np.max(np.abs(jc), axis=0) + 1e-300)) <= 1e-8


@pytest.mark.slow
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
    # the step into the residual tolerance is quadratic; a further iteration, taken only for the
    # update criterion, sits at the round-off floor and is not judged
    k = next(i for i, v in enumerate(h) if v <= VISCOUS.residual_tol)
    assert h[k - 1] <= 1e-4 and h[k] <= 10 * h[k - 1] ** 1.8


@pytest.mark.slow
def test_solution_does_not_depend_on_the_start(naca2412):
    """5 deg from the starting march and from the 3 deg solution: same Cl, Cd, Cm, transition.
    The 3 deg solution is a polar point (``solve_point``): since the upwinding of FID eq. (13) its
    start from the march stalls on an isolated H_k clip and is reached by the repair restart."""
    coords, geo = naca2412
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold = V.solve_viscous(coords, 5.0, 1e6, geometry=geo)
        r3 = V.solve_point(coords, 3.0, 1e6, geometry=geo)
        warm = V.solve_continued(coords, 5.0, 1e6, start=r3.state, alpha_start=3.0, geometry=geo)
    assert r3.converged and cold.converged and warm.converged
    for q in ("cl", "cd", "cd_te", "cm", "xtr_upper"):
        assert abs(getattr(cold, q) - getattr(warm, q)) <= 1e-7, q


@pytest.mark.slow
def test_stagnation_point_on_a_node():
    """NACA 0012 at 0 deg: the stagnation point lies on the leading-edge node to round-off (FID
    p. 7). The solution converges, is symmetric (Cl = 0, equal layers on both surfaces), and the
    complex-step Jacobian there matches finite differences, including the node's rows."""
    coords = canonical("NACA0012").coords
    geo = V.build_geometry(coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = V.solve_viscous(coords, 0.0, 1e6, geometry=geo)
    assert r.converged and abs(r.cl) < 1e-10
    up, lo = r.surface["upper"], r.surface["lower"]
    n = min(up["theta"].size, lo["theta"].size) - 1
    assert np.allclose(up["theta"][-n:], lo["theta"][-n:], rtol=1e-8)
    assert r.xtr_upper == pytest.approx(r.xtr_lower, abs=1e-8)
    x, st = r.state
    assert sum(b.k0 for b in st.branches) == 1
    V.finalize(st)
    x = x * (1.0 + 1e-3 * np.random.default_rng(2).standard_normal(x.shape))
    j = V.jacobian_complex_step(x, st, 1e6, VISCOUS)
    m = geo.ps.num_nodes
    le = [b.nodes[0] for b in st.branches] + [b.nodes[1] for b in st.branches]
    xf = x.ravel()
    for k in np.concatenate([4 * s + np.arange(4) for s in le + [0, m - 1, m, m + 1, m + 5]]):
        # ue at the node on the stagnation point is ~1e-15: a step relative to it would move s_stag
        # below round-off, so ue columns are stepped relative to 0.1 (the residual is smooth in ue
        # through zero there; the node's own rows do not use its xi)
        h = 1e-6 * max(abs(xf[k]), 0.1 if k % 4 == 3 else 1e-8)
        xp_, xm = xf.copy(), xf.copy()
        xp_[k] += h
        xm[k] -= h
        fd = (V.residual(xp_, st, 1e6, VISCOUS) - V.residual(xm, st, 1e6, VISCOUS)) / (2 * h)
        assert np.max(np.abs(fd - j[:, k])) <= 1e-5 * max(np.max(np.abs(fd)), 1e-12), k


def test_isolated_clips_finds_one_station_spikes():
    """A turbulent station on the H_k clip between two turbulent neighbours above it is found;
    a clipped station next to another clipped or a laminar station is not."""
    from types import SimpleNamespace
    prm = VISCOUS
    th = np.full(8, 1e-3)
    ds = 1.4 * th
    ds[3] = prm.hk_min * th[3]                       # isolated spike
    ds[6] = ds[7] = prm.hk_min * th[6]               # two adjacent clipped stations
    turb = np.array([False, True, True, True, True, False, True, True])
    st = SimpleNamespace(branches=[SimpleNamespace(nodes=np.arange(8))], turbulent=turb)
    x = np.stack([th, ds, np.zeros(8), np.ones(8)], axis=1)
    assert V.isolated_clips(x, st, prm).tolist() == [3]


@pytest.mark.slow
def test_restart_reaches_the_cold_start_solution():
    """NACA 4412, Re 5e5: -1 deg from the converged 0 deg stalls with one turbulent station on the
    H_k clip (C3); the restarts of solve_point reach the solution of a cold start at -1 deg, and the
    result says how (DECISIONS.md M3, decision 3c after report 6)."""
    coords = canonical("NACA4412").coords
    geo = V.build_geometry(coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0, r1 = V.polar(coords, [0.0, -1.0], 5e5, geometry=geo)
        cold = V.solve_viscous(coords, -1.0, 5e5, geometry=geo)
    assert r0.converged and r1.converged and cold.converged
    assert "restart" in r1.path and r1.iterations_total > r1.iterations
    for k in ("cl", "cd", "cm", "xtr_upper", "xtr_lower"):
        assert getattr(r1, k) == pytest.approx(getattr(cold, k), abs=1e-6)
