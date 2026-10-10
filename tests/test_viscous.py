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
    """E387 with a 0.25 % trailing-edge gap, Re 1e6: -1 deg from the converged 0 deg fails as a warm
    start (C3 on the Track A code); the restarts of solve_point reach the solution of a cold start at
    -1 deg, and the result says how (DECISIONS.md M3, decision 3c after report 6). The case was NACA
    4412, Re 5e5, -1 deg until the continuity rule of report 9, with which that point converges
    directly and its cold start no longer converges (decision 4 after report 9)."""
    from foil_lab.benchmark.sections import SectionUnavailable
    try:
        coords = canonical("E387_GAP025").coords      # downloaded section (UIUC, cached)
    except SectionUnavailable as exc:
        pytest.skip(str(exc))
    geo = V.build_geometry(coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0, r1 = V.polar(coords, [0.0, -1.0], 1e6, geometry=geo)
        cold = V.solve_viscous(coords, -1.0, 1e6, geometry=geo)
    assert r0.converged and r1.converged and cold.converged
    assert "restart" in r1.path and r1.iterations_total > r1.iterations
    for k in ("cl", "cd", "cm", "xtr_upper", "xtr_lower"):
        assert getattr(r1, k) == pytest.approx(getattr(cold, k), abs=1e-6)


def test_edge_velocity_tie_only_on_an_open_trailing_edge():
    """The first wake station takes the trailing-edge speed (``tie``) on a body with an open
    trailing edge; a closed (sharp) edge keeps the wake-line velocity there (DECISIONS.md M3,
    decisions after report 9, interim state)."""
    from foil_lab.generators import generate_naca4
    for coords, is_open in ((canonical("NACA2412").coords, True),
                            (generate_naca4("2412", 160, trailing_edge="closed"), False)):
        geo = V.build_geometry(coords)
        assert geo.ps.open_te == (is_open,)
        cpl = V.build_coupling(geo, 5.0)
        st = V.make_structure(geo, cpl, cpl.gamma_inv)
        w0 = cpl.wake_nodes[0][0]
        m = geo.ps.num_nodes
        if is_open:
            assert st.tie[w0, 0] == st.tie[w0, m - 1] == 0.5 and np.count_nonzero(st.tie) == 2
            assert st.ue_inv[w0] == 0.0 and not st.dmat[w0].any()
        else:
            assert not st.tie.any()
            assert st.ue_inv[w0] == pytest.approx(cpl.ue_inv_wake[0]) and st.dmat[w0].any()


@pytest.mark.slow
def test_closed_trailing_edge_result_warns_about_wake_drag():
    """A viscous result on a closed trailing edge carries the wake-drag warning; ``cd`` is still the
    wake-end Squire-Young value (no substitution). An open edge carries no warning."""
    from foil_lab.generators import generate_naca4
    closed = generate_naca4("2412", 160, trailing_edge="closed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = V.solve_viscous(closed, 5.0, 1e6)
        ro = V.solve_viscous(canonical("NACA2412").coords, 5.0, 1e6)
    assert rc.converged and ro.converged
    assert rc.warnings == [V.WARN_CLOSED_TE_DRAG] and ro.warnings == []
    w = rc.surface["wake"]
    assert rc.cd == 2.0 * w["theta"][-1] * w["ue"][-1] ** ((w["h"][-1] + 5.0) / 2.0)
    assert rc.cd != rc.cd_te


def test_reach_from_nearest_walks_from_the_nearest_converged_angle(monkeypatch):
    """Walking logic of ``reach_from_nearest`` (DECISIONS.md M3, decision 3 after report 8), with a
    stand-in for ``solve_point`` that converges from a start except at the angles in ``bad``: cold
    points are reached from the nearest converged angle one sweep angle at a time, the walk stops at
    the first failure, a point that failed from a warm start is not walked to, and the path and the
    total iterations record how each point was reached."""
    calls, bad = [], {7.0}

    def fake(coords, alpha, reynolds, params=None, trips=None, start=None, alpha_start=None,
             geometry=None):
        calls.append((alpha, alpha_start))
        ok = start is not None and alpha not in bad
        return V.ViscousResult(alpha=alpha, reynolds=reynolds, converged=ok, iterations=5,
                               iterations_total=7, reason="converged" if ok else "max_iter",
                               state=("x", alpha) if ok else None, path="warm start")

    monkeypatch.setattr(V, "solve_point", fake)
    alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    conv = {2.0, 3.0, 5.0}
    res = [V.ViscousResult(alpha=a, reynolds=1e6, converged=a in conv, iterations=4, iterations_total=10,
                           reason="", state=("x", a), path="march") for a in alphas]
    cold = [a in (0.0, 1.0, 2.0, 8.0) for a in alphas]       # 4 and 6-7 failed from a warm start
    seen = []
    V.reach_from_nearest(None, alphas, res, cold, 1e6, geometry=object(),
                         on_result=lambda k, r, s: seen.append((k, r.converged)))
    # 0 deg: nearest converged is 2 deg -> walk 1, 0; 8 deg: nearest converged 5 deg -> 6 ok, 7 fails
    assert calls == [(1.0, 2.0), (0.0, 1.0), (6.0, 5.0), (7.0, 6.0)]
    assert seen == [(1, True), (0, True), (6, True), (7, False)]
    assert [r.converged for r in res] == [True, True, True, True, False, True, True, False, False]
    assert res[0].path == res[1].path == "from 2 deg: warm start"
    assert res[6].path == "from 5 deg: warm start"
    assert res[1].iterations == 5 and res[1].iterations_total == 7 + 10
    assert res[4].path == "march" and res[8].path == "march"   # not walked to / walk stopped before
