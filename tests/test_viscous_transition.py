"""Transition-interval cycles in the coupled solver (DECISIONS.md v2.1, T1).

At low Reynolds number with a long laminar bubble the interval holding transition onset can switch
back and forth between Newton iterations. The solver then limits its motion to one node per
iteration, pins it after a two-cycle and accepts the result only if the amplification of the
converged state agrees with the pinned interval (:func:`foil_lab.viscous.transition_mismatch`).
"""

import dataclasses
import warnings

import numpy as np
import pytest

from foil_lab import viscous as V
from foil_lab.benchmark.sections import canonical
from foil_lab.params import VISCOUS


def _branch(p_tr=None, forced_w=None):
    return V.Branch(0, True, np.arange(80), p_tr=p_tr, forced_w=forced_w)


def _st(branches):
    return V.Structure(None, np.array([0]), np.ones(1), branches, np.zeros(1, dtype=bool))


def test_reversals_count_changes_of_direction_only():
    assert V._reversals([]) == 0
    assert V._reversals([50, 51, 52, 53]) == 0
    assert V._reversals([50, 52, 50]) == 1
    assert V._reversals([50, 50, 52, 52, 50, 51]) == 2       # repeats are not moves
    assert V._reversals([50, 52, None, 50, 52]) == 0       # a step into or out of "no transition" is not a move


def _feed(seq, res, prm=VISCOUS, **kw):
    b = _branch(**kw)
    st, ph, rh = _st([b]), [[]], []
    pinned_at = None
    for p, r in zip(seq, res):
        b.p_tr = p
        if V._freeze_two_cycles(st, ph, rh, prm) and pinned_at is None:
            pinned_at = len(rh)
        rh.append(r)
    return b, pinned_at


def test_two_cycle_between_neighbours_is_pinned_to_the_smaller_residual():
    # 52 <-> 53 from the start; 52's iterations have the smaller residuals
    b, at = _feed([53, 52, 53, 52, 53, 52, 53], [0.24, 0.19, 0.21, 0.17, 0.30, 0.18, 0.2])
    assert b.pin == 52 and b.limited
    assert at == 5                       # needs transition_cycle_switches (4) switches plus one more entry
    b, _ = _feed([53, 52, 53, 52, 53, 52], [0.1, 0.19, 0.1, 0.17, 0.1, 0.18])
    assert b.pin == 53


def test_no_pin_for_jumps_over_a_node_forced_trips_or_when_disabled():
    b, _ = _feed([50, 52, 50, 52, 50, 52, 50], [1.0] * 7)
    assert b.pin is None and b.limited   # oscillating: limited to one node per iteration, not pinned
    b, _ = _feed([53, 52, 53, 52, 53, 52, 53], [1.0] * 7, forced_w=0.5)
    assert b.pin is None
    off = dataclasses.replace(VISCOUS, transition_cycle_switches=0)
    b, _ = _feed([53, 52, 53, 52, 53, 52, 53], [1.0] * 7, prm=off)
    assert b.pin is None


def test_a_moving_interval_is_neither_limited_nor_pinned():
    b, _ = _feed([60, 58, 56, 55, 54, 54, 54], [1.0] * 7)
    assert b.pin is None and not b.limited


@pytest.fixture(scope="module")
def naca4412_re5e4():
    """NACA 4412, Re 5e4, 5 deg warm-started from 4 deg: the v2.0 release run did not converge here
    (transition interval two-cycle, docs/study_re5e4.md)."""
    coords = canonical("NACA4412").coords
    geo = V.build_geometry(coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r4 = V.solve_viscous(coords, 4.0, 5e4, geometry=geo)
        r5 = V.solve_viscous(coords, 5.0, 5e4, start=r4.state, geometry=geo)
        again = V.solve_viscous(coords, 5.0, 5e4, start=r4.state, geometry=geo)
    return coords, geo, r4, r5, again


@pytest.mark.slow
def test_naca4412_re5e4_5deg_converges_with_a_consistent_pinned_interval(naca4412_re5e4):
    _, _, r4, r5, again = naca4412_re5e4
    assert r4.converged and r5.converged
    x, st = r5.state
    upper = [b for b in st.branches if b.upper][0]
    assert upper.pin is not None                      # reached through the pin
    assert V.transition_mismatch(x, st, 5e4, VISCOUS) == []
    assert r5.residual <= VISCOUS.residual_tol
    # determinism: the same start gives the bit-identical result
    assert (again.cl, again.cd, again.cm, again.xtr_upper) == (r5.cl, r5.cd, r5.cm, r5.xtr_upper)
    # measured 2026-11-19: Cl 0.7318 against XFOIL 0.6987, transition x/c 0.675 against 0.694
    assert 0.65 < r5.xtr_upper < 0.70


@pytest.mark.slow
def test_mismatch_is_reported_for_a_pin_away_from_the_amplification(naca4412_re5e4):
    _, _, _, r5, _ = naca4412_re5e4
    x, st = r5.state
    x = x.copy()
    upper = [b for b in st.branches if b.upper][0]
    saved = upper.pin, upper.p_tr
    try:
        upper.pin = upper.p_tr = saved[1] + 3
        bad = V.transition_mismatch(x, st, 5e4, VISCOUS)
        assert len(bad) == 1 and bad[0][0] is upper and bad[0][1] < upper.pin
    finally:
        upper.pin, upper.p_tr = saved


@pytest.mark.slow
def test_unpinned_results_have_no_mismatch(naca4412_re5e4):
    _, _, r4, _, _ = naca4412_re5e4
    x, st = r4.state
    assert all(b.pin is None for b in st.branches)
    assert V.transition_mismatch(x, st, 5e4, VISCOUS) == []


@pytest.mark.slow
def test_march_guard_rejects_a_spurious_transition_root():
    """NACA 2412 (generator, 120 panels), Re 3e6, 0 deg: without the guards the starting march's
    transition interval on the lower surface lands on a root with H 457 and every station after it
    inherits it (v2.0: Newton diverged to Cl -5e6). With them the starting state stays physical
    (DECISIONS.md v2.1, T2)."""
    from foil_lab.generators import generate_naca4
    geo = V.build_geometry(generate_naca4("2412", 120))
    cpl = V.build_coupling(geo, 0.0)
    m = geo.ps.num_nodes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_off, _ = V._initial_state(geo, cpl, 3e6, dataclasses.replace(VISCOUS, init_march_guards=False), None)
        x_on, _ = V._initial_state(geo, cpl, 3e6, VISCOUS, None)
    assert np.max(x_off[:m, 1] / x_off[:m, 0]) > 400.0           # the v2.0 defect, documented
    assert np.max(x_on[:m, 1] / x_on[:m, 0]) < 20.0
