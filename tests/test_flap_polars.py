"""Polars across flap deflection and interpolation in the deflection angle (M4 step 6)."""

import numpy as np
import pytest

import foil_lab as fl
from foil_lab.flap_polars import FlapPolarSet, interpolate_in_deflection

D = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
ALPHAS = [0.0, 4.0, 8.0]


@pytest.fixture(scope="module")
def inviscid_set():
    return fl.flap_polars(fl.Airfoil.naca("0012", 160), 0.75, D, ALPHAS)


def test_nodes_are_reproduced_exactly(inviscid_set):
    fps = inviscid_set
    for i, d in enumerate(D):
        s = fps.at(d)
        assert not s.interpolated and s.converged.all()
        assert np.array_equal(s.cl, fps.polars[i].cl) and np.array_equal(s.ch, fps.polars[i].ch)
        assert np.array_equal(s.cm, fps.polars[i].cm)


def test_interpolated_polar_is_monotone_between_nodes_and_marked(inviscid_set):
    fps = inviscid_set
    prev = fps.at(5.0).cl
    for d in (6.0, 7.5, 9.0, 10.0):
        cur = fps.at(d).cl
        assert np.all(cur > prev)                   # Cl rises with downward deflection
        prev = cur
    s = fps.at(7.5)
    assert s.interpolated and s.converged.all() and s.deflection == 7.5
    lo, hi = fps.at(5.0).cl, fps.at(10.0).cl
    assert np.all((s.cl > lo) & (s.cl < hi))        # no overshoot (monotone interpolant)
    assert fps.value("cl", 4.0, 7.5) == pytest.approx(s.cl[1], abs=0)


def test_no_extrapolation_and_no_interpolation_in_alpha(inviscid_set):
    with pytest.raises(ValueError, match="outside the solved range"):
        inviscid_set.at(25.0)
    with pytest.raises(ValueError, match="outside the solved range"):
        inviscid_set.at(-1.0)
    with pytest.raises(ValueError, match="not an angle"):
        inviscid_set.at(10.0, [3.0])
    assert inviscid_set.at(10.0, [8.0, 0.0]).alpha.tolist() == [8.0, 0.0]


def test_two_deflections_interpolate_linearly():
    v = np.array([[0.0, 1.0], [2.0, 5.0]])
    assert interpolate_in_deflection(np.array([0.0, 10.0]), v, 5.0).tolist() == [1.0, 3.0]


def test_a_node_that_did_not_converge_is_never_interpolated():
    v = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5]])
    v[1, 0] = np.nan                       # 5 deg, first angle
    v[4, 2] = np.nan                       # 20 deg, last angle
    out = interpolate_in_deflection(D, v, 12.0)           # interval 10-15, window 5..20
    assert np.isnan(out[0]) and np.isnan(out[2]) and np.isfinite(out[1])
    out = interpolate_in_deflection(D, v, 2.0)            # interval 0-5, window 0..10
    assert np.isnan(out[0]) and np.isfinite(out[2])
    out = interpolate_in_deflection(D, v, 7.5)            # interval 5-10, window 0..15
    assert np.isnan(out[0]) and np.isfinite(out[1]) and np.isfinite(out[2])
    assert np.isnan(interpolate_in_deflection(D, v, 5.0)[0])      # a node: its own value only
    assert interpolate_in_deflection(D, v, 15.0)[2] == 0.4


def test_polar_set_validates_and_warns_on_wide_spacing():
    tab = np.zeros((3, 2))
    with pytest.warns(UserWarning, match="apart"):
        FlapPolarSet.from_tables([0.0, 12.0, 24.0], [0.0, 1.0], tab, tab, tab, tab)
    with pytest.raises(ValueError):
        FlapPolarSet.from_tables([0.0], [0.0, 1.0], tab[:1], tab[:1], tab[:1], tab[:1])
    with pytest.raises(ValueError):
        FlapPolarSet.from_tables([0.0, 5.0, 3.0], [0.0, 1.0], tab, tab, tab, tab)


def test_leave_one_out_error_is_small_for_the_smooth_inviscid_flap_polar(inviscid_set):
    loo = inviscid_set.leave_one_out(10.0)
    assert set(loo) == {"cl", "cd", "cm", "ch"}
    assert loo["cl"][2] == 3 and loo["cl"][1] < 0.02 and loo["cm"][1] < 0.01 and loo["ch"][1] < 0.002
    assert inviscid_set.leave_one_out(15.0, use=[10.0, 20.0])["cl"][1] < 0.02
    with pytest.raises(ValueError, match="not a solved"):
        inviscid_set.leave_one_out(12.0)


def test_slice_to_dicts_and_polar_hinge_moment(inviscid_set):
    rows = inviscid_set.at(7.5).to_dicts()
    assert len(rows) == 3 and rows[0]["interpolated"] and set(rows[0]) >= {"alpha", "cl", "cd", "cm", "ch"}
    p = fl.polar(fl.Airfoil.naca("0012", 160), [0.0])
    assert np.isnan(p.ch).all()                     # no flap: nan


@pytest.mark.slow
def test_viscous_flap_polars_carry_the_hinge_moment_and_interpolate():
    fps = fl.flap_polars(fl.Airfoil.naca("2412", 160), 0.75, [0.0, 10.0], [2.0], re=1e6)
    assert fps.polars[1].converged.all() and np.isfinite(fps.tables["ch"]).all()
    assert fps.tables["ch"][1, 0] < 0.0                         # lift on the flap
    mid = fps.at(5.0)
    assert mid.interpolated and mid.converged.all()
    assert mid.cl[0] == pytest.approx(0.5 * (fps.tables["cl"][0, 0] + fps.tables["cl"][1, 0]))
