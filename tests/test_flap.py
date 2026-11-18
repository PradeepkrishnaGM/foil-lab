"""Plain flap geometry (M4 step 1; SPEC.md 4.5, DECISIONS.md M4)."""

import dataclasses

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import flap as F
from foil_lab.params import GEOMETRY

AF = fl.Airfoil.naca("2412", 160)
SYM = fl.Airfoil.naca("0012", 160)
DEFLECTIONS = (-30.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 30.0)


def _signed_area(c):
    return 0.5 * float(np.sum(c[:-1, 0] * c[1:, 1] - c[1:, 0] * c[:-1, 1]))


def _turning(c):
    t = np.diff(c, axis=0)
    return np.degrees(np.abs(np.arctan2(t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0],
                                        np.sum(t[:-1] * t[1:], axis=1))))


def _crosses(c):
    try:
        F._check_simple(c)
    except ValueError:
        return True
    return False


# --------------------------------------------------------------------------------------- #
# Deflection 0 and the reference frame
# --------------------------------------------------------------------------------------- #
def test_zero_deflection_is_the_original():
    fa = AF.with_flap(0.75, 0.0)
    assert np.array_equal(fa.coords, AF.coords)
    assert fa.flap is not None and fa.flap.deflection == 0.0
    mid_x = 0.5 * (fa.coords[:-1, 0] + fa.coords[1:, 0])
    assert np.array_equal(fa.flap.flap_panels, mid_x > 0.75)


@pytest.mark.parametrize("d", (-30.0, 12.0, 30.0))
def test_reference_frame_is_the_undeflected_one(d):
    fa = AF.with_flap(0.75, d)
    assert fa.chord == AF.chord
    assert np.array_equal(fa.leading_edge, AF.leading_edge)
    assert np.array_equal(fa.quarter_chord, AF.quarter_chord)
    cfg = fl.Configuration.of(fa)
    assert cfg.reference() == fl.Configuration.of(AF).reference()


# --------------------------------------------------------------------------------------- #
# Valid geometry over the whole range
# --------------------------------------------------------------------------------------- #
def _check_valid(d, hx, hy):
    fa = AF.with_flap(hx, d, hy)
    c = fa.coords
    assert fa.num_panels == AF.num_panels
    assert np.all(np.isfinite(c))
    assert not _crosses(c)
    assert np.sign(_signed_area(c)) == np.sign(_signed_area(AF.coords))
    assert np.hypot(*np.diff(c, axis=0).T).min() > 1e-6          # no repeated nodes
    # the flap is rigid: the trailing-edge nodes are the original ones rotated about the hinge
    h = fa.hinge
    phi = np.radians(-d)
    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    for k in (0, -1):
        expect = h + rot @ (AF.coords[k] - h)
        assert np.allclose(c[k], expect, atol=1e-12)
    # the hinge lies at hinge_x in the chord frame of the undeflected section
    le, e, chord = F.geo._ArcSpline(AF.coords).chord_frame()
    assert abs(((fa.hinge - le) @ e) / chord - hx) < 1e-12


@pytest.mark.parametrize("d", (-30.0, -10.0, 30.0))
@pytest.mark.parametrize("hy", (0.0, 0.5, 1.0))
def test_valid_contour(d, hy):
    _check_valid(d, 0.75, hy)


@pytest.mark.slow
@pytest.mark.parametrize("d", DEFLECTIONS)
@pytest.mark.parametrize("hx", (0.5, 0.75, 0.9))
@pytest.mark.parametrize("hy", (0.0, 0.5, 1.0))
def test_valid_contour_full_matrix(d, hx, hy):
    _check_valid(d, hx, hy)


@pytest.mark.parametrize("d", (-30.0, -10.0, 10.0, 30.0))
def test_hinge_arc_is_circular_about_the_hinge(d):
    fa = AF.with_flap(0.75, d, 0.5)
    arc = fa.coords[:-1][fa.flap.zones == F.ZONE_ARC]
    assert len(arc) >= 1
    r = np.hypot(*(arc - fa.hinge).T)
    assert np.ptp(r) < 1e-12
    # diagnostics are in chords of the undeflected section
    assert abs(r.mean() / AF.chord - fa.flap.diagnostics["arc_radius"]) < 1e-12
    # no gap: the arc spans the deflection angle about the hinge (the ends are nodes)
    assert fa.flap.diagnostics["arc_length"] == pytest.approx(
        fa.flap.diagnostics["arc_radius"] * np.radians(abs(d)), rel=1e-12)


def test_zones_run_in_order_along_the_contour():
    fa = AF.with_flap(0.75, 15.0)
    z = fa.flap.zones
    assert len(z) == fa.num_panels
    # upper surface from the trailing edge: flap, arc, main, then lower: main, flap
    changes = [z[0]] + [b for a, b in zip(z[:-1], z[1:]) if a != b]
    assert changes == [F.ZONE_FLAP, F.ZONE_ARC, F.ZONE_MAIN, F.ZONE_FLAP]


# --------------------------------------------------------------------------------------- #
# Smoothness: tangent continuity of the pieces, and the node spacing
# --------------------------------------------------------------------------------------- #
def _junction_turns(L, hx, hy, delta_deg, prm):
    upper, lower, *_ = F._build_down(L, hx, hy, np.radians(delta_deg), prm)
    out = []
    eps = 1e-7
    for pieces in (upper, lower):
        for a, b in zip(pieces[:-1], pieces[1:]):
            ta = a.f(np.array([1.0])) - a.f(np.array([1.0 - eps]))
            tb = b.f(np.array([eps])) - b.f(np.array([0.0]))
            ta, tb = ta[0] / np.hypot(*ta[0]), tb[0] / np.hypot(*tb[0])
            out.append(np.degrees(abs(np.arctan2(ta[0] * tb[1] - ta[1] * tb[0], ta @ tb))))
    return out


@pytest.mark.parametrize("d", (2.0, 10.0, 20.0, 30.0))
@pytest.mark.parametrize("hy", (0.3, 0.5, 0.7))
def test_tangent_is_continuous_at_every_junction(d, hy):
    L = F._Local(AF.coords)
    turns = _junction_turns(L, 0.75, hy, d, GEOMETRY)
    assert max(turns) < 1e-3                                       # degrees


def test_fillet_size_zero_gives_the_plain_notch():
    prm = dataclasses.replace(GEOMETRY, flap_fillet_radius=0.0)
    L = F._Local(AF.coords)
    turns = _junction_turns(L, 0.75, 0.5, 20.0, prm)
    assert max(turns) > 15.0                                       # the notch corner (about 20 deg)
    c, info = F.make_flap(AF.coords, 0.75, 20.0, 0.5, params=prm)
    assert not _crosses(c) and info.diagnostics["fillet_deviation"] == 0.0


def test_fairings_deviation_is_reported_and_small():
    for d in (5.0, 20.0, 30.0):
        dg = AF.with_flap(0.75, d).flap.diagnostics
        assert 0.0 < dg["fairing_deviation"] < 0.002
        assert dg["fillet_deviation"] < 0.002


def test_few_panels_warn_that_the_hinge_zone_was_relaxed():
    with pytest.warns(UserWarning, match="few for the hinge zone"):
        fa = AF.with_flap(0.75, 10.0, num_panels=80)
    assert fa.flap.diagnostics["zone_spacing_relaxed"] > 1.0 and not _crosses(fa.coords)


@pytest.mark.parametrize("n", (160, 320))
def test_node_spacing_near_the_hinge(n):
    fa = AF.with_flap(0.75, 20.0, num_panels=n)
    c = fa.coords
    seg = np.hypot(*np.diff(c, axis=0).T)
    z = fa.flap.zones
    assert len(c) == n + 1
    assert fa.flap.diagnostics["arc_panels"] >= 1
    # neighbouring panel lengths change by at most the grading ratio plus a rounding allowance,
    # away from the leading-edge and trailing-edge clusters
    ratio = np.maximum(seg[1:] / seg[:-1], seg[:-1] / seg[1:])
    mid_x = 0.5 * (c[1:-1, 0] + c[2:, 0])
    away = (mid_x > 0.3) & (mid_x < 0.9)
    assert ratio[away].max() < 1.8
    # the hinge zone is finer than the cosine spacing of the unflapped section there
    base = np.hypot(*np.diff(AF.repanelled(n).coords, axis=0).T)
    bx = 0.5 * (AF.repanelled(n).coords[:-1, 0] + AF.repanelled(n).coords[1:, 0])
    near = (np.abs(bx - 0.75) < 0.02)
    assert seg[(z == F.ZONE_ARC)].max() < base[near].mean()


def test_negative_deflection_is_the_mirror_of_positive_on_a_symmetric_section():
    a = SYM.with_flap(0.75, 17.0).coords
    b = SYM.with_flap(0.75, -17.0).coords
    mirrored = a[::-1] * np.array([1.0, -1.0])
    assert np.allclose(b, mirrored, atol=1e-9)


# --------------------------------------------------------------------------------------- #
# Operations on flapped sections
# --------------------------------------------------------------------------------------- #
def test_transforms_commute_with_the_flap():
    for op in (lambda a: a.scaled(2.5), lambda a: a.translated(0.3, -0.2),
               lambda a: a.rotated(12.0, pivot=(0.3, 0.1))):
        one = op(AF.with_flap(0.7, 14.0, 0.4)).repanelled(160)
        two = op(AF).with_flap(0.7, 14.0, 0.4, num_panels=160)
        assert np.allclose(one.coords, two.coords, atol=1e-9)
        assert np.allclose(one.hinge, two.hinge, atol=1e-9)
        assert one.chord == pytest.approx(two.chord)


@pytest.mark.parametrize("op,inv", (
    (lambda a: a.scaled(2.0), lambda c: c / 2.0),
    (lambda a: a.translated(1.0, 0.5), lambda c: c - np.array([1.0, 0.5])),
))
def test_flap_does_not_depend_on_size_or_position(op, inv):
    fa = SYM.with_flap(0.75, 8.0)
    again = op(fa).repanelled(160)
    assert np.allclose(inv(again.coords), fa.coords, atol=1e-9)
    assert np.array_equal(again.flap.zones, fa.flap.zones)


def test_flap_does_not_depend_on_rotation():
    fa = SYM.with_flap(0.75, 8.0)
    again = fa.rotated(17.0, pivot=(0.3, 0.1)).repanelled(160)
    back = fl.Airfoil(again.coords).rotated(-17.0, pivot=(0.3, 0.1))
    assert np.allclose(back.coords, fa.coords, atol=1e-9)


def test_repanelling_keeps_the_hinge_refinement():
    fa = AF.with_flap(0.75, 20.0)
    again = fa.repanelled(160)
    assert np.array_equal(again.coords, fa.coords)
    fine = fa.repanelled(320)
    assert fine.num_panels == 320 and fine.flap.diagnostics["arc_panels"] >= 2
    assert fine.flap.deflection == 20.0


def test_shape_operations_act_on_the_undeflected_section():
    thin = AF.with_flap(0.75, 10.0).with_thickness_scaled(0.5)
    ref = AF.with_thickness_scaled(0.5).with_flap(0.75, 10.0)
    assert np.allclose(thin.coords, ref.coords, atol=1e-12)


def test_flap_is_deterministic():
    a = AF.with_flap(0.75, 13.0).coords
    b = AF.with_flap(0.75, 13.0).coords
    assert a.tobytes() == b.tobytes()


def test_invalid_input_has_a_clear_message():
    with pytest.raises(ValueError, match="within"):
        AF.with_flap(0.75, 31.0)
    with pytest.raises(ValueError, match="hinge_x"):
        AF.with_flap(1.0, 10.0)
    with pytest.raises(ValueError, match="hinge_y"):
        AF.with_flap(0.75, 10.0, 1.5)
    with pytest.raises(ValueError, match="room|few|thick"):
        AF.with_flap(0.99, 10.0)


def test_flapped_section_gives_lift_at_zero_angle():
    cl0 = fl.analyze(SYM, 0.0).cl
    cl = fl.analyze(SYM.with_flap(0.75, 10.0), 0.0).cl
    assert abs(cl0) < 1e-9
    assert cl > 0.2                                                # downward flap adds camber
    assert fl.analyze(SYM.with_flap(0.75, -10.0), 0.0).cl == pytest.approx(-cl, abs=1e-6)
