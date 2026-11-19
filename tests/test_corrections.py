"""M6: compressibility, ground effect, sweep and aspect-ratio corrections (SPEC.md 4.2, 4.7)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import inviscid as core
from foil_lab.corrections import (critical_cp, critical_mach, helmbold_factor,
                                  karman_tsien_correction)
from foil_lab.geometry import rotate

AF = fl.Airfoil.naca("2412")


def test_no_option_no_correction():
    """Without options nothing is corrected and the result is bit-identical to the plain solve."""
    r = fl.analyze(AF, 4.0)
    assert r.corrections == () and r.mach == 0.0 and r.cd_induced is None
    b = np.asarray(AF.repanelled(160).coords)
    ps = core.PanelSet.from_bodies([b])
    cl, _, _ = core.pressure_loads(ps, np.asarray(core.solve(ps, 4.0).cp), 4.0, 0, r.ref_chord, r.ref_point)
    assert r.cl == float(cl)


def test_karman_tsien_applied_once_through_the_api():
    """Cl at Mach 0.5 is the integral of the once-corrected incompressible Cp, and the correction is
    listed once."""
    r = fl.analyze(AF, 4.0, mach=0.5)
    b = np.asarray(AF.repanelled(160).coords)
    ps = core.PanelSet.from_bodies([b])
    cp = karman_tsien_correction(np.asarray(core.solve(ps, 4.0).cp), 0.5)
    cl, _, cm = core.pressure_loads(ps, cp, 4.0, 0, r.ref_chord, r.ref_point)
    assert r.cl == pytest.approx(float(cl), abs=1e-12) and r.cm == pytest.approx(float(cm), abs=1e-12)
    assert sum("Kármán-Tsien" in c for c in r.corrections) == 1


def test_critical_mach_definition_and_warning():
    """At the reported critical Mach number the corrected minimum Cp equals the critical Cp; a Mach
    number above it gives a warning, below it none."""
    r0 = fl.analyze(AF, 2.0)
    m = r0.mach_critical
    assert 0.5 < m < 0.8
    cp_min = min(np.min(e.cp) for e in r0.elements)
    assert float(karman_tsien_correction(np.array(cp_min), m)) == pytest.approx(critical_cp(m), abs=1e-9)
    assert not fl.analyze(AF, 2.0, mach=m - 0.02).warnings
    hi = fl.analyze(AF, 2.0, mach=m + 0.02)
    assert any("critical Mach" in w for w in hi.warnings)
    assert np.isnan(critical_mach(0.2))


def test_critical_cp_known_values():
    """Isentropic critical Cp (gamma 1.4): -0.5912 at Mach 0.75, -2.1334 at Mach 0.5 (computed from
    the closed form; the sonic condition Cp*(1) = 0)."""
    assert critical_cp(1.0) == pytest.approx(0.0, abs=1e-12)
    assert critical_cp(0.75) == pytest.approx(-0.59121, abs=1e-4)
    assert critical_cp(0.5) == pytest.approx(-2.13340, abs=1e-4)


QC = tuple(float(v) for v in AF.repanelled(160).quarter_chord)   # default pivot = moment point


def _ground_pair(alpha, h, pivot=QC):
    real = rotate(np.asarray(AF.repanelled(160).coords), alpha, pivot=pivot)
    real[:, 1] += h - pivot[1]
    return real, real * np.array([1.0, -1.0])


@pytest.mark.parametrize("alpha,h", [(4.0, 0.5), (8.0, 0.25), (-2.0, 1.0)])
def test_ground_effect_equals_the_mirror_problem_in_free_air(alpha, h):
    """SPEC.md 4.2: the ground result equals the two-body mirror problem solved in free air (the
    pitched section and its image as a free-air configuration at alpha 0)."""
    g = fl.analyze(AF.repanelled(160), alpha, ground_height=h, repanel=False)
    real, img = _ground_pair(alpha, h * fl.Airfoil.from_coords(AF.repanelled(160).coords).chord)
    ref = (QC[0], h * g.ref_chord)
    cfg = fl.Configuration((fl.Element(fl.Airfoil.from_coords(real)), fl.Element(fl.Airfoil.from_coords(img))),
                           ref_chord=g.ref_chord, ref_point=tuple(ref))
    free = fl.analyze(cfg, 0.0, repanel=False)
    assert len(g.elements) == 1
    assert g.cl == pytest.approx(free.elements[0].cl, abs=1e-12)
    assert g.cm == pytest.approx(free.elements[0].cm, abs=1e-12)
    np.testing.assert_allclose(g.elements[0].cp, free.elements[0].cp, atol=1e-12)


def test_ground_is_a_streamline_and_far_ground_is_free_air():
    """The velocity normal to the ground vanishes on y = 0, and 50 chords up the result is the
    free-air result at the same angle (pitch about the quarter chord)."""
    real, img = _ground_pair(5.0, 0.4)
    ps = core.PanelSet.from_bodies([real, img])
    sol = core.solve(ps, 0.0)
    x = np.linspace(-2.0, 3.0, 41)
    _, v = core.induced_velocity(ps, np.asarray(sol.gamma), x, np.zeros_like(x))
    assert np.max(np.abs(v)) < 1e-10
    far = fl.analyze(AF, 5.0, ground_height=50.0)
    assert far.cl == pytest.approx(fl.analyze(AF, 5.0).cl, rel=2e-3)
    near = fl.analyze(AF, 5.0, ground_height=0.4)
    assert near.cl > far.cl and any("ground effect" in c for c in near.corrections)


def test_ground_effect_pivot():
    """The section pitches about the given pivot; a different pivot gives the same result as moving
    the height accordingly (pivot on the chord line, small check at 0 deg pitch)."""
    a = fl.analyze(AF, 0.0, ground_height=0.5, pivot=(0.0, 0.0))
    b = fl.analyze(AF, 0.0, ground_height=0.5, pivot=(0.25, 0.0))
    assert a.cl == pytest.approx(b.cl, abs=1e-12)
    c = fl.analyze(AF, 6.0, ground_height=0.5, pivot=(0.0, 0.0))
    d = fl.analyze(AF, 6.0, ground_height=0.5)
    assert abs(c.cl - d.cl) > 1e-3          # pitching about the leading edge raises the trailing edge


def test_aspect_ratio_large_is_the_section_and_factor_formula():
    for mach in (0.0, 0.5):
        two = fl.analyze(AF, 4.0, mach=mach)
        big = fl.analyze(AF, 4.0, mach=mach, aspect_ratio=1e7)
        assert big.cl == pytest.approx(two.cl, rel=1e-6) and big.cd_induced < 1e-7
    r = fl.analyze(AF, 4.0, aspect_ratio=8.0)
    two = fl.analyze(AF, 4.0)
    slope = (fl.analyze(AF, 4.5).cl - fl.analyze(AF, 3.5).cl) / np.radians(1.0)
    f = helmbold_factor(slope, 8.0)
    assert r.cl == pytest.approx(f * two.cl, rel=1e-12)
    assert r.cd_induced == pytest.approx(r.cl ** 2 / (np.pi * 8.0), rel=1e-12)
    assert r.cm != two.cm          # the moment is that of the effective angle (test below)
    k = 2 * np.pi / (np.pi * 20.0)
    assert helmbold_factor(2 * np.pi, 20.0) == pytest.approx(1.0 / (np.sqrt(1 + k * k) + k), rel=1e-15)
    assert helmbold_factor(2 * np.pi, 1e6) == pytest.approx(1.0 / (1.0 + k * 20.0 / 1e6), rel=1e-9)


def test_compressibility_not_applied_twice_with_aspect_ratio():
    """The finite-wing factor uses the compressible section slope; its ratio CL/Cl at Mach 0.5 is
    Helmbold's factor of that slope, with no extra Prandtl-Glauert or Kármán-Tsien factor."""
    r = fl.analyze(AF, 4.0, mach=0.5, aspect_ratio=8.0)
    two = fl.analyze(AF, 4.0, mach=0.5)
    slope = (fl.analyze(AF, 4.5, mach=0.5).cl - fl.analyze(AF, 3.5, mach=0.5).cl) / np.radians(1.0)
    assert r.cl / two.cl == pytest.approx(helmbold_factor(slope, 8.0), rel=1e-12)


def test_sweep_normal_plane_and_scaling():
    lam = np.radians(25.0)
    r = fl.analyze(AF, 4.0, sweep=25.0)
    a_n = np.degrees(np.arctan(np.tan(np.radians(4.0)) / np.cos(lam)))
    n = fl.analyze(AF, a_n)
    assert r.cl == pytest.approx(n.cl * np.cos(lam) ** 2, rel=1e-12)
    assert r.cm == pytest.approx(n.cm * np.cos(lam) ** 2, rel=1e-12)
    assert r.mach_critical == pytest.approx(n.mach_critical / np.cos(lam), rel=1e-12)
    rm = fl.analyze(AF, 4.0, sweep=25.0, mach=0.6)
    nm = fl.analyze(AF, a_n, mach=0.6 * np.cos(lam))
    assert rm.cl == pytest.approx(nm.cl * np.cos(lam) ** 2, rel=1e-12)
    with pytest.raises(ValueError):
        fl.analyze(AF, 4.0, sweep=75.0)


def test_corrections_recorded_in_fixed_order():
    r = fl.analyze(AF, 4.0, mach=0.3, sweep=20.0, aspect_ratio=6.0, ground_height=1.0)
    keys = ["ground effect", "simple sweep", "Kármán-Tsien", "finite wing"]
    pos = [next(i for i, c in enumerate(r.corrections) if c.startswith(k) or k in c) for k in keys]
    assert pos == sorted(pos) and len(r.corrections) == 4
    assert (r.mach, r.sweep, r.aspect_ratio, r.ground_height) == (0.3, 20.0, 6.0, 1.0)


def test_invalid_conditions():
    with pytest.raises(NotImplementedError):
        fl.analyze(AF, 4.0, re=1e6, mach=0.3)
    with pytest.raises(ValueError):
        fl.analyze(AF, 4.0, mach=1.2)
    with pytest.raises(ValueError):
        fl.analyze(AF, 4.0, aspect_ratio=0.0)
    with pytest.raises(ValueError):
        fl.analyze(AF, 4.0, ground_height=-0.1)


@pytest.mark.slow
def test_viscous_ground_effect_and_sweep():
    """Viscous ground effect solves the section and its image together (M5 multi-element solver);
    only the real section's drag is counted. Viscous sweep runs the normal plane at Re cos(sweep)."""
    warnings.simplefilter("ignore")
    g = fl.analyze(AF, 2.0, re=1e6, ground_height=1.0)
    assert g.converged and len(g.elements) == 1
    assert g.cd == pytest.approx(g.elements[0].cd, rel=1e-12)
    free = fl.analyze(AF, 2.0, re=1e6)
    assert abs(g.cd - free.cd) / free.cd < 0.2 and g.cl != free.cl
    s = fl.analyze(AF, 2.0, re=1e6, sweep=30.0)
    a_n = np.degrees(np.arctan(np.tan(np.radians(2.0)) / np.cos(np.radians(30.0))))
    n = fl.analyze(AF, a_n, re=1e6 * np.cos(np.radians(30.0)))
    c = np.cos(np.radians(30.0))
    assert s.cl == pytest.approx(n.cl * c * c, rel=1e-12) and s.cd == pytest.approx(n.cd * c ** 3, rel=1e-12)


def test_finite_wing_moment_and_profile_drag_at_the_effective_angle():
    """Profile drag and moment come from the section at alpha_eff = alpha - (Cl_inf - CL)/a, where
    the section's lift equals the wing's CL (to the lift curve's linearity); lift stays CL
    (DECISIONS.md v2.2, E1)."""
    import re as _re
    r = fl.analyze(AF, 4.0, aspect_ratio=6.0)
    a_eff = float(_re.search(r"effective angle (-?[\d.]+) deg", " ".join(r.corrections)).group(1))
    sec = fl.analyze(AF, a_eff)
    assert a_eff < 4.0
    assert sec.cl == pytest.approx(r.cl, abs=2e-3)
    assert r.cm == pytest.approx(sec.cm, abs=1e-4)          # 4 decimals in the correction text


@pytest.mark.slow
def test_finite_wing_viscous_profile_drag_is_the_section_drag_at_the_effective_angle():
    import re as _re
    r = fl.analyze(AF, 4.0, re=3e6, aspect_ratio=8.0)
    a_eff = float(_re.search(r"effective angle (-?[\d.]+) deg", " ".join(r.corrections)).group(1))
    sec = fl.analyze(AF, a_eff, re=3e6)
    assert r.converged and sec.converged
    assert r.cd - r.cd_induced == pytest.approx(sec.cd, rel=1e-3)
    assert sec.cd < fl.analyze(AF, 4.0, re=3e6).cd
