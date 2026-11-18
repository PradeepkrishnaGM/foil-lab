"""Flap cases of the benchmark: canonical sections and the plumbing (M4 step 3)."""

import hashlib

import numpy as np
import pytest

import foil_lab as fl
from foil_lab.benchmark.flap_run import case_names
from foil_lab.benchmark.sections import canonical, flap_name
from foil_lab.params import BENCHMARK


def test_flap_case_names():
    assert flap_name("NACA2412", 10.0) == "NACA2412_F10"
    assert flap_name("NACA0012", 5.0) == "NACA0012_F05"
    names = case_names()
    assert len(names) == len(BENCHMARK.flap_bases) * len(BENCHMARK.flap_deflections) == 10
    assert "NACA0012_F00" in names and "NACA2412_F20" in names


def test_canonical_flap_section_is_the_flapped_base_section():
    base = canonical("NACA2412")
    sec = canonical("NACA2412_F10")
    af = fl.Airfoil.from_coords(base.coords).with_flap(0.75, 10.0, 0.5)
    assert np.array_equal(sec.coords, af.coords)
    assert sec.hinge == (float(af.hinge[0]), float(af.hinge[1]))
    assert sec.flap_panels.shape == (160,) and sec.flap_panels.any() and not sec.flap_panels.all()
    assert "with_flap" in sec.source and "hinge point" in sec.source
    assert sec.trailing_edge_gap == pytest.approx(base.trailing_edge_gap, rel=1e-9)   # rigid flap


def test_undeflected_flap_case_is_the_base_section_with_a_hinge():
    base = canonical("NACA0012")
    sec = canonical("NACA0012_F00")
    assert np.array_equal(sec.coords, base.coords)
    assert sec.hinge is not None and base.hinge is None


def test_flap_coordinate_file_is_deterministic():
    a = hashlib.sha256(canonical("NACA2412_F15").coords.tobytes()).hexdigest()
    b = hashlib.sha256(canonical("NACA2412_F15").coords.tobytes()).hexdigest()
    assert a == b


def test_dump_script_solves_each_angle_and_dumps_cp_and_cf():
    from foil_lab.benchmark.xfoil import alpha_tag, build_script
    s = build_script("a.dat", "a.pol", 1e6, 0.0, -2.0, -1.0, hinge=(0.75, 0.0), dump=True)
    lines = s.splitlines()
    assert alpha_tag(-2.0) == "m2" and alpha_tag(2.5) == "2p5"
    assert "SAVE nodes.dat" in lines and lines.index("SAVE nodes.dat") < lines.index("OPER")
    for a, tag in ((0, "0"), (-1, "m1"), (-2, "m2")):
        i = lines.index(f"ALFA {a}")
        assert lines[i + 1:i + 3] == [f"CPWR cp_{tag}.txt", f"DUMP du_{tag}.txt"]
    assert "ASEQ" not in s and "FMOM" in lines


def test_hinge_moment_from_xfoil_data_with_uniform_pressure_is_exact():
    """Uniform Cp on XFOIL's nodes (here the section's own nodes): the same exact value as the
    panel identity of Foil-Lab's hinge_moment (sign: XFOIL's counter-clockwise moment reversed)."""
    from foil_lab import inviscid as core
    from foil_lab.benchmark.flap_run import xfoil_hinge_moment
    from foil_lab.benchmark.xfoil import XfoilDumps
    sec = canonical("NACA2412_F10")
    n = len(sec.coords)
    d = XfoilDumps(sec.coords, [0.0], np.full((1, n), 0.8), np.zeros((1, n)), np.ones((1, n)), [])
    (ch_p, ch_f), = xfoil_hinge_moment(d, sec.coords, sec.flap_panels, np.asarray(sec.hinge)).values()
    ps = core.PanelSet.from_bodies([sec.coords])
    ref, _ = core.hinge_moment(ps, np.full(ps.num_panels, 0.8), 0, sec.flap_panels, np.asarray(sec.hinge))
    assert ch_f == 0.0 and ch_p == pytest.approx(ref, abs=1e-12)
