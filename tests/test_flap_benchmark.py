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
