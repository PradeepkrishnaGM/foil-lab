"""Geometry and IO round-trip tests."""

import numpy as np

from foil_lab.generators import generate_naca4
from foil_lab.geometry import normalize_airfoil, repanel_cosine


def test_normalize_unit_chord():
    af = generate_naca4("2412", 120, trailing_edge="closed") * 3.0  # scale up
    norm = normalize_airfoil(af)
    chord = norm[:, 0].max() - norm[:, 0].min()
    assert abs(chord - 1.0) < 1e-6


def test_repanel_point_count():
    af = generate_naca4("2412", 120)
    rp = repanel_cosine(af, num_panels=100)
    # num_panels+1 nodes (within rounding for the LE-skip assembly)
    assert 99 <= len(rp) <= 102


def test_flap_deflection_changes_lift():
    import foil_lab as fl
    af = fl.Airfoil.from_coords(generate_naca4("0012", 160))
    base = fl.analyze(af, 0.0, repanel=False).cl
    flapped_cl = fl.analyze(af.with_flap(0.7, 10.0), 0.0, repanel=False).cl
    # A downward flap adds camber -> positive lift at alpha=0.
    assert flapped_cl > base + 0.1
