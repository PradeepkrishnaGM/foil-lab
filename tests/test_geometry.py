"""Geometry and IO round-trip tests."""

import numpy as np

from foil_lab.generators import generate_naca4
from foil_lab.geometry import normalize_airfoil, repanel_cosine, deflect_flap


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
    from foil_lab.solver import solve_flow
    af = generate_naca4("0012", 160)
    base = solve_flow(af, 0.0).cl
    flapped = deflect_flap(af, hinge_x=0.7, deflection_deg=10.0)
    flapped_cl = solve_flow(repanel_cosine(flapped, 160), 0.0).cl
    # A downward flap adds camber -> positive lift at alpha=0.
    assert flapped_cl > base + 0.1
