"""Tests for inverse design / optimization."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4, generate_parsec
from foil_lab.solver import solve_flow
from foil_lab.design import (design_for_cl, match_cp, default_start,
                             _vec_to_params, _params_to_vec, PARSEC_KEYS)


def test_param_vector_roundtrip():
    p = default_start()
    assert _vec_to_params(_params_to_vec(p)) == pytest.approx(p)


@pytest.mark.parametrize("target", [0.6, 0.9, 1.1])
def test_design_for_cl_hits_target(target):
    r = design_for_cl(target_cl=target, alpha_deg=5.0, num_panels=100, max_iter=60)
    achieved = solve_flow(r.coords, 5.0).cl
    assert abs(achieved - target) < 0.03, f"got {achieved}, wanted {target}"


def test_design_respects_thickness_bounds():
    r = design_for_cl(target_cl=0.8, alpha_deg=5.0, num_panels=100,
                      min_thickness=0.10, max_thickness=0.14, max_iter=60)
    t = r.coords[:, 1].max() - r.coords[:, 1].min()
    assert 0.09 < t < 0.15   # within bounds (small slack for discretization)


def test_design_produces_valid_geometry():
    r = design_for_cl(target_cl=0.7, alpha_deg=5.0, num_panels=100, max_iter=40)
    # finite, ordered, sensible airfoil
    assert np.all(np.isfinite(r.coords))
    assert r.coords[:, 0].min() < 0.05      # has a leading edge near x=0
    assert r.coords[:, 0].max() > 0.95      # extends to trailing edge


def test_match_cp_recovers_similar_shape():
    target = generate_naca4("2412", 120)
    r = match_cp(target, alpha_deg=5.0, num_panels=120, max_iter=60)
    # Recovered thickness close to the 2412's 12%.
    t = r.coords[:, 1].max() - r.coords[:, 1].min()
    assert 0.10 < t < 0.14
    # Recovered Cl close to target Cl.
    tgt_cl = solve_flow(target, 5.0).cl
    got_cl = solve_flow(r.coords, 5.0).cl
    assert abs(got_cl - tgt_cl) < 0.10


def test_match_cp_reduces_error():
    """Optimization should reduce the Cp error from the starting shape."""
    target = generate_naca4("4412", 120)
    r = match_cp(target, alpha_deg=4.0, num_panels=120, max_iter=60)
    # history[0] is the starting error, history[-1] the final.
    assert r.history[-1] < r.history[0]
