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


@pytest.mark.slow
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


@pytest.mark.slow
def test_match_cp_reduces_error():
    """Optimization should reduce the Cp error from the starting shape."""
    target = generate_naca4("4412", 120)
    r = match_cp(target, alpha_deg=4.0, num_panels=120, max_iter=60)
    # history[0] is the starting error, history[-1] the final.
    assert r.history[-1] < r.history[0]


def test_trailing_edge_type_fixed_within_one_optimisation():
    """A closed start stays closed (gap held at 0); an open design keeps the gap >= the bound."""
    from foil_lab.params import DESIGN
    closed = design_for_cl(target_cl=0.8, alpha_deg=5.0, num_panels=80, max_iter=15)
    assert closed.params["dy_te"] == 0.0
    opened = design_for_cl(target_cl=0.8, alpha_deg=5.0, num_panels=80, max_iter=15,
                           trailing_edge="open")
    assert opened.params["dy_te"] >= DESIGN.min_te_gap


@pytest.mark.slow
def test_maximize_ld_runs_on_the_coupled_viscous_solver(monkeypatch):
    """Design smoke test (C5): every L/D evaluation is the coupled viscous solver at the requested
    Reynolds number, the search from the direct start only, and the returned design's final solve
    (full single-point logic) converged. Reduced settings (60 panels, 2 optimiser iterations, one
    angle) for about 2 minutes; the full demo is far slower (DECISIONS.md, Later, M7 speed)."""
    from foil_lab import design
    calls = []
    real = design.solve_flow

    def spy(*a, **kw):
        s = real(*a, **kw)
        calls.append((kw.get("reynolds"), bool(s.converged), s.iterations, kw.get("fallback")))
        return s

    monkeypatch.setattr(design, "solve_flow", spy)
    r = design.maximize_ld(alpha_deg=4.0, reynolds=3e6, num_panels=60, max_iter=2)
    assert calls and all(re == 3e6 and it > 0 for re, _, it, _ in calls)   # Newton iterations ran
    assert calls[-1][3] is True and all(fb is False for *_, fb in calls[:-1])
    assert calls[-1][1], "the returned design's final viscous solve did not converge"
    assert r.n_failed == sum(not c for _, c, _, _ in calls[:-1])
    assert "NOT converged" not in r.message
    assert 20.0 < r.objective < 300.0                                   # a realistic 2D L/D
