"""Tests for the v0.5 feature set: advanced flight, viscous mods, ground effect,
generators, and geometry smoothing.
"""

import numpy as np
import pytest

from foil_lab.generators import (generate_naca4, generate_biconvex, generate_wedge,
                                  generate_parsec, generate_naca6)
from foil_lab.geometry import smooth_coordinates
from foil_lab.solver import solve_flow
from foil_lab.advanced import solve_swept_wing, solve_constant_wing_loading
from foil_lab.ground_effect import solve_ground_effect


# --- Advanced flight -------------------------------------------------------- #
def test_busemann_sweep_cos_squared_scaling():
    af = generate_naca4("2412", 160)
    cl0 = solve_swept_wing(af, 5.0, 0.0).cl_3d
    cl30 = solve_swept_wing(af, 5.0, 30.0).cl_3d
    expected = np.cos(np.radians(30.0)) ** 2
    assert cl30 / cl0 == pytest.approx(expected, abs=1e-3)


def test_constant_wing_loading_converges():
    af = generate_naca4("2412", 160)
    p = solve_constant_wing_loading(af, 5.0, wing_loading=600.0, chord=1.2)
    assert p.converged
    assert p.speed > 0
    assert p.reynolds > 0


# --- Viscous modifiers ------------------------------------------------------ #
def test_roughness_increases_drag():
    af = generate_naca4("2412", 200)
    cds = [solve_flow(af, 5.0, reynolds=3e6, roughness=r).cd_visc for r in range(4)]
    assert all(cds[i] <= cds[i + 1] + 1e-9 for i in range(3))
    assert cds[3] > cds[0]


def test_post_stall_caps_lift():
    af = generate_naca4("2412", 200)
    # Without post-stall, Cl keeps rising; with it, a peak forms.
    cl_high_plain = solve_flow(af, 20.0, reynolds=3e6).cl
    cl_high_ps = solve_flow(af, 20.0, reynolds=3e6, enable_post_stall=True).cl
    assert cl_high_ps < cl_high_plain
    # Peak Cl should be physically bounded.
    cls = [solve_flow(af, float(a), reynolds=3e6, enable_post_stall=True).cl
           for a in range(0, 24, 2)]
    assert max(cls) < 2.2


# --- Ground effect ---------------------------------------------------------- #
def test_ground_effect_increases_lift():
    af = generate_naca4("2412", 160)
    free = solve_flow(af, 5.0).cl
    near = solve_ground_effect(af, 5.0, height=0.5).cl
    assert near > free, "lift should rise in ground effect"


def test_ground_effect_far_field_limit():
    af = generate_naca4("2412", 160)
    free = solve_flow(af, 5.0).cl
    far = solve_ground_effect(af, 5.0, height=50.0).cl
    assert abs(far - free) < 0.01


def test_ground_effect_free_surface_not_implemented():
    af = generate_naca4("2412", 120)
    with pytest.raises(NotImplementedError):
        solve_ground_effect(af, 5.0, height=1.0, wall_type="free_surface")


# --- Generators ------------------------------------------------------------- #
@pytest.mark.parametrize("gen", [generate_biconvex, generate_wedge])
def test_symmetric_supersonic_sections(gen):
    af = gen(0.06, 120)
    assert np.allclose(af[0], af[-1], atol=1e-6)  # closed TE
    assert abs(solve_flow(af, 0.0).cl) < 1e-6      # symmetric


def test_parsec_produces_valid_geometry():
    p = dict(rle=0.01, x_up=0.4, y_up=0.06, yxx_up=-0.5, x_lo=0.4, y_lo=-0.04,
             yxx_lo=0.5, y_te=0.0, dy_te=0.0, ang_te=-8.0, ang_wedge=8.0)
    af = generate_parsec(p, 160)
    assert af.shape[1] == 2
    assert af[:, 1].max() > 0 and af[:, 1].min() < 0  # has thickness
    sol = solve_flow(af, 0.0)
    assert np.isfinite(sol.cl)


def test_naca6_runs():
    af = generate_naca6("63", 0.12, 0.4, num_panels=160)
    assert np.isfinite(solve_flow(af, 0.0).cl)


# --- Geometry smoothing ----------------------------------------------------- #
def test_smoothing_reduces_noise_and_preserves_te():
    af = generate_naca4("2412", 120)
    rng = np.random.RandomState(0)
    noisy = af.copy()
    noisy[1:-1, 1] += rng.normal(0, 0.002, len(af) - 2)
    sm = smooth_coordinates(noisy, strength=0.5, iterations=3)
    rough = lambda c: np.sum(np.abs(np.diff(c[:, 1], 2)))
    assert rough(sm) < 0.5 * rough(noisy)
    assert np.allclose(sm[0], noisy[0]) and np.allclose(sm[-1], noisy[-1])
