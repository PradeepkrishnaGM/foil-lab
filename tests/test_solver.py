"""Core solver tests: orientation-independence, symmetry, and lift slope.

These lock in the physics of the inviscid core (foil_lab.inviscid). If a lumped-vortex style
regression ever returns, the symmetry and slope assertions fail loudly.
"""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab import inviscid as core


def test_symmetric_airfoil_zero_lift_at_zero_alpha():
    af = generate_naca4("0012", 200)
    sol = solve_flow(af, 0.0)
    assert abs(sol.cl) < 1e-6, f"symmetric foil must give Cl=0, got {sol.cl}"


def test_lift_slope_near_2pi():
    af = generate_naca4("0012", 240)
    cl5 = solve_flow(af, 5.0).cl
    cl0 = solve_flow(af, 0.0).cl
    slope_per_deg = (cl5 - cl0) / 5.0
    slope_per_rad = slope_per_deg * 180.0 / np.pi
    # Thin-airfoil theory: 2*pi ~= 6.28 per rad. Panel methods land ~6.0-6.9.
    assert 5.8 < slope_per_rad < 7.2, f"lift slope {slope_per_rad:.2f}/rad off"


def test_cambered_zero_lift_angle():
    af = generate_naca4("2412", 240)
    # Find alpha where Cl=0 by bracketing.
    a = np.linspace(-5, 0, 51)
    cls = np.array([solve_flow(af, float(ai)).cl for ai in a])
    alpha_l0 = np.interp(0.0, cls, a)
    assert -3.0 < alpha_l0 < -1.2, f"2412 alpha_L0 {alpha_l0:.2f} off (~-2.1 deg)"


def test_orientation_independence():
    """Reversing node order must not change the result (panels handle CW/CCW)."""
    af = generate_naca4("2412", 160)
    cl_fwd = solve_flow(af, 5.0).cl
    cl_rev = solve_flow(af[::-1].copy(), 5.0).cl
    assert abs(cl_fwd - cl_rev) < 1e-9


def test_near_field_matches_circulation():
    """Pressure-integral lift and Kutta--Joukowski lift should agree.

    Circulation is the closed-loop integral of tangential velocity around the
    surface, Gamma = -∮ V_t ds (sign per traversal). Cl = 2*Gamma / (V_inf * c).
    """
    af = generate_naca4("2412", 240)
    sol = solve_flow(af, 5.0)
    gamma_total = float(core.solve(core.PanelSet.from_bodies(af), 5.0).circulation[0])
    cl_kj = 2.0 * gamma_total
    assert abs(abs(cl_kj) - abs(sol.cl)) < 0.05


def test_panels_are_clockwise():
    af = generate_naca4("0012", 80)  # generator emits CCW
    ps = core.PanelSet.from_bodies(af)
    assert ps.reversed_input[0]
    assert core._signed_area(np.column_stack([ps.node_x, ps.node_y])) < 0.0


def test_stagnation_cp_near_one():
    af = generate_naca4("2412", 240)
    sol = solve_flow(af, 5.0)
    assert sol.cp_dist.max() == pytest.approx(1.0, abs=0.02)


def test_panel_convergence():
    """Cl should converge as panel count increases."""
    cls = [solve_flow(generate_naca4("2412", n), 5.0).cl for n in (80, 160, 320)]
    assert abs(cls[2] - cls[1]) < abs(cls[1] - cls[0]) + 1e-6
    assert abs(cls[2] - cls[1]) < 0.01
