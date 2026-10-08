"""Validation against published thin-airfoil / NACA reference values.

These are looser tolerance "does the physics land in the right ballpark" checks
rather than unit tests. They guard against silent physics regressions.
"""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.analysis import run_alpha_sweep, find_zero_lift_angle, calculate_aerodynamic_center


@pytest.mark.parametrize("naca,alpha_l0_expected", [
    ("0012", 0.0),    # symmetric
    ("2412", -2.1),   # ~ -2.1 deg
    ("4412", -4.0),   # ~ -4 deg
])
def test_zero_lift_angles(naca, alpha_l0_expected):
    af = generate_naca4(naca, 200)
    polar = run_alpha_sweep(af, -8, 8, 1.0)
    al0 = find_zero_lift_angle(polar)
    assert abs(al0 - alpha_l0_expected) < 1.0, \
        f"NACA{naca} alpha_L0={al0:.2f}, expected ~{alpha_l0_expected}"


def test_aerodynamic_center_near_quarter_chord():
    af = generate_naca4("2412", 240)
    polar = run_alpha_sweep(af, -4, 8, 1.0)
    x_ac, _ = calculate_aerodynamic_center(polar)
    # Thin-airfoil theory places the AC at 0.25c; panel methods land 0.24-0.27.
    assert 0.22 < x_ac < 0.28, f"x_ac={x_ac:.3f} off quarter chord"


def test_lift_curve_linear_in_attached_range():
    af = generate_naca4("2412", 200)
    polar = run_alpha_sweep(af, -4, 8, 1.0)
    # In the attached range the lift curve should be near-linear: second
    # differences small relative to the slope.
    d2 = np.diff(polar.cls, 2)
    slope = np.mean(np.diff(polar.cls))
    assert np.max(np.abs(d2)) < 0.05 * abs(slope) + 1e-3
