"""Validation against published thin-airfoil / NACA reference values.

These are looser tolerance "does the physics land in the right ballpark" checks
rather than unit tests. They guard against silent physics regressions.
"""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
import foil_lab as fl


def _polar(naca: str, num_panels: int, a0: float, a1: float) -> fl.Polar:
    """Inviscid polar on the generator's points (no repanelling), 1 deg steps."""
    af = fl.Airfoil.from_coords(generate_naca4(naca, num_panels))
    return fl.polar(af, np.arange(a0, a1 + 0.5), repanel=False)


@pytest.mark.parametrize("naca,alpha_l0_expected", [
    ("0012", 0.0),    # symmetric
    ("2412", -2.1),   # ~ -2.1 deg
    ("4412", -4.0),   # ~ -4 deg
])
def test_zero_lift_angles(naca, alpha_l0_expected):
    polar = _polar(naca, 200, -8, 8)
    al0 = np.interp(0.0, polar.cl, polar.alpha)
    assert abs(al0 - alpha_l0_expected) < 1.0, \
        f"NACA{naca} alpha_L0={al0:.2f}, expected ~{alpha_l0_expected}"


def test_aerodynamic_center_near_quarter_chord():
    polar = _polar("2412", 240, -4, 8)
    # Cm about the reference point (x_ref): Cm = Cm_ac - (x_ac - x_ref) / c * Cl
    slope, _ = np.polyfit(polar.cl, polar.cm, 1)
    r = polar.results[0]
    x_ac = r.ref_point[0] / r.ref_chord - slope
    # Thin-airfoil theory places the AC at 0.25c; panel methods land 0.24-0.27.
    assert 0.22 < x_ac < 0.28, f"x_ac={x_ac:.3f} off quarter chord"


def test_lift_curve_linear_in_attached_range():
    polar = _polar("2412", 200, -4, 8)
    # In the attached range the lift curve should be near-linear: second
    # differences small relative to the slope.
    d2 = np.diff(polar.cl, 2)
    slope = np.mean(np.diff(polar.cl))
    assert np.max(np.abs(d2)) < 0.05 * abs(slope) + 1e-3
