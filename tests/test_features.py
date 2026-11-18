"""Tests for the v0.5 feature set, through the v2 interface: simple sweep and ground effect
(``fl.analyze``), generators and geometry smoothing.
"""

import numpy as np
import pytest

import foil_lab as fl
from foil_lab.generators import (generate_naca4, generate_biconvex, generate_wedge,
                                  generate_parsec)
from foil_lab.geometry import smooth_coordinates
from tests import section


# --- Simple sweep ----------------------------------------------------------- #
def test_simple_sweep_lift_slope_scales_with_cos_sweep():
    """Simple sweep theory (M6): the swept wing's lift equals the normal-plane section's lift at
    alpha_n = atan(tan(alpha)/cos(sweep)) times cos^2(sweep); for the lift slope about the zero-lift
    angle this is the classic cos(sweep) (v1.3.1 gave cos^2)."""
    af = fl.Airfoil.from_coords(generate_naca4("0012", 160))
    lam = np.radians(30.0)
    a_n = np.degrees(np.arctan(np.tan(np.radians(2.0)) / np.cos(lam)))
    r = fl.analyze(af, 2.0, sweep=30.0, repanel=False)
    assert r.cl == pytest.approx(fl.analyze(af, a_n, repanel=False).cl * np.cos(lam) ** 2, rel=1e-12)

    def cl(alpha, sweep):
        return fl.analyze(af, alpha, sweep=sweep, repanel=False).cl
    slope = (cl(1.0, 30.0) - cl(-1.0, 30.0)) / 2.0
    slope0 = (cl(1.0, 0.0) - cl(-1.0, 0.0)) / 2.0
    assert slope / slope0 == pytest.approx(np.cos(lam), rel=1e-3)


# --- Ground effect ---------------------------------------------------------- #
def test_ground_effect_increases_lift():
    af = fl.Airfoil.from_coords(generate_naca4("2412", 160))
    free = fl.analyze(af, 5.0, repanel=False).cl
    near = fl.analyze(af, 5.0, ground_height=0.5, repanel=False).cl
    assert near > free, "lift should rise in ground effect"


def test_ground_effect_far_field_limit():
    af = fl.Airfoil.from_coords(generate_naca4("2412", 160))
    free = fl.analyze(af, 5.0, repanel=False).cl
    far = fl.analyze(af, 5.0, ground_height=50.0, repanel=False).cl
    assert abs(far - free) < 0.01


# --- Generators ------------------------------------------------------------- #
@pytest.mark.parametrize("gen", [generate_biconvex, generate_wedge])
def test_symmetric_supersonic_sections(gen):
    af = gen(0.06, 120)
    assert np.allclose(af[0], af[-1], atol=1e-6)  # closed TE
    assert abs(section.cl(af, 0.0)) < 1e-6      # symmetric


def test_parsec_produces_valid_geometry():
    p = dict(rle=0.01, x_up=0.4, y_up=0.06, yxx_up=-0.5, x_lo=0.4, y_lo=-0.04,
             yxx_lo=0.5, y_te=0.0, dy_te=0.0, ang_te=-8.0, ang_wedge=8.0)
    af = generate_parsec(p, 160)
    assert af.shape[1] == 2
    assert af[:, 1].max() > 0 and af[:, 1].min() < 0  # has thickness
    assert np.isfinite(section.cl(af, 0.0))


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
