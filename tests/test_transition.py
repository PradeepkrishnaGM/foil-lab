"""Tests for the e^n (envelope) transition model and its BL/VII integration."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.transition import (find_transition_en, log10_re_theta_crit,
                                 dn_dre_theta)
from foil_lab.bl_integral import march_boundary_layer
from foil_lab.vii import solve_vii


# --- Correlations ----------------------------------------------------------- #
def test_amplification_rate_nonnegative():
    H = np.linspace(2.0, 6.0, 50)
    assert np.all(dn_dre_theta(H) >= 0.0)


def test_neutral_point_increases_for_favorable_gradient():
    """Lower H (favorable gradient / fuller profile) => more stable => higher
    critical Re_theta. So Re_theta_crit should fall as H rises (adverse)."""
    rc_low_H = 10 ** log10_re_theta_crit(np.array([2.2]))[0]
    rc_high_H = 10 ** log10_re_theta_crit(np.array([3.5]))[0]
    assert rc_low_H > rc_high_H


# --- Flat-plate transition location ----------------------------------------- #
def test_flat_plate_transition_ncrit9():
    """N_crit=9 flat-plate transition near the textbook Re_x ~ 2.5-3e6."""
    Re = 1e7
    x = np.linspace(1e-4, 1.0, 2000)
    theta = 0.664 * np.sqrt(x / Re)
    H = np.full_like(x, 2.59)
    Re_theta = Re * theta
    idx, _ = find_transition_en(x, Re_theta, H, n_crit=9.0)
    Re_x_tr = Re * x[idx]
    assert 2.0e6 < Re_x_tr < 3.5e6


def test_ncrit_monotonic_transition():
    """Higher N_crit (quieter) delays transition; lower advances it."""
    Re = 1e7
    x = np.linspace(1e-4, 1.0, 2000)
    theta = 0.664 * np.sqrt(x / Re)
    H = np.full_like(x, 2.59)
    Re_theta = Re * theta
    xs = []
    for nc in (5, 9, 13):
        idx, _ = find_transition_en(x, Re_theta, H, n_crit=nc)
        xs.append(x[idx])
    assert xs[0] < xs[1] < xs[2]


# --- BL marcher integration ------------------------------------------------- #
def test_en_and_michel_similar_ballpark():
    Re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    ue = np.ones_like(s)
    p_en = march_boundary_layer(s, ue, Re, transition_model="en")
    p_mi = march_boundary_layer(s, ue, Re, transition_model="michel")
    # Same order of magnitude (both physical transition estimates).
    assert abs(s[p_en.transition_idx] - s[p_mi.transition_idx]) < 0.25


def test_trip_overrides_en():
    Re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    ue = np.ones_like(s)
    p = march_boundary_layer(s, ue, Re, trip_s=0.1, transition_model="en")
    assert s[p.transition_idx] <= 0.12


# --- VII with e^n ----------------------------------------------------------- #
def test_vii_en_ncrit_affects_drag():
    """Dirtier (low N_crit) => earlier transition => more drag."""
    af = generate_naca4("2412", 200)
    cd_dirty = solve_vii(af, 4.0, reynolds=3e6, transition_model="en",
                         n_crit=5.0, max_iter=200, tol=1e-3).cd_wake
    cd_clean = solve_vii(af, 4.0, reynolds=3e6, transition_model="en",
                         n_crit=12.0, max_iter=200, tol=1e-3).cd_wake
    assert cd_dirty > cd_clean


def test_vii_en_converges():
    af = generate_naca4("2412", 200)
    r = solve_vii(af, 4.0, reynolds=3e6, transition_model="en", n_crit=9.0,
                  max_iter=200, tol=1e-3)
    assert r.converged
