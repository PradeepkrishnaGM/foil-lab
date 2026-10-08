"""Tests for the viscous-inviscid interaction (VII) solver and integral BL."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab.bl_integral import march_boundary_layer
from foil_lab.vii import solve_vii


# --- Integral boundary layer ------------------------------------------------ #
def test_blasius_laminar_flat_plate():
    """Laminar flat plate must match the Blasius similarity solution."""
    Re = 1e6
    s = np.linspace(1e-4, 1.0, 400)
    ue = np.ones_like(s)
    prof = march_boundary_layer(s, ue, Re)
    i = np.argmin(np.abs(s - 0.1))
    theta_blasius = 0.664 * np.sqrt(0.1 / Re)
    dstar_blasius = 1.721 * np.sqrt(0.1 / Re)
    assert abs(prof.theta[i] / theta_blasius - 1.0) < 0.05
    assert abs(prof.delta_star[i] / dstar_blasius - 1.0) < 0.05
    assert 2.5 < prof.H[i] < 2.7   # Blasius H = 2.59


def test_turbulent_flat_plate_cf():
    """Tripped turbulent flat plate matches the cf ~ 0.0592 Re_x^-0.2 law."""
    Re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    ue = np.ones_like(s)
    prof = march_boundary_layer(s, ue, Re, trip_s=0.1)
    for x in (0.3, 0.6, 0.9):
        i = np.argmin(np.abs(s - x))
        cf_ref = 0.0592 * (Re * x) ** -0.2
        assert abs(prof.cf[i] / cf_ref - 1.0) < 0.10
    # Attached turbulent shape factor stays in range.
    it = prof.transition_idx
    assert 1.3 < np.mean(prof.H[it:]) < 1.5


def test_transition_triggers_at_high_Re():
    Re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    prof = march_boundary_layer(s, np.ones_like(s), Re)
    # Natural transition should occur on the plate (not stuck at TE).
    assert prof.transition_idx < len(s) - 1


# --- VII coupling ----------------------------------------------------------- #
def test_vii_converges_attached():
    af = generate_naca4("2412", 200)
    r = solve_vii(af, 4.0, reynolds=3e6, max_iter=200, tol=1e-3)
    assert r.converged, f"VII failed to converge (res={r.residual})"


def test_vii_drag_rises_with_alpha():
    """A hallmark of real VII: profile drag increases with incidence."""
    af = generate_naca4("2412", 200)
    cds = []
    for a in (0.0, 4.0, 8.0):
        r = solve_vii(af, a, reynolds=3e6, max_iter=200, tol=1e-3)
        cds.append(r.cd_wake)
    assert cds[1] > cds[0] and cds[2] > cds[1], f"Cd not monotone: {cds}"


def test_vii_reduces_lift_vs_inviscid():
    """Displacement decambering: VII lift sits below the inviscid value."""
    af = generate_naca4("2412", 200)
    r = solve_vii(af, 4.0, reynolds=3e6, max_iter=200, tol=1e-3)
    inv = solve_flow(af, 4.0)  # pure inviscid
    assert r.solution.cl < inv.cl


def test_vii_drag_physically_plausible():
    """VII drag should be in a realistic range, unlike the too-low one-way Cd."""
    af = generate_naca4("2412", 200)
    r = solve_vii(af, 4.0, reynolds=3e6, max_iter=200, tol=1e-3)
    # NACA 2412 at Re=3e6 has Cd ~ 0.006-0.009 experimentally.
    assert 0.004 < r.cd_wake < 0.012


def test_vii_deterministic():
    af = generate_naca4("0012", 160)
    r1 = solve_vii(af, 3.0, reynolds=2e6, max_iter=150, tol=1e-3)
    r2 = solve_vii(af, 3.0, reynolds=2e6, max_iter=150, tol=1e-3)
    assert abs(r1.solution.cl - r2.solution.cl) < 1e-9
    assert abs(r1.cd_wake - r2.cd_wake) < 1e-9
