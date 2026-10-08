"""Experimental quasi-simultaneous VII — what is validated, what is not.

These tests pin the CURRENT state of the work-in-progress Veldman coupling:
  * the interaction law DOES bound displacement thickness through separation
    (validated, reusable machinery);
  * the full coupled solve DOES NOT yet converge at low Re (documented limit).

If a future change makes solve_vii_qs converge at Re=50k, the xfail below will
start passing unexpectedly (XPASS) — that is the intended signal to update the
status banner, VALIDATION.md, and the documented envelope.
"""

import warnings

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.vii_qs import (march_coupled, interaction_coefficients,
                             solve_vii_qs)
from foil_lab.bl_integral import march_boundary_layer


# --- Validated machinery: the interaction law bounds delta* ----------------- #
def test_interaction_law_bounds_displacement_through_separation():
    """On an adverse gradient that separates the direct BL, the coupled march
    keeps delta* an order of magnitude smaller (no Goldstein blow-up)."""
    s = np.linspace(1e-4, 1.0, 200)
    ue_inv = np.maximum(1.3 - 0.8 * s, 0.3)   # decelerating -> separation-prone

    direct = march_boundary_layer(s, ue_inv, Re=2e5, trip_s=0.1)
    coupled = march_coupled(s, ue_inv, Re=2e5, trip_s=0.1)

    assert np.all(np.isfinite(coupled.delta_star))
    # Coupled displacement stays well below the direct method's runaway value.
    assert coupled.delta_star.max() < 0.5 * direct.delta_star.max()


def test_interaction_coefficients_positive():
    s = np.linspace(0, 1, 50)
    C = interaction_coefficients(s)
    assert np.all(C > 0) and np.all(np.isfinite(C))


def test_coupled_march_edge_velocity_responds():
    """The coupled march must actually modify ue (else it's not coupling)."""
    s = np.linspace(1e-4, 1.0, 150)
    ue_inv = np.maximum(1.2 - 0.6 * s, 0.3)
    p = march_coupled(s, ue_inv, Re=2e5, trip_s=0.1)
    assert np.max(np.abs(p.ue - ue_inv)) > 1e-3
    assert p.ue.min() > 0.0


def test_solve_vii_qs_emits_experimental_warning():
    af = generate_naca4("2412", 120)
    with pytest.warns(UserWarning, match="EXPERIMENTAL"):
        solve_vii_qs(af, 4.0, reynolds=50000, max_iter=20, tol=1e-3)


# --- Documented limitation: does NOT converge at low Re yet ----------------- #
@pytest.mark.xfail(reason="QS-VII does not yet converge at low Re (WIP); "
                          "see vii_qs status banner", strict=False)
@pytest.mark.slow
def test_qs_vii_converges_at_low_re_KNOWN_FAIL():
    af = generate_naca4("2412", 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_vii_qs(af, 4.0, reynolds=50000, max_iter=200, tol=1e-3)
    assert r.converged   # currently False — xfail documents the WIP state
