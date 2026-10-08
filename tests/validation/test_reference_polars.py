"""Validation against reference polars — honest envelope checks.

These tests assert what Foil Lab *can* legitimately claim and document the known
limits (low-Re VII non-convergence) rather than asserting accuracy it does not
have. They use sourced reference data under data/.
"""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.validation import (load_reference, validate_inviscid,
                                 validate_vii, lift_curve_metrics)


def test_inviscid_matches_thin_airfoil_theory():
    """The inviscid core must match analytical thin-airfoil theory closely."""
    ref = load_reference("naca0012_thin_airfoil.csv")
    rep = validate_inviscid(generate_naca4("0012", 200), ref)
    # Panel method vs flat-plate theory: small residual from finite thickness.
    assert rep.cl_mae < 0.06


def test_reference_data_loads():
    ref = load_reference("naca2412_xfoil_re50k.csv")
    assert ref.reynolds == 50000
    assert len(ref.alpha) > 5
    assert "XFOIL" in ref.source or "xfoil" in ref.source.lower()


def test_inviscid_zero_lift_angle_physical():
    """Inviscid alpha_L0 for NACA 2412 must be ~ -2.1 deg (thin-airfoil)."""
    ref = load_reference("naca2412_xfoil_re50k.csv")
    m = lift_curve_metrics(generate_naca4("2412", 200), ref)
    assert -3.0 < m["model_alpha_L0_deg"] < -1.2


@pytest.mark.slow
def test_documented_low_re_vii_limitation():
    """DOCUMENTED LIMITATION: VII does not converge at Re=50k (bubble regime).

    This test pins the *known* limit so a future fix is visible as a change.
    If VII is improved to converge here, this test will start failing and
    should be updated — that's the intended signal.
    """
    ref = load_reference("naca2412_xfoil_re50k.csv")
    rep = validate_vii(generate_naca4("2412", 200), ref, n_crit=9.0,
                       alpha_max=8.0, max_iter=120)
    n_converged = np.sum(~np.isnan(rep.cl_model))
    # Currently essentially none converge at this Reynolds number.
    assert n_converged <= 2, (
        f"VII now converges at {n_converged} low-Re points — if this is a real "
        "improvement, update this test and the documented envelope.")
