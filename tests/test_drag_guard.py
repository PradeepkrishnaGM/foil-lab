"""A viscous result with non-finite drag, Cd < 0 or Cd > 1 is never reported as converged."""

import numpy as np
import pytest

from foil_lab import batch
from foil_lab.generators import generate_naca4
from foil_lab.solver import drag_is_plausible, solve_flow


@pytest.mark.parametrize("cd,ok", [(0.006, True), (0.0, True), (1.0, True), (-1e-6, False),
                                   (1.0001, False), (26836.7, False), (np.nan, False),
                                   (np.inf, False)])
def test_drag_is_plausible(cd, ok):
    assert drag_is_plausible(cd) is ok


@pytest.mark.parametrize("cd_visc", [2.0, -0.01, np.nan])
def test_batch_row_with_absurd_drag_is_not_converged(monkeypatch, cd_visc):
    af = generate_naca4("2412", 80)
    real = solve_flow

    def fake(*a, **kw):
        s = real(*a, **kw)
        s.cd_visc = cd_visc
        return s

    monkeypatch.setattr(batch, "solve_flow", fake)
    batch.register_airfoils({"a": af})
    row = batch._solve_one(batch.Case(airfoil="a", alpha_deg=4.0, reynolds=1e6))
    assert row["converged"] is False


def test_batch_inviscid_row_still_converged():
    batch.register_airfoils({"a": generate_naca4("2412", 80)})
    assert batch._solve_one(batch.Case(airfoil="a", alpha_deg=4.0))["converged"] is True
