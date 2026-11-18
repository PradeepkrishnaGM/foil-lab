"""A viscous result with non-finite drag, Cd < 0 or Cd > 1 is never reported as converged."""

import numpy as np
import pytest

from foil_lab import batch
from foil_lab.generators import generate_naca4
from foil_lab.viscous import drag_is_plausible


@pytest.mark.parametrize("cd,ok", [(0.006, True), (0.0, True), (1.0, True), (-1e-6, False),
                                   (1.0001, False), (26836.7, False), (np.nan, False),
                                   (np.inf, False)])
def test_drag_is_plausible(cd, ok):
    assert drag_is_plausible(cd) is ok


@pytest.mark.parametrize("cd_visc", [2.0, -0.01, np.nan])
def test_batch_row_with_absurd_drag_is_not_converged(monkeypatch, cd_visc):
    import foil_lab as fl
    from foil_lab import viscous

    def fake_sweep(coords, alphas, reynolds, params=viscous.VISCOUS, trips=None, *a, **kw):
        # converged viscous results with the given drag, without running the coupled solver
        return [viscous.ViscousResult(alpha=al, reynolds=reynolds, converged=True, iterations=3, reason="",
                                      cl=0.5, cd=cd_visc, cdf=0.004, cdp=0.0, cm=-0.05, xtr_upper=0.3,
                                      xtr_lower=0.6, path="march") for al in alphas]
    monkeypatch.setattr(viscous, "sweep", fake_sweep)
    af = fl.Airfoil.from_coords(generate_naca4("2412", 80))
    rows = batch.run_batch([batch.Case("a", 4.0, re=1e6)], {"a": af})
    assert rows[0]["converged"] is False


def test_batch_inviscid_row_still_converged():
    rows = batch.run_batch([batch.Case("a", 4.0)], {"a": generate_naca4("2412", 80)})
    assert rows[0]["converged"] is True
