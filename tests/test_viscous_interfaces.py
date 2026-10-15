"""C5 (M3): every interface with a Reynolds number reaches the coupled viscous solver.

``fl.analyze`` / ``fl.polar`` (``re``), ``solver.solve_flow`` / ``solve_flow_sweep``, the command
line, the batch runner, ``design.maximize_ld``, ``advanced`` and the GUI; the v1.3.1 viscous
modules are gone. Fast tests check the routing with a stand-in for the solver; the end-to-end
smoke tests run the real solver (about 10 s per point here) and are marked slow.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import solver, viscous
from foil_lab.generators import generate_naca4

OLD_MODULES = ("boundary_layer", "bl_integral", "transition", "vii", "vii_newton", "vii_qs",
               "viscous_mods")


@pytest.mark.parametrize("name", OLD_MODULES)
def test_old_viscous_modules_are_removed(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"foil_lab.{name}")


@pytest.fixture
def fake_viscous(monkeypatch):
    """Replace the coupled solver by a stand-in that records its calls (no solve)."""
    calls = []

    def result(alpha, reynolds, params, trips):
        return viscous.ViscousResult(alpha=alpha, reynolds=reynolds, converged=True, iterations=3, reason="",
                                     cl=0.5, cd=0.01, cdf=0.006, cdp=0.004, cm=-0.05,
                                     xtr_upper=0.3, xtr_lower=0.6, path="march")

    def solve_point(coords, alpha, reynolds, params=viscous.VISCOUS, trips=None, *a, **kw):
        calls.append(("point", alpha, reynolds, params.n_crit, trips))
        return result(alpha, reynolds, params, trips)

    def sweep(coords, alphas, reynolds, params=viscous.VISCOUS, trips=None, *a, **kw):
        calls.append(("sweep", list(alphas), reynolds, params.n_crit, trips))
        return [result(a, reynolds, params, trips) for a in alphas]

    monkeypatch.setattr(viscous, "solve_point", solve_point)
    monkeypatch.setattr(viscous, "sweep", sweep)
    return calls


def test_solve_flow_with_reynolds_uses_the_coupled_solver(fake_viscous):
    af = generate_naca4("2412", 120)
    s = solver.solve_flow(af, 4.0, reynolds=1e6, n_crit=7.0, trip_x_upper=0.1)
    assert fake_viscous == [("point", 4.0, 1e6, 7.0, {(0, "upper"): 0.1})]
    assert s.converged and s.cd_visc == 0.01 and s.cd_friction == 0.006 and s.xtr_upper == 0.3
    sols = solver.solve_flow_sweep(af, [0.0, 2.0], 1e6)
    assert fake_viscous[-1][0] == "sweep" and [x.alpha_deg for x in sols] == [0.0, 2.0]


def test_solve_flow_without_reynolds_stays_inviscid(fake_viscous):
    s = solver.solve_flow(generate_naca4("2412", 120), 4.0)
    assert not fake_viscous and s.cd_visc == 0.0 and s.converged


def test_viscous_requests_outside_the_solver_raise():
    af = generate_naca4("2412", 120)
    with pytest.raises(NotImplementedError):
        solver.solve_flow(af, 4.0, reynolds=1e6, mach=0.3)
    with pytest.raises(NotImplementedError):
        solver.solve_flow([af, af + [1.5, 0.0]], 4.0, reynolds=1e6)
    from foil_lab.ground_effect import solve_ground_effect
    with pytest.raises(NotImplementedError):
        solve_ground_effect(af, 4.0, 0.5, reynolds=1e6)
    with pytest.raises(ValueError):
        fl.analyze(fl.Airfoil.naca("2412"), 4.0, re=-1e6)


def test_api_accepts_re_and_reports_viscous_fields(fake_viscous):
    af = fl.Airfoil.naca("2412")
    r = fl.analyze(af, 4.0, re=1e6, n_crit=8.0, trip_lower=0.4)
    chord = r.ref_chord            # the solver works per unit length: it receives re / ref_chord
    assert fake_viscous[-1][:4] == ("point", 4.0, pytest.approx(1e6 / chord, rel=1e-15), 8.0)
    assert fake_viscous[-1][4] == {(0, "lower"): 0.4}
    assert r.reynolds == pytest.approx(1e6, rel=1e-15) and r.cd == pytest.approx(0.01 / chord)
    assert r.cd_friction == pytest.approx(0.006 / chord) and r.path == "march"
    assert r.n_crit == 8.0 and r.trips == (None, 0.4) and r.method == fl.api.METHOD_VISCOUS
    p = fl.polar(af, [2.0, 0.0], re=1e6)
    assert fake_viscous[-1][:2] == ("sweep", [2.0, 0.0])
    assert list(p.alpha) == [2.0, 0.0] and np.all(p.converged) and np.allclose(p.cd, 0.01 / chord)


def test_cli_routes_reynolds_to_the_coupled_solver(fake_viscous, capsys):
    from foil_lab.cli import main
    main(["analyze", "--naca", "2412", "--alpha", "4", "--re", "1e6", "--ncrit", "7"])
    out = capsys.readouterr().out
    assert fake_viscous[-1][:4] == ("point", 4.0, 1e6, 7.0) and "converged" in out
    main(["polar", "--naca", "2412", "--start", "0", "--end", "2", "--re", "1e6"])
    assert fake_viscous[-1][:2] == ("sweep", [0.0, 1.0, 2.0])
    assert capsys.readouterr().out.count("converged") == 3


def test_cli_reports_invalid_input_without_a_traceback():
    from foil_lab.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--naca", "2412", "--re", "-5"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--naca", "241"])
    assert "error" in str(exc.value.code)


def test_batch_rows_carry_the_solver_flag_and_drag_parts(fake_viscous):
    from foil_lab import batch
    batch.register_airfoils({"a": generate_naca4("2412", 120)})
    row = batch._solve_one(batch.Case(airfoil="a", alpha_deg=4.0, reynolds=1e6, n_crit=6.0))
    assert fake_viscous[-1][:4] == ("point", 4.0, 1e6, 6.0)
    assert row["converged"] is True and row["cd_friction"] == 0.006 and row["path"] == "march"


def test_maximize_ld_uses_the_coupled_solver(monkeypatch):
    """Every L/D evaluation is ``solve_flow`` with the Reynolds number (no other drag path)."""
    from foil_lab import design
    seen = []

    def fake(coords, alpha, reynolds=0.0, **kw):
        seen.append(reynolds)
        s = solver.solve_flow(coords, alpha)
        s.reynolds, s.cd_visc = reynolds, 0.01
        return s

    monkeypatch.setattr(design, "solve_flow", fake)
    res = design.maximize_ld(alpha_deg=4.0, reynolds=2e6, max_iter=1)
    assert seen and set(seen) == {2e6}
    assert "viscous" in res.message


def test_maximize_ld_penalises_unconverged_candidates(monkeypatch):
    from foil_lab import design

    def fake(coords, alpha, reynolds=0.0, **kw):
        s = solver.solve_flow(coords, alpha)
        s.reynolds, s.cd_visc, s.converged = reynolds, 1e-6, False
        return s

    monkeypatch.setattr(design, "solve_flow", fake)
    res = design.maximize_ld(alpha_deg=4.0, reynolds=2e6, max_iter=1)
    assert not res.success and "NOT converged" in res.message
    assert all(h == 0.0 for h in res.history)


# ---------------------------------------------------------------------------------------- #
# End to end with the real solver (slow)
# ---------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_analyze_and_polar_with_re_end_to_end():
    af = fl.Airfoil.naca("2412")
    r = fl.analyze(af, 4.0, re=1e6)
    assert r.converged and r.residual <= viscous.VISCOUS.residual_tol
    assert 0.004 < r.cd < 0.012 and abs(r.cd_friction + r.cd_pressure - r.cd) <= 1e-12
    assert 0.0 < r.xtr_upper < 1.0 and r.warnings == ()
    p = fl.polar(af, [0.0, 2.0, 4.0], re=1e6)
    assert np.all(p.converged) and np.all(np.diff(p.cl) > 0)
    assert abs(p.cl[2] - r.cl) <= 1e-3 and abs(p.cd[2] - r.cd) <= 1e-2 * r.cd


@pytest.mark.slow
def test_closed_trailing_edge_result_carries_the_wake_drag_warning():
    r = fl.analyze(fl.Airfoil.naca("2412", trailing_edge="closed"), 4.0, re=1e6)
    assert viscous.WARN_CLOSED_TE_DRAG in r.warnings


@pytest.mark.slow
def test_run_alpha_sweep_and_batch_end_to_end():
    from foil_lab.analysis import run_alpha_sweep
    from foil_lab.batch import build_matrix, run_batch
    af = generate_naca4("2412", 120)
    pol = run_alpha_sweep(af, 0.0, 2.0, 2.0, reynolds=3e6)
    assert all(s.converged for s in pol.solutions) and np.all(pol.cds > 0.003)
    rows = run_batch(build_matrix({"a": af}, alpha=[2.0], reynolds=[3e6]))
    assert rows[0]["converged"] and abs(rows[0]["cd_total"] - pol.cds[1]) <= 1e-3


@pytest.mark.slow
def test_gui_single_case_runs_the_coupled_solver(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from PySide6.QtWidgets import QApplication
    from foil_lab.gui.main_window import FoilLabMainWindow
    app = QApplication.instance() or QApplication([])
    w = FoilLabMainWindow()
    w.txt_naca.setText("2412")
    w.generate_naca()
    w.spin_re.setValue(1e6)
    w.spin_alpha.setValue(4.0)
    w.run_single_analysis()
    text = w.lbl_status.text()
    assert text.startswith("Viscous: converged"), text
    w.close()
    del app
