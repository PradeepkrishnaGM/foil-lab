"""C5 (M3): every interface with a Reynolds number reaches the coupled viscous solver.

``fl.analyze`` / ``fl.polar`` (``re``), the command line, the batch runner,
``design.maximize_ld`` and the GUI; the v1.3.1 viscous modules and the v1.3.1-shaped interface
(``solver``, ``analysis``, ``advanced``) are gone. Fast tests check the routing with a stand-in for the solver; the end-to-end
smoke tests run the real solver (about 10 s per point here) and are marked slow.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import viscous
from tests import section
from foil_lab.generators import generate_naca4

OLD_MODULES = ("boundary_layer", "bl_integral", "transition", "vii", "vii_newton", "vii_qs",
               "viscous_mods", "solver", "analysis", "advanced")


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

    def solve_single(coords, alpha, reynolds, params=viscous.VISCOUS, trips=None, *a, **kw):
        calls.append(("point", alpha, reynolds, params.n_crit, trips))
        return result(alpha, reynolds, params, trips)

    def sweep(coords, alphas, reynolds, params=viscous.VISCOUS, trips=None, *a, **kw):
        calls.append(("sweep", list(alphas), reynolds, params.n_crit, trips))
        return [result(a, reynolds, params, trips) for a in alphas]

    monkeypatch.setattr(viscous, "solve_single", solve_single)
    monkeypatch.setattr(viscous, "sweep", sweep)
    return calls


def test_analyze_with_re_uses_the_coupled_solver(fake_viscous):
    af = fl.Airfoil.from_coords(generate_naca4("2412", 120))
    r = fl.analyze(af, 4.0, re=1e6, n_crit=7.0, trip_upper=0.1, repanel=False)
    assert fake_viscous == [("point", 4.0, 1e6 / af.chord, 7.0, {(0, "upper"): 0.1})]
    assert r.converged and r.method != "inviscid" and r.xtr_upper == 0.3
    p = fl.polar(af, [0.0, 2.0], re=1e6, repanel=False)
    assert fake_viscous[-1][0] == "sweep" and list(p.alpha) == [0.0, 2.0]


def test_analyze_without_re_stays_inviscid(fake_viscous):
    r = fl.analyze(fl.Airfoil.naca("2412"), 4.0)
    assert not fake_viscous and r.cd is None and r.converged


def test_viscous_requests_outside_the_solver_raise():
    with pytest.raises(NotImplementedError):
        fl.analyze(fl.Airfoil.naca("2412"), 4.0, re=1e6, mach=0.3)
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
    assert fake_viscous[-1][0] == "point" and fake_viscous[-1][1] == 4.0 and fake_viscous[-1][3] == 7.0
    assert "converged" in out
    main(["polar", "--naca", "2412", "--start", "0", "--end", "2", "--re", "1e6"])
    assert fake_viscous[-1][:2] == ("sweep", [0.0, 1.0, 2.0])
    assert capsys.readouterr().out.count("converged") == 3


def test_cli_reports_invalid_input_without_a_traceback():
    from foil_lab.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--naca", "2412", "--alpha", "2", "--re", "-5"])
    assert exc.value.code == 2                       # argparse usage error
    for argv in (["analyze", "--naca", "241", "--alpha", "2"],
                 ["analyze", "--dat", "no_such_file.dat", "--alpha", "2"],
                 ["analyze", "--naca", "2412", "--alpha", "2", "--re", "1e6", "--mach", "0.3"],
                 ["analyze", "--naca", "2412", "--alpha", "2", "--ncrit", "7"],
                 ["polar", "--naca", "2412", "--step", "0"],
                 ["analyze", "--naca", "2412", "--flap", "0.75", "45", "--alpha", "2"]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert str(exc.value.code).startswith("error:"), argv


def test_batch_rows_carry_the_solver_flag_and_drag_parts(fake_viscous):
    from foil_lab import batch
    af = generate_naca4("2412", 120)
    rows = batch.run_batch([batch.Case("a", 4.0, re=1e6, n_crit=6.0)], {"a": af})
    assert fake_viscous[-1][0] == "sweep" and fake_viscous[-1][3] == 6.0
    row = rows[0]
    assert row["converged"] is True and row["path"] == "march"
    assert row["cd_friction"] == pytest.approx(0.006 / fl.Airfoil.from_coords(af).chord)


def test_maximize_ld_uses_the_coupled_solver(monkeypatch):
    """Every L/D evaluation is the coupled solver at the Reynolds number (no other drag path): the
    search from the direct start only, the final design with the full single-point logic."""
    from foil_lab import design
    seen = []

    def fake(coords, alpha, reynolds, fallback):
        seen.append((reynolds, fallback))
        return viscous.ViscousResult(alpha=alpha, reynolds=reynolds, converged=True, iterations=3,
                                     reason="", cl=section.cl(coords, alpha), cd=0.01, cdf=0.006,
                                     cdp=0.004, cm=0.0, xtr_upper=0.3, xtr_lower=0.6, path="march")

    monkeypatch.setattr(design, "_viscous", fake)
    res = design.maximize_ld(alpha_deg=4.0, reynolds=2e6, max_iter=1)
    assert seen and {re for re, _ in seen} == {2e6}
    assert seen[-1][1] is True and all(fb is False for _, fb in seen[:-1])
    assert "viscous" in res.message and res.n_failed == 0


def test_maximize_ld_penalises_unconverged_candidates(monkeypatch):
    from foil_lab import design

    def fake(coords, alpha, reynolds, fallback):
        return viscous.ViscousResult(alpha=alpha, reynolds=reynolds, converged=False, iterations=3,
                                     reason="", cl=section.cl(coords, alpha), cd=1e-6, cdf=1e-6,
                                     cdp=0.0, cm=0.0, xtr_upper=0.3, xtr_lower=0.6, path="march")

    monkeypatch.setattr(design, "_viscous", fake)
    res = design.maximize_ld(alpha_deg=4.0, reynolds=2e6, max_iter=1)
    assert not res.success and "NOT converged" in res.message
    assert all(h == 0.0 for h in res.history)
    assert res.n_failed == res.n_evals and f"{res.n_failed} candidates failed" in res.message


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
def test_closed_trailing_edge_kept_sharp_carries_the_wake_drag_warning():
    """Default: the closed edge is opened (a correction, no warning). Kept sharp
    (``closed_te_gap`` 0): the result carries the closed-edge drag warning."""
    import dataclasses
    af = fl.Airfoil.naca("2412", trailing_edge="closed")
    r = fl.analyze(af, 4.0, re=1e6)
    assert r.corrections and viscous.WARN_CLOSED_TE_DRAG not in r.warnings
    sharp = fl.analyze(af, 4.0, re=1e6, params=dataclasses.replace(viscous.VISCOUS, closed_te_gap=0.0))
    assert sharp.corrections == () and viscous.WARN_CLOSED_TE_DRAG in sharp.warnings


@pytest.mark.slow
def test_polar_and_batch_end_to_end():
    """The problem of the former ``run_alpha_sweep`` test: the raw generator points at a Reynolds
    number of 3e6 per unit length (``re`` is per reference chord, here 1.000078). At ``re=3e6``
    itself the 0 deg point does not converge on these points (DECISIONS.md, Later)."""
    from foil_lab.batch import build_matrix, run_batch
    af = generate_naca4("2412", 120)
    foil = fl.Airfoil.from_coords(af)
    pol = fl.polar(foil, [0.0, 2.0], re=3e6 * foil.chord, repanel=False)
    assert np.all(pol.converged) and np.all(pol.cd > 0.003)
    rows = run_batch(build_matrix({"a": af}, alpha=[2.0], re=[3e6]), {"a": af})
    assert rows[0]["converged"] and abs(rows[0]["cd"] - pol.cd[1]) <= 1e-3


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
    import time
    while w._thread is not None:                     # the solve runs in a worker thread (M8)
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    text = w.lbl_status.text()
    assert text.startswith("Viscous, Re 1e+06: converged"), text
    w.close()
    del app


# ---------------------------------------------------------------------------------------- #
# Single points: direct start, else continuation (user, after report 11)
# ---------------------------------------------------------------------------------------- #
@pytest.fixture
def scripted(monkeypatch):
    """``viscous.solve_point`` replaced by a script: angles in ``cold_ok`` converge from the march,
    every warm-started angle converges unless it, or the pair (angle, start angle), is in
    ``warm_fail``."""
    cfg = {"cold_ok": set(), "warm_fail": set(), "calls": []}

    def solve_point(coords, alpha, reynolds, params=viscous.VISCOUS, trips=None, start=None,
                    alpha_start=None, geometry=None, *a, **kw):
        cold = start is None
        cfg["calls"].append((float(alpha), "cold" if cold else f"warm from {alpha_start:g}"))
        ok = alpha in cfg["cold_ok"] if cold else not ({alpha, (alpha, alpha_start)} & cfg["warm_fail"])
        return viscous.ViscousResult(alpha=alpha, reynolds=reynolds, converged=ok, iterations=2,
                                     reason="" if ok else "max_iter", iterations_total=2,
                                     state=("x", alpha), path="march" if cold else "warm start")

    monkeypatch.setattr(viscous, "solve_point", solve_point)
    monkeypatch.setattr(viscous, "build_geometry", lambda c: None)
    return cfg


def test_single_point_direct_start(scripted):
    scripted["cold_ok"] = {5.0}
    r = viscous.solve_single(None, 5.0, 1e6)
    assert r.converged and r.path == "direct: march" and scripted["calls"] == [(5.0, "cold")]


def test_single_point_continuation_from_zero(scripted):
    scripted["cold_ok"] = {0.0}
    r = viscous.solve_single(None, -2.5, 1e6)
    assert r.converged and r.path == "continuation from 0 deg: warm start"
    assert scripted["calls"] == [(-2.5, "cold"), (0.0, "cold"), (-1.0, "warm from 0"),
                                 (-2.0, "warm from -1"), (-2.5, "warm from -2")]
    assert r.iterations == 2 and r.iterations_total == 10


def test_single_point_from_the_nearest_converged_angle_when_zero_fails(scripted):
    scripted["cold_ok"] = {2.0, 7.0}             # 0 fails; 7 (after 4, 6 and 3) is the nearest to converge
    r = viscous.solve_single(None, 5.0, 1e6)
    assert r.converged and r.path == "from 7 deg: warm start"
    assert [c for c in scripted["calls"] if c[1] == "cold"] == [(5.0, "cold"), (0.0, "cold"), (4.0, "cold"),
                                                                (6.0, "cold"), (3.0, "cold"), (7.0, "cold")]


def test_single_point_that_fails_says_what_was_tried(scripted):
    scripted["cold_ok"] = {0.0}
    scripted["warm_fail"] = {3.0}
    r = viscous.solve_single(None, 3.0, 1e6)
    assert not r.converged and r.path.startswith("direct: march; continuation from 0 deg failed; 2 deg: no start")
    assert r.path.endswith("cap of 10 cold starts reached")
    assert sum(c == "cold" for _, c in scripted["calls"]) == 12        # direct, 0 deg and the cap of 10
    scripted["cold_ok"], scripted["warm_fail"] = set(), set()
    r = viscous.solve_single(None, 1.0, 1e6)
    assert not r.converged and r.path.startswith("direct: march; 0 deg failed; 2 deg: no start")
    assert r.path.endswith("cap of 10 cold starts reached")


def test_single_point_moves_on_after_a_failed_walk(scripted):
    """As Clark Y Re 5e5 11 deg (report 12): the starts at 12 and 13 deg converge but their walks
    fail; 8 deg is reached and its walk converges."""
    scripted["cold_ok"] = {12.0, 13.0, 8.0}
    scripted["warm_fail"] = {(11.0, 12.0)}
    r = viscous.solve_single(None, 11.0, 1e6)
    assert r.converged and r.path == "from 8 deg: warm start"
    assert [a for a, c in scripted["calls"] if c == "cold"] == [11.0, 0.0, 10.0, 12.0, 9.0, 13.0, 8.0]


def test_single_point_without_fallback_is_the_direct_start(scripted):
    r = viscous.solve_single(None, 5.0, 1e6, fallback=False)
    assert not r.converged and r.path == "direct: march" and scripted["calls"] == [(5.0, "cold")]


# ---------------------------------------------------------------------------------------- #
# Closed trailing edge opened for viscous analysis (user, after report 11)
# ---------------------------------------------------------------------------------------- #
def test_closed_trailing_edge_is_opened_and_recorded():
    import dataclasses
    from foil_lab.geometry import trailing_edge_gap
    closed = generate_naca4("2412", 160, "closed")
    c, corr = viscous.open_closed_te(closed)
    assert abs(trailing_edge_gap(c) - 0.0005) <= 1e-6 and len(corr) == 1 and "0.0005 c" in corr[0]
    opened = generate_naca4("2412", 160)
    c2, corr2 = viscous.open_closed_te(opened)
    assert corr2 == [] and np.array_equal(c2, opened)
    sharp = dataclasses.replace(viscous.VISCOUS, closed_te_gap=0.0)
    c3, corr3 = viscous.open_closed_te(closed, sharp)
    assert corr3 == [] and np.array_equal(c3, closed)


def test_interfaces_record_the_opened_trailing_edge(fake_viscous):
    r = fl.analyze(fl.Airfoil.naca("2412", trailing_edge="closed"), 4.0, re=1e6)
    assert len(r.corrections) == 1 and "opened" in r.corrections[0]
    assert fl.analyze(fl.Airfoil.naca("2412"), 4.0, re=1e6).corrections == ()


@pytest.mark.slow
def test_sharp_naca2412_re3e6_0deg_now_agrees_with_xfoil_at_the_small_gap():
    """Closed NACA 2412 (160 panels), Re 3e6, 0 deg: the sharp path converged to Cl 0.059 (XFOIL
    sharp 0.2336). With the default opening (0.0005 c), Cl is within 0.03 of XFOIL 6.99 at 0.001 c
    (Cl 0.2425; PANE 160, N_crit 9; step 1 after report 10), or the point is reported as not
    converged."""
    s = fl.analyze(fl.Airfoil.from_coords(generate_naca4("2412", 160, "closed")), 0.0, re=3e6,
                   repanel=False)
    assert s.corrections
    assert (not s.converged) or abs(s.cl - 0.2425) <= 0.03, (s.cl, s.path)
