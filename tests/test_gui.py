"""M8: the GUI is a thin client of the API (SPEC.md 4.9), driven headless (Qt offscreen platform)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

import foil_lab as fl  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


@pytest.fixture
def win(app):
    from foil_lab.gui.main_window import FoilLabMainWindow
    w = FoilLabMainWindow()
    w.show()
    yield w
    w.close()


def _wait(app, win, timeout=120.0):
    t0 = time.time()
    while win._thread is not None and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert win._thread is None, "solve did not finish"


def _create(win, text):
    win.txt_naca.setText(text)
    win.btn_gen_naca.click()


def test_title_and_geometry_redraw(app, win):
    assert fl.__version__ in win.windowTitle()
    _create(win, "2412")
    assert len(win.elements) == 1 and win.btn_run_single.isEnabled()
    ax = win.single_plot.fig.axes[0]
    xs = np.concatenate([ln.get_xdata() for ln in ax.lines])
    assert xs.max() == pytest.approx(1.0, abs=2e-3)
    win.spin_scale.setValue(0.5)
    win.btn_apply_mod.click()                                    # redrawn after the change
    xs = np.concatenate([ln.get_xdata() for ln in win.single_plot.fig.axes[0].lines])
    assert xs.max() < 0.6
    win.btn_apply_flap.click()
    assert win.elements[0].airfoil.flap is not None
    win.btn_add_slotted.click()
    assert len(win.elements) == 2 and len(win.single_plot.fig.axes[0].lines) == 2
    win.btn_remove_extra.click()
    assert len(win.elements) == 1


def test_single_case_matches_the_api_and_labels_surfaces(app, win):
    _create(win, "2412")
    win.spin_re.setValue(0.0)
    win.spin_alpha.setValue(4.0)
    win.btn_run_single.click()
    _wait(app, win)
    r = win.last_result
    ref = fl.analyze(fl.Airfoil.naca("2412"), 4.0)
    assert r.cl == pytest.approx(ref.cl, abs=1e-12)
    cp_ax = win.single_plot.fig.axes[1]
    lines = {ln.get_label(): ln for ln in cp_ax.lines}
    up, lo = lines["main upper"], lines["main lower"]
    assert np.min(up.get_ydata()) < np.min(lo.get_ydata())     # suction peak on the upper surface
    assert f"Cl {ref.cl:.4f}" in win.lbl_status.text()
    # flow field: axes limits are the grid, not a tiny blob; no flow inside the body
    fax = win.flow_plot.fig.axes[0]
    assert fax.get_xlim()[1] - fax.get_xlim()[0] < 3.0
    assert "inviscid" in fax.get_title()


def test_flowfield_has_no_flow_through_the_surface():
    from foil_lab.flowfield import velocity_field
    from foil_lab import inviscid as core
    af = fl.Airfoil.naca("2412")
    ff = velocity_field(af, 4.0, grid_res=40)
    assert ff.inside.any() and np.all(ff.u[ff.inside] == 0.0)
    # just outside the surface the normal velocity is small (no-penetration of the same solution)
    b = np.asarray(af.repanelled(160).coords)
    ps = core.PanelSet.from_bodies([b])
    sol = core.solve(ps, 4.0)
    nx, ny = np.asarray(ps.normal[0]), np.asarray(ps.normal[1])
    k = np.r_[10:60, 100:150]                     # away from the nose and the trailing edge
    px = np.asarray(ps.mid_x)[k] + 0.002 * nx[k]
    py = np.asarray(ps.mid_y)[k] + 0.002 * ny[k]
    u, v = core.induced_velocity(ps, np.asarray(sol.gamma), px, py)
    a = np.radians(4.0)
    vn = (np.cos(a) + u) * nx[k] + (np.sin(a) + v) * ny[k]
    assert np.max(np.abs(vn)) < 0.02


def test_drag_shown_once_with_parts():
    from foil_lab.gui.main_window import FoilLabMainWindow
    r = fl.analyze(fl.Airfoil.naca("2412"), 4.0, aspect_ratio=8.0, sweep=20.0)
    text = FoilLabMainWindow.summary(r)
    assert text.count(f"{r.cd_induced:.5f}") == 1 and "simple sweep" in text and "finite wing" in text


def test_polar_with_progress_cancel_and_export(app, win, tmp_path):
    _create(win, "0012")
    win.spin_re.setValue(0.0)
    win.spin_start.setValue(-2.0)
    win.spin_end.setValue(4.0)
    win.btn_run_polar.click()
    _wait(app, win)
    assert len(win.last_polar.results) == 7 and win.progress.value() == win.progress.maximum()
    from foil_lab import export
    export.to_csv(win.last_polar, tmp_path / "p.csv")
    assert (tmp_path / "p.csv").read_text().count("\n") == 8
    # cancel: the worker's progress callback raises; the status says so and no polar is kept
    win.last_polar = None
    win.spin_end.setValue(20.0)
    win.btn_run_polar.click()
    win.cancel()
    _wait(app, win)
    assert win.last_polar is None and "Cancel" in win.lbl_status.text()


@pytest.mark.slow
def test_viscous_multi_element_and_ground_runs(app, win):
    _create(win, "2412")
    win.btn_add_slotted.click()
    win.spin_re.setValue(3e6)
    win.spin_alpha.setValue(4.0)
    win.btn_run_single.click()
    _wait(app, win, 600)
    r = win.last_result
    assert r is not None and r.converged and len(r.elements) == 2
    assert "element 1" in win.lbl_status.text()
    win.btn_remove_extra.click()
    win.spin_re.setValue(0.0)
    win.chk_ground.setChecked(True)
    win.spin_height.setValue(0.5)
    win.btn_run_single.click()
    _wait(app, win)
    assert win.last_result.ground_height == 0.5 and "ground effect" in win.lbl_status.text()
