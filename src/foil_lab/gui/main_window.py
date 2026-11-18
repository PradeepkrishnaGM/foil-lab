"""Foil-Lab main window (PySide6): a thin client of the analysis API (M8, SPEC.md 4.9).

* State: a list of elements (``foil_lab.Element``), element 0 the main section; the geometry shown
  is always ``Configuration(elements)``, redrawn after every change.
* Every solve is ``fl.analyze`` or ``fl.polar`` with the conditions of the Conditions tab, run in a
  worker thread (:class:`SolveWorker`) with a progress bar and a Cancel button. A cancelled polar
  stops after the point being solved; a cancelled single point is discarded when it finishes.
* Plots read what the result says: surface labels from ``ElementResult.surface``, drag from
  ``Result.cd`` once (with its friction, pressure and induced parts listed), corrections and
  warnings in the status text. The flow-field tab draws the streamlines of the inviscid solution of
  the same geometry and conditions (:func:`foil_lab.flowfield.velocity_field`), labelled as such.
"""

from __future__ import annotations

import traceback
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

import foil_lab as fl
from foil_lab import export
from foil_lab.api import Cancelled

from .views import AirfoilPlotCanvas


class SolveWorker(QObject):
    """Runs one analysis call in a worker thread; emits progress, the result or an error."""

    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, kind: str, geometry, alphas, conditions: dict):
        super().__init__()
        self.kind, self.geometry, self.alphas, self.conditions = kind, geometry, alphas, conditions
        self.cancelled = False

    def _progress(self, done: int, total: int) -> None:
        if self.cancelled:
            raise Cancelled()
        self.progress.emit(done, total)

    def run(self) -> None:
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.kind == "single":
                    out = fl.analyze(self.geometry, self.alphas[0], **self.conditions)
                else:
                    out = fl.polar(self.geometry, self.alphas, progress=self._progress, **self.conditions)
            if self.cancelled:
                self.failed.emit("cancelled")
            else:
                self.finished.emit(out)
        except Cancelled:
            self.failed.emit("cancelled")
        except Exception as exc:                      # shown in the status line, never a crash
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


class FoilLabMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Foil-Lab {fl.__version__}")
        self.resize(1250, 820)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._create_sidebar())
        scroll.setMinimumWidth(320)
        splitter.addWidget(scroll)
        self.tabs = QTabWidget()
        self.single_plot = AirfoilPlotCanvas(self)
        self.polar_plot = AirfoilPlotCanvas(self)
        self.flow_plot = AirfoilPlotCanvas(self)
        self.tabs.addTab(self.single_plot, "Single Analysis")
        self.tabs.addTab(self.polar_plot, "Polar Sweep")
        self.tabs.addTab(self.flow_plot, "Flowfield (Streamlines)")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 1000])
        layout.addWidget(splitter)

        self.elements: List[fl.Element] = []
        self.last_result: Optional[fl.Result] = None
        self.last_polar: Optional[fl.Polar] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[SolveWorker] = None

    # ------------------------------------------------------------------ sidebar
    def _create_sidebar(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setAlignment(Qt.AlignTop)
        side = QTabWidget()
        outer.addWidget(side)
        for title, groups in (("Geometry", (self._geometry_group(), self._flap_group(), self._modifier_group())),
                              ("Conditions", (self._flow_group(), self._ground_group())),
                              ("Run", (self._single_group(), self._polar_group()))):
            tab = QWidget()
            lay = QVBoxLayout(tab)
            lay.setAlignment(Qt.AlignTop)
            for g in groups:
                lay.addWidget(g)
            side.addTab(tab, title)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        self.lbl_status = QLabel("No section loaded.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.progress)
        outer.addWidget(self.btn_cancel)
        outer.addWidget(self.lbl_status)
        return w

    def _spin(self, lo, hi, val, step=None, decimals=None, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        if decimals is not None:
            s.setDecimals(decimals)
        s.setValue(val)
        if step is not None:
            s.setSingleStep(step)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _geometry_group(self):
        g = QGroupBox("Section")
        lay = QVBoxLayout()
        self.btn_load = QPushButton("Load coordinate file")
        self.btn_load.clicked.connect(self.load_airfoil)
        row = QHBoxLayout()
        self.txt_naca = QLineEdit()
        self.txt_naca.setPlaceholderText("2412 | 23012 | e387 | bc:0.06 | wedge:0.05")
        self.txt_naca.setToolTip("NACA digits, a UIUC database name (downloaded once), "
                                 "or bc:/wedge: with a thickness")
        self.btn_gen_naca = QPushButton("Create")
        self.btn_gen_naca.clicked.connect(self.generate_naca)
        row.addWidget(self.txt_naca)
        row.addWidget(self.btn_gen_naca)
        drow = QHBoxLayout()
        self.spin_design_cl = self._spin(-0.5, 1.8, 0.8, 0.1)
        self.spin_design_cl.setPrefix("Cl* ")
        self.btn_design = QPushButton("Design for Cl")
        self.btn_design.setToolTip("PARSEC inverse design (inviscid, alpha 5 deg); replaces the section")
        self.btn_design.clicked.connect(self.design_for_target_cl)
        drow.addWidget(self.spin_design_cl)
        drow.addWidget(self.btn_design)
        lay.addWidget(self.btn_load)
        lay.addLayout(row)
        lay.addLayout(drow)
        g.setLayout(lay)
        return g

    def _flap_group(self):
        g = QGroupBox("Flaps")
        f = QFormLayout()
        self.spin_hinge = self._spin(0.1, 0.95, 0.75, 0.05, suffix=" c")
        self.spin_hinge_y = self._spin(0.0, 1.0, 0.5, 0.1)
        self.spin_hinge_y.setToolTip("hinge height as a fraction of the local thickness (0 lower, 1 upper)")
        self.spin_deflect = self._spin(-30.0, 30.0, 10.0, 1.0, suffix=" °")
        self.btn_apply_flap = QPushButton("Apply plain flap to the main element")
        self.btn_apply_flap.clicked.connect(self.apply_flap)
        self.btn_add_slotted = QPushButton("Add slotted flap element")
        self.btn_add_slotted.setToolTip("NACA 2412 of 0.3 c, 20 deg, leading edge at (0.97, -0.04) "
                                        "(examples/slotted_flap.py)")
        self.btn_add_slotted.clicked.connect(self.add_slotted_flap)
        self.btn_remove_extra = QPushButton("Remove extra elements")
        self.btn_remove_extra.clicked.connect(self.remove_extra_elements)
        f.addRow("Hinge x:", self.spin_hinge)
        f.addRow("Hinge height:", self.spin_hinge_y)
        f.addRow("Deflection (TE down +):", self.spin_deflect)
        f.addRow(self.btn_apply_flap)
        f.addRow(self.btn_add_slotted)
        f.addRow(self.btn_remove_extra)
        g.setLayout(f)
        return g

    def _modifier_group(self):
        g = QGroupBox("Place the main element")
        f = QFormLayout()
        self.spin_scale = self._spin(0.1, 10.0, 1.0, 0.1)
        self.spin_rot = self._spin(-180.0, 180.0, 0.0, 1.0, suffix=" °")
        self.spin_dx = self._spin(-10.0, 10.0, 0.0, 0.1)
        self.spin_dy = self._spin(-10.0, 10.0, 0.0, 0.1)
        self.spin_px = self._spin(-1.0, 2.0, 0.25, 0.05)
        self.spin_py = self._spin(-1.0, 1.0, 0.0, 0.05)
        self.btn_apply_mod = QPushButton("Apply placement")
        self.btn_apply_mod.clicked.connect(self.apply_modifiers)
        for lab, wdg in (("Scale:", self.spin_scale), ("Rotate, nose up +:", self.spin_rot),
                         ("Move x:", self.spin_dx), ("Move y:", self.spin_dy), ("Pivot x:", self.spin_px),
                         ("Pivot y:", self.spin_py)):
            f.addRow(lab, wdg)
        f.addRow(self.btn_apply_mod)
        g.setLayout(f)
        return g

    def _flow_group(self):
        g = QGroupBox("Flow conditions")
        f = QFormLayout()
        self.spin_re = self._spin(0.0, 1e8, 1e6, 1e5, decimals=0)
        self.spin_re.setToolTip("0: inviscid. Validated range 1e5 to 6e6.")
        self.spin_mach = self._spin(0.0, 0.95, 0.0, 0.05)
        self.spin_mach.setToolTip("Kármán-Tsien, inviscid only (set Re to 0)")
        self.spin_ar = self._spin(0.0, 100.0, 0.0, 1.0)
        self.spin_ar.setToolTip("0: the section (infinite wing)")
        self.spin_sweep = self._spin(-60.0, 60.0, 0.0, 5.0, suffix=" °")
        self.spin_ncrit = self._spin(2.0, 14.0, 9.0, 0.5)
        self.spin_trip_up = self._spin(0.0, 1.0, 0.0, 0.05, suffix=" x/c")
        self.spin_trip_lo = self._spin(0.0, 1.0, 0.0, 0.05, suffix=" x/c")
        for w in (self.spin_trip_up, self.spin_trip_lo):
            w.setToolTip("forced transition on the main element (0: free transition)")
        for lab, wdg in (("Reynolds number:", self.spin_re), ("Mach number:", self.spin_mach),
                         ("Aspect ratio:", self.spin_ar), ("Sweep:", self.spin_sweep),
                         ("N_crit (e^n):", self.spin_ncrit), ("Trip upper:", self.spin_trip_up),
                         ("Trip lower:", self.spin_trip_lo)):
            f.addRow(lab, wdg)
        g.setLayout(f)
        return g

    def _ground_group(self):
        g = QGroupBox("Ground effect")
        f = QFormLayout()
        self.chk_ground = QCheckBox("Ground (image method)")
        self.spin_height = self._spin(0.05, 10.0, 1.0, 0.1, suffix=" h/c")
        self.spin_height.setToolTip("height of the moment point (quarter chord) above the ground")
        f.addRow(self.chk_ground)
        f.addRow("Height:", self.spin_height)
        g.setLayout(f)
        return g

    def _single_group(self):
        g = QGroupBox("Single case")
        f = QFormLayout()
        self.spin_alpha = self._spin(-20.0, 25.0, 5.0, 0.5)
        self.btn_run_single = QPushButton("Run case")
        self.btn_run_single.clicked.connect(self.run_single_analysis)
        f.addRow("Alpha (deg):", self.spin_alpha)
        f.addRow(self.btn_run_single)
        g.setLayout(f)
        return g

    def _polar_group(self):
        g = QGroupBox("Polar")
        f = QFormLayout()
        self.spin_start = self._spin(-20.0, 25.0, -4.0, 1.0)
        self.spin_end = self._spin(-20.0, 25.0, 12.0, 1.0)
        self.spin_step = self._spin(0.25, 5.0, 1.0, 0.25)
        self.btn_run_polar = QPushButton("Run polar")
        self.btn_run_polar.clicked.connect(self.run_polar_analysis)
        self.btn_export_csv = QPushButton("Export polar (CSV)")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self.export_csv)
        for lab, wdg in (("Start alpha:", self.spin_start), ("End alpha:", self.spin_end),
                         ("Step:", self.spin_step)):
            f.addRow(lab, wdg)
        f.addRow(self.btn_run_polar)
        f.addRow(self.btn_export_csv)
        g.setLayout(f)
        self._enable(False)
        return g

    # ------------------------------------------------------------------ geometry
    def configuration(self) -> fl.Configuration:
        return fl.Configuration(tuple(self.elements), name=self.elements[0].airfoil.name)

    def _set_section(self, airfoil: fl.Airfoil, message: str) -> None:
        self.elements = [fl.Element(airfoil)]
        self.lbl_status.setText(message)
        self._enable(True)
        self._plot_geometry()

    def load_airfoil(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open coordinate file", "", "Coordinate files (*.dat *.txt);;All (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        try:
            af = fl.Airfoil.from_file(path)
        except Exception as exc:
            self.lbl_status.setText(f"Could not read {path}:\n{exc}")
            return
        self._set_section(af, f"Loaded {af.name} ({af.num_panels} panels as read; repanelled to "
                              f"{fl.params.GEOMETRY.num_panels} for analysis)")

    def generate_naca(self):
        text = self.txt_naca.text().strip()
        low = text.lower()
        try:
            if low.startswith("bc:"):
                af = fl.Airfoil.biconvex(float(low[3:] or 0.06))
            elif low.startswith("wedge:"):
                af = fl.Airfoil.wedge(float(low[6:] or 0.06))
            elif low.isdigit() and len(low) in (4, 5):
                af = fl.Airfoil.naca(low)
            elif low:
                af = fl.Airfoil.from_name(text)
            else:
                raise ValueError("enter NACA digits, a UIUC name, bc:t or wedge:t")
        except Exception as exc:
            self.lbl_status.setText(f"Could not create the section:\n{exc}")
            return
        self._set_section(af, f"Created {af.name}\nsource: {af.source}")

    def design_for_target_cl(self):
        from foil_lab.design import design_for_cl
        target = self.spin_design_cl.value()
        try:
            r = design_for_cl(target_cl=target, alpha_deg=5.0, num_panels=120, max_iter=60)
        except Exception as exc:
            self.lbl_status.setText(f"Design failed:\n{exc}")
            return
        self._set_section(fl.Airfoil.from_coords(r.coords, f"PARSEC for Cl {target:.2f}"),
                          f"Designed: {r.message} ({r.n_evals} evaluations)")

    def apply_flap(self):
        if not self.elements:
            return
        main = self.elements[0]
        base = main.airfoil
        if base.flap is not None:                     # re-flap the undeflected section
            base = fl.Airfoil.from_coords(base.flap.base, base.name, base.source)
        try:
            flapped = base.with_flap(self.spin_hinge.value(), self.spin_deflect.value(), self.spin_hinge_y.value())
        except Exception as exc:
            self.lbl_status.setText(f"Flap not applied:\n{exc}")
            return
        self.elements[0] = fl.Element(flapped, main.offset, main.scale, main.rotation_deg, main.pivot)
        self.lbl_status.setText(f"Plain flap: hinge {self.spin_hinge.value():g} c, "
                                f"{self.spin_deflect.value():g} deg (trailing edge down +)")
        self._plot_geometry()

    def add_slotted_flap(self):
        if not self.elements:
            return
        self.elements.append(fl.Element(fl.Airfoil.naca("2412"), scale=0.3, rotation_deg=20.0,
                                        pivot=(0.0, 0.0), offset=(0.97, -0.04)))
        self.lbl_status.setText(f"{len(self.elements)} elements. Limits: no wake merging, no slot "
                                "separation model, no maximum-lift prediction; not validated against "
                                "a multi-element reference.")
        self._plot_geometry()

    def remove_extra_elements(self):
        if len(self.elements) > 1:
            self.elements = self.elements[:1]
            self.lbl_status.setText("Extra elements removed.")
            self._plot_geometry()

    def apply_modifiers(self):
        if not self.elements:
            return
        main = self.elements[0]
        self.elements[0] = fl.Element(main.airfoil, offset=(self.spin_dx.value(), self.spin_dy.value()),
                                      scale=self.spin_scale.value(), rotation_deg=self.spin_rot.value(),
                                      pivot=(self.spin_px.value(), self.spin_py.value()))
        self.lbl_status.setText("Placement applied to the main element (reference chord and moment "
                                "point follow it).")
        self._plot_geometry()

    # ------------------------------------------------------------------ conditions and runs
    def conditions(self) -> dict:
        re = self.spin_re.value()
        c = dict(re=re or None, mach=self.spin_mach.value(), sweep=self.spin_sweep.value(),
                 aspect_ratio=self.spin_ar.value() or None,
                 ground_height=self.spin_height.value() if self.chk_ground.isChecked() else None)
        if re:
            c.update(n_crit=self.spin_ncrit.value(), trip_upper=self.spin_trip_up.value() or None,
                     trip_lower=self.spin_trip_lo.value() or None)
        return c

    def _enable(self, on: bool) -> None:
        for b in (getattr(self, n, None) for n in ("btn_apply_flap", "btn_add_slotted", "btn_remove_extra",
                                                   "btn_apply_mod", "btn_run_single", "btn_run_polar")):
            if b is not None:
                b.setEnabled(on)

    def _start(self, kind: str, alphas) -> None:
        if not self.elements:
            self.lbl_status.setText("Load or create a section first.")
            return
        if self._thread is not None:
            return
        self._worker = SolveWorker(kind, self.configuration(), alphas, self.conditions())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._enable(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, len(alphas) if kind == "polar" else 0)   # busy indicator for one point
        self.progress.setValue(0)
        self.lbl_status.setText(f"Solving {'1 point' if kind == 'single' else f'{len(alphas)} points'}...")
        self._kind = kind
        self._thread.start()

    def run_single_analysis(self):
        self._start("single", [self.spin_alpha.value()])

    def run_polar_analysis(self):
        a0, a1, da = self.spin_start.value(), self.spin_end.value(), self.spin_step.value()
        if a1 < a0:
            self.lbl_status.setText("End alpha is below start alpha.")
            return
        self._start("polar", [round(float(a), 10) for a in np.arange(a0, a1 + 1e-9, da)])

    def cancel(self):
        if self._worker is not None:
            self._worker.cancelled = True
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling after the point being solved...")

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_failed(self, message: str) -> None:
        self.lbl_status.setText("Cancelled." if message == "cancelled" else f"Solver error:\n{message}")

    def _on_finished(self, out) -> None:
        if self._kind == "single":
            self.last_result = out
            self.show_result(out)
        else:
            self.last_polar = out
            self.show_polar(out)

    def _cleanup(self) -> None:
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.btn_cancel.setEnabled(False)
        self._enable(bool(self.elements))

    # ------------------------------------------------------------------ display
    @staticmethod
    def summary(r: fl.Result) -> str:
        """Status text of one result: drag once, with its parts; corrections and warnings."""
        if r.reynolds:
            head = f"Viscous, Re {r.reynolds:.3g}: " + (f"converged ({r.iterations} it)" if r.converged
                                                         else "NOT converged")
        else:
            head = "Inviscid"
        lines = [head, f"Cl {r.cl:.4f}   Cm {r.cm:+.4f}"]
        if r.cd is not None:
            parts = []
            if r.cd_friction is not None:
                parts += [f"friction {r.cd_friction:.5f}", f"pressure {r.cd_pressure:.5f}"]
            if r.cd_induced is not None:
                parts.append(f"induced {r.cd_induced:.5f}")
            lines.append(f"Cd {r.cd:.5f} ({', '.join(parts)})")
        elif r.cd_induced is not None:
            lines.append(f"induced drag {r.cd_induced:.5f} (no profile drag: inviscid)")
        flapped = [e for e in r.elements if e.ch is not None]
        if len(flapped) == 1:
            lines.append(f"hinge moment Ch {r.ch:+.5f} (trailing edge down +)")
        if len(r.elements) > 1:
            lines += [f"  element {k}: Cl {e.cl:.4f}" + (f", Cd {e.cd:.5f}" if e.cd is not None else "")
                      for k, e in enumerate(r.elements)]
        if r.mach_critical is not None and np.isfinite(r.mach_critical):
            lines.append(f"critical Mach {r.mach_critical:.3f}")
        lines += [f"correction: {c}" for c in r.corrections]
        lines += [f"WARNING: {w}" for w in r.warnings]
        return "\n".join(lines)

    def show_result(self, r: fl.Result) -> None:
        self.lbl_status.setText(self.summary(r))
        fig = self.single_plot.fig
        fig.clear()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for k, e in enumerate(r.elements):
            ax1.plot(e.x, e.y, "k-", lw=1.0)
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.axis("off")
        cd = "-" if r.cd is None else f"{r.cd:.5f}"
        ax1.set_title(f"alpha {r.alpha:g} deg: Cl {r.cl:.4f}, Cd {cd}, Cm {r.cm:+.4f}"
                      + ("" if r.converged else "  (NOT converged)"))
        for k, e in enumerate(r.elements):
            name = "main" if k == 0 else f"element {k}"
            for surf, ls in (("upper", "-"), ("lower", "--")):
                m = np.asarray(e.surface) == surf
                ax2.plot(np.asarray(e.x)[m], np.asarray(e.cp)[m], ls, color=f"C{k}", label=f"{name} {surf}")
        ax2.invert_yaxis()
        ax2.set_xlabel("x")
        ax2.set_ylabel("Cp" + (" (normal plane)" if r.sweep else ""))
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize="small")
        fig.tight_layout()
        self.single_plot.redraw()
        self.tabs.setCurrentIndex(0)
        self.show_flowfield(r)

    def show_flowfield(self, r: fl.Result) -> None:
        from foil_lab.flowfield import velocity_field
        fig = self.flow_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        try:
            ff = velocity_field(self.configuration(), r.alpha, mach=0.0, sweep=r.sweep,
                                ground_height=r.ground_height, grid_res=110)
        except Exception as exc:
            ax.set_title(f"flow field not available: {exc}")
            self.flow_plot.redraw()
            return
        speed = np.hypot(ff.u, ff.v)
        speed = np.ma.masked_where(ff.inside, speed)
        strm = ax.streamplot(ff.x, ff.y, np.ma.masked_where(ff.inside, ff.u),
                             np.ma.masked_where(ff.inside, ff.v), color=speed, cmap="viridis",
                             linewidth=1.0, density=1.6, arrowsize=0.8)
        for b in ff.bodies:
            ax.fill(b[:, 0], b[:, 1], color="0.6", zorder=5)
            ax.plot(b[:, 0], b[:, 1], "k-", lw=1.0, zorder=6)
        if r.ground_height is not None:
            ax.axhline(0.0, color="k", lw=2.0, zorder=6)
        ax.set_xlim(ff.x.min(), ff.x.max())
        ax.set_ylim(ff.y.min(), ff.y.max())
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Streamlines of the inviscid solution, alpha {r.alpha:g} deg"
                     + (" (viscous result: displacement effect not shown)" if r.reynolds else ""), fontsize=10)
        fig.colorbar(strm.lines, ax=ax, label="speed / freestream", fraction=0.04, pad=0.03)
        fig.tight_layout()
        self.flow_plot.redraw()

    def show_polar(self, pol: fl.Polar) -> None:
        from foil_lab import plot
        n_ok = int(np.sum(pol.converged))
        viscous = bool(pol.results and pol.results[0].reynolds)
        self.lbl_status.setText(f"Polar: {n_ok} of {len(pol.results)} points converged"
                                + ("" if viscous else " (inviscid: no profile drag)")
                                + "".join(f"\ncorrection: {c}" for c in (pol.results[0].corrections if pol.results else ())))
        fig = self.polar_plot.fig
        fig.clear()
        axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
        plot.polars(pol, labels=[self.configuration().name], axes=axes)
        for ax, title in zip(axes, ("Lift", "Drag polar (total Cd)", "Moment")):
            ax.set_title(title)
        fig.tight_layout()
        self.polar_plot.redraw()
        self.btn_export_csv.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    def export_csv(self):
        if self.last_polar is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save polar", "", "CSV files (*.csv)")
        if path:
            try:
                export.to_csv(self.last_polar, path)
                self.lbl_status.setText(f"Polar written to {path}")
            except Exception as exc:
                self.lbl_status.setText(f"Export failed:\n{exc}")

    def _plot_geometry(self) -> None:
        """Redraw the geometry after every change (new axes on a cleared figure)."""
        from foil_lab import plot
        fig = self.single_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        plot.geometry(self.configuration(), ax=ax)
        ax.set_title(f"{self.configuration().name}: {len(self.elements)} element(s)")
        ax.legend(fontsize="small")
        fig.tight_layout()
        self.single_plot.redraw()
        self.tabs.setCurrentIndex(0)


def run():
    """Console entry point for the GUI (``foillab-gui``)."""
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FoilLabMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
