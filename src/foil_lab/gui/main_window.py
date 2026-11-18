import numpy as np
import matplotlib.pyplot as plt
from typing import List
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QGroupBox, 
                               QDoubleSpinBox, QFormLayout, QTabWidget, QLineEdit,
                               QCheckBox, QScrollArea, QSplitter)
from PySide6.QtCore import Qt

from .views import AirfoilPlotCanvas
from foil_lab.utils.io import load_dat, export_polar_csv
from foil_lab.geometry import (normalize_airfoil, repanel_cosine,
                               translate, scale, rotate, smooth_coordinates)
from foil_lab.solver import solve_flow, compute_panel_geometry
from foil_lab.analysis import run_alpha_sweep
from foil_lab.generators import (generate_naca4, generate_naca5, generate_parsec,
                                 generate_biconvex, generate_wedge)
from foil_lab.flowfield import compute_velocity_field
from foil_lab.ground_effect import solve_ground_effect
from foil_lab.advanced import solve_swept_wing

class FoilLabMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foil Lab v1.3 — First Low-Re VII Convergence")
        self.resize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Resizable split between sidebar and display (draggable divider) ---
        splitter = QSplitter(Qt.Horizontal)

        # --- Sidebar (Left), wrapped in a scroll area so it never overflows ---
        self.sidebar = self._create_sidebar()
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(self.sidebar)
        sidebar_scroll.setMinimumWidth(300)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        splitter.addWidget(sidebar_scroll)

        # --- Main Display Area (Right) ---
        self.tabs = QTabWidget()

        self.single_plot = AirfoilPlotCanvas(self)
        self.tabs.addTab(self.single_plot, "Single Analysis")

        self.polar_plot = AirfoilPlotCanvas(self)
        self.tabs.addTab(self.polar_plot, "Polar Sweep")

        self.flow_plot = AirfoilPlotCanvas(self)
        self.tabs.addTab(self.flow_plot, "Flowfield (Streamlines)")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)   # sidebar keeps its size
        splitter.setStretchFactor(1, 1)   # plot area takes extra space
        splitter.setSizes([340, 1000])
        main_layout.addWidget(splitter)

        # State Variables
        self.current_coords = None   
        # Stores a list of bodies to support Multi-Element physics
        self.panelled_bodies: List[np.ndarray] = []  
        self.last_polar = None       

    def _create_sidebar(self) -> QWidget:
        # The sidebar groups are organized into logical tabs so the control list
        # stays compact and navigable instead of one long overflowing column.
        sidebar_widget = QWidget()
        outer = QVBoxLayout(sidebar_widget)
        outer.setAlignment(Qt.AlignTop)
        side_tabs = QTabWidget()
        outer.addWidget(side_tabs)

        # Build the individual groups (defined below as helper methods).
        geom_group = self._build_geometry_group()
        flap_group = self._build_flap_group()
        mod_group = self._build_mod_group()
        flight_group, ground_group = self._build_flow_groups()
        single_group, polar_group = self._build_run_groups()

        # Tab 1: Geometry & shape.
        geom_tab = QWidget(); gl = QVBoxLayout(geom_tab); gl.setAlignment(Qt.AlignTop)
        gl.addWidget(geom_group); gl.addWidget(flap_group); gl.addWidget(mod_group)
        side_tabs.addTab(geom_tab, "Geometry")

        # Tab 2: Flow conditions.
        flow_tab = QWidget(); fl = QVBoxLayout(flow_tab); fl.setAlignment(Qt.AlignTop)
        fl.addWidget(flight_group); fl.addWidget(ground_group)
        side_tabs.addTab(flow_tab, "Conditions")

        # Tab 3: Run / sweep.
        run_tab = QWidget(); rl = QVBoxLayout(run_tab); rl.setAlignment(Qt.AlignTop)
        rl.addWidget(single_group); rl.addWidget(polar_group)
        side_tabs.addTab(run_tab, "Run")

        return sidebar_widget

    # --- Sidebar group builders (each returns a QGroupBox) ---

    def _build_geometry_group(self):
        geom_group = QGroupBox("Geometry")
        geom_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load .dat File")
        self.btn_load.clicked.connect(self.load_airfoil)
        naca_layout = QHBoxLayout()
        self.txt_naca = QLineEdit()
        self.txt_naca.setPlaceholderText("2412 | 23012 | bc:0.06 | wedge:0.05")
        self.btn_gen_naca = QPushButton("Generate NACA")
        self.btn_gen_naca.clicked.connect(self.generate_naca)
        naca_layout.addWidget(self.txt_naca)
        naca_layout.addWidget(self.btn_gen_naca)
        # Inverse design: generate a PARSEC airfoil hitting a target Cl.
        design_layout = QHBoxLayout()
        self.spin_design_cl = QDoubleSpinBox()
        self.spin_design_cl.setRange(-0.5, 1.8); self.spin_design_cl.setValue(0.8)
        self.spin_design_cl.setSingleStep(0.1); self.spin_design_cl.setPrefix("Cl* ")
        self.btn_design = QPushButton("Design for Cl")
        self.btn_design.clicked.connect(self.design_for_target_cl)
        design_layout.addWidget(self.spin_design_cl)
        design_layout.addWidget(self.btn_design)
        self.lbl_status = QLabel("No airfoil loaded.")
        self.lbl_status.setWordWrap(True)
        geom_layout.addWidget(self.btn_load)
        geom_layout.addLayout(naca_layout)
        geom_layout.addLayout(design_layout)
        geom_layout.addWidget(self.lbl_status)
        geom_group.setLayout(geom_layout)
        return geom_group

    def design_for_target_cl(self):
        """Run inverse design for the target Cl and load the resulting shape."""
        from foil_lab.design import design_for_cl
        target = self.spin_design_cl.value()
        self.lbl_status.setText(f"Designing for Cl={target:.2f}...")
        try:
            r = design_for_cl(target_cl=target, alpha_deg=5.0, num_panels=120,
                              max_iter=60)
            self.current_coords = r.coords
            self.panelled_bodies = [np.copy(r.coords)]
            self.lbl_status.setText(
                f"Designed: {r.message}\n"
                f"t/c={r.coords[:,1].max()-r.coords[:,1].min():.3f}, "
                f"{r.n_evals} evals")
            self._enable_ui_buttons()
            self._plot_geometry()
        except Exception as e:
            self.lbl_status.setText(f"Design Error:\n{str(e)}")

    def _build_flap_group(self):
        flap_group = QGroupBox("Flap Tool")
        flap_layout = QFormLayout()
        self.spin_hinge = QDoubleSpinBox(); self.spin_hinge.setRange(0.1, 0.95); self.spin_hinge.setValue(0.75); self.spin_hinge.setSingleStep(0.05); self.spin_hinge.setSuffix(" c")
        self.spin_deflect = QDoubleSpinBox(); self.spin_deflect.setRange(-30.0, 30.0); self.spin_deflect.setValue(10.0); self.spin_deflect.setSuffix(" °")
        self.btn_apply_flap = QPushButton("Apply Plain Flap")
        self.btn_apply_flap.setEnabled(False)
        self.btn_apply_flap.clicked.connect(self.apply_flap)
        self.btn_add_slotted = QPushButton("Add Slotted Flap")
        self.btn_add_slotted.setEnabled(False)
        self.btn_add_slotted.clicked.connect(self.add_slotted_flap)
        flap_layout.addRow("Hinge Location:", self.spin_hinge)
        flap_layout.addRow("Deflection:", self.spin_deflect)
        flap_layout.addRow(self.btn_apply_flap)
        flap_layout.addRow(self.btn_add_slotted)
        flap_group.setLayout(flap_layout)
        return flap_group

    def _build_mod_group(self):
        mod_group = QGroupBox("Geometry Modifiers")
        mod_layout = QFormLayout()
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setRange(0.1, 10.0); self.spin_scale.setValue(1.0); self.spin_scale.setSingleStep(0.1)
        self.spin_rot = QDoubleSpinBox(); self.spin_rot.setRange(-180, 180); self.spin_rot.setValue(0.0)
        self.spin_dx = QDoubleSpinBox(); self.spin_dx.setRange(-10, 10); self.spin_dx.setValue(0.0); self.spin_dx.setSingleStep(0.1)
        self.spin_dy = QDoubleSpinBox(); self.spin_dy.setRange(-10, 10); self.spin_dy.setValue(0.0); self.spin_dy.setSingleStep(0.1)
        self.spin_px = QDoubleSpinBox(); self.spin_px.setRange(-1.0, 2.0); self.spin_px.setValue(0.25); self.spin_px.setSingleStep(0.1)
        self.spin_py = QDoubleSpinBox(); self.spin_py.setRange(-1.0, 1.0); self.spin_py.setValue(0.0); self.spin_py.setSingleStep(0.1)
        self.btn_apply_mod = QPushButton("Apply Transformations")
        self.btn_apply_mod.setEnabled(False)
        self.btn_apply_mod.clicked.connect(self.apply_modifiers)
        mod_layout.addRow("Scale:", self.spin_scale)
        mod_layout.addRow("Rotate (°):", self.spin_rot)
        mod_layout.addRow("Translate X:", self.spin_dx)
        mod_layout.addRow("Translate Y:", self.spin_dy)
        mod_layout.addRow("Pivot X (Rot):", self.spin_px)
        mod_layout.addRow("Pivot Y (Rot):", self.spin_py)
        mod_layout.addRow(self.btn_apply_mod)
        mod_group.setLayout(mod_layout)
        return mod_group

    def _build_flow_groups(self):
        flight_group = QGroupBox("Flight Conditions")
        flight_layout = QFormLayout()
        # Re 0: inviscid. Re > 0: the coupled viscous-inviscid solver (one body, Mach 0).
        self.spin_re = QDoubleSpinBox(); self.spin_re.setRange(0.0, 1e8); self.spin_re.setValue(1000000); self.spin_re.setDecimals(0); self.spin_re.setSingleStep(100000)
        self.spin_mach = QDoubleSpinBox(); self.spin_mach.setRange(0.0, 0.75); self.spin_mach.setValue(0.0); self.spin_mach.setSingleStep(0.05)
        self.spin_ar = QDoubleSpinBox(); self.spin_ar.setRange(0.0, 100.0); self.spin_ar.setValue(0.0); self.spin_ar.setToolTip("Set to 0 for infinite 2D wing")
        self.spin_ncrit = QDoubleSpinBox(); self.spin_ncrit.setRange(2.0, 14.0); self.spin_ncrit.setValue(9.0); self.spin_ncrit.setSingleStep(0.5)
        self.spin_ncrit.setToolTip("e^n transition N_crit: ~9 clean, ~4-7 noisy/rough, ~11-13 very quiet")
        self.spin_sweep = QDoubleSpinBox(); self.spin_sweep.setRange(0.0, 75.0); self.spin_sweep.setValue(0.0); self.spin_sweep.setSuffix(" °"); self.spin_sweep.setToolTip("Leading-edge sweep (Busemann simple-sweep approximation)")
        # Trip wires: explicit x/c transition flags per surface (0 = natural).
        self.spin_trip_up = QDoubleSpinBox(); self.spin_trip_up.setRange(0.0, 1.0); self.spin_trip_up.setValue(0.0); self.spin_trip_up.setSingleStep(0.05); self.spin_trip_up.setSuffix(" x/c"); self.spin_trip_up.setToolTip("Force upper-surface transition at this x/c (0 = natural)")
        self.spin_trip_lo = QDoubleSpinBox(); self.spin_trip_lo.setRange(0.0, 1.0); self.spin_trip_lo.setValue(0.0); self.spin_trip_lo.setSingleStep(0.05); self.spin_trip_lo.setSuffix(" x/c"); self.spin_trip_lo.setToolTip("Force lower-surface transition at this x/c (0 = natural)")
        flight_layout.addRow("Reynolds Number:", self.spin_re)
        flight_layout.addRow("Mach Number:", self.spin_mach)
        flight_layout.addRow("Aspect Ratio:", self.spin_ar)
        flight_layout.addRow("Sweep Angle:", self.spin_sweep)
        flight_layout.addRow("Trip Upper:", self.spin_trip_up)
        flight_layout.addRow("Trip Lower:", self.spin_trip_lo)
        flight_layout.addRow("e^n N_crit:", self.spin_ncrit)
        flight_group.setLayout(flight_layout)

        ground_group = QGroupBox("Surface / Ground Effect")
        ground_layout = QFormLayout()
        self.chk_ground = QCheckBox("Enable Surface Effect")
        self.chk_ground.setToolTip("Rigid-wall Method of Images (ground / low Froude)")
        self.spin_height = QDoubleSpinBox(); self.spin_height.setRange(0.1, 10.0); self.spin_height.setValue(1.0); self.spin_height.setSingleStep(0.1); self.spin_height.setSuffix(" h/c")
        ground_layout.addRow(self.chk_ground)
        ground_layout.addRow("Clearance/Depth:", self.spin_height)
        ground_group.setLayout(ground_layout)
        return flight_group, ground_group

    def _build_run_groups(self):
        single_group = QGroupBox("Single Case")
        single_layout = QFormLayout()
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(-20.0, 20.0)
        self.spin_alpha.setValue(5.0)
        self.btn_run_single = QPushButton("Run Case")
        self.btn_run_single.setEnabled(False)
        self.btn_run_single.clicked.connect(self.run_single_analysis)
        single_layout.addRow("Alpha:", self.spin_alpha)
        single_layout.addRow(self.btn_run_single)
        single_group.setLayout(single_layout)

        polar_group = QGroupBox("Polar Sweep")
        polar_layout = QFormLayout()
        self.spin_start = QDoubleSpinBox(); self.spin_start.setRange(-20, 20); self.spin_start.setValue(-5)
        self.spin_end = QDoubleSpinBox(); self.spin_end.setRange(-20, 20); self.spin_end.setValue(15)
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0.1, 5); self.spin_step.setValue(1.0)
        self.btn_run_polar = QPushButton("Run Alpha Sweep")
        self.btn_run_polar.setEnabled(False)
        self.btn_run_polar.clicked.connect(self.run_polar_analysis)
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self.export_csv)
        polar_layout.addRow("Start Alpha:", self.spin_start)
        polar_layout.addRow("End Alpha:", self.spin_end)
        polar_layout.addRow("Step:", self.spin_step)
        polar_layout.addRow(self.btn_run_polar)
        polar_layout.addRow(self.btn_export_csv)
        polar_group.setLayout(polar_layout)
        return single_group, polar_group

    # --- Core Logic Methods ---

    def load_airfoil(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Airfoil", "", "Data Files (*.dat)")
        if filepath:
            raw_coords = load_dat(filepath)
            self.current_coords = normalize_airfoil(raw_coords)
            self.panelled_bodies = [repanel_cosine(self.current_coords, num_panels=120)]
            self.lbl_status.setText(f"Loaded: {filepath.split('/')[-1]}")
            self._enable_ui_buttons()
            self._plot_geometry()

    def generate_naca(self):
        naca_str = self.txt_naca.text().strip().lower()
        try:
            # Advanced shapes via simple prefixes; plain digits stay NACA 4/5.
            if naca_str.startswith("bc:"):
                tc = float(naca_str[3:]) if naca_str[3:] else 0.06
                self.current_coords = generate_biconvex(tc, num_panels=120)
                label = f"Biconvex (t/c={tc})"
            elif naca_str.startswith("wedge:"):
                tc = float(naca_str[6:]) if naca_str[6:] else 0.06
                self.current_coords = generate_wedge(tc, num_panels=120)
                label = f"Double-wedge (t/c={tc})"
            elif len(naca_str) == 4 and naca_str.isdigit():
                self.current_coords = generate_naca4(naca_str, num_panels=120)
                label = f"NACA {naca_str}"
            elif len(naca_str) == 5 and naca_str.isdigit():
                self.current_coords = generate_naca5(naca_str, num_panels=120)
                label = f"NACA {naca_str}"
            else:
                raise ValueError(
                    "Enter 4/5 NACA digits, 'bc:tc', or 'wedge:tc'."
                )

            # Reset the bodies list to just the newly generated wing
            self.panelled_bodies = [np.copy(self.current_coords)]
            self.lbl_status.setText(f"Generated: {label}")
            self._enable_ui_buttons()
            self._plot_geometry()
        except Exception as e:
            self.lbl_status.setText(f"Generation Error:\n{str(e)}")

    def apply_flap(self):
        if self.current_coords is None: return
        hinge = self.spin_hinge.value()
        deflect = self.spin_deflect.value()
        from foil_lab.flap import make_flap
        try:
            deflected, _ = make_flap(self.current_coords, hinge, deflect, num_panels=120)
        except ValueError as e:
            self.lbl_status.setText(f"Flap not applied:\n{e}")
            return
        # Resets to a single body (overwriting any slotted flaps)
        self.panelled_bodies = [deflected]
        self.lbl_status.setText(f"Flap Applied:\nHinge: {hinge}c, Deflection: {deflect}°")
        self._plot_geometry()

    def add_slotted_flap(self):
        if not self.panelled_bodies: return
        
        # 1. Spawn a symmetric NACA 0012 to use as a flap
        flap_coords = generate_naca4("0012", num_panels=80)
        
        # 2. Scale it down to 30% chord, pitch it down 25 deg, and translate it to the trailing edge
        flap_coords = scale(flap_coords, 0.3)
        flap_coords = rotate(flap_coords, 25.0, pivot=(0.0, 0.0))
        flap_coords = translate(flap_coords, 0.95, -0.05)
        
        # 3. Append it as a second independent body!
        self.panelled_bodies.append(flap_coords)
        self.lbl_status.setText("Added Slotted Flap")
        self._plot_geometry()

    def apply_modifiers(self):
        if self.current_coords is None: return
        coords = np.copy(self.current_coords)
        s = self.spin_scale.value()
        if s != 1.0: coords = scale(coords, s)
        r = self.spin_rot.value()
        if r != 0.0:
            pivot = (self.spin_px.value(), self.spin_py.value())
            coords = rotate(coords, r, pivot)
        dx, dy = self.spin_dx.value(), self.spin_dy.value()
        if dx != 0.0 or dy != 0.0:
            coords = translate(coords, dx, dy)
            
        self.panelled_bodies = [repanel_cosine(coords, num_panels=120)]
        self.lbl_status.setText("Transformations applied.")
        self._plot_geometry()

    def run_single_analysis(self):
        alpha = self.spin_alpha.value()
        mach = self.spin_mach.value()
        ar = self.spin_ar.value()
        reynolds = self.spin_re.value()
        sweep = self.spin_sweep.value()
        ground = self.chk_ground.isChecked()
        height = self.spin_height.value()
        trip_up = self.spin_trip_up.value() or None
        trip_lo = self.spin_trip_lo.value() or None

        try:
            if ground:
                # Ground effect operates on a single body via the image method.
                sol = solve_ground_effect(self.panelled_bodies[0], alpha, height,
                                          mach=mach, aspect_ratio=ar, reynolds=reynolds)
            elif sweep > 0.0:
                # Busemann simple-sweep: report the swept (3D-referred) solution.
                swept = solve_swept_wing(self.panelled_bodies[0], alpha, sweep,
                                         mach=mach, reynolds=reynolds, aspect_ratio=ar)
                sol = swept.normal_solution
                sol.cl, sol.cd_ind = swept.cl_3d, swept.cd_3d
                sol.cm_qc = swept.cm_qc_3d
            else:
                sol = solve_flow(self.panelled_bodies, alpha, mach=mach,
                                 reynolds=reynolds, aspect_ratio=ar,
                                 trip_x_upper=trip_up, trip_x_lower=trip_lo,
                                 n_crit=self.spin_ncrit.value())
        except Exception as e:
            self.lbl_status.setText(f"Solver Error:\n{str(e)}")
            return
        if sol.reynolds > 0:
            tag = f"converged ({sol.iterations} it)" if sol.converged else "NOT converged"
            self.lbl_status.setText(f"Viscous: {tag}\nCl={sol.cl:.3f}  Cd={sol.cd_total:.5f}"
                                    + "".join(f"\n{w}" for w in sol.warnings))
        else:
            self.lbl_status.setText(f"Inviscid: Cl={sol.cl:.3f}")
        
        # --- Update Tab 1: Single Analysis (Cp Distribution) ---
        self.tabs.setCurrentIndex(0) 
        self.single_plot.fig.clear()
        
        ax1 = self.single_plot.fig.add_subplot(211)
        ax2 = self.single_plot.fig.add_subplot(212)
        
        # Draw all bodies
        for body in self.panelled_bodies:
            ax1.plot(body[:,0], body[:,1], 'k-')
            
        title = f"Pressure (α={alpha}°, M={mach}) | Global Cl: {sol.cl:.3f} | Total Cd: {sol.cd_total:.4f}"
        ax1.set_title(title)
        ax1.axis("equal")
        ax1.axis("off")
        
        # Extract global geometry mapping to plot Cp dynamically for N bodies
        _, _, _, _, mid_x, _, _, _, _, _, _, _, body_counts = compute_panel_geometry(self.panelled_bodies)
        
        idx_start = 0
        colors_up = ['b-', 'g-', 'm-']
        colors_low = ['r-', 'orange', 'c-']
        
        for k, n_p in enumerate(body_counts):
            m_x = mid_x[idx_start : idx_start + n_p]
            cp_vals = sol.cp_dist[idx_start : idx_start + n_p]
            le_idx = np.argmin(m_x) # Find local leading edge for this specific body
            
            label_up = 'Main Upper' if k == 0 else f'Flap {k} Upper'
            label_low = 'Main Lower' if k == 0 else f'Flap {k} Lower'
            c_up = colors_up[k % len(colors_up)]
            c_low = colors_low[k % len(colors_low)]
            
            ax2.plot(m_x[:le_idx+1], cp_vals[:le_idx+1], c_up, label=label_up)
            ax2.plot(m_x[le_idx:], cp_vals[le_idx:], c_low, label=label_low)
            
            idx_start += n_p
            
        ax2.invert_yaxis()
        ax2.set_xlabel("x/c")
        ax2.set_ylabel("Cp")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        self.single_plot.fig.tight_layout()
        self.single_plot.redraw()

        # --- Update Tab 3: Flowfield (Streamlines) ---
        self.flow_plot.fig.clear()
        ax_flow = self.flow_plot.fig.add_subplot(111)
        
        X, Y, U, V = compute_velocity_field(sol, grid_res=80)
        speed = np.sqrt(U**2 + V**2)
        
        strm = ax_flow.streamplot(X, Y, U, V, color=speed, cmap='jet', 
                                  linewidth=1.2, density=1.5, arrowsize=1.0)
        
        # Overlay the solved bodies (for ground effect: pitched and raised above the wall)
        for body in sol.coords_list:
            ax_flow.fill(body[:,0], body[:,1], color='gray', zorder=5)
            ax_flow.plot(body[:,0], body[:,1], color='black', linewidth=1.5, zorder=6)
        
        ax_flow.set_title(f"Streamlines & Velocity Field (α={alpha}°)")
        ax_flow.axis("equal")
        ax_flow.set_xlim(-0.5, 1.5)
        ax_flow.set_ylim(-0.5, 0.5)
        self.flow_plot.fig.colorbar(strm.lines, ax=ax_flow, label="Velocity / V_inf", fraction=0.046, pad=0.04)
        
        self.flow_plot.fig.tight_layout()
        self.flow_plot.redraw()

    def run_polar_analysis(self):
        self.tabs.setCurrentIndex(1)
        mach = self.spin_mach.value()
        ar = self.spin_ar.value()
        reynolds = self.spin_re.value()

        # The polar sweep runs on a single body (the main element).
        sweep_body = self.panelled_bodies[0] if self.panelled_bodies else None
        if sweep_body is None:
            self.lbl_status.setText("Load or generate an airfoil first.")
            return

        try:
            self.last_polar = run_alpha_sweep(
                sweep_body,
                self.spin_start.value(), 
                self.spin_end.value(), 
                self.spin_step.value(),
                mach=mach, reynolds=reynolds, aspect_ratio=ar,
                trip_x_upper=self.spin_trip_up.value() or None,
                trip_x_lower=self.spin_trip_lo.value() or None,
                n_crit=self.spin_ncrit.value(),
            )
        except Exception as e:
            self.lbl_status.setText(f"Sweep Error:\n{str(e)}")
            return
        
        self.btn_export_csv.setEnabled(True) 
        if reynolds > 0:
            sols = self.last_polar.solutions
            n_ok = sum(bool(s.converged) for s in sols)
            self.lbl_status.setText(f"Viscous polar: {n_ok} of {len(sols)} points converged")
        
        self.polar_plot.fig.clear()
        
        ax1 = self.polar_plot.fig.add_subplot(131)
        ax2 = self.polar_plot.fig.add_subplot(132)
        ax3 = self.polar_plot.fig.add_subplot(133)
        
        ax1.plot(self.last_polar.alphas, self.last_polar.cls, 'o-', color='navy', markersize=4)
        ax1.set_title("Lift Curve (Cl vs α)")
        ax1.set_xlabel("α (deg)")
        ax1.set_ylabel("Cl")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.last_polar.cds, self.last_polar.cls, '^-', color='forestgreen', markersize=4)
        ax2.set_title("Drag Polar (Cl vs Cd)")
        ax2.set_xlabel("Cd (Total Drag)")
        ax2.set_ylabel("Cl")
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.last_polar.alphas, self.last_polar.cm_qcs, 's-', color='darkred', markersize=4)
        ax3.set_title("Moment (Cm vs α)")
        ax3.set_xlabel("α (deg)")
        ax3.set_ylabel("Cm")
        ax3.grid(True, alpha=0.3)
        
        self.polar_plot.fig.tight_layout()
        self.polar_plot.redraw()

    def export_csv(self):
        if self.last_polar is None: return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Polar Data", "", "CSV Files (*.csv)")
        if filepath:
            try:
                export_polar_csv(filepath, self.last_polar)
                self.lbl_status.setText("Polar exported successfully.")
            except Exception as e:
                self.lbl_status.setText(f"Export Error:\n{str(e)}")

    def _enable_ui_buttons(self):
        self.btn_apply_flap.setEnabled(True)
        self.btn_add_slotted.setEnabled(True) # NEW
        self.btn_apply_mod.setEnabled(True)
        self.btn_run_single.setEnabled(True)
        self.btn_run_polar.setEnabled(True)
        
    def _plot_geometry(self):
        self.tabs.setCurrentIndex(0)
        self.single_plot.clear_plot()
        for body in self.panelled_bodies:
            self.single_plot.ax.plot(body[:,0], body[:,1], 'k-')
        self.single_plot.ax.axis("equal")
        self.single_plot.redraw()


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
