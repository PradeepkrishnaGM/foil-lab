"""Solver orchestrator: wire geometry -> influence -> linear solve -> loads.

The heavy lifting lives in :mod:`panels`, :mod:`influence`,
:mod:`linear_system`, and :mod:`forces`. This module stays thin so the pipeline
is easy to read and each stage is independently testable and swappable.

Backward compatibility: ``solve_flow``, ``SolutionData``, and
``compute_panel_geometry`` keep their original signatures/return shapes so the
GUI, plotting, and analysis layers need no changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .boundary_layer import calculate_viscous_drag
from .corrections import apply_3d_corrections, karman_tsien_correction
from .forces import integrate_loads
from .influence import DEFAULT_MODEL, InfluenceModel
from .linear_system import assemble_and_solve
from .panels import CoordsLike, PanelGeometry, build_panels
from .viscous_mods import apply_post_stall, estimate_separation_from_cp


@dataclass
class SolutionData:
    alpha_deg: float
    mach: float
    reynolds: float
    aspect_ratio: float
    cp_dist: np.ndarray
    cl: float
    cd_ind: float
    cd_visc: float
    cm_le: float
    cm_qc: float
    sources: np.ndarray            # per-panel source strengths
    gamma: np.ndarray              # shared vortex strength(s), one entry
    coords_list: List[np.ndarray]  # supports multiple bodies

    @property
    def cd_total(self) -> float:
        return self.cd_ind + self.cd_visc


def compute_panel_geometry(coords_list: CoordsLike):
    """Backward-compatible 13-tuple view of :func:`panels.build_panels`."""
    g: PanelGeometry = build_panels(coords_list)
    return (g.x1, g.y1, g.x2, g.y2, g.mid_x, g.mid_y, g.lengths,
            g.theta, g.nx, g.ny, g.tx, g.ty, g.body_counts)


def geom_to_coords_list(coords_input: CoordsLike) -> List[np.ndarray]:
    if isinstance(coords_input, np.ndarray):
        return [coords_input]
    return list(coords_input)


def solve_flow(
    coords_input: CoordsLike,
    alpha_deg: float,
    v_inf: float = 1.0,
    mach: float = 0.0,
    reynolds: float = 0.0,
    aspect_ratio: float = 0.0,
    influence_model: Optional[InfluenceModel] = None,
    real_body_index: Optional[int] = 0,
    roughness: int = 0,
    enable_post_stall: bool = False,
    trip_x_upper: Optional[float] = None,
    trip_x_lower: Optional[float] = None,
) -> SolutionData:
    """Solve incompressible inviscid flow over one or more bodies.

    influence_model : swap the singularity kernel (defaults to Hess--Smith).
    real_body_index : which body carries the reported loads (body 0 normally;
        for a Method-of-Images ground-effect run the mirror body is appended and
        loads are integrated over the real body only).
    roughness : surface roughness state 0-3, threaded into the viscous solve
        (earlier transition + higher drag).
    enable_post_stall : if True, apply empirical post-stall Cl/Cd corrections
        when the boundary layer reports significant separation (single body,
        Re > 0 only).
    """
    model = influence_model or DEFAULT_MODEL
    alpha_rad = np.deg2rad(alpha_deg)

    geom = build_panels(coords_input)
    coeffs = model.compute(geom)
    flow = assemble_and_solve(geom, coeffs, alpha_rad, v_inf)

    cp_inc = 1.0 - (flow.vt_total / v_inf) ** 2
    cp_dist = karman_tsien_correction(cp_inc, mach)

    body_idx = real_body_index if geom.num_bodies > 1 else None
    loads = integrate_loads(geom, cp_dist, alpha_rad, body_index=body_idx)

    cl, cd_ind, cm_qc = apply_3d_corrections(
        loads.cl, loads.cm_qc, alpha_deg, aspect_ratio, mach
    )

    cd_visc = 0.0
    sep_fraction = 0.0
    if reynolds > 0 and geom.num_bodies == 1:
        cd_visc, sep_fraction = calculate_viscous_drag(
            geom.mid_x, geom.mid_y, geom.lengths, flow.vt_total, reynolds,
            roughness=roughness, return_details=True,
            trip_x_upper=trip_x_upper, trip_x_lower=trip_x_lower,
        )

    # Empirical post-stall knock-down, keyed to the inviscid suction peak
    # (a more robust stall proxy than the integral BL separation point).
    if enable_post_stall:
        cp_min = float(cp_dist.min())
        stall_frac = estimate_separation_from_cp(cp_min)
        if stall_frac > 0.0:
            result = apply_post_stall(cl, cd_ind + cd_visc, stall_frac,
                                      sep_threshold=0.0)
            cl = result.cl
            cd_visc = result.cd - cd_ind

    sources = flow.sources
    gamma = flow.gammas

    return SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds, aspect_ratio=aspect_ratio,
        cp_dist=cp_dist, cl=cl, cd_ind=cd_ind, cd_visc=cd_visc,
        cm_le=loads.cm_le, cm_qc=cm_qc,
        sources=sources, gamma=gamma,
        coords_list=geom_to_coords_list(coords_input),
    )
