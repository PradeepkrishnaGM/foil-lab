"""Solver entry point kept for the GUI, CLI, batch and design layers.

``solve_flow`` runs the inviscid core in :mod:`foil_lab.inviscid` (linear-strength vortex
panels, one assembly for every mode) and adds the existing compressibility, 3D, one-way viscous
drag and post-stall layers unchanged. ``SolutionData`` and ``compute_panel_geometry`` keep their
shapes so those layers need no changes; ``SolutionData.panels`` carries the core's panel set so
off-body velocities can be rebuilt from the same singularity model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from . import inviscid as core
from .boundary_layer import calculate_viscous_drag
from .corrections import apply_3d_corrections, karman_tsien_correction
from .viscous_mods import apply_post_stall, estimate_separation_from_cp

CoordsLike = Union[np.ndarray, Sequence[np.ndarray]]


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
    sources: np.ndarray            # per-panel known source strength (transpiration), else zeros
    gamma: np.ndarray              # nodal vortex strengths of the core (all bodies)
    coords_list: List[np.ndarray]  # supports multiple bodies
    panels: Optional[core.PanelSet] = field(default=None, repr=False)

    @property
    def cd_total(self) -> float:
        return self.cd_ind + self.cd_visc


def compute_panel_geometry(coords_list: CoordsLike):
    """Panel geometry as a 13-tuple, in the solver's (clockwise) panel order.

    (x1, y1, x2, y2, mid_x, mid_y, lengths, theta, nx, ny, tx, ty, body_counts)
    """
    g = core.PanelSet.from_bodies(geom_to_coords_list(coords_list))
    nx, ny = g.normal
    tx, ty = g.tangent
    counts = [sl.stop - sl.start for sl in g.body_panels]
    return (g.x1, g.y1, g.x2, g.y2, g.mid_x, g.mid_y, g.lengths, g.theta, nx, ny, tx, ty, counts)


def geom_to_coords_list(coords_input: CoordsLike) -> List[np.ndarray]:
    if isinstance(coords_input, np.ndarray):
        return [coords_input]
    return list(coords_input)


def loads_from_cp(ps: core.PanelSet, cp: np.ndarray, alpha_deg: float, body=None):
    """(cl, cm about the leading edge (0, 0), cm about the quarter chord (0.25, 0))."""
    cl, _, cm_le = core.pressure_loads(ps, cp, alpha_deg, body, ref_point=(0.0, 0.0))
    _, _, cm_qc = core.pressure_loads(ps, cp, alpha_deg, body, ref_point=(0.25, 0.0))
    return float(cl), float(cm_le), float(cm_qc)


def solve_flow(
    coords_input: CoordsLike,
    alpha_deg: float,
    v_inf: float = 1.0,
    mach: float = 0.0,
    reynolds: float = 0.0,
    aspect_ratio: float = 0.0,
    real_body_index: Optional[int] = 0,
    roughness: int = 0,
    enable_post_stall: bool = False,
    trip_x_upper: Optional[float] = None,
    trip_x_lower: Optional[float] = None,
) -> SolutionData:
    """Solve incompressible inviscid flow over one or more bodies with the core.

    real_body_index : with more than one body, the body whose loads are reported (None: all
        bodies together).
    roughness : surface roughness state 0-3, threaded into the viscous solve
        (earlier transition + higher drag).
    enable_post_stall : if True, apply empirical post-stall Cl/Cd corrections
        when the boundary layer reports significant separation (single body,
        Re > 0 only).
    """
    bodies = geom_to_coords_list(coords_input)
    ps = core.PanelSet.from_bodies(bodies)
    sol = core.solve(ps, alpha_deg)
    vt = np.asarray(sol.vt) * v_inf

    cp_dist = karman_tsien_correction(np.asarray(sol.cp), mach)

    body_idx = real_body_index if ps.num_bodies > 1 else None
    cl0, cm_le, cm_qc0 = loads_from_cp(ps, cp_dist, alpha_deg, body_idx)

    cl, cd_ind, cm_qc = apply_3d_corrections(cl0, cm_qc0, alpha_deg, aspect_ratio, mach)

    cd_visc = 0.0
    sep_fraction = 0.0
    if reynolds > 0 and ps.num_bodies == 1:
        cd_visc, sep_fraction = calculate_viscous_drag(
            ps.mid_x, ps.mid_y, ps.lengths, vt, reynolds,
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

    return SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds, aspect_ratio=aspect_ratio,
        cp_dist=cp_dist, cl=cl, cd_ind=cd_ind, cd_visc=cd_visc,
        cm_le=cm_le, cm_qc=cm_qc,
        sources=np.zeros(ps.num_panels), gamma=np.asarray(sol.gamma),
        coords_list=bodies, panels=ps,
    )
