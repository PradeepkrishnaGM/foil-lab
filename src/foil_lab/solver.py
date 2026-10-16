"""Solver entry point kept for the GUI, CLI, batch and design layers.

``solve_flow`` runs the inviscid core in :mod:`foil_lab.inviscid` (linear-strength vortex
panels, one assembly for every mode) and adds the existing compressibility and 3D layers. With a
Reynolds number it runs the coupled viscous-inviscid solver of :mod:`foil_lab.viscous` (M3): lift,
moment, surface pressure and drag then come from the coupled solution, and the result carries its
``converged`` flag, iterations, residual, path and warnings. ``SolutionData`` and
``compute_panel_geometry`` keep their shapes; ``SolutionData.panels`` carries the core's panel set
so off-body velocities can be rebuilt from the same singularity model.

Viscous analysis is for one body at Mach 0 in this version (DECISIONS.md M3, Q4 and C5); other
requests with a Reynolds number raise ``NotImplementedError`` instead of returning a result
without viscous drag. A closed (sharp) trailing edge is opened to ``ViscousParams.closed_te_gap``
for viscous analysis and the result's ``corrections`` say so (user, after report 11); a single
point whose direct start fails is reached by continuation (:func:`foil_lab.viscous.solve_single`,
its ``path`` says how).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from . import inviscid as core
from .corrections import apply_3d_corrections, karman_tsien_correction
from .params import VISCOUS, ViscousParams

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
    sources: np.ndarray            # per-panel source strength (viscous displacement), else zeros
    gamma: np.ndarray              # nodal vortex strengths of the core (all bodies)
    coords_list: List[np.ndarray]  # supports multiple bodies
    panels: Optional[core.PanelSet] = field(default=None, repr=False)
    #: viscous results only (``reynolds > 0``); an inviscid result keeps the defaults
    converged: bool = True         # inviscid: finite solution; viscous: the residual criteria were met
    cd_friction: float = float("nan")
    cd_pressure: float = float("nan")
    xtr_upper: float = float("nan")
    xtr_lower: float = float("nan")
    iterations: int = 0
    residual: float = float("nan")
    path: str = ""
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)  # e.g. the closed trailing edge opened
    viscous: object = field(default=None, repr=False)   # the ``viscous.ViscousResult``

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


def drag_is_plausible(cd: float) -> bool:
    """False for non-finite drag, Cd < 0 or Cd > 1: such a viscous result is never reported as
    converged (DECISIONS.md, M2 check 3)."""
    return bool(np.isfinite(cd) and 0.0 <= cd <= 1.0)


def loads_from_cp(ps: core.PanelSet, cp: np.ndarray, alpha_deg: float, body=None):
    """(cl, cm about the leading edge (0, 0), cm about the quarter chord (0.25, 0))."""
    cl, _, cm_le = core.pressure_loads(ps, cp, alpha_deg, body, ref_point=(0.0, 0.0))
    _, _, cm_qc = core.pressure_loads(ps, cp, alpha_deg, body, ref_point=(0.25, 0.0))
    return float(cl), float(cm_le), float(cm_qc)


def viscous_params(n_crit: Optional[float] = None, params: Optional[ViscousParams] = None
                   ) -> ViscousParams:
    """``params`` (default ``params.VISCOUS``) with ``n_crit`` replaced when it is given."""
    prm = VISCOUS if params is None else params
    return prm if n_crit is None else dataclasses.replace(prm, n_crit=float(n_crit))


def trips_dict(trip_x_upper: Optional[float] = None, trip_x_lower: Optional[float] = None,
               body: int = 0) -> Optional[dict]:
    """Forced transition locations (x of the body, x/c for a unit chord) in the form of
    :func:`foil_lab.viscous.solve_viscous`; ``None`` or 0 means free transition."""
    trips = {}
    if trip_x_upper:
        trips[(body, "upper")] = float(trip_x_upper)
    if trip_x_lower:
        trips[(body, "lower")] = float(trip_x_lower)
    return trips or None


def _check_viscous(num_bodies: int, mach: float) -> None:
    if num_bodies != 1:
        raise NotImplementedError("viscous analysis (a Reynolds number) of several bodies is added in M5; "
                                  "use reynolds=0 for an inviscid multi-element result")
    if mach > 0.0:
        raise NotImplementedError("viscous results are Mach 0 in this version (compressibility in M6); "
                                  "use mach=0 with a Reynolds number, or reynolds=0")


def solution_from_viscous(body: np.ndarray, vres, mach: float = 0.0, aspect_ratio: float = 0.0,
                          corrections: Sequence[str] = ()) -> SolutionData:
    """``SolutionData`` of one coupled viscous result (``viscous.ViscousResult``) on ``body`` (the
    geometry that was solved)."""
    ps = core.PanelSet.from_bodies([body])
    alpha = float(vres.alpha)
    cp = np.asarray(vres.cp) if vres.cp is not None else np.full(ps.num_panels, np.nan)
    _, cm_le, _ = loads_from_cp(ps, cp, alpha)
    cl, cd_ind, cm_qc = apply_3d_corrections(float(vres.cl), float(vres.cm), alpha, aspect_ratio, mach)
    return SolutionData(
        alpha_deg=alpha, mach=mach, reynolds=float(vres.reynolds), aspect_ratio=aspect_ratio,
        cp_dist=cp, cl=cl, cd_ind=cd_ind, cd_visc=float(vres.cd), cm_le=cm_le, cm_qc=cm_qc,
        sources=(np.zeros(ps.num_panels) if vres.sources is None else np.asarray(vres.sources)),
        gamma=(np.full(ps.num_nodes, np.nan) if vres.gamma is None else np.asarray(vres.gamma)),
        coords_list=[body], panels=ps, converged=bool(vres.converged), cd_friction=float(vres.cdf),
        cd_pressure=float(vres.cdp), xtr_upper=float(vres.xtr_upper), xtr_lower=float(vres.xtr_lower),
        iterations=int(vres.iterations), residual=vres.residual, path=vres.path,
        warnings=list(vres.warnings), corrections=list(corrections), viscous=vres)


def solve_flow(
    coords_input: CoordsLike,
    alpha_deg: float,
    v_inf: float = 1.0,
    mach: float = 0.0,
    reynolds: float = 0.0,
    aspect_ratio: float = 0.0,
    real_body_index: Optional[int] = 0,
    trip_x_upper: Optional[float] = None,
    trip_x_lower: Optional[float] = None,
    n_crit: Optional[float] = None,
    params: Optional[ViscousParams] = None,
) -> SolutionData:
    """Solve the flow over one or more bodies.

    ``reynolds`` 0: inviscid core (Kármán-Tsien on Cp, aspect-ratio correction). ``reynolds`` > 0:
    the coupled viscous solver (:func:`foil_lab.viscous.solve_single`: the direct start, else
    continuation; one body, Mach 0), with e^n transition (``n_crit``, default
    ``params.VISCOUS.n_crit``) and forced trips ``trip_x_upper`` / ``trip_x_lower`` (x/c; ``None``
    or 0 = free transition). A closed trailing edge is opened to ``params.closed_te_gap`` first
    (recorded in ``corrections``; 0 keeps it sharp).

    real_body_index : with more than one body, the body whose loads are reported (None: all
        bodies together).
    """
    bodies = geom_to_coords_list(coords_input)
    if reynolds > 0:
        from . import viscous
        _check_viscous(len(bodies), mach)
        prm = viscous_params(n_crit, params)
        body, corr = viscous.open_closed_te(bodies[0], prm)
        vres = viscous.solve_single(body, float(alpha_deg), float(reynolds), prm,
                                    trips_dict(trip_x_upper, trip_x_lower))
        return solution_from_viscous(body, vres, mach, aspect_ratio, corr)

    ps = core.PanelSet.from_bodies(bodies)
    sol = core.solve(ps, alpha_deg)

    cp_dist = karman_tsien_correction(np.asarray(sol.cp), mach)

    body_idx = real_body_index if ps.num_bodies > 1 else None
    cl0, cm_le, cm_qc0 = loads_from_cp(ps, cp_dist, alpha_deg, body_idx)

    cl, cd_ind, cm_qc = apply_3d_corrections(cl0, cm_qc0, alpha_deg, aspect_ratio, mach)

    return SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds, aspect_ratio=aspect_ratio,
        cp_dist=cp_dist, cl=cl, cd_ind=cd_ind, cd_visc=0.0,
        cm_le=cm_le, cm_qc=cm_qc,
        sources=np.zeros(ps.num_panels), gamma=np.asarray(sol.gamma),
        coords_list=bodies, panels=ps, converged=bool(np.all(np.isfinite(np.asarray(sol.cp)))),
    )


def solve_flow_sweep(coords: np.ndarray, alphas: Sequence[float], reynolds: float, mach: float = 0.0,
                     aspect_ratio: float = 0.0, trip_x_upper: Optional[float] = None,
                     trip_x_lower: Optional[float] = None, n_crit: Optional[float] = None,
                     params: Optional[ViscousParams] = None) -> List[SolutionData]:
    """Viscous results at ``alphas`` (in that order) for one body, reached with warm starts
    (:func:`foil_lab.viscous.sweep`); each result says how it was reached (``path``). A closed
    trailing edge is opened as in :func:`solve_flow`."""
    from . import viscous
    bodies = geom_to_coords_list(coords)
    _check_viscous(len(bodies), mach)
    prm = viscous_params(n_crit, params)
    body, corr = viscous.open_closed_te(bodies[0], prm)
    res = viscous.sweep(body, [float(a) for a in alphas], float(reynolds), prm,
                        trips_dict(trip_x_upper, trip_x_lower))
    return [solution_from_viscous(body, r, mach, aspect_ratio, corr) for r in res]
