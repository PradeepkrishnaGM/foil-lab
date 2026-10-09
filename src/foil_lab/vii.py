"""Viscous-Inviscid Interaction (VII) via the transpiration-velocity method.

Couples the inviscid panel solver to the integral boundary layer so that the
boundary-layer displacement feeds back onto the inviscid flow. This predicts the
viscous pressure distribution and drag far better than the one-way (inviscid ->
BL -> drag) path, and captures the lift loss and drag rise approaching stall.

Method (transpiration / surface-blowing)
----------------------------------------
The displacement thickness delta* makes the body look thicker to the outer flow.
Rather than physically re-shaping the geometry each iteration, we model the
displacement with an equivalent *transpiration* (blowing) velocity injected at
the wall:

    v_n(s) = d/ds ( u_e(s) * delta*(s) )

This wall-normal source is added to the inviscid flow-tangency right-hand side.
The coupled fixed point is then iterated with under-relaxation:

    1. inviscid solve (with current transpiration) -> edge velocities u_e
    2. march the integral BL on each surface       -> delta*(s)
    3. recompute transpiration v_n from u_e, delta*
    4. blend (under-relax) and repeat until u_e converges

Scope and honesty
-----------------
* Converges for attached and mildly separated flow and clearly improves the
  pressure/drag over the one-way coupling.
* It is the *direct* (prescribed-edge-velocity) formulation, so it does NOT
  robustly handle deep separation -- near the Goldstein singularity the integral
  BL under prescribed u_e becomes ill-posed and the iteration will stall or
  refuse to converge. Capturing large separation bubbles needs the inverse /
  simultaneous-Newton formulation, which is the documented follow-on.
* Single body. Uses the inviscid core (:mod:`foil_lab.inviscid`), where the transpiration
  enters as known equivalent sources (Lighthill 1958), and the integral BL in ``bl_integral``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import inviscid as core
from .bl_integral import march_boundary_layer
from .corrections import karman_tsien_correction
from .solver import SolutionData, drag_is_plausible, geom_to_coords_list, loads_from_cp


@dataclass
class VIIResult:
    solution: SolutionData
    converged: bool
    iterations: int
    residual: float
    cd_wake: float          # Squire-Young profile drag from the BL at the TE
    delta_star: np.ndarray  # displacement thickness per panel
    transpiration: np.ndarray


def _inviscid_with_transpiration(ps, alpha_rad, v_n, v_inf=1.0, assembly=None):
    """Solve the core with a wall transpiration v_n (outward normal velocity per panel).

    v_n enters the core as a known source strength per panel, so the boundary condition is
    "outward normal velocity = v_n" (:func:`foil_lab.inviscid.solve`). Returns the nodal vortex
    strengths and the tangential edge velocity (solver direction) scaled by ``v_inf``.
    ``assembly`` (``core.assemble(ps, sources=True)``) avoids rebuilding the matrices.
    """
    sol = core.solve(ps, float(np.rad2deg(alpha_rad)), source_strength=v_n, assembly=assembly)
    return np.asarray(sol.gamma), np.asarray(sol.vt) * v_inf


def _split_surfaces(geom, vt):
    """Split panels into two surface branches at the stagnation point.

    Returns (idx_a, s_a, ue_a, idx_b, s_b, ue_b) where idx_* map branch stations
    back to global panel indices.
    """
    n = geom.num_panels
    guard = max(2, n // 20)
    stag = int(np.argmin(np.abs(vt[guard:n - guard]))) + guard

    idx_a = np.arange(stag, -1, -1)               # stag -> 0
    idx_b = np.arange(stag, n)                     # stag -> n-1
    lengths = geom.lengths

    def arc(idx):
        ds = lengths[idx]
        return np.concatenate([[0.0], np.cumsum(ds)])[:-1]

    s_a = arc(idx_a); ue_a = np.abs(vt[idx_a])
    s_b = arc(idx_b); ue_b = np.abs(vt[idx_b])
    return idx_a, s_a, ue_a, idx_b, s_b, ue_b, stag


def solve_vii(coords, alpha_deg, reynolds, v_inf=1.0, mach=0.0,
              max_iter=80, tol=1e-4, relax=0.05, trip_x=None,
              transition_model="en", n_crit=9.0,
              verbose=False) -> VIIResult:
    """Solve a single airfoil with transpiration VII coupling.

    Parameters
    ----------
    relax : under-relaxation factor for the transpiration update (small is
        stable but slow; 0.02-0.1 is typical for attached flow).
    tol : convergence tolerance on the max change in edge velocity between
        iterations.
    trip_x : optional explicit transition x/c (applied to both surfaces).
    """
    alpha_rad = np.deg2rad(alpha_deg)
    geom = core.PanelSet.from_bodies(coords)
    asm = core.assemble(geom, sources=True)
    n = geom.num_panels

    v_n = np.zeros(n)             # transpiration, starts at zero (inviscid)
    delta_star = np.zeros(n)
    ds_prev = None
    converged = False
    residual = np.inf
    prev_res = np.inf
    it = 0

    def _smooth(y, passes=2):
        """Light Laplacian smoothing to damp panel-scale noise in delta*."""
        out = y.copy()
        for _ in range(passes):
            out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        return out

    for it in range(1, max_iter + 1):
        strengths, vt = _inviscid_with_transpiration(geom, alpha_rad, v_n, v_inf, asm)
        v_n_solved = v_n.copy()     # the transpiration that produced (strengths, vt)

        idx_a, s_a, ue_a, idx_b, s_b, ue_b, stag = _split_surfaces(geom, vt)

        # Map a trip x/c to arc length on each branch if requested.
        def trip_s_for(idx, s_arr):
            if trip_x is None:
                return None
            xb = geom.mid_x[idx]
            j = int(np.argmin(np.abs(xb - trip_x)))
            return float(s_arr[j])

        prof_a = march_boundary_layer(s_a, ue_a, reynolds, trip_s_for(idx_a, s_a),
                                      transition_model=transition_model, n_crit=n_crit)
        prof_b = march_boundary_layer(s_b, ue_b, reynolds, trip_s_for(idx_b, s_b),
                                      transition_model=transition_model, n_crit=n_crit)

        # Assemble delta* back onto global panels (smoothed per branch).
        ds_new = np.zeros(n)
        ds_new[idx_a] = _smooth(prof_a.delta_star)
        ds_new[idx_b] = _smooth(prof_b.delta_star)

        # Transpiration v_n = d(ue * delta*)/ds on each branch, from the SMOOTHED
        # displacement, then clipped to a physical range. The blowing velocity
        # cannot realistically exceed a small fraction of the freestream; large
        # excursions are panel-scale numerical artifacts (esp. at the TE) and
        # destabilize the coupling, so we clamp them.
        v_n_new = np.zeros(n)
        for idx, s_arr, ue_arr, prof in [(idx_a, s_a, ue_a, prof_a),
                                         (idx_b, s_b, ue_b, prof_b)]:
            dstar_sm = _smooth(prof.delta_star)
            m = ue_arr * dstar_sm
            dvn = np.gradient(m, s_arr)
            v_n_new[idx] = dvn
        v_n_new = np.clip(v_n_new, -0.25, 0.25)

        # Under-relax both transpiration and delta* for stability.
        v_n = (1 - relax) * v_n + relax * v_n_new
        delta_star = (1 - relax) * delta_star + relax * ds_new

        # Convergence on the coupling variable delta* (more robust than raw u_e,
        # which spikes at the suction peak).
        if ds_prev is not None:
            denom = max(np.max(delta_star), 1e-6)
            residual = np.max(np.abs(delta_star - ds_prev)) / denom
            if residual < tol:
                converged = True
                if verbose:
                    print(f"  VII converged in {it} iters, res={residual:.2e}")
                break
        ds_prev = delta_star.copy()
        if verbose and it % 10 == 0:
            print(f"  iter {it}: res={residual:.2e}")

        # Adaptive relaxation: if the residual stalls (not dropping), back off
        # the relaxation factor to regain stability near the convergence limit.
        if it > 20 and residual > 0.5 * prev_res:
            relax = max(0.02, relax * 0.9)
        prev_res = residual

    # Final loads from the converged (viscous) pressure field.
    cp_inc = 1.0 - (vt / v_inf) ** 2
    cp = karman_tsien_correction(cp_inc, mach)
    cl, cm_le, cm_qc = loads_from_cp(geom, cp, alpha_deg)

    # Squire-Young profile drag from the TE momentum thickness & shape factor:
    # cd = 2 * theta_TE * (ue_TE)^((H_TE+5)/2), summed over both surfaces.
    def squire_young(prof, ue_arr):
        H_te = prof.H[-1]
        ue_te = max(abs(ue_arr[-1]), 0.3)
        return 2.0 * prof.theta[-1] * ue_te ** ((H_te + 5.0) / 2.0)

    cd_wake = squire_young(prof_a, ue_a) + squire_young(prof_b, ue_b)

    sol = SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds, aspect_ratio=0.0,
        cp_dist=cp, cl=cl, cd_ind=0.0, cd_visc=cd_wake,
        cm_le=cm_le, cm_qc=cm_qc,
        sources=v_n_solved, gamma=strengths,
        coords_list=geom_to_coords_list(coords), panels=geom,
    )
    converged = bool(converged and drag_is_plausible(cd_wake))
    return VIIResult(solution=sol, converged=converged, iterations=it,
                     residual=float(residual), cd_wake=float(cd_wake),
                     delta_star=delta_star, transpiration=v_n)
