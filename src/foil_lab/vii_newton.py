"""Simultaneous-Newton viscous-inviscid interaction (VII).

================================================================
EXPERIMENTAL — the simultaneous Newton coupling. This is the principled fix for
low-Reynolds convergence that the quasi-simultaneous hybrid (``vii_qs``) could
not achieve. Status is reported honestly by the solver (``converged`` flag) and
by tests; read the run notes / CHANGELOG before relying on it.
================================================================

Idea
----
The transpiration VII (``vii.solve_vii``) iterates inviscid <-> BL with
under-relaxation and stalls near separation. Here we instead solve the *coupled*
system in one Newton iteration.

The key enabling fact: the inviscid edge-velocity response to a wall
transpiration ``v_n`` is exactly affine (the panel system is linear):

    u_e = u_e0 + D @ v_n                         (D constant, built once)

The viscous law gives the transpiration required by the current edge velocity:

    v_n^BL = T(u_e)        (T = d/ds (u_e * delta*(u_e)), from the BL march)

A consistent (converged) solution satisfies the fixed point ``v_n = T(u_e0 + D
v_n)``. Define the residual

    R(v_n) = v_n - T(u_e0 + D @ v_n)

and solve R = 0 by Newton's method:

    J = I - (dT/du_e) @ D
    v_n <- v_n - J^{-1} R

``dT/du_e`` (how the required transpiration changes with edge velocity) is
obtained by finite differences of the BL march -- cheap and local. ``D`` is the
exact inviscid sensitivity. Inverting the full ``J`` (rather than relaxing a
diagonal) is what lets the scheme pass through separation, where the
quasi-simultaneous hybrid limit-cycled.

Honesty
-------
Whether this fully converges at Re=50k is reported by the ``converged`` flag and
the validation tests, not assumed. The Newton machinery is first validated at a
condition where VII already works (moderate Re), then pushed to low Re.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import inviscid as core
from .bl_integral import march_boundary_layer
from .vii import (_inviscid_with_transpiration, _split_surfaces,
                  VIIResult)


def _build_sensitivity_matrix(geom, alpha_rad, v_inf=1.0, assembly=None):
    """Exact constant matrix D mapping transpiration v_n -> edge velocity vt.

    Because the panel system is linear, vt = vt0 + D @ v_n exactly. Built by
    n one-time linear solves.
    """
    n = geom.num_panels
    if assembly is None:
        assembly = core.assemble(geom, sources=True)
    _, vt0 = _inviscid_with_transpiration(geom, alpha_rad, np.zeros(n), v_inf, assembly)
    D = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        _, vt_j = _inviscid_with_transpiration(geom, alpha_rad, e, v_inf, assembly)
        D[:, j] = vt_j - vt0
    return vt0, D


def _transpiration_from_ue(geom, vt, reynolds, n_crit, trip_x, smooth_passes=2):
    """The viscous law T(u_e): edge velocity -> required transpiration v_n.

    Marches the BL on both surfaces and returns v_n = d(u_e*delta*)/ds mapped to
    panels, plus the per-surface profiles (for drag afterwards).
    """
    n = geom.num_panels
    idx_a, s_a, ue_a, idx_b, s_b, ue_b, stag = _split_surfaces(geom, vt)

    def trip_s_for(idx, s_arr):
        if trip_x is None:
            return None
        xb = geom.mid_x[idx]
        j = int(np.argmin(np.abs(xb - trip_x)))
        return float(s_arr[j])

    def smooth(y):
        out = y.copy()
        for _ in range(smooth_passes):
            out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        return out

    prof_a = march_boundary_layer(s_a, ue_a, reynolds, trip_s_for(idx_a, s_a),
                                  n_crit=n_crit)
    prof_b = march_boundary_layer(s_b, ue_b, reynolds, trip_s_for(idx_b, s_b),
                                  n_crit=n_crit)

    v_n = np.zeros(n)
    for idx, s_arr, ue_arr, prof in [(idx_a, s_a, ue_a, prof_a),
                                     (idx_b, s_b, ue_b, prof_b)]:
        m = ue_arr * smooth(prof.delta_star)
        v_n[idx] = np.gradient(m, s_arr)
    v_n = np.clip(v_n, -0.25, 0.25)
    return v_n, (idx_a, s_a, ue_a, prof_a, idx_b, s_b, ue_b, prof_b)


def solve_vii_newton(coords, alpha_deg, reynolds, v_inf=1.0, mach=0.0,
                     max_iter=40, tol=1e-4, trip_x=None, n_crit=9.0,
                     fd_eps=1e-4, damping=1.0, verbose=False):
    """Simultaneous-Newton VII solve. Returns a VIIResult.

    Parameters
    ----------
    fd_eps : finite-difference step for the viscous Jacobian dT/du_e.
    damping : Newton step damping (1.0 = full step; <1 for robustness).
    """
    import warnings
    warnings.warn(
        "solve_vii_newton is EXPERIMENTAL. It converges at moderate Re and, with "
        "the laminar-bubble closure, at low Re for low-to-moderate alpha; higher "
        "low-Re angles may not converge (check the `converged` flag). Drag at low "
        "Re is approximate. Use vii.solve_vii for validated high-Re analysis.",
        stacklevel=2,
    )
    from .corrections import karman_tsien_correction
    from .solver import SolutionData, geom_to_coords_list, loads_from_cp

    alpha_rad = np.deg2rad(alpha_deg)
    geom = core.PanelSet.from_bodies(coords)
    asm = core.assemble(geom, sources=True)
    n = geom.num_panels

    # Exact constant inviscid sensitivity.
    vt0, D = _build_sensitivity_matrix(geom, alpha_rad, v_inf, asm)

    def ue_of_vn(v_n):
        return vt0 + D @ v_n

    def residual(v_n):
        vt = ue_of_vn(v_n)
        v_n_bl, aux = _transpiration_from_ue(geom, vt, reynolds, n_crit, trip_x)
        return v_n - v_n_bl, aux

    v_n = np.zeros(n)
    converged = False
    res_norm = np.inf

    for it in range(1, max_iter + 1):
        R, aux = residual(v_n)
        res_norm = np.max(np.abs(R))
        if verbose and (it % 5 == 0 or it == 1):
            print(f"  Newton iter {it}: |R|={res_norm:.2e}")
        if res_norm < tol:
            converged = True
            if verbose:
                print(f"  Newton converged in {it} iters, |R|={res_norm:.2e}")
            break

        # Viscous Jacobian dT/d(v_n) via finite differences of T(u_e(v_n)).
        # We need d(v_n_bl)/d(v_n) = (dT/du_e) @ D. Rather than form dT/du_e
        # separately, perturb v_n directly: column k = (T(v_n+eps e_k) - T)/eps.
        # That is exactly d(v_n_bl)/d(v_n). Then J = I - that.
        dVbl = np.zeros((n, n))
        v_n_bl0, _ = _transpiration_from_ue(geom, ue_of_vn(v_n), reynolds,
                                            n_crit, trip_x)
        # Finite-difference column by column would be n BL marches per Newton
        # iter; do it but reuse the same marches cheaply.
        for k in range(n):
            vp = v_n.copy(); vp[k] += fd_eps
            v_n_bl_k, _ = _transpiration_from_ue(geom, ue_of_vn(vp), reynolds,
                                                 n_crit, trip_x)
            dVbl[:, k] = (v_n_bl_k - v_n_bl0) / fd_eps

        J = np.eye(n) - dVbl
        try:
            step = np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, R, rcond=None)[0]

        # Damped line search: pick the fraction in (0,1] that most reduces the
        # residual norm. The coupled Jacobian is near-singular (transpiration has
        # a weak null space) and the BL transition point makes the finite-diff
        # Jacobian noisy, so a full Newton step can overshoot; the line search
        # restores robustness without hand-tuned damping.
        best_v, best_norm = v_n - step, None
        for frac in (1.0, 0.5, 0.3, 0.15, 0.07, 0.03):
            trial = v_n - frac * damping * step
            R_trial, _ = residual(trial)
            nrm = np.max(np.abs(R_trial))
            if best_norm is None or nrm < best_norm:
                best_norm, best_v = nrm, trial
        v_n = best_v

    # Final state.
    vt = ue_of_vn(v_n)
    _, aux = residual(v_n)
    (idx_a, s_a, ue_a, prof_a, idx_b, s_b, ue_b, prof_b) = aux

    cp_inc = 1.0 - (vt / v_inf) ** 2
    cp = karman_tsien_correction(cp_inc, mach)
    cl, cm_le, cm_qc = loads_from_cp(geom, cp, alpha_deg)

    def squire_young(prof, ue_arr):
        H_te = prof.H[-1]
        ue_te = max(abs(ue_arr[-1]), 0.3)
        return 2.0 * prof.theta[-1] * ue_te ** ((H_te + 5.0) / 2.0)

    cd_wake = squire_young(prof_a, ue_a) + squire_young(prof_b, ue_b)
    delta_star = np.zeros(n)
    delta_star[idx_a] = prof_a.delta_star
    delta_star[idx_b] = prof_b.delta_star

    sol = SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds, aspect_ratio=0.0,
        cp_dist=cp, cl=cl, cd_ind=0.0, cd_visc=cd_wake,
        cm_le=cm_le, cm_qc=cm_qc,
        sources=np.asarray(v_n, dtype=float).copy(),
        gamma=_inviscid_with_transpiration(geom, alpha_rad, v_n, v_inf, asm)[0],
        coords_list=geom_to_coords_list(coords), panels=geom)
    return VIIResult(solution=sol, converged=converged, iterations=it,
                     residual=float(res_norm), cd_wake=float(cd_wake),
                     delta_star=delta_star, transpiration=v_n)
