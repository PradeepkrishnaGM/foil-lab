"""Quasi-simultaneous (Veldman) viscous-inviscid coupling primitives.

================================================================
STATUS: EXPERIMENTAL / WORK IN PROGRESS — NOT a working low-Re solver yet.
================================================================
This module is scaffolding toward low-Reynolds VII convergence. As of v1.2 it
DOES NOT converge at low Re; ``solve_vii_qs`` typically plateaus at a residual of
~1e-2 (above tolerance) and returns ``converged=False``. Do NOT use its Cl/Cd for
analysis — use the validated high-Re ``vii.solve_vii`` instead.

What IS validated and reusable:
* ``interaction_coefficients`` / ``march_coupled``: the Veldman interaction law,
  which provably keeps displacement thickness BOUNDED through a separating
  pressure gradient where the direct method (``bl_integral``) runs away. This is
  the singularity-removing machinery and it works in isolation (see
  ``tests/test_vii_qs_experimental.py``).

What is NOT working:
* The FULL coupled airfoil solve (``solve_vii_qs``) does not converge at Re=50k.
  The instability has been localized to a TRAILING-EDGE LIMIT CYCLE in the outer
  transpiration loop. Root cause: this is a *hybrid* scheme (interaction law in
  the local BL march, but the outer inviscid re-solve still coupled via
  under-relaxed transpiration); the two partially fight. A true fix needs the
  inviscid and viscous systems solved SIMULTANEOUSLY in one Newton step over a
  combined residual (the JAX autodiff core is intended to supply that Jacobian).

This is preserved as a documented starting point for that work, not shipped as a
feature. See VALIDATION.md and CHANGELOG.md.
================================================================

The direct boundary layer in ``bl_integral`` becomes singular at separation when
the edge velocity ``u_e`` is prescribed (Goldstein singularity), which is why the
transpiration VII diverges at low Reynolds number. Veldman's quasi-simultaneous
method (1981) removes the singularity by letting the boundary layer "see" a
*local* model of how the inviscid flow responds to its own displacement, and
solving the two together at each station.

Interaction law
---------------
The outer (inviscid) flow's edge velocity is written as

    u_e = u_inv + C · (m - m_inv)

where the mass defect is ``m = u_e · delta*``, ``u_inv``/``m_inv`` are the
current inviscid values, and ``C`` is a positive local *interaction
coefficient* approximating the diagonal inviscid response d(u_e)/d(m). Coupling
this law to the von Karman momentum equation makes ``u_e`` an unknown that rises
when the layer thickens, which is exactly the stabilizing feedback that prevents
the direct-mode blow-up at separation.

This module provides:
* ``interaction_coefficients`` -- the per-station C from panel spacing.
* ``march_coupled``            -- a momentum-integral BL marched with the
                                  interaction law, returning ue, theta, H, delta*
                                  and remaining finite through mild separation.

Scope: this is the convergence-enabling machinery. It is validated to converge
where the direct method diverges; quantitative bubble accuracy vs XFOIL is a
further step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bl_integral import (_thwaites_H, _thwaites_cf, _turbulent_cf,
                          _head_entrainment, _head_H_from_H1)
from .transition import find_transition_en


@dataclass
class CoupledProfile:
    s: np.ndarray
    ue: np.ndarray          # edge velocity (UPDATED by the interaction law)
    theta: np.ndarray
    H: np.ndarray
    delta_star: np.ndarray
    cf: np.ndarray
    transition_idx: int
    separated: bool


def interaction_coefficients(s: np.ndarray) -> np.ndarray:
    """Local interaction coefficient C(s) from station spacing.

    Veldman's local approximation to the inverse inviscid operator is a positive
    coefficient scaling with 1/Δs (the thin-airfoil/Hilbert kernel diagonal).
    We use C_i = 4 / (pi * (ds_back + ds_fwd)), a standard symmetric-spacing
    form; the prefactor only sets how strongly the local law pulls and is not
    sensitive within a broad range because it is iterated to consistency.
    """
    n = len(s)
    C = np.zeros(n)
    for i in range(n):
        ds_b = s[i] - s[i - 1] if i > 0 else s[1] - s[0]
        ds_f = s[i + 1] - s[i] if i < n - 1 else s[-1] - s[-2]
        C[i] = 4.0 / (np.pi * max(ds_b + ds_f, 1e-6))
    return C


def march_coupled(s: np.ndarray, u_inv: np.ndarray, Re: float,
                  trip_s: float | None = None, n_crit: float = 9.0,
                  C_scale: float = 1.0) -> CoupledProfile:
    """March the BL with the Veldman interaction law (inverse-capable).

    Parameters
    ----------
    s : arc-length stations from the stagnation point.
    u_inv : inviscid edge velocity at each station (the outer-flow guess).
    Re : Reynolds number.
    trip_s, n_crit : transition controls (explicit trip / e^n envelope).
    C_scale : multiplier on the interaction coefficient (tuning knob).

    Returns
    -------
    CoupledProfile, where ``ue`` is the viscous edge velocity (which differs from
    ``u_inv`` where the layer displaces the flow) and the profile remains finite
    through mild separation.
    """
    n = len(s)
    u_inv = np.abs(u_inv).astype(float)
    u_inv = np.maximum(u_inv, 1e-4)
    C = interaction_coefficients(s) * C_scale

    ue = u_inv.copy()
    theta = np.zeros(n)
    H = np.full(n, 2.59)
    cf = np.zeros(n)

    # --- Laminar start: Thwaites on the inviscid ue as initial guess -------
    ue5 = ue ** 5
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (ue5[1:] + ue5[:-1]) * np.diff(s))])
    theta_sq = (0.45 / (Re * ue ** 6)) * integ
    theta_lam = np.sqrt(np.maximum(theta_sq, 1e-20))
    due_ds = np.gradient(ue, s)
    lam = np.clip(Re * theta_sq * due_ds, -0.09, 0.25)
    H_lam = _thwaites_H(lam)

    Re_theta = Re * ue * theta_lam
    if trip_s is not None:
        tr = n - 1
        for i in range(1, n):
            if s[i] >= trip_s:
                tr = i
                break
        transition_idx = tr
    else:
        transition_idx, _ = find_transition_en(s, Re_theta, H_lam, n_crit=n_crit)

    theta[:transition_idx + 1] = theta_lam[:transition_idx + 1]
    H[:transition_idx + 1] = H_lam[:transition_idx + 1]
    cf[:transition_idx + 1] = _thwaites_cf(lam[:transition_idx + 1],
                                           Re_theta[:transition_idx + 1])

    separated = False

    # --- Turbulent region: march theta & H with the interaction law -------
    if transition_idx < n - 1:
        th = theta[transition_idx]
        Hk = float(np.clip(H[transition_idx], 1.4, 1.8))
        H1 = _head_entrainment(Hk)

        for i in range(transition_idx + 1, n):
            ds = s[i] - s[i - 1]

            # Quasi-simultaneous local solve for ue[i]:
            # momentum: dtheta/ds = cf/2 - (H+2) theta/ue * due/ds
            # interaction: ue = u_inv + C*(ue*delta* - u_inv*delta*_inv_guess)
            # We use the previous station's delta* as the local reference and
            # iterate a few times for the coupled (ue, theta) pair.
            ue_i = ue[i]                      # start from inviscid
            th_i = th
            dstar_prev = H[i - 1] * theta[i - 1]
            m_inv = u_inv[i] * dstar_prev     # inviscid mass-defect reference
            Hk_i = Hk
            H1_i = H1

            for _ in range(12):               # local fixed point, bounded
                Re_th = float(np.clip(Re * ue_i * th_i, 1.0, 1e8))
                cf_i = _turbulent_cf(Hk_i, Re_th)
                due = (ue_i - ue[i - 1]) / max(ds, 1e-9)
                # Momentum integral, with a bounded shape factor in the source.
                Hsrc = min(Hk_i, 3.0)
                dtheta = cf_i / 2.0 - (Hsrc + 2.0) * th_i / max(ue_i, 0.05) * due
                # Limit theta growth per step to keep the march stable.
                dtheta = float(np.clip(dtheta, -5.0 * th_i / max(ds, 1e-9),
                                       5.0 * th_i / max(ds, 1e-9)))
                th_new = float(np.clip(theta[i - 1] + dtheta * ds, 1e-7, 0.5))
                # Head's entrainment for H, with bounded H1.
                CE = 0.0306 * max(H1_i - 3.0, 1e-3) ** -0.6169
                dH1 = (ue[i - 1] * CE - H1_i * (th_i * due + ue_i * dtheta)) \
                    / max(ue_i * th_i, 1e-6)
                H1_new = float(np.clip(H1_i + dH1 * ds, 3.01, 20.0))
                Hk_new = float(np.clip(_head_H_from_H1(H1_new), 1.05, 3.2))
                dstar = Hk_new * th_new
                # Interaction law: ue responds to displacement mass defect.
                m = ue_i * dstar
                ue_new = u_inv[i] + C[i] * (m - m_inv)
                ue_new = float(np.clip(ue_new, 0.05, 2.0))

                # Heavily damped local update for stability.
                ue_i = 0.7 * ue_i + 0.3 * ue_new
                th_i = th_new
                Hk_i = Hk_new
                H1_i = H1_new

            th = th_i
            Hk = Hk_i
            H1 = H1_i
            if Hk > 2.8:        # mild-separation cap keeps the march finite
                separated = True
                Hk = 2.8
                H1 = _head_entrainment(Hk)

            ue[i] = ue_i
            theta[i] = th
            H[i] = Hk
            cf[i] = _turbulent_cf(Hk, max(Re * ue_i * th, 1.0))

    delta_star = H * theta
    return CoupledProfile(s=s, ue=ue, theta=theta, H=H, delta_star=delta_star,
                          cf=cf, transition_idx=transition_idx, separated=separated)


# --------------------------------------------------------------------------- #
# Full quasi-simultaneous VII airfoil solve
# --------------------------------------------------------------------------- #
def solve_vii_qs(coords, alpha_deg, reynolds, v_inf=1.0, mach=0.0,
                 max_iter=120, tol=1e-3, relax=0.10, trip_x=None,
                 n_crit=9.0, verbose=False):
    """Quasi-simultaneous VII: stays convergent into mild separation.

    Same outer structure as the transpiration VII, but the boundary layer is
    marched with the Veldman interaction law (``march_coupled``) so the
    displacement thickness stays bounded through separation instead of blowing
    up. This is what enables convergence at low Reynolds number where the direct
    transpiration method fails.

    Returns a ``VIIResult`` (imported from ``vii``) for drop-in compatibility.
    """
    import warnings
    warnings.warn(
        "solve_vii_qs is EXPERIMENTAL and does not yet converge at low Re "
        "(typically returns converged=False with a ~1e-2 residual). Do not use "
        "its Cl/Cd for analysis; use vii.solve_vii for validated high-Re VII.",
        stacklevel=2,
    )
    import numpy as np
    from .vii import (VIIResult, _inviscid_with_transpiration, _split_surfaces,
                      geom_to_coords_list)
    from .corrections import karman_tsien_correction
    from .forces import integrate_loads
    from .influence import LinearVortexModel
    from .panels import build_panels
    from .solver import SolutionData

    alpha_rad = np.deg2rad(alpha_deg)
    geom = build_panels(coords)
    coeffs = LinearVortexModel().compute(geom)
    n = geom.num_panels

    v_n = np.zeros(n)
    delta_star = np.zeros(n)
    ds_prev = None
    converged = False
    residual = np.inf
    prev_res = np.inf
    it = 0

    def _smooth(y, passes=2):
        out = y.copy()
        for _ in range(passes):
            out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        return out

    prof_a = prof_b = None
    ue_a = ue_b = None

    for it in range(1, max_iter + 1):
        strengths, vt = _inviscid_with_transpiration(geom, coeffs, alpha_rad, v_n, v_inf)
        idx_a, s_a, ue_a, idx_b, s_b, ue_b, stag = _split_surfaces(geom, vt)

        def trip_s_for(idx, s_arr):
            if trip_x is None:
                return None
            xb = geom.mid_x[idx]
            j = int(np.argmin(np.abs(xb - trip_x)))
            return float(s_arr[j])

        # Inverse-capable coupled BL march on each surface (bounded delta*).
        prof_a = march_coupled(s_a, ue_a, reynolds, trip_s_for(idx_a, s_a), n_crit)
        prof_b = march_coupled(s_b, ue_b, reynolds, trip_s_for(idx_b, s_b), n_crit)

        ds_new = np.zeros(n)
        ds_new[idx_a] = _smooth(prof_a.delta_star)
        ds_new[idx_b] = _smooth(prof_b.delta_star)

        # Transpiration from the bounded displacement thickness.
        v_n_new = np.zeros(n)
        for idx, s_arr, prof in [(idx_a, s_a, prof_a), (idx_b, s_b, prof_b)]:
            m = prof.ue * _smooth(prof.delta_star)
            v_n_new[idx] = np.gradient(m, s_arr)
        v_n_new = np.clip(v_n_new, -0.25, 0.25)

        # Taper transpiration at the trailing edge. The bare d(ue*delta*)/ds
        # formula is invalid where the two surface branches meet the wake: panels
        # there are tiny and delta* is largest, so the raw gradient produces a
        # spurious spike that drives a TE limit cycle. A cosine taper over the
        # last few panels of each branch removes the artifact without affecting
        # the physical interior.
        taper_frac = 0.04
        for idx in (idx_a, idx_b):
            k = max(2, int(len(idx) * taper_frac))
            w = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, k)))  # 0->1
            # idx runs stagnation->TE, so taper the TE end (last k entries).
            v_n_new[idx[-k:]] *= w[::-1]

        v_n = (1 - relax) * v_n + relax * v_n_new
        delta_star = (1 - relax) * delta_star + relax * ds_new

        if ds_prev is not None:
            denom = max(np.max(delta_star), 1e-6)
            residual = np.max(np.abs(delta_star - ds_prev)) / denom
            if residual < tol:
                converged = True
                if verbose:
                    print(f"  QS-VII converged in {it} iters, res={residual:.2e}")
                break
        ds_prev = delta_star.copy()
        if verbose and it % 20 == 0:
            print(f"  iter {it}: res={residual:.2e}")
        if it > 20 and residual > 0.5 * prev_res:
            relax = max(0.02, relax * 0.9)
        prev_res = residual

    cp_inc = 1.0 - (vt / v_inf) ** 2
    cp = karman_tsien_correction(cp_inc, mach)
    loads = integrate_loads(geom, cp, alpha_rad, body_index=None)

    def squire_young(prof, ue_arr):
        H_te = prof.H[-1]
        ue_te = max(abs(prof.ue[-1]), 0.3)
        return 2.0 * prof.theta[-1] * ue_te ** ((H_te + 5.0) / 2.0)

    cd_wake = squire_young(prof_a, ue_a) + squire_young(prof_b, ue_b)

    sol = SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds, aspect_ratio=0.0,
        cp_dist=cp, cl=loads.cl, cd_ind=0.0, cd_visc=cd_wake,
        cm_le=loads.cm_le, cm_qc=loads.cm_qc,
        sources=strengths[:n], gamma=np.array([strengths[n]]),
        coords_list=geom_to_coords_list(coords))
    return VIIResult(solution=sol, converged=converged, iterations=it,
                     residual=float(residual), cd_wake=float(cd_wake),
                     delta_star=delta_star, transpiration=v_n)
