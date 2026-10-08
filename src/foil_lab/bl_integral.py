"""Integral boundary-layer marching for viscous-inviscid interaction.

This is a *self-contained* integral BL solver that returns the full streamwise
distributions of momentum thickness theta(s), shape factor H(s), and
displacement thickness delta*(s) = H * theta along one surface. The existing
``boundary_layer.py`` only returns a single drag number; VII needs the whole
delta*(s) profile to build the transpiration boundary condition, so we keep this
separate rather than disturbing the validated drag path.

Method
------
* Laminar region: Thwaites' one-equation method (momentum thickness from an
  integral of the edge velocity), with H from Thwaites' correlation.
* Transition: Michel's criterion (or an explicit trip), same as the drag path,
  so behaviour is consistent.
* Turbulent region: Head's entrainment method -- march theta and H downstream
  using the entrainment closure (Green's correlations for the entrainment
  coefficient and skin friction). This is the standard integral turbulent
  closure used in coupled panel/BL codes.

All references are Cebeci & Bradshaw / Drela-style integral closures. The solver
returns arrays aligned with the input arc-length stations so the caller can map
delta* back onto panels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transition import find_transition_en


@dataclass
class BLProfile:
    s: np.ndarray            # arc length stations
    theta: np.ndarray        # momentum thickness
    H: np.ndarray            # shape factor delta*/theta
    delta_star: np.ndarray   # displacement thickness
    cf: np.ndarray           # skin friction coefficient
    transition_idx: int
    separated: bool


def _thwaites_H(lam: np.ndarray) -> np.ndarray:
    """Thwaites correlation for shape factor H from the pressure-gradient
    parameter lambda (Cebeci & Bradshaw piecewise fit)."""
    H = np.where(
        lam < 0.0,
        2.088 + 0.0731 / (lam + 0.14),
        2.61 - 3.75 * lam + 5.24 * lam ** 2,
    )
    return np.clip(H, 2.0, 4.0)


def _thwaites_cf(lam: np.ndarray, Re_theta: np.ndarray) -> np.ndarray:
    """Thwaites laminar skin friction from shear correlation."""
    ell = np.where(
        lam < 0.0,
        0.22 + 1.402 * lam + (0.018 * lam) / (lam + 0.107),
        0.22 + 1.57 * lam - 1.8 * lam ** 2,
    )
    return 2.0 * ell / np.maximum(Re_theta, 1e-6)


def _head_entrainment(H: float) -> float:
    """Head's entrainment shape parameter H1 as a function of H."""
    if H <= 1.6:
        return 3.3 + 0.8234 * (H - 1.1) ** -1.287
    return 3.3 + 1.5501 * (H - 0.6778) ** -3.064


def _head_H_from_H1(H1: float) -> float:
    """Invert Head's H1(H) correlation to get H from H1 (for marching)."""
    # Solve numerically; H1 is monotincreasing in 1/H region. Use bracketed search.
    if H1 <= 3.3:
        return 3.0
    # Two correlation branches; pick the physical root by bracketed bisection.
    def H1_of_H(H):
        return _head_entrainment(H)
    lo, hi = 1.05, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if H1_of_H(mid) > H1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _turbulent_cf(H: float, Re_theta: float) -> float:
    """Ludwieg-Tillmann skin friction for the turbulent boundary layer."""
    Re_theta = max(Re_theta, 1.0)
    return 0.246 * 10.0 ** (-0.678 * H) * Re_theta ** -0.268


def march_boundary_layer(s: np.ndarray, ue: np.ndarray, Re: float,
                         trip_s: float | None = None,
                         transition_model: str = "en",
                         n_crit: float = 9.0):
    """March the integral BL along one surface from stagnation to TE.

    Parameters
    ----------
    s : arc length stations (monotone increasing from the stagnation point).
    ue : edge velocity magnitude at each station (normalized by V_inf).
    Re : chord Reynolds number.
    trip_s : optional explicit transition arc length (overrides the model).
    transition_model : 'en' (Drela envelope e^n, physical, default) or
        'michel' (empirical correlation, legacy).
    n_crit : critical amplification factor for the e^n model (environment knob;
        ~9 clean, ~4-7 noisy, ~11-13 very quiet). Ignored for 'michel'.

    Returns
    -------
    BLProfile with theta, H, delta* = H*theta, cf at every station.
    """
    n = len(s)
    ue = np.abs(ue).astype(float)
    ue = np.maximum(ue, 1e-4)

    theta = np.zeros(n)
    H = np.full(n, 2.59)        # Blasius flat-plate start
    cf = np.zeros(n)

    # --- Laminar region: Thwaites -----------------------------------------
    # theta^2 = 0.45/Re * ue^-6 * integral(ue^5 ds)
    ue5 = ue ** 5
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (ue5[1:] + ue5[:-1]) * np.diff(s))])
    theta_sq = (0.45 / (Re * ue ** 6)) * integ
    theta_lam = np.sqrt(np.maximum(theta_sq, 1e-20))

    # Pressure gradient parameter lambda = theta^2/nu * due/ds = Re*theta^2*due/ds
    due_ds = np.gradient(ue, s)
    lam = np.clip(Re * theta_sq * due_ds, -0.09, 0.25)
    H_lam = _thwaites_H(lam)

    # --- Find transition ---------------------------------------------------
    Re_theta = Re * ue * theta_lam

    # Transition location: explicit trip wins; otherwise the chosen model.
    trip_idx = n - 1
    if trip_s is not None:
        for i in range(1, n):
            if s[i] >= trip_s:
                trip_idx = i
                break

    if transition_model == "michel":
        Re_s = Re * ue * s
        michel = 1.535 * np.abs(Re_s) ** 0.444
        model_idx = n - 1
        for i in range(1, n):
            if Re_theta[i] > michel[i]:
                model_idx = i
                break
    else:  # 'en' (Drela envelope e^n)
        model_idx, _ = find_transition_en(s, Re_theta, H_lam, n_crit=n_crit)

    # Laminar separation bubble closure: if the laminar shape factor reaches the
    # separation value (H ~ 3.5, Thwaites lambda <= -0.09) upstream of the e^n
    # point, the laminar layer separates and transitions in the detached shear
    # layer, reattaching turbulent (a short bubble). Model this by forcing
    # transition at the laminar separation station. This is what lets the march
    # produce a finite, physical result at low Re instead of capping H. Enabled
    # by default; the high-Re path is unaffected because separation occurs at or
    # past the (earlier) e^n transition there.
    lam_sep_idx = n - 1
    for i in range(1, n):
        if H_lam[i] >= 3.5:
            lam_sep_idx = i
            break
    model_idx = min(model_idx, lam_sep_idx)

    # Earliest of the model prediction and any explicit trip.
    transition_idx = min(trip_idx, model_idx)

    # Fill laminar part.
    theta[:transition_idx + 1] = theta_lam[:transition_idx + 1]
    H[:transition_idx + 1] = H_lam[:transition_idx + 1]
    cf[:transition_idx + 1] = _thwaites_cf(lam[:transition_idx + 1],
                                           Re_theta[:transition_idx + 1])

    # --- Turbulent region: Head's entrainment march -----------------------
    separated = False
    if transition_idx < n - 1:
        th = theta[transition_idx]
        Hk = max(min(H[transition_idx], 1.8), 1.4)   # reset to turbulent range
        H1 = _head_entrainment(Hk)
        for i in range(transition_idx + 1, n):
            ds = s[i] - s[i - 1]
            ue_i = ue[i - 1]
            due = (ue[i] - ue[i - 1]) / max(ds, 1e-9)
            Re_th = max(Re * ue_i * th, 1.0)
            cf_i = _turbulent_cf(Hk, Re_th)

            # von Karman momentum integral:
            # dtheta/ds = cf/2 - (H+2) theta/ue due/ds
            dtheta = cf_i / 2.0 - (Hk + 2.0) * th / ue_i * due
            # Head's entrainment: d(ue theta H1)/ds = ue * CE
            CE = 0.0306 * (H1 - 3.0) ** -0.6169
            dH1 = (ue_i * CE - H1 * (th * due + ue_i * dtheta)) / max(ue_i * th, 1e-9)

            th = max(th + dtheta * ds, 1e-7)
            H1 = max(H1 + dH1 * ds, 3.0 + 1e-6)
            Hk = _head_H_from_H1(H1)

            # Separation when H exceeds ~2.4-3.0 (turbulent separation).
            if Hk > 2.8:
                separated = True
                Hk = 2.8
            theta[i] = th
            H[i] = Hk
            cf[i] = _turbulent_cf(Hk, Re * ue_i * th)

    delta_star = H * theta
    return BLProfile(s=s, theta=theta, H=H, delta_star=delta_star, cf=cf,
                     transition_idx=transition_idx, separated=separated)
