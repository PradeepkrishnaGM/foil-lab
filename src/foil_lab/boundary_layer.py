# src/foil_lab/boundary_layer.py

import numpy as np
import scipy.integrate as integrate

def analyze_boundary_layer(s: np.ndarray, v: np.ndarray, Re: float,
                           roughness: int = 0, trip_s: float | None = None) -> tuple:
    """
    Integrates the boundary layer along a single surface (upper or lower)
    starting from the stagnation point.

    Parameters
    ----------
    roughness : int (0-3)
        Surface roughness state. Higher values trip the boundary layer to
        turbulence earlier by lowering the effective Michel transition
        threshold (dirt/bugs/paint imperfections). 0 = clean.
    trip_s : float or None
        Explicit transition trip location, in surface arc length from the
        stagnation point. If set, natural transition is overridden: the flow is
        forced turbulent at the first panel at or beyond this arc length
        (whichever comes first, the trip or natural transition).

    Returns
    -------
    theta_te : float
        Momentum thickness at the trailing edge.
    transition_idx : int
        Panel index where flow tripped to turbulent.
    separated : bool
        True if laminar separation (Thwaites lambda < -0.09) was detected.
    sep_fraction : float
        Fraction of surface arc length downstream of the separation/transition
        point (0 if attached to TE); used by post-stall corrections.
    """
    v = np.abs(v).astype(float)
    v[v < 1e-5] = 1e-5

    # 1. Laminar integration: Thwaites' method.
    v5 = v ** 5
    integral_v5 = integrate.cumulative_trapezoid(v5, s, initial=0.0)
    theta_sq = (0.45 / (Re * v ** 6)) * integral_v5
    theta = np.sqrt(np.abs(theta_sq))

    # 2. Transition: Michel's criterion, with roughness lowering the threshold.
    re_theta = Re * v * theta
    re_s = Re * v * s
    # Each roughness level removes ~12% of the threshold, tripping earlier.
    rough_factor = max(0.1, 1.0 - 0.12 * int(np.clip(roughness, 0, 3)))
    michel_threshold = 1.535 * np.abs(re_s) ** 0.444 * rough_factor

    transition_idx = len(s) - 1
    laminar_sep_idx = None      # laminar separation (usually a short bubble)
    turbulent_sep_idx = None    # turbulent separation (real stall)

    for i in range(1, len(s)):
        if i < len(s) - 1:
            dv_ds = (v[i + 1] - v[i - 1]) / (s[i + 1] - s[i - 1])
            thwaites_lambda = theta_sq[i] * Re * dv_ds
            if thwaites_lambda < -0.09 and laminar_sep_idx is None:
                # Laminar separation -> transition trigger (bubble reattachment).
                laminar_sep_idx = i
                transition_idx = i
        if re_theta[i] > michel_threshold[i] and transition_idx == len(s) - 1:
            transition_idx = i

    # Explicit trip wire: force transition at/just past the requested arc length,
    # but never later than natural transition already found.
    if trip_s is not None and trip_s > 0:
        trip_idx = int(np.searchsorted(s, trip_s))
        trip_idx = min(max(trip_idx, 1), len(s) - 1)
        transition_idx = min(transition_idx, trip_idx)

    # Turbulent separation: downstream of transition, a strongly adverse
    # gradient (Stratford-like) detaches the turbulent layer => genuine stall.
    for i in range(transition_idx + 1, len(s) - 1):
        dv_ds = (v[i + 1] - v[i - 1]) / (s[i + 1] - s[i - 1])
        # Turbulent separation when the deceleration is strong relative to the
        # local edge speed (empirical threshold).
        if dv_ds < -2.5 * v[i] / max(s[i], 1e-6):
            turbulent_sep_idx = i
            break

    separated = turbulent_sep_idx is not None

    # 3. Turbulent growth (flat-plate 1/7th-power approximation).
    theta_te = theta[-1]
    if transition_idx < len(s) - 1:
        s_turb = s[-1] - s[transition_idx]
        if s_turb > 0:
            theta_turb_growth = 0.036 * s_turb / (Re * s_turb) ** 0.2
            theta_te = theta[transition_idx] + theta_turb_growth

    # Separation fraction is measured from the TURBULENT separation point only
    # (laminar bubbles reattach and do not constitute stall).
    total_s = s[-1] if s[-1] > 0 else 1.0
    if separated:
        sep_fraction = (s[-1] - s[turbulent_sep_idx]) / total_s
    else:
        sep_fraction = 0.0

    return theta_te, transition_idx, separated, sep_fraction


def calculate_viscous_drag(mid_x: np.ndarray, mid_y: np.ndarray, lengths: np.ndarray,
                           vt: np.ndarray, Re: float, roughness: int = 0,
                           return_details: bool = False,
                           trip_x_upper: float | None = None,
                           trip_x_lower: float | None = None):
    """Find the stagnation point, integrate the boundary layer on each surface,
    and apply the Squire--Young formula for profile drag.

    Ordering-robust: the surface is split at the stagnation point (minimum
    surface speed) into the two branches running from the stagnation point to
    each trailing-edge end of the node list. Works for either panel orientation.

    Parameters
    ----------
    roughness : int (0-3)
        Threaded into the transition model and used to scale up the final drag
        (rougher surface => more skin friction).
    return_details : bool
        If True, also return the worst-case separation fraction across the two
        surfaces (for post-stall corrections).
    trip_x_upper, trip_x_lower : float or None
        Explicit transition trip x/c locations for each surface branch. Converted
        to surface arc length internally and passed to the BL integrator.
    """
    if Re <= 1000:  # Inviscid methods break down below ~Re 1000.
        return (0.0, 0.0) if return_details else 0.0

    npan = vt.size
    guard = max(2, npan // 20)
    stagnation_idx = int(np.argmin(np.abs(vt[guard:npan - guard]))) + guard

    s_a = np.insert(np.cumsum(lengths[stagnation_idx::-1]), 0, 0.0)[:-1]
    v_a = vt[stagnation_idx::-1]
    x_a = mid_x[stagnation_idx::-1]
    s_b = np.insert(np.cumsum(lengths[stagnation_idx:]), 0, 0.0)[:-1]
    v_b = vt[stagnation_idx:]
    x_b = mid_x[stagnation_idx:]

    # Map a requested trip x/c to an arc-length position on each branch. Branch
    # identity (which is "upper") is resolved by mean y-sign of the branch.
    y_a = mid_y[stagnation_idx::-1]
    y_b = mid_y[stagnation_idx:]
    a_is_upper = np.mean(y_a) >= np.mean(y_b)
    trip_a = trip_x_upper if a_is_upper else trip_x_lower
    trip_b = trip_x_lower if a_is_upper else trip_x_upper

    def x_to_s(trip_x, x_arr, s_arr):
        if trip_x is None or trip_x <= 0:
            return None
        idx = int(np.argmin(np.abs(x_arr - trip_x)))
        return float(s_arr[idx])

    trip_s_a = x_to_s(trip_a, x_a, s_a)
    trip_s_b = x_to_s(trip_b, x_b, s_b)

    theta_a, _, sep_a, frac_a = analyze_boundary_layer(s_a, v_a, Re, roughness, trip_s_a)
    theta_b, _, sep_b, frac_b = analyze_boundary_layer(s_b, v_b, Re, roughness, trip_s_b)

    v_te = 0.5 * (np.abs(v_a[-1]) + np.abs(v_b[-1]))
    v_te = max(v_te, 0.5)
    cd = 2.0 * (theta_a + theta_b) * (v_te ** 3.2)

    # Roughness linearly scales up total drag (paint/dirt skin-friction penalty).
    cd *= (1.0 + 0.10 * int(np.clip(roughness, 0, 3)))

    sep_fraction = max(frac_a, frac_b)
    cd = float(cd)
    return (cd, sep_fraction) if return_details else cd

