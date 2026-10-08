"""e^n (envelope) transition prediction via Tollmien-Schlichting amplification.

Replaces the empirical Michel criterion with a physically-grounded estimate of
where laminar-to-turbulent transition occurs, by tracking the growth of the
most-amplified Tollmien-Schlichting (TS) instability wave.

Method: Drela's envelope e^n
----------------------------
Rather than integrating every disturbance frequency (the full e^n method), the
*envelope* method follows the envelope of the most-amplified wave. The
log-amplification factor ``n`` accumulates along the surface,

    dn/ds = (dn/dRe_theta) * (dRe_theta/ds),

where ``dn/dRe_theta`` is a closed-form function of the local shape factor H
(Drela & Giles, AIAA J. 1987). Transition is declared when ``n`` reaches a
user-set critical value ``N_crit``:

    N_crit ~ 9      clean wind tunnel / free flight (default)
    N_crit ~ 4-7    noisy stream, some surface roughness / vibration
    N_crit ~ 11-13  very quiet facility

Amplification only accumulates once ``Re_theta`` exceeds the critical value
``Re_theta_crit(H)`` below which the boundary layer is stable. Both correlations
below are the standard XFOIL-family fits.

This is an estimate, like all transition models, but unlike Michel it responds
physically to pressure gradient (through H) and to environment (through
``N_crit``).
"""

from __future__ import annotations

import numpy as np


def log10_re_theta_crit(H: np.ndarray) -> np.ndarray:
    """log10 of the critical momentum-thickness Reynolds number vs H.

    Below this Re_theta the laminar BL is stable and no TS waves amplify
    (Drela/Giles envelope correlation).
    """
    H = np.clip(H, 2.0, 20.0)
    Hm1 = 1.0 / (H - 1.0)
    return (0.7 * np.tanh(14.0 * Hm1 - 9.24)
            + 2.492 * Hm1 ** 0.43
            + 0.62)


def dn_dre_theta(H: np.ndarray) -> np.ndarray:
    """Envelope amplification rate dn/dRe_theta as a function of H.

    Drela & Giles (1987) closed-form fit for the slope of the envelope of the
    most-amplified TS wave. Returns 0 where the fit would go negative
    (effectively stable).
    """
    H = np.clip(H, 2.0, 20.0)
    Hm1 = H - 1.0
    # Drela's correlation:
    #   dn/dRe_theta = 0.01 * sqrt( (2.4 H - 3.7 + 2.5 tanh(1.5 H - 4.65))^2 + 0.25 )
    term = 2.4 * H - 3.7 + 2.5 * np.tanh(1.5 * H - 4.65)
    rate = 0.01 * np.sqrt(term ** 2 + 0.25)
    return np.maximum(rate, 0.0)


def find_transition_en(s: np.ndarray, Re_theta: np.ndarray, H: np.ndarray,
                       n_crit: float = 9.0):
    """Integrate the envelope amplification and return the transition index.

    Parameters
    ----------
    s : arc-length stations.
    Re_theta : momentum-thickness Reynolds number at each station.
    H : laminar shape factor at each station.
    n_crit : critical amplification factor (environment-dependent).

    Returns
    -------
    (transition_idx, n_profile) : the first station index where n >= n_crit
    (or len-1 if never reached), and the accumulated amplification array.
    """
    nstations = len(s)
    n_amp = np.zeros(nstations)
    re_crit = 10.0 ** log10_re_theta_crit(H)

    transition_idx = nstations - 1
    for i in range(1, nstations):
        # Amplify only once the local Re_theta exceeds the neutral point.
        if Re_theta[i] > re_crit[i]:
            dRe = Re_theta[i] - Re_theta[i - 1]
            # Use the rate at the current H; only positive dRe contributes.
            rate = 0.5 * (dn_dre_theta(H[i]) + dn_dre_theta(H[i - 1]))
            n_amp[i] = n_amp[i - 1] + rate * max(dRe, 0.0)
        else:
            n_amp[i] = n_amp[i - 1]

        if n_amp[i] >= n_crit and transition_idx == nstations - 1:
            transition_idx = i
            break

    return transition_idx, n_amp
