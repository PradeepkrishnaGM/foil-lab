"""Closure set "fidkowski2022": the closures as published by Fidkowski (Mach 0).

Source (FID): K. J. Fidkowski, "A Coupled Inviscid-Viscous Airfoil Analysis Solver, Revisited",
appendix B, eqs. (45)-(79), Table 2, and eqs. (8), (17); read from page images, never from code.
The paper states that its closures in large part follow XFOIL and its earlier documentation
(DG87, LNE89, BTE89), that deviations are mainly XFOIL updates after those publications, and
that eqs. (58), (74) and (76) are the author's own modifications for Newton robustness. Those
three are implemented as published and marked "author's own" below.

At M_e = 0: H_k = H (eq. 45), H** = 0 (eq. 46), H*_turb = H*_turb,inc (eq. 49), F_c = 1 (eq. 60).
"Small H_k values are limited to ... 1.05 on the airfoil" (p. 18): applied in ``hk``.
Constants from Table 2: G_A 6.7, G_beta 0.75, G_C 18, eta_D 1 on the airfoil (0.9 in the wake),
K_lag 5.6, C_tau 1.8, E_tau 3.3.
"""

from __future__ import annotations

from ._xp import branch, maximum, minimum, ns, re

NAME = "fidkowski2022"

G_A, G_BETA, G_C = 6.7, 0.75, 18.0
K_LAG, C_TAU, E_TAU = 5.6, 1.8, 3.3
#: wall/wake dissipation length ratio on the airfoil (Table 2: 0.9 in the wake)
ETA_D = 1.0
HK_MIN_AIRFOIL = 1.05
US_MAX_AIRFOIL = 0.98

PROVENANCE = {
    "hk": "FID eq. (45), M_e = 0; limited to >= 1.05 on the airfoil (FID p. 18)",
    "hstar_lam": "FID eqs. (47)-(48) (later XFOIL form; differs from DG87 eq. 10)",
    "cf_lam": "FID eq. (56) (later XFOIL form; differs from DG87 eq. 11)",
    "cd_lam": "FID eq. (61) (= DG87 eq. 12 for H_k < 4; -0.0016 instead of -0.003 above)",
    "cf_turb": "FID eqs. (57)-(60) (= DG87 eq. 14 with guards; eq. 58 is the author's own)",
    "hstar_turb": "FID eqs. (49)-(54) (later form; differs from DG87 eq. 18)",
    "us": "FID eq. (68) (= DG87 eq. 21 with G_beta = 0.75, max 0.98 on the airfoil)",
    "delta": "FID eq. (55) (= DG87 eq. 23, capped at 12 theta)",
    "ctau_eq": "FID eqs. (78)-(79) (H_kc = H_k - 1 - 18/Re_theta; DG87 eq. 24 has (H_k - 1)^3, 0.015)",
    "cd_turb": "FID eqs. (62)-(65) (differs from DG87 eq. 20: low-Re wall factor, 0.995, stress term, max with laminar)",
    "lag": "FID eqs. (8), (77)-(78): K = 5.6/(G_beta (1 + U_s)), eta_D = 1, u_q with H_kc (LNE89 eq. 23 has K = 5.6, H_k - 1)",
    "dn_dxi": "FID eqs. (69)-(76) (eqs. 74 and 76 are the author's own)",
    "ctau_transition": "FID eq. (17): C_tau^1/2 = 1.8 exp(-3.3/(H_k - 1)) C_tau,EQ^1/2 (DG87: 0.7)",
}


def hk(h, xp=None):
    """FID eq. (45) at M_e = 0, limited to 1.05 on the airfoil (FID p. 18)."""
    xp = xp or ns(h)
    return maximum(xp, h, HK_MIN_AIRFOIL + 0.0 * h)


# --------------------------------------------------------------------------------------- #
# Laminar
# --------------------------------------------------------------------------------------- #
def hstar_lam(hk, ret=None, xp=None):
    """FID eqs. (47)-(48)."""
    xp = xp or ns(hk)
    return branch(xp, re(hk) < 4.35, hk, 3.0, 5.0,
                  lambda h: ((0.0111 * (h - 4.35) ** 2 - 0.0278 * (h - 4.35) ** 3) / (h + 1.0)
                             + 1.528 - 0.0002 * ((h - 4.35) * h) ** 2),
                  lambda h: 0.015 * (h - 4.35) ** 2 / h + 1.528)


def cf_lam(hk, ret, xp=None):
    """FID eq. (56) (the paper gives cf, local edge normalisation)."""
    xp = xp or ns(hk)
    f = branch(xp, re(hk) < 5.5, hk, 3.0, 6.0,
               lambda h: 0.0727 * (5.5 - h) ** 3 / (h + 1.0) - 0.07,
               lambda h: 0.015 * (1.0 - 1.0 / (h - 4.5)) ** 2 - 0.07)
    return f / ret


def _cdi_lam(hk, ret, xp):
    """2 C_D/H* laminar, FID eq. (61)."""
    g = branch(xp, re(hk) < 4.0, hk, 3.0, 5.0,
               lambda h: 0.00205 * (4.0 - h) ** 5.5 + 0.207,
               lambda h: -0.0016 * (h - 4.0) ** 2 / (1.0 + 0.02 * (h - 4.0) ** 2) + 0.207)
    return g / ret


def cd_lam(hk, ret, xp=None):
    """C_D laminar from FID eq. (61)."""
    xp = xp or ns(hk)
    return 0.5 * hstar_lam(hk, ret, xp) * _cdi_lam(hk, ret, xp)


# --------------------------------------------------------------------------------------- #
# Turbulent
# --------------------------------------------------------------------------------------- #
def cf_turb(hk, ret, xp=None):
    """FID eqs. (57)-(60) at M_e = 0. Eq. (58) (limit on A_cf) is the author's own."""
    xp = xp or ns(hk)
    a = -1.33 * hk
    a = branch(xp, re(a) < -17.0, a, -18.0, -10.0,
               lambda v: -20.0 + 3.0 * xp.exp((v + 17.0) / 3.0), lambda v: v)
    b = maximum(xp, xp.log10(ret), 1.303 + 0.0 * ret)
    return 0.3 * xp.exp(a) * b ** (-1.74 - 0.31 * hk) + 0.00011 * (xp.tanh(4.0 - hk / 0.875) - 1.0)


def hstar_turb(hk, ret, xp=None):
    """FID eqs. (49)-(54) at M_e = 0."""
    xp = xp or ns(hk)
    h0 = minimum(xp, 3.0 + 400.0 / ret, 4.0 + 0.0 * ret)              # eq. (52)
    rt = maximum(xp, ret, 200.0 + 0.0 * ret)                           # eq. (53)
    base = 1.5 + 4.0 / rt
    below = re(hk) < re(h0)
    hk_lo = xp.where(below, hk, h0 - 0.5)
    hk_hi = xp.where(below, h0 + 0.5, hk)
    hr = (h0 - hk_lo) / (h0 - 1.0)                                     # eq. (51)
    lo = base + (0.5 - 4.0 / rt) * 1.5 * hr ** 2 / (hk_lo + 0.5)
    a = hk_hi - h0 + 4.0 / xp.log(rt)                                  # eq. (54)
    hi = base + (hk_hi - h0) ** 2 * (0.007 * xp.log(rt) / a ** 2 + 0.015 / hk_hi)
    return xp.where(below, lo, hi)


def us(hstar, h, hk, xp=None):
    """FID eq. (68), max 0.98 on the airfoil."""
    xp = xp or ns(h)
    u = 0.5 * hstar * (1.0 - (hk - 1.0) / (h * G_BETA))
    return minimum(xp, u, US_MAX_AIRFOIL + 0.0 * u)


def delta(theta, hk, dstar, xp=None):
    """FID eq. (55)."""
    xp = xp or ns(theta)
    return minimum(xp, 3.15 * theta + 1.72 * theta / (hk - 1.0) + dstar, 12.0 * theta)


def _hkc(hk, ret):
    """FID eq. (78) on the airfoil."""
    return hk - 1.0 - G_C / ret


def ctau_eq(hstar, h, hk, us_, ret, xp=None):
    """FID eq. (79): C_tau,EQ = H* (H_k - 1) H_kc^2 / [2 G_A^2 G_beta (1 - U_s) H H_k^2]."""
    return hstar * (hk - 1.0) * _hkc(hk, ret) ** 2 / (2.0 * G_A ** 2 * G_BETA * (1.0 - us_) * h * hk ** 2)


def cd_turb(hk, ret, cf, us_, ctau, hstar, xp=None):
    """C_D turbulent from FID eqs. (62)-(65): 2C_D/H* = max(wall + outer + stress, laminar)."""
    xp = xp or ns(hk)
    wall = 0.5 * cf * us_ * (2.0 / hstar) * 0.5 * (1.0 + xp.tanh((hk - 1.0) * xp.log(ret) / 2.1))
    outer = ctau * (0.995 - us_) * 2.0 / hstar
    stress = 0.3 * (0.995 - us_) ** 2 / (hstar * ret)
    cdi = maximum(xp, wall + outer + stress, _cdi_lam(hk, ret, xp))
    return 0.5 * hstar * cdi


def lag_k(us_, xp=None):
    """Rate constant of the lag equation, FID eq. (8): K_lag/(G_beta (1 + U_s))."""
    return K_LAG / (G_BETA * (1.0 + us_))


def uq(cf, hk, ret, dstar, xp=None):
    """Equilibrium (1/ue) due/dxi, FID eq. (77) with eta_D = 1 on the airfoil."""
    return (0.5 * cf - (_hkc(hk, ret) / (G_A * ETA_D * hk)) ** 2) / (G_BETA * dstar)


# --------------------------------------------------------------------------------------- #
# Transition
# --------------------------------------------------------------------------------------- #
def dn_dxi(hk, theta, ret, n, n_crit, xp=None):
    """FID eqs. (69)-(76). Eqs. (74) (L_0) and (76) (epsilon) are the author's own."""
    xp = xp or ns(hk)
    hh = 1.0 / (hk - 1.0)                                              # eq. (75)
    f = -0.05 + 2.7 * hh - 5.5 * hh ** 2 + 3.0 * hh ** 3 + 0.1 * xp.exp(-20.0 * hh)   # eq. (70)
    g = 0.028 / hh - 0.0345 * xp.exp(-(3.87 * hh - 2.52) ** 2)        # eq. (71)
    l0 = 2.492 * hh ** 0.43 + 0.7 * (1.0 + xp.tanh(14.0 * hh - 9.24))  # eq. (74)
    s = (xp.log10(ret) - (l0 - 0.1)) / 0.2                             # eq. (73)
    r = xp.where(re(s) < 0.0, 0.0 * s, xp.where(re(s) > 1.0, 1.0 + 0.0 * s, 3.0 * s ** 2 - 2.0 * s ** 3))
    eps = 0.001 * (1.0 + xp.tanh(5.0 * (n - n_crit)))                  # eq. (76)
    return (r * f * g + eps) / theta                                   # eq. (69)


def ctau_transition(hk, ctau_eq_, xp=None):
    """FID eq. (17): C_tau^1/2 = C_tau exp(-E_tau/(H_k - 1)) C_tau,EQ^1/2."""
    xp = xp or ns(hk)
    return (C_TAU * xp.exp(-E_TAU / (hk - 1.0))) ** 2 * ctau_eq_



# --------------------------------------------------------------------------------------- #
# Wake (FID pp. 7, 18-19): one layer; closures use delta* - h_w (eq. 15)
# --------------------------------------------------------------------------------------- #
#: wall/wake dissipation length ratio in the wake (Table 2)
ETA_D_WAKE = 0.9
HK_MIN_WAKE = 1.00005
US_MAX_WAKE = 0.99995
F_W = 2.5
#: clip of the trailing-edge thickness slope dt/dxi in eq. (15): +-3/f_w
WAKE_DTDX_CLIP = 3.0 / F_W
PROVENANCE.update({
    "wake_start": "FID eq. (14): theta, delta* + h_TE summed; C_tau^1/2 theta-weighted; laminar TE: eq. (17)",
    "dead_air": "FID eq. (15): h_w = h_TE [1 + (2 + f_w dt/dxi) xi_w/L_w](1 - xi_w/L_w)^2, L_w = 2.5 h_TE, |dt/dxi| <= 3/f_w",
    "cd_wake": "FID eqs. (66)-(67); Cf = 0 in the wake (p. 17)",
    "wake_closures": "FID: H_k >= 1.00005, U_s <= 0.99995, eta_D = 0.9, G~C = 0 in the wake (eqs. 68, 77-78, Table 2)",
})


def hk_wake(h, xp=None):
    """FID eq. (45), limited to 1.00005 in the wake (p. 18)."""
    xp = xp or ns(h)
    return maximum(xp, h, HK_MIN_WAKE + 0.0 * h)


def us_wake(hstar, h, hk, xp=None):
    """FID eq. (68), max 0.99995 in the wake."""
    xp = xp or ns(h)
    u = 0.5 * hstar * (1.0 - (hk - 1.0) / (h * G_BETA))
    return minimum(xp, u, US_MAX_WAKE + 0.0 * u)


def ctau_eq_wake(hstar, h, hk, us_, ret, xp=None):
    """FID eq. (79) with H_kc = H_k - 1 (G~C = 0 in the wake, eq. 78)."""
    return hstar * (hk - 1.0) ** 3 / (2.0 * G_A ** 2 * G_BETA * (1.0 - us_) * h * hk ** 2)


def cd_wake(hk, ret, us_, ctau, hstar, dead_frac=0.0, xp=None):
    """C_D in the wake from FID eqs. (66)-(67) (``dead_frac`` unused in this set)."""
    xp = xp or ns(hk)
    outer = ctau * (0.995 - us_) * 2.0 / hstar
    stress = 0.3 * (0.995 - us_) ** 2 / (hstar * ret)
    lamwake = 2.2 * (1.0 - 1.0 / hk) ** 2 * (1.0 / hk) / (hstar * ret)
    return 0.5 * hstar * maximum(xp, outer + stress, lamwake)


def uq_wake(hk, ret, dstar, xp=None):
    """FID eq. (77) in the wake: Cf = 0, eta_D = 0.9, H_kc = H_k - 1."""
    return (0.0 - ((hk - 1.0) / (G_A * ETA_D_WAKE * hk)) ** 2) / (G_BETA * dstar)


def ctau_wake_start(theta_u, ctau_u, theta_l, ctau_l, xp=None):
    """FID eq. (14): C_tau^1/2 of the first wake station, theta-weighted."""
    xp = xp or ns(theta_u)
    return ((theta_u * xp.sqrt(ctau_u) + theta_l * xp.sqrt(ctau_l)) / (theta_u + theta_l)) ** 2
