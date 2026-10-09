"""Closure set "drela1987": Drela & Giles (1987) with Drela (1989), as published (Mach 0).

Sources, read from page images (transcription in the M3 record of DECISIONS.md):

* DG87: M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low Reynolds
  number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355.
* LNE89: M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils", in
  T. J. Mueller (ed.), Low Reynolds Number Aerodynamics, Lecture Notes in Engineering 54,
  Springer, 1989, pp. 1-12.

With the edge Mach number M_e = 0: H_k = H (DG87 eq. 9), F_c = 1 (eq. 15), H** = 0 (eq. 13).
Interpretations recorded in DECISIONS.md (M3): DG87 eq. (18) is printed with "H_k" on the left
and is read as H* (as its text states); its unqualified "log" is read as the natural logarithm.
The papers give no limits on H_k or Re_theta; none are applied here.
"""

from __future__ import annotations

from ._xp import branch, ns, re

NAME = "drela1987"

#: provenance of every closure of the set (source, equation)
PROVENANCE = {
    "hk": "DG87 eq. (9), M_e = 0: H_k = H",
    "hstar_lam": "DG87 eq. (10)",
    "cf_lam": "DG87 eq. (11)",
    "cd_lam": "DG87 eq. (12)",
    "cf_turb": "DG87 eq. (14), F_c = 1 (eq. 15)",
    "hstar_turb": "DG87 eqs. (18)-(19) (read as H*, natural log; DECISIONS.md M3)",
    "us": "DG87 eq. (21)",
    "delta": "DG87 eq. (23)",
    "ctau_eq": "DG87 eq. (24)",
    "cd_turb": "DG87 eq. (20)",
    "lag": "LNE89 eq. (23): K = 5.6, eta_D = 1, u_q = (4/(3 delta*))[Cf/2 - ((H_k-1)/(6.7 H_k))^2]",
    "dn_dxi": "DG87 eqs. (29), (30), (33)-(35); zero below Re_theta0 (step)",
    "ctau_transition": "DG87 p. 1351: C_tau^1/2 = 0.7 C_tau,EQ^1/2 at onset",
}


# --------------------------------------------------------------------------------------- #
# Laminar closure (Falkner-Skan family), DG87 eqs. (10)-(12)
# --------------------------------------------------------------------------------------- #
def hk(h, xp=None):
    """DG87 eq. (9) at M_e = 0."""
    return h


def hstar_lam(hk, ret=None, xp=None):
    """H* laminar, DG87 eq. (10)."""
    xp = xp or ns(hk)
    return branch(xp, re(hk) < 4.0, hk, 3.0, 5.0,
                  lambda h: 1.515 + 0.076 * (4.0 - h) ** 2 / h,
                  lambda h: 1.515 + 0.040 * (h - 4.0) ** 2 / h)


def cf_lam(hk, ret, xp=None):
    """Skin friction (local edge normalisation) laminar: Re_theta Cf/2 from DG87 eq. (11)."""
    xp = xp or ns(hk)
    f = branch(xp, re(hk) < 7.4, hk, 3.0, 8.0,
               lambda h: -0.067 + 0.01977 * (7.4 - h) ** 2 / (h - 1.0),
               lambda h: -0.067 + 0.022 * (1.0 - 1.4 / (h - 6.0)) ** 2)
    return 2.0 * f / ret


def cd_lam(hk, ret, xp=None):
    """Dissipation coefficient C_D laminar, from Re_theta 2C_D/H* of DG87 eq. (12)."""
    xp = xp or ns(hk)
    g = branch(xp, re(hk) < 4.0, hk, 3.0, 5.0,
               lambda h: 0.207 + 0.00205 * (4.0 - h) ** 5.5,
               lambda h: 0.207 - 0.003 * (h - 4.0) ** 2 / (1.0 + 0.02 * (h - 4.0) ** 2))
    return hstar_lam(hk, ret, xp) * g / (2.0 * ret)


# --------------------------------------------------------------------------------------- #
# Turbulent closure (Swafford profiles), DG87 eqs. (14), (18)-(24); lag LNE89 eq. (23)
# --------------------------------------------------------------------------------------- #
def cf_turb(hk, ret, xp=None):
    """Skin friction (local edge normalisation) turbulent, DG87 eq. (14) with F_c = 1."""
    xp = xp or ns(hk)
    return (0.3 * xp.exp(-1.33 * hk) * (xp.log10(ret)) ** (-1.74 - 0.31 * hk)
            + 0.00011 * (xp.tanh(4.0 - hk / 0.875) - 1.0))


def h0_turb(ret, xp=None):
    """H_0 of DG87 eq. (19)."""
    xp = xp or ns(ret)
    return branch(xp, re(ret) < 400.0, ret, 100.0, 1000.0,
                  lambda r: 4.0 + 0.0 * r,
                  lambda r: 3.0 + 400.0 / r)


def hstar_turb(hk, ret, xp=None):
    """H* turbulent, DG87 eq. (18) (read as H*, natural logarithm; DECISIONS.md M3)."""
    xp = xp or ns(hk)
    h0 = h0_turb(ret, xp)
    base = 1.505 + 4.0 / ret
    below = re(hk) < re(h0)
    hk_lo = xp.where(below, hk, h0 - 0.5)
    hk_hi = xp.where(below, h0 + 0.5, hk)
    lo = base + (0.165 - 1.6 / xp.sqrt(ret)) * (h0 - hk_lo) ** 1.6 / hk_lo
    lnr = xp.log(ret)
    hi = base + (hk_hi - h0) ** 2 * (0.04 / hk_hi + 0.007 * lnr / (hk_hi - h0 + 4.0 / lnr) ** 2)
    return xp.where(below, lo, hi)


def us(hstar, h, hk, xp=None):
    """Normalised slip velocity, DG87 eq. (21)."""
    return 0.5 * hstar * (1.0 - (4.0 / 3.0) * (hk - 1.0) / h)


def delta(theta, hk, dstar, xp=None):
    """Nominal boundary-layer thickness, DG87 eq. (23)."""
    return theta * (3.15 + 1.72 / (hk - 1.0)) + dstar


def ctau_eq(hstar, h, hk, us_, ret=None, xp=None):
    """Equilibrium shear stress coefficient, DG87 eq. (24)."""
    return hstar * 0.015 / (1.0 - us_) * (hk - 1.0) ** 3 / (hk ** 2 * h)


def cd_turb(hk, ret, cf, us_, ctau, hstar, xp=None):
    """Dissipation coefficient, DG87 eq. (20): C_D = (Cf/2) U_s + C_tau (1 - U_s)."""
    return 0.5 * cf * us_ + ctau * (1.0 - us_)


#: wall/wake dissipation length ratio in the lag equation (LNE89 eq. 23 has none: 1)
ETA_D = 1.0


def lag_k(us_, xp=None):
    """Rate constant of the lag equation, LNE89 eq. (23): 5.6."""
    return 5.6 + 0.0 * us_


def uq(cf, hk, ret, dstar, xp=None):
    """Equilibrium (1/ue) due/dxi of LNE89 eq. (23): (4/(3 delta*)) [Cf/2 - ((H_k - 1)/(6.7 H_k))^2]."""
    return (4.0 / (3.0 * dstar)) * (0.5 * cf - ((hk - 1.0) / (6.7 * hk)) ** 2)


# --------------------------------------------------------------------------------------- #
# Transition: envelope e^n, DG87 eqs. (29), (30), (33)-(35)
# --------------------------------------------------------------------------------------- #
def dn_dretheta(hk, xp=None):
    """DG87 eq. (29)."""
    xp = xp or ns(hk)
    return 0.01 * xp.sqrt((2.4 * hk - 3.7 + 2.5 * xp.tanh(1.5 * hk - 4.65)) ** 2 + 0.25)


def log10_retheta0(hk, xp=None):
    """Critical Re_theta, DG87 eq. (30): log10 Re_theta0."""
    xp = xp or ns(hk)
    a = 1.0 / (hk - 1.0)
    return (1.415 * a - 0.489) * xp.tanh(20.0 * a - 12.9) + 3.295 * a + 0.44


def ell_fs(hk):
    """l(H_k) = rho ue theta^2 / (mu xi), DG87 eq. (33)."""
    return (6.54 * hk - 14.07) / hk ** 2


def m_fs(hk):
    """m(H_k) = (xi/ue) due/dxi, DG87 eq. (34)."""
    return (0.058 * (hk - 4.0) ** 2 / (hk - 1.0) - 0.068) / ell_fs(hk)


def dn_dxi(hk, theta, ret, n=None, n_crit=None, xp=None):
    """Amplification rate per unit length, DG87 eq. (35), zero below Re_theta0 or where
    l(H_k) <= 0 (DECISIONS.md M3, item 4). Independent of n."""
    xp = xp or ns(hk)
    ell = ell_fs(hk)
    rate = dn_dretheta(hk, xp) * 0.5 * (m_fs(hk) + 1.0) * ell / theta
    active = (re(xp.log10(ret)) > re(log10_retheta0(hk, xp))) & (re(ell) > 0.0)
    return xp.where(active, rate, 0.0 * rate)


def ctau_transition(hk, ctau_eq_, xp=None):
    """Initial C_tau at transition onset: C_tau^1/2 = 0.7 C_tau,EQ^1/2 (DG87 p. 1351, Fig. 5)."""
    return 0.49 * ctau_eq_
