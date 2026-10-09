"""Discrete integral boundary-layer equations on one interval between two stations (Mach 0).

The equations, their discretisation and the transition-interval treatment exist once; every
closure comes from a closure set (:mod:`foil_lab.bl.closure_sets`), passed as ``cs``.

Equations (DG87 = Drela & Giles 1987; LNE89 = Drela 1989; FID = Fidkowski):

* momentum, DG87 eq. (4) = FID eq. (5); shape parameter, DG87 eq. (5) = FID eq. (6) (H** = 0);
* laminar: amplification dn/dxi from the set (DG87 eq. 35 / FID eq. 69);
* turbulent: shear-lag equation in the common form of FID eq. (8),

      (delta/C_tau) dC_tau/dxi = K (C_tau,EQ^1/2 - eta_D C_tau^1/2) + 2 delta (u_q - (1/ue) due/dxi),

  of which LNE89 eq. (23) is the case K = 5.6, eta_D = 1, u_q = (4/(3 delta*))[Cf/2 - ((H_k - 1)/
  (6.7 H_k))^2]; the set supplies K(U_s), eta_D and u_q;
* transition interval split into a laminar and a turbulent subinterval at the onset xi_t
  (DG87 pp. 1351-1352, Fig. 5; FID eqs. 16-18): the two subintervals' residuals are summed; the
  onset is the root of the amplification residual on the laminar subinterval, with the state
  interpolated linearly to xi_t and n = N_crit there (FID eq. 18); C_tau at onset from the set.

Two-point central differences (DG87 p. 1351) in logarithmic form, so every residual is
dimensionless (scaling in DECISIONS.md, M3):

    R1 = ln(theta2/theta1) + (2 + H)_a ln(ue2/ue1) - dxi (Cf/2)_a / theta_a
    R2 = ln(H*2/H*1) + (1 - H)_a ln(ue2/ue1) - dxi (2C_D/H* - Cf/2)_a / theta_a
    R3 = n2 - n1 - dxi (dn/dxi)_a                                                  laminar
    R3 = ln(C_tau2/C_tau1) - dxi (dln C_tau/dxi)_a                                 turbulent

(_a = average of the two stations; the shape equation may be biased toward the downstream station
with ``bias``, 0.5 = central). Every function accepts complex values (complex-step derivatives)
and is written against an array namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import closure_sets
from ._xp import re as _re


@dataclass
class Station:
    """Secondary quantities of one station."""

    theta: Any
    dstar: Any
    third: Any          # n (laminar) or C_tau (turbulent)
    ue: Any
    turbulent: bool
    h: Any = None
    hk: Any = None
    ret: Any = None
    hstar: Any = None
    cf: Any = None
    cd: Any = None
    us: Any = None
    delta: Any = None
    ctau_eq: Any = None


def station(theta, dstar, third, ue, reynolds, turbulent: bool, xp=np, cs="fidkowski2022") -> Station:
    """Closure quantities at one station (turbulent C_D uses the station's C_tau)."""
    cs = closure_sets.get(cs)
    h = dstar / theta
    hk = cs.hk(h, xp)
    ret = reynolds * ue * theta
    st = Station(theta, dstar, third, ue, turbulent, h=h, hk=hk, ret=ret)
    if turbulent:
        st.hstar = cs.hstar_turb(hk, ret, xp)
        st.cf = cs.cf_turb(hk, ret, xp)
        st.us = cs.us(st.hstar, h, hk, xp)
        st.delta = cs.delta(theta, hk, dstar, xp)
        st.ctau_eq = cs.ctau_eq(st.hstar, h, hk, st.us, ret, xp)
        st.cd = cs.cd_turb(hk, ret, st.cf, st.us, third, st.hstar, xp)
    else:
        st.hstar = cs.hstar_lam(hk, ret, xp)
        st.cf = cs.cf_lam(hk, ret, xp)
        st.cd = cs.cd_lam(hk, ret, xp)
    return st


def _common(s1: Station, s2: Station, dxi, xp, bias):
    lu = xp.log(s2.ue / s1.ue)
    th_a = 0.5 * (s1.theta + s2.theta)
    h_a = 0.5 * (s1.h + s2.h)
    r1 = xp.log(s2.theta / s1.theta) + (2.0 + h_a) * lu - dxi * 0.5 * (s1.cf + s2.cf) * 0.5 / th_a
    src1 = 2.0 * s1.cd / s1.hstar - 0.5 * s1.cf
    src2 = 2.0 * s2.cd / s2.hstar - 0.5 * s2.cf
    src = (1.0 - bias) * src1 + bias * src2
    th_b = (1.0 - bias) * s1.theta + bias * s2.theta
    r2 = (xp.log(s2.hstar / s1.hstar) + (1.0 - ((1.0 - bias) * s1.h + bias * s2.h)) * lu
          - dxi * src / th_b)
    return r1, r2


def lam_rate(s: Station, xp, cs, n_crit):
    return closure_sets.get(cs).dn_dxi(s.hk, s.theta, s.ret, s.third, n_crit, xp)


def turb_lograte(s: Station, due_dxi, xp, cs):
    """d ln C_tau / d xi of the common lag equation (FID eq. 8; LNE89 eq. 23 as a special case)."""
    cs = closure_sets.get(cs)
    k = cs.lag_k(s.us, xp)
    return (k * (xp.sqrt(s.ctau_eq) - cs.ETA_D * xp.sqrt(s.third)) / s.delta
            + 2.0 * (cs.uq(s.cf, s.hk, s.ret, s.dstar, xp) - due_dxi / s.ue))


def laminar_interval(s1: Station, s2: Station, dxi, xp=np, bias=0.5, cs="fidkowski2022", n_crit=9.0):
    r1, r2 = _common(s1, s2, dxi, xp, bias)
    r3 = s2.third - s1.third - dxi * 0.5 * (lam_rate(s1, xp, cs, n_crit) + lam_rate(s2, xp, cs, n_crit))
    return xp.stack([r1, r2, r3])


def turbulent_interval(s1: Station, s2: Station, dxi, xp=np, bias=0.5, cs="fidkowski2022"):
    r1, r2 = _common(s1, s2, dxi, xp, bias)
    due = (s2.ue - s1.ue) / dxi
    g = 0.5 * (turb_lograte(s1, due, xp, cs) + turb_lograte(s2, due, xp, cs))
    r3 = xp.log(s2.third / s1.third) - dxi * g
    return xp.stack([r1, r2, r3])


ONSET_BISECTIONS = 40


def transition_onset(s1: Station, theta2, dstar2, ue2, dxi, reynolds, n_crit, xp=np, cs="fidkowski2022"):
    """Onset fraction w = (xi_t - xi1)/dxi in [0, 1]: root of the amplification residual on the
    laminar subinterval (FID eq. 18), R_t(w) = N_crit - n1 - w dxi (rate(u1) + rate(u_t^-))/2, with
    u_t^- interpolated linearly to xi_t and n = N_crit there.

    Solved by bisection (whose iterates do not depend on the state, so they carry no derivative)
    followed by one Newton correction, so that the derivatives of w are those of the implicit
    function theorem, -dR_t/dx / dR_t/dw (cf. FID eq. 19), in every array namespace. If R_t has no
    root in [0, 1] (a provisional Newton state), w is taken at the nearer end: 1 if N_crit is not
    reached, 0 if it was already exceeded (DECISIONS.md M3)."""
    r1 = lam_rate(s1, xp, cs, n_crit)

    def rt(w):
        st = station(s1.theta + w * (theta2 - s1.theta), s1.dstar + w * (dstar2 - s1.dstar),
                     n_crit, s1.ue + w * (ue2 - s1.ue), reynolds, False, xp, cs)
        return n_crit - s1.third - w * dxi * 0.5 * (r1 + lam_rate(st, xp, cs, n_crit))

    zero = 0.0 * s1.theta
    r_lo, r_hi = rt(zero), rt(zero + 1.0)
    lo, hi = zero, zero + 1.0
    for _ in range(ONSET_BISECTIONS):
        mid = 0.5 * (lo + hi)
        pos = _re(rt(mid)) > 0.0
        lo = xp.where(pos, mid, lo)
        hi = xp.where(pos, hi, mid)
    w0 = 0.5 * (lo + hi)
    h = 1e-7
    drdw = (rt(w0 + h) - rt(w0 - h)) / (2.0 * h)
    w = w0 - rt(w0) / drdw
    bracketed = (_re(r_lo) > 0.0) & (_re(r_hi) <= 0.0)
    w = xp.where(bracketed, w, xp.where(_re(r_lo) <= 0.0, zero, zero + 1.0))
    return w


def transition_interval(theta1, dstar1, n1, ue1, theta2, dstar2, ctau2, ue2, dxi, reynolds,
                        n_crit, xp=np, bias=0.5, forced_w=None, cs="fidkowski2022"):
    """Residuals of an interval containing transition onset (DG87 Fig. 5; FID eqs. 16-18).

    The state at the onset is interpolated linearly between the stations; the laminar
    subinterval ends with n = N_crit, the turbulent one starts from the set's C_tau at transition;
    the two subintervals' residuals are added (FID eq. 16). The third residual is the lag equation
    on the turbulent subinterval (the onset itself is defined by the amplification residual).
    ``forced_w``: onset fraction of a forced trip instead of the amplification criterion.
    Returns (residuals, w).
    """
    s1 = station(theta1, dstar1, n1, ue1, reynolds, False, xp, cs)
    w = (transition_onset(s1, theta2, dstar2, ue2, dxi, reynolds, n_crit, xp, cs)
         if forced_w is None else forced_w)
    th_t = theta1 + w * (theta2 - theta1)
    ds_t = dstar1 + w * (dstar2 - dstar1)
    ue_t = ue1 + w * (ue2 - ue1)
    stl = station(th_t, ds_t, n_crit, ue_t, reynolds, False, xp, cs)
    st_tmp = station(th_t, ds_t, 1.0, ue_t, reynolds, True, xp, cs)
    ct_t = closure_sets.get(cs).ctau_transition(st_tmp.hk, st_tmp.ctau_eq, xp)
    stt = station(th_t, ds_t, ct_t, ue_t, reynolds, True, xp, cs)
    s2 = station(theta2, dstar2, ctau2, ue2, reynolds, True, xp, cs)
    d1, d2 = w * dxi, (1.0 - w) * dxi
    a1, a2 = _common(s1, stl, d1, xp, bias)
    b1, b2 = _common(stt, s2, d2, xp, bias)
    due = (s2.ue - stt.ue) / xp.where(_re(d2) > 1e-12 * _re(dxi), d2, 1e-12 * dxi)
    g = 0.5 * (turb_lograte(stt, due, xp, cs) + turb_lograte(s2, due, xp, cs))
    r3 = xp.log(s2.third / stt.third) - d2 * g
    return xp.stack([a1 + b1, a2 + b2, r3]), w


def similarity_start(theta, dstar, ue, xi, m, reynolds, xp=np, cs="fidkowski2022"):
    """Residuals of the first station of a branch: the momentum and shape equations for a local
    similarity flow ue ~ xi^m, theta^2 ~ xi^(1 - m), H constant, written with the set's own
    laminar closures. With G = Re ue theta^2/xi:

        G [(1 - m)/2 + (2 + H) m] = Re_theta Cf/2,     G (1 - H) m = Re_theta (2C_D/H* - Cf/2).

    At m = 1 these are the stagnation-point equations of FID p. 7 (ue = K xi); at m = 0 they give
    the flat plate (2C_D/H* = Cf/2). DECISIONS.md M3, "Two closure sets"."""
    s = station(theta, dstar, 0.0, ue, reynolds, False, xp, cs)
    g = reynolds * ue * theta ** 2 / xi
    fc = s.ret * 0.5 * s.cf
    fd = s.ret * 2.0 * s.cd / s.hstar
    return xp.stack([g * ((1.0 - m) / 2.0 + (2.0 + s.h) * m) / fc - 1.0,
                     (g * (1.0 - s.h) * m - (fd - fc)) / fc])
