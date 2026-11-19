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

Two-point differences in logarithmic form, so every residual is dimensionless (scaling in
DECISIONS.md, M3); the source terms of the momentum and shape equations are integrated in ln xi,
with xi carried inside the average (FID eqs. 9-10), which keeps the first interval next to a
stagnation point consistent (there Cf and C_D grow like 1/ue ~ 1/xi; DECISIONS.md M3, C2):

    R1 = ln(theta2/theta1) + (2 + H)_a ln(ue2/ue1) - ln(xi2/xi1) (xi Cf/(2 theta))_a
    R2 = ln(H*2/H*1) + (1 - H)_a ln(ue2/ue1) - ln(xi2/xi1) (xi (2C_D/H* - Cf/2)/theta)_u
    R3 = n2 - n1 - dxi (dn/dxi)_a                                                  laminar
    R3 = ln(C_tau2/C_tau1) - dxi [K_a (C_tau,EQ^1/2_u - eta_D C_tau^1/2_u)/delta_a
                                  + 2 (u_q(Cf_u, H_k,u, Re_theta,a, delta*_a) - (due/dxi)(1/ue)_a)]
                                                                                   turbulent

(_a = average of the two stations; _u = upwinded with the H_k-dependent factor of FID eq. (13),
q_u = (1 - eta_up) q1 + eta_up q2; dxi = xi2 - xi1). Which quantities are upwinded and which are
averaged is FID p. 6: in the momentum equation everything is averaged; in the shape equation the
Cf and C_D source terms are upwinded; in the lag equation C_tau^1/2 and C_tau,EQ^1/2 are
upwinded, delta is averaged, and u_q uses an upwinded Cf and H_k with an averaged delta* and
Re_theta; dn/dxi is averaged (DECISIONS.md M3, "H_k-dependent upwinding").
Every function accepts complex values (complex-step derivatives)
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
    h_full: Any = None      # delta*/theta (equals h except in the wake with a dead-air region)
    k_lag: Any = None       # lag-equation rate constant, eta_D, equilibrium u_q
    eta_d: Any = None
    uq: Any = None
    wake: bool = False      # wake station (C_up of FID eq. 13; wake u_q)


def station(theta, dstar, third, ue, reynolds, turbulent: bool, xp=np, cs="fidkowski2022",
            wake: bool = False, hw=0.0, dead_frac=0.0) -> Station:
    """Closure quantities at one station (turbulent C_D uses the station's C_tau).

    Wake stations (``wake=True``) are turbulent; their closures use delta* - h_w, h_w being the
    dead-air thickness (FID eq. 15, BTE89 eqs. 20-22), while the momentum and shape equations use
    the full delta*/theta (``h_full``); Cf = 0; ``dead_frac`` = h_w/h_TE (BTE89 eq. 33)."""
    cs = closure_sets.get(cs)
    h_full = dstar / theta
    ret = reynolds * ue * theta
    if wake:
        dcl = dstar - hw
        h = dcl / theta
        hk = cs.hk_wake(h, xp)
        st = Station(theta, dcl, third, ue, True, h=h, hk=hk, ret=ret, h_full=h_full, wake=True)
        st.hstar = cs.hstar_turb(hk, ret, xp)
        st.cf = 0.0 * h
        st.us = cs.us_wake(st.hstar, h, hk, xp)
        st.delta = cs.delta(theta, hk, dcl, xp)
        st.ctau_eq = cs.ctau_eq_wake(st.hstar, h, hk, st.us, ret, xp)
        st.cd = cs.cd_wake(hk, ret, st.us, third, st.hstar, dead_frac, xp)
        st.k_lag, st.eta_d, st.uq = cs.lag_k(st.us, xp), cs.ETA_D_WAKE, cs.uq_wake(hk, ret, dcl, xp)
        return st
    h = h_full
    hk = cs.hk(h, xp)
    st = Station(theta, dstar, third, ue, turbulent, h=h, hk=hk, ret=ret, h_full=h_full)
    if turbulent:
        st.hstar = cs.hstar_turb(hk, ret, xp)
        st.cf = cs.cf_turb(hk, ret, xp)
        st.us = cs.us(st.hstar, h, hk, xp)
        st.delta = cs.delta(theta, hk, dstar, xp)
        st.ctau_eq = cs.ctau_eq(st.hstar, h, hk, st.us, ret, xp)
        st.cd = cs.cd_turb(hk, ret, st.cf, st.us, third, st.hstar, xp)
        st.k_lag, st.eta_d, st.uq = cs.lag_k(st.us, xp), cs.ETA_D, cs.uq(st.cf, hk, ret, dstar, xp)
    else:
        st.hstar = cs.hstar_lam(hk, ret, xp)
        st.cf = cs.cf_lam(hk, ret, xp)
        st.cd = cs.cd_lam(hk, ret, xp)
    return st


#: C_up of FID eq. (13) on the airfoil and in the wake (published constants of the method)
UPWIND_C_AIRFOIL = 5.0
UPWIND_C_WAKE = 1.0


def upwind_factor(s1: Station, s2: Station, xp=np):
    """eta_up of FID eq. (13): 1 - exp(-(ln|f_Hu|)^2 C_up/H_k2^2)/2, f_Hu = (H_k2 - 1)/(H_k1 - 1).

    0.5 (central) where H_k changes little across the interval, towards 1 (backward Euler) where it
    changes strongly. C_up from the downstream station (wake or airfoil)."""
    f = (s2.hk - 1.0) / (s1.hk - 1.0)
    lf = xp.log(xp.where(_re(f) > 0.0, f, -f))
    c_up = UPWIND_C_WAKE if s2.wake else UPWIND_C_AIRFOIL
    return 1.0 - 0.5 * xp.exp(-lf ** 2 * c_up / s2.hk ** 2)


def _common(s1: Station, s2: Station, xi1, xi2, xp, eta):
    lu = xp.log(s2.ue / s1.ue)
    lx = xp.log(xi2 / xi1)
    h_a = 0.5 * (s1.h_full + s2.h_full)
    f1 = xi1 * s1.cf / s1.theta
    f2 = xi2 * s2.cf / s2.theta
    r1 = xp.log(s2.theta / s1.theta) + (2.0 + h_a) * lu - 0.5 * lx * 0.5 * (f1 + f2)
    src1 = xi1 * (2.0 * s1.cd / s1.hstar - 0.5 * s1.cf) / s1.theta
    src2 = xi2 * (2.0 * s2.cd / s2.hstar - 0.5 * s2.cf) / s2.theta
    src = (1.0 - eta) * src1 + eta * src2
    r2 = xp.log(s2.hstar / s1.hstar) + (1.0 - h_a) * lu - lx * src
    return r1, r2


def lam_rate(s: Station, xp, cs, n_crit):
    return closure_sets.get(cs).dn_dxi(s.hk, s.theta, s.ret, s.third, n_crit, xp)


def turb_lograte(s1: Station, s2: Station, eta, due_dxi, xp, cs):
    """d ln C_tau / d xi over an interval, the common lag equation (FID eq. 8; LNE89 eq. 23 as a
    special case) divided by the averaged delta, with the upwinding of FID p. 6: C_tau^1/2 and
    C_tau,EQ^1/2 upwinded, u_q from the upwinded Cf and H_k and the averaged delta* and Re_theta;
    delta, the rate constant K, eta_D and 1/ue averaged. The set supplies K, eta_D and u_q
    (surface or wake closures)."""
    csm = closure_sets.get(cs)

    def up(a, b):
        return (1.0 - eta) * a + eta * b

    hk_u = up(s1.hk, s2.hk)
    ret_a = 0.5 * (s1.ret + s2.ret)
    ds_a = 0.5 * (s1.dstar + s2.dstar)
    if s2.wake:
        uq = csm.uq_wake(hk_u, ret_a, ds_a, xp)
    else:
        uq = csm.uq(up(s1.cf, s2.cf), hk_u, ret_a, ds_a, xp)
    k_a = 0.5 * (s1.k_lag + s2.k_lag)
    eta_d = 0.5 * (s1.eta_d + s2.eta_d)
    sq_eq = up(xp.sqrt(s1.ctau_eq), xp.sqrt(s2.ctau_eq))
    sq_c = up(xp.sqrt(s1.third), xp.sqrt(s2.third))
    delta_a = 0.5 * (s1.delta + s2.delta)
    inv_ue = 0.5 * (1.0 / s1.ue + 1.0 / s2.ue)
    return k_a * (sq_eq - eta_d * sq_c) / delta_a + 2.0 * (uq - due_dxi * inv_ue)


def laminar_interval(s1: Station, s2: Station, xi1, xi2, xp=np, cs="fidkowski2022", n_crit=9.0):
    r1, r2 = _common(s1, s2, xi1, xi2, xp, upwind_factor(s1, s2, xp))
    dxi = xi2 - xi1
    r3 = s2.third - s1.third - dxi * 0.5 * (lam_rate(s1, xp, cs, n_crit) + lam_rate(s2, xp, cs, n_crit))
    return xp.stack([r1, r2, r3])


def turbulent_interval(s1: Station, s2: Station, xi1, xi2, xp=np, cs="fidkowski2022"):
    eta = upwind_factor(s1, s2, xp)
    r1, r2 = _common(s1, s2, xi1, xi2, xp, eta)
    dxi = xi2 - xi1
    due = (s2.ue - s1.ue) / dxi
    r3 = xp.log(s2.third / s1.third) - dxi * turb_lograte(s1, s2, eta, due, xp, cs)
    return xp.stack([r1, r2, r3])


ONSET_BISECTIONS = 40


_BISECT_CACHE: dict = {}


def _bisect_real(s1: Station, theta2, dstar2, ue2, dxi, reynolds, n_crit, r1, csm):
    """The bisection of :func:`transition_onset` in NumPy on the real parts (its iterates carry no
    derivative; only the sign of the real part of R_t steers them), cached on those real inputs:
    a complex-step Jacobian evaluates the same onset again with perturbations that leave the real
    parts unchanged (speed; the golden values check that the result is unchanged)."""
    vals = [np.real(np.asarray(v)).astype(float) for v in
            (s1.theta, s1.dstar, s1.third, s1.ue, theta2, dstar2, ue2, dxi, r1)]
    key = (csm.NAME, float(reynolds), float(n_crit)) + tuple((v.shape, v.tobytes()) for v in vals)
    hit = _BISECT_CACHE.get(key)
    if hit is not None:
        return hit
    th1, ds1, n1, u1, th2, ds2, u2, dx, rr1 = vals

    def rt(w):
        th = th1 + w * (th2 - th1)
        ds = ds1 + w * (ds2 - ds1)
        u = u1 + w * (u2 - u1)
        hk = csm.hk(ds / th, np)
        rate = csm.dn_dxi(hk, th, reynolds * u * th, n_crit, n_crit, np)
        return n_crit - n1 - w * dx * 0.5 * (rr1 + rate)

    lo, hi = 0.0 * th1, 0.0 * th1 + 1.0
    for _ in range(ONSET_BISECTIONS):
        mid = 0.5 * (lo + hi)
        pos = rt(mid) > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    w0 = 0.5 * (lo + hi)
    if len(_BISECT_CACHE) > 4096:
        _BISECT_CACHE.clear()
    _BISECT_CACHE[key] = w0
    return w0


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

    csm = closure_sets.get(cs)

    def rt(w):
        # the laminar rate at the onset needs only H_k, theta and Re_theta: the same operations as
        # station() for those, without the closures the rate does not use (speed; bit-identical)
        th = s1.theta + w * (theta2 - s1.theta)
        ds = s1.dstar + w * (dstar2 - s1.dstar)
        u = s1.ue + w * (ue2 - s1.ue)
        hk = csm.hk(ds / th, xp)
        rate = csm.dn_dxi(hk, th, reynolds * u * th, n_crit, n_crit, xp)
        return n_crit - s1.third - w * dxi * 0.5 * (r1 + rate)

    zero = 0.0 * s1.theta
    r_lo, r_hi = rt(zero), rt(zero + 1.0)
    if xp is np:
        w0 = _bisect_real(s1, theta2, dstar2, ue2, dxi, reynolds, n_crit, r1, csm)
    else:
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


def transition_interval(theta1, dstar1, n1, ue1, theta2, dstar2, ctau2, ue2, xi1, xi2, reynolds,
                        n_crit, xp=np, forced_w=None, cs="fidkowski2022"):
    """Residuals of an interval containing transition onset (DG87 Fig. 5; FID eqs. 16-18).

    The state at the onset is interpolated linearly between the stations; the laminar
    subinterval ends with n = N_crit, the turbulent one starts from the set's C_tau at transition;
    the two subintervals' residuals are added (FID eq. 16). The third residual is the lag equation
    on the turbulent subinterval (the onset itself is defined by the amplification residual).
    ``forced_w``: onset fraction of a forced trip instead of the amplification criterion.
    Returns (residuals, w).
    """
    dxi = xi2 - xi1
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
    d2 = (1.0 - w) * dxi
    xt = xi1 + w * dxi
    a1, a2 = _common(s1, stl, xi1, xt, xp, upwind_factor(s1, stl, xp))
    eta = upwind_factor(stt, s2, xp)
    b1, b2 = _common(stt, s2, xt, xi2, xp, eta)
    due = (s2.ue - stt.ue) / xp.where(_re(d2) > 1e-12 * _re(dxi), d2, 1e-12 * dxi)
    r3 = xp.log(s2.third / stt.third) - d2 * turb_lograte(stt, s2, eta, due, xp, cs)
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
