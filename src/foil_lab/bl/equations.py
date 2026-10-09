"""Discrete integral boundary-layer equations on one interval between two stations (Mach 0).

Equations (DG87 = Drela & Giles 1987; LNE89 = Drela 1989, Lecture Notes in Engineering 54):

* momentum, DG87 eq. (4); shape parameter, DG87 eq. (5);
* laminar: amplification, DG87 eq. (35); turbulent: lag, LNE89 eq. (23);
* transition interval split into a laminar and a turbulent subinterval at an onset point defined
  implicitly by the amplification reaching N_crit (DG87 pp. 1351-1352, Fig. 5), with
  C_tau^1/2 = 0.7 C_tau,EQ^1/2 at onset.

Two-point central differences (DG87 p. 1351) in logarithmic form, so every residual is
dimensionless (scaling in DECISIONS.md, M3):

    R1 = ln(theta2/theta1) + (2 + H)_a ln(ue2/ue1) - dxi (Cf/2)_a / theta_a
    R2 = ln(H*2/H*1) + (1 - H)_a ln(ue2/ue1) - dxi (2C_D/H* - Cf/2)_a / theta_a    (H** = 0)
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

from . import closures as cl


@dataclass
class Station:
    """Secondary quantities of one station."""

    theta: Any
    dstar: Any
    third: Any          # n (laminar) or C_tau (turbulent)
    ue: Any
    turbulent: bool
    h: Any = None
    ret: Any = None
    hstar: Any = None
    cf: Any = None
    cd: Any = None
    us: Any = None
    delta: Any = None
    ctau_eq: Any = None


def station(theta, dstar, third, ue, reynolds, turbulent: bool, xp=np) -> Station:
    """Closure quantities at one station (C_D for turbulent uses the station's C_tau)."""
    h = dstar / theta
    ret = reynolds * ue * theta
    st = Station(theta, dstar, third, ue, turbulent, h=h, ret=ret)
    if turbulent:
        st.hstar = cl.hstar_turb(h, ret, xp)
        st.cf = cl.cf_turb(h, ret, xp)
        st.us = cl.us_slip(st.hstar, h, h)
        st.delta = cl.delta_bl(theta, h, dstar)
        st.ctau_eq = cl.ctau_eq(st.hstar, h, h, st.us)
        st.cd = cl.cd_turb(st.cf, st.us, third)
    else:
        st.hstar = cl.hstar_lam(h, xp)
        st.cf = cl.cf_lam(h, ret, xp)
        st.cd = cl.cd_lam(h, ret, xp)
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


def lam_rate(s: Station, xp):
    return cl.dn_dxi(s.h, s.theta, s.ret, xp)


def turb_lograte(s: Station, due_dxi, xp):
    """d ln C_tau / d xi from LNE89 eq. (23)."""
    return cl.lag_rate(s.third, s.ctau_eq, s.delta, s.dstar, s.cf, s.h, s.ue, due_dxi, xp) / s.third


def laminar_interval(s1: Station, s2: Station, dxi, xp=np, bias=0.5):
    r1, r2 = _common(s1, s2, dxi, xp, bias)
    r3 = s2.third - s1.third - dxi * 0.5 * (lam_rate(s1, xp) + lam_rate(s2, xp))
    return xp.stack([r1, r2, r3])


def turbulent_interval(s1: Station, s2: Station, dxi, xp=np, bias=0.5):
    r1, r2 = _common(s1, s2, dxi, xp, bias)
    due = (s2.ue - s1.ue) / dxi
    g = 0.5 * (turb_lograte(s1, due, xp) + turb_lograte(s2, due, xp))
    r3 = xp.log(s2.third / s1.third) - dxi * g
    return xp.stack([r1, r2, r3])


def transition_onset(s1: Station, s2_lam: Station, dxi, n_crit, xp=np):
    """Onset fraction w = (xi_tr - xi1)/dxi in [0, 1], from the discrete amplification equation
    on the laminar subinterval with the rate interpolated linearly in xi (DECISIONS.md M3):
    N_crit = n1 + dxi w (r1 + w (r2 - r1)/2), solved in closed form."""
    r1, r2 = lam_rate(s1, xp), lam_rate(s2_lam, xp)
    a = 0.5 * dxi * (r2 - r1)
    b = dxi * r1
    c = s1.third - n_crit
    disc = b * b - 4.0 * a * c
    has_root = _re(disc) >= 0.0
    disc_s = xp.where(has_root, disc, 1.0 + 0.0 * disc)
    small = xp.abs(_re(a)) < 1e-14 * (xp.abs(_re(b)) + 1e-300)
    a_safe = xp.where(small, 1.0, a)
    w_quad = (-b + xp.sqrt(disc_s)) / (2.0 * a_safe)
    w_lin = -c / xp.where(_re(b) == 0, 1e-300, b)
    w = xp.where(small, w_lin, w_quad)
    # A provisional Newton state may put the onset outside the interval: then the onset is taken
    # at the nearer end (w = 1 if the amplification does not reach N_crit inside the interval,
    # w = 0 if it was already exceeded). A numerical definition, DECISIONS.md M3.
    w = xp.where(has_root | small, w, 1.0 + 0.0 * w)
    w = xp.where(_re(w) > 1.0, 1.0 + 0.0 * w, w)
    return xp.where(_re(w) < 0.0, 0.0 * w, w)


def _re(x):
    return x.real if hasattr(x, "real") else x


def transition_interval(theta1, dstar1, n1, ue1, theta2, dstar2, ctau2, ue2, dxi, reynolds,
                        n_crit, xp=np, bias=0.5, forced_w=None):
    """Residuals of an interval containing transition onset (DG87 Fig. 5).

    The state at the onset is interpolated linearly between the stations; the two subintervals'
    log-form residuals are added (their sum discretises the whole interval); the third equation
    is the lag equation on the turbulent subinterval starting from C_tau = 0.49 C_tau,EQ.
    ``forced_w``: onset fraction of a forced trip instead of the amplification criterion.
    """
    s1 = station(theta1, dstar1, n1, ue1, reynolds, False, xp)
    s2l = station(theta2, dstar2, n1, ue2, reynolds, False, xp)
    w = transition_onset(s1, s2l, dxi, n_crit, xp) if forced_w is None else forced_w
    th_t = theta1 + w * (theta2 - theta1)
    ds_t = dstar1 + w * (dstar2 - dstar1)
    ue_t = ue1 + w * (ue2 - ue1)
    stl = station(th_t, ds_t, n_crit, ue_t, reynolds, False, xp)
    st_tmp = station(th_t, ds_t, 1.0, ue_t, reynolds, True, xp)
    ct_t = cl.ctau_at_transition(st_tmp.ctau_eq)
    stt = station(th_t, ds_t, ct_t, ue_t, reynolds, True, xp)
    s2 = station(theta2, dstar2, ctau2, ue2, reynolds, True, xp)
    d1, d2 = w * dxi, (1.0 - w) * dxi
    a1, a2 = _common(s1, stl, d1, xp, bias)
    b1, b2 = _common(stt, s2, d2, xp, bias)
    due = (s2.ue - stt.ue) / xp.where(_re(d2) > 1e-12 * _re(dxi), d2, 1e-12 * dxi)
    g = 0.5 * (turb_lograte(stt, due, xp) + turb_lograte(s2, due, xp))
    r3 = xp.log(s2.third / stt.third) - d2 * g
    return xp.stack([a1 + b1, a2 + b2, r3]), w
