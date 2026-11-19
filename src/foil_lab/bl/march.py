"""Station-by-station march of the discrete boundary-layer equations along one branch.

Two modes, used for diagnosis (checkpoint C1) and for the cold start of the coupled solver:

* direct: ue prescribed at every station, unknowns (theta, delta*, n or C_tau);
* inverse: delta* prescribed, unknowns (theta, ue, n or C_tau).

The first station takes the local similarity state of the set's own closures
(:func:`foil_lab.bl.equations.similarity_start`). Each station is solved by Newton's method on the
three interval residuals of :mod:`foil_lab.bl.equations`; the 3x3 Jacobian is obtained by
complex-step differentiation of those residuals (never hand-derived). Transition: an interval
becomes the transition interval when the laminar amplification at its downstream station reaches
N_crit, or when it contains a forced trip. All closures come from the closure set ``cs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import closure_sets
from . import equations as eq


@dataclass
class MarchResult:
    xi: np.ndarray
    ue: np.ndarray
    theta: np.ndarray
    dstar: np.ndarray
    third: np.ndarray            # n (laminar stations) or C_tau (turbulent stations)
    turbulent: np.ndarray        # bool per station
    h: np.ndarray
    cf: np.ndarray
    xi_transition: float | None
    failed: list = field(default_factory=list)   # stations where the local Newton did not converge
    closure_set: str = ""


def _jac(fun, x, h=1e-30):
    n = len(x)
    r0 = fun(x.astype(complex))
    j = np.empty((len(r0), n))
    for k in range(n):
        xc = x.astype(complex)
        xc[k] += 1j * h
        j[:, k] = fun(xc).imag / h
    return np.real(r0), j


def _admissible(x, direct: bool, hw: float = 0.0) -> bool:
    """theta and delta* (direct) or ue (inverse) positive; in direct mode H > 1.02, with H formed
    from delta* - h_w in the wake (``hw``, the dead-air thickness)."""
    if not np.all(x[:2] > 0):
        return False
    return bool(x[1] - hw > 1.02 * x[0]) if direct else True


def _newton(fun, x0, direct=True, tol=1e-12, max_iter=40, hw=0.0):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        r, j = _jac(fun, x)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):
            return x, False
        try:
            dx = np.linalg.solve(j, -r)
        except np.linalg.LinAlgError:
            return x, False
        # step halving keeps the state admissible (DECISIONS.md M3, item 4)
        lam = 1.0
        for _k in range(30):
            if _admissible(x + lam * dx, direct, hw):
                break
            lam *= 0.5
        x = x + lam * dx
        if np.max(np.abs(r)) < tol and np.max(np.abs(lam * dx) / np.maximum(np.abs(x), 1e-30)) < 1e-10:
            return x, True
    r, _ = _jac(fun, x)
    return x, bool(np.max(np.abs(r)) < 1e-9)


def similarity_start_state(xi0, ue0, ue1, xi1, reynolds, cs, m=None):
    """(theta, delta*) of the first station from the local similarity equations, with
    m = (xi/ue) due/dxi from the first two stations, clipped to [-0.05, 1] (DECISIONS.md M3)."""
    if m is None:
        m = float(np.clip(xi0 / ue0 * (ue1 - ue0) / (xi1 - xi0), -0.05, 1.0))

    def f(x):
        return eq.similarity_start(x[0], x[1], ue0, xi0, m, reynolds, cs=cs)
    th0 = np.sqrt(0.3 * xi0 / (reynolds * ue0))
    x, ok = _newton(f, [th0, 2.4 * th0])
    if not ok:
        raise RuntimeError(f"similarity start failed (m = {m:.3f})")
    return x[0], x[1]


def march(xi, reynolds, ue=None, dstar=None, n_crit=9.0, xi_trip=None,
          ue_start=None, start_state=None, cs="fidkowski2022") -> MarchResult:
    """March one branch. ``xi``: distance from the stagnation point (first station xi[0] > 0).
    Exactly one of ``ue`` (direct) or ``dstar`` (inverse) is prescribed. In the inverse mode the
    first station's ue is ``ue_start`` if given, else from the stagnation (m = 1) similarity state.
    ``start_state`` = (theta, delta*, n or C_tau, ue, turbulent) replaces the similarity start at
    the first station (used to begin a march part-way along a branch)."""
    if (ue is None) == (dstar is None):
        raise ValueError("prescribe exactly one of ue (direct) or dstar (inverse)")
    cs = closure_sets.get(cs)
    direct = dstar is None
    n = len(xi)
    out_ue = np.zeros(n); th = np.zeros(n); ds = np.zeros(n); third = np.zeros(n)
    turb = np.zeros(n, dtype=bool)
    xtr, failed, turbulent = None, [], False
    if start_state is not None:
        th[0], ds[0], third[0], out_ue[0], turbulent = start_state
        turb[0] = turbulent
    elif direct:
        out_ue[0] = ue[0]
        th[0], ds[0] = similarity_start_state(xi[0], ue[0], ue[1], xi[1], reynolds, cs)
    else:
        u0 = ue_start if ue_start is not None else 1.0
        t1, d1 = similarity_start_state(xi[0], u0, None, None, reynolds, cs, m=1.0)
        if ue_start is None:     # theta^2 ue ~ xi at a stagnation point: scale to the given delta*
            u0 = u0 * (d1 / dstar[0]) ** 2
            t1, d1 = similarity_start_state(xi[0], u0, None, None, reynolds, cs, m=1.0)
        out_ue[0], ds[0], th[0] = u0, dstar[0], t1 * dstar[0] / d1
    for i in range(1, n):
        dxi = xi[i] - xi[i - 1]
        prev = (th[i - 1], ds[i - 1], third[i - 1], out_ue[i - 1])
        trip_here = (not turbulent and xi_trip is not None and xi[i - 1] < xi_trip <= xi[i])

        def unpack(x, i=i):
            if direct:
                return x[0], x[1], x[2], ue[i]
            return x[0], dstar[i], x[2], x[1]

        def lam_res(x, prev=prev, dxi=dxi, unpack=unpack):
            t, d, c, u = unpack(x)
            s1 = eq.station(*prev, reynolds, False, cs=cs)
            s2 = eq.station(t, d, c, u, reynolds, False, cs=cs)
            return eq.laminar_interval(s1, s2, xi[i - 1], xi[i], cs=cs, n_crit=n_crit)

        def turb_res(x, prev=prev, dxi=dxi, unpack=unpack):
            t, d, c, u = unpack(x)
            s1 = eq.station(*prev, reynolds, True, cs=cs)
            s2 = eq.station(t, d, c, u, reynolds, True, cs=cs)
            return eq.turbulent_interval(s1, s2, xi[i - 1], xi[i], cs=cs)

        def trans_res(x, forced=None, prev=prev, dxi=dxi, unpack=unpack):
            t, d, c, u = unpack(x)
            return eq.transition_interval(*prev, t, d, c, u, xi[i - 1], xi[i], reynolds, n_crit,
                                          forced_w=forced, cs=cs)
        guess = np.array([prev[0], prev[1] if direct else prev[3], prev[2]])
        if not turbulent:
            x, ok = _newton(lam_res, guess, direct) if not trip_here else (guess, False)
            # transition interval: forced trip, n reaching N_crit, or a laminar step without a
            # solution (accepted only if the onset then lies inside the interval)
            if trip_here or not ok or x[2] >= n_crit:
                forced = (xi_trip - xi[i - 1]) / dxi if trip_here else None
                st = eq.station(prev[0], prev[1], 1.0, prev[3], reynolds, True, cs=cs)
                c0 = float(np.real(cs.ctau_transition(st.hk, st.ctau_eq)))
                g2 = np.array([guess[0], guess[1], c0 if np.isfinite(c0) and c0 > 0 else 1e-3])
                xt, okt = _newton(lambda x, fw=forced: trans_res(x, fw)[0], g2, direct)
                w = float(np.real(trans_res(xt, forced)[1])) if okt else 1.0
                if okt and (trip_here or w < 1.0):
                    x, ok, turbulent = xt, True, True
                    xtr = float(xi[i - 1] + w * dxi)
                elif ok:
                    x, ok = xt, okt          # n >= N_crit but no transition solution: report failure
        else:
            x, ok = _newton(turb_res, guess, direct)
        if not ok:
            failed.append(i)
        t, d, c, u = unpack(x)
        th[i], ds[i], third[i], out_ue[i] = np.real(t), np.real(d), np.real(c), np.real(u)
        turb[i] = turbulent
    h = ds / th
    ret = reynolds * out_ue * th
    hk = cs.hk(h)
    with np.errstate(all="ignore"):
        cf = np.where(turb, cs.cf_turb(hk, np.maximum(ret, 1.0001)), cs.cf_lam(hk, ret))
    return MarchResult(np.asarray(xi), out_ue, th, ds, third, turb, h, cf, xtr, failed, cs.NAME)


def march_init(xi, ue_inv, reynolds, n_crit=9.0, xi_trip=None, cs="fidkowski2022",
               hk_max_lam=3.8, hk_max_turb=2.5, guards: bool = False) -> MarchResult:
    """Starting state of the coupled Newton solver on one branch (FID section V.A, p. 9).

    Direct march on the inviscid edge velocity. Where the direct Newton fails or H_k exceeds
    ``hk_max_lam`` / ``hk_max_turb`` (FID: 3.8, 2.5), the station is solved in an inverse mode with
    ue unknown and a prescribed H_k (FID eqs. 20-21, as printed, with Delta H_k = (xi2 - xi1)/theta1):

        laminar:   H_k2 = max(H_k1 + 0.03 Delta H_k, H_k,max,lam)
        turbulent: H_k2 = max(H_k1 - 0.15 Delta H_k, H_k,max,turb)

    If that fails too, theta and delta* are scaled from the previous station by (xi2/xi1)^0.5 with
    the inviscid ue (FID p. 9). Only a starting state: the converged coupled solution does not
    depend on it. ``failed`` lists the stations that needed the inverse mode or the fallback.
    """
    cs = closure_sets.get(cs)
    n = len(xi)
    out_ue = np.array(ue_inv, dtype=float)
    th = np.zeros(n); ds = np.zeros(n); third = np.zeros(n)
    turb = np.zeros(n, dtype=bool)
    th[0], ds[0] = similarity_start_state(xi[0], ue_inv[0], None, None, reynolds, cs, m=1.0)   # next to stagnation
    xtr, failed, turbulent = None, [], False
    for i in range(1, n):
        dxi = xi[i] - xi[i - 1]
        prev = (th[i - 1], ds[i - 1], third[i - 1], out_ue[i - 1])
        trip_here = (not turbulent and xi_trip is not None and xi[i - 1] < xi_trip <= xi[i])
        forced = (xi_trip - xi[i - 1]) / dxi if trip_here else None

        def residual(x, kind, mode, hk_t=None, prev=prev, dxi=dxi, i=i, forced=forced):
            if mode == "direct":
                t, d, c, u = x[0], x[1], x[2], ue_inv[i]
            else:                                   # prescribed H_k: x = (theta, ue, third)
                t, d, c, u = x[0], hk_t * x[0], x[2], x[1]
            if kind == "transition":
                return eq.transition_interval(*prev, t, d, c, u, xi[i - 1], xi[i], reynolds, n_crit,
                                              forced_w=forced, cs=cs)
            tb = kind == "turbulent"
            s1 = eq.station(*prev, reynolds, tb, cs=cs)
            s2 = eq.station(t, d, c, u, reynolds, tb, cs=cs)
            if tb:
                return eq.turbulent_interval(s1, s2, xi[i - 1], xi[i], cs=cs)
            return eq.laminar_interval(s1, s2, xi[i - 1], xi[i], cs=cs, n_crit=n_crit)

        def solve(kind, mode, hk_t=None):
            if kind == "transition":
                st = eq.station(prev[0], prev[1], 1.0, prev[3], reynolds, True, cs=cs)
                c0 = float(np.real(cs.ctau_transition(st.hk, st.ctau_eq)))
                c_guess = c0 if np.isfinite(c0) and c0 > 0 else 1e-3
            else:
                c_guess = prev[2]
            g = np.array([prev[0], prev[1] if mode == "direct" else prev[3], c_guess])
            f = ((lambda x: residual(x, kind, mode, hk_t)[0]) if kind == "transition"
                 else (lambda x: residual(x, kind, mode, hk_t)))
            x, ok = _newton(f, g, direct=(mode == "direct"))
            w = float(np.real(residual(x, kind, mode, hk_t)[1])) if (ok and kind == "transition") else None
            return x, ok, w

        def to_state(x, mode, hk_t=None):
            if mode == "direct":
                return x[0], x[1], x[2], ue_inv[i]
            return x[0], hk_t * x[0], x[2], x[1]

        # 1. direct
        mode, hk_t = "direct", None
        kind = "turbulent" if turbulent else "laminar"
        x, ok, _ = solve(kind, mode) if not trip_here else (None, False, None)
        new_turb = turbulent
        if not turbulent and (trip_here or not ok or x[2] >= n_crit):
            xt, okt, w = solve("transition", mode)
            if okt and (trip_here or w < 1.0):
                x, ok, new_turb, xtr_i = xt, True, True, float(xi[i - 1] + w * dxi)
            else:
                xtr_i = None
        else:
            xtr_i = None
        hk_max = hk_max_turb if new_turb else hk_max_lam
        # with the guards, a turbulent station after a separated one (H_k above the limit) stays
        # in the inverse mode until the prescribed H_k has come down to the limit: the direct
        # Newton can otherwise land on a root with an H_k far below it, a reattachment within one
        # station (our choice; DECISIONS.md v2.1, T2)
        relaxing = guards and turbulent and new_turb and prev[1] / prev[0] > hk_max_turb
        # the H_k switch is not applied on the transition interval itself, whose downstream
        # station still carries the laminar H_k (DECISIONS.md M3, C2); with the guards, a
        # transition interval whose downstream H_k jumps above both the upstream H_k and the
        # laminar limit is a spurious root (H_k in the hundreds was seen) and goes to the inverse
        # mode (our choice; DECISIONS.md v2.1, T2)
        jump = (guards and ok and new_turb and not turbulent
                and x[1] / x[0] > max(prev[1] / prev[0], hk_max_lam))
        if ok and not relaxing and not jump and (x[1] / x[0] <= hk_max or (new_turb and not turbulent)):
            state = to_state(x, mode)
        else:
            # 2. inverse with prescribed H_k (FID eqs. 20-21)
            failed.append(i)
            hk1 = prev[1] / prev[0]
            dhk = dxi / prev[0]
            hk_t = (max(hk1 + 0.03 * dhk, hk_max_lam) if not new_turb
                    else max(hk1 - 0.15 * dhk, hk_max_turb))
            kind = "transition" if (new_turb and not turbulent) else ("turbulent" if turbulent else "laminar")
            xi_, ok, w = solve(kind, "inverse", hk_t)
            if ok and kind == "laminar" and xi_[2] >= n_crit:
                xt, okt, w = solve("transition", "inverse", hk_t)
                if okt and w < 1.0:
                    xi_, new_turb, xtr_i = xt, True, float(xi[i - 1] + w * dxi)
            if ok:
                state = to_state(xi_, "inverse", hk_t)
            else:
                # 3. scaling fallback (FID p. 9)
                fac = np.sqrt(xi[i] / xi[i - 1])
                c_fb = prev[2]
                if new_turb and not turbulent:      # third variable becomes C_tau
                    stt = eq.station(prev[0], prev[1], 1.0, prev[3], reynolds, True, cs=cs)
                    c_fb = float(np.real(cs.ctau_transition(stt.hk, stt.ctau_eq)))
                state = (prev[0] * fac, prev[1] * fac, c_fb, ue_inv[i])
        if new_turb and not turbulent:
            xtr = xtr_i if xtr_i is not None else float(xi[i])
        turbulent = new_turb
        th[i], ds[i], third[i], out_ue[i] = (float(np.real(v)) for v in state)
        turb[i] = turbulent
    h = ds / th
    ret = reynolds * out_ue * th
    hk = cs.hk(h)
    with np.errstate(all="ignore"):
        cf = np.where(turb, cs.cf_turb(hk, np.maximum(ret, 1.0001)), cs.cf_lam(hk, ret))
    return MarchResult(np.asarray(xi), out_ue, th, ds, third, turb, h, cf, xtr, failed, cs.NAME)


def wake_hk_inverse(hk1, dhk, iterations=6):
    """Prescribed H_k at the next wake station in the inverse mode of the starting march (FID
    eq. 22, as printed): Newton iterations on H_k2 + 0.03 Delta H_k (H_k2 - 1)^3 = H_k1, started
    from H_k2 = H_k1, six iterations (FID p. 9)."""
    hk2 = hk1
    for _ in range(iterations):
        hk2 = hk2 - (hk2 + 0.03 * dhk * (hk2 - 1.0) ** 3 - hk1) / (1.0 + 0.09 * dhk * (hk2 - 1.0) ** 2)
    return hk2


def march_wake(xi, ue_inv, hw, dead_frac, start, reynolds, cs="fidkowski2022",
               hk_max=2.5) -> MarchResult:
    """Starting state of one wake (FID section V.A, p. 9). ``start`` = (theta, delta*, C_tau) of
    the first wake station (FID eq. 14); ``hw``, ``dead_frac``: dead-air thickness and h_w/h_TE
    per station (FID eq. 15, BTE89 eqs. 20-22, 33). Direct march on the inviscid edge velocity;
    where the direct Newton fails or the closure H_k = (delta* - h_w)/theta exceeds ``hk_max``
    (the turbulent limit of FID p. 9; FID gives none of its own for the wake), the inverse mode
    with H_k from FID eq. (22); if that fails too, the extrapolation of FID p. 9, theta2 = theta1,
    delta*2 = (delta*1 + theta1 r)/(1 + r), r = (xi2 - xi1)/(10 delta*1), applied to delta* - h_w
    (our reading), with the inviscid ue. ``failed`` lists the stations that needed either."""
    cs = closure_sets.get(cs)
    n = len(xi)
    out_ue = np.array(ue_inv, dtype=float)
    th = np.zeros(n); ds = np.zeros(n); third = np.zeros(n)
    th[0], ds[0], third[0] = start
    failed = []
    for i in range(1, n):
        prev = (th[i - 1], ds[i - 1], third[i - 1], out_ue[i - 1])

        def res(t, d, c, u, prev=prev, i=i):
            s1 = eq.station(*prev, reynolds, True, cs=cs, wake=True, hw=hw[i - 1], dead_frac=dead_frac[i - 1])
            s2 = eq.station(t, d, c, u, reynolds, True, cs=cs, wake=True, hw=hw[i], dead_frac=dead_frac[i])
            return eq.turbulent_interval(s1, s2, xi[i - 1], xi[i], cs=cs)

        x, ok = _newton(lambda x: res(x[0], x[1], x[2], ue_inv[i]), np.array(prev[:3]), True, hw=hw[i])
        if ok and (x[1] - hw[i]) / x[0] <= hk_max:
            th[i], ds[i], third[i] = (float(np.real(v)) for v in x)
            continue
        failed.append(i)
        hk_t = wake_hk_inverse((prev[1] - hw[i - 1]) / prev[0], (xi[i] - xi[i - 1]) / prev[0])
        g = np.array([prev[0], prev[3], prev[2]])
        x, ok = _newton(lambda x: res(x[0], hk_t * x[0] + hw[i], x[2], x[1]), g, False)
        if ok:
            th[i], ds[i], third[i], out_ue[i] = (float(np.real(v)) for v in (x[0], hk_t * x[0] + hw[i], x[2], x[1]))
            continue
        dcl = prev[1] - hw[i - 1]
        r = (xi[i] - xi[i - 1]) / (10.0 * dcl)
        th[i], ds[i], third[i] = prev[0], (dcl + prev[0] * r) / (1.0 + r) + hw[i], prev[2]
    h = ds / th
    return MarchResult(np.asarray(xi), out_ue, th, ds, third, np.ones(n, dtype=bool), h, np.zeros(n), None,
                       failed, cs.NAME)
