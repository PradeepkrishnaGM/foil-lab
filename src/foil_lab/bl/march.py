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


def _admissible(x, direct: bool) -> bool:
    """theta and delta* (direct) or ue (inverse) positive; in direct mode H > 1.02."""
    if not np.all(x[:2] > 0):
        return False
    return bool(x[1] > 1.02 * x[0]) if direct else True


def _newton(fun, x0, direct=True, tol=1e-12, max_iter=40):
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
            if _admissible(x + lam * dx, direct):
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


def march(xi, reynolds, ue=None, dstar=None, n_crit=9.0, xi_trip=None, bias=0.5,
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
            return eq.laminar_interval(s1, s2, dxi, bias=bias, cs=cs, n_crit=n_crit)

        def turb_res(x, prev=prev, dxi=dxi, unpack=unpack):
            t, d, c, u = unpack(x)
            s1 = eq.station(*prev, reynolds, True, cs=cs)
            s2 = eq.station(t, d, c, u, reynolds, True, cs=cs)
            return eq.turbulent_interval(s1, s2, dxi, bias=bias, cs=cs)

        def trans_res(x, forced=None, prev=prev, dxi=dxi, unpack=unpack):
            t, d, c, u = unpack(x)
            return eq.transition_interval(*prev, t, d, c, u, dxi, reynolds, n_crit, bias=bias,
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
