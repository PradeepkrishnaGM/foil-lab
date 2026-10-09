"""Station-by-station march of the discrete boundary-layer equations along one branch.

Two modes, used for diagnosis (checkpoint C1) and for the cold start of the coupled solver:

* direct: ue prescribed at every station, unknowns (theta, delta*, n or C_tau);
* inverse: delta* prescribed, unknowns (theta, ue, n or C_tau).

The branch starts at the stagnation point with the Hiemenz similarity state of DG87 eqs. (33)-(34)
(DECISIONS.md M3, item 3). Each station is solved by Newton's method on the three interval
residuals of :mod:`foil_lab.bl.equations`; the 3x3 Jacobian is obtained by complex-step
differentiation of those residuals (never hand-derived). Transition: an interval becomes the
transition interval when the laminar amplification at its downstream station reaches N_crit, or
when it contains a forced trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import closures as cl
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


def _jac(fun, x, h=1e-30):
    n = len(x)
    r0 = fun(x.astype(complex))
    j = np.empty((len(r0), n))
    for k in range(n):
        xc = x.astype(complex)
        xc[k] += 1j * h
        j[:, k] = fun(xc).imag / h
    return np.real(r0), j


def _newton(fun, x0, tol=1e-12, max_iter=40):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        r, j = _jac(fun, x)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):
            return x, False
        try:
            dx = np.linalg.solve(j, -r)
        except np.linalg.LinAlgError:
            return x, False
        # keep theta, delta*/ue and C_tau positive and H_k above 1 (DECISIONS.md M3, item 4)
        lam = 1.0
        for _k in range(30):
            xn = x + lam * dx
            if np.all(xn[:2] > 0) and xn[1] > 1.02 * xn[0] if fun.mode == "direct" else np.all(xn[:2] > 0):
                break
            lam *= 0.5
        x = x + lam * dx
        if np.max(np.abs(r)) < tol and np.max(np.abs(lam * dx) / np.maximum(np.abs(x), 1e-30)) < 1e-10:
            return x, True
    r, _ = _jac(fun, x)
    return x, bool(np.max(np.abs(r)) < 1e-9)


def march(xi, reynolds, ue=None, dstar=None, n_crit=9.0, xi_trip=None, bias=0.5,
          ue_start=None, start_state=None) -> MarchResult:
    """March one branch. ``xi``: distance from the stagnation point (first station xi[0] > 0).
    Exactly one of ``ue`` (direct) or ``dstar`` (inverse) is prescribed. In the inverse mode the
    first station's ue is ``ue_start`` if given (the similarity relation ue = l xi/(Re theta^2) is
    ill-conditioned with delta* prescribed close to the stagnation point).
    ``start_state`` = (theta, delta*, n or C_tau, ue, turbulent) replaces the similarity start at the
    first station (used to begin a march part-way along a branch)."""
    if (ue is None) == (dstar is None):
        raise ValueError("prescribe exactly one of ue (direct) or dstar (inverse)")
    mode = "direct" if dstar is None else "inverse"
    n = len(xi)
    out_ue = np.zeros(n); th = np.zeros(n); ds = np.zeros(n); third = np.zeros(n)
    turb = np.zeros(n, dtype=bool)
    # start: local Falkner-Skan similarity state, DG87 eqs. (33)-(34) (DECISIONS.md M3, item 3)
    if mode == "direct":
        m0 = float(np.clip(xi[0] / ue[0] * (ue[1] - ue[0]) / (xi[1] - xi[0]), -0.05, 1.0))
        hk0 = cl.similarity_hk(m0)
        out_ue[0] = ue[0]
        th[0] = cl.similarity_theta(xi[0], ue[0], reynolds, m0)
        ds[0] = hk0 * th[0]
    else:
        hk0 = cl.stagnation_hk()
        ds[0] = dstar[0]
        th[0] = ds[0] / hk0
        out_ue[0] = (ue_start if ue_start is not None
                     else cl.ell_fs(hk0) * xi[0] / (reynolds * th[0] ** 2))   # DG87 eq. (33)
    xtr, failed, turbulent = None, [], False
    if start_state is not None:
        th[0], ds[0], third[0], out_ue[0], turbulent = start_state
        turb[0] = turbulent
    for i in range(1, n):
        dxi = xi[i] - xi[i - 1]
        prev = (th[i - 1], ds[i - 1], third[i - 1], out_ue[i - 1])
        trip_here = (not turbulent and xi_trip is not None and xi[i - 1] < xi_trip <= xi[i])

        def unpack(x):
            if mode == "direct":
                return x[0], x[1], x[2], ue[i]
            return x[0], dstar[i], x[2], x[1]

        def lam_res(x):
            t, d, c, u = unpack(x)
            s1 = eq.station(prev[0], prev[1], prev[2], prev[3], reynolds, False)
            s2 = eq.station(t, d, c, u, reynolds, False)
            return eq.laminar_interval(s1, s2, dxi, bias=bias)

        def turb_res(x):
            t, d, c, u = unpack(x)
            s1 = eq.station(prev[0], prev[1], prev[2], prev[3], reynolds, True)
            s2 = eq.station(t, d, c, u, reynolds, True)
            return eq.turbulent_interval(s1, s2, dxi, bias=bias)

        def trans_res(x, forced=None):
            t, d, c, u = unpack(x)
            r, _ = eq.transition_interval(prev[0], prev[1], prev[2], prev[3], t, d, c, u, dxi,
                                          reynolds, n_crit, bias=bias, forced_w=forced)
            return r
        guess = np.array([prev[0], prev[1] if mode == "direct" else prev[3], prev[2]])
        if not turbulent:
            lam_res.mode = mode
            x, ok = _newton(lam_res, guess) if not trip_here else (guess, False)
            if trip_here or (ok and x[2] >= n_crit):
                forced = (xi_trip - xi[i - 1]) / dxi if trip_here else None
                f = (lambda x, fw=forced: trans_res(x, fw))
                f.mode = mode
                st = eq.station(prev[0], prev[1], 1.0, prev[3], reynolds, True)
                g2 = np.array([guess[0], guess[1], cl.ctau_at_transition(float(np.real(st.ctau_eq)))
                               if np.isfinite(st.ctau_eq) else 1e-3])
                x, ok = _newton(f, g2)
                if ok:
                    turbulent = True
                    _, w = eq.transition_interval(prev[0], prev[1], prev[2], prev[3],
                                                  *([x[0], x[1], x[2], ue[i]] if mode == "direct"
                                                    else [x[0], dstar[i], x[2], x[1]]),
                                                  dxi, reynolds, n_crit, bias=bias, forced_w=forced)
                    xtr = float(xi[i - 1] + np.real(w) * dxi)
        else:
            turb_res.mode = mode
            x, ok = _newton(turb_res, guess)
        if not ok:
            failed.append(i)
        t, d, c, u = unpack(x)
        th[i], ds[i], third[i], out_ue[i] = np.real(t), np.real(d), np.real(c), np.real(u)
        turb[i] = turbulent
    h = ds / th
    ret = reynolds * out_ue * th
    cf = np.where(turb, cl.cf_turb(h, np.maximum(ret, 1.0001)), cl.cf_lam(h, ret))
    return MarchResult(np.asarray(xi), out_ue, th, ds, third, turb, h, cf, xtr, failed)

