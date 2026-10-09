"""Exact laminar similarity relations from the Falkner-Skan equation (reference for closures, M3).

Falkner-Skan equation in Hartree's form,

    f''' + f f'' + beta (1 - f'^2) = 0,   f(0) = f'(0) = 0,   f'(inf) = 1,

with u/ue = f'(eta), eta = y sqrt((m + 1) ue / (2 nu x)), beta = 2m/(m + 1). With
theta_eta = int f'(1 - f') d eta, dstar_eta = int (1 - f') d eta, thetastar_eta = int f'(1 - f'^2) d eta,
the closure groups are scale-free:

    H = dstar_eta/theta_eta,  H* = thetastar_eta/theta_eta,
    Re_theta Cf/2 = theta_eta f''(0),  Re_theta 2 C_D/H* = 2 theta_eta int f''^2 d eta / H*

(C_D = (1/(rho ue^3)) int tau du/dy dy, laminar). The attached branch is solved by shooting on
f''(0), from stagnation flow (beta = 1, m = 1) to separation (beta = -0.1988). Print the table with
``python -m tests.diagnostics.falkner_skan``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ETA_MAX = 14.0


def _rhs(beta):
    def f(_eta, y):
        # y = [f, f', f'', int f'(1-f'), int (1-f'), int f'(1-f'^2), int f''^2]
        return [y[1], y[2], -y[0] * y[2] - beta * (1.0 - y[1] ** 2),
                y[1] * (1.0 - y[1]), 1.0 - y[1], y[1] * (1.0 - y[1] ** 2), y[2] ** 2]
    return f


def _shoot(beta, s, eta_max=ETA_MAX):
    # On the attached branch f' rises monotonically to 1: an overshoot (f' > 1) means f''(0) too
    # large, a turn-back (f'' < 0 while f' < 1) means f''(0) too small.
    def over(_e, y):
        return y[1] - 1.0 - 1e-9
    over.terminal, over.direction = True, 1

    def back(_e, y):
        return y[2] + 0.0 * y[1]
    back.terminal, back.direction = True, -1
    return solve_ivp(_rhs(beta), (0.0, eta_max), [0, 0, s, 0, 0, 0, 0], method="DOP853",
                     rtol=1e-12, atol=1e-13, events=(over, back))


def solve(beta: float) -> dict:
    """Attached Falkner-Skan solution for one beta (beta >= -0.1988)."""
    def g(s):
        sol = _shoot(beta, s)
        if sol.status == 1:
            return 1.0 if len(sol.t_events[0]) else -1.0
        return sol.y[1, -1] - 1.0
    lo = 1e-7 if beta < 0 else 1e-3
    s = brentq(g, lo, 3.0, xtol=1e-14, rtol=1e-14)
    y = _shoot(beta, s).y[:, -1]
    th, ds, ts, diss = y[3], y[4], y[5], y[6]
    hstar = ts / th
    return dict(beta=beta, fpp0=s, h=ds / th, hstar=hstar, ret_cf2=th * s,
                ret_2cd_hstar=2.0 * th * diss / hstar, m=beta / (2.0 - beta))


def branch(betas=None) -> list[dict]:
    if betas is None:
        betas = np.concatenate([np.linspace(1.6, 0.0, 33), np.linspace(-0.01, -0.1988, 40),
                                [-0.19882, -0.19883, -0.198834, -0.198836, -0.1988372]])
    return [solve(float(b)) for b in betas]


def table(h_values=None) -> dict:
    """Exact relations interpolated (cubic in beta, monotonic in H) at the given H values."""
    from scipy.interpolate import PchipInterpolator
    sols = branch()
    h = np.array([d["h"] for d in sols])
    order = np.argsort(h)
    h = h[order]
    if h_values is None:
        h_values = np.linspace(2.2, 4.0, 15)
    out = {"h": np.asarray(h_values, float)}
    for k in ("ret_cf2", "hstar", "ret_2cd_hstar", "beta", "m"):
        v = np.array([d[k] for d in sols])[order]
        out[k] = PchipInterpolator(h, v)(out["h"])
    return out


def main() -> None:
    for b, ref in ((0.0, "Blasius: f''(0) 0.4696, H 2.591"), (1.0, "Hiemenz: f''(0) 1.2326, H 2.216"),
                   (-0.1988372, "near separation, beta_s = -0.19884: f''(0) -> 0, H -> 4.03")):
        d = solve(b)
        print(f"beta {b:+.4f}: f''(0) {d['fpp0']:.5f}, H {d['h']:.4f}, H* {d['hstar']:.4f}, "
              f"Re_theta Cf/2 {d['ret_cf2']:.5f}, Re_theta 2CD/H* {d['ret_2cd_hstar']:.5f}   ({ref})")
    t = table()
    print("\n   H     beta     m       Re_th Cf/2   H*       Re_th 2CD/H*")
    for i in range(len(t["h"])):
        print(f"  {t['h'][i]:.3f}  {t['beta'][i]:+.4f}  {t['m'][i]:+.4f}  {t['ret_cf2'][i]:.5f}     "
              f"{t['hstar'][i]:.5f}  {t['ret_2cd_hstar'][i]:.5f}")


if __name__ == "__main__":
    main()
