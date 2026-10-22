"""Checkpoint C1: the new boundary layer on XFOIL's own state (no coupling).

* direct: XFOIL's edge velocity prescribed; compare theta, delta*, H, Cf, transition;
* inverse: XFOIL's delta* prescribed; compare the resulting edge velocity (plus theta, H).

Stations are XFOIL's dump points; xi is measured from the stagnation point, where XFOIL's signed
Ue/Vinf changes sign (linear interpolation in s). XFOIL's Cf is divided by ue^2 (it is relative to
the freestream dynamic pressure). XFOIL transition locations are those of the polar file of the
same run (Top_Xtr / Bot_Xtr). Print with ``python -m tests.diagnostics.c1_compare``.
"""

from __future__ import annotations

import warnings

import numpy as np

from foil_lab.bl.march import march
from tests.diagnostics.bl_compare import DATA, dump_name, read_dump

RE = 1e6
#: XFOIL 6.99 polar of the same run (canonical NACA 2412, PANE 160, Re 1e6, N_crit 9): x/c
XFOIL_XTR = {0.0: {"upper": 0.6571, "lower": 0.6806}, 5.0: {"upper": 0.2932, "lower": 0.9999}}
STATIONS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def branches(alpha: float):
    """{'upper'|'lower': (xi, x, ue, delta*, theta, Cf_local, H)} from the committed dump."""
    air, _ = read_dump(DATA / dump_name(alpha))
    s, x, ue = air[:, 0], air[:, 1], air[:, 3]
    k = int(np.flatnonzero(np.sign(ue[1:]) != np.sign(ue[:-1]))[0])
    s_stag = s[k] + (s[k + 1] - s[k]) * ue[k] / (ue[k] - ue[k + 1])
    out = {}
    for lab, idx in (("upper", np.arange(k, -1, -1)), ("lower", np.arange(k + 1, len(air)))):
        a = air[idx]
        ue_b = np.abs(a[:, 3])
        out[lab] = (np.abs(a[:, 0] - s_stag), a[:, 1], ue_b, a[:, 4], a[:, 5], a[:, 6] / ue_b ** 2,
                    a[:, 7])
    return out


def run(alpha: float, cs: str = "fidkowski2022") -> dict:
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lab, (xi, x, ue, ds, th, cf, h) in branches(alpha).items():
            d = march(xi, RE, ue=ue, n_crit=9.0, cs=cs)
            # inverse from x/c 0.2 (XFOIL's state there; n from the direct march), where the direct
            # march approaches laminar separation, transition and the trailing edge
            k0 = int(np.argmin(np.abs(x - 0.2)))
            st = (th[k0], ds[k0], d.third[k0], ue[k0], bool(d.turbulent[k0]))
            inv_part = march(xi[k0:], RE, dstar=ds[k0:], n_crit=9.0, start_state=st, cs=cs)
            inv = _prepend(inv_part, k0, ue)
            # transition forced at XFOIL's location (isolates the turbulent part)
            xt = XFOIL_XTR[alpha][lab]
            xi_t = float(np.interp(xt, x, xi)) if xt < 0.999 else None
            tr = march(xi, RE, ue=ue, n_crit=1e9 if xi_t is not None else 9.0, xi_trip=xi_t, cs=cs)
            res[lab] = dict(xi=xi, x=x, ue=ue, dstar=ds, theta=th, cf=cf, h=h, direct=d, inverse=inv,
                            tripped=tr, xtr_xfoil=XFOIL_XTR[alpha][lab])
    return res


def _prepend(part, k0, ue):
    """Inverse result on the whole branch: XFOIL's ue upstream of the start station (not solved)."""
    import dataclasses
    full_ue = np.concatenate([ue[:k0], part.ue])
    return dataclasses.replace(part, ue=full_ue)


def _xtr_inv(inv, r):
    k0 = len(r["x"]) - len(inv.theta)
    if inv.xi_transition is None:
        return float("nan")
    return float(np.interp(inv.xi_transition, r["xi"], r["x"]))


def _xtr(m, x, xi):
    if m.xi_transition is None:
        return float("nan")
    return float(np.interp(m.xi_transition, xi, x))


def summary(alpha: float, res: dict | None = None, cs: str = "fidkowski2022") -> dict:
    res = res or run(alpha, cs)
    out = {}
    for lab, r in res.items():
        d, inv = r["direct"], r["inverse"]
        out[lab] = dict(
            te_dstar_ratio=d.dstar[-1] / r["dstar"][-1], te_theta_ratio=d.theta[-1] / r["theta"][-1],
            te_h_diff=d.h[-1] - r["h"][-1], max_h_diff=float(np.max(np.abs(d.h - r["h"]))),
            xtr=_xtr(d, r["x"], r["xi"]), xtr_xfoil=r["xtr_xfoil"],
            inverse_max_ue_diff=float(np.max(np.abs(inv.ue - r["ue"])[1:])),
            inverse_xtr=_xtr_inv(inv, r),
            inverse_te_theta_ratio=inv.theta[-1] / r["theta"][-1],
            inverse_te_h_diff=inv.h[-1] - r["h"][-1],
            tripped_te_dstar_ratio=r["tripped"].dstar[-1] / r["dstar"][-1],
            tripped_te_theta_ratio=r["tripped"].theta[-1] / r["theta"][-1],
            tripped_te_h_diff=r["tripped"].h[-1] - r["h"][-1],
            inverse_te_ue_diff=float(inv.ue[-1] - r["ue"][-1]),
            failed_direct=list(d.failed), failed_inverse=list(inv.failed))
    return out


def main(cs: str = "fidkowski2022") -> None:
    print(f"closure set: {cs}")
    for alpha in (0.0, 5.0):
        res = run(alpha, cs)
        sm = summary(alpha, res)
        print(f"\n===== alpha {alpha:g} =====")
        for lab, r in res.items():
            s = sm[lab]
            print(f" {lab}: transition x/c new {s['xtr']:.3f}, XFOIL {s['xtr_xfoil']:.3f}; TE delta* ratio "
                  f"{s['te_dstar_ratio']:.3f}, theta ratio {s['te_theta_ratio']:.3f}, H diff {s['te_h_diff']:+.3f}; "
                  f"inverse (from x/c 0.2): max |ue diff| {s['inverse_max_ue_diff']:.4f}, TE ue diff {s['inverse_te_ue_diff']:+.4f}, transition {s['inverse_xtr']:.3f}; "
                  f"failed direct {s['failed_direct'][:5]} inverse {s['failed_inverse'][:5]}")
            print(f"   inverse at the TE: theta ratio {s['inverse_te_theta_ratio']:.3f}, H diff {s['inverse_te_h_diff']:+.3f}")
            print(f"   transition forced at XFOIL's x/c: TE delta* ratio {s['tripped_te_dstar_ratio']:.3f}, "
                  f"theta ratio {s['tripped_te_theta_ratio']:.3f}, H diff {s['tripped_te_h_diff']:+.3f}")
            d, inv = r["direct"], r["inverse"]
            print("   x/c   | d* XF     d* new   d* trip  | th XF     th new   | H XF  H new H trip | Cf XF     Cf new   | ue XF  ue inv")
            for xs in STATIONS:
                i = int(np.argmin(np.abs(r["x"] - xs)))
                t = r["tripped"]
                print(f"   {r['x'][i]:.3f} | {r['dstar'][i]:.6f} {d.dstar[i]:.6f} {t.dstar[i]:.6f} | {r['theta'][i]:.6f} {d.theta[i]:.6f} |"
                      f" {r['h'][i]:.2f} {d.h[i]:.2f} {t.h[i]:.2f} | {r['cf'][i]:+.6f} {d.cf[i]:+.6f} | {r['ue'][i]:.4f} {inv.ue[i]:.4f}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
