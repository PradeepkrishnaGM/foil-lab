"""Every difference between the closure sets, evaluated on XFOIL's dumped states (M3).

For each closure set of :mod:`foil_lab.bl.closure_sets`:

1. Cf and H* evaluated on XFOIL's dumped (H, Re_theta) in laminar and turbulent regions, against
   the dumped values (Cf divided by ue^2);
2. transition: the set's amplification rate integrated along XFOIL's dumped states (trapezoid,
   implicit in n where the rate depends on n), against XFOIL's transition locations;
3. the C1 march on XFOIL's ue (free transition and transition forced at XFOIL's location):
   transition and trailing-edge delta*, theta, H;
4. formula-level differences as functions of H_k: critical Re_theta, amplification rate,
   C_tau^1/2 / C_tau,EQ^1/2 at transition, lag rate constant K.

Print with ``python -m tests.diagnostics.closure_set_compare``.
"""

from __future__ import annotations

import warnings

import numpy as np

from foil_lab.bl import closure_sets, drela1987, fidkowski2022
from tests.diagnostics import c1_compare
from tests.diagnostics.bl_compare import DATA, dump_name, read_dump

RE = 1e6
XTR = {0.0: (0.6571, 0.6806), 5.0: (0.2932, 0.9999)}


def dump_points():
    """Rows (H, Re_theta, Cf_local, H*, laminar flag) of both dumps, airfoil only, x/c 0.02-0.98."""
    rows = []
    for alpha, (xu, xl) in XTR.items():
        air, _ = read_dump(DATA / dump_name(alpha))
        x, ue, th, cf, h, hs = (air[:, 1], np.abs(air[:, 3]), air[:, 5], air[:, 6] / air[:, 3] ** 2,
                                air[:, 7], air[:, 8])
        upper = np.arange(len(x)) < np.argmin(x)
        lam = np.where(upper, x < xu, x < xl)
        k = (x > 0.02) & (x < 0.98)
        rows.append(np.column_stack([h, RE * ue * th, cf, hs, lam])[k])
    return np.vstack(rows)


def closure_on_dump(cs):
    cs = closure_sets.get(cs)
    p = dump_points()
    out = {}
    for regime, sel, bands in (("laminar", p[:, 4] == 1, ((2.2, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.2))),
                               ("turbulent", p[:, 4] == 0, ((1.3, 1.45), (1.45, 1.6), (1.6, 2.2)))):
        for lo, hi in bands:
            k = sel & (p[:, 0] >= lo) & (p[:, 0] < hi)
            if not k.any():
                continue
            hk, ret = cs.hk(p[k, 0]), p[k, 1]
            if regime == "laminar":
                cf, hs = cs.cf_lam(hk, ret), cs.hstar_lam(hk, ret)
            else:
                cf, hs = cs.cf_turb(hk, ret), cs.hstar_turb(hk, ret)
            out[(regime, lo, hi)] = (int(k.sum()), float(np.mean(cf / p[k, 2])),
                                     float(np.max(np.abs(cf / p[k, 2] - 1.0))),
                                     float(np.mean(hs / p[k, 3])), float(np.max(np.abs(hs / p[k, 3] - 1.0))))
    return out


def transition_on_dump(cs, n_crit=9.0):
    """x/c where the set's amplification, integrated on XFOIL's states, reaches N_crit."""
    cs = closure_sets.get(cs)
    out = {}
    for alpha in XTR:
        for lab, (xi, x, ue, ds, th, cf, h) in c1_compare.branches(alpha).items():
            hk, ret = cs.hk(h), RE * ue * th
            n = 0.0
            r_prev = cs.dn_dxi(hk[0], th[0], ret[0], n, n_crit)
            xt = float("nan")
            for k in range(1, len(xi)):
                d = xi[k] - xi[k - 1]
                n2 = n
                for _ in range(20):                          # implicit trapezoid (rate may depend on n)
                    n2 = n + 0.5 * d * (r_prev + cs.dn_dxi(hk[k], th[k], ret[k], n2, n_crit))
                if n2 >= n_crit:
                    xt = float(x[k - 1] + (x[k] - x[k - 1]) * (n_crit - n) / (n2 - n))
                    break
                n, r_prev = n2, cs.dn_dxi(hk[k], th[k], ret[k], n2, n_crit)
            out[(alpha, lab)] = xt
    return out


def formula_table():
    hk = np.array([1.4, 1.8, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0])
    theta, ret = 1.0, 500.0
    rows = []
    for h in hk:
        d_l = float(drela1987.log10_retheta0(h))
        hh = 1.0 / (h - 1.0)
        f_l = 2.492 * hh ** 0.43 + 0.7 * (1.0 + np.tanh(14.0 * hh - 9.24))    # FID eq. (74), ramp centre
        d_r = float(drela1987.dn_dretheta(h) * 0.5 * (drela1987.m_fs(h) + 1.0) * drela1987.ell_fs(h))
        r_big = 1e5                                          # well above both critical values
        f_r = float(fidkowski2022.dn_dxi(h, theta, r_big, 0.0, 9.0))
        d_r_big = float(drela1987.dn_dxi(h, theta, r_big))
        rows.append((h, d_l, f_l, d_r_big, f_r, 0.7, 1.8 * np.exp(-3.3 / (h - 1.0))))
    return rows


def main() -> None:
    names = list(closure_sets.CLOSURE_SETS)
    print("1. Closures on XFOIL's dumped states: mean ratio closure/XFOIL (max |ratio - 1|)")
    res = {n: closure_on_dump(n) for n in names}
    for key in res[names[0]]:
        regime, lo, hi = key
        cells = []
        for n in names:
            cnt, cfm, cfx, hsm, hsx = res[n][key]
            cells.append(f"{n}: Cf {cfm:.3f} ({cfx:.3f}), H* {hsm:.4f} ({hsx:.4f})")
        print(f"   {regime:9s} H {lo:.2f}-{hi:.2f} n={cnt:3d} | " + " | ".join(cells))
    print("\n2. Transition from each set's amplification integrated on XFOIL's states (x/c)")
    tr = {n: transition_on_dump(n) for n in names}
    for key in tr[names[0]]:
        a, lab = key
        print(f"   alpha {a:g} {lab:5s}: XFOIL {XTR[a][0 if lab == 'upper' else 1]:.3f} | "
              + " | ".join(f"{n} {tr[n][key]:.3f}" for n in names))
    print("\n3. C1 march on XFOIL's ue: transition; TE delta*, theta ratio, H diff; forced at XFOIL's transition")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in names:
            for a in (0.0, 5.0):
                sm = c1_compare.summary(a, cs=n)
                for lab, s in sm.items():
                    print(f"   {n:13s} alpha {a:g} {lab:5s}: xtr {s['xtr']:.3f} (XFOIL {s['xtr_xfoil']:.3f}); "
                          f"TE {s['te_dstar_ratio']:.3f} {s['te_theta_ratio']:.3f} {s['te_h_diff']:+.3f}; "
                          f"forced {s['tripped_te_dstar_ratio']:.3f} {s['tripped_te_theta_ratio']:.3f} "
                          f"{s['tripped_te_h_diff']:+.3f}; failed {s['failed_direct'][:3]}")
    print("\n4. Formulas against H_k")
    print("   H_k  | log10 Re_th0: DG87 eq.30  FID eq.74 | theta dn/dxi (Re_th 1e5): DG87  FID | C_tau^1/2/C_tau,EQ^1/2 at transition: DG87  FID")
    for h, dl, fl, dr, fr, dc, fc in formula_table():
        print(f"   {h:.2f} | {dl:8.3f} {fl:8.3f}               | {dr:.5f} {fr:.5f}            | {dc:.3f} {fc:.3f}")
    us = np.array([0.3, 0.5, 0.7])
    print("   lag rate constant K: LNE89 5.6; FID 5.6/(0.75 (1 + U_s)) = "
          + ", ".join(f"{fidkowski2022.lag_k(u):.2f} at U_s {u}" for u in us))


if __name__ == "__main__":
    main()
