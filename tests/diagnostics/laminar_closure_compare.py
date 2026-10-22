"""Laminar closures of each closure set against the exact Falkner-Skan relations (M3).

Compares, at 15 values of H from 2.2 to 4.0, the laminar closures of every closure set in
:mod:`foil_lab.bl.closure_sets` ("drela1987": DG87 eqs. 10-12; "fidkowski2022": FID eqs. 47, 56,
61) and XFOIL's dumped laminar states (Re_theta Cf/2 with Cf divided by ue^2, and H*) with the
exact similarity values of :mod:`tests.diagnostics.falkner_skan`. The measured errors are kept
as they are (DECISIONS.md M3, "Two closure sets"; validation document in M9).

Print with ``python -m tests.diagnostics.laminar_closure_compare``.
"""

from __future__ import annotations

import numpy as np

from foil_lab.bl import closure_sets
from tests.diagnostics import falkner_skan as fs
from tests.diagnostics.bl_compare import DATA, dump_name, read_dump

RANGES = ((2.2, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.0001))
QUANTITIES = ("ret_cf2", "hstar", "ret_2cd_hstar")
#: XFOIL transition x/c of the dumped runs (polar files), to select laminar points
XTR = {0.0: (0.6571, 0.6806), 5.0: (0.2932, 0.9999)}


def set_groups(cs, h):
    """Re_theta Cf/2, H*, Re_theta 2C_D/H* of a set's laminar closures (independent of Re_theta)."""
    cs = closure_sets.get(cs)
    ret = 1000.0
    hk = cs.hk(np.asarray(h, float))
    hstar = cs.hstar_lam(hk, ret)
    return dict(ret_cf2=cs.cf_lam(hk, ret) * ret / 2.0, hstar=hstar,
                ret_2cd_hstar=cs.cd_lam(hk, ret) * 2.0 * ret / hstar)


def xfoil_points():
    """Laminar points of the dumps (x/c > 0.02, upstream of XFOIL's transition): H, Re_th Cf/2, H*."""
    out = []
    for alpha, (xu, xl) in XTR.items():
        air, _ = read_dump(DATA / dump_name(alpha))
        x, ue, th, cf, h, hs = (air[:, 1], np.abs(air[:, 3]), air[:, 5], air[:, 6] / air[:, 3] ** 2,
                                air[:, 7], air[:, 8])
        upper = np.arange(len(x)) < np.argmin(x)
        lam = np.where(upper, x < xu, x < xl) & (x > 0.02)
        ret = 1e6 * ue * th
        out.append(np.column_stack([h, ret * cf / 2.0, hs])[lam])
    return np.vstack(out)


def errors():
    """{quantity: {source: [(lo, hi, max abs error, max rel error[, n])]}} against Falkner-Skan."""
    t = fs.table()
    h = t["h"]
    res = {q: {} for q in QUANTITIES}
    for name in closure_sets.CLOSURE_SETS:
        v = set_groups(name, h)
        for q in QUANTITIES:
            err = v[q] - t[q]
            res[q][name] = [(lo, hi, float(np.max(np.abs(err[(h >= lo) & (h < hi)]))),
                             float(np.max(np.abs(err / t[q])[(h >= lo) & (h < hi)])))
                            for lo, hi in RANGES]
    xf = xfoil_points()
    ex = fs.table(np.clip(xf[:, 0], 2.2, 4.0))
    for q, col in (("ret_cf2", 1), ("hstar", 2)):
        err = xf[:, col] - ex[q]
        res[q]["XFOIL dump"] = []
        for lo, hi in RANGES:
            k = (xf[:, 0] >= lo) & (xf[:, 0] < hi)
            res[q]["XFOIL dump"].append(
                (lo, hi, float(np.max(np.abs(err[k]))) if k.any() else np.nan,
                 float(np.max(np.abs(err[k] / ex[q][k]))) if k.any() else np.nan, int(k.sum())))
    return t, res


def main() -> None:
    t, res = errors()
    vals = {n: set_groups(n, t["h"]) for n in closure_sets.CLOSURE_SETS}
    names = list(vals)
    print("   H    | Re_th Cf/2: exact " + " ".join(f"{n:>13s}" for n in names)
          + " | H*: exact " + " ".join(f"{n:>13s}" for n in names))
    for i, hh in enumerate(t["h"]):
        print(f"  {hh:.3f} | {t['ret_cf2'][i]:.5f}           " + " ".join(f"{vals[n]['ret_cf2'][i]:+13.5f}" for n in names)
              + f" | {t['hstar'][i]:.5f}  " + " ".join(f"{vals[n]['hstar'][i]:13.5f}" for n in names))
    for q, by in res.items():
        print(f"\n{q}: max |error| against Falkner-Skan (absolute; relative)")
        for name, lst in by.items():
            cells = []
            for item in lst:
                lo, hi, a, r = item[:4]
                n = f", n={item[4]}" if len(item) > 4 else ""
                cells.append(f"H {lo:.1f}-{min(hi, 4.0):.1f}: {a:.4f}; {100 * r:.1f} %{n}")
            print(f"  {name:14s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
