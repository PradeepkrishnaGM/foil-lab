"""Laminar closures against the exact Falkner-Skan relations (M3, closure choice on physics).

Compares, at 15 values of H from 2.2 to 4.0:

* DG87 eqs. (10)-(12) as implemented in ``foil_lab.bl.closures``;
* the corresponding closures of Fidkowski (FID eqs. 47, 56, 61; "A Coupled Inviscid-Viscous
  Airfoil Analysis Solver, Revisited"), written out here from the paper;
* XFOIL's dumped laminar states (Re_theta Cf/2 with Cf divided by ue^2, and H*), where available.

Print with ``python -m tests.diagnostics.laminar_closure_compare``.
"""

from __future__ import annotations

import numpy as np

from foil_lab.bl import closures as cl
from tests.diagnostics import falkner_skan as fs
from tests.diagnostics.bl_compare import DATA, dump_name, read_dump

RANGES = ((2.2, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.0001))
#: XFOIL transition x/c of the dumped runs (polar files), to select laminar points
XTR = {0.0: (0.6571, 0.6806), 5.0: (0.2932, 0.9999)}


# ---- FID closures (laminar), from the paper's appendix (page images checked) ----------- #
def hstar_lam_fid(hk):
    """FID eqs. (47)-(48)."""
    ht = hk - 4.35
    return np.where(hk < 4.35, (0.0111 * ht ** 2 - 0.0278 * ht ** 3) / (hk + 1.0) + 1.528
                    - 0.0002 * (ht * hk) ** 2, 0.015 * ht ** 2 / hk + 1.528)


def ret_cf2_lam_fid(hk):
    """Re_theta Cf/2 from FID eq. (56) (the paper gives Re_theta cf)."""
    return 0.5 * np.where(hk < 5.5, 0.0727 * (5.5 - hk) ** 3 / (hk + 1.0) - 0.07,
                          0.015 * (1.0 - 1.0 / (hk - 4.5)) ** 2 - 0.07)


def ret_2cd_hstar_lam_fid(hk):
    """Re_theta 2C_D/H* from FID eq. (61)."""
    return np.where(hk < 4.0, 0.00205 * np.abs(4.0 - hk) ** 5.5 + 0.207,
                    -0.0016 * (hk - 4.0) ** 2 / (1.0 + 0.02 * (hk - 4.0) ** 2) + 0.207)


def dg87(hk):
    ret = 1000.0                                       # groups are independent of Re_theta
    return dict(ret_cf2=cl.cf_lam(hk, ret) * ret / 2.0, hstar=cl.hstar_lam(hk),
                ret_2cd_hstar=cl.cd_lam(hk, ret) * 2.0 * ret / cl.hstar_lam(hk))


def fid(hk):
    return dict(ret_cf2=ret_cf2_lam_fid(hk), hstar=hstar_lam_fid(hk),
                ret_2cd_hstar=ret_2cd_hstar_lam_fid(hk))


def xfoil_points():
    """Laminar points of the dumps (x/c > 0.02, upstream of XFOIL's transition)."""
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


def compare():
    t = fs.table()
    h = t["h"]
    rows = {"DG87": dg87(h), "FID": fid(h)}
    xf = xfoil_points()
    exact_at_xf = fs.table(np.clip(xf[:, 0], 2.2, 4.0))
    res = {}
    for q in ("ret_cf2", "hstar", "ret_2cd_hstar"):
        res[q] = {}
        for name, v in rows.items():
            err = v[q] - t[q]
            res[q][name] = [(lo, hi, float(np.max(np.abs(err[(h >= lo) & (h < hi)]))),
                             float(np.max(np.abs(err / t[q])[(h >= lo) & (h < hi)])))
                            for lo, hi in RANGES]
        if q in ("ret_cf2", "hstar"):
            col = 1 if q == "ret_cf2" else 2
            err = xf[:, col] - exact_at_xf[q]
            res[q]["XFOIL dump"] = []
            for lo, hi in RANGES:
                k = (xf[:, 0] >= lo) & (xf[:, 0] < hi)
                res[q]["XFOIL dump"].append(
                    (lo, hi, float(np.max(np.abs(err[k]))) if k.any() else np.nan,
                     float(np.max(np.abs(err[k] / exact_at_xf[q][k]))) if k.any() else np.nan,
                     int(k.sum())))
    return t, rows, res


def main() -> None:
    t, rows, res = compare()
    print("   H    | Re_th Cf/2: exact  DG87     FID     | H*: exact  DG87    FID     | Re_th 2CD/H*: exact DG87  FID")
    for i, hh in enumerate(t["h"]):
        print(f"  {hh:.3f} | {t['ret_cf2'][i]:.5f} {rows['DG87']['ret_cf2'][i]:+.5f} {rows['FID']['ret_cf2'][i]:+.5f} |"
              f" {t['hstar'][i]:.5f} {rows['DG87']['hstar'][i]:.5f} {rows['FID']['hstar'][i]:.5f} |"
              f" {t['ret_2cd_hstar'][i]:.5f} {rows['DG87']['ret_2cd_hstar'][i]:.5f} {rows['FID']['ret_2cd_hstar'][i]:.5f}")
    for q, by in res.items():
        print(f"\n{q}: max |error| against Falkner-Skan (absolute; relative)")
        for name, lst in by.items():
            cells = []
            for item in lst:
                lo, hi, a, r = item[:4]
                n = f", n={item[4]}" if len(item) > 4 else ""
                cells.append(f"H {lo:.1f}-{min(hi, 4.0):.1f}: {a:.4f}; {100 * r:.1f} %{n}")
            print(f"  {name:10s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
