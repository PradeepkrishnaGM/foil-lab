"""Checkpoint C2: coupled Newton solver on NACA 2412, Re 1e6, -4 to 16 deg, against XFOIL.

The polar is run like the XFOIL reference: two sweeps from 0 deg (up to 16, down to -4), each
angle started from the previous converged one (DG87 p. 1353), with angle-of-attack continuation
when a point fails (FID section VIII.C; ``viscous.solve_continued``). Compared per angle with the
committed XFOIL reference (canonical section, PANE 160): convergence, Newton iterations, Cl,
Cd, Cm and transition. Print with ``python -m tests.diagnostics.c2_polar [closure_set]``.
"""

from __future__ import annotations

import dataclasses
import time
import warnings

import numpy as np

from foil_lab import viscous as V
from foil_lab.benchmark.references import load_reference
from foil_lab.benchmark.sections import canonical
from foil_lab.params import VISCOUS

SECTION, RE = "NACA2412", 1e6
UP = [float(a) for a in range(0, 17)]
DOWN = [float(a) for a in range(-1, -5, -1)]


def run(cs: str = "fidkowski2022", up=UP, down=DOWN) -> dict:
    """Two sweeps from 0 deg, as the XFOIL reference (``viscous.polar``: each angle from the last
    converged one, with continuation)."""
    prm = dataclasses.replace(VISCOUS, closure_set=cs)
    coords = canonical(SECTION).coords
    geo = V.build_geometry(coords)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sweep in (up, down):
            start, a_prev = None, None
            if sweep[0] < 0 and 0.0 in out and out[0.0].converged:
                start, a_prev = out[0.0].state, 0.0
            for a in sweep:
                t0 = time.perf_counter()
                r = V.polar(coords, [a], RE, prm, geometry=geo, start=start, alpha_start=a_prev)[0]
                r.runtime = time.perf_counter() - t0
                out[a] = r
                if r.converged:
                    start, a_prev = r.state, a
    return out



def table(res: dict) -> list[dict]:
    ref = load_reference(SECTION, RE, 160)
    rows = []
    for a in sorted(res):
        r = res[a]
        x = ref.rows.get(a)
        rows.append(dict(alpha=a, converged=r.converged, iterations=r.iterations, total=r.iterations_total,
                         cl=r.cl, cd=r.cd, cd_te=r.cd_te, cm=r.cm,
                         xtr_u=r.xtr_upper, xtr_l=r.xtr_lower, runtime=getattr(r, "runtime", np.nan),
                         scaled=r.scaled_steps, reason=r.reason, cont=r.continuation,
                         cl_x=x.cl if x else np.nan, cd_x=x.cd if x else np.nan, cm_x=x.cm if x else np.nan,
                         xtr_u_x=x.xtr_top if x else np.nan, xtr_l_x=x.xtr_bot if x else np.nan))
    return rows


def main(cs: str = "fidkowski2022") -> None:
    res = run(cs)
    rows = table(res)
    print(f"closure set {cs}; NACA 2412, Re 1e6; XFOIL reference PANE 160; Cd = Squire-Young at the wake end,"
          " Cd TE = Squire-Young at the trailing edge (cross-check)")
    print(" alpha conv it (total) | Cl      XFOIL   dCl     | Cd       XFOIL    rel dCd  Cd TE   rel  | "
          "Cm      XFOIL   dCm     | xtr u  XF     xtr l  XF    | s")
    for r in rows:
        print(f" {r['alpha']:5.1f} {'Y' if r['converged'] else 'n'}  {r['iterations']:3d} ({r['total']:3d}) | "
              f"{r['cl']:+.4f} {r['cl_x']:+.4f} {r['cl'] - r['cl_x']:+.4f} | "
              f"{r['cd']:.5f} {r['cd_x']:.5f} {100 * (r['cd'] / r['cd_x'] - 1):+6.1f} % "
              f"{r['cd_te']:.5f} {100 * (r['cd_te'] / r['cd_x'] - 1):+5.1f} % | "
              f"{r['cm']:+.4f} {r['cm_x']:+.4f} {r['cm'] - r['cm_x']:+.4f} | "
              f"{r['xtr_u']:.3f} {r['xtr_u_x']:.3f}  {r['xtr_l']:.3f} {r['xtr_l_x']:.3f} | {r['runtime']:.1f}"
              + (f"  (continuation x{r['cont']})" if r["cont"] else "")
              + ("" if r["converged"] else f"  ({r['reason']})"))
    bl_vs_dump(res, cs)


def bl_vs_dump(res: dict, cs: str = "fidkowski2022") -> None:
    """Coupled trailing-edge states against XFOIL's committed dumps (0, 5, 10 deg): delta*, theta,
    H and ue at the trailing edge per surface, transition, delta* at x/c 0.5 and 0.9; ue and theta at
    the first and last wake station."""
    from tests.diagnostics.bl_compare import DATA, dump_name, read_dump
    from tests.diagnostics.c1_compare import branches
    ref = load_reference(SECTION, RE, 160)
    for a in (0.0, 5.0, 10.0):
        r = res.get(a)
        if r is None:
            continue
        x = ref.rows[a]
        print(f" alpha {a:g} ({cs}, converged {r.converged}):")
        for lab, (xi, xs_, ue, ds, th, cf, h) in branches(a).items():
            s = r.surface[lab]
            xtr_new = r.xtr_upper if lab == "upper" else r.xtr_lower
            xtr_xf = x.xtr_top if lab == "upper" else x.xtr_bot
            cells = []
            for xs in (0.5, 0.9):
                i, j = int(np.argmin(np.abs(s["x"] - xs))), int(np.argmin(np.abs(xs_ - xs)))
                cells.append(f"d* x/c {xs}: {s['dstar'][i] / ds[j]:.3f}")
            print(f"   {lab}: transition {xtr_new:.3f} (XFOIL {xtr_xf:.3f}); TE delta* {s['dstar'][-1]:.5f} "
                  f"(XFOIL {ds[-1]:.5f}, ratio {s['dstar'][-1] / ds[-1]:.3f}), theta ratio "
                  f"{s['theta'][-1] / th[-1]:.3f}, H diff {s['h'][-1] - h[-1]:+.3f}, TE ue {s['ue'][-1]:.4f} "
                  f"(XFOIL {ue[-1]:.4f}, diff {s['ue'][-1] - ue[-1]:+.4f}); " + ", ".join(cells))
        _, wake = read_dump(DATA / dump_name(a))
        w = r.surface["wake"]
        print(f"   wake: first station ue {w['ue'][0]:.4f} (XFOIL {abs(wake[0, 3]):.4f}), theta ratio "
              f"{w['theta'][0] / wake[0, 5]:.3f}; last station theta ratio {w['theta'][-1] / wake[-1, 5]:.3f}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
