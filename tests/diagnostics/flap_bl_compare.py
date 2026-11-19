"""Boundary-layer states of a flap case against XFOIL's dump, station by station (v2.2).

Reaches the angle the way the benchmark does (:func:`foil_lab.viscous.sweep` over the benchmark's
up and down sweeps, so that the point is reached from converged neighbours), runs XFOIL on the same
coordinate file (benchmark script, ``DUMP`` at each angle from 0), and prints theta, delta*, H and
ue on both surfaces at fixed x/c. Used for the F20 lift and drag deficit (DECISIONS.md v2.2, E2).

Needs XFOIL; takes about 15 minutes for NACA 2412 F20 at Re 3e6 (the whole benchmark sweep).
Print with ``python -m tests.diagnostics.flap_bl_compare [NAME RE ALPHA]``.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from foil_lab import viscous as V
from foil_lab.benchmark.sections import canonical
from foil_lab.benchmark.viscous_run import _sweeps
from foil_lab.benchmark.xfoil import (alpha_tag, build_script, parse_polar_text, run_xfoil,
                                      write_coordinate_file)

STATIONS = (0.1, 0.3, 0.5, 0.7, 0.74, 0.76, 0.8, 0.85, 0.9, 0.95, 0.99)


def main(name: str = "NACA2412_F20", reynolds: float = 3e6, alpha: float = 4.0) -> None:
    sec = canonical(name)
    with tempfile.TemporaryDirectory(prefix="foil-lab-flap-bl-") as tmp:
        wd = Path(tmp)
        write_coordinate_file(wd / "c.dat", sec.coords, name)
        run = run_xfoil(build_script("c.dat", "p.pol", reynolds, 0.0, alpha, 1.0 if alpha >= 0 else -1.0,
                                     hinge=sec.hinge, dump=True), wd, "p.pol")
        row = [r for r in parse_polar_text(run.polar_text)[1] if abs(r.alpha - alpha) < 1e-6][0]
        # XFOIL DUMP columns used: x (1), Ue/Vinf (3), Dstar (4), Theta (5), H (7)
        du = np.loadtxt(wd / f"du_{alpha_tag(alpha)}.txt", skiprows=1, usecols=range(8))
    got = {}
    up, down = _sweeps()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        V.sweep(sec.coords, up + down, reynolds,
                on_result=lambda a, r, s, walked, err: got.__setitem__(a, r) if r is not None and r.converged else None)
    if alpha not in got:
        print(f"Foil-Lab did not converge at {alpha} deg")
        return
    r = got[alpha]
    print(f"XFOIL    cl={row.cl:.4f} cd={row.cd:.5f} cm={row.cm:.4f} xtr {row.xtr_top:.3f}/{row.xtr_bot:.3f}")
    print(f"Foil-Lab cl={r.cl:.4f} cd={r.cd:.5f} cm={r.cm:.4f} xtr {r.xtr_upper:.3f}/{r.xtr_lower:.3f}")
    x, st = r.state
    geo = V.build_geometry(sec.coords)
    nx = geo.ps.node_x
    m = len(nx)
    k_le = int(np.argmin(du[:, 1][du[:, 1] <= 1.0001]))
    for upper in (True, False):
        print(("upper" if upper else "lower") + " x/c | XFOIL theta*1e3 d*1e3 H ue | Foil-Lab theta*1e3 d*1e3 H ue")
        xs = du[:k_le + 1] if upper else du[k_le:]
        xs = xs[xs[:, 1] <= 1.0001]
        nodes = [b for b in st.branches if b.upper == upper][0].nodes
        nodes = nodes[nodes < m]
        for xc in STATIONS:
            j = int(np.argmin(np.abs(xs[:, 1] - xc)))
            k = nodes[int(np.argmin(np.abs(nx[nodes] - xc)))]
            th, ds, ue = x[k, 0], x[k, 1], abs(x[k, 3])
            print(f" {xc:4.2f} | {xs[j, 5] * 1e3:6.3f} {xs[j, 4] * 1e3:7.3f} {xs[j, 7]:5.2f} {abs(xs[j, 3]):5.3f}"
                  f" | {th * 1e3:6.3f} {ds * 1e3:7.3f} {ds / th:5.2f} {ue:5.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2])) if len(a) == 3 else main()
