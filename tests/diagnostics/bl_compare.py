"""Boundary-layer diagnostic: our coupled VII against XFOIL's boundary-layer dump (M3).

NACA 2412 (canonical open-trailing-edge file, 160 panels), Re 1e6, N_crit 9. XFOIL is run only
as an external program; its documented ``DUMP`` command writes s, x, y, Ue/Vinf, delta*, theta,
Cf, H (surface rows, then wake rows). XFOIL's dumped Cf is relative to the freestream dynamic
pressure, ours to the local edge value; the comparison divides XFOIL's by ue^2.

Three comparisons per angle:

* coupled: our ``vii.solve_vii`` (benchmark settings) against XFOIL, station by station;
* fed: our boundary-layer marcher on XFOIL's own edge velocity (closures without coupling);
* inviscid check: our core driven by XFOIL's delta* as transpiration, giving Cl and Cm.

Committed dumps (0, 5 and 10 deg) live in ``tests/diagnostics/data`` with provenance headers, so the
comparison runs without XFOIL. Print the tables with ``python -m tests.diagnostics.bl_compare``.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
ANGLES = (0.0, 5.0, 10.0)
RE = 1e6
STATIONS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def dump_name(alpha: float) -> str:
    return f"xfoil_bl_naca2412_re1e6_a{alpha:g}.dat"


def canonical_coords() -> np.ndarray:
    from foil_lab.benchmark.sections import canonical
    return canonical("NACA2412").coords


def run_xfoil_dumps(out_dir: Path, angles=ANGLES) -> dict:
    """Run XFOIL on the canonical NACA 2412 and write the dumps with provenance headers."""
    from foil_lab.benchmark.xfoil import run_xfoil, write_coordinate_file, xfoil_package
    from foil_lab.params import XFOIL
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sha = write_coordinate_file(work / "naca2412.dat", canonical_coords(), "NACA2412")
        lines = ["LOAD naca2412.dat", "PPAR", "N 160", "", "", "PANE", "OPER", "VISC 1000000",
                 "VPAR", "N 9", "", "ITER 200"]
        for a in angles:
            lines += [f"ALFA {a:g}", f"DUMP bl{a:g}.dat"]
        script = "\n".join(lines + ["", "QUIT", ""])
        run = run_xfoil(script, work, f"bl{angles[-1]:g}.dat", params=XFOIL)
        out = {}
        for a in angles:
            body = (work / f"bl{a:g}.dat").read_text()
            head = [f"# XFOIL boundary-layer dump (DUMP), run as an external program",
                    f"# xfoil_package: {xfoil_package()}",
                    f"# command: {' '.join(XFOIL.command)}",
                    f"# geometry: canonical NACA2412 (foil_lab.benchmark.sections), sha256 {sha}",
                    f"# settings: PANE 160, VISC 1e6, N_crit 9, ITER 200, ALFA {a:g}",
                    f"# date_utc: {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
                    f"# xfoil returncode {run.returncode}, timed out {run.timed_out}",
                    "# Cf is relative to the freestream dynamic pressure; Ue/Vinf is signed."]
            path = out_dir / dump_name(a)
            path.write_text("\n".join(head) + "\n" + body)
            out[a] = path
    return out


def read_dump(path: Path):
    """(surface rows, wake rows); columns s, x, y, Ue/Vinf, delta*, theta, Cf, H, ..."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            r = [float(v) for v in line.split()[:12]]
            rows.append(r + [np.nan] * (12 - len(r)))
    a = np.array(rows)
    n_air = int(np.argmax(np.isnan(a[:, 8]))) if np.isnan(a[:, 8]).any() else len(a)
    return a[:n_air], a[n_air:]


def provenance_sha(path: Path) -> str:
    for line in Path(path).read_text().splitlines():
        if line.startswith("# geometry:"):
            return line.rsplit("sha256", 1)[1].strip()
    return ""


def _surfaces(air):
    k = int(np.argmin(np.abs(air[:, 3])))
    return air[:k + 1][::-1], air[k:]          # upper, lower: stagnation point -> trailing edge


@contextlib.contextmanager
def _capture():
    from foil_lab import vii
    rec = {"split": None, "prof": []}
    orig_split, orig_march = vii._split_surfaces, vii.march_boundary_layer

    def split(geom, vt):
        out = orig_split(geom, vt)
        rec["split"] = (out, np.asarray(geom.mid_x), np.asarray(geom.mid_y))
        return out

    def march(*a, **k):
        p = orig_march(*a, **k)
        rec["prof"].append(p)
        return p
    vii._split_surfaces, vii.march_boundary_layer = split, march
    try:
        yield rec
    finally:
        vii._split_surfaces, vii.march_boundary_layer = orig_split, orig_march


def compare(alpha: float, dump: Path | None = None) -> dict:
    """All three comparisons at one angle. Returns tables and summary numbers."""
    from foil_lab import bl_integral, inviscid as core, vii
    coords = canonical_coords()
    air, wake = read_dump(dump or DATA / dump_name(alpha))
    up, lo = _surfaces(air)
    with warnings.catch_warnings(), _capture() as rec:
        warnings.simplefilter("ignore")
        r = vii.solve_vii(coords, alpha, reynolds=RE, max_iter=200, tol=1e-3, transition_model="en",
                          n_crit=9.0)
    (ia, s_a, ue_a, ib, s_b, ue_b, _), mx, my = rec["split"]
    pa, pb = rec["prof"][-2], rec["prof"][-1]
    a_up = np.mean(my[ia]) > np.mean(my[ib])
    ours = {"upper": (ia, pa, ue_a) if a_up else (ib, pb, ue_b),
            "lower": (ib, pb, ue_b) if a_up else (ia, pa, ue_a)}
    out = {"alpha": alpha, "cl": r.solution.cl, "cd": r.cd_wake, "converged": r.converged,
           "iterations": r.iterations, "tables": {}}
    for lab, xf in (("upper", up), ("lower", lo)):
        ii, p, ue = ours[lab]
        s = np.abs(xf[:, 0] - xf[0, 0])
        fed = bl_integral.march_boundary_layer(s, np.abs(xf[:, 3]), RE, transition_model="en",
                                               n_crit=9.0)
        rows = []
        for x in STATIONS:
            i = int(np.argmin(np.abs(xf[:, 1] - x)))
            j = int(np.argmin(np.abs(mx[ii] - x)))
            uex = abs(xf[i, 3])
            rows.append(dict(x=x, ue_xf=uex, ue_cpl=ue[j], dstar_xf=xf[i, 4], dstar_cpl=p.delta_star[j],
                             dstar_fed=fed.delta_star[i], theta_xf=xf[i, 5], theta_cpl=p.theta[j],
                             theta_fed=fed.theta[i], H_xf=xf[i, 7], H_cpl=p.H[j], H_fed=fed.H[i],
                             cf_xf_local=xf[i, 6] / uex ** 2, cf_cpl=p.cf[j], cf_fed=fed.cf[i]))
        out["tables"][lab] = rows
        out[f"xtr_{lab}_coupled"] = float(mx[ii][p.transition_idx])
        out[f"xtr_{lab}_fed"] = float(xf[fed.transition_idx, 1])
    # inviscid check: our core with XFOIL's delta* as transpiration (surface only)
    ps = core.PanelSet.from_bodies(coords)
    sig = np.zeros(ps.num_panels)
    for surf in (up, lo):
        s = np.abs(surf[:, 0] - surf[0, 0])
        dm = np.gradient(np.abs(surf[:, 3]) * surf[:, 4], s)
        for j in range(ps.num_panels):
            d_up = np.min(np.hypot(up[:, 1] - ps.mid_x[j], up[:, 2] - ps.mid_y[j]))
            d_lo = np.min(np.hypot(lo[:, 1] - ps.mid_x[j], lo[:, 2] - ps.mid_y[j]))
            if (d_up <= d_lo) == (surf is up):
                i = int(np.argmin(np.hypot(surf[:, 1] - ps.mid_x[j], surf[:, 2] - ps.mid_y[j])))
                sig[j] = dm[i]
    sol = core.solve(ps, alpha, source_strength=sig)
    cl, _, cm = core.pressure_loads(ps, np.asarray(sol.cp), alpha)
    out["cl_core_with_xfoil_dstar"], out["cm_core_with_xfoil_dstar"] = float(cl), float(cm)
    return out


def main() -> None:
    for a in ANGLES:
        c = compare(a)
        print(f"\n== alpha {a:g}: ours Cl {c['cl']:.4f} Cd {c['cd']:.5f} converged {c['converged']} "
              f"({c['iterations']} it); core with XFOIL delta*: Cl {c['cl_core_with_xfoil_dstar']:.4f} "
              f"Cm {c['cm_core_with_xfoil_dstar']:+.4f}")
        for lab in ("upper", "lower"):
            print(f"   {lab}: transition coupled {c[f'xtr_{lab}_coupled']:.3f}, fed {c[f'xtr_{lab}_fed']:.3f}")
            print("     x/c   ue XF/cpl        d* XF / cpl / fed            H XF / cpl / fed      Cf(local) XF / cpl / fed")
            for r in c["tables"][lab]:
                print(f"    {r['x']:.2f}  {r['ue_xf']:.3f} {r['ue_cpl']:.3f}   {r['dstar_xf']:.6f} {r['dstar_cpl']:.6f} "
                      f"{r['dstar_fed']:.6f}   {r['H_xf']:.2f} {r['H_cpl']:.2f} {r['H_fed']:.2f}   "
                      f"{r['cf_xf_local']:+.5f} {r['cf_cpl']:+.5f} {r['cf_fed']:+.5f}")


if __name__ == "__main__":
    main()
