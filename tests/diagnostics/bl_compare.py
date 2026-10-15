"""Boundary-layer diagnostic: our coupled VII against XFOIL's boundary-layer dump (M3).

NACA 2412 (canonical open-trailing-edge file, 160 panels), Re 1e6, N_crit 9. XFOIL is run only
as an external program; its documented ``DUMP`` command writes s, x, y, Ue/Vinf, delta*, theta,
Cf, H (surface rows, then wake rows). XFOIL's dumped Cf is relative to the freestream dynamic
pressure, ours to the local edge value; the comparison divides XFOIL's by ue^2.

The comparison kept here is the inviscid check: our core driven by XFOIL's delta* as
transpiration, giving Cl and Cm. The diagnosis before the rewrite (DECISIONS.md M3, "Diagnosis")
also compared the v1.3.1 transpiration VII and its boundary-layer marcher with these dumps; both
were deleted in M3 C5, and so were those two comparisons. The coupled solver of M3 is compared
with the same dumps by ``c1_compare`` and ``transition_compare``.

Committed dumps (0, 5 and 10 deg) live in ``tests/diagnostics/data`` with provenance headers, so the
comparison runs without XFOIL. Print the tables with ``python -m tests.diagnostics.bl_compare``.
"""

from __future__ import annotations

import datetime
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
ANGLES = (0.0, 5.0, 10.0)
RE = 1e6


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


def compare(alpha: float, dump: Path | None = None) -> dict:
    """The inviscid check at one angle: our core with XFOIL's delta* as transpiration (surface
    only). Returns Cl, Cm and the number of surface and wake rows of the dump."""
    from foil_lab import inviscid as core
    coords = canonical_coords()
    air, wake = read_dump(dump or DATA / dump_name(alpha))
    up, lo = _surfaces(air)
    out = {"alpha": alpha, "rows_upper": len(up), "rows_lower": len(lo), "rows_wake": len(wake)}
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
        print(f"alpha {a:g}: core with XFOIL delta*: Cl {c['cl_core_with_xfoil_dstar']:.4f} "
              f"Cm {c['cm_core_with_xfoil_dstar']:+.4f}")


if __name__ == "__main__":
    main()
