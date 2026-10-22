"""Station-by-station comparison around transition at a point where our solver does not converge
(M3, decision 3b after report 6), against XFOIL's boundary-layer dump at the same point.

XFOIL is run only as an external program on the canonical file of the section (PANE 160, N_crit 9,
ITER 200), reaching the angle by the reference recipe's sweep from 0 deg in 1 deg steps; its
documented ``DUMP`` output is stored with a provenance header in ``tests/diagnostics/data``. Our
side: the polar of the benchmark (``viscous.polar`` from 0 deg in 1 deg steps); the last converged
angle and the final iterate of the failing angle are compared with the dump.

    python -m tests.diagnostics.transition_compare NACA0012 3e6 10      # needs XFOIL for a new dump
"""

from __future__ import annotations

import datetime
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


def dump_path(section: str, reynolds: float, alpha: float) -> Path:
    return DATA / f"xfoil_bl_{section.lower()}_re{reynolds:.0f}_a{alpha:g}.dat"


def run_xfoil_dump(section: str, reynolds: float, alpha: float) -> Path:
    """XFOIL's dump at ``alpha`` reached from 0 deg in 1 deg steps, with a provenance header."""
    from foil_lab.benchmark.sections import canonical
    from foil_lab.benchmark.xfoil import run_xfoil, write_coordinate_file, xfoil_package
    from foil_lab.params import XFOIL
    step = 1.0 if alpha >= 0 else -1.0
    seq = [round(k * step, 6) + 0.0 for k in range(int(round(abs(alpha))) + 1)]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sha = write_coordinate_file(work / "foil.dat", canonical(section).coords, section)
        lines = ["LOAD foil.dat", "PPAR", "N 160", "", "", "PANE", "OPER", f"VISC {reynolds:g}",
                 "VPAR", "N 9", "", "ITER 200"] + [f"ALFA {a:g}" for a in seq] + ["DUMP bl.dat", "", "QUIT", ""]
        run = run_xfoil("\n".join(lines), work, "bl.dat", params=XFOIL)
        body = (work / "bl.dat").read_text()
    head = ["# XFOIL boundary-layer dump (DUMP), run as an external program",
            f"# xfoil_package: {xfoil_package()}",
            f"# command: {' '.join(XFOIL.command)}",
            f"# geometry: canonical {section} (foil_lab.benchmark.sections), sha256 {sha}",
            f"# settings: PANE 160, VISC {reynolds:g}, N_crit 9, ITER 200, ALFA {' '.join(f'{a:g}' for a in seq)}",
            f"# date_utc: {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
            f"# xfoil returncode {run.returncode}, timed out {run.timed_out}",
            "# Cf is relative to the freestream dynamic pressure; Ue/Vinf is signed."]
    path = dump_path(section, reynolds, alpha)
    path.write_text("\n".join(head) + "\n" + body)
    return path


def our_polar(section: str, reynolds: float, alpha: float):
    """The sweep from 0 deg to ``alpha`` in 1 deg steps as C3 ran it before the restarts of
    ``solve_point``: each angle by ``solve_continued`` from the last converged one, so that the last
    entry is the failed iterate as C3 had it. List of results."""
    from foil_lab import viscous as V
    from foil_lab.benchmark.sections import canonical
    coords = canonical(section).coords
    geo = V.build_geometry(coords)
    step = 1.0 if alpha >= 0 else -1.0
    seq = [round(k * step, 6) + 0.0 for k in range(int(round(abs(alpha))) + 1)]
    out, start, a_prev = [], None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in seq:
            r = V.solve_continued(coords, a, reynolds, start=start, alpha_start=a_prev, geometry=geo)
            out.append(r)
            if r.converged:
                start, a_prev = r.state, a
    return out


def surfaces_xfoil(path: Path):
    from .bl_compare import read_dump
    air, wake = read_dump(path)
    k = int(np.argmin(np.abs(air[:, 3])))
    up, lo = air[:k + 1][::-1], air[k:]
    return {"upper": up, "lower": lo}, wake


def table(section: str, reynolds: float, alpha: float, results, side: str, x_lo: float, x_hi: float) -> str:
    """Rows x, then (ours converged neighbour | ours failing iterate | XFOIL) for ue, theta, H, n/C_tau."""
    xf, _ = surfaces_xfoil(dump_path(section, reynolds, alpha))
    xs = xf[side]
    lines = []
    conv = [r for r in results if r.converged]
    last = conv[-1] if conv else None
    fail = results[-1]
    lines.append(f"{section} Re {reynolds:g} {alpha:g} deg, {side} surface; neighbour = ours converged at "
                 f"{last.alpha if last else '-'} deg; iterate = ours at {fail.alpha:g} deg "
                 f"(converged {fail.converged}, {fail.iterations} it)")
    for tag, r in (("neighbour", last), ("iterate", fail)):
        if r is not None:
            lines.append(f"  {tag}: xtr upper {r.xtr_upper:.4f} lower {r.xtr_lower:.4f}, Cl {r.cl:.4f}, "
                         f"step limits {r.step_limits}")
    lines.append("     x      | ue nb / it / XF        | theta*1e3 nb / it / XF  | H nb / it / XF       | "
                 "turb nb/it | 3rd nb / it")
    s_it = fail.surface[side]
    s_nb = last.surface[side] if last else None
    # downstream of the leading edge only, where x is monotonic along the surface (the stagnation
    # point can lie on the other side of the leading edge)
    xs = xs[int(np.argmin(xs[:, 1])):]
    j0 = int(np.argmin(s_it["x"]))
    for j in np.flatnonzero((s_it["x"] >= x_lo) & (s_it["x"] <= x_hi) & (np.arange(s_it["x"].size) >= j0)):
        x = s_it["x"][j]
        o = np.argsort(xs[:, 1])
        xu = np.interp(x, xs[o, 1], np.abs(xs[o, 3]))
        xt = np.interp(x, xs[o, 1], xs[o, 5])
        xh = np.interp(x, xs[o, 1], xs[o, 7])
        nb = (s_nb["ue"][j], s_nb["theta"][j], s_nb["h"][j], int(s_nb["turbulent"][j]), s_nb["third"][j]) \
            if s_nb is not None and j < s_nb["x"].size else (np.nan,) * 5
        it = (s_it["ue"][j], s_it["theta"][j], s_it["h"][j], int(s_it["turbulent"][j]), s_it["third"][j])
        lines.append(f"  {x:.5f} | {nb[0]:.4f} {it[0]:.4f} {xu:.4f} | {1e3 * nb[1]:.4f} {1e3 * it[1]:.4f} "
                     f"{1e3 * xt:.4f} | {nb[2]:.3f} {it[2]:.3f} {xh:.3f} | {nb[3]} {it[3]} | {nb[4]:.4g} {it[4]:.4g}")
    return "\n".join(lines)


if __name__ == "__main__":
    sec, re, a = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    if not dump_path(sec, re, a).exists():
        print("XFOIL dump:", run_xfoil_dump(sec, re, a))
    res = our_polar(sec, re, a)
    lo, hi = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (0.0, 0.2)
    for side in ("upper", "lower"):
        print(table(sec, re, a, res, side, lo, hi))
