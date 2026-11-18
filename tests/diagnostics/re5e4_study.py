"""Re 5e4 study (v2.2; DECISIONS.md "v2.2 study: Re 5e4"): why the low Reynolds numbers miss the
SPEC.md 4.1 targets. Diagnostic only; not part of the test suite and changes no solver code.

    python tests/diagnostics/re5e4_study.py run --re 70000          # XFOIL references + Foil-Lab
    python tests/diagnostics/re5e4_study.py breakdown --re 50000    # per-point table (markdown)
    python tests/diagnostics/re5e4_study.py bl --re 50000 --section NACA4412 --alphas 3 4 5
                                        # upper/lower BL of both codes: transition, separation, H
    python tests/diagnostics/re5e4_study.py trace --re 50000 --section NACA4412 --alphas 4 4.25
                                        # Newton iterations at the last angle, warm-started from
                                        # the previous ones: largest residual, step, transition

Output goes to ``benchmark_output/study_re5e4`` (not in git). For Re 5e4 the release run's rows are
copied there and the committed references are used; for other Reynolds numbers the references are
the ones ``run`` wrote to ``benchmark_output/study_re5e4/refs``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

from foil_lab.benchmark.references import (generate_references, read_reference_csv,
                                           reference_name)
from foil_lab.benchmark.report import attached_filter
from foil_lab.benchmark.viscous_run import (case_file, read_case, reached_by_restart, run_matrix,
                                            unreliable_quantities)
from foil_lab.params import BENCHMARK

ROOT = Path("benchmark_output/study_re5e4")
REFS = ROOT / "refs"
#: per-point (max) targets of SPEC.md 4.1
MAX = {"cl": 0.10, "cd": 0.30, "cm": 0.03}


def _bench(re: float):
    return dataclasses.replace(BENCHMARK, reynolds_low=(float(re),), reynolds_validated=())


def load_ref(section: str, re: float, nodes: int):
    local = REFS / reference_name(section, re, nodes)
    if local.exists():
        return read_reference_csv(local.read_text())
    from foil_lab.benchmark.references import load_reference
    return load_reference(section, re, nodes)


def run(re: float, workers: int | None) -> None:
    generate_references(_bench(re), out_dir=REFS, output_root=ROOT, workers=workers)
    run_matrix(ROOT, _bench(re), workers=workers, reynolds=[float(re)])


def breakdown(re: float) -> str:
    """Every attached-filter point of every section at ``re``: Foil-Lab and XFOIL values, the
    differences, and which per-point targets fail (``!``), with the reference-reliability flags."""
    lines = ["| section | alpha | conv | path | dCl | rel dCd | dCm | xtr top FL / XF | xtr bot FL / XF | "
             "fails | ref unreliable |", "|" + "---|" * 11]
    totals = {"points": 0, "unconv": 0, "cl": 0, "cd": 0, "cm": 0}
    for sec in BENCHMARK.sections:
        try:
            ref = load_ref(sec, re, 160)
        except FileNotFoundError:
            continue
        try:
            ref240 = load_ref(sec, re, 240)
        except FileNotFoundError:
            ref240 = None
        if float(ref.provenance.get("trailing_edge_gap", "nan")) == 0.0:
            continue                                     # sharp sections: no targets (SPEC.md 4.1)
        filt, _ = attached_filter(ref, BENCHMARK)
        rows = {r["alpha"]: r for r in read_case(case_file(ROOT, sec, re))
                if r["method"].startswith("viscous:")}
        for a in filt:
            x, r = ref.rows[a], rows.get(a)
            if x is None or r is None:
                continue
            totals["points"] += 1
            unrel = unreliable_quantities(x, None if ref240 is None else ref240.rows.get(a))
            conv = r["converged"] == "1" and math.isfinite(r["cl"])
            path = r.get("path", "") or r.get("reason", "")
            if not conv:
                totals["unconv"] += 1
                lines.append(f"| {sec} | {a:g} | no | {path} | | | | | | not converged | "
                             f"{','.join(sorted(unrel))} |")
                continue
            d = {"cl": abs(r["cl"] - x.cl), "cd": abs(r["cd"] - x.cd) / x.cd, "cm": abs(r["cm"] - x.cm)}
            fails = [q for q in MAX if d[q] > MAX[q] and q not in unrel]
            for q in fails:
                totals[q] += 1
            if not fails and not unrel:
                continue                                 # only the interesting points are listed
            lines.append(
                f"| {sec} | {a:g} | yes | {'restart: ' if reached_by_restart(path) else ''}{path} | "
                f"{d['cl']:.3f} | {100 * d['cd']:.1f} % | {d['cm']:.3f} | "
                f"{r['xtr_top']:.3f} / {x.xtr_top:.3f} | {r['xtr_bot']:.3f} / {x.xtr_bot:.3f} | "
                f"{','.join(fails)} | {','.join(sorted(unrel))} |")
    head = (f"Re {re:g}: {totals['points']} filter points (XFOIL converged, open trailing edges); "
            f"{totals['unconv']} not converged; per-point target missed (reference reliable): "
            f"Cl {totals['cl']}, Cd {totals['cd']}, Cm {totals['cm']}.\n\n")
    return head + "\n".join(lines) + "\n"


def xfoil_bl(section: str, re: float, alphas, start: float = 0.0) -> dict:
    """XFOIL's boundary layer at ``alphas`` (one process, ``ALFA`` from ``start`` in 1 deg steps up
    or down to each angle, ``DUMP`` after each; the benchmark's XFOIL settings). Returns
    {alpha: (x, y, H, Cf)} at the panel nodes of the converged angles (H = delta*/theta)."""
    import tempfile
    import numpy as np
    from foil_lab.benchmark.sections import canonical
    from foil_lab.benchmark.xfoil import (XFOIL, alpha_tag, build_script, parse_polar_text,
                                          run_xfoil, write_coordinate_file)
    out = {}
    for sign in (1.0, -1.0):
        targets = [a for a in alphas if (a - start) * sign >= 0]
        if not targets:
            continue
        end = max(targets) if sign > 0 else min(targets)
        with tempfile.TemporaryDirectory(prefix="foil-lab-study-") as wd:
            wd = Path(wd)
            write_coordinate_file(wd / f"{section}.dat", canonical(section).coords, section)
            script = build_script(f"{section}.dat", "p.pol", re, start, end, sign * 1.0, XFOIL, None,
                                  dump=True)
            run = run_xfoil(script, wd, "p.pol", XFOIL)
            n = len(np.loadtxt(wd / "nodes.dat", skiprows=1))
            for r in parse_polar_text(run.polar_text or "")[1]:
                a = round(r.alpha, 6)
                f = wd / f"du_{alpha_tag(a)}.txt"
                if a in targets and f.exists():
                    d = np.loadtxt(f, skiprows=1, usecols=range(7), max_rows=n)
                    out[a] = (d[:, 1], d[:, 2], d[:, 4] / d[:, 5], d[:, 6])
    return out


def _sep(x, cf, xmax: float = 1.0):
    """(first x/c where Cf < 0, last x/c where Cf < 0) on one surface, or None."""
    import numpy as np
    neg = np.where((np.asarray(cf) < 0) & (np.asarray(x) <= xmax))[0]
    return None if not len(neg) else (float(x[neg[0]]), float(x[neg[-1]]))


def bl(section: str, re: float, alphas) -> str:
    """Both codes' boundary layer at ``alphas``: Cl, transition, the laminar separation region
    (Cf < 0) and the largest H on each surface. Foil-Lab is reached as ``fl.polar`` reaches the
    angles (``viscous.sweep``)."""
    import warnings
    import numpy as np
    from foil_lab import viscous as V
    from foil_lab.benchmark.sections import canonical
    xf = xfoil_bl(section, re, alphas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = dict(zip(alphas, V.sweep(canonical(section).coords, list(alphas), re)))
    lines = [f"{section}, Re {re:g}", "| alpha | code | conv | Cl | xtr top | sep top (Cf<0) | max H top | "
             "xtr bot | sep bot | max H bot |", "|" + "---|" * 10]
    for a in alphas:
        r = ours[a]
        cells = []
        for key in ("upper", "lower"):
            srf = r.surface.get(key) if r.surface else None
            if srf is None:
                cells += ["", "", ""]
                continue
            lam = ~np.asarray(srf["turbulent"], bool)
            sp = _sep(srf["x"], srf["cf"])
            cells += [f"{r.xtr_upper if key == 'upper' else r.xtr_lower:.3f}",
                      "-" if sp is None else f"{sp[0]:.3f}-{sp[1]:.3f}",
                      f"{np.max(np.asarray(srf['h'])[lam]) if lam.any() else float('nan'):.2f} (lam)"]
        lines.append(f"| {a:g} | Foil-Lab | {r.converged} | {r.cl:.3f} | " + " | ".join(
            [cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]]) + " |")
        if a in xf:
            x, y, h, cf = xf[a]
            le = int(np.argmin(x))
            up = slice(0, le + 1) if y[0] >= y[-1] and np.mean(y[:le]) > np.mean(y[le:]) else slice(le, None)
            lo = slice(le, None) if up.start == 0 else slice(0, le + 1)
            ref = load_ref(section, re, 160).rows.get(a)
            c = []
            for sl, xt in ((up, ref.xtr_top if ref else math.nan), (lo, ref.xtr_bot if ref else math.nan)):
                sp = _sep(x[sl], cf[sl])
                hs = h[sl][np.asarray(x[sl]) <= (xt if math.isfinite(xt) else 1.0)]
                c += [f"{xt:.3f}", "-" if sp is None else f"{sp[0]:.3f}-{sp[1]:.3f}",
                      f"{np.max(hs) if len(hs) else float('nan'):.2f} (lam)"]
            lines.append(f"| {a:g} | XFOIL | True | {ref.cl if ref else math.nan:.3f} | " + " | ".join(c) + " |")
        else:
            lines.append(f"| {a:g} | XFOIL | False | | | | | | | |")
    return "\n".join(lines) + "\n"


def trace(section: str, re: float, alphas, iterations: int = 20) -> str:
    """Newton iterations of :func:`foil_lab.viscous.solve_viscous` at ``alphas[-1]``, warm-started
    through the earlier ``alphas``: per iteration the largest scaled residual and where it is, the
    step scale and its limit, whether a transition interval moved, and the x/c of every laminar to
    turbulent switch. The solver's functions are wrapped from outside; nothing is changed."""
    import warnings
    import numpy as np
    from foil_lab import viscous as V
    from foil_lab.benchmark.sections import canonical
    c = canonical(section).coords
    geo = V.build_geometry(c)
    m = geo.ps.num_nodes
    nx, ny = np.asarray(geo.ps.node_x), np.asarray(geo.ps.node_y)
    log, cur = [], {}
    o_res, o_ut, o_ss = V.residual, V.update_transition, V.step_scale

    def res(x, st, rn, prm, xp=np):
        r = o_res(x, st, rn, prm, xp)
        if xp is np and np.isrealobj(r):
            rr = np.abs(np.asarray(r)).reshape(-1, V.NV)
            k, q = np.unravel_index(np.argmax(rr), rr.shape)
            sw = np.flatnonzero(np.diff(st.turbulent[:m].astype(int)) != 0)
            cur.update(rmax=float(rr.max()), eq=int(q),
                       where=(f"x {nx[k]:.3f}, y {ny[k]:+.3f}" if k < m else f"wake {k - m}"),
                       trans=" ".join(f"{nx[t]:.3f}" for t in sw))
        return r

    def ss(x, dx, st, rn, prm):
        om, why = o_ss(x, dx, st, rn, prm)
        cur.update(om=om, why=why)
        return om, why

    def ut(x, st, rn, prm):
        moved = o_ut(x, st, rn, prm)
        if cur:
            log.append(dict(cur, moved=moved))
            cur.clear()
        return moved

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = None
        for a in alphas[:-1]:
            r = V.solve_viscous(c, a, re, start=state, geometry=geo)
            state = r.state
        V.residual, V.step_scale, V.update_transition = res, ss, ut
        try:
            r = V.solve_viscous(c, alphas[-1], re, start=state, geometry=geo)
        finally:
            V.residual, V.step_scale, V.update_transition = o_res, o_ss, o_ut
    lines = [f"{section}, Re {re:g}, {alphas[-1]:g} deg from {alphas[:-1]}: converged {r.converged}, "
             f"{r.iterations} iterations", "| it | max residual | equation | at | step | limit | "
             "transition moved | laminar/turbulent switches at x/c |", "|" + "---|" * 8]
    for i, d in enumerate(log[:iterations]):
        lines.append(f"| {i} | {d['rmax']:.1e} | {d['eq']} | {d['where']} | {d.get('om', math.nan):.2f} | "
                     f"{d.get('why', '')} | {d['moved']} | {d['trans']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "breakdown", "bl", "trace"))
    ap.add_argument("--section", default="NACA4412")
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.command == "run":
        run(a.re, a.workers)
    elif a.command == "trace":
        print(trace(a.section, a.re, a.alphas))
    elif a.command == "bl":
        print(bl(a.section, a.re, a.alphas))
    else:
        print(breakdown(a.re))


if __name__ == "__main__":
    main()
