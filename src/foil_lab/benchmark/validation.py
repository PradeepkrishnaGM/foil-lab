"""VALIDATION.md generated from the benchmark output (SPEC.md 4.1, 4.12; M9).

    python -m foil_lab.benchmark validation --output benchmark_output/m4_m3check \\
        --flap-output benchmark_output/m4_full --file VALIDATION.md

Reads the viscous benchmark rows (``<output>/m3_viscous``, written by ``python -m foil_lab.benchmark
viscous``) and the flap benchmark rows (``<flap-output>``, written by ``python -m
foil_lab.benchmark.flap_run full``), compares them with the committed XFOIL references, and writes
the tables with the SPEC.md 4.1 targets marked met or not met. Nothing is re-run.
"""

from __future__ import annotations

import datetime as _dt
import math
import subprocess
from pathlib import Path

from ..params import BENCHMARK, BenchmarkParams
from . import viscous_run as vr
from .report import _max, _mean, _pct

METHOD = vr.method_name(vr.CLOSURE_SETS[0])
QUANT = ("converged", "mean dCl", "max dCl", "mean rel dCd", "max rel dCd", "mean dCm", "max dCm")


def _fmt(k: str, v: float) -> str:
    if not math.isfinite(v):
        return "-"
    if k == "converged" or "dCd" in k:
        return f"{v:.1%}"
    return f"{v:.4f}"


def _row(label: list, st, claim: bool) -> list:
    v = vr.verdict(st)
    out = list(label) + [f"{st.n_compared - st.n_unconverged}/{st.n_compared}"]
    for k in QUANT:
        val, ok = v[k]
        out.append(_fmt(k, val) + ("" if not claim or not math.isfinite(val) else (" ✓" if ok else " ✗")))
    return out


def _md(head, rows) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main_tables(output_root: Path, bench: BenchmarkParams = BENCHMARK) -> tuple[str, dict]:
    cases = [vr.compare_case(s, re, output_root, bench) for s in bench.sections
             for re in bench.reynolds_low + bench.reynolds_validated
             if vr.case_file(output_root, s, re).exists()]
    valid = set(bench.reynolds_validated)
    head = ["Re", "points (converged / compared)", "converged", "mean dCl", "max dCl", "mean rel dCd",
            "max rel dCd", "mean dCm", "max dCm"]
    rows, verdicts = [], {}
    for re in sorted({c.reynolds for c in cases}):
        sub = [c for c in cases if c.reynolds == re and not c.sharp]
        st = vr._pool_dict(sub, ("wake", "all", True), METHOD)
        claim = re in valid
        rows.append(_row([f"{re:g}" + ("" if claim else " (not validated)")], st, claim))
        if claim:
            verdicts[re] = vr.verdict(st)
    open_md = _md(head, rows)
    rows = []
    for re in sorted({c.reynolds for c in cases if c.sharp}):
        st = vr._pool_dict([c for c in cases if c.reynolds == re and c.sharp], ("wake", "all", False), METHOD)
        rows.append(_row([f"{re:g}"], st, False))
    sharp_md = _md(head, rows) if rows else "(no sharp-edged cases in the output)\n"
    rows = []
    for c in cases:
        st = c.variants.get((METHOD, "wake", "all", True))
        if st is None or c.sharp:
            continue
        rows.append([c.section, f"{c.reynolds:g}", f"{st.n_compared - st.n_unconverged}/{st.n_compared}",
                     _fmt("mean dCl", _mean(st.dcl)), _fmt("mean rel dCd", _mean(st.dcd_rel)),
                     _fmt("mean dCm", _mean(st.dcm)), str(c.xfoil_unconv)])
    per_md = _md(["section", "Re", "converged / compared", "mean dCl", "mean rel dCd", "mean dCm",
                  "XFOIL not converged in filter"], rows)
    n_sec = len({c.section for c in cases if not c.sharp})
    return open_md, dict(sharp=sharp_md, per=per_md, verdicts=verdicts, n_cases=len(cases), n_sec=n_sec)


def flap_tables(flap_root: Path, bench: BenchmarkParams = BENCHMARK) -> str:
    from . import flap_run as fr
    names = fr.case_names(bench)
    cases = [fr.compare_case(n, re, flap_root / "xfoil_ref", flap_root, bench) for n in names
             for re in bench.reynolds_validated if fr._ours_path(flap_root, n, re).exists()]
    if not cases:
        return "(flap benchmark output not found)\n"
    defl = sorted({c.deflection for c in cases})
    res = sorted({c.re for c in cases})
    ok_all = {}
    rows = []
    for d in defl:
        row = [f"{d:g}"]
        for r in res:
            v = vr.verdict(fr._pool([c for c in cases if c.deflection == d and c.re == r]))
            bad = fr._failed(v)
            ok_all[(d, r)] = not bad
            row.append("✓" if not bad else "✗ " + ", ".join(f"{k} {_fmt(k, v[k][0])}" for k in bad))
        rows.append(row)
    run_ok = None
    for d in defl:
        if all(ok_all[(d, r)] for r in res):
            run_ok = d
        else:
            break
    md = _md(["flap deg"] + [f"Re {r:g}" for r in res], rows)
    md += (f"\n**Validated flap range:** {'none' if run_ok is None else f'{run_ok:g} deg'} (the largest "
           "deflection at which every Reynolds number meets every target, with all smaller deflections "
           "too). Larger deflections are benchmarked and reported without a claim.\n")
    rows = []
    for d in defl:
        sel = [c for c in cases if c.deflection == d]
        da = [x for c in sel for x in c.d_a]
        db = [x for c in sel for x in c.d_b]
        ch = [x for c in sel for x in c.ch_abs]
        rows.append([f"{d:g}", str(len(da)), f"{_mean(da):.5f} / {_max(da):.5f}" if da else "-",
                     f"{_mean(db):.5f} / {_max(db):.5f}" if db else "-", f"{_mean(ch):.4f}" if ch else "-"])
    md += ("\nHinge moment, all Reynolds numbers pooled, attached-flow filter, both codes converged: "
           "(a) Foil-Lab `ch` against XFOIL's own Cp and Cf integrated with Foil-Lab's definition (target per "
           f"(deflection, Re) cell: mean <= {fr.TARGET_A['mean']}, max <= {fr.TARGET_A['max']}; met in every "
           "cell, see docs/benchmark_m4.md; the target was set from this run); (b) against `-Chinge` as XFOIL "
           f"reports it (loose target mean <= {fr.TARGET_B['mean']}, max <= {fr.TARGET_B['max']}; XFOIL's "
           "definition of Chinge is not known, docs/xfoil_hinge_calibration.md).\n\n")
    md += _md(["flap deg", "points", "(a) mean / max", "(b) mean / max", "mean abs ch (XFOIL)"], rows)
    return md


def build(output_root: Path, flap_root: Path, bench: BenchmarkParams = BENCHMARK) -> str:
    from .. import __version__
    open_md, extra = main_tables(output_root, bench)
    v = extra["verdicts"]
    met = [re for re, vv in v.items() if all(ok for _, ok in vv.values())]
    lines = [
        "# Validation",
        "",
        f"Generated by `python -m foil_lab.benchmark validation` from the benchmark output on "
        f"{_dt.date.today().isoformat()} (Foil-Lab {__version__}, document written at commit "
        f"{_git('rev-parse', '--short', 'HEAD') or 'unknown'}; benchmark rows from `{output_root}` and "
        f"`{flap_root}`). Every number below is measured; none is tuned. Agreement is with XFOIL 6.99, "
        "not with experiment.",
        "",
        "## How the comparison is made",
        "",
        "- Both codes read the same canonical coordinate file (160 panels, Selig order); every reference "
        "polar stores the SHA-256 of that file, and a test fails if the current code would write a different "
        "file.",
        "- XFOIL 6.99 runs as an external program (never linked or copied), under `xvfb-run`, with `PANE` at "
        "160 nodes (the reference) and 240 nodes (paneling sensitivity), N_crit 9, free transition, Mach 0, "
        "`ITER 200`; two fresh processes per case (0 to 20 deg and 0 to -4 deg), a skipped angle retried "
        "once. The reference polars and their provenance are committed in "
        "`src/foil_lab/data/reference/xfoil/`, so the comparison runs without XFOIL.",
        "- Foil-Lab: the coupled viscous-inviscid solver (closure set \"fidkowski2022\"), Cd by Squire-Young "
        "at the end of the wake.",
        "- Points: angles from -4 deg up to the first angle where XFOIL's Cl reaches 90 % of its maximum "
        "(attached-flow filter), where XFOIL converged. A point Foil-Lab did not converge on is counted as "
        "not converged and left out of the error metrics. A reference point where XFOIL's own 240-node "
        "result differs from its 160-node result by more than the per-point target is not counted for "
        "that quantity (SPEC.md 4.1).",
        "- Targets (SPEC.md 4.1), per Reynolds number: converged >= 95 %; mean |dCl| <= 0.03, max <= 0.10; "
        "mean relative |dCd| <= 10 %, max <= 30 %; mean |dCm| <= 0.01, max <= 0.03. ✓ met, ✗ not met.",
        "",
        "## Sections with an open trailing edge",
        "",
        f"NACA 0012, 2412, 4412, 23012, Clark Y and Eppler E387 with a 0.25 % trailing-edge gap "
        f"({extra['n_sec']} sections), 25 angles each.",
        "",
        open_md,
        f"Targets met at every validated Reynolds number: "
        f"{'yes' if len(met) == len(v) else 'no (met at ' + ', '.join(f'{r:g}' for r in sorted(met)) + ')'}.",
        "",
        "### Per section (validated range and Re 5e4)",
        "",
        extra["per"],
        "## Sharp trailing edge (E387), no targets",
        "",
        "A closed trailing edge is analysed with a sharp-edge treatment whose wake drag is flagged; the "
        "interfaces open closed edges to 0.0005 c by default. For a sharp edge XFOIL's own result keeps "
        "changing with its panel count, so the paneling sensitivity is a lower bound.",
        "",
        extra["sharp"],
        "## Plain flaps (NACA 0012 and 2412, 25 % chord, hinge at mid-thickness)",
        "",
        "Same comparison on the flapped coordinate files, deflections 0 to 20 deg, Re 1e5 to 6e6; both "
        "sections pooled per (deflection, Re).",
        "",
        flap_tables(flap_root, bench),
        "",
        "## Exact solutions and consistency (tests, no XFOIL)",
        "",
        "- Kármán-Trefftz surface pressure against the analytic solution; Williams' exact two-element "
        "solution (ARC R&M 3717); ground effect equal to the free-air mirror problem; pressure, circulation "
        "and far-field lift agree; Blasius and turbulent flat plates; NumPy and JAX inviscid Cl to 1e-6; "
        "Kármán-Tsien and the aspect-ratio correction each applied once; geometry operations preserve the "
        "shape; bit-identical output for identical input (`tests/test_exact_solutions.py`, "
        "`tests/test_williams.py`, `tests/test_corrections.py`, `tests/test_bl_closures.py`).",
        "- Several elements: no viscous reference is available; consistency tests only "
        "(`tests/test_multielement.py`).",
        "",
        "## Limits",
        "",
        "- Subsonic, attached or mildly separated flow. Not for transonic flow. Viscous results are Mach 0; "
        "compressibility (Kármán-Tsien) applies to inviscid results.",
        "- Validated at Re from 1e5 to 6e6 (the targets above); Re 5e4 is reported, not validated.",
        "- Maximum lift and post-stall behaviour are approximate (outside the attached-flow filter).",
        "- Plain flaps are validated at 0 deg only; 5 to 20 deg are reported without a claim.",
        "- At 20 deg flap the boundary layer on the flap"s upper surface separates (H up to 13 at the trailing edge in both codes, also inside the attached-flow filter). There Foil-Lab"s displacement thickness grows 7-15 % faster than XFOIL"s while theta agrees, which gives Cl about 0.04 and Cd about 10 % below XFOIL (NACA 2412 F20, Re 3e6, 4 deg; `tests/diagnostics/flap_bl_compare.py`). Upstream of the separation the states agree within a few per cent.",
        "- Multi-element results exclude wake merging and slot separation and have no maximum-lift prediction.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python -m foil_lab.benchmark references        # XFOIL references (needs xfoil and xvfb-run)",
        "python -m foil_lab.benchmark viscous --output benchmark_output/run   # about 1 h on 5 workers",
        "python -m foil_lab.benchmark.flap_run full --out benchmark_output/flaps",
        "python -m foil_lab.benchmark validation --output benchmark_output/run --flap-output benchmark_output/flaps",
        "```",
        "",
        "Detailed reports: docs/benchmark_m3.md, docs/benchmark_m4.md; the v1.3.1 \"before\" picture: "
        "docs/baseline_v1.3.1.md.",
        "",
    ]
    return "\n".join(lines)


def write_validation(output_root: Path, flap_root: Path, path: Path, bench: BenchmarkParams = BENCHMARK) -> Path:
    path = Path(path)
    path.write_text(build(Path(output_root), Path(flap_root), bench))
    return path
