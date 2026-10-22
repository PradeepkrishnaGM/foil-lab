"""Flap cases of the benchmark (SPEC.md 4.1 and 4.5; DECISIONS.md, M4).

A flap case is a canonical section with a plain flap (``<section>_F<deflection>``, for example
``NACA2412_F10``; :func:`foil_lab.benchmark.sections.flap_name`), made by ``Airfoil.with_flap`` and
written once to a coordinate file that both codes read. XFOIL gets the hinge point with ``FNEW`` and
reports its hinge moment (``Chinge``, trailing edge up positive); Foil-Lab's ``ch`` is trailing edge
down positive, so the comparison is ``ch`` against ``-Chinge``.

This module runs XFOIL and Foil-Lab on a list of flap cases and writes a comparison table. The
pilot (``python -m foil_lab.benchmark.flap_run pilot``) runs a few cases and writes nothing into the
package data; the references of a full run are written there by ``--package``.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..params import BENCHMARK, SWEEP, XFOIL, BenchmarkParams
from .references import SOURCE_REFERENCE_DIR, _one, reference_name
from .report import attached_filter
from .sections import canonical, flap_name
from .viscous_run import CLOSURE_SETS, _write, method_name, read_case, run_case, unreliable_quantities
from .xfoil import XfoilPolar, read_reference_csv, write_coordinate_file


def case_names(bench: BenchmarkParams = BENCHMARK) -> list[str]:
    """All flap cases of the matrix."""
    return [flap_name(b, d) for b in bench.flap_bases for d in bench.flap_deflections]


def _ref_path(ref_dir: Path, name: str, re: float, nodes: int) -> Path:
    return ref_dir / reference_name(name, re, nodes)


def _ours_path(out_dir: Path, name: str, re: float) -> Path:
    return out_dir / "ours" / f"{name}_Re{int(round(re))}.csv"


def run_references(cases: list[tuple[str, float]], ref_dir: Path, out_dir: Path,
                   bench: BenchmarkParams = BENCHMARK, workers: int | None = None) -> None:
    """XFOIL references (160 and 240 PANE nodes) of the cases ``(name, Reynolds number)``."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    secs = {}
    for name, _ in cases:
        if name not in secs:
            secs[name] = canonical(name, bench.num_panels)
            (out_dir / "coords").mkdir(parents=True, exist_ok=True)
            write_coordinate_file(out_dir / "coords" / f"{name}.dat", secs[name].coords, name)
    jobs = [(secs[n], re, nodes) for n, re in cases for nodes in bench.pane_nodes]
    with ProcessPoolExecutor(max_workers=workers or bench.workers) as pool:
        futs = [pool.submit(_one, sec, re, nodes, str(ref_dir), str(out_dir / "xfoil_logs"))
                for sec, re, nodes in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, conv, secs_ = fut.result()
            print(f"[xfoil {i}/{len(futs)}] {name}: {conv} converged, {secs_:.0f} s", flush=True)


def run_ours(cases: list[tuple[str, float]], out_dir: Path, workers: int | None = None,
             bench: BenchmarkParams = BENCHMARK) -> None:
    """Foil-Lab sweeps of the cases (one CSV per case, with the hinge moment)."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers or bench.workers, mp_context=ctx) as pool:
        futs = [pool.submit(run_case, n, re, CLOSURE_SETS[0]) for n, re in cases]
        for i, fut in enumerate(as_completed(futs), 1):
            name, re, rows = fut.result()
            _write(_ours_path(out_dir, name, re), rows)
            print(f"[ours {i}/{len(futs)}] {name} Re {re:g}: "
                  f"{sum(r['converged'] == '1' for r in rows)}/{len(rows)} converged, "
                  f"{sum(r['runtime_s'] for r in rows):.0f} s", flush=True)


def _stats(v: list[float]) -> tuple[float, float]:
    return (sum(v) / len(v), max(v)) if v else (math.nan, math.nan)


def compare(name: str, re: float, ref_dir: Path, out_dir: Path, bench: BenchmarkParams = BENCHMARK):
    """Per-angle table and summary of one case.

    Returns ``(rows, summary)``. ``rows``: one dict per angle with XFOIL (160 nodes), the 240-node
    hinge moment and ours. ``summary``: counts and mean / max differences inside the attached-flow
    filter of SPEC.md 4.1 over the points where both codes converged, with the reference-reliability
    rule applied to Cl, Cd and Cm (not to ``ch``, which has no target yet).
    """
    ref: XfoilPolar = read_reference_csv(_ref_path(ref_dir, name, re, bench.pane_nodes[0]).read_text())
    p240 = _ref_path(ref_dir, name, re, bench.pane_nodes[1])
    ref240 = read_reference_csv(p240.read_text()) if p240.exists() else None
    ours = {r["alpha"]: r for r in read_case(_ours_path(out_dir, name, re))
            if r["method"] == method_name(CLOSURE_SETS[0])}
    filt, cut = attached_filter(ref, bench)
    rows = []
    for a in ref.alphas:
        x, x2, o = ref.rows[a], (ref240.rows.get(a) if ref240 else None), ours.get(a)
        conv = o is not None and o["converged"] == "1"
        rows.append(dict(
            alpha=a, in_filter=a in filt, xfoil=x is not None, ours=conv,
            cl_x=x.cl if x else math.nan, cd_x=x.cd if x else math.nan, cm_x=x.cm if x else math.nan,
            ch_x=x.ch if x else math.nan,
            ch_x240=x2.ch if x2 is not None and x2.ch is not None else math.nan,
            cl=o["cl"] if conv else math.nan, cd=o["cd"] if conv else math.nan,
            cm=o["cm"] if conv else math.nan, ch=o["ch"] if conv else math.nan,
            ch_friction=o["ch_friction"] if conv else math.nan,
            runtime=o["runtime_s"] if o else math.nan))
    use = [r for r in rows if r["in_filter"] and r["xfoil"] and r["ours"]]
    unrel = {}
    for a in filt:
        x, x2 = ref.rows[a], (ref240.rows.get(a) if ref240 else None)
        unrel[a] = unreliable_quantities(x, x2)
    dcl = [abs(r["cl"] - r["cl_x"]) for r in use if "cl" not in unrel[r["alpha"]]]
    dcd = [abs(r["cd"] - r["cd_x"]) / r["cd_x"] for r in use if "cd" not in unrel[r["alpha"]]]
    dcm = [abs(r["cm"] - r["cm_x"]) for r in use if "cm" not in unrel[r["alpha"]]]
    dch = [abs(r["ch"] - r["ch_x"]) for r in use if math.isfinite(r["ch_x"])]
    sens = [abs(r["ch_x240"] - r["ch_x"]) for r in rows
            if r["in_filter"] and r["xfoil"] and math.isfinite(r["ch_x240"])]
    summary = dict(name=name, re=re, cut=cut, n_filter=len(filt),
                   n_xfoil=sum(ref.rows[a] is not None for a in filt), n_both=len(use),
                   n_ours_unconverged=sum(1 for r in rows if r["in_filter"] and r["xfoil"] and not r["ours"]),
                   dcl=_stats(dcl), dcd=_stats(dcd), dcm=_stats(dcm), dch=_stats(dch),
                   ch_xfoil_sensitivity=_stats(sens), ch_range=_stats([abs(r["ch_x"]) for r in use]),
                   runtime_mean=_stats([r["runtime"] for r in rows if math.isfinite(r["runtime"])])[0])
    return rows, summary


def write_tables(cases, ref_dir: Path, out_dir: Path, bench: BenchmarkParams = BENCHMARK) -> str:
    """Markdown of all cases: a summary table, then one per-angle table per case."""
    head = ("| case | Re | filter angles | XFOIL conv. | both conv. | mean / max dCl | mean / max rel dCd | "
            "mean / max dCm | mean / max d(ch) | XFOIL ch 240 vs 160 (mean / max) | mean abs ch (XFOIL) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    summ, detail = [], []
    for name, re in cases:
        rows, s = compare(name, re, ref_dir, out_dir, bench)
        f = lambda t, fmt: f"{t[0]:{fmt}} / {t[1]:{fmt}}"
        summ.append(f"| {name} | {re:g} | {s['n_filter']} (cut {s['cut']}) | {s['n_xfoil']} | {s['n_both']} | "
                    f"{f(s['dcl'], '.4f')} | {s['dcd'][0] * 100:.1f} % / {s['dcd'][1] * 100:.1f} % | "
                    f"{f(s['dcm'], '.4f')} | {f(s['dch'], '.5f')} | {f(s['ch_xfoil_sensitivity'], '.5f')} | "
                    f"{s['ch_range'][0]:.5f} |")
        d = [f"\n#### {name}, Re {re:g}\n", "| alpha | filter | Cl X / ours | Cd X / ours | Cm X / ours | "
             "ch = -Chinge X (160, 240) / ours (friction part) |", "|---|---|---|---|---|---|"]
        for r in rows:
            d.append(f"| {r['alpha']:g} | {'y' if r['in_filter'] else ''} | {r['cl_x']:.4f} / {r['cl']:.4f} | "
                     f"{r['cd_x']:.5f} / {r['cd']:.5f} | {r['cm_x']:.4f} / {r['cm']:.4f} | "
                     f"{r['ch_x']:.5f} ({r['ch_x240']:.5f}) / {r['ch']:.5f} ({r['ch_friction']:.5f}) |")
        detail.append("\n".join(d))
    return head + "\n".join(summ) + "\n" + "\n".join(detail) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("pilot", "full"))
    ap.add_argument("--out", type=Path, default=Path("benchmark_output/m4"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-xfoil", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    args = ap.parse_args(argv)
    bench = BENCHMARK
    if args.what == "pilot":
        cases = [(flap_name("NACA2412", d), re) for d in (10.0, 20.0) for re in (1e6, 3e6)]
    else:
        cases = [(n, re) for n in case_names(bench) for re in bench.reynolds_validated]
    ref_dir = args.out / "xfoil_ref"
    t0 = time.perf_counter()
    if not args.skip_xfoil:
        run_references(cases, ref_dir, args.out, bench, args.workers)
    if not args.skip_ours:
        run_ours(cases, args.out, args.workers, bench)
    text = write_tables(cases, ref_dir, args.out, bench)
    (args.out / f"{args.what}.md").write_text(text)
    print(text)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
