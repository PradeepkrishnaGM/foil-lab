"""Checkpoint C3 of M3: the coupled viscous solver (``foil_lab.viscous``) on the benchmark matrix.

One "viscous" method per closure set (DECISIONS.md M3, Q5 and "Two closure sets"), run like the
XFOIL reference: per section and Reynolds number two sweeps from 0 deg (up to ``alpha_max``, down to
``alpha_min``), each angle started from the last converged one with angle-of-attack continuation
(``viscous.polar``; the down sweep starts from the converged 0 deg state). Both codes read the same
canonical coordinate file. Drag is Squire-Young at the end of the wake; the trailing-edge value
(``cd_te``) is reported next to it as a cross-check.

Metrics and the attached-flow filter are those of ``report.py`` (SPEC.md 4.1): a point enters the
metrics only if XFOIL and our solver converged there; points we did not converge on are counted.
The SPEC.md 4.1 targets apply to open trailing edges at Re 5e5-6e6, per Reynolds number; the
sharp-edged E387 and the low Reynolds numbers are reported without targets.

    python -m foil_lab.benchmark viscous          # run (writes benchmark_output/m3_viscous)
    python -m foil_lab.benchmark viscous-report   # tables in benchmark_output/m3_viscous/report.md
"""

from __future__ import annotations

import csv
import dataclasses
import math
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..params import BENCHMARK, SWEEP, VISCOUS, BenchmarkParams
from .references import load_reference
from .report import MethodStats, _f, _max, _md, _mean, _pct, attached_filter

#: SPEC.md 4.1 targets (open trailing edge, Re 5e5-6e6, attached-flow filter, per Reynolds number)
TARGETS = {"converged": 0.95, "cl_mean": 0.03, "cl_max": 0.10, "cd_mean": 0.10, "cd_max": 0.30,
           "cm_mean": 0.01, "cm_max": 0.03}
CLOSURE_SETS = ("fidkowski2022", "drela1987")
COLUMNS = ("method", "alpha", "converged", "cl", "cd", "cd_te", "cm", "xtr_top", "xtr_bot",
           "iterations", "iterations_total", "continuation", "runtime_s", "reason", "error")


def method_name(closure_set: str) -> str:
    return f"viscous:{closure_set}"


def case_file(output_root: Path, section: str, reynolds: float) -> Path:
    return output_root / "m3_viscous" / f"{section}_Re{int(round(reynolds))}.csv"


def _sweeps() -> tuple[list[float], list[float]]:
    n = int(round((SWEEP.alpha_max - SWEEP.alpha_min) / SWEEP.alpha_step))
    alphas = [round(SWEEP.alpha_min + i * SWEEP.alpha_step, 6) for i in range(n + 1)]
    up = [a for a in alphas if a >= 0.0]
    down = sorted((a for a in alphas if a < 0.0), reverse=True)
    return up, down


def _row(r, closure_set: str, runtime: float, error: str = "") -> dict:
    if r is None:
        return dict(method=method_name(closure_set), alpha=math.nan, converged="0", cl=math.nan, cd=math.nan,
                    cd_te=math.nan, cm=math.nan, xtr_top=math.nan, xtr_bot=math.nan, iterations=0,
                    iterations_total=0, continuation=0, runtime_s=runtime, reason="exception", error=error)
    return dict(method=method_name(closure_set), alpha=r.alpha, converged="1" if r.converged else "0",
                cl=r.cl, cd=r.cd, cd_te=r.cd_te, cm=r.cm, xtr_top=r.xtr_upper, xtr_bot=r.xtr_lower,
                iterations=r.iterations, iterations_total=r.iterations_total, continuation=r.continuation,
                runtime_s=runtime, reason=r.reason, error=error)


def run_case(section: str, reynolds: float, closure_set: str) -> tuple[str, float, list[dict]]:
    """Both sweeps of one (section, Reynolds number) with one closure set."""
    from .. import viscous as V
    from .sections import canonical
    prm = dataclasses.replace(VISCOUS, closure_set=closure_set)
    coords = canonical(section).coords
    geo = V.build_geometry(coords)
    up, down = _sweeps()
    rows, state0 = [], None
    for sweep in (up, down):
        start, a_prev = (state0, 0.0) if (sweep is down and state0 is not None) else (None, None)
        for a in sweep:
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = V.polar(coords, [a], reynolds, prm, geometry=geo, start=start, alpha_start=a_prev)[0]
                row = _row(r, closure_set, time.perf_counter() - t0)
            except Exception as exc:  # recorded, never hidden
                r = None
                row = _row(None, closure_set, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")
                row["alpha"] = a
            rows.append(row)
            if r is not None and r.converged:
                start, a_prev = r.state, a
                if a == 0.0:
                    state0 = r.state
    return section, reynolds, rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["method"], r["alpha"]))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in COLUMNS})


def read_case(path: Path) -> list[dict]:
    with path.open() as f:
        out = []
        for r in csv.DictReader(f):
            for k in ("alpha", "cl", "cd", "cd_te", "cm", "xtr_top", "xtr_bot", "runtime_s"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else math.nan
            for k in ("iterations", "iterations_total", "continuation"):
                r[k] = int(r[k])
            out.append(r)
        return out


def run_matrix(output_root: Path = Path("benchmark_output"), bench: BenchmarkParams = BENCHMARK,
               workers: int | None = None, closure_sets=CLOSURE_SETS, sections=None,
               reynolds=None) -> None:
    """Run every (section, Reynolds number, closure set) of the matrix; one CSV per case (both
    sets in one file). Worker processes are spawned with one BLAS thread each."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(s, re, cs) for s in (sections or bench.sections)
            for re in (reynolds or bench.reynolds_validated + bench.reynolds_low) for cs in closure_sets]
    results: dict[tuple[str, float], list[dict]] = {}
    for (s, re, cs) in jobs:                                   # keep rows of sets not rerun
        p = case_file(output_root, s, re)
        if p.exists() and (s, re) not in results:
            results[(s, re)] = [r for r in read_case(p) if r["method"] not in
                                {method_name(c) for c in closure_sets}]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers or bench.workers, mp_context=ctx) as pool:
        futs = [pool.submit(run_case, s, re, cs) for s, re, cs in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            sec, re, rows = fut.result()
            results.setdefault((sec, re), []).extend(rows)
            n_conv = sum(r["converged"] == "1" for r in rows)
            print(f"[{i}/{len(futs)}] {sec} Re {re:g} {rows[0]['method']}: {n_conv}/{len(rows)} converged, "
                  f"{sum(r['runtime_s'] for r in rows):.0f} s", flush=True)
            _write(case_file(output_root, sec, re), results[(sec, re)])


# --------------------------------------------------------------------------------------- #
# Comparison and tables
# --------------------------------------------------------------------------------------- #
@dataclasses.dataclass
class ViscousCase:
    section: str
    reynolds: float
    sharp: bool
    filter_alphas: list
    xfoil_unconv: int
    stats: dict            # method -> MethodStats (Cd at the wake end)
    stats_te: dict         # method -> MethodStats (Cd TE)
    iters: dict            # method -> list of final-solve iterations (used points)


def compare_case(section: str, reynolds: float, output_root: Path,
                 bench: BenchmarkParams = BENCHMARK) -> ViscousCase:
    ref = load_reference(section, reynolds, bench.pane_nodes[0])
    filt, _ = attached_filter(ref, bench)
    rows = read_case(case_file(output_root, section, reynolds))
    stats, stats_te, iters = {}, {}, {}
    for m in sorted({r["method"] for r in rows}):
        mrows = {r["alpha"]: r for r in rows if r["method"] == m}
        st, stt, it = MethodStats(), MethodStats(), []
        for a in filt:
            r, x = mrows.get(a), ref.rows[a]
            if r is None or x is None:
                continue
            for s in (st, stt):
                s.n_compared += 1
                s.runtime.append(r["runtime_s"])
            if r["converged"] != "1" or not math.isfinite(r["cl"]):
                st.n_unconverged += 1
                stt.n_unconverged += 1
                continue
            it.append(r["iterations"])
            for s, cd in ((st, r["cd"]), (stt, r["cd_te"])):
                s.dcl.append(abs(r["cl"] - x.cl))
                s.dcm.append(abs(r["cm"] - x.cm))
                s.dcd_rel.append(abs(cd - x.cd) / x.cd)
                if math.isfinite(r["xtr_top"]):
                    s.dxtr_top.append(abs(r["xtr_top"] - x.xtr_top))
                if math.isfinite(r["xtr_bot"]):
                    s.dxtr_bot.append(abs(r["xtr_bot"] - x.xtr_bot))
        stats[m], stats_te[m], iters[m] = st, stt, it
    sharp = float(ref.provenance.get("trailing_edge_gap", "nan")) == 0.0
    return ViscousCase(section, reynolds, sharp, filt, sum(ref.rows[a] is None for a in filt), stats,
                       stats_te, iters)


def verdict(st: MethodStats) -> dict:
    """SPEC.md 4.1 targets for pooled stats of one Reynolds number: {name: (value, met)}."""
    conv = 1.0 - st.n_unconverged / st.n_compared if st.n_compared else math.nan
    v = {"converged": (conv, conv >= TARGETS["converged"]),
         "mean dCl": (_mean(st.dcl), _mean(st.dcl) <= TARGETS["cl_mean"]),
         "max dCl": (_max(st.dcl), _max(st.dcl) <= TARGETS["cl_max"]),
         "mean rel dCd": (_mean(st.dcd_rel), _mean(st.dcd_rel) <= TARGETS["cd_mean"]),
         "max rel dCd": (_max(st.dcd_rel), _max(st.dcd_rel) <= TARGETS["cd_max"]),
         "mean dCm": (_mean(st.dcm), _mean(st.dcm) <= TARGETS["cm_mean"]),
         "max dCm": (_max(st.dcm), _max(st.dcm) <= TARGETS["cm_max"])}
    return v


def _pool_dict(cases, key: str, method: str) -> MethodStats:
    out = MethodStats()
    for c in cases:
        st = getattr(c, key).get(method)
        if st is None:
            continue
        out.n_compared += st.n_compared
        out.n_unconverged += st.n_unconverged
        for k in ("dcl", "dcm", "dcd_rel", "dxtr_top", "dxtr_bot", "runtime"):
            getattr(out, k).extend(getattr(st, k))
    return out


def _cells(st: MethodStats, v: dict | None) -> list[str]:
    def mark(k):
        return "" if v is None or not math.isfinite(v[k][0]) else (" ✓" if v[k][1] else " ✗")
    conv = 1.0 - st.n_unconverged / st.n_compared if st.n_compared else math.nan
    return [f"{st.n_compared - st.n_unconverged}/{st.n_compared} ({_pct(conv)}){mark('converged')}",
            _f(_mean(st.dcl), ".4f") + mark("mean dCl"), _f(_max(st.dcl), ".4f") + mark("max dCl"),
            _pct(_mean(st.dcd_rel)) + mark("mean rel dCd"), _pct(_max(st.dcd_rel)) + mark("max rel dCd"),
            _f(_mean(st.dcm), ".4f") + mark("mean dCm"), _f(_max(st.dcm), ".4f") + mark("max dCm"),
            f"{_f(_mean(st.dxtr_top), '.3f')} / {_f(_mean(st.dxtr_bot), '.3f')}"]


HEAD = ["converged / XFOIL converged", "mean dCl", "max dCl", "mean rel dCd", "max rel dCd", "mean dCm",
        "max dCm", "mean dx_tr up / low"]


def build_markdown(cases: list[ViscousCase], bench: BenchmarkParams = BENCHMARK) -> str:
    methods = sorted({m for c in cases for m in c.stats})
    valid = set(bench.reynolds_validated)
    lines = ["# M3 C3: coupled viscous solver against XFOIL (SPEC.md 4.1)", "",
             "Reference XFOIL 6.99, 160 PANE nodes, N_crit 9, Mach 0. Attached-flow filter and metrics as "
             "in the baseline report; a point is used only if both codes converged. Targets (SPEC.md 4.1, "
             "open trailing edge, Re 5e5-6e6, per Reynolds number): converged >= 95 %, mean |dCl| <= 0.03 "
             "and max <= 0.10, mean rel |dCd| <= 10 % and max <= 30 %, mean |dCm| <= 0.01 and max <= 0.03 "
             "(✓ met, ✗ not met). Cd: Squire-Young at the end of the wake; 'Cd TE' rows use the "
             "Squire-Young value at the trailing edge (cross-check). Iterations are those of the final "
             "solve.", ""]
    for group, sharp in (("open trailing edge (NACA 0012, 2412, 4412, 23012, Clark Y, E387 with a 0.25 % "
                          "gap)", False), ("sharp trailing edge (E387), no targets", True)):
        lines += [f"## {group}", ""]
        rows = []
        for re in sorted({c.reynolds for c in cases}):
            sub = [c for c in cases if c.reynolds == re and c.sharp == sharp]
            if not sub:
                continue
            tag = f"{re:g}" + ("" if re in valid else " (not validated)")
            for m in methods:
                for key, lab in (("stats", ""), ("stats_te", ", Cd TE")):
                    st = _pool_dict(sub, key, m)
                    v = verdict(st) if (re in valid and not sharp) else None
                    rows.append([tag, m.split(":")[1] + lab] + _cells(st, v))
        lines += [_md(["Re", "closure set"] + HEAD, rows), ""]
    lines += ["## Per case", ""]
    rows = []
    for c in cases:
        for m in methods:
            st = c.stats.get(m)
            if st is None:
                continue
            it = c.iters.get(m, [])
            rows.append([c.section, f"{c.reynolds:g}", m.split(":")[1]] + _cells(st, None)
                        + [_pct(_mean(c.stats_te[m].dcd_rel)), str(c.xfoil_unconv),
                           f"{_f(_mean(it), '.1f')} / {max(it) if it else '-'}",
                           _f(_mean(st.runtime), ".1f")])
    lines += [_md(["section", "Re", "closure set"] + HEAD + ["mean rel dCd (Cd TE)", "XFOIL not converged",
                                                              "iterations mean / max", "run time per point (s)"],
                  rows), ""]
    return "\n".join(lines)


def write_report(output_root: Path = Path("benchmark_output"), bench: BenchmarkParams = BENCHMARK
                 ) -> list[ViscousCase]:
    cases = []
    for sec in bench.sections:
        for re in bench.reynolds_low + bench.reynolds_validated:
            if case_file(output_root, sec, re).exists():
                cases.append(compare_case(sec, re, output_root, bench))
    cases.sort(key=lambda c: (bench.sections.index(c.section), c.reynolds))
    out = output_root / "m3_viscous" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_markdown(cases, bench))
    return cases


