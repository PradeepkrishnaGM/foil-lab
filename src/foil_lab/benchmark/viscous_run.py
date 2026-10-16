"""Checkpoint C3 of M3: the coupled viscous solver (``foil_lab.viscous``) on the benchmark matrix.

One "viscous" method per benchmarked closure set (``CLOSURE_SETS``; DECISIONS.md M3, Q5, "Two closure
sets" and the parking of "drela1987"), run like the
XFOIL reference: per section and Reynolds number two sweeps from 0 deg (up to ``alpha_max``, down to
``alpha_min``), each angle started from the last converged one with angle-of-attack continuation
(``viscous.polar``; the down sweep starts from the converged 0 deg state); a point whose start from
the starting march failed is then reached from the nearest converged angle of its sweep, and if
that fails, from the nearest converged angle of the other sweep where that is at least as near
(``viscous.reach_from_nearest``; the ``path`` column says so). Both codes read the same
canonical coordinate file. Drag is Squire-Young at the end of the wake; the trailing-edge value
(``cd_te``) is reported next to it as a cross-check.

Metrics and the attached-flow filter are those of ``report.py`` (SPEC.md 4.1): a point enters the
metrics only if XFOIL and our solver converged there; points we did not converge on are counted.
The SPEC.md 4.1 targets apply to open trailing edges at Re 1e5-6e6, per Reynolds number; the
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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..params import BENCHMARK, SWEEP, VISCOUS, BenchmarkParams
from .references import load_reference
from .report import MethodStats, _f, _max, _md, _mean, _pct, attached_filter

#: SPEC.md 4.1 targets (open trailing edge, Re 1e5-6e6, attached-flow filter, per Reynolds number)
TARGETS = {"converged": 0.95, "cl_mean": 0.03, "cl_max": 0.10, "cd_mean": 0.10, "cd_max": 0.30,
           "cm_mean": 0.01, "cm_max": 0.03}
#: closure sets run and reported by the benchmark. "drela1987" is parked: it does not converge with the
#: wake (its far wake ends on the H_k clip) and its wake treatment is decided after v2.0; the set,
#: its laminar Falkner-Skan comparison and its tests are kept (DECISIONS.md M3, report 6 decisions)
CLOSURE_SETS = ("fidkowski2022",)
COLUMNS = ("method", "alpha", "converged", "cl", "cd", "cd_te", "cm", "xtr_top", "xtr_bot",
           "iterations", "iterations_total", "continuation", "runtime_s", "reason", "path", "warning", "error")


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
                    iterations_total=0, continuation=0, runtime_s=runtime, reason="exception", path="",
                    warning="", error=error)
    return dict(method=method_name(closure_set), alpha=r.alpha, converged="1" if r.converged else "0",
                cl=r.cl, cd=r.cd, cd_te=r.cd_te, cm=r.cm, xtr_top=r.xtr_upper, xtr_bot=r.xtr_lower,
                iterations=r.iterations, iterations_total=r.iterations_total, continuation=r.continuation,
                runtime_s=runtime, reason=r.reason, path=r.path, warning="; ".join(r.warnings),
                error=error)


def run_case(section: str, reynolds: float, closure_set: str) -> tuple[str, float, list[dict]]:
    """Both sweeps of one (section, Reynolds number) with one closure set: the angles are reached by
    :func:`foil_lab.viscous.sweep` (the same path as ``fl.polar``); this adds the rows and the run
    time per point. A point a walk reaches keeps its row, with the run time and iterations of the
    walk added (DECISIONS.md M3, decision 3 after report 8)."""
    from .. import viscous as V
    from .sections import canonical
    prm = dataclasses.replace(VISCOUS, closure_set=closure_set)
    coords = canonical(section).coords
    up, down = _sweeps()
    rows: dict[float, dict] = {}

    def record(a, r, seconds, walked, error):
        if not walked:
            if error is None:
                rows[a] = _row(r, closure_set, seconds)
            else:
                rows[a] = _row(None, closure_set, seconds, error)
                rows[a]["alpha"] = a
            return
        old = rows[a]
        if r.converged:
            rows[a] = _row(r, closure_set, old["runtime_s"] + seconds)
        else:
            old["runtime_s"] += seconds
            old["iterations_total"] += r.iterations_total

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        V.sweep(coords, up + down, reynolds, prm, on_result=record)
    return section, reynolds, [rows[a] for a in up + down]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["method"], r["alpha"]))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (format(v, ".17g") if isinstance(v, float) else v)
                        for k in COLUMNS for v in [r.get(k, "")]})


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
    parked = [c for c in closure_sets if c not in CLOSURE_SETS]
    if parked:
        raise ValueError(f"closure set(s) {parked} not benchmarked (parked); benchmarked: {CLOSURE_SETS}")
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
#: subsets of the converged points by how they were reached (``ViscousResult.path``): "direct" = warm
#: start, continuation or starting march; "restart" = repair or march restart (DECISIONS.md M3, 3c), or
#: reached afterwards from the nearest converged angle (path "from <angle> deg: ..."; decision 3 after
#: report 8)
SUBSETS = ("all", "direct", "restart")
#: quantities the reference-reliability rule is applied to, with the target it is compared with
RELIABILITY = {"cl": "cl_max", "cd": "cd_max", "cm": "cm_max"}


def reached_by_restart(path: str) -> bool:
    return "restart" in (path or "") or (path or "").startswith("from ")


def unreliable_quantities(x160, x240) -> frozenset:
    """Reference reliability (SPEC.md 4.1; DECISIONS.md M3, decision 4 after report 8):
    the quantities for which XFOIL's own result at 240 PANE nodes differs from that at 160 by more
    than the target, read as the per-point (max) target: |dCl| > 0.10, |dCd|/Cd > 30 %, |dCm| > 0.03.
    No 240-node result at that angle: nothing is excluded."""
    if x160 is None or x240 is None:
        return frozenset()
    d = {"cl": abs(x240.cl - x160.cl), "cd": abs(x240.cd - x160.cd) / x160.cd,
         "cm": abs(x240.cm - x160.cm)}
    return frozenset(q for q, t in RELIABILITY.items() if d[q] > TARGETS[t])


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
    #: (method, "wake" | "te", subset, rule applied) -> MethodStats; subsets other than "all" hold
    #: only converged points (n_compared = their number)
    variants: dict = dataclasses.field(default_factory=dict)
    #: filter angles (XFOIL converged at 160 nodes) whose reference is not counted, per quantity
    unreliable: dict = dataclasses.field(default_factory=dict)


def compare_case(section: str, reynolds: float, output_root: Path,
                 bench: BenchmarkParams = BENCHMARK) -> ViscousCase:
    ref = load_reference(section, reynolds, bench.pane_nodes[0])
    try:
        ref240 = load_reference(section, reynolds, bench.pane_nodes[1])
    except FileNotFoundError:
        ref240 = None
    filt, _ = attached_filter(ref, bench)
    unrel = {a: unreliable_quantities(ref.rows[a], None if ref240 is None else ref240.rows.get(a))
             for a in filt}
    rows = read_case(case_file(output_root, section, reynolds))
    stats, stats_te, iters, variants = {}, {}, {}, {}
    for m in sorted({r["method"] for r in rows} & {method_name(c) for c in CLOSURE_SETS}):
        mrows = {r["alpha"]: r for r in rows if r["method"] == m}
        var = {(k, sub, rule): MethodStats() for k in ("wake", "te") for sub in SUBSETS
               for rule in (False, True)}
        it = []
        for a in filt:
            r, x = mrows.get(a), ref.rows[a]
            if r is None or x is None:
                continue
            conv = r["converged"] == "1" and math.isfinite(r["cl"])
            sub = "restart" if conv and reached_by_restart(r.get("path", "")) else "direct"
            for (k, s_, rule), s in var.items():
                if s_ != "all" and (not conv or s_ != sub):
                    continue
                s.n_compared += 1
                s.runtime.append(r["runtime_s"])
                if not conv:
                    s.n_unconverged += 1
                    continue
                skip = unrel[a] if rule else frozenset()
                cd = r["cd"] if k == "wake" else r["cd_te"]
                if "cl" not in skip:
                    s.dcl.append(abs(r["cl"] - x.cl))
                if "cm" not in skip:
                    s.dcm.append(abs(r["cm"] - x.cm))
                if "cd" not in skip:
                    s.dcd_rel.append(abs(cd - x.cd) / x.cd)
                if math.isfinite(r["xtr_top"]):
                    s.dxtr_top.append(abs(r["xtr_top"] - x.xtr_top))
                if math.isfinite(r["xtr_bot"]):
                    s.dxtr_bot.append(abs(r["xtr_bot"] - x.xtr_bot))
            if conv:
                it.append(r["iterations"])
        stats[m], stats_te[m], iters[m] = var[("wake", "all", False)], var[("te", "all", False)], it
        variants.update({(m,) + key: v for key, v in var.items()})
    sharp = float(ref.provenance.get("trailing_edge_gap", "nan")) == 0.0
    n_unrel = {q: sum(q in unrel[a] for a in filt if ref.rows[a] is not None) for q in RELIABILITY}
    return ViscousCase(section, reynolds, sharp, filt, sum(ref.rows[a] is None for a in filt), stats,
                       stats_te, iters, variants, n_unrel)


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


def _pool_dict(cases, key, method: str) -> MethodStats:
    """Pool one method over cases; ``key`` is "stats", "stats_te" or a variant key (kind, subset,
    rule)."""
    out = MethodStats()
    for c in cases:
        st = getattr(c, key).get(method) if isinstance(key, str) else c.variants.get((method,) + key)
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
             f"open trailing edge, Re {min(valid):g}-{max(valid):g}, per Reynolds number): converged >= 95 %, "
             "mean |dCl| <= 0.03 "
             "and max <= 0.10, mean rel |dCd| <= 10 % and max <= 30 %, mean |dCm| <= 0.01 and max <= 0.03 "
             "(✓ met, ✗ not met). Cd: Squire-Young at the end of the wake; 'Cd TE' rows use the "
             "Squire-Young value at the trailing edge (cross-check). Iterations are those of the final "
             "solve.", ""]
    lines += ["Reference reliability (SPEC.md 4.1): a reference point where XFOIL's own result at 240 PANE "
              "nodes differs from 160 nodes by more than the per-point (maximum) target for a quantity is "
              "reported but not counted towards that quantity's targets (|dCl| > 0.10, |dCd|/Cd > 30 %, "
              "|dCm| > 0.03); the convergence count is not affected. Rows 'rule: yes' apply it (the SPEC.md "
              "4.1 measure), 'rule: no' do not. The "
              "number of such points per case is in the per-case table.", ""]
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
                for rule in (False, True):
                    for kind, lab in (("wake", ""), ("te", ", Cd TE")):
                        st = _pool_dict(sub, (kind, "all", rule), m)
                        v = verdict(st) if (re in valid and not sharp) else None
                        rows.append([tag, m.split(":")[1] + lab, "yes" if rule else "no"] + _cells(st, v))
        lines += [_md(["Re", "closure set", "rule"] + HEAD, rows), ""]
    lines += ["## Points reached directly and by a restart (open trailing edge, rule: no)", "",
              "Direct: warm start, continuation or starting march; restart: repair restart or restart from "
              "the starting march (`path` column; DECISIONS.md M3, 3c), or reached afterwards from the nearest "
              "converged angle of the sweep (`path` 'from <angle> deg: ...'). Converged points only.", ""]
    rows = []
    for re in sorted({c.reynolds for c in cases}):
        sub = [c for c in cases if c.reynolds == re and not c.sharp]
        for m in methods:
            for s_ in ("direct", "restart"):
                st = _pool_dict(sub, ("wake", s_, False), m)
                st_te = _pool_dict(sub, ("te", s_, False), m)
                rows.append([f"{re:g}", m.split(":")[1], s_, str(st.n_compared),
                             _f(_mean(st.dcl), ".4f"), _f(_max(st.dcl), ".4f"),
                             _pct(_mean(st.dcd_rel)), _pct(_max(st.dcd_rel)),
                             _pct(_mean(st_te.dcd_rel)), _pct(_max(st_te.dcd_rel)),
                             _f(_mean(st.dcm), ".4f"), _f(_max(st.dcm), ".4f"),
                             f"{_f(_mean(st.dxtr_top), '.3f')} / {_f(_mean(st.dxtr_bot), '.3f')}"])
    lines += [_md(["Re", "closure set", "reached", "points", "mean dCl", "max dCl", "mean rel dCd",
                   "max rel dCd", "mean rel dCd (Cd TE)", "max rel dCd (Cd TE)", "mean dCm", "max dCm",
                   "mean dx_tr up / low"], rows), ""]
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
                           _f(_mean(st.runtime), ".1f"),
                           " / ".join(str(c.unreliable.get(q, 0)) for q in RELIABILITY)])
    lines += [_md(["section", "Re", "closure set"] + HEAD + ["mean rel dCd (Cd TE)", "XFOIL not converged",
                                                              "iterations mean / max", "run time per point (s)",
                                                              "unreliable reference Cl / Cd / Cm"],
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


