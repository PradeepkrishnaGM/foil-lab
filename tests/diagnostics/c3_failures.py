"""C3 failure classification for "fidkowski2022" (M3, decision 3a after report 6).

Replays the C3 benchmark exactly as ``foil_lab.benchmark.viscous_run.run_case`` runs it (same
sweeps, warm starts and continuation), with the solver's clip and transition steps observed (not
changed): after every Newton update the stations clipped to ``hk_min``, the transition interval of
every branch and the state of the laminar station ahead of it are recorded. Writes, under
``<output>/m3_diag/``:

* ``results.json``: every point with full-precision Cl, Cd, Cd TE, Cm, transition, iterations
  (the baseline for "every point that converged before converges to the same values"), and for
  every non-converged point its features and class.

``--summary`` prints the classes of the non-converged points inside the attached-flow filter where
XFOIL converged (the points counted against the SPEC.md 4.1 convergence target), per Reynolds
number, open trailing edges at Re 5e5-6e6 (and the other groups separately).

Classes of a non-converged point (judged on the direct attempt, which is the result ``polar``
returns for a failed point, over its last ``WINDOW`` Newton iterations):

* ``cold start``: the attempt started from the starting march (no converged state before it in
  its sweep);
* ``H_k clip after laminar-separated transition``: in at least ``MIN_HITS`` of those iterations a
  station at most two positions downstream of a free transition interval is clipped to ``hk_min``
  while the laminar station ahead of the interval is separated (laminar Cf <= 0);
* ``transition relocation``: else, the transition interval of some branch moved in at least
  ``MIN_HITS`` of those iterations;
* ``isolated H_k clip in the turbulent layer``: else, in at least ``MIN_HITS`` of those iterations
  a turbulent airfoil station more than two positions downstream of the transition interval is
  clipped to ``hk_min``;
* ``other``.

A cold-start failure is also given the class its iterations would have without the cold start
(``signature``), so that the mechanisms can be counted over all failures.

The thresholds are those of this diagnostic, not of the solver. Run with
``python -m tests.diagnostics.c3_failures [--workers N] [--sections A,B] [--reynolds R1,R2]
[--write-benchmark]``; with ``--write-benchmark`` the run is also the benchmark run (its rows
replace the "fidkowski2022" rows of ``<output>/m3_viscous``; then ``python -m foil_lab.benchmark
viscous-report``).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from foil_lab import viscous as V
from foil_lab.benchmark import viscous_run as vr
from foil_lab.bl import closure_sets
from foil_lab.params import BENCHMARK

WINDOW = 15
WRITE_BENCHMARK = os.environ.get("C3_WRITE_BENCHMARK") == "1"
OUTPUT = os.environ.get("C3_OUTPUT", "benchmark_output")
MIN_HITS = 3
CLASSES = ("cold start", "H_k clip after laminar-separated transition", "transition relocation",
           "isolated H_k clip in the turbulent layer", "other")

# --------------------------------------------------------------------------------------- #
# Observation hooks (module level, so that spawned workers install them on import)
# --------------------------------------------------------------------------------------- #
_TRACE: list | None = None
_RESULTS: dict = {}
_solve, _clip, _update, _polar = V.solve_viscous, V.clip_state, V.update_transition, V.polar


def _clip_obs(x, st, prm):
    if _TRACE is not None:
        hw = st.cpl.hw
        low = np.flatnonzero((x[:, 3] > 0) & ((x[:, 1] - hw) < prm.hk_min * x[:, 0]))
        _TRACE.append({"clip": low.tolist()})
    return _clip(x, st, prm)


def _update_obs(x, st, reynolds, prm):
    before = [(b.p_tr, b.forced_w is None) for b in st.branches]
    moved = _update(x, st, reynolds, prm)
    if _TRACE:
        rec = _TRACE[-1]
        if "branches" in rec:            # update_transition at iteration 0 (no clip record yet)
            return moved
        cs = closure_sets.get(prm.closure_set)
        clipped = set(rec["clip"])
        br = []
        for b, (p0, _) in zip(st.branches, before):
            d = {"upper": b.upper, "p_tr": b.p_tr, "moved": b.p_tr != p0, "free": b.forced_w is None,
                 "x_tr": None, "lam_cf": None, "lam_hk": None, "clip_rel": []}
            nd = b.nodes
            if b.p_tr is not None:
                d["x_tr"] = float(st.cpl.s[nd[b.p_tr]])
                q = nd[b.p_tr - 1]
                hk = float(cs.hk(x[q, 1] / x[q, 0]))
                ret = float(reynolds * x[q, 3] * x[q, 0])
                d["lam_hk"] = hk
                d["lam_cf"] = float(np.real(cs.cf_lam(hk, ret)))
                d["clip_rel"] = [int(p - b.p_tr) for p in range(nd.size) if nd[p] in clipped]
            br.append(d)
        rec["branches"] = br
        rec["moved"] = bool(moved)
        n_air = st.cpl.m_air
        rec["clip_wake"] = sum(1 for j in clipped if j >= n_air)
    return moved


def _solve_obs(coords, alpha, reynolds, params=V.VISCOUS, trips=None, start=None, geometry=None):
    global _TRACE
    _TRACE = []
    try:
        r = _solve(coords, alpha, reynolds, params, trips, start, geometry)
    finally:
        trace, _TRACE = _TRACE, None
    r.trace = {"cold": start is None, "its": trace}
    return r


def _polar_obs(coords, alphas, reynolds, params=V.VISCOUS, trips=None, geometry=None, start=None,
               alpha_start=None):
    out = _polar(coords, alphas, reynolds, params, trips, geometry, start, alpha_start)
    for r in out:
        _RESULTS[r.alpha] = r
    return out


V.solve_viscous, V.clip_state, V.update_transition, V.polar = _solve_obs, _clip_obs, _update_obs, _polar_obs


# --------------------------------------------------------------------------------------- #
# Features and classes
# --------------------------------------------------------------------------------------- #
def features(r) -> dict:
    its = r.trace["its"][-WINDOW:]
    sep_clip = tr_moves = air_clip = wake_clip = iso_clip = 0
    x_tr_range = 0.0
    for rec in its:
        br = rec.get("branches", [])
        if any(b["free"] and b["p_tr"] is not None and b["lam_cf"] is not None and b["lam_cf"] <= 0.0
               and any(0 <= k <= 2 for k in b["clip_rel"]) for b in br):
            sep_clip += 1
        if any(any(k > 2 for k in b["clip_rel"]) for b in br):
            iso_clip += 1
        if any(b["moved"] and b["free"] for b in br):
            tr_moves += 1
        wake_clip += rec.get("clip_wake", 0) > 0
        air_clip += len(rec["clip"]) > rec.get("clip_wake", 0)
    for side in (True, False):
        xs = [b["x_tr"] for rec in its for b in rec.get("branches", []) if b["upper"] == side and b["x_tr"] is not None]
        if xs:
            x_tr_range = max(x_tr_range, max(xs) - min(xs))
    return dict(cold=r.trace["cold"], sep_clip=sep_clip, tr_moves=tr_moves, iso_clip=iso_clip, air_clip=air_clip,
                wake_clip=wake_clip, s_tr_range=x_tr_range, iterations=r.iterations,
                iterations_total=r.iterations_total, continuation=r.continuation, reason=r.reason,
                last_residual=r.residual_history[-1] if r.residual_history else math.nan,
                step_limits=r.step_limits)


def classify(f: dict, ignore_cold: bool = False) -> str:
    if f["cold"] and not ignore_cold:
        return CLASSES[0]
    if f["sep_clip"] >= MIN_HITS:
        return CLASSES[1]
    if f["tr_moves"] >= MIN_HITS:
        return CLASSES[2]
    if f["iso_clip"] >= MIN_HITS:
        return CLASSES[3]
    return CLASSES[4]


def work(section: str, reynolds: float) -> tuple:
    _RESULTS.clear()
    t0 = time.perf_counter()
    _, _, rows = vr.run_case(section, reynolds, "fidkowski2022")
    if WRITE_BENCHMARK:                    # the rows of the benchmark itself (same run_case)
        path = vr.case_file(Path(OUTPUT), section, reynolds)
        old = [r for r in vr.read_case(path) if r["method"] != vr.method_name("fidkowski2022")] \
            if path.exists() else []
        vr._write(path, old + rows)
    pts = []
    for row in rows:
        r = _RESULTS.get(row["alpha"])
        p = dict(alpha=row["alpha"], converged=row["converged"] == "1", error=row["error"], path=row["path"])
        if p["converged"] and (r is None or not r.converged):
            # reached afterwards from the nearest converged angle (``viscous.reach_from_nearest``)
            p.update({k: float(row[q]) for k, q in (("cl", "cl"), ("cd", "cd"), ("cd_te", "cd_te"), ("cm", "cm"),
                                                    ("xtr_upper", "xtr_top"), ("xtr_lower", "xtr_bot"))})
            p.update(iterations=int(row["iterations"]), iterations_total=int(row["iterations_total"]),
                     continuation=int(row["continuation"]))
        elif r is not None:
            p.update(cl=r.cl, cd=r.cd, cd_te=r.cd_te, cm=r.cm, xtr_upper=r.xtr_upper, xtr_lower=r.xtr_lower,
                     iterations=r.iterations, iterations_total=r.iterations_total,
                     continuation=r.continuation)
            if not r.converged:
                p["features"] = features(r)
                p["class"] = classify(p["features"])
                p["signature"] = classify(p["features"], ignore_cold=True)
                p["trace_tail"] = r.trace["its"][-WINDOW:]
        pts.append(p)
    return section, reynolds, pts, time.perf_counter() - t0


def summary(path: Path, bench=BENCHMARK) -> dict:
    """{group: {Re: {class: count}}} of the non-converged points counted by the SPEC.md 4.1 metrics."""
    from foil_lab.benchmark.references import load_reference
    from foil_lab.benchmark.report import attached_filter
    data = json.loads(Path(path).read_text())
    out: dict = {}
    for key, v in data.items():
        sec, re = key.split("|")
        re = float(re)
        ref = load_reference(sec, re, bench.pane_nodes[0])
        filt, _ = attached_filter(ref, bench)
        sharp = float(ref.provenance.get("trailing_edge_gap", "nan")) == 0.0
        group = "sharp" if sharp else ("validated" if re in bench.reynolds_validated else "low Re")
        cnt = out.setdefault(group, {}).setdefault(re, {c: 0 for c in CLASSES} | {"points": 0, "cases": [],
                                                                                 "cold signatures": {}})
        for p in v["points"]:
            if p["alpha"] not in filt or ref.rows[p["alpha"]] is None:
                continue
            cnt["points"] += 1
            if not p["converged"]:
                c = p.get("class", "other")
                cnt[c] += 1
                cnt["cases"].append((sec, p["alpha"], c))
                if c == CLASSES[0] and "signature" in p:
                    sg = cnt["cold signatures"]
                    sg[p["signature"]] = sg.get(p["signature"], 0) + 1
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmark_output")
    ap.add_argument("--workers", type=int, default=BENCHMARK.workers)
    ap.add_argument("--sections", default=None)
    ap.add_argument("--reynolds", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--write-benchmark", action="store_true",
                    help="also write the benchmark CSVs (as `python -m foil_lab.benchmark viscous`)")
    ap.add_argument("--file", default="results.json", help="results file in <output>/m3_diag")
    a = ap.parse_args(argv)
    if a.summary:
        for group, by_re in summary(Path(a.output) / "m3_diag" / a.file).items():
            for re, c in sorted(by_re.items()):
                n_fail = sum(c[k] for k in CLASSES)
                print(f"{group:9s} Re {re:7g}: {n_fail:3d} of {c['points']:3d} not converged; "
                      + "; ".join(f"{k} {c[k]}" for k in CLASSES)
                      + (f"; cold-start signatures {c['cold signatures']}" if c["cold signatures"] else ""))
        return
    if a.write_benchmark:                  # read by the spawned workers on import
        os.environ["C3_WRITE_BENCHMARK"] = "1"
        os.environ["C3_OUTPUT"] = a.output
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    secs = a.sections.split(",") if a.sections else BENCHMARK.sections
    res = [float(v) for v in a.reynolds.split(",")] if a.reynolds else \
        BENCHMARK.reynolds_validated + BENCHMARK.reynolds_low
    out = Path(a.output) / "m3_diag"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(work, s, re) for s in secs for re in res]
        for i, fut in enumerate(as_completed(futs), 1):
            s, re, pts, t = fut.result()
            results[f"{s}|{re:g}"] = dict(points=pts, runtime_s=t)
            print(f"[{i}/{len(futs)}] {s} Re {re:g}: {sum(p['converged'] for p in pts)}/{len(pts)} converged, "
                  f"{t:.0f} s", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
