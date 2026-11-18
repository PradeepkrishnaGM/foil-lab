"""Inviscid rows of the M1 "before" benchmark run (the v1.3.1 adapter, reduced in M3 C5).

M1 ran four methods of v1.3.1 on the benchmark matrix (DECISIONS.md, M1 Q3 and E2): ``inviscid``,
``oneway`` (one-way boundary layer with Michel transition), ``vii`` (transpiration VII) and
``newton`` (Newton VII). The three viscous methods were deleted with their modules in M3 C5
(DECISIONS.md M3, Q3 and C5); their measured results stay in the committed before pictures
``docs/baseline_v1.3.1.md`` and ``docs/benchmark_m2.md``, and :mod:`.report` still reads their rows
from result files of earlier runs. ``run_baseline`` now writes the inviscid rows only (the
inviscid core of :mod:`foil_lab.inviscid`, loads per unit chord with the moment about (0.25, 0),
as the v1.3.1 interface reported them). The coupled viscous solver is benchmarked by
:mod:`.viscous_run`.

Both codes read the same canonical coordinate file (160 panels).
"""

from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..params import BENCHMARK, SWEEP, BenchmarkParams
from ..utils.io import load_dat

COLUMNS = ("method", "alpha", "converged", "cl", "cd", "cm", "xtr_top", "xtr_bot", "runtime_s",
           "error")


def _alphas() -> list[float]:
    n = int(round((SWEEP.alpha_max - SWEEP.alpha_min) / SWEEP.alpha_step))
    return [round(SWEEP.alpha_min + i * SWEEP.alpha_step, 6) for i in range(n + 1)]


def _run_point(method: str, coords: np.ndarray, alpha: float, reynolds: float,
               bench: BenchmarkParams) -> dict:
    from .. import inviscid as core

    t0 = time.perf_counter()
    try:
        if method != "inviscid":
            raise ValueError(f"method {method!r} was deleted in M3 C5; only 'inviscid' runs")
        s = core.body_loads(core.solve(core.PanelSet.from_bodies([coords]), alpha), 0)
        row = dict(converged="n/a", cl=s.cl, cd=math.nan, cm=s.cm,
                   xtr_top=math.nan, xtr_bot=math.nan)
        row["error"] = ""
    except Exception as exc:  # recorded, never hidden
        row = dict(converged="0", cl=math.nan, cd=math.nan, cm=math.nan, xtr_top=math.nan,
                   xtr_bot=math.nan, error=f"{type(exc).__name__}: {exc}")
    row.update(method=method, alpha=alpha, runtime_s=time.perf_counter() - t0)
    return row


def _run_case(section: str, coord_path: str, reynolds: float, methods: tuple[str, ...],
              alphas: tuple[float, ...], bench: BenchmarkParams) -> tuple[str, float, list[dict]]:
    coords = load_dat(coord_path)
    rows = [_run_point(m, coords, a, reynolds, bench) for m in methods for a in alphas]
    return section, reynolds, rows


def case_file(output_root: Path, section: str, reynolds: float) -> Path:
    """Result file of one (section, Reynolds number) case; Reynolds 0 holds inviscid results."""
    return output_root / "baseline_v131" / f"{section}_Re{int(round(reynolds))}.csv"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["method"], r["alpha"]))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in COLUMNS})


def read_case(path: Path) -> list[dict]:
    """Read a baseline result file (floats parsed, ``nan`` for missing values)."""
    with path.open() as f:
        out = []
        for r in csv.DictReader(f):
            for k in ("alpha", "cl", "cd", "cm", "xtr_top", "xtr_bot", "runtime_s"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else math.nan
            out.append(r)
        return out


def run_baseline(output_root: Path = Path("benchmark_output"), bench: BenchmarkParams = BENCHMARK,
                 workers: int | None = None) -> None:
    """Run the inviscid method on every section of the matrix (the viscous methods of v1.3.1
    were deleted in M3 C5).

    Reads the canonical coordinate files written by the reference generation
    (``output_root/coords``) and writes one CSV per case to ``output_root/baseline_v131``.
    """
    alphas = tuple(_alphas())
    jobs = []
    for sec in bench.sections:
        path = output_root / "coords" / f"{sec}.dat"
        if not path.exists():
            print(f"skipped {sec}: no canonical coordinate file (run 'references' first)")
            continue
        jobs.append((sec, str(path), 0.0, ("inviscid",), alphas))
    results: dict[tuple[str, float], list[dict]] = {}
    with ProcessPoolExecutor(max_workers=workers or bench.workers) as pool:
        futs = [pool.submit(_run_case, s, p, abs(re), m, a, bench) for s, p, re, m, a in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            sec, re, rows = fut.result()
            results.setdefault((sec, re), []).extend(rows)
            print(f"[{i}/{len(futs)}] {sec} Re {re:g}: {len(rows)} points, "
                  f"{sum(r['runtime_s'] for r in rows):.1f} s", flush=True)
    for (sec, re), rows in results.items():
        _write(case_file(output_root, sec, re), rows)
