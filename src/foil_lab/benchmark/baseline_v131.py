"""Run Foil-Lab v1.3.1, unchanged, on the benchmark matrix (the "before" picture of M1).

Methods (DECISIONS.md, M1 Q3 and E2):

* ``inviscid``: ``solver.solve_flow`` without Reynolds number (Cl, Cm).
* ``oneway``: ``solver.solve_flow`` with Reynolds number (default viscous path, one-way
  boundary layer with Michel transition).
* ``vii``: ``vii.solve_vii`` with the v1.3.1 command-line settings (``max_iter=200``,
  ``tol=1e-3``, e^n transition, N_crit 9).
* ``newton``: ``vii_newton.solve_vii_newton`` (``max_iter=60``, ``tol=1e-4``) on a small subset.

Both codes read the same canonical coordinate file (160 panels). Transition locations are not
exposed by v1.3.1; they are captured here by wrapping internal boundary-layer functions inside
the worker process only (M1 Q15). No library code is changed. This adapter is removed when the
v1.3.1 solvers are replaced.
"""

from __future__ import annotations

import csv
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..params import BENCHMARK, SWEEP, BenchmarkParams
from ..utils.io import load_dat

COLUMNS = ("method", "alpha", "converged", "cl", "cd", "cm", "xtr_top", "xtr_bot", "runtime_s",
           "error")


def _alphas() -> list[float]:
    n = int(round((SWEEP.alpha_max - SWEEP.alpha_min) / SWEEP.alpha_step))
    return [round(SWEEP.alpha_min + i * SWEEP.alpha_step, 6) for i in range(n + 1)]


@contextmanager
def _patched(module: object, name: str, wrapper: Callable) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, wrapper(original))
    try:
        yield
    finally:
        setattr(module, name, original)


def _branch_xtr(mid_x: np.ndarray, mid_y: np.ndarray, branches: list[tuple[np.ndarray, int]]
                ) -> tuple[float, float]:
    """Map (panel indices of a branch, transition station) pairs to (x_tr upper, x_tr lower)."""
    (ia, ta), (ib, tb) = branches
    xa, xb = float(mid_x[ia][ta]), float(mid_x[ib][tb])
    a_upper = np.mean(mid_y[ia]) >= np.mean(mid_y[ib])
    return (xa, xb) if a_upper else (xb, xa)


def _oneway(coords: np.ndarray, alpha: float, reynolds: float) -> dict:
    from .. import boundary_layer, solver

    calls: dict = {"drag_args": None, "tidx": []}

    def wrap_drag(fn):
        def inner(mid_x, mid_y, lengths, vt, Re, **kw):
            calls["drag_args"] = (np.asarray(mid_x), np.asarray(mid_y), np.asarray(vt))
            return fn(mid_x, mid_y, lengths, vt, Re, **kw)
        return inner

    def wrap_bl(fn):
        def inner(*a, **kw):
            out = fn(*a, **kw)
            calls["tidx"].append(int(out[1]))
            return out
        return inner

    with _patched(solver, "calculate_viscous_drag", wrap_drag), \
            _patched(boundary_layer, "analyze_boundary_layer", wrap_bl):
        s = solver.solve_flow(coords, alpha, reynolds=reynolds)
    xt = xb = math.nan
    if calls["drag_args"] is not None and len(calls["tidx"]) >= 2:
        mid_x, mid_y, vt = calls["drag_args"]
        n = vt.size
        guard = max(2, n // 20)  # same stagnation search as boundary_layer.calculate_viscous_drag
        stag = int(np.argmin(np.abs(vt[guard:n - guard]))) + guard
        ia, ib = np.arange(stag, -1, -1), np.arange(stag, n)
        xt, xb = _branch_xtr(mid_x, mid_y, [(ia, calls["tidx"][-2]), (ib, calls["tidx"][-1])])
    ok = all(math.isfinite(v) for v in (s.cl, s.cd_total, s.cm_qc))
    return dict(converged="n/a" if ok else "0", cl=s.cl, cd=s.cd_total, cm=s.cm_qc,
                xtr_top=xt, xtr_bot=xb)


def _coupled(coords: np.ndarray, alpha: float, reynolds: float, newton: bool,
             bench: BenchmarkParams) -> dict:
    from .. import vii, vii_newton
    from ..panels import build_panels

    mod = vii_newton if newton else vii
    calls: dict = {"split": [], "prof": []}

    def wrap_split(fn):
        def inner(geom, vt):
            out = fn(geom, vt)
            calls["split"].append((out[0], out[3]))
            return out
        return inner

    def wrap_march(fn):
        def inner(*a, **kw):
            prof = fn(*a, **kw)
            calls["prof"].append(int(prof.transition_idx))
            return prof
        return inner

    with _patched(mod, "_split_surfaces", wrap_split), \
            _patched(mod, "march_boundary_layer", wrap_march), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if newton:
            r = vii_newton.solve_vii_newton(coords, alpha, reynolds=reynolds,
                                            max_iter=bench.newton_max_iter, tol=bench.newton_tol)
        else:
            r = vii.solve_vii(coords, alpha, reynolds=reynolds, max_iter=bench.vii_max_iter,
                              tol=bench.vii_tol, transition_model="en", n_crit=9.0)
    g = build_panels(coords)
    (ia, ib), (ta, tb) = calls["split"][-1], calls["prof"][-2:]
    xt, xb = _branch_xtr(g.mid_x, g.mid_y, [(ia, ta), (ib, tb)])
    return dict(converged="1" if r.converged else "0", cl=r.solution.cl, cd=r.cd_wake,
                cm=r.solution.cm_qc, xtr_top=xt, xtr_bot=xb)


def _run_point(method: str, coords: np.ndarray, alpha: float, reynolds: float,
               bench: BenchmarkParams) -> dict:
    from ..solver import solve_flow

    t0 = time.perf_counter()
    try:
        if method == "inviscid":
            s = solve_flow(coords, alpha)
            row = dict(converged="n/a", cl=s.cl, cd=math.nan, cm=s.cm_qc,
                       xtr_top=math.nan, xtr_bot=math.nan)
        elif method == "oneway":
            row = _oneway(coords, alpha, reynolds)
        else:
            row = _coupled(coords, alpha, reynolds, method == "newton", bench)
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
    """Run all v1.3.1 methods on every section and Reynolds number of the matrix.

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
        for re in bench.reynolds_low + bench.reynolds_validated:
            jobs.append((sec, str(path), re, ("oneway", "vii"), alphas))
        if sec in bench.newton_sections:
            for re in bench.newton_reynolds:
                jobs.append((sec, str(path), -re, ("newton",), bench.newton_alphas))
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
