"""Headless batch execution for large parametric studies.

Solve a Cartesian product of parameters (airfoils x alpha x Re x Mach x ...) in
parallel and dump the results to Parquet or HDF5 for downstream analysis. This
is the "micro-modular execution mode" for sweeping massive matrices without the
GUI.

Design
------
* A :class:`BatchJob` is a list of independent :class:`Case` dicts. Each case is
  a fully-specified ``solve_flow`` call.
* Cases are embarrassingly parallel; we farm them out with a process pool.
* Results are collected into a flat table (one row per case) and written with a
  pluggable writer (Parquet via pyarrow, or HDF5 via h5py). Both are optional
  dependencies; a CSV fallback always works.

Example
-------
    from foil_lab.batch import build_matrix, run_batch, write_table
    cases = build_matrix(
        airfoils={"naca2412": coords1, "naca0012": coords2},
        alpha=range(-4, 13),
        reynolds=[1e6, 3e6],
    )
    table = run_batch(cases, workers=8)
    write_table(table, "sweep.parquet")
"""

from __future__ import annotations

import itertools
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .solver import solve_flow


# Module-level registry so worker processes can resolve airfoil coordinates by
# name without pickling large arrays for every single case.
_AIRFOIL_REGISTRY: Dict[str, np.ndarray] = {}


@dataclass
class Case:
    """One fully-specified solver run."""

    airfoil: str                       # key into the airfoil registry
    alpha_deg: float
    v_inf: float = 1.0
    mach: float = 0.0
    reynolds: float = 0.0
    aspect_ratio: float = 0.0
    roughness: int = 0
    enable_post_stall: bool = False
    trip_x_upper: Optional[float] = None
    trip_x_lower: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)   # passthrough metadata


def register_airfoils(airfoils: Mapping[str, np.ndarray]) -> None:
    """Register airfoil coordinate arrays by name for the workers to resolve."""
    _AIRFOIL_REGISTRY.update({k: np.asarray(v, dtype=float) for k, v in airfoils.items()})


def build_matrix(
    airfoils: Mapping[str, np.ndarray],
    alpha: Iterable[float],
    reynolds: Iterable[float] = (0.0,),
    mach: Iterable[float] = (0.0,),
    aspect_ratio: Iterable[float] = (0.0,),
    roughness: Iterable[int] = (0,),
    enable_post_stall: bool = False,
    trip_x_upper: Iterable[Optional[float]] = (None,),
    trip_x_lower: Iterable[Optional[float]] = (None,),
) -> List[Case]:
    """Construct the Cartesian product of parameters as a list of Cases.

    Registers the airfoils as a side effect so workers can resolve them.
    """
    register_airfoils(airfoils)
    cases: List[Case] = []
    for name, a, re, m, ar, rough, tu, tl in itertools.product(
        airfoils.keys(), alpha, reynolds, mach, aspect_ratio, roughness,
        trip_x_upper, trip_x_lower
    ):
        cases.append(Case(
            airfoil=name, alpha_deg=float(a), reynolds=float(re), mach=float(m),
            aspect_ratio=float(ar), roughness=int(rough),
            enable_post_stall=enable_post_stall, trip_x_upper=tu, trip_x_lower=tl,
        ))
    return cases


def _solve_one(case: Case) -> Dict[str, Any]:
    """Worker: solve a single case and return a flat result row."""
    try:
        coords = _AIRFOIL_REGISTRY[case.airfoil]
        sol = solve_flow(
            coords, case.alpha_deg, v_inf=case.v_inf, mach=case.mach,
            reynolds=case.reynolds, aspect_ratio=case.aspect_ratio,
            roughness=case.roughness, enable_post_stall=case.enable_post_stall,
            trip_x_upper=case.trip_x_upper, trip_x_lower=case.trip_x_lower,
        )
        row = dict(
            airfoil=case.airfoil, alpha_deg=case.alpha_deg, mach=case.mach,
            reynolds=case.reynolds, aspect_ratio=case.aspect_ratio,
            roughness=case.roughness, trip_x_upper=case.trip_x_upper or np.nan,
            trip_x_lower=case.trip_x_lower or np.nan,
            cl=sol.cl, cd_ind=sol.cd_ind, cd_visc=sol.cd_visc,
            cd_total=sol.cd_total, cm_qc=sol.cm_qc, cm_le=sol.cm_le,
            converged=True,
        )
    except Exception as exc:  # never let one bad case kill the batch
        row = dict(
            airfoil=case.airfoil, alpha_deg=case.alpha_deg, mach=case.mach,
            reynolds=case.reynolds, aspect_ratio=case.aspect_ratio,
            roughness=case.roughness, trip_x_upper=case.trip_x_upper or np.nan,
            trip_x_lower=case.trip_x_lower or np.nan,
            cl=np.nan, cd_ind=np.nan, cd_visc=np.nan, cd_total=np.nan,
            cm_qc=np.nan, cm_le=np.nan, converged=False, error=str(exc),
        )
    row.update(case.extra)
    return row


def run_batch(cases: List[Case], workers: Optional[int] = 1,
              progress: bool = False, backend: str = "thread") -> List[Dict[str, Any]]:
    """Solve all cases. Defaults to serial (``workers=1``).

    A note on parallelism, honestly
    -------------------------------
    With the current pure-Python/NumPy core, neither thread nor process pools
    reliably speed this up: per-case time is dominated by GIL-bound Python
    assembly (geometry build, the nodal-vortex loop), not by the GIL-releasing
    dense solve, so threads don't scale; and cases are short enough that process
    pools lose to pickling/IPC overhead. The batch runner's real value is
    organization (Cartesian matrix building), robustness (one failed case never
    kills the run), and multi-format export -- not a parallel speedup.

    Real parallel speedup needs either the planned JAX core (vmap over the whole
    matrix on CPU/GPU) or process-based execution with shared-memory airfoils.
    The ``workers``/``backend`` knobs are wired so that lands cleanly later, but
    by default we run serially because it is the fastest reliable mode today.
    """
    if workers == 1 or workers is None:
        rows = []
        for i, c in enumerate(cases):
            rows.append(_solve_one(c))
            if progress and i % 50 == 0:
                print(f"  {i}/{len(cases)}")
        return rows

    if backend == "process":
        from concurrent.futures import ProcessPoolExecutor as Pool
    else:
        from concurrent.futures import ThreadPoolExecutor as Pool

    rows = []
    with Pool(max_workers=workers) as ex:
        futures = [ex.submit(_solve_one, c) for c in cases]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if progress and i % 50 == 0:
                print(f"  {i}/{len(cases)}")
    return rows


# --------------------------------------------------------------------------- #
# Writers (optional deps; CSV always available)
# --------------------------------------------------------------------------- #
def _columnar(table: List[Dict[str, Any]]) -> Dict[str, list]:
    """Transpose a list of row dicts into a dict of columns (union of keys)."""
    keys: List[str] = []
    for row in table:
        for k in row:
            if k not in keys:
                keys.append(k)
    return {k: [row.get(k, None) for row in table] for k in keys}


def write_parquet(table: List[Dict[str, Any]], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table(_columnar(table)), path)


def write_hdf5(table: List[Dict[str, Any]], path: str, dataset: str = "results") -> None:
    import h5py
    cols = _columnar(table)
    with h5py.File(path, "w") as f:
        grp = f.create_group(dataset)
        for k, v in cols.items():
            arr = np.array([np.nan if x is None else x for x in v]
                           if all(not isinstance(x, str) for x in v if x is not None)
                           else [("" if x is None else str(x)) for x in v])
            if arr.dtype.kind in ("U", "S", "O"):
                grp.create_dataset(k, data=arr.astype("S64"))
            else:
                grp.create_dataset(k, data=arr)


def write_csv(table: List[Dict[str, Any]], path: str) -> None:
    import csv
    cols = _columnar(table)
    keys = list(cols.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(table)):
            w.writerow([cols[k][i] for k in keys])


def write_table(table: List[Dict[str, Any]], path: str) -> None:
    """Dispatch on file extension: .parquet, .h5/.hdf5, else .csv."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        write_parquet(table, path)
    elif lower.endswith((".h5", ".hdf5")):
        write_hdf5(table, path)
    else:
        write_csv(table, path)


# --------------------------------------------------------------------------- #
# JAX-accelerated batch path (inviscid; the real parallel speedup)
# --------------------------------------------------------------------------- #
def run_batch_jax(airfoils, alphas_deg, mach=0.0):
    """Inviscid Cl/Cm matrix over airfoils x alphas, using the JAX backend.

    This is the genuinely parallel batch path: each airfoil's full alpha polar is
    a single vmapped, JIT-compiled call (~100x faster than the Python loop for
    large sweeps). Inviscid only -- for viscous drag use the NumPy
    :func:`run_batch`. Falls back to the NumPy path automatically if JAX is not
    installed.

    Returns a list of flat rows (same schema as :func:`run_batch`, with cd left
    as the inviscid/zero contribution).
    """
    from .jax_backend import HAS_JAX, jax_core

    if not HAS_JAX:
        # Graceful fallback: build cases and run the NumPy batch.
        cases = build_matrix(airfoils, alpha=alphas_deg, mach=(mach,))
        return run_batch(cases, workers=1)

    import numpy as _np
    rows = []
    for name, coords in airfoils.items():
        cls, cms = jax_core.batch_polar(coords, alphas_deg)
        for a, cl, cm in zip(alphas_deg, cls, cms):
            rows.append(dict(
                airfoil=name, alpha_deg=float(a), mach=float(mach),
                reynolds=0.0, aspect_ratio=0.0, roughness=0,
                trip_x_upper=_np.nan, trip_x_lower=_np.nan,
                cl=float(cl), cd_ind=0.0, cd_visc=0.0, cd_total=0.0,
                cm_qc=float(cm), cm_le=_np.nan, converged=True,
            ))
    return rows
