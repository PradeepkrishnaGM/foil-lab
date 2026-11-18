"""Batch runner for parametric studies (SPEC.md 4.8), on the public analysis API.

    from foil_lab import batch
    import foil_lab as fl

    airfoils = {"2412": fl.Airfoil.naca("2412"), "e387": fl.Airfoil.from_name("e387")}
    cases = batch.build_matrix(airfoils, alpha=range(-4, 11), re=[1e6, 3e6])
    rows = batch.run_batch(cases, airfoils, backend="process", workers=4)
    batch.write_table(rows, "study.parquet")            # .parquet, .h5 / .hdf5, else CSV

Cases that differ only in the angle of attack form a group, solved with one :func:`foil_lab.polar`
call (so viscous angles are reached with warm starts from their neighbours, as in the benchmark).
Backends: ``"serial"`` (default) and ``"process"`` (a process pool started with the ``spawn``
method, which behaves the same on every supported Python version and platform). One failing group
never stops the batch: its rows carry ``converged`` False and the error. A viscous row whose drag is
not plausible (negative, above 1 or not finite) is never converged (DECISIONS.md, M2 check 3).
Rows have ``api.ROW_COLUMNS`` plus ``airfoil``, ``error`` and the case's ``extra`` items, in the order
of the cases.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .api import ROW_COLUMNS
from .solver import drag_is_plausible


@dataclass(frozen=True)
class Case:
    """One operating point: the airfoil's key in the geometry mapping and the conditions of
    :func:`foil_lab.analyze`."""

    airfoil: str
    alpha: float
    re: float = 0.0
    mach: float = 0.0
    sweep: float = 0.0
    aspect_ratio: Optional[float] = None
    ground_height: Optional[float] = None
    n_crit: Optional[float] = None
    trip_upper: Optional[float] = None
    trip_lower: Optional[float] = None
    extra: tuple = field(default=(), compare=False)   # (key, value) pairs copied to the row

    def group(self) -> tuple:
        return (self.airfoil, self.re, self.mach, self.sweep, self.aspect_ratio, self.ground_height,
                self.n_crit, self.trip_upper, self.trip_lower)


def build_matrix(airfoils: Mapping[str, Any], alpha: Iterable[float], re: Iterable[float] = (0.0,),
                 mach: Iterable[float] = (0.0,), sweep: Iterable[float] = (0.0,),
                 aspect_ratio: Iterable[Optional[float]] = (None,),
                 ground_height: Iterable[Optional[float]] = (None,),
                 n_crit: Iterable[Optional[float]] = (None,),
                 trip_upper: Iterable[Optional[float]] = (None,),
                 trip_lower: Iterable[Optional[float]] = (None,)) -> List[Case]:
    """The Cartesian product of the parameters as a list of cases (airfoils by their keys)."""
    alpha = [float(a) for a in alpha]
    out = []
    for name, r, m, sw, ar, gh, nc, tu, tl, a in itertools.product(
            airfoils.keys(), re, mach, sweep, aspect_ratio, ground_height, n_crit, trip_upper,
            trip_lower, alpha):
        out.append(Case(name, a, float(r), float(m), float(sw), ar, gh, nc, tu, tl))
    return out


def _geometry(g):
    from .airfoil import Airfoil, Configuration, Element
    if isinstance(g, (Airfoil, Element, Configuration)):
        return g
    return Airfoil.from_coords(np.asarray(g, dtype=float))


def _empty_row(case: Case, error: str) -> Dict[str, Any]:
    row = {k: None for k in ROW_COLUMNS}
    row.update(alpha=case.alpha, converged=False, reynolds=case.re or None, mach=case.mach)
    return row


def _solve_group(geometry, cases: List[Case]) -> List[Dict[str, Any]]:
    """Worker: one polar for cases that differ only in alpha; rows in the order of ``cases``."""
    import warnings
    from .api import polar
    c0 = cases[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pol = polar(_geometry(geometry), [c.alpha for c in cases], re=c0.re or None, mach=c0.mach,
                        sweep=c0.sweep, aspect_ratio=c0.aspect_ratio, ground_height=c0.ground_height,
                        n_crit=c0.n_crit, trip_upper=c0.trip_upper, trip_lower=c0.trip_lower)
        rows = []
        for c, r in zip(cases, pol.results):
            row = r.row()
            if c.re and not drag_is_plausible(r.cd):
                row["converged"] = False
            row.update(airfoil=c.airfoil, error="", **dict(c.extra))
            rows.append(row)
        return rows
    except Exception as exc:                    # one bad group never stops the batch
        rows = []
        for c in cases:
            row = _empty_row(c, str(exc))
            row.update(airfoil=c.airfoil, error=f"{type(exc).__name__}: {exc}", **dict(c.extra))
            rows.append(row)
        return rows


def run_batch(cases: List[Case], airfoils: Mapping[str, Any], backend: str = "serial",
              workers: Optional[int] = None, progress=None) -> List[Dict[str, Any]]:
    """Solve every case; rows in the order of ``cases``.

    ``airfoils``: key -> ``Airfoil``, ``Element``, ``Configuration`` or an (n, 2) coordinate array.
    ``backend``: ``"serial"`` or ``"process"`` (``workers`` processes, default the CPU count).
    ``progress``: optional callable ``progress(done_groups, total_groups)``.
    A case whose airfoil key is missing gets a row with the error.
    """
    if backend not in ("serial", "process"):
        raise ValueError(f"backend must be 'serial' or 'process', got {backend!r}")
    groups: Dict[tuple, List[int]] = {}
    for i, c in enumerate(cases):
        groups.setdefault(c.group(), []).append(i)
    out: List[Optional[Dict[str, Any]]] = [None] * len(cases)
    jobs = []
    for key, idx in groups.items():
        name = key[0]
        if name not in airfoils:
            for i in idx:
                row = _empty_row(cases[i], "")
                row.update(airfoil=name, error=f"KeyError: no airfoil named {name!r}", **dict(cases[i].extra))
                out[i] = row
            continue
        jobs.append((airfoils[name], idx))
    if backend == "serial":
        for n, (g, idx) in enumerate(jobs, 1):
            for i, row in zip(idx, _solve_group(g, [cases[i] for i in idx])):
                out[i] = row
            if progress:
                progress(n, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(_solve_group, _geometry(g), [cases[i] for i in idx]): idx for g, idx in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                for i, row in zip(futs[fut], fut.result()):
                    out[i] = row
                if progress:
                    progress(n, len(jobs))
    return out


def to_dataframe(rows: List[Dict[str, Any]]):
    """pandas DataFrame of batch rows."""
    from .api import _pandas
    return _pandas().DataFrame(rows)


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
            numeric = all(isinstance(x, (int, float, bool, np.number)) or x is None for x in v)
            if numeric:
                grp.create_dataset(k, data=np.array([np.nan if x is None else float(x) for x in v]))
            else:
                grp.create_dataset(k, data=np.array([("" if x is None else str(x)) for x in v]).astype("S"))


def write_csv(table: List[Dict[str, Any]], path: str) -> None:
    import csv
    cols = _columnar(table)
    keys = list(cols.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(table)):
            w.writerow(["" if cols[k][i] is None else cols[k][i] for k in keys])


def write_table(table: List[Dict[str, Any]], path: str) -> None:
    """Dispatch on file extension: .parquet, .h5/.hdf5, else .csv."""
    lower = str(path).lower()
    if lower.endswith(".parquet"):
        write_parquet(table, path)
    elif lower.endswith((".h5", ".hdf5")):
        write_hdf5(table, path)
    else:
        write_csv(table, path)


# --------------------------------------------------------------------------- #
# JAX-accelerated batch path (inviscid)
# --------------------------------------------------------------------------- #
def run_batch_jax(airfoils: Mapping[str, Any], alphas_deg, mach: float = 0.0) -> List[Dict[str, Any]]:
    """Inviscid Cl and Cm over airfoils x alphas with the JAX core (each airfoil's polar one vmapped,
    compiled call; coordinates used as given, Cm about (0.25, 0)). Without JAX: :func:`run_batch`.
    Rows have the keys ``airfoil``, ``alpha``, ``cl``, ``cm``, ``converged``, ``method``."""
    from .jax_backend import HAS_JAX
    if not HAS_JAX or mach:
        rows = run_batch(build_matrix(airfoils, alpha=alphas_deg, mach=(mach,)), airfoils)
        return [dict(airfoil=r["airfoil"], alpha=r["alpha"], cl=r["cl"], cm=r["cm"],
                     converged=r["converged"], method=r["method"]) for r in rows]
    from . import jax_core
    rows = []
    for name, g in airfoils.items():
        from .airfoil import Configuration
        cfg = Configuration.of(_geometry(g))
        if len(cfg.elements) != 1:
            raise ValueError(f"{name}: the JAX batch path takes single sections")
        coords = np.asarray(cfg.elements[0].coords, dtype=float)
        cls, cms = jax_core.batch_polar(coords, alphas_deg)
        for a, cl, cm in zip(alphas_deg, cls, cms):
            rows.append(dict(airfoil=name, alpha=float(a), cl=float(cl), cm=float(cm),
                             converged=bool(np.isfinite(cl)), method="inviscid (JAX core)"))
    return rows
