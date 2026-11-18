"""Polar database (SPEC.md 4.8): results stored in SQLite, keyed by geometry hash, conditions,
method and code version, with the converged flag.

    import foil_lab as fl
    db = fl.PolarDatabase("polars.sqlite")
    pol = db.polar(fl.Airfoil.naca("2412"), range(-4, 11), re=1e6)   # computes what is missing
    df = db.to_dataframe(name="NACA 2412")

The geometry hash (:func:`geometry_hash`) is the SHA-256 of every element's placed coordinates
(rounded to 1e-12), the reference chord and the moment reference point, so two geometries share
rows only if they are the same section in the same place. A result computed by another Foil-Lab
version is a different row (the version is part of the key); :meth:`PolarDatabase.lookup` only
returns rows of the running version.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .airfoil import Configuration
from .api import Polar, Result

#: condition keys that, with the geometry, the angle, the method and the version, identify a row
CONDITIONS = ("reynolds", "mach", "sweep", "aspect_ratio", "ground_height", "n_crit", "trip_upper",
              "trip_lower", "num_panels")
VALUES = ("converged", "cl", "cd", "cm", "ch", "cd_friction", "cd_pressure", "cd_induced", "xtr_upper",
          "xtr_lower", "mach_critical", "iterations", "path", "corrections", "warnings")


def geometry_hash(geometry) -> str:
    """SHA-256 (hex) of the placed coordinates of every element (rounded to 1e-12), the reference
    chord and the moment reference point."""
    cfg = Configuration.of(geometry)
    h = hashlib.sha256()
    for el in cfg.elements:
        c = np.round(np.asarray(el.placed.coords, dtype=float), 12) + 0.0     # -0.0 -> 0.0
        h.update(np.ascontiguousarray(c).tobytes())
        h.update(b"|")
    chord, point = cfg.reference()
    h.update(np.round(np.array([chord, *point], float), 12).tobytes())
    return h.hexdigest()


def _r(v):
    """Condition value as stored: 9 significant digits (a Reynolds number that went through
    re / chord * chord compares equal), ``None`` kept."""
    return None if v is None else float(f"{float(v):.9g}")


def _conditions(r: Result, num_panels: Optional[int]) -> dict:
    tu, tl = (r.trips or (None, None))
    return _canon(dict(reynolds=r.reynolds or 0.0, mach=r.mach, sweep=r.sweep, aspect_ratio=r.aspect_ratio,
                ground_height=r.ground_height, n_crit=r.n_crit, trip_upper=tu, trip_lower=tl,
                num_panels=num_panels if num_panels is not None else (r.num_panels[0] if r.elements else None)))


def _canon(cond: dict) -> dict:
    return {k: _r(cond.get(k)) for k in CONDITIONS}


def _key(cond: dict) -> str:
    return json.dumps({k: cond.get(k) for k in CONDITIONS}, sort_keys=True)


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


class PolarDatabase:
    """Results in one SQLite file (``":memory:"`` for a temporary database)."""

    def __init__(self, path: Union[str, Path] = ":memory:"):
        self.path = str(path)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        cols = ", ".join(f"{k} REAL" for k in CONDITIONS)
        self.con.execute(
            f"""CREATE TABLE IF NOT EXISTS results (
                geometry_hash TEXT NOT NULL, name TEXT, alpha REAL NOT NULL, {cols},
                conditions TEXT NOT NULL, method TEXT NOT NULL, version TEXT NOT NULL,
                converged INTEGER NOT NULL, cl REAL, cd REAL, cm REAL, ch REAL, cd_friction REAL,
                cd_pressure REAL, cd_induced REAL, xtr_upper REAL, xtr_lower REAL, mach_critical REAL,
                iterations INTEGER, path TEXT, corrections TEXT, warnings TEXT, created TEXT,
                PRIMARY KEY (geometry_hash, conditions, alpha, method, version))""")
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- writing --------------------------------------------------------------------------
    def add(self, result: Union[Result, Polar], geometry, name: Optional[str] = None,
            num_panels: Optional[int] = None) -> int:
        """Store a ``Result`` or every result of a ``Polar`` for ``geometry`` (replacing a row with
        the same key). Returns the number of rows written."""
        results = result.results if isinstance(result, Polar) else (result,)
        gh = geometry_hash(geometry)
        nm = name or Configuration.of(geometry).name
        now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        n = 0
        for r in results:
            cond = _conditions(r, num_panels)
            row = r.row()
            vals = dict(geometry_hash=gh, name=nm, alpha=r.alpha, **{k: cond[k] for k in CONDITIONS},
                        conditions=_key(cond), method=r.method, version=r.version,
                        converged=int(bool(r.converged)),
                        **{k: _num(row.get(k)) for k in VALUES if k not in ("converged",)},
                        created=now)
            names = ", ".join(vals)
            self.con.execute(f"INSERT OR REPLACE INTO results ({names}) VALUES ({', '.join('?' * len(vals))})",
                             tuple(vals.values()))
            n += 1
        self.con.commit()
        return n

    # --- reading --------------------------------------------------------------------------
    def query(self, geometry=None, name: Optional[str] = None, method: Optional[str] = None,
              version: Optional[str] = None, converged: Optional[bool] = None, **conditions) -> list:
        """Rows (dicts, ordered by geometry, conditions and alpha) matching every given filter.
        ``conditions``: any of ``CONDITIONS`` (``None`` matches a missing value)."""
        where, args = [], []
        if geometry is not None:
            where.append("geometry_hash = ?")
            args.append(geometry_hash(geometry))
        for col, v in (("name", name), ("method", method), ("version", version)):
            if v is not None:
                where.append(f"{col} = ?")
                args.append(v)
        if converged is not None:
            where.append("converged = ?")
            args.append(int(converged))
        for k, v in conditions.items():
            if k not in CONDITIONS:
                raise ValueError(f"unknown condition {k!r}; one of {CONDITIONS}")
            if v is None:
                where.append(f"{k} IS NULL")
            else:
                where.append(f"{k} = ?")
                args.append(_r(v))
        sql = "SELECT * FROM results" + (" WHERE " + " AND ".join(where) if where else "") + \
              " ORDER BY geometry_hash, conditions, alpha"
        return [dict(r) for r in self.con.execute(sql, args)]

    def to_dataframe(self, **filters):
        """:meth:`query` as a pandas DataFrame."""
        from .api import _pandas
        return _pandas().DataFrame(self.query(**filters))

    def lookup(self, geometry, alpha: float, method: str, **conditions) -> Optional[dict]:
        """The stored row of the running Foil-Lab version for exactly these inputs, or ``None``."""
        from . import __version__
        cond = _canon(conditions)
        row = self.con.execute(
            "SELECT * FROM results WHERE geometry_hash = ? AND conditions = ? AND alpha = ? AND method = ? "
            "AND version = ?", (geometry_hash(geometry), _key(cond), float(alpha), method, __version__)).fetchone()
        return None if row is None else dict(row)

    def polar(self, geometry, alphas: Iterable[float], re: Optional[float] = None,
              num_panels: Optional[int] = None, **kwargs) -> list:
        """Rows for ``alphas`` (in that order): stored rows of the running version where they exist,
        the missing angles computed with ``fl.polar(geometry, missing, re=re, **kwargs)`` and stored
        first. ``kwargs`` are the conditions of :func:`foil_lab.polar` (``mach``, ``n_crit``, ...)."""
        from .api import METHOD, METHOD_VISCOUS, polar
        from .params import GEOMETRY, VISCOUS
        alphas = [float(a) for a in alphas]
        n = GEOMETRY.num_panels if num_panels is None else int(num_panels)
        method = METHOD_VISCOUS if re else METHOD
        cond = dict(reynolds=float(re or 0.0), mach=float(kwargs.get("mach", 0.0)),
                    sweep=float(kwargs.get("sweep", 0.0)), aspect_ratio=kwargs.get("aspect_ratio"),
                    ground_height=kwargs.get("ground_height"),
                    n_crit=(kwargs.get("n_crit") or VISCOUS.n_crit) if re else None,
                    trip_upper=kwargs.get("trip_upper") or None if re else None,
                    trip_lower=kwargs.get("trip_lower") or None if re else None, num_panels=n)
        missing = [a for a in alphas if self.lookup(geometry, a, method, **cond) is None]
        if missing:
            self.add(polar(geometry, missing, re=re, num_panels=n, **kwargs), geometry, num_panels=n)
        return [self.lookup(geometry, a, method, **cond) for a in alphas]
