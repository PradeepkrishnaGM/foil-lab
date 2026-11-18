"""Exports (SPEC.md 4.8): CSV and JSON for results, XFOIL-style polar text, coordinate files.

    from foil_lab import export
    export.to_csv(pol, "polar.csv")            # Polar, Result, FlapPolarSet or a list of Results
    export.to_json(pol, "polar.json")
    export.to_polar_text(pol, "polar.pol")     # column layout of an XFOIL polar save file
    export.save_coordinates(airfoil, "section.dat")

None of these needs pandas.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Union

import numpy as np

from .api import ROW_COLUMNS, Polar, Result

PathLike = Union[str, Path]


def _rows(obj) -> list:
    from .flap_polars import FlapPolarSet, FlapSlice
    if isinstance(obj, Result):
        return [obj.row()]
    if isinstance(obj, Polar):
        return obj.rows()
    if isinstance(obj, FlapPolarSet):
        return [
            dict(deflection=float(d), alpha=float(a),
                 **{q: float(obj.tables[q][i, j]) for q in ("cl", "cd", "cm", "ch")})
            for i, d in enumerate(obj.deflections) for j, a in enumerate(obj.alphas)]
    if isinstance(obj, FlapSlice):
        return obj.to_dicts()
    if isinstance(obj, (list, tuple)) and all(isinstance(r, Result) for r in obj):
        return [r.row() for r in obj]
    raise TypeError(f"cannot export {type(obj).__name__}: expected Result, Polar, FlapPolarSet, "
                    "FlapSlice or a list of Results")


def _clean(v):
    """JSON-safe value: nan and inf become null, numpy scalars plain numbers."""
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def to_csv(obj, path: PathLike) -> Path:
    """One row per operating point (``api.ROW_COLUMNS``; flap sets: deflection, alpha, cl, cd, cm,
    ch). Not converged points are kept, with ``converged`` False."""
    rows = _rows(obj)
    cols = list(ROW_COLUMNS) if rows and "path" in rows[0] else list(rows[0].keys()) if rows else []
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


def to_json(obj, path: PathLike) -> Path:
    """``{"foil_lab": version, "written": UTC time, "rows": [...]}``; a single ``Result`` also gets
    its per-element loads. nan is written as null."""
    from . import __version__
    doc = dict(foil_lab=__version__, written=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               rows=_rows(obj))
    if isinstance(obj, Result):
        doc["elements"] = obj.to_dict()["elements"]
    path = Path(path)
    path.write_text(json.dumps(_clean(doc), indent=1))
    return path


def polar_text(polar: Polar, name: str = "airfoil") -> str:
    """A polar as text in the column layout of an XFOIL polar save file (alpha, CL, CD, CDp, CM,
    Top_Xtr, Bot_Xtr), so tools that read those files can read it. The header says Foil-Lab wrote
    it. Only converged points are written (as XFOIL does); an inviscid polar has CD = CDp = 0 and
    transition at 1."""
    from . import __version__
    rs = [r for r in polar.results if r.converged]
    re_ = rs[0].reynolds if rs and rs[0].reynolds else 0.0
    mach = rs[0].mach if rs else 0.0
    ncrit = rs[0].n_crit if rs and rs[0].n_crit is not None else 9.0
    mant, expo = (f"{re_:.3e}".split("e") if re_ else ("0.000", "0"))
    lines = [f" Foil-Lab {__version__} polar (XFOIL polar-file column layout)", "",
             f" Calculated polar for: {name}", "",
             " 1 1 Reynolds number fixed          Mach number fixed", "",
             " xtrf =   1.000 (top)        1.000 (bottom)",
             f" Mach = {mach:7.3f}     Re = {float(mant):9.3f} e {int(expo)}     Ncrit = {ncrit:7.3f}", "",
             "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr",
             "  ------ -------- --------- --------- -------- -------- --------"]
    for r in rs:
        cd = r.cd if r.cd is not None else 0.0
        cdp = r.cd_pressure if r.cd_pressure is not None else 0.0
        xu = r.xtr_upper if r.xtr_upper is not None and math.isfinite(r.xtr_upper) else 1.0
        xl = r.xtr_lower if r.xtr_lower is not None and math.isfinite(r.xtr_lower) else 1.0
        lines.append(f"{r.alpha:8.3f}{r.cl:9.4f}{cd:10.5f}{cdp:10.5f}{r.cm:9.4f}{xu:9.4f}{xl:9.4f}")
    return "\n".join(lines) + "\n"


def to_polar_text(polar: Polar, path: PathLike, name: str = "airfoil") -> Path:
    """Write :func:`polar_text` to ``path``."""
    path = Path(path)
    path.write_text(polar_text(polar, name))
    return path


def save_coordinates(airfoil, path: PathLike) -> Path:
    """Selig-format coordinate file of an ``Airfoil`` (name line, then x y pairs)."""
    from .utils.io import save_dat
    path = Path(path)
    save_dat(path, np.asarray(airfoil.coords), airfoil.name)
    return path
