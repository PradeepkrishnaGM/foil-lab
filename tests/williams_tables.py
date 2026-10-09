"""Williams (ARC R&M 3717, 1973) two-element test case from the transcribed tables (option b).

Test helper. The tabulated Table 2 geometry (xi, eta) of each element is used directly: a cubic
spline in arc length through the tabulated points, starting and ending at the trailing-edge
row (a sharp trailing edge, so the corner is kept), repanelled with cosine clustering on each
side of the leading edge. Data files and their page numbers: ``tests/data/williams/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

DATA = Path(__file__).parent / "data" / "williams"
#: configuration letter -> Table 3 row key in tables_1_3_forces.csv
CONFIG = {"A": "T3_flap30", "B": "T3_flap10"}


def read_table2(config: str, element: str) -> np.ndarray:
    """Rows (xi, eta, cp) of Table 2 in printed order; the last row is the trailing edge."""
    path = DATA / f"table2{config.lower()}_{element}.csv"
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")][1:]
    return np.array([[float(v) for v in r.split(",")] for r in rows])


def read_forces() -> dict[tuple[str, str, str], float]:
    out = {}
    for ln in (DATA / "tables_1_3_forces.csv").read_text().splitlines():
        if ln and not ln.startswith("#") and not ln.startswith("table,"):
            t, q, e, v = ln.split(",")
            out[(t, q, e)] = float(v)
    return out


@dataclass
class Element:
    """One tabulated element as a closed curve through its trailing-edge point."""

    table: np.ndarray            # (n, 3) printed rows
    s_knots: np.ndarray          # arc length of [TE, rows 1..n-1, TE]
    spline: tuple[CubicSpline, CubicSpline]
    s_le: float                  # arc length of the leading edge (farthest point from the TE)

    @classmethod
    def from_table(cls, table: np.ndarray) -> "Element":
        pts = np.vstack([table[-1, :2], table[:-1, :2], table[-1, :2]])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        sx, sy = CubicSpline(s, pts[:, 0]), CubicSpline(s, pts[:, 1])
        fine = np.linspace(0.0, s[-1], 20001)
        d = np.hypot(sx(fine) - pts[0, 0], sy(fine) - pts[0, 1])
        return cls(table, s, (sx, sy), float(fine[np.argmax(d)]))

    @property
    def s_table(self) -> np.ndarray:
        """Arc length of each printed row (the trailing-edge row at the end of the curve)."""
        return np.concatenate([self.s_knots[1:-1], [self.s_knots[-1]]])

    def nodes(self, num_panels: int) -> np.ndarray:
        """Closed node array (num_panels + 1 nodes), cosine-clustered at both edges."""
        h = num_panels // 2
        t1 = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, h + 1)))
        t2 = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, num_panels - h + 1)))
        s = np.concatenate([self.s_le * t1, self.s_le + (self.s_knots[-1] - self.s_le) * t2[1:]])
        return np.column_stack([self.spline[0](s), self.spline[1](s)]), s

    def x_over_c(self, xy: np.ndarray) -> np.ndarray:
        """Position along the element's own chord (leading edge 0, trailing edge 1)."""
        te = self.table[-1, :2]
        le = np.array([self.spline[0](self.s_le), self.spline[1](self.s_le)])
        ch = te - le
        return (xy - le) @ ch / (ch @ ch)
