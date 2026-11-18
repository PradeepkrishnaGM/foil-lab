"""Coordinate files: reading (Selig and Lednicer) and writing (Selig)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def parse_coordinates(text: str) -> tuple[np.ndarray, int]:
    """Parse a coordinate file in Selig or Lednicer format; return Selig order.

    Selig: name line, then points from the trailing edge over the upper surface to the leading
    edge and back over the lower surface. Lednicer: name line, a line with the point counts of
    the two surfaces (values above 1), then the upper surface from leading to trailing edge and,
    after a blank line, the lower surface from leading to trailing edge.

    Consecutive duplicate points are removed; their number is returned with the coordinates.
    """
    rows: list[list[float]] = []
    blocks: list[list[list[float]]] = [[]]
    for raw in text.replace("\r", "").splitlines():
        parts = raw.split()
        if not parts:
            if blocks[-1]:
                blocks.append([])
            continue
        try:
            vals = [float(v) for v in parts[:2]]
        except ValueError:
            continue  # name or comment line
        if len(vals) < 2:
            continue
        rows.append(vals)
        blocks[-1].append(vals)
    blocks = [b for b in blocks if b]
    if not rows:
        raise ValueError("no coordinates found")
    if rows[0][0] > 1.5 and rows[0][1] > 1.5:   # Lednicer: point counts first
        n_up, n_lo = int(rows[0][0]), int(rows[0][1])
        data = [b for b in blocks if not (len(b) == 1 and b[0] == rows[0])]
        if len(data) != 2 or len(data[0]) != n_up or len(data[1]) != n_lo:
            raise ValueError("Lednicer file: surface blocks do not match the point counts")
        upper = np.array(data[0])[::-1]          # TE -> LE
        lower = np.array(data[1])                 # LE -> TE
        if np.allclose(upper[-1], lower[0]):
            lower = lower[1:]
        coords = np.vstack((upper, lower))
    else:
        coords = np.array(rows)
    seg = np.hypot(*np.diff(coords, axis=0).T)
    keep = np.concatenate(([True], seg > 0.0))
    return coords[keep], int((~keep).sum())


def load_dat(filepath: str | Path) -> np.ndarray:
    """Read a coordinate file in Selig or Lednicer format; return Selig order
    (:func:`parse_coordinates`; consecutive duplicate points removed)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Airfoil file not found: {filepath}")
    return parse_coordinates(filepath.read_text(errors="replace"))[0]


def save_dat(filepath: str | Path, coords: np.ndarray, header: str = "Foil-Lab airfoil",
             decimals: int = 10) -> None:
    """Write coordinates in Selig format: a name line, then one "x y" pair per line."""
    with open(filepath, "w") as f:
        f.write(f"{header}\n")
        for x, y in np.asarray(coords, dtype=float):
            f.write(f"{x:.{decimals}f} {y:.{decimals}f}\n")


def export_polar_csv(filepath: str | Path, polar) -> None:
    """Coefficients of a v1.3.1-shaped ``analysis.PolarData`` as CSV (used by the GUI until M8;
    the API's exports are in :mod:`foil_lab.export`)."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Alpha (deg)", "Cl", "Cm_LE", "Cm_c/4"])
        for i in range(len(polar.alphas)):
            writer.writerow([f"{polar.alphas[i]:.3f}", f"{polar.cls[i]:.4f}",
                             f"{polar.cm_les[i]:.4f}", f"{polar.cm_qcs[i]:.4f}"])
