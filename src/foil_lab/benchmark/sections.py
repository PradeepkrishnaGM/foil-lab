"""Benchmark sections and their canonical coordinates.

NACA sections come from Foil-Lab's own generators with the open trailing edge of NACA
Report 460 (DECISIONS.md, M1 D1). Non-NACA sections are downloaded on demand from the UIUC
Airfoil Coordinates Database into a cache outside the repository; only their URL and the
SHA-256 of the downloaded file are part of Foil-Lab. Coordinate data is not bundled.

The canonical coordinates of a section are what both Foil-Lab and XFOIL read in the
benchmark (written by :func:`foil_lab.benchmark.xfoil.write_coordinate_file`).
"""

from __future__ import annotations

import hashlib
import re
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..generators import generate_naca4, generate_naca5
from ..geometry import normalize_airfoil, repanel_cosine
from ..params import BENCHMARK


class SectionUnavailable(RuntimeError):
    """A downloadable section is neither cached nor downloadable (for example, no network)."""


@dataclass(frozen=True)
class SectionSpec:
    """Where a benchmark section comes from."""

    name: str
    kind: str                 # "naca4", "naca5", "uiuc" or "gap" (a variant of another section)
    code: str = ""            # NACA digits
    url: str = ""             # download URL (UIUC sections)
    sha256: str = ""          # SHA-256 of the downloaded file, as first retrieved 2026-10-08
    base: str = ""            # "gap": the section the variant is made from
    te_gap: float = 0.0       # "gap": trailing-edge gap set with geometry.set_trailing_edge_gap


SECTIONS: dict[str, SectionSpec] = {
    "NACA0012": SectionSpec("NACA0012", "naca4", code="0012"),
    "NACA2412": SectionSpec("NACA2412", "naca4", code="2412"),
    "NACA4412": SectionSpec("NACA4412", "naca4", code="4412"),
    "NACA23012": SectionSpec("NACA23012", "naca5", code="23012"),
    "CLARKY": SectionSpec(
        "CLARKY", "uiuc", url="https://m-selig.ae.illinois.edu/ads/coord/clarky.dat",
        sha256="bd1822f02d71bf66357774ec58279ff26f75f911414cded2fb34cafd15935f13"),
    "E387": SectionSpec(
        "E387", "uiuc", url="https://m-selig.ae.illinois.edu/ads/coord/e387.dat",
        sha256="98aa8a4076edbfd1c52e3a2d0cf17b1ef1cbc51ae5da3ab8e5361047b89ee523"),
    # E387 with a 0.25 % chord trailing-edge gap (M2 plan point 5; DECISIONS.md M2 step 5)
    "E387_GAP025": SectionSpec("E387_GAP025", "gap", base="E387", te_gap=0.0025),
}


@dataclass(frozen=True)
class CanonicalSection:
    """Canonical coordinates of one section and how they were made."""

    name: str
    coords: np.ndarray
    source: str                       # free text for the provenance header
    trailing_edge_gap: float          # distance between first and last point, chord units
    repanel_deviation: float | None   # max distance of new points to the original polyline
    removed_duplicates: int           # consecutive duplicate points dropped from the download
    hinge: tuple[float, float] | None = None     # flap cases: hinge point in the file's coordinates
    flap_panels: np.ndarray | None = None        # flap cases: True for the panels of the flap


def cache_dir() -> Path:
    """Download cache: ``$FOIL_LAB_CACHE`` or ``~/.cache/foil-lab``, plus ``airfoils/``."""
    root = os.environ.get("FOIL_LAB_CACHE") or Path.home() / ".cache" / "foil-lab"
    return Path(root) / "airfoils"


def fetch(spec: SectionSpec, timeout_s: float = 30.0) -> bytes:
    """Return the downloaded coordinate file for a UIUC section, using the cache if possible.

    The file's SHA-256 must match ``spec.sha256``; a mismatch is an error, so a changed
    upstream file is noticed rather than silently used.

    Raises
    ------
    SectionUnavailable
        If the file is not cached and cannot be downloaded.
    ValueError
        If the checksum does not match.
    """
    path = cache_dir() / f"{spec.name.lower()}.dat"
    if path.exists():
        data = path.read_bytes()
    else:
        try:
            with urllib.request.urlopen(spec.url, timeout=timeout_s) as resp:
                data = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise SectionUnavailable(f"{spec.name}: not cached and download failed ({exc})") from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec.sha256:
            raise ValueError(f"{spec.name}: checksum {digest} does not match {spec.sha256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"{spec.name}: cached file checksum {digest} does not match {spec.sha256}")
    return data


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


def _polyline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of each point to the nearest segment of the polyline ``poly``."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.einsum("pij,ij->pi", points[:, None, :] - a[None], ab) / np.einsum("ij,ij->i", ab, ab)
    t = np.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(points[:, None, :] - proj, axis=2), axis=1)


_FLAP_NAME = re.compile(r"(.+)_F(\d+)")


def flap_name(base: str, deflection: float) -> str:
    """Name of a flap case: ``flap_name("NACA2412", 10.0) == "NACA2412_F10"``."""
    return f"{base}_F{int(round(deflection)):02d}"


def _flap_canonical(name: str, base: str, deflection: float, num_panels: int) -> CanonicalSection:
    """Canonical section of a flap case: the canonical base section with a plain flap."""
    from ..airfoil import Airfoil
    from ..params import BENCHMARK
    sec = canonical(base, num_panels)
    af = Airfoil.from_coords(sec.coords, base).with_flap(BENCHMARK.flap_hinge_x, deflection,
                                                          BENCHMARK.flap_hinge_y)
    src = (f"canonical {base} ({sec.source}), then Airfoil.with_flap(hinge_x={BENCHMARK.flap_hinge_x:g}, "
           f"deflection={deflection:g}, hinge_y={BENCHMARK.flap_hinge_y:g}); hinge point "
           f"({af.hinge[0]:.6f}, {af.hinge[1]:.6f})")
    gap = float(np.hypot(*(af.coords[0] - af.coords[-1])))
    return CanonicalSection(name, np.asarray(af.coords, dtype=float), src, gap, sec.repanel_deviation,
                            sec.removed_duplicates, (float(af.hinge[0]), float(af.hinge[1])),
                            np.asarray(af.flap.flap_panels))


def canonical(name: str, num_panels: int = BENCHMARK.num_panels) -> CanonicalSection:
    """Canonical coordinates of a benchmark section (``num_panels`` panels, Selig order).

    NACA sections: Foil-Lab generator with the open trailing edge. UIUC sections: downloaded
    file, converted to Selig order, normalised to unit chord and repanelled with the v1.3.1
    cosine repanelling (``geometry.repanel_cosine``). The largest distance of the repanelled
    points to the original polyline is recorded.

    Raises
    ------
    SectionUnavailable
        For a UIUC section that is not cached and cannot be downloaded.
    """
    m = _FLAP_NAME.fullmatch(name)
    if m:
        return _flap_canonical(name, m.group(1), float(m.group(2)), num_panels)
    spec = SECTIONS[name]
    if spec.kind == "gap":
        from ..geometry import set_trailing_edge_gap
        base = canonical(spec.base, num_panels)
        coords = set_trailing_edge_gap(base.coords, spec.te_gap)
        gap = float(np.hypot(*(coords[0] - coords[-1])))
        src = (f"canonical {spec.base} ({base.source}), then "
               f"geometry.set_trailing_edge_gap({spec.te_gap:g})")
        return CanonicalSection(name, np.asarray(coords, dtype=float), src, gap,
                                base.repanel_deviation, base.removed_duplicates)
    if spec.kind == "naca4":
        coords = generate_naca4(spec.code, num_panels, trailing_edge="open")
        src, dev, dups = (f"foil_lab.generators.generate_naca4('{spec.code}', {num_panels}, "
                          f"trailing_edge='open')"), None, 0
    elif spec.kind == "naca5":
        coords = generate_naca5(spec.code, num_panels, trailing_edge="open")
        src, dev, dups = (f"foil_lab.generators.generate_naca5('{spec.code}', {num_panels}, "
                          f"trailing_edge='open')"), None, 0
    else:
        raw, dups = parse_coordinates(fetch(spec).decode("ascii", errors="replace"))
        norm = normalize_airfoil(raw)
        coords = repanel_cosine(norm, num_panels=num_panels)
        dev = float(np.max(_polyline_distance(coords, norm)))
        src = (f"{spec.url} (sha256 {spec.sha256[:16]}...), Selig order, "
               f"normalize_airfoil + repanel_cosine({num_panels}) (arc-length, M2)")
    gap = float(np.hypot(*(coords[0] - coords[-1])))
    return CanonicalSection(name, np.asarray(coords, dtype=float), src, gap, dev, dups)
