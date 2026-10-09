"""Geometry model: ``Airfoil``, ``Element`` and ``Configuration``.

Coordinates are in Selig order (trailing edge, upper surface, leading edge, lower surface,
trailing edge), as in SPEC.md section 3. All operations return new objects; the coordinate
arrays are read-only. Operations that depend on the shape (repanelling, trailing-edge gap,
thickness and camber scaling) work in the section's own chord frame and do not depend on its
position, rotation or scale (see :mod:`foil_lab.geometry`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import geometry as geo
from .params import GEOMETRY


def _frozen(coords) -> np.ndarray:
    c = np.array(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 4:
        raise ValueError(f"coordinates must be an (n, 2) array with n >= 4, got {c.shape}")
    c.setflags(write=False)
    return c


@dataclass(frozen=True)
class Airfoil:
    """A section: Selig-ordered coordinates, a name and where it came from."""

    coords: np.ndarray
    name: str = "airfoil"
    source: str = "coordinates"

    def __post_init__(self):
        object.__setattr__(self, "coords", _frozen(self.coords))

    # --- constructors ----------------------------------------------------------------
    @classmethod
    def naca(cls, code: str, num_panels: Optional[int] = None,
             trailing_edge: str = "open") -> "Airfoil":
        """NACA 4- or 5-digit section (``trailing_edge`` "open" or "closed")."""
        from .generators import generate_naca4, generate_naca5
        n = GEOMETRY.num_panels if num_panels is None else num_panels
        code = str(code).strip()
        gen = generate_naca5 if len(code) == 5 else generate_naca4
        return cls(gen(code, n, trailing_edge=trailing_edge), f"NACA {code}",
                   f"foil_lab.generators, {trailing_edge} trailing edge")

    @classmethod
    def from_file(cls, path: Union[str, Path], name: Optional[str] = None) -> "Airfoil":
        """Read a Selig or Lednicer coordinate file."""
        from .utils.io import load_dat
        return cls(load_dat(str(path)), name or Path(path).stem, f"file {path}")

    @classmethod
    def from_coords(cls, coords, name: str = "airfoil", source: str = "coordinates") -> "Airfoil":
        return cls(coords, name, source)

    @classmethod
    def parsec(cls, params: dict, num_panels: Optional[int] = None) -> "Airfoil":
        from .generators import generate_parsec
        n = GEOMETRY.num_panels if num_panels is None else num_panels
        return cls(generate_parsec(params, n), "PARSEC", "foil_lab.generators.generate_parsec")

    @classmethod
    def biconvex(cls, thickness: float = 0.06, num_panels: Optional[int] = None) -> "Airfoil":
        from .generators import generate_biconvex
        n = GEOMETRY.num_panels if num_panels is None else num_panels
        return cls(generate_biconvex(thickness, n), f"biconvex {thickness:g}",
                   "foil_lab.generators.generate_biconvex")

    @classmethod
    def wedge(cls, thickness: float = 0.06, num_panels: Optional[int] = None) -> "Airfoil":
        from .generators import generate_wedge
        n = GEOMETRY.num_panels if num_panels is None else num_panels
        return cls(generate_wedge(thickness, n), f"double wedge {thickness:g}",
                   "foil_lab.generators.generate_wedge")

    # --- properties ------------------------------------------------------------------
    @property
    def num_panels(self) -> int:
        return len(self.coords) - 1

    @property
    def te_gap(self) -> float:
        return geo.trailing_edge_gap(self.coords)

    @property
    def leading_edge(self) -> np.ndarray:
        return geo._ArcSpline(self.coords).chord_frame()[0]

    @property
    def trailing_edge(self) -> np.ndarray:
        """Midpoint of the trailing edge."""
        return 0.5 * (self.coords[0] + self.coords[-1])

    @property
    def chord(self) -> float:
        return geo._ArcSpline(self.coords).chord_frame()[2]

    @property
    def quarter_chord(self) -> np.ndarray:
        le = self.leading_edge
        return le + 0.25 * (self.trailing_edge - le)

    # --- operations ------------------------------------------------------------------
    def _new(self, coords, what: str) -> "Airfoil":
        return Airfoil(coords, self.name, f"{self.source}; {what}")

    def scaled(self, factor: float) -> "Airfoil":
        return self._new(geo.scale(self.coords, factor), f"scaled {factor:g}")

    def translated(self, dx: float, dy: float) -> "Airfoil":
        return self._new(geo.translate(self.coords, dx, dy), f"translated ({dx:g}, {dy:g})")

    def rotated(self, angle_deg: float, pivot: Tuple[float, float] = (0.25, 0.0)) -> "Airfoil":
        """Pitch by ``angle_deg`` (nose up positive) about ``pivot``."""
        return self._new(geo.rotate(self.coords, angle_deg, pivot=pivot),
                         f"rotated {angle_deg:g} deg about {tuple(pivot)}")

    def repanelled(self, num_panels: Optional[int] = None) -> "Airfoil":
        n = GEOMETRY.num_panels if num_panels is None else num_panels
        return self._new(geo.repanel_cosine(self.coords, n), f"repanelled to {n}")

    def with_te_gap(self, gap: float, blend_start: Optional[float] = None) -> "Airfoil":
        """Trailing-edge gap ``gap`` (fraction of this section's chord)."""
        return self._new(geo.set_trailing_edge_gap(self.coords, gap * self.chord, blend_start),
                         f"trailing-edge gap {gap:g} c")

    def with_thickness_scaled(self, factor: float) -> "Airfoil":
        return self._new(geo.scale_thickness_camber(self.coords, thickness=factor),
                         f"thickness x{factor:g}")

    def with_camber_scaled(self, factor: float) -> "Airfoil":
        return self._new(geo.scale_thickness_camber(self.coords, camber=factor),
                         f"camber x{factor:g}")


@dataclass(frozen=True)
class Element:
    """An airfoil placed in a configuration.

    The airfoil is scaled by ``scale`` about the origin, pitched by ``rotation_deg`` (nose up
    positive) about ``pivot`` (in the scaled airfoil's frame), then moved by ``offset``.
    """

    airfoil: Airfoil
    offset: Tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    rotation_deg: float = 0.0
    pivot: Tuple[float, float] = (0.0, 0.0)

    @property
    def placed(self) -> Airfoil:
        a = self.airfoil
        if self.scale != 1.0:
            a = a.scaled(self.scale)
        if self.rotation_deg != 0.0:
            a = a.rotated(self.rotation_deg, pivot=self.pivot)
        if tuple(self.offset) != (0.0, 0.0):
            a = a.translated(*self.offset)
        return a

    @property
    def coords(self) -> np.ndarray:
        return self.placed.coords


@dataclass(frozen=True)
class Configuration:
    """One or more elements, with the reference chord and moment reference point.

    By default (Q10) the reference chord is the first element's chord and the moment reference
    point its quarter-chord point, both of the placed element; either can be set explicitly.
    """

    elements: Tuple[Element, ...]
    ref_chord: Optional[float] = None
    ref_point: Optional[Tuple[float, float]] = None
    name: str = "configuration"

    def __post_init__(self):
        els = tuple(e if isinstance(e, Element) else Element(e) for e in self.elements)
        if not els:
            raise ValueError("a configuration needs at least one element")
        object.__setattr__(self, "elements", els)

    @classmethod
    def of(cls, geometry: Union[Airfoil, Element, "Configuration"]) -> "Configuration":
        if isinstance(geometry, Configuration):
            return geometry
        if isinstance(geometry, Airfoil):
            return cls((Element(geometry),), name=geometry.name)
        if isinstance(geometry, Element):
            return cls((geometry,), name=geometry.airfoil.name)
        raise TypeError(f"expected Airfoil, Element or Configuration, got {type(geometry).__name__}")

    def reference(self) -> Tuple[float, Tuple[float, float]]:
        """(reference chord, moment reference point)."""
        first = self.elements[0].placed
        chord = self.ref_chord if self.ref_chord is not None else first.chord
        if self.ref_point is not None:
            point = tuple(float(v) for v in self.ref_point)
        else:
            point = tuple(float(v) for v in first.quarter_chord)
        return float(chord), point
