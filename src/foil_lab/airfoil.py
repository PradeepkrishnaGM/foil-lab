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
from .flap import FlapInfo, make_flap
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
    #: set by :meth:`with_flap`; the undeflected section and hinge (see :mod:`foil_lab.flap`)
    flap: Optional[FlapInfo] = None

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
    def _reference(self) -> np.ndarray:
        """Coordinates that define the reference frame: the undeflected section if flapped."""
        return self.coords if self.flap is None else self.flap.base

    @property
    def leading_edge(self) -> np.ndarray:
        return geo._ArcSpline(self._reference).chord_frame()[0]

    @property
    def trailing_edge(self) -> np.ndarray:
        """Midpoint of the trailing edge (of the undeflected section if flapped)."""
        r = self._reference
        return 0.5 * (r[0] + r[-1])

    @property
    def chord(self) -> float:
        """Chord length; for a flapped section the chord of the undeflected one (SPEC.md 4.5)."""
        return geo._ArcSpline(self._reference).chord_frame()[2]

    @property
    def quarter_chord(self) -> np.ndarray:
        le = self.leading_edge
        return le + 0.25 * (self.trailing_edge - le)

    @property
    def hinge(self) -> Optional[np.ndarray]:
        """Hinge point of a flapped section (``None`` without a flap)."""
        return None if self.flap is None else self.flap.hinge

    # --- operations ------------------------------------------------------------------
    def _new(self, coords, what: str, flap: Optional[FlapInfo] = None) -> "Airfoil":
        return Airfoil(coords, self.name, f"{self.source}; {what}", flap)

    def _moved(self, fn, what: str) -> "Airfoil":
        """Apply a point transformation to the coordinates and to the flap definition."""
        flap = None if self.flap is None else self.flap.transformed(fn)
        return self._new(fn(self.coords), what, flap)

    def _reflapped(self, base_op, what: str) -> "Airfoil":
        """A shape operation on a flapped section: change the undeflected one, flap it again."""
        f = self.flap
        coords, info = make_flap(base_op(f.base), f.hinge_x, f.deflection, f.hinge_y,
                                 self.num_panels, f.params)
        return self._new(coords, what, info)

    def scaled(self, factor: float) -> "Airfoil":
        return self._moved(lambda p: geo.scale(p, factor), f"scaled {factor:g}")

    def translated(self, dx: float, dy: float) -> "Airfoil":
        return self._moved(lambda p: geo.translate(p, dx, dy), f"translated ({dx:g}, {dy:g})")

    def rotated(self, angle_deg: float, pivot: Tuple[float, float] = (0.25, 0.0)) -> "Airfoil":
        """Pitch by ``angle_deg`` (nose up positive) about ``pivot``."""
        return self._moved(lambda p: geo.rotate(p, angle_deg, pivot=pivot),
                           f"rotated {angle_deg:g} deg about {tuple(pivot)}")

    def repanelled(self, num_panels: Optional[int] = None) -> "Airfoil":
        n = GEOMETRY.num_panels if num_panels is None else num_panels
        if self.flap is not None:                      # rebuilt, so the hinge refinement stays
            f = self.flap
            coords, info = make_flap(f.base, f.hinge_x, f.deflection, f.hinge_y, n, f.params)
            return self._new(coords, f"repanelled to {n}", info)
        return self._new(geo.repanel_cosine(self.coords, n), f"repanelled to {n}")

    def with_te_gap(self, gap: float, blend_start: Optional[float] = None) -> "Airfoil":
        """Trailing-edge gap ``gap`` (fraction of this section's chord)."""
        what = f"trailing-edge gap {gap:g} c"
        if self.flap is not None:
            ch = self.chord
            return self._reflapped(lambda b: geo.set_trailing_edge_gap(b, gap * ch, blend_start), what)
        return self._new(geo.set_trailing_edge_gap(self.coords, gap * self.chord, blend_start), what)

    def with_thickness_scaled(self, factor: float) -> "Airfoil":
        what = f"thickness x{factor:g}"
        if self.flap is not None:
            return self._reflapped(lambda b: geo.scale_thickness_camber(b, thickness=factor), what)
        return self._new(geo.scale_thickness_camber(self.coords, thickness=factor), what)

    def with_camber_scaled(self, factor: float) -> "Airfoil":
        what = f"camber x{factor:g}"
        if self.flap is not None:
            return self._reflapped(lambda b: geo.scale_thickness_camber(b, camber=factor), what)
        return self._new(geo.scale_thickness_camber(self.coords, camber=factor), what)

    def with_flap(self, hinge_x: float, deflection: float, hinge_y: float = 0.5,
                  num_panels: Optional[int] = None) -> "Airfoil":
        """Plain flap: the part aft of ``hinge_x`` (fraction of the chord) deflected by
        ``deflection`` degrees about the hinge point (trailing edge down positive, -30 to +30).

        ``hinge_y`` is the fraction of the local thickness at ``hinge_x`` (0 lower surface, 1 upper
        surface, default 0.5). The hinge region is closed by an arc about the hinge point with
        tangent-continuous fairings and, on the closed side, a fillet (sizes in
        ``params.GEOMETRY``; :mod:`foil_lab.flap`). The panels are redistributed with a refinement
        at the hinge; ``num_panels`` defaults to the panel count of this section. A flapped section
        keeps the chord, leading edge and quarter-chord point of the undeflected one, so the angle
        of attack and ``Cm`` refer to them. A deflection of 0 leaves the coordinates unchanged.
        Called on a flapped section, it replaces the flap.
        """
        base = self._reference
        coords, info = make_flap(base, hinge_x, deflection, hinge_y,
                                 self.num_panels if num_panels is None else num_panels,
                                 self.flap.params if self.flap else GEOMETRY)
        src = self.source if self.flap is None else self.source.split("; flap")[0]
        return Airfoil(coords, self.name, f"{src}; flap hinge ({hinge_x:g}, {hinge_y:g}) "
                       f"deflection {deflection:g} deg", info)


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
