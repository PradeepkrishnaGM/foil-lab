"""Public analysis interface: ``analyze`` and ``polar`` returning ``Result`` and ``Polar``.

Inviscid only in M2 (the inviscid core of :mod:`foil_lab.inviscid`); passing a Reynolds number
raises ``NotImplementedError`` until M3. ``solver.solve_flow`` keeps its own behaviour for the
GUI, CLI and batch tools (DECISIONS.md, M2 Q8).

Conventions (SPEC.md section 3): Cl, Cd and Cm per unit span, normalised by the reference
chord; Cm about the moment reference point, nose up positive; Cp and surface labels in Selig
order per element. Drag is ``None`` for an inviscid result, never 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from . import inviscid as core
from .airfoil import Airfoil, Configuration, Element
from .params import GEOMETRY

Geometry = Union[Airfoil, Element, Configuration]
METHOD = "inviscid linear-strength vortex panel method"


@dataclass(frozen=True)
class ElementResult:
    """Loads and surface pressure of one element (Selig order, panel midpoints)."""

    name: str
    cl: float
    cm: float
    cd: Optional[float]
    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray
    surface: np.ndarray            # "upper" or "lower" per panel
    num_panels: int


@dataclass(frozen=True)
class Result:
    """One operating point. ``cl``, ``cm``, ``cd`` are totals over all elements."""

    alpha: float
    cl: float
    cm: float
    cd: Optional[float]
    elements: Tuple[ElementResult, ...]
    ref_chord: float
    ref_point: Tuple[float, float]
    converged: bool
    method: str
    corrections: Tuple[str, ...]
    repanelled: bool
    version: str
    reynolds: Optional[float] = None

    @property
    def num_panels(self) -> Tuple[int, ...]:
        return tuple(e.num_panels for e in self.elements)

    def to_dict(self) -> dict:
        """Scalar summary (per-element loads included; Cp arrays left out)."""
        d = dict(alpha=self.alpha, cl=self.cl, cm=self.cm, cd=self.cd, ref_chord=self.ref_chord,
                 ref_point=self.ref_point, converged=self.converged, method=self.method,
                 corrections=list(self.corrections), repanelled=self.repanelled,
                 num_panels=list(self.num_panels), version=self.version, reynolds=self.reynolds)
        d["elements"] = [dict(name=e.name, cl=e.cl, cm=e.cm, cd=e.cd) for e in self.elements]
        return d


@dataclass(frozen=True)
class Polar:
    """Results over a list of angles for one geometry."""

    results: Tuple[Result, ...]

    @property
    def alpha(self) -> np.ndarray:
        return np.array([r.alpha for r in self.results])

    @property
    def cl(self) -> np.ndarray:
        return np.array([r.cl for r in self.results])

    @property
    def cm(self) -> np.ndarray:
        return np.array([r.cm for r in self.results])

    @property
    def converged(self) -> np.ndarray:
        return np.array([r.converged for r in self.results])

    def to_dicts(self) -> list:
        return [r.to_dict() for r in self.results]


def _prepare(geometry: Geometry, repanel: bool, num_panels: Optional[int]):
    config = Configuration.of(geometry)
    n = GEOMETRY.num_panels if num_panels is None else int(num_panels)
    bodies, names = [], []
    for el in config.elements:
        placed = el.placed
        coords = placed.repanelled(n).coords if repanel else placed.coords
        bodies.append(np.asarray(coords))
        names.append(placed.name)
    return config, bodies, names


def _check_re(re) -> None:
    if re is not None and re != 0:
        raise NotImplementedError("viscous analysis (a Reynolds number) is added in M3; "
                                  "fl.analyze and fl.polar are inviscid in this version")


def _result(ps, asm, bodies, names, config, alpha: float, repanel: bool) -> Result:
    from . import __version__
    sol = core.solve(ps, alpha, assembly=asm)
    cp_all = np.asarray(sol.cp)
    chord, point = config.reference()
    elements = []
    cl_t = cm_t = 0.0
    for k, (b, name) in enumerate(zip(bodies, names)):
        cl, _, cm = core.pressure_loads(ps, cp_all, alpha, k, chord, point)
        cp = core.to_selig_order(sol, cp_all, k)
        mid = 0.5 * (b[:-1] + b[1:])
        te_mid = 0.5 * (b[0] + b[-1])                  # leading edge: node farthest from it
        le = int(np.argmax(np.hypot(*(b - te_mid).T)))
        surface = np.where(np.arange(len(mid)) < le, "upper", "lower")
        elements.append(ElementResult(name, float(cl), float(cm), None, mid[:, 0].copy(),
                                      mid[:, 1].copy(), np.asarray(cp).copy(), surface, len(mid)))
        cl_t += float(cl)
        cm_t += float(cm)
    converged = bool(np.all(np.isfinite(cp_all)))
    return Result(float(alpha), cl_t, cm_t, None, tuple(elements), chord, point, converged,
                  METHOD, (), repanel, __version__)


def analyze(geometry: Geometry, alpha: float, re: Optional[float] = None, repanel: bool = True,
            num_panels: Optional[int] = None) -> Result:
    """Analyse one operating point (inviscid).

    ``geometry``: an ``Airfoil``, ``Element`` or ``Configuration``. ``repanel``: repanel every
    element by arc length to ``num_panels`` (default ``params.GEOMETRY.num_panels``); with
    ``repanel=False`` the given points are used unchanged. ``re``: not supported until M3.
    """
    _check_re(re)
    config, bodies, names = _prepare(geometry, repanel, num_panels)
    ps = core.PanelSet.from_bodies(bodies)
    return _result(ps, core.assemble(ps), bodies, names, config, float(alpha), repanel)


def polar(geometry: Geometry, alphas: Iterable[float], re: Optional[float] = None,
          repanel: bool = True, num_panels: Optional[int] = None) -> Polar:
    """Analyse a list of angles (inviscid); the system is assembled once."""
    _check_re(re)
    config, bodies, names = _prepare(geometry, repanel, num_panels)
    ps = core.PanelSet.from_bodies(bodies)
    asm = core.assemble(ps)
    return Polar(tuple(_result(ps, asm, bodies, names, config, float(a), repanel) for a in alphas))
