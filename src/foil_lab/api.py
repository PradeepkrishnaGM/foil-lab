"""Public analysis interface: ``analyze`` and ``polar`` returning ``Result`` and ``Polar``.

Without a Reynolds number: the inviscid core of :mod:`foil_lab.inviscid`. With ``re``: the coupled
viscous-inviscid solver of :mod:`foil_lab.viscous` (M3; one element, Mach 0), with transition by
the e^n envelope method (``n_crit``) and forced trip locations per surface (``trip_upper``,
``trip_lower``, x/c), as SPEC.md 4.3 states. A viscous result carries the total drag and its
friction and pressure parts, the transition location on each surface, the Newton iterations, the
final residual, how the point was reached (``path``), warnings, and ``converged``, which is true
only if the residual criteria of ``params.ViscousParams`` were met.

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
from .params import GEOMETRY, ViscousParams

Geometry = Union[Airfoil, Element, Configuration]
METHOD = "inviscid linear-strength vortex panel method"
METHOD_VISCOUS = "coupled viscous-inviscid (integral boundary layer and wake, e^n transition)"


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
    #: flapped elements only (``None`` otherwise): hinge-moment coefficient about the hinge point,
    #: from the pressure on the flap panels and, for a viscous result, the wall friction on them;
    #: positive when it tends to deflect the flap trailing edge down (SPEC.md 3 and 4.5)
    ch: Optional[float] = None
    ch_pressure: Optional[float] = None
    ch_friction: Optional[float] = None


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
    #: viscous results only; ``None`` for an inviscid result
    cd_friction: Optional[float] = None
    cd_pressure: Optional[float] = None
    xtr_upper: Optional[float] = None       # transition x/c (nan: laminar to the trailing edge)
    xtr_lower: Optional[float] = None
    iterations: Optional[int] = None        # Newton iterations of the final solve
    iterations_total: Optional[int] = None  # including failed attempts and continuation
    residual: Optional[float] = None        # largest scaled residual at the last iteration
    path: Optional[str] = None              # how the point was reached ("march", "warm start", ...)
    warnings: Tuple[str, ...] = ()
    n_crit: Optional[float] = None
    trips: Optional[Tuple[Optional[float], Optional[float]]] = None   # (upper, lower) x/c

    @property
    def num_panels(self) -> Tuple[int, ...]:
        return tuple(e.num_panels for e in self.elements)

    def _flap_value(self, name: str):
        vals = [getattr(e, name) for e in self.elements if e.ch is not None]
        if len(vals) > 1:
            raise ValueError("several flapped elements: read the hinge moments from result.elements")
        return vals[0] if vals else None

    @property
    def ch(self) -> Optional[float]:
        """Hinge-moment coefficient of the (single) flap; ``None`` without a flap."""
        return self._flap_value("ch")

    @property
    def ch_pressure(self) -> Optional[float]:
        return self._flap_value("ch_pressure")

    @property
    def ch_friction(self) -> Optional[float]:
        """Friction part of ``ch``; ``None`` without a flap or for an inviscid result."""
        return self._flap_value("ch_friction")

    def to_dict(self) -> dict:
        """Scalar summary (per-element loads included; Cp arrays left out)."""
        d = dict(alpha=self.alpha, cl=self.cl, cm=self.cm, cd=self.cd, ref_chord=self.ref_chord,
                 ref_point=self.ref_point, converged=self.converged, method=self.method,
                 corrections=list(self.corrections), repanelled=self.repanelled,
                 num_panels=list(self.num_panels), version=self.version, reynolds=self.reynolds,
                 cd_friction=self.cd_friction, cd_pressure=self.cd_pressure, xtr_upper=self.xtr_upper,
                 xtr_lower=self.xtr_lower, iterations=self.iterations,
                 iterations_total=self.iterations_total, residual=self.residual, path=self.path,
                 warnings=list(self.warnings), n_crit=self.n_crit,
                 trips=None if self.trips is None else list(self.trips))
        d["elements"] = [dict(name=e.name, cl=e.cl, cm=e.cm, cd=e.cd, ch=e.ch,
                              ch_pressure=e.ch_pressure, ch_friction=e.ch_friction)
                         for e in self.elements]
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
    def cd(self) -> np.ndarray:
        """Total drag (nan for inviscid results)."""
        return np.array([np.nan if r.cd is None else r.cd for r in self.results])

    @property
    def ch(self) -> np.ndarray:
        """Hinge-moment coefficient of the flap (nan without a flap)."""
        return np.array([np.nan if r.ch is None else r.ch for r in self.results])

    @property
    def converged(self) -> np.ndarray:
        return np.array([r.converged for r in self.results])

    def to_dicts(self) -> list:
        return [r.to_dict() for r in self.results]


def _prepare(geometry: Geometry, repanel: bool, num_panels: Optional[int]):
    config = Configuration.of(geometry)
    n = GEOMETRY.num_panels if num_panels is None else int(num_panels)
    bodies, names, flaps = [], [], []
    for el in config.elements:
        placed = el.placed
        placed = placed.repanelled(n) if repanel else placed
        bodies.append(np.asarray(placed.coords))
        names.append(placed.name)
        flaps.append(placed.flap)
    return config, bodies, names, flaps


def _hinge(ps, cp_all, k, flap, chord, shear=None):
    """(ch, ch_pressure, ch_friction) of body ``k`` if it has a flap, else three ``None``."""
    if flap is None:
        return None, None, None
    ch_p, ch_f = core.hinge_moment(ps, cp_all, k, flap.flap_panels, flap.hinge, chord, shear)
    return ch_p + ch_f, ch_p, (None if shear is None else ch_f)


def _viscous_setup(config, bodies, re, n_crit, trip_upper, trip_lower, params):
    from .solver import trips_dict, viscous_params
    if len(bodies) != 1:
        raise NotImplementedError("viscous analysis (re) of several elements is added in M5; "
                                  "omit re for an inviscid multi-element result")
    if not re > 0:
        raise ValueError(f"re must be a positive Reynolds number, got {re!r}")
    return viscous_params(n_crit, params), trips_dict(trip_upper, trip_lower)


def _viscous_result(vres, body, names, config, repanel: bool, prm, trip_upper, trip_lower,
                    corrections=(), flap=None) -> Result:
    """``Result`` of one viscous solution on ``body``, the geometry that was solved (a closed
    trailing edge already opened, see ``corrections``)."""
    from . import __version__
    b = body
    chord, point = config.reference()
    ps = core.PanelSet.from_bodies([b])
    cp_all = np.asarray(vres.cp) if vres.cp is not None else np.full(ps.num_panels, np.nan)
    # the solver's loads are about the quarter chord of the unit chord at the origin; recompute them
    # from the viscous surface pressure about the configuration's reference (as the inviscid path)
    cl, _, cm = core.pressure_loads(ps, cp_all, vres.alpha, 0, chord, point)
    cp = cp_all[::-1] if ps.reversed_input[0] else cp_all             # Selig (input) order
    mid = 0.5 * (b[:-1] + b[1:])
    te_mid = 0.5 * (b[0] + b[-1])
    le = int(np.argmax(np.hypot(*(b - te_mid).T)))
    surface = np.where(np.arange(len(mid)) < le, "upper", "lower")
    cd = float(vres.cd) / chord
    ch, ch_p, ch_f = _hinge(ps, cp_all, 0, flap, chord,
                            None if vres.shear is None else np.asarray(vres.shear))
    el = ElementResult(names[0], float(cl), float(cm), cd, mid[:, 0].copy(), mid[:, 1].copy(),
                       np.asarray(cp).copy(), surface, len(mid), ch, ch_p, ch_f)
    return Result(float(vres.alpha), float(cl), float(cm), cd, (el,), chord, point,
                  bool(vres.converged), METHOD_VISCOUS, tuple(corrections), repanel, __version__,
                  float(vres.reynolds) * chord,
                  cd_friction=float(vres.cdf) / chord, cd_pressure=float(vres.cdp) / chord,
                  xtr_upper=float(vres.xtr_upper), xtr_lower=float(vres.xtr_lower),
                  iterations=int(vres.iterations), iterations_total=int(vres.iterations_total),
                  residual=vres.residual, path=vres.path, warnings=tuple(vres.warnings),
                  n_crit=prm.n_crit, trips=(trip_upper or None, trip_lower or None))


def _result(ps, asm, bodies, names, config, alpha: float, repanel: bool, flaps=()) -> Result:
    from . import __version__
    sol = core.solve(ps, alpha, assembly=asm)
    cp_all = np.asarray(sol.cp)
    chord, point = config.reference()
    elements = []
    cl_t = cm_t = 0.0
    for k, (b, name) in enumerate(zip(bodies, names)):
        cl, _, cm = core.pressure_loads(ps, cp_all, alpha, k, chord, point)
        ch, ch_p, ch_f = _hinge(ps, cp_all, k, flaps[k] if flaps else None, chord)
        cp = core.to_selig_order(sol, cp_all, k)
        mid = 0.5 * (b[:-1] + b[1:])
        te_mid = 0.5 * (b[0] + b[-1])                  # leading edge: node farthest from it
        le = int(np.argmax(np.hypot(*(b - te_mid).T)))
        surface = np.where(np.arange(len(mid)) < le, "upper", "lower")
        elements.append(ElementResult(name, float(cl), float(cm), None, mid[:, 0].copy(),
                                      mid[:, 1].copy(), np.asarray(cp).copy(), surface, len(mid),
                                      ch, ch_p, ch_f))
        cl_t += float(cl)
        cm_t += float(cm)
    converged = bool(np.all(np.isfinite(cp_all)))
    return Result(float(alpha), cl_t, cm_t, None, tuple(elements), chord, point, converged,
                  METHOD, (), repanel, __version__)


def analyze(geometry: Geometry, alpha: float, re: Optional[float] = None, repanel: bool = True,
            num_panels: Optional[int] = None, n_crit: Optional[float] = None,
            trip_upper: Optional[float] = None, trip_lower: Optional[float] = None,
            params: Optional[ViscousParams] = None) -> Result:
    """Analyse one operating point.

    ``geometry``: an ``Airfoil``, ``Element`` or ``Configuration``. ``repanel``: repanel every
    element by arc length to ``num_panels`` (default ``params.GEOMETRY.num_panels``); with
    ``repanel=False`` the given points are used unchanged.

    ``re``: Reynolds number based on the reference chord (the solver works per unit length, so it
    receives ``re / ref_chord``); ``None`` or 0 gives the inviscid result.
    With ``re`` the coupled viscous solver runs (one element, Mach 0): the direct start from the
    march, and if that fails continuation from 0 deg (:func:`foil_lab.viscous.solve_single`; the
    result's ``path`` says which). A closed trailing edge is opened to
    ``params.ViscousParams.closed_te_gap`` of the chord first and ``corrections`` says so (0 keeps
    the sharp edge, with a drag warning). e^n transition with ``n_crit`` (default
    ``params.VISCOUS.n_crit``), forced trips at x/c ``trip_upper`` / ``trip_lower`` (``None`` = free
    transition), other solver settings from ``params`` (default ``params.VISCOUS``). For several
    angles use :func:`polar`, which reaches each angle from a converged neighbour.
    """
    config, bodies, names, flaps = _prepare(geometry, repanel, num_panels)
    if re:
        from . import viscous
        prm, trips = _viscous_setup(config, bodies, re, n_crit, trip_upper, trip_lower, params)
        body, corr = viscous.open_closed_te(bodies[0], prm)
        vres = viscous.solve_single(body, float(alpha), float(re) / config.reference()[0], prm, trips)
        return _viscous_result(vres, body, names, config, repanel, prm, trip_upper, trip_lower, corr,
                               flaps[0])
    ps = core.PanelSet.from_bodies(bodies)
    return _result(ps, core.assemble(ps), bodies, names, config, float(alpha), repanel, flaps)


def polar(geometry: Geometry, alphas: Iterable[float], re: Optional[float] = None,
          repanel: bool = True, num_panels: Optional[int] = None, n_crit: Optional[float] = None,
          trip_upper: Optional[float] = None, trip_lower: Optional[float] = None,
          params: Optional[ViscousParams] = None) -> Polar:
    """Analyse a list of angles; results in the order given.

    Inviscid: the system is assembled once. With ``re`` (see :func:`analyze`): the angles are
    reached as the benchmark reaches them (:func:`foil_lab.viscous.sweep`: from the angle nearest
    0 deg up and down with warm starts, then failed starts from the nearest converged angle); each
    result's ``path`` says how it was reached.
    """
    config, bodies, names, flaps = _prepare(geometry, repanel, num_panels)
    alphas = [float(a) for a in alphas]
    if re:
        from . import viscous
        prm, trips = _viscous_setup(config, bodies, re, n_crit, trip_upper, trip_lower, params)
        body, corr = viscous.open_closed_te(bodies[0], prm)
        res = viscous.sweep(body, alphas, float(re) / config.reference()[0], prm, trips)
        return Polar(tuple(_viscous_result(r, body, names, config, repanel, prm, trip_upper, trip_lower,
                                           corr, flaps[0]) for r in res))
    ps = core.PanelSet.from_bodies(bodies)
    asm = core.assemble(ps)
    return Polar(tuple(_result(ps, asm, bodies, names, config, a, repanel, flaps) for a in alphas))
