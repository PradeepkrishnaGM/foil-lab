"""Public analysis interface: ``analyze`` and ``polar`` returning ``Result`` and ``Polar``.

Without a Reynolds number: the inviscid core of :mod:`foil_lab.inviscid`. With ``re``: the coupled
viscous-inviscid solver of :mod:`foil_lab.viscous` (M3; several elements from M5, Mach 0), with transition by
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
from .params import CORRECTIONS, GEOMETRY, ViscousParams

Geometry = Union[Airfoil, Element, Configuration]
METHOD = "inviscid linear-strength vortex panel method"
METHOD_VISCOUS = "coupled viscous-inviscid (integral boundary layer and wake, e^n transition)"


class Cancelled(Exception):
    """Raised by a ``progress`` callback of :func:`polar` to stop the polar (the point being solved
    finishes first); :func:`polar` lets it propagate."""


def _pandas():
    try:
        import pandas
    except ImportError as exc:                      # pragma: no cover - depends on the install
        raise ImportError("pandas is needed for tables: pip install 'foil-lab[pandas]'") from exc
    return pandas


#: scalar columns of a result row (``Result.to_dict`` without the element list), in table order
ROW_COLUMNS = ("alpha", "cl", "cd", "cm", "ch", "converged", "reynolds", "mach", "sweep", "aspect_ratio",
               "ground_height", "cd_friction", "cd_pressure", "cd_induced", "xtr_upper", "xtr_lower",
               "mach_critical", "n_crit", "iterations", "iterations_total", "residual", "path", "method",
               "ref_chord", "version", "corrections", "warnings")


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
    #: viscous results only: this element's friction drag and transition x (nan: laminar to the
    #: trailing edge); ``cd`` above is the Squire-Young drag at the end of this element's wake
    cd_friction: Optional[float] = None
    xtr_upper: Optional[float] = None
    xtr_lower: Optional[float] = None


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
    #: conditions and corrections (M6; :mod:`foil_lab.corrections`)
    mach: float = 0.0
    sweep: float = 0.0                       # deg, simple sweep theory
    aspect_ratio: Optional[float] = None     # None: the section (infinite wing)
    ground_height: Optional[float] = None    # pivot height above the ground (reference chords)
    mach_critical: Optional[float] = None    # nan: no surface point faster than the freestream
    cd_induced: Optional[float] = None       # finite wing only; included in ``cd`` when ``cd`` is set

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
                 trips=None if self.trips is None else list(self.trips), mach=self.mach,
                 sweep=self.sweep, aspect_ratio=self.aspect_ratio, ground_height=self.ground_height,
                 mach_critical=self.mach_critical, cd_induced=self.cd_induced)
        d["ch"] = self.ch if sum(e.ch is not None for e in self.elements) <= 1 else None
        d["elements"] = [dict(name=e.name, cl=e.cl, cm=e.cm, cd=e.cd, ch=e.ch,
                              ch_pressure=e.ch_pressure, ch_friction=e.ch_friction,
                              cd_friction=e.cd_friction, xtr_upper=e.xtr_upper, xtr_lower=e.xtr_lower)
                         for e in self.elements]
        return d

    def row(self) -> dict:
        """One flat row (``ROW_COLUMNS``): scalars only, lists joined with "; "."""
        d = self.to_dict()
        out = {k: d.get(k) for k in ROW_COLUMNS}
        out["corrections"] = "; ".join(d["corrections"])
        out["warnings"] = "; ".join(d["warnings"])
        return out

    def to_dataframe(self):
        """One-row pandas DataFrame of :meth:`row`."""
        return _pandas().DataFrame([self.row()], columns=list(ROW_COLUMNS))

    def elements_dataframe(self):
        """pandas DataFrame with one row per element (name, cl, cm, cd, ch, ...)."""
        return _pandas().DataFrame(self.to_dict()["elements"])

    def surface_dataframe(self):
        """pandas DataFrame of the surface pressure: element, panel, x, y, surface, cp."""
        rows = [dict(element=k, name=e.name, panel=i, x=float(e.x[i]), y=float(e.y[i]),
                     surface=str(e.surface[i]), cp=float(e.cp[i]))
                for k, e in enumerate(self.elements) for i in range(e.num_panels)]
        return _pandas().DataFrame(rows)


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

    def rows(self) -> list:
        """Flat rows (``ROW_COLUMNS``), one per angle."""
        return [r.row() for r in self.results]

    def to_dataframe(self):
        """pandas DataFrame, one row per angle, columns ``ROW_COLUMNS``."""
        return _pandas().DataFrame(self.rows(), columns=list(ROW_COLUMNS))


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


@dataclass(frozen=True)
class _Problem:
    """The two-dimensional flow problem actually solved at one angle (M6)."""

    bodies: list                  # real bodies, then their images (ground effect)
    n_real: int
    names: list
    flaps: list                   # FlapInfo or None per real body
    chord: float
    point: Tuple[float, float]
    alpha: float                  # angle given to the solver (0 with ground: the section is pitched)


def _problem(config, geometry, repanel, num_panels, alpha_n: float, ground_height, pivot) -> _Problem:
    """Bodies of one operating point. Without ground: the placed elements, freestream at
    ``alpha_n``. With ground (SPEC.md 4.7): every element pitched by ``alpha_n`` about ``pivot``
    (default the moment reference point), raised so that the pivot is ``ground_height`` reference
    chords above the ground y = 0, plus the mirror image of each about y = 0; freestream parallel
    to the ground (:mod:`foil_lab.ground_effect`)."""
    chord, point = config.reference()
    if ground_height is None:
        _, bodies, names, flaps = _prepare(geometry, repanel, num_panels)
        return _Problem(bodies, len(bodies), names, flaps, chord, point, alpha_n)
    from .ground_effect import ground_frame, image
    pv = tuple(float(v) for v in (pivot if pivot is not None else point))
    to_ground = ground_frame(alpha_n, pv, float(ground_height) * chord)
    n = GEOMETRY.num_panels if num_panels is None else int(num_panels)
    real, names, flaps = [], [], []
    for el in config.elements:
        a = el.placed._moved(to_ground, f"ground frame: pitched {alpha_n:g} deg about {pv}, "
                                        f"pivot {ground_height:g} c above the ground")
        a = a.repanelled(n) if repanel else a
        real.append(np.asarray(a.coords))
        names.append(a.name)
        flaps.append(a.flap)
    ref = tuple(float(v) for v in to_ground(np.array([point], float))[0])
    return _Problem(real + [image(b) for b in real], len(real), names, flaps, chord, ref, 0.0)


def _hinge(ps, cp_all, k, flap, chord, shear=None):
    """(ch, ch_pressure, ch_friction) of body ``k`` if it has a flap, else three ``None``."""
    if flap is None:
        return None, None, None
    ch_p, ch_f = core.hinge_moment(ps, cp_all, k, flap.flap_panels, flap.hinge, chord, shear)
    return ch_p + ch_f, ch_p, (None if shear is None else ch_f)


def _viscous_setup(config, bodies, re, n_crit, trip_upper, trip_lower, params):
    from .viscous import trips_dict, viscous_params
    if not re > 0:
        raise ValueError(f"re must be a positive Reynolds number, got {re!r}")
    return viscous_params(n_crit, params), trips_dict(trip_upper, trip_lower)


def _viscous_result(vres, bodies, names, config, repanel: bool, prm, trip_upper, trip_lower,
                    corrections=(), flaps=(), n_real: Optional[int] = None, chord=None,
                    point=None) -> Result:
    """``Result`` of one viscous solution on ``bodies``, the geometry that was solved (closed
    trailing edges already opened, see ``corrections``). Only the first ``n_real`` bodies are
    reported; the others are ground images (their drag is not counted)."""
    from . import __version__
    from .corrections import critical_mach
    if chord is None:
        chord, point = config.reference()
    nr = len(bodies) if n_real is None else n_real
    ps = core.PanelSet.from_bodies(list(bodies))
    cp_all = np.asarray(vres.cp) if vres.cp is not None else np.full(ps.num_panels, np.nan)
    shear = None if vres.shear is None else np.asarray(vres.shear)
    per = vres.bodies or [{}] * len(bodies)
    elements = []
    cl_t = cm_t = 0.0
    for k, b in enumerate(bodies[:nr]):
        # the solver's loads are about the quarter chord of the unit chord at the origin; recompute
        # them from the viscous surface pressure about the configuration's reference (as inviscid)
        cl, _, cm = core.pressure_loads(ps, cp_all, vres.alpha, k, chord, point)
        sl = ps.body_panels[k]
        cp_k = cp_all[sl]
        cp = cp_k[::-1] if ps.reversed_input[k] else cp_k             # Selig (input) order
        mid = 0.5 * (b[:-1] + b[1:])
        te_mid = 0.5 * (b[0] + b[-1])
        le = int(np.argmax(np.hypot(*(b - te_mid).T)))
        surface = np.where(np.arange(len(mid)) < le, "upper", "lower")
        cd_k = float(per[k].get("cd", vres.cd if len(bodies) == 1 else np.nan)) / chord
        ch, ch_p, ch_f = _hinge(ps, cp_all, k, flaps[k] if flaps else None, chord, shear)
        elements.append(ElementResult(
            names[k], float(cl), float(cm), cd_k, mid[:, 0].copy(), mid[:, 1].copy(),
            np.asarray(cp).copy(), surface, len(mid), ch, ch_p, ch_f,
            cd_friction=float(per[k].get("cdf", vres.cdf if len(bodies) == 1 else np.nan)) / chord,
            xtr_upper=float(per[k].get("xtr_upper", vres.xtr_upper if len(bodies) == 1 else np.nan)),
            xtr_lower=float(per[k].get("xtr_lower", vres.xtr_lower if len(bodies) == 1 else np.nan))))
        cl_t += float(cl)
        cm_t += float(cm)
    if nr == len(bodies):
        cd, cdf, cdp = float(vres.cd) / chord, float(vres.cdf) / chord, float(vres.cdp) / chord
    else:
        cd = sum(per[k]["cd"] for k in range(nr)) / chord
        cdf = sum(per[k]["cdf"] for k in range(nr)) / chord
        cdp = cd - cdf
    real = np.concatenate([cp_all[ps.body_panels[k]] for k in range(nr)])
    return Result(float(vres.alpha), cl_t, cm_t, cd, tuple(elements), chord, point,
                  bool(vres.converged), METHOD_VISCOUS, tuple(corrections), repanel, __version__,
                  float(vres.reynolds) * chord,
                  cd_friction=cdf, cd_pressure=cdp,
                  xtr_upper=float(vres.xtr_upper), xtr_lower=float(vres.xtr_lower),
                  iterations=int(vres.iterations), iterations_total=int(vres.iterations_total),
                  residual=vres.residual, path=vres.path, warnings=tuple(vres.warnings),
                  n_crit=prm.n_crit, trips=(trip_upper or None, trip_lower or None),
                  mach_critical=critical_mach(float(np.min(real))))


def _solver_input(bodies):
    """One body as an array (bit-identical to M3/M4), several as a list (M5)."""
    return bodies[0] if len(bodies) == 1 else list(bodies)


def _result(ps, asm, bodies, names, config, alpha: float, repanel: bool, flaps=(), mach: float = 0.0,
            n_real: Optional[int] = None, chord=None, point=None) -> Result:
    """Inviscid result; Kármán-Tsien at ``mach`` on the surface pressure, applied once before the
    loads (SPEC.md 4.7). Only the first ``n_real`` bodies are reported (the rest are images)."""
    from . import __version__
    from .corrections import critical_mach, karman_tsien_correction
    sol = core.solve(ps, alpha, assembly=asm)
    cp_inc = np.asarray(sol.cp)
    cp_all = np.asarray(karman_tsien_correction(cp_inc, mach))
    if chord is None:
        chord, point = config.reference()
    nr = len(bodies) if n_real is None else n_real
    elements = []
    cl_t = cm_t = 0.0
    for k, (b, name) in enumerate(zip(bodies[:nr], names)):
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
    real = np.concatenate([cp_inc[ps.body_panels[k]] for k in range(nr)])
    return Result(float(alpha), cl_t, cm_t, None, tuple(elements), chord, point, converged,
                  METHOD, (), repanel, __version__, mach=float(mach),
                  mach_critical=critical_mach(float(np.min(real))))


def _corrected(r: Result, alpha: float, plane, mach: float, sweep: float, aspect_ratio, ground_height,
               slope) -> Result:
    """Steps 4 and 5 of :mod:`foil_lab.corrections` on a normal-plane result: back to the freestream
    (sweep), then the finite wing; conditions, critical Mach number, corrections and warnings
    recorded. ``slope``: callable giving the infinite wing's lift slope (per rad), used only with an
    aspect ratio."""
    import dataclasses
    from .corrections import finite_wing
    corr = list(r.corrections)
    warn = list(r.warnings)
    c = plane.cos_sweep
    if ground_height is not None:
        corr.append(f"ground effect: image method, pivot {ground_height:g} c above the ground, "
                    "freestream parallel to the ground")
    if sweep:
        corr.append(f"simple sweep theory: sweep {sweep:g} deg, normal-plane alpha {plane.alpha:.4f} deg, "
                    f"Mach {plane.mach:.4f}, Re {plane.reynolds:.6g}; Cl, Cm x cos^2, Cd x cos^3")
    if mach:
        corr.append(f"Kármán-Tsien at Mach {plane.mach:.4f} on the surface pressure")
    fw_f, cdi = 1.0, None
    if aspect_ratio is not None:
        a = float(slope())
        cl_inf = r.cl * c * c
        fw = finite_wing(cl_inf, a, float(aspect_ratio))
        fw_f, cdi = fw.factor, fw.cdi
        corr.append(f"finite wing: aspect ratio {aspect_ratio:g}, Helmbold factor {fw.factor:.5f} "
                    f"(lift slope {a:.4f} /rad), induced drag CL^2/(pi e AR)")
    els = []
    for e in r.elements:
        els.append(dataclasses.replace(
            e, cl=e.cl * c * c * fw_f, cm=e.cm * c * c, cd=None if e.cd is None else e.cd * c ** 3,
            cd_friction=None if e.cd_friction is None else e.cd_friction * c ** 3,
            ch=None if e.ch is None else e.ch * c * c,
            ch_pressure=None if e.ch_pressure is None else e.ch_pressure * c * c,
            ch_friction=None if e.ch_friction is None else e.ch_friction * c * c))
    cd = None if r.cd is None else r.cd * c ** 3 + (cdi or 0.0)
    mcr = None if r.mach_critical is None else r.mach_critical / c
    if mach and mcr is not None and np.isfinite(mcr) and mach > mcr:
        warn.append(f"Mach {mach:g} is above the critical Mach number {mcr:.4f}: supersonic flow on "
                    "the surface; the Kármán-Tsien rule and the panel method do not hold there")
    return dataclasses.replace(
        r, alpha=float(alpha), cl=r.cl * c * c * fw_f, cm=r.cm * c * c, cd=cd, elements=tuple(els),
        cd_friction=None if r.cd_friction is None else r.cd_friction * c ** 3,
        cd_pressure=None if r.cd_pressure is None else r.cd_pressure * c ** 3,
        corrections=tuple(corr), warnings=tuple(warn), mach=float(mach), sweep=float(sweep),
        aspect_ratio=None if aspect_ratio is None else float(aspect_ratio),
        ground_height=None if ground_height is None else float(ground_height),
        mach_critical=mcr, cd_induced=cdi)


def _check_conditions(re, mach, sweep, aspect_ratio, ground_height):
    if not 0.0 <= mach < 1.0:
        raise ValueError(f"mach must be in [0, 1), got {mach!r}")
    if re and mach:
        raise NotImplementedError("viscous results are Mach 0 in v2.0 (the boundary layer is "
                                  "incompressible); omit re for a compressible inviscid result")
    if aspect_ratio is not None and not aspect_ratio > 0.0:
        raise ValueError(f"aspect_ratio must be positive (None: the section), got {aspect_ratio!r}")
    if ground_height is not None and not ground_height > 0.0:
        raise ValueError(f"ground_height must be positive, got {ground_height!r}")


def _run(geometry, alphas, re, repanel, num_panels, n_crit, trip_upper, trip_lower, params, mach,
         sweep, aspect_ratio, ground_height, pivot, single: bool, progress=None) -> list:
    """The pipeline of :mod:`foil_lab.corrections` for a list of angles (DECISIONS.md M6)."""
    from .corrections import NormalPlane, to_normal_plane
    _check_conditions(re, mach, sweep, aspect_ratio, ground_height)
    config = Configuration.of(geometry)
    planes = [to_normal_plane(a, sweep, mach, re or 0.0) if sweep else NormalPlane(a, mach, re or 0.0, 1.0)
              for a in alphas]
    out = []
    if re:
        from . import viscous
        chord = config.reference()[0]
        if ground_height is None:
            pb = _problem(config, geometry, repanel, num_panels, 0.0, None, pivot)
            prm, trips = _viscous_setup(config, pb.bodies, re, n_crit, trip_upper, trip_lower, params)
            solved, corr = viscous.open_closed_te_bodies(pb.bodies, prm)
            rn = planes[0].reynolds / chord
            if single:
                vr = [viscous.solve_single(_solver_input(solved), planes[0].alpha, rn, prm, trips)]
            else:
                done = [0]

                def on_result(a, r, seconds, walked, error):
                    if progress is not None and not walked:
                        done[0] += 1
                        progress(done[0], len(planes))
                vr = viscous.sweep(_solver_input(solved), [p.alpha for p in planes], rn, prm, trips,
                                   on_result=on_result)
            raw = [_viscous_result(v, solved, pb.names, config, repanel, prm, trip_upper, trip_lower,
                                   corr, pb.flaps) for v in vr]
        else:                                  # the geometry changes with the angle: one solve each
            raw = []
            for p in planes:
                pb = _problem(config, geometry, repanel, num_panels, p.alpha, ground_height, pivot)
                prm, trips = _viscous_setup(config, pb.bodies, re, n_crit, trip_upper, trip_lower, params)
                solved, corr = viscous.open_closed_te_bodies(pb.bodies, prm)
                v = viscous.solve_single(_solver_input(solved), 0.0, p.reynolds / chord, prm, trips)
                raw.append(_viscous_result(v, solved, pb.names, config, repanel, prm, trip_upper,
                                           trip_lower, corr, pb.flaps, pb.n_real, pb.chord, pb.point))
                if progress is not None:
                    progress(len(raw), len(planes))
    else:
        raw = []
        fixed = None
        for p in planes:
            if ground_height is None and fixed is not None:
                pb, ps, asm = fixed
            else:
                pb = _problem(config, geometry, repanel, num_panels, p.alpha, ground_height, pivot)
                ps = core.PanelSet.from_bodies(pb.bodies)
                asm = core.assemble(ps)
                if ground_height is None:
                    fixed = (pb, ps, asm)
            raw.append(_result(ps, asm, pb.bodies, pb.names, config, p.alpha if ground_height is None
                               else 0.0, repanel, pb.flaps, p.mach, pb.n_real, pb.chord, pb.point))
            if progress is not None:
                progress(len(raw), len(planes))
    for a, p, r in zip(alphas, planes, raw):
        def slope(a=a):
            h = CORRECTIONS.lift_slope_step
            up, lo = (_run(geometry, [a + d], None, repanel, num_panels, None, None, None, None, mach,
                           sweep, None, ground_height, pivot, True)[0].cl for d in (h, -h))
            return (up - lo) / np.deg2rad(2.0 * h)
        out.append(_corrected(r, a, p, mach, sweep, aspect_ratio, ground_height, slope))
    return out


def analyze(geometry: Geometry, alpha: float, re: Optional[float] = None, repanel: bool = True,
            num_panels: Optional[int] = None, n_crit: Optional[float] = None,
            trip_upper: Optional[float] = None, trip_lower: Optional[float] = None,
            params: Optional[ViscousParams] = None, mach: float = 0.0, sweep: float = 0.0,
            aspect_ratio: Optional[float] = None, ground_height: Optional[float] = None,
            pivot: Optional[Tuple[float, float]] = None) -> Result:
    """Analyse one operating point.

    ``geometry``: an ``Airfoil``, ``Element`` or ``Configuration``. ``repanel``: repanel every
    element by arc length to ``num_panels`` (default ``params.GEOMETRY.num_panels``); with
    ``repanel=False`` the given points are used unchanged.

    ``re``: Reynolds number based on the reference chord (the solver works per unit length, so it
    receives ``re / ref_chord``); ``None`` or 0 gives the inviscid result.
    With ``re`` the coupled viscous solver runs (one or several elements, Mach 0): the direct start
    from the march, and if that fails continuation from 0 deg (:func:`foil_lab.viscous.solve_single`;
    the result's ``path`` says which). A closed trailing edge is opened to
    ``params.ViscousParams.closed_te_gap`` of the chord first and ``corrections`` says so (0 keeps
    the sharp edge, with a drag warning). e^n transition with ``n_crit`` (default
    ``params.VISCOUS.n_crit``), forced trips at x/c ``trip_upper`` / ``trip_lower`` on element 0
    (``None`` = free transition), other solver settings from ``params`` (default
    ``params.VISCOUS``). For several angles use :func:`polar`, which reaches each angle from a
    converged neighbour.

    Corrections (M6, applied once each in the order of :mod:`foil_lab.corrections` and listed in
    ``corrections``): ``ground_height`` (pivot height above the ground in reference chords; the
    configuration is pitched by alpha about ``pivot``, default the moment reference point, and the
    freestream is parallel to the ground), ``sweep`` (deg, simple sweep theory), ``mach``
    (Kármán-Tsien; inviscid only; the critical Mach number is always reported, with a warning above
    it), ``aspect_ratio`` (finite wing: Helmbold's lift factor and induced drag, in ``cd`` and
    ``cd_induced``; ``None`` = the section). With sweep, Cp is that of the normal plane.
    """
    return _run(geometry, [float(alpha)], re, repanel, num_panels, n_crit, trip_upper, trip_lower,
                params, float(mach), float(sweep), aspect_ratio, ground_height, pivot, True)[0]


def polar(geometry: Geometry, alphas: Iterable[float], re: Optional[float] = None,
          repanel: bool = True, num_panels: Optional[int] = None, n_crit: Optional[float] = None,
          trip_upper: Optional[float] = None, trip_lower: Optional[float] = None,
          params: Optional[ViscousParams] = None, mach: float = 0.0, sweep: float = 0.0,
          aspect_ratio: Optional[float] = None, ground_height: Optional[float] = None,
          pivot: Optional[Tuple[float, float]] = None, progress=None) -> Polar:
    """Analyse a list of angles; results in the order given (options as :func:`analyze`).

    Inviscid: the system is assembled once (per angle with ground effect, where the section is
    pitched). With ``re``: the angles are reached as the benchmark reaches them
    (:func:`foil_lab.viscous.sweep`: from the angle nearest 0 deg up and down with warm starts, then
    failed starts from the nearest converged angle); each result's ``path`` says how it was reached.
    With ground effect and ``re`` each angle is a separate solve (the geometry changes with alpha).

    ``progress``: optional callable ``progress(done, total)`` called after each angle is solved
    (viscous: in the order the sweep reaches them). If it raises (for example :class:`Cancelled`),
    the polar stops and the exception propagates.
    """
    return Polar(tuple(_run(geometry, [float(a) for a in alphas], re, repanel, num_panels, n_crit,
                            trip_upper, trip_lower, params, float(mach), float(sweep), aspect_ratio,
                            ground_height, pivot, False, progress)))
