"""Polars across flap deflection, with interpolation in the deflection angle (SPEC.md 4.5; M4).

``flap_polars`` solves one polar per deflection of a plain flap and returns a
:class:`FlapPolarSet`. ``FlapPolarSet.at(deflection)`` returns the polar for any deflection inside
the solved range, found by interpolating ``Cl``, ``Cd``, ``Cm`` and the hinge moment ``ch`` in the
deflection angle at every angle of attack.

Method. At fixed angle of attack each quantity is interpolated with a monotone piecewise cubic
Hermite interpolant (PCHIP; Fritsch & Carlson, SIAM J. Numer. Anal. 17 (1980) 238-246, as in
``scipy.interpolate.PchipInterpolator``): it reproduces the solved values exactly, has no overshoot
between them and uses the two nodes on each side of the interval at most. With two deflections it is
linear.

Limits (each one is enforced or reported, none is hidden):

* No extrapolation: a deflection outside the solved range raises ``ValueError``.
* The angles of attack of the interpolated polar are those of the set (or a subset of them); there is
  no interpolation in alpha.
* A point that did not converge is never interpolated (SPEC.md 4.1): if any node the interpolant of
  the interval uses did not converge at that angle, the interpolated point is not converged and its
  values are NaN. A deflection that is a node returns the solved values.
* Only numbers are interpolated, not flow physics. Near flap separation ``Cd`` and ``ch`` change
  quickly with deflection, and the spacing of the solved deflections sets the accuracy; a spacing
  above ``MAX_SPACING`` degrees gives a warning, and :meth:`FlapPolarSet.leave_one_out` measures the
  error of a given set directly (DECISIONS.md, M4, for the measured values).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .api import Polar, polar as _polar

#: Largest spacing (degrees) of the solved deflections without a warning.
MAX_SPACING = 10.0
QUANTITIES = ("cl", "cd", "cm", "ch")


def interpolate_in_deflection(deflections: np.ndarray, values: np.ndarray, target: float,
                              tol: float = 1e-9) -> np.ndarray:
    """Values at ``target`` of ``values`` (``(n_deflections, n_alpha)``, NaN where not converged).

    PCHIP along the deflection axis for every column; NaN in a column if any node that the
    interpolant of the interval uses is NaN there (the two nodes of the interval and their
    neighbours). A ``target`` equal to a node returns that node's column unchanged. ``deflections``
    must be increasing. Raises ``ValueError`` outside the range (no extrapolation).
    """
    d = np.asarray(deflections, dtype=float)
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] != len(d) or len(d) < 2 or np.any(np.diff(d) <= 0):
        raise ValueError("need increasing deflections (at least two) and a values array (n_def, n_alpha)")
    if target < d[0] - tol or target > d[-1] + tol:
        raise ValueError(f"deflection {target:g} deg is outside the solved range "
                         f"{d[0]:g} to {d[-1]:g} deg (no extrapolation)")
    hit = np.nonzero(np.abs(d - target) <= tol)[0]
    if len(hit):
        return v[hit[0]].copy()
    i = int(np.searchsorted(d, target) - 1)                    # interval [d[i], d[i + 1]]
    lo, hi = max(i - 1, 0), min(i + 2, len(d) - 1)
    win = slice(lo, hi + 1)
    out = np.full(v.shape[1], np.nan)
    ok = np.all(np.isfinite(v[win]), axis=0)
    if ok.any():
        out[ok] = PchipInterpolator(d[win], v[win][:, ok], axis=0)(target)
    return out


@dataclass(frozen=True)
class FlapSlice:
    """A polar at one flap deflection, solved or interpolated (arrays over the angles of attack)."""

    deflection: float
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    ch: np.ndarray
    converged: np.ndarray            # False where a value is NaN (not converged, or a node was not)
    interpolated: bool               # False when the deflection is one of the solved ones

    def to_dicts(self) -> list:
        return [dict(alpha=float(a), deflection=self.deflection, cl=float(cl), cd=float(cd), cm=float(cm),
                     ch=float(ch), converged=bool(c), interpolated=self.interpolated)
                for a, cl, cd, cm, ch, c in zip(self.alpha, self.cl, self.cd, self.cm, self.ch,
                                                self.converged)]


@dataclass(frozen=True)
class FlapPolarSet:
    """Polars of one section and flap hinge at several deflections.

    ``tables[q]`` (``q`` in ``cl``, ``cd``, ``cm``, ``ch``) has shape ``(n_deflections, n_alphas)``,
    NaN where the point did not converge. ``polars`` holds the solved :class:`Polar` objects when the
    set came from :func:`flap_polars`.
    """

    deflections: np.ndarray
    alphas: np.ndarray
    tables: dict
    reynolds: Optional[float] = None
    hinge_x: Optional[float] = None
    hinge_y: Optional[float] = None
    polars: Optional[tuple] = field(default=None, repr=False)

    def __post_init__(self):
        d = np.asarray(self.deflections, dtype=float)
        if len(d) < 2 or np.any(np.diff(d) <= 0):
            raise ValueError("a flap polar set needs at least two increasing deflections")
        for q in QUANTITIES:
            if np.asarray(self.tables[q]).shape != (len(d), len(self.alphas)):
                raise ValueError(f"table {q!r} must have shape (n_deflections, n_alphas)")
        gaps = np.diff(d)
        if gaps.max() > MAX_SPACING + 1e-9:
            warnings.warn(f"flap deflections are {gaps.max():g} deg apart (more than {MAX_SPACING:g}); "
                          f"interpolation between them can be inaccurate", stacklevel=3)

    @classmethod
    def from_tables(cls, deflections, alphas, cl, cd, cm, ch, **meta) -> "FlapPolarSet":
        """A set from arrays of shape ``(n_deflections, n_alphas)`` (NaN = not converged)."""
        return cls(np.asarray(deflections, float), np.asarray(alphas, float),
                   dict(cl=np.asarray(cl, float), cd=np.asarray(cd, float), cm=np.asarray(cm, float),
                        ch=np.asarray(ch, float)), **meta)

    def at(self, deflection: float, alphas: Optional[Iterable[float]] = None) -> FlapSlice:
        """The polar at ``deflection`` (degrees), interpolated unless it is a solved deflection.

        ``alphas`` (default: all) must be angles of the set. Raises ``ValueError`` outside the solved
        deflection range or for an angle that is not in the set.
        """
        d = np.asarray(self.deflections, float)
        if alphas is None:
            idx = np.arange(len(self.alphas))
        else:
            idx = []
            for a in alphas:
                k = np.nonzero(np.abs(self.alphas - a) < 1e-9)[0]
                if not len(k):
                    raise ValueError(f"alpha {a:g} deg is not an angle of this set; there is no "
                                     f"interpolation in alpha")
                idx.append(int(k[0]))
            idx = np.array(idx)
        vals = {q: interpolate_in_deflection(d, np.asarray(self.tables[q], float)[:, idx], deflection)
                for q in QUANTITIES}
        conv = np.all([np.isfinite(vals[q]) for q in ("cl", "cm")], axis=0)
        exact = bool(np.any(np.abs(d - deflection) <= 1e-9))
        return FlapSlice(float(deflection), np.asarray(self.alphas)[idx], vals["cl"], vals["cd"], vals["cm"],
                         vals["ch"], conv, not exact)

    def value(self, quantity: str, alpha: float, deflection: float) -> float:
        """One interpolated value (``quantity`` in ``cl``, ``cd``, ``cm``, ``ch``)."""
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        return float(getattr(self.at(deflection, [alpha]), quantity)[0])

    def leave_one_out(self, deflection: float, use: Optional[Sequence[float]] = None) -> dict:
        """Error of interpolating ``deflection`` (a solved one) from the others (or from ``use``).

        Returns ``{quantity: (mean abs error, max abs error, n points)}`` over the angles where the
        left-out value and everything the interpolant uses converged (``cd``: relative error).
        """
        d = np.asarray(self.deflections, float)
        k = int(np.argmin(np.abs(d - deflection)))
        if abs(d[k] - deflection) > 1e-9:
            raise ValueError(f"{deflection:g} deg is not a solved deflection")
        keep = [i for i in range(len(d)) if i != k and (use is None or any(abs(d[i] - u) < 1e-9 for u in use))]
        out = {}
        for q in QUANTITIES:
            est = interpolate_in_deflection(d[keep], np.asarray(self.tables[q], float)[keep], deflection)
            ref = np.asarray(self.tables[q])[k]
            m = np.isfinite(est) & np.isfinite(ref)
            err = np.abs(est - ref)[m] / (np.abs(ref[m]) if q == "cd" else 1.0)
            out[q] = (float(err.mean()) if m.any() else float("nan"),
                      float(err.max()) if m.any() else float("nan"), int(m.sum()))
        return out


def flap_polars(airfoil, hinge_x: float, deflections: Iterable[float], alphas: Iterable[float],
                re: Optional[float] = None, hinge_y: float = 0.5, **kwargs) -> FlapPolarSet:
    """Solve one polar per flap deflection and return the set (see the module docstring).

    ``airfoil``: an ``Airfoil`` (without flap, or its flap is replaced); ``deflections`` in degrees,
    trailing edge down positive, within +-30 and at least two; ``alphas`` and ``re`` as
    :func:`foil_lab.polar` (inviscid without ``re``); other keyword arguments go to ``polar``
    (``num_panels``, ``n_crit``, ``params`` ...). Use 160 panels or more for flaps.
    """
    d = sorted({float(x) for x in deflections})
    al = [float(a) for a in alphas]
    polars = []
    for x in d:
        polars.append(_polar(airfoil.with_flap(hinge_x, x, hinge_y), al, re=re, **kwargs))
    tables = {q: np.full((len(d), len(al)), np.nan) for q in QUANTITIES}
    for i, p in enumerate(polars):
        ok = p.converged
        for q, arr in (("cl", p.cl), ("cd", p.cd), ("cm", p.cm), ("ch", p.ch)):
            tables[q][i] = np.where(ok, arr, np.nan)
    return FlapPolarSet(np.array(d), np.array(al), tables, re, hinge_x, hinge_y, tuple(polars))
