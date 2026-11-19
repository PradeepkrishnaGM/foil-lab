"""Plot helpers (SPEC.md 4.8). Every function returns ``(figure, axes)`` and never shows or blocks;
pass ``ax`` (or ``axes``) to draw into an existing figure.

    import foil_lab as fl
    from foil_lab import plot

    fig, ax = plot.geometry(fl.Airfoil.naca("2412").with_flap(0.75, 15))
    fig, ax = plot.cp(fl.analyze(fl.Airfoil.naca("2412"), 4.0))
    fig, axes = plot.polars([pol_a, pol_b], labels=["Re 1e6", "Re 3e6"], reference=xfoil_polar)
    fig.savefig("polar.png")
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

#: default panels of :func:`polars`: (x, y) quantity pairs
POLAR_PANELS = (("alpha", "cl"), ("cd", "cl"), ("alpha", "cm"))
LABELS = {"alpha": r"$\alpha$ (deg)", "cl": "$C_l$", "cd": "$C_d$", "cm": "$C_m$", "ch": "$C_h$",
          "deflection": r"flap deflection (deg)"}


def _new(ax=None, ncols: int = 1, size=(6.0, 4.0)):
    import matplotlib.pyplot as plt
    if ax is not None:
        axes = np.atleast_1d(ax)
        return axes[0].figure, ax
    fig, ax = plt.subplots(1, ncols, figsize=(size[0] * ncols, size[1]), squeeze=True)
    return fig, ax


def _coords_of(geometry):
    from .airfoil import Configuration
    cfg = Configuration.of(geometry)
    return [(el.placed.name, np.asarray(el.placed.coords)) for el in cfg.elements]


def geometry(geometry, ax=None, panels: bool = False, ground: Optional[float] = None):
    """Outline of an ``Airfoil``, ``Element`` or ``Configuration`` (equal axes). ``panels``: mark the
    nodes. ``ground``: draw the ground line at this y."""
    fig, ax = _new(ax, size=(8.0, 3.0))
    for name, c in _coords_of(geometry):
        ax.plot(c[:, 0], c[:, 1], ".-" if panels else "-", lw=1.2, ms=3, label=name)
    if ground is not None:
        ax.axhline(ground, color="0.3", lw=1.0, ls="--", label="ground")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, ls=":", alpha=0.6)
    return fig, ax


def cp(result, ax=None, labels: Optional[Sequence[str]] = None):
    """Surface pressure of one ``Result`` or a list of them (Cp axis inverted, suction up), each
    element's upper and lower surfaces as solid and dashed lines."""
    results = result if isinstance(result, (list, tuple)) else [result]
    fig, ax = _new(ax)
    for i, r in enumerate(results):
        tag = labels[i] if labels else f"alpha {r.alpha:g}"
        for k, e in enumerate(r.elements):
            for surf, ls in (("upper", "-"), ("lower", "--")):
                m = np.asarray(e.surface) == surf
                name = f"{tag}" + (f", element {k}" if len(r.elements) > 1 else "") + f", {surf}"
                ax.plot(np.asarray(e.x)[m], np.asarray(e.cp)[m], ls, lw=1.2, label=name,
                        color=f"C{(i * len(r.elements) + k) % 10}")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("$C_p$")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize="small")
    return fig, ax


def _series(p, q: str) -> np.ndarray:
    """Values of quantity ``q`` from a ``Polar``, ``FlapSlice``, XFOIL reference polar, pandas
    DataFrame or mapping of arrays (nan where not converged)."""
    from .benchmark.xfoil import XfoilPolar
    if isinstance(p, XfoilPolar):
        al = p.alphas
        if q == "alpha":
            return np.array(al, float)
        rows = [p.rows[a] for a in al]
        if q == "ch":
            return np.array([np.nan if r is None or r.chinge is None else -r.chinge for r in rows])
        return np.array([np.nan if r is None else getattr(r, q) for r in rows], float)
    if hasattr(p, "results"):                       # api.Polar
        v = np.asarray(getattr(p, q), float)
        if q != "alpha":
            v = np.where(p.converged, v, np.nan)
        return v
    if hasattr(p, "columns"):                       # pandas DataFrame
        return np.asarray(p[q], float)
    v = np.asarray(getattr(p, q) if hasattr(p, q) else p[q], float)
    if q != "alpha" and hasattr(p, "converged"):
        v = np.where(np.asarray(p.converged), v, np.nan)
    return v


def polars(polar_list, labels: Optional[Sequence[str]] = None,
           panels: Iterable = POLAR_PANELS, reference=None, reference_label: str = "XFOIL",
           axes=None):
    """Overlay several polars (sections, Reynolds numbers, flap angles, or reference polars) on one
    figure with one axes per (x, y) pair of ``panels`` (default Cl-alpha, Cl-Cd, Cm-alpha).
    ``polar_list``: ``Polar``, ``FlapSlice``, ``XfoilPolar``, DataFrame, or a list of them.
    ``reference``: one more polar (or a list) drawn as markers. Panels whose quantity a polar lacks
    (inviscid Cd) stay empty for it."""
    pl = polar_list if isinstance(polar_list, (list, tuple)) else [polar_list]
    panels = tuple(panels)
    own = axes is None
    fig, axes = _new(axes, ncols=len(panels), size=(4.6, 4.0))
    axes = np.atleast_1d(axes)
    refs = [] if reference is None else (reference if isinstance(reference, (list, tuple)) else [reference])
    for ax, (xq, yq) in zip(axes, panels):
        for i, p in enumerate(pl):
            x, y = _series(p, xq), _series(p, yq)
            if np.all(~np.isfinite(y)) or np.all(~np.isfinite(x)):
                continue
            ax.plot(x, y, ".-", lw=1.2, ms=4, color=f"C{i % 10}",
                    label=labels[i] if labels else f"polar {i + 1}")
        for j, r in enumerate(refs):
            x, y = _series(r, xq), _series(r, yq)
            ax.plot(x, y, "o", mfc="none", ms=5, color="k" if len(refs) == 1 else f"C{(len(pl) + j) % 10}",
                    label=reference_label if len(refs) == 1 else f"{reference_label} {j + 1}")
        ax.set_xlabel(LABELS.get(xq, xq))
        ax.set_ylabel(LABELS.get(yq, yq))
        ax.grid(True, ls=":", alpha=0.6)
    axes[0].legend(fontsize="small")
    if own:                                       # a figure passed in keeps the caller's layout
        fig.tight_layout()
    return fig, axes


def flap_set(fps, quantity: str = "cl", ax=None, deflections: Optional[Sequence[float]] = None):
    """One curve of ``quantity`` against alpha per deflection of a ``FlapPolarSet`` (solved
    deflections, or ``deflections`` interpolated, drawn dashed)."""
    fig, ax = _new(ax)
    ds = list(fps.deflections) if deflections is None else list(deflections)
    for i, d in enumerate(ds):
        sl = fps.at(d)
        ax.plot(sl.alpha, getattr(sl, quantity), "--" if sl.interpolated else ".-", color=f"C{i % 10}",
                label=f"{d:g} deg" + (" (interpolated)" if sl.interpolated else ""))
    ax.set_xlabel(LABELS["alpha"])
    ax.set_ylabel(LABELS.get(quantity, quantity))
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize="small", title="flap")
    return fig, ax
