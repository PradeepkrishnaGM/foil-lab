"""Tunable parameters of Foil-Lab, in one place.

Every number a user or developer might want to change lives here as a documented default.
Changing a default changes results, so it needs an entry in ``DECISIONS.md``.

Contents: benchmark settings (M1); geometry, repanelling and shape-optimisation settings (M2).
The inviscid core has no tunable numbers apart from ``inviscid.CLOSED_GAP_FRACTION``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XfoilParams:
    """How the external XFOIL program is run to produce reference polars.

    Attributes
    ----------
    command:
        Program and wrapper used to start XFOIL. XFOIL 6.99 as packaged for Fedora crashes
        with a floating-point exception when graphics are disabled, so it runs with graphics
        on inside a virtual X server (``xvfb-run -a``).
    pane_nodes:
        Number of panel nodes XFOIL's ``PANE`` command distributes over the loaded
        coordinates. ``None`` means the coordinates are used as given (no ``PANE``).
    n_crit:
        Critical amplification factor of XFOIL's e^n transition model (both surfaces).
    xtr_top, xtr_bottom:
        Forced transition locations in x/c; 1.0 means free transition.
    mach:
        Freestream Mach number.
    max_iter:
        Viscous iteration limit per operating point (XFOIL ``ITER``).
    timeout_s:
        Wall-clock limit for one XFOIL process, in seconds.
    """

    command: tuple[str, ...] = ("xvfb-run", "-a", "xfoil")
    pane_nodes: int | None = 160
    n_crit: float = 9.0
    xtr_top: float = 1.0
    xtr_bottom: float = 1.0
    mach: float = 0.0
    max_iter: int = 200
    timeout_s: float = 300.0


@dataclass(frozen=True)
class SweepParams:
    """Angle-of-attack sweep used by the benchmark (degrees).

    XFOIL is run twice per case from a fresh start: from ``0`` up to ``alpha_max`` and from
    ``0`` down to ``alpha_min``, so that the two branches do not influence each other.
    """

    alpha_min: float = -4.0
    alpha_max: float = 20.0
    alpha_step: float = 1.0


@dataclass(frozen=True)
class BenchmarkParams:
    """Benchmark matrix (SPEC.md 4.1 without flap cases; DECISIONS.md M1).

    Attributes
    ----------
    sections:
        Section names, see ``foil_lab.benchmark.sections``.
    reynolds_validated:
        Reynolds numbers of the validated range.
    reynolds_low:
        Reynolds numbers reported but outside the validated range.
    pane_nodes:
        XFOIL ``PANE`` node counts. The first is the reference; the difference to the second
        is reported as paneling sensitivity.
    num_panels:
        Panels in the canonical coordinate file read by both codes.
    attached_fraction:
        A point is in the attached-flow filter if its alpha is at least ``alpha_min`` and below
        the first positive angle at which XFOIL's Cl reaches this fraction of its maximum.
    newton_sections, newton_reynolds, newton_alphas:
        Small subset on which the v1.3.1 Newton VII solver is run.
    vii_max_iter, vii_tol, newton_max_iter, newton_tol:
        Iteration settings of the v1.3.1 viscous solvers, as used by its command line.
    workers:
        Parallel worker processes for reference generation and baseline runs.
    """

    sections: tuple[str, ...] = ("NACA0012", "NACA2412", "NACA4412", "NACA23012",
                                 "CLARKY", "E387", "E387_GAP025")
    reynolds_validated: tuple[float, ...] = (5e5, 1e6, 3e6, 6e6)
    reynolds_low: tuple[float, ...] = (5e4, 1e5, 2e5)
    pane_nodes: tuple[int, int] = (160, 240)
    num_panels: int = 160
    attached_fraction: float = 0.9
    newton_sections: tuple[str, ...] = ("NACA0012", "NACA2412")
    newton_reynolds: tuple[float, ...] = (5e4, 2e5)
    newton_alphas: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0)
    vii_max_iter: int = 200
    vii_tol: float = 1e-3
    newton_max_iter: int = 60
    newton_tol: float = 1e-4
    workers: int = 6


@dataclass(frozen=True)
class GeometryParams:
    """Geometry model and operations (``foil_lab.geometry``, ``foil_lab.airfoil``).

    Attributes
    ----------
    num_panels:
        Panels per element after repanelling; ``fl.analyze`` repanels to this by default (Q9).
    te_gap_warn:
        The trailing-edge gap tool warns above this gap, as a fraction of chord (Q2).
    te_gap_blend_start:
        The gap tool changes the thickness only aft of this chordwise position, blended with
        ((x - x_b) / (1 - x_b))^2 so the shape ahead of x_b is unchanged.
    le_search_samples:
        Samples per input point used to bracket the leading edge (the point of the arc-length
        spline farthest from the trailing-edge midpoint) before it is refined by root finding.
    """

    num_panels: int = 160
    te_gap_warn: float = 0.01
    te_gap_blend_start: float = 0.8
    le_search_samples: int = 20


@dataclass(frozen=True)
class DesignParams:
    """Shape optimisation (``foil_lab.design``).

    Attributes
    ----------
    min_te_gap:
        Lower bound on the PARSEC trailing-edge gap ``dy_te`` (fraction of chord) when an open
        trailing edge is designed. The closed (zero gap) and open (trailing-edge panel)
        treatments differ at zero gap by a discretisation amount, so the optimiser must never
        reach zero gap from the open side (DECISIONS.md, M2). 0.001 c keeps the gap far from the
        switch and from finite-difference steps (about 1.5e-8). Proposed default, awaiting
        approval.
    """

    min_te_gap: float = 0.001


XFOIL = XfoilParams()
SWEEP = SweepParams()
BENCHMARK = BenchmarkParams()
GEOMETRY = GeometryParams()
DESIGN = DesignParams()
