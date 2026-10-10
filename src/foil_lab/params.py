"""Tunable parameters of Foil-Lab, in one place.

Every number a user or developer might want to change lives here as a documented default.
Changing a default changes results, so it needs an entry in ``DECISIONS.md``.

Contents: benchmark settings (M1); geometry, repanelling and shape-optimisation settings (M2);
the coupled viscous solver (M3).
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


@dataclass(frozen=True)
class ViscousParams:
    """Coupled viscous-inviscid solver (``foil_lab.viscous``, ``foil_lab.bl``; M3).

    Attributes
    ----------
    closure_set:
        Closure set of the boundary layer, ``"fidkowski2022"`` or ``"drela1987"``
        (``foil_lab.bl.closure_sets``). Provisional default: the set the SPEC 4.1 targets are
        judged with; the user chooses the default after checkpoint C3 (DECISIONS.md M3).
    n_crit:
        Critical amplification factor of the e^n envelope method (both surfaces).
    residual_tol:
        Converged when the largest scaled residual is at most this (approved starting value).
    update_tol:
        ... and the last Newton update of theta and m = ue delta* is at most this, relative.
    max_iter:
        Newton iterations per operating point (approved starting value).
    hk_min:
        After each Newton update delta* is raised where needed so that H_k is at least this at
        every station (in the wake the H_k of the closures), as FID p. 11 does; each clip is
        counted in the result. FID p. 11 uses 1.02 in the wake and 1.00005 on the airfoil; one
        value for both is our choice. Approved (DECISIONS.md M3, C4).
    re_theta_min:
        A Newton step is scaled down so that Re_theta stays at or above this at turbulent stations,
        where the turbulent Cf contains log10 Re_theta (FID eq. 59 limits log10 Re_theta to 1.303,
        i.e. Re_theta 20; using it as a bound on the step is our choice). Approved.
    shape_eq_bias:
        Weight of the downstream station in the source term of the shape equation; 0.5 = central
        (DG87 p. 1351). Changed only if a checkpoint shows the stiffness DG87 describes.
    init_hk_max_lam, init_hk_max_turb:
        H_k above which the starting march switches to its inverse mode (FID p. 9: 3.8 and 2.5;
        the turbulent value also in the wake, where FID gives none of its own). Approved.
    jacobian:
        ``"complex_step"`` (NumPy: complex step of the local residuals with the chain rule through
        the edge velocity and the stagnation point) or ``"jax"`` (automatic differentiation of the
        full residual; needs JAX). Both are tested against each other and finite differences.
        Default complex step (our choice: 0.08 s per Jacobian against 7.9 s for eager JAX at 161
        stations). Approved.
    """

    closure_set: str = "fidkowski2022"
    n_crit: float = 9.0
    residual_tol: float = 1e-8
    update_tol: float = 1e-6
    max_iter: int = 50
    hk_min: float = 1.02
    re_theta_min: float = 20.0
    shape_eq_bias: float = 0.5
    init_hk_max_lam: float = 3.8
    init_hk_max_turb: float = 2.5
    jacobian: str = "complex_step"


XFOIL = XfoilParams()
SWEEP = SweepParams()
BENCHMARK = BenchmarkParams()
GEOMETRY = GeometryParams()
DESIGN = DesignParams()
VISCOUS = ViscousParams()
