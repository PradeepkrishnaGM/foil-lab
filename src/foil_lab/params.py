"""Tunable parameters of Foil-Lab, in one place.

Every number a user or developer might want to change lives here as a documented default.
Changing a default changes results, so it needs an entry in ``DECISIONS.md``.

At present this module holds only the benchmark settings (milestone M1); the solver
parameters move here in milestone M2.
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


XFOIL = XfoilParams()
SWEEP = SweepParams()
