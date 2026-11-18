"""Inviscid section loads for tests on raw coordinate arrays.

Per unit chord with the moment about (0.25, 0), straight from the core of
:mod:`foil_lab.inviscid` (no repanelling, no reference chord from the geometry). ``fl.analyze`` on
the same points gives the same surface Cp; its loads use the section's own chord and quarter-chord
point instead (SPEC.md 3), which differ from these by up to about 5e-4 in Cl for a cambered NACA
section whose leading edge is not exactly at x = 0.
"""

from __future__ import annotations

import numpy as np

from foil_lab import inviscid as core


def solve(coords, alpha: float) -> core.InviscidSolution:
    """Inviscid solution of one body (any node orientation)."""
    return core.solve(core.PanelSet.from_bodies([np.asarray(coords, float)]), float(alpha))


def loads(coords, alpha: float) -> core.BodyLoads:
    """(cl, cd, cm) of one body, per unit chord, moment about (0.25, 0)."""
    return core.body_loads(solve(coords, alpha), 0)


def cl(coords, alpha: float) -> float:
    return loads(coords, alpha).cl
