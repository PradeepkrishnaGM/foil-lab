"""Ground effect via the method of images (rigid wall at y = 0), on the inviscid core.

The section is pitched by ``alpha_deg`` about its quarter-chord point (0.25, 0), raised by
``height`` and solved together with its mirror image about y = 0 in a stream parallel to the
ground. By symmetry the mirrored pair has y = 0 as a streamline, so it is the flow over the
section above a rigid wall. Both bodies go through the one assembly of
:mod:`foil_lab.inviscid` (each with its own Kutta condition; the solution is symmetric), so the
image is not a separate kernel. Loads are those of the real body, in the ground (wind) axes,
with the moment about the displaced quarter-chord point.

Behaviour change from v1.3.1: there the stream was inclined by alpha to an unrotated section,
so it crossed the wall; here the section is pitched and the stream is parallel to the ground
(DECISIONS.md, M2 step 3).

Free surface (high Froude) is out of scope; only the rigid-wall image is built.

:func:`ground_frame` and :func:`image` are the one implementation of this construction; ``fl.analyze``
(``ground_height``, ``pivot``; any number of elements, inviscid or viscous) and
:func:`solve_ground_effect` both use them (M6). Validation: tests/test_corrections.py solves the
same pair of bodies as a free-air two-element configuration and finds the same loads, and checks
that the velocity normal to the ground vanishes on y = 0 (SPEC.md 4.2).
"""

from __future__ import annotations

import numpy as np

from . import inviscid as core
from .corrections import apply_3d_corrections, karman_tsien_correction
from .geometry import rotate
from .solver import SolutionData


def ground_frame(alpha_deg: float, pivot, height: float):
    """Point map into the ground frame (ground y = 0): pitch by ``alpha_deg`` (nose up positive)
    about ``pivot``, then move up so that the pivot is ``height`` above the ground."""
    px, py = float(pivot[0]), float(pivot[1])

    def fn(p):
        out = rotate(np.asarray(p, dtype=float), alpha_deg, pivot=(px, py))
        out[:, 1] += height - py
        return out
    return fn


def image(coords) -> np.ndarray:
    """Mirror image of a body about the ground y = 0 (the panel set accepts either orientation)."""
    return np.asarray(coords, dtype=float) * np.array([1.0, -1.0])


def solve_ground_effect(
    coords: np.ndarray,
    alpha_deg: float,
    height: float,
    v_inf: float = 1.0,
    mach: float = 0.0,
    aspect_ratio: float = 0.0,
    reynolds: float = 0.0,
    wall_type: str = "ground",
) -> SolutionData:
    """Solve a single section at ``height`` chords (quarter-chord point) above a rigid wall."""
    if reynolds > 0:
        raise NotImplementedError("ground effect is inviscid in this version (viscous ground effect: M6); "
                                  "use reynolds=0")
    if wall_type != "ground":
        raise NotImplementedError(
            "Only the rigid 'ground' image is implemented; the deformable "
            "free-surface (high-Froude) case is out of scope."
        )
    real = ground_frame(alpha_deg, (0.25, 0.0), height)(coords)

    ps = core.PanelSet.from_bodies([real, image(real)])
    sol = core.solve(ps, 0.0)
    cp = karman_tsien_correction(np.asarray(sol.cp), mach)

    ref = (0.25, float(height))
    cl0, _, cm_qc0 = core.pressure_loads(ps, cp, 0.0, 0, ref_point=ref)
    _, _, cm_le = core.pressure_loads(ps, cp, 0.0, 0, ref_point=(float(real[np.argmin(real[:, 0]), 0]),
                                                                  float(real[np.argmin(real[:, 0]), 1])))
    slope = 2.0 * np.pi
    if aspect_ratio > 0:
        from .params import CORRECTIONS
        h = CORRECTIONS.lift_slope_step
        up, lo = (solve_ground_effect(coords, alpha_deg + d, height, v_inf, mach).cl for d in (h, -h))
        slope = (up - lo) / np.deg2rad(2.0 * h)
    cl, cd_ind, cm_qc = apply_3d_corrections(float(cl0), float(cm_qc0), alpha_deg, aspect_ratio, mach,
                                             slope)

    n_real = ps.body_panels[0].stop
    return SolutionData(
        alpha_deg=0.0, mach=mach, reynolds=reynolds,
        aspect_ratio=aspect_ratio, cp_dist=cp[:n_real], cl=cl, cd_ind=cd_ind, cd_visc=0.0,
        cm_le=float(cm_le), cm_qc=cm_qc,
        sources=np.zeros(ps.num_panels), gamma=np.asarray(sol.gamma),
        coords_list=[real], panels=ps,
    )
