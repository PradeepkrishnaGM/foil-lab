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
"""

from __future__ import annotations

import numpy as np

from . import inviscid as core
from .corrections import apply_3d_corrections, karman_tsien_correction
from .geometry import rotate
from .solver import SolutionData


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
    if wall_type != "ground":
        raise NotImplementedError(
            "Only the rigid 'ground' image is implemented; the deformable "
            "free-surface (high-Froude) case is out of scope."
        )
    real = rotate(np.asarray(coords, dtype=float), alpha_deg, pivot=(0.25, 0.0))
    real[:, 1] += height
    image = real * np.array([1.0, -1.0])

    ps = core.PanelSet.from_bodies([real, image])
    sol = core.solve(ps, 0.0)
    cp = karman_tsien_correction(np.asarray(sol.cp), mach)

    ref = (0.25, float(height))
    cl0, _, cm_qc0 = core.pressure_loads(ps, cp, 0.0, 0, ref_point=ref)
    _, _, cm_le = core.pressure_loads(ps, cp, 0.0, 0, ref_point=(float(real[np.argmin(real[:, 0]), 0]),
                                                                  float(real[np.argmin(real[:, 0]), 1])))
    cl, cd_ind, cm_qc = apply_3d_corrections(float(cl0), float(cm_qc0), alpha_deg, aspect_ratio, mach)

    n_real = ps.body_panels[0].stop
    return SolutionData(
        alpha_deg=0.0, mach=mach, reynolds=reynolds,
        aspect_ratio=aspect_ratio, cp_dist=cp[:n_real], cl=cl, cd_ind=cd_ind, cd_visc=0.0,
        cm_le=float(cm_le), cm_qc=cm_qc,
        sources=np.zeros(ps.num_panels), gamma=np.asarray(sol.gamma),
        coords_list=[real], panels=ps,
    )
