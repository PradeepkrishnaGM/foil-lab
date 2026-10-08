"""Ground effect via the Method of Images (rigid wall at y=0).

Why not just append a mirror body
----------------------------------
The feature spec suggested mirroring the airfoil across y=0 and appending it to
the bodies list. That does NOT model a wall: this solver gives every body its
own Kutta condition and free circulation, so an appended mirror is solved as a
second free airfoil. The correct image *slaves* the mirror singularities to the
real ones so that y=0 is held as a streamline, and keeps the unknown count equal
to the real body alone.

Correct image signs (rigid wall)
--------------------------------
For a wall at y=0, reflect each singularity to (x, -y) with:
  * source strength: UNCHANGED (+) -- a source and its image both push fluid
    away from the wall, making y=0 a streamline for the source field.
  * vortex strength: NEGATED (-) -- the mirror vortex must spin oppositely so
    the wall-normal velocities cancel on y=0.

A lifting airfoil (bound vortex +Gamma) then sees its image (-Gamma) below the
wall; the image induces extra UPWASH at the real airfoil, which for a fixed
geometry shows up as increased lift as the ground is approached -- the textbook
ground-effect lift gain.

Implementation
--------------
We solve only the real body's (n+1) unknowns. The real-on-real influence comes
from the standard kernel. The image-on-real influence is computed directly: each
real collocation point sees the velocity induced by the reflected panels, with
the source part added and the vortex part subtracted. Both fold linearly into
the system matrix, so a single dense solve suffices.

Free surface (high Froude) is out of scope; only the rigid-wall image is built.
"""

from __future__ import annotations

import numpy as np

from .corrections import apply_3d_corrections, karman_tsien_correction
from .forces import integrate_loads
from .influence import LinearVortexModel
from .panels import build_panels
from .solver import SolutionData, geom_to_coords_list

_TWO_PI = 2.0 * np.pi


def _panel_influence_on_points(px, py, geom, sign_source=1.0, sign_vortex=1.0):
    """Velocity influence of each panel on arbitrary field points (px, py).

    Returns four (n_points, n_panels) arrays giving, per panel, the global-frame
    (u, v) induced per unit source strength and per unit (shared) vortex
    strength. Source and vortex parts can be independently sign-flipped, which is
    exactly what the image method needs.
    """
    x1, y1 = geom.x1, geom.y1
    theta, S = geom.theta, geom.lengths

    dx = px[:, None] - x1[None, :]
    dy = py[:, None] - y1[None, :]
    cos_t = np.cos(theta)[None, :]
    sin_t = np.sin(theta)[None, :]

    x_loc = dx * cos_t + dy * sin_t
    y_loc = -dx * sin_t + dy * cos_t
    lj = S[None, :]

    r1_sq = np.maximum(x_loc ** 2 + y_loc ** 2, 1e-12)
    r2_sq = np.maximum((x_loc - lj) ** 2 + y_loc ** 2, 1e-12)

    u_src = (0.5 * np.log(r1_sq / r2_sq)) / _TWO_PI
    v_src = np.arctan2(y_loc * lj, x_loc * (x_loc - lj) + y_loc ** 2) / _TWO_PI
    u_vtx, v_vtx = v_src, -u_src

    # rotate local -> global
    u_src_g = u_src * cos_t - v_src * sin_t
    v_src_g = u_src * sin_t + v_src * cos_t
    u_vtx_g = u_vtx * cos_t - v_vtx * sin_t
    v_vtx_g = u_vtx * sin_t + v_vtx * cos_t

    return (sign_source * u_src_g, sign_source * v_src_g,
            sign_vortex * u_vtx_g, sign_vortex * v_vtx_g)


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
    """Solve a single airfoil at ``height`` chords above a rigid wall at y=0."""
    if wall_type != "ground":
        raise NotImplementedError(
            "Only the rigid 'ground' image is implemented; the deformable "
            "free-surface (high-Froude) case is out of scope."
        )

    alpha_rad = np.deg2rad(alpha_deg)

    real = coords.copy().astype(float)
    real[:, 1] += height
    image = real.copy()
    image[:, 1] = -image[:, 1]   # reflect across y=0

    geom_r = build_panels(real)
    geom_i = build_panels(image)
    n = geom_r.num_panels
    theta = geom_r.theta
    nx, ny = geom_r.nx, geom_r.ny
    tx, ty = geom_r.tx, geom_r.ty

    model = LinearVortexModel()
    C_rr = model.compute(geom_r)   # real-on-real: S_n,S_t (n,n) V_n,V_t (n,n)

    # Image-on-real induced velocity, per real source column and the shared
    # vortex column. Source sign +, vortex sign - (rigid wall).
    us, vs, uv, vv = _panel_influence_on_points(
        geom_r.mid_x, geom_r.mid_y, geom_i, sign_source=1.0, sign_vortex=-1.0
    )
    # Project onto real-panel normal and tangent.
    img_n_src = us * nx[:, None] + vs * ny[:, None]   # (n, n) per source j
    img_t_src = us * tx[:, None] + vs * ty[:, None]
    img_n_vtx = uv * nx[:, None] + vv * ny[:, None]   # (n, n) per panel's vortex
    img_t_vtx = uv * tx[:, None] + vv * ty[:, None]

    # Real-body vortex collapsed to a single shared circulation column.
    Vn_real = np.sum(C_rr.V_n, axis=1)   # (n,)
    Vt_real = np.sum(C_rr.V_t, axis=1)

    # Assemble system for real unknowns [sources(n) | shared vortex(1)].
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)

    # Source columns: real self-influence + image (slaved to real source).
    A[:n, :n] = C_rr.S_n + img_n_src
    # Vortex column: real shared vortex + image vortex (slaved, sign already -).
    A[:n, n] = Vn_real + np.sum(img_n_vtx, axis=1)
    rhs[:n] = np.sin(theta - alpha_rad)

    # Kutta row (single body): first + last panel tangential closure.
    St = C_rr.S_t + img_t_src
    Vt_col = Vt_real + np.sum(img_t_vtx, axis=1)
    A[n, :n] = St[0, :] + St[n - 1, :]
    A[n, n] = Vt_col[0] + Vt_col[n - 1]
    rhs[n] = -(np.cos(theta[0] - alpha_rad) + np.cos(theta[n - 1] - alpha_rad))

    strengths = np.linalg.solve(A, rhs)
    sources_sol = strengths[:n]
    gamma_sol = strengths[n]
    vt = (np.cos(theta - alpha_rad) + St @ sources_sol + Vt_col * gamma_sol)
    vt *= v_inf

    cp_inc = 1.0 - (vt / v_inf) ** 2
    cp = karman_tsien_correction(cp_inc, mach)

    loads = integrate_loads(geom_r, cp, alpha_rad, body_index=None)
    cl, cd_ind, cm_qc = apply_3d_corrections(
        loads.cl, loads.cm_qc, alpha_deg, aspect_ratio, mach
    )

    return SolutionData(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds,
        aspect_ratio=aspect_ratio, cp_dist=cp, cl=cl, cd_ind=cd_ind, cd_visc=0.0,
        cm_le=loads.cm_le, cm_qc=cm_qc,
        sources=sources_sol, gamma=np.array([gamma_sol]),
        coords_list=geom_to_coords_list(coords),
    )
