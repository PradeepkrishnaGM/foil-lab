"""Exact-solution and consistency tests (SPEC.md 4.2; tolerances fixed in DECISIONS.md, M1 Q11).

Tests that fail on v1.3.1 are marked ``xfail(strict=True, raises=AssertionError)`` with the
milestone that fixes them. The laminar (Blasius) and turbulent flat-plate checks of 4.2 are in
``tests/test_vii.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from foil_lab import inviscid as core
from foil_lab.flowfield import compute_velocity_field
from foil_lab.generators import generate_naca4, generate_naca5
from foil_lab.geometry import repanel_cosine, rotate, scale, translate
from foil_lab.ground_effect import solve_ground_effect
from foil_lab.solver import loads_from_cp, solve_flow


def _xfail(milestone: str, why: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason=f"fixed in {milestone}: {why}")


# --------------------------------------------------------------------------------------- #
# Kármán-Trefftz airfoil: analytic surface pressure
# --------------------------------------------------------------------------------------- #
# Kármán-Trefftz transformation and the flow about a circle with circulation fixed by the Kutta
# condition, as in L. M. Milne-Thomson, "Theoretical Aerodynamics" (1958), ch. 7, and
# J. Katz and A. Plotkin, "Low-Speed Aerodynamics" (2001), sec. 6.7.
#
# Section parameters: circle centre zeta_0 = -0.08 + 0.08 i in the zeta plane, passing through
# the trailing-edge point zeta = 1 (b = 1); trailing-edge angle tau = 10 deg, so the exponent is
# n = 2 - tau/pi. The section is about 12 % thick and 3 % cambered; z ~ zeta far away, so the
# freestream angle is the same in both planes.
KT_CENTRE = -0.08 + 0.08j
KT_TAU_DEG = 10.0


def _kt(n_panels: int, alpha_deg: float):
    """Nodes (Selig order, closed), panel-midpoint parameter values and analytic Cp there."""
    n = 2.0 - np.deg2rad(KT_TAU_DEG) / np.pi
    radius = abs(1.0 - KT_CENTRE)
    beta = -np.angle(1.0 - KT_CENTRE)              # trailing edge seen from the centre at -beta
    alpha = np.deg2rad(alpha_deg)

    def z_of(zeta):
        a, b = (zeta + 1.0) ** n, (zeta - 1.0) ** n
        return n * (a + b) / (a - b)

    def cp_of(zeta):
        w = (np.exp(-1j * alpha) - radius**2 * np.exp(1j * alpha) / (zeta - KT_CENTRE) ** 2
             + 1j * 4 * np.pi * radius * np.sin(alpha + beta) / (2 * np.pi * (zeta - KT_CENTRE)))
        dzdz = (4 * n**2 * ((zeta - 1.0) * (zeta + 1.0)) ** (n - 1)
                / ((zeta + 1.0) ** n - (zeta - 1.0) ** n) ** 2)
        return 1.0 - np.abs(w / dzdz) ** 2

    th = np.linspace(0.0, 2 * np.pi, n_panels + 1)   # counter-clockwise from the trailing edge
    zeta = KT_CENTRE + radius * np.exp(1j * (th - beta))
    z = z_of(zeta)
    z[0] = z[-1] = n                                   # exact trailing-edge point
    th_mid = 0.5 * (th[:-1] + th[1:])
    zeta_mid = KT_CENTRE + radius * np.exp(1j * (th_mid - beta))
    return np.column_stack((z.real, z.imag)), z_of(zeta_mid), cp_of(zeta_mid)


def _kt_error(n_panels: int, alpha_deg: float) -> float:
    coords, z_mid, cp_exact = _kt(n_panels, alpha_deg)
    sol = solve_flow(coords, alpha_deg)
    g = core.PanelSet.from_bodies(coords)
    # The solver orders panels clockwise; our nodes are counter-clockwise: reverse to match.
    cp_num = sol.cp_dist[::-1]
    assert np.allclose(g.mid_x[::-1], 0.5 * (coords[:-1, 0] + coords[1:, 0]))
    x_le, x_te = coords[:, 0].min(), coords[0, 0]
    xc = (z_mid.real - x_le) / (x_te - x_le)
    band = (xc >= 0.02) & (xc <= 0.98)
    return float(np.max(np.abs(cp_num[band] - cp_exact[band])))


@pytest.mark.parametrize("alpha", [0.0, 4.0])
def test_karman_trefftz_surface_pressure(alpha):
    """max |dCp| <= 0.02 for x/c in [0.02, 0.98] at 200 panels; error falls from 160 to 320."""
    assert _kt_error(200, alpha) <= 0.02
    assert _kt_error(320, alpha) < _kt_error(160, alpha)


# --------------------------------------------------------------------------------------- #
# Multi-element, data-free checks (the Williams exact solution is added in M2)
# --------------------------------------------------------------------------------------- #
def _two_body_cl(bodies, alpha_deg):
    sol = core.solve(core.PanelSet.from_bodies(bodies), alpha_deg)
    return [core.body_loads(sol, k).cl for k in range(len(bodies))]


def test_two_different_bodies_far_apart_recover_isolated_lift():
    """NACA 0012 and NACA 4412 50 chords apart: each Cl within 0.5 % of its isolated value.

    The bodies are stacked (offset across the stream), not in tandem. In 2D the velocity induced
    by a lifting body decays only as 1/distance; in tandem at 50 chords the physical interference
    on the 0012 is about +2.3 % (point-vortex estimate +2.1 %), so a tandem test could never pass.
    Stacked, the induced velocity is nearly parallel to the stream and the physical interference
    is about -0.18 % / +0.11 % (measured with a per-body-closed assembly, DECISIONS.md M1).
    """
    a, b = generate_naca4("0012", 160), generate_naca4("4412", 160)
    cl_a, cl_b = _two_body_cl([a, translate(b, 0.0, 50.0)], 4.0)
    assert abs(cl_a / solve_flow(a, 4.0).cl - 1.0) <= 0.005
    assert abs(cl_b / solve_flow(b, 4.0).cl - 1.0) <= 0.005


def test_mirrored_pair_lift_equal_and_opposite():
    """A section and its mirror image about y = 0 in a horizontal stream: Cl1 = -Cl2 (1e-9)."""
    real = translate(rotate(generate_naca4("2412", 160), 4.0, pivot=(0.25, 0.0)), 0.0, 0.5)
    mirror = real * np.array([1.0, -1.0])
    cl_real, cl_mirror = _two_body_cl([real, mirror], 0.0)
    assert abs(cl_real + cl_mirror) <= 1e-9


# --------------------------------------------------------------------------------------- #
# Ground effect against the mirrored two-body problem in free air
# --------------------------------------------------------------------------------------- #
def test_ground_effect_matches_mirror_problem():
    """Section rotated 4 deg about c/4 at h/c = 0.5, stream parallel to the ground: the image
    solution and the free-air mirrored pair agree to |dCl| <= 1e-6."""
    sec = rotate(generate_naca4("2412", 160), 4.0, pivot=(0.25, 0.0))
    ge = solve_ground_effect(sec, 0.0, height=0.5)
    real = translate(sec, 0.0, 0.5)
    cl_pair = _two_body_cl([real, real * np.array([1.0, -1.0])], 0.0)[0]
    assert abs(ge.cl - cl_pair) <= 1e-6


def test_ground_plane_is_a_streamline():
    """Normal velocity on the ground plane y = 0 is zero to 1e-8."""
    sec = rotate(generate_naca4("2412", 160), 4.0, pivot=(0.25, 0.0))
    ge = solve_ground_effect(sec, 0.0, height=0.5)
    _, _, _, v = compute_velocity_field(ge, x_bounds=(-1.0, 2.0), y_bounds=(0.0, 0.0), grid_res=61)
    assert np.max(np.abs(v)) <= 1e-8


# --------------------------------------------------------------------------------------- #
# Lift: surface pressure = circulation = far-field momentum
# --------------------------------------------------------------------------------------- #
def _pressure_and_circulation_cl(af, alpha_deg):
    sol = core.solve(core.PanelSet.from_bodies(af), alpha_deg)
    cl_p = solve_flow(af, alpha_deg).cl
    cl_g = abs(2.0 * float(sol.circulation[0]))               # Kutta-Joukowski, c = V = 1
    return cl_p, cl_g


def test_pressure_lift_equals_circulation_lift():
    """Within 0.5 %."""
    cl_p, cl_g = _pressure_and_circulation_cl(generate_naca4("2412", 200), 4.0)
    assert abs(cl_g / cl_p - 1.0) <= 0.005


def _farfield_cl(sol, alpha_deg, half=10.0, res=801):
    """Lift from momentum flux and pressure on a square contour of side 2*half (rho = V = 1)."""
    x, y, u, v = compute_velocity_field(sol, x_bounds=(0.5 - half, 0.5 + half),
                                        y_bounds=(-half, half), grid_res=res)
    p = 0.5 * (1.0 - (u**2 + v**2))
    fx = fy = 0.0
    # (edge selector, outward normal) for bottom, top, left, right
    for sl, (nx, ny), along in (((0, slice(None)), (0.0, -1.0), x[0]),
                                ((-1, slice(None)), (0.0, 1.0), x[-1]),
                                ((slice(None), 0), (-1.0, 0.0), y[:, 0]),
                                ((slice(None), -1), (1.0, 0.0), y[:, -1])):
        ue, ve, pe = u[sl], v[sl], p[sl]
        vn = ue * nx + ve * ny
        fx += -np.trapezoid(pe * nx + ue * vn, along)
        fy += -np.trapezoid(pe * ny + ve * vn, along)
    a = np.deg2rad(alpha_deg)
    return 2.0 * (fy * np.cos(a) - fx * np.sin(a))


def test_farfield_lift_equals_pressure_lift():
    """Within 0.5 %."""
    af = generate_naca4("2412", 200)
    sol = solve_flow(af, 4.0)
    assert abs(_farfield_cl(sol, 4.0) / sol.cl - 1.0) <= 0.005


# --------------------------------------------------------------------------------------- #
# Geometry operations on scaled, translated and rotated shapes
# --------------------------------------------------------------------------------------- #
def test_exact_transforms_are_invertible():
    """Without repanelling, transform and inverse transform reproduce the shape to 1e-12."""
    af = generate_naca4("2412", 160)
    assert np.max(np.abs(translate(translate(af, 0.3, -0.2), -0.3, 0.2) - af)) <= 1e-12
    assert np.max(np.abs(scale(scale(af, 2.5), 1 / 2.5) - af)) <= 1e-12
    back = rotate(rotate(af, 7.0, pivot=(0.25, 0.01)), -7.0, pivot=(0.25, 0.01))
    assert np.max(np.abs(back - af)) <= 1e-12


def _distance_to_curve(points, curve):
    a, b = curve[:-1], curve[1:]
    ab = b - a
    t = np.clip(np.einsum("pij,ij->pi", points[:, None] - a[None], ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    return np.min(np.linalg.norm(points[:, None] - (a[None] + t[..., None] * ab[None]), axis=2), axis=1)


@_xfail("M2", "arc-length repanelling independent of chord length and position")
@pytest.mark.parametrize("op", ["scaled", "translated", "rotated"])
def test_repanelling_preserves_transformed_shape(op):
    """Repanelling a transformed section keeps every point within 1e-4 chord of the original
    curve (a 4000-panel generation of the same section, transformed the same way)."""
    fine, coarse = generate_naca4("2412", 4000), generate_naca4("2412", 160)
    f = {"scaled": lambda c: scale(c, 0.5), "translated": lambda c: translate(c, 0.2, 0.1),
         "rotated": lambda c: rotate(c, 10.0, pivot=(0.25, 0.0))}[op]
    chord = 0.5 if op == "scaled" else 1.0
    new = repanel_cosine(f(coarse), num_panels=160)
    assert np.max(_distance_to_curve(new, f(fine))) / chord <= 1e-4


# --------------------------------------------------------------------------------------- #
# Corrections applied exactly once
# --------------------------------------------------------------------------------------- #
def test_karman_tsien_applied_once():
    """Cl at Mach 0.5 equals the integral of the once-corrected incompressible Cp (1e-3)."""
    from foil_lab.corrections import karman_tsien_correction
    af = generate_naca4("2412", 160)
    ps = core.PanelSet.from_bodies(af)
    cp_inc = solve_flow(af, 4.0).cp_dist
    cl_once = loads_from_cp(ps, karman_tsien_correction(cp_inc, 0.5), 4.0)[0]
    assert abs(solve_flow(af, 4.0, mach=0.5).cl - cl_once) <= 1e-3


@_xfail("M6", "aspect-ratio correction must not apply compressibility a second time")
def test_aspect_ratio_correction_does_not_reapply_compressibility():
    """With a very large aspect ratio the Mach 0.5 Cl equals the 2D Mach 0.5 Cl (1e-3)."""
    af = generate_naca4("2412", 160)
    assert abs(solve_flow(af, 4.0, mach=0.5, aspect_ratio=1e6).cl
               - solve_flow(af, 4.0, mach=0.5).cl) <= 1e-3


# --------------------------------------------------------------------------------------- #
# NumPy and JAX agree; determinism
# --------------------------------------------------------------------------------------- #
_SECTIONS = {"0012": lambda: generate_naca4("0012", 200), "2412": lambda: generate_naca4("2412", 200),
             "4412": lambda: generate_naca4("4412", 200), "23012": lambda: generate_naca5("23012", 200)}


def _jax_cases():
    for alpha in (-4.0, 4.0, 10.0):
        for name, make in _SECTIONS.items():
            yield pytest.param(make, alpha, id=f"{alpha}-{name}")


@pytest.mark.parametrize("make,alpha", list(_jax_cases()))
def test_numpy_and_jax_agree(make, alpha):
    """Inviscid Cl agrees to 1e-6."""
    from foil_lab.jax_backend import HAS_JAX
    if not HAS_JAX:
        pytest.skip("JAX not installed")
    from foil_lab import jax_core
    af = make()
    assert abs(solve_flow(af, alpha).cl - jax_core.solve_numpy_compatible(af, alpha)[0]) <= 1e-6


def test_deterministic_inviscid_and_viscous():
    """Same input, bit-identical output (inviscid Cp, one-way drag, VII)."""
    from foil_lab.vii import solve_vii
    af = generate_naca4("2412", 160)
    s1, s2 = solve_flow(af, 4.0, reynolds=1e6), solve_flow(af, 4.0, reynolds=1e6)
    assert np.array_equal(s1.cp_dist, s2.cp_dist) and s1.cd_visc == s2.cd_visc
    v1 = solve_vii(af, 4.0, reynolds=1e6, max_iter=200, tol=1e-3)
    v2 = solve_vii(af, 4.0, reynolds=1e6, max_iter=200, tol=1e-3)
    assert np.array_equal(v1.solution.cp_dist, v2.solution.cp_dist) and v1.cd_wake == v2.cd_wake
