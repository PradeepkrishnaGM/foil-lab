"""Tests for the JAX backend: NumPy parity, autodiff correctness, vmap, batch.

These are skipped automatically if JAX is not installed, so the suite still runs
on a JAX-free environment.
"""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab.jax_backend import HAS_JAX

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


@pytest.mark.parametrize("naca,alpha", [
    ("0012", 5.0), ("2412", 5.0), ("2412", 0.0), ("4412", 8.0), ("0012", -3.0),
])
def test_jax_matches_numpy(naca, alpha):
    from foil_lab import jax_core
    af = generate_naca4(naca, 200)
    cl_np = solve_flow(af, alpha).cl
    cl_jax, _ = jax_core.solve_numpy_compatible(af, alpha)
    assert abs(cl_jax - cl_np) < 1e-5


def test_symmetric_zero_lift_jax():
    from foil_lab import jax_core
    cl, _ = jax_core.solve_numpy_compatible(generate_naca4("0012", 200), 0.0)
    assert abs(cl) < 1e-6


@pytest.mark.slow
def test_autodiff_matches_finite_difference():
    from foil_lab import jax_core
    cw = jax_core.ensure_cw_static(generate_naca4("2412", 200))
    a0 = np.deg2rad(5.0)
    dcl = float(jax_core.dcl_dalpha(cw, a0))
    h = 1e-6
    fd = float((jax_core.cl_of_alpha(cw, a0 + h)
                - jax_core.cl_of_alpha(cw, a0 - h)) / (2 * h))
    assert abs(dcl - fd) < 1e-5


@pytest.mark.slow
def test_geometry_gradient_shape_and_fd():
    from foil_lab import jax_core
    af = generate_naca4("2412", 120)
    cw = jax_core.ensure_cw_static(af)
    a0 = np.deg2rad(5.0)
    g = jax_core.dcl_dgeom(cw, a0)
    assert g.shape == cw.shape
    # FD spot-check one node's y-sensitivity.
    i = 30
    base = float(jax_core.cl_of_alpha(cw, a0))
    pert = cw.copy(); pert[i, 1] += 1e-6
    fd = (float(jax_core.cl_of_alpha(pert, a0)) - base) / 1e-6
    assert abs(float(g[i, 1]) - fd) < 1e-2


def test_vmap_batch_matches_loop():
    from foil_lab import jax_core
    af = generate_naca4("2412", 160)
    alphas = np.linspace(-5, 12, 18)
    cls_batch, _ = jax_core.batch_polar(af, alphas)
    cls_loop = np.array([solve_flow(af, a).cl for a in alphas])
    assert np.max(np.abs(cls_batch - cls_loop)) < 1e-5


def test_lift_slope_and_alpha0():
    from foil_lab import jax_core
    slope, a0 = jax_core.lift_slope_and_alpha0(generate_naca4("2412", 200))
    assert 5.8 < slope < 7.2          # ~2*pi
    assert -3.0 < a0 < -1.2           # ~ -2.1 deg


def test_jax_batch_path():
    from foil_lab.batch import run_batch_jax
    afs = {"a": generate_naca4("2412", 160), "b": generate_naca4("0012", 160)}
    rows = run_batch_jax(afs, np.linspace(-4, 10, 15))
    assert len(rows) == 30
    assert all(r["converged"] for r in rows)


def test_jax_core_leaves_global_x64_unchanged():
    import jax
    from foil_lab import jax_core
    before = jax.config.x64_enabled
    jax_core.batch_polar(generate_naca4("2412", 120), [0.0, 4.0])
    jax_core.dcl_dalpha(jax_core.ensure_cw_static(generate_naca4("2412", 120)), 0.05)
    assert jax.config.x64_enabled == before
