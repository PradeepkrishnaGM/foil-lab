"""JAX entry points to the inviscid core (single body): jit, vmap and grad.

The assembly and solve are those of :mod:`foil_lab.inviscid`, run with ``xp = jax.numpy``, so
the JAX and NumPy paths do the same arithmetic. This module only adds compiled, batched and
differentiated wrappers:

* ``jax.jit``  -> the solve compiles to XLA;
* ``jax.vmap`` -> polars over many angles in one call;
* ``jax.grad`` -> exact dCl/dalpha and dCl/d(node coordinates) for design.

Precision: every public function runs inside ``jax.enable_x64(True)`` (a scoped setting) and
takes NumPy arrays; JAX's global configuration is not changed. Inputs may be in any
orientation; they are made clockwise in NumPy before tracing (:func:`ensure_cw_static`).
Whether the trailing edge is open (a Drela trailing-edge panel) is decided from the concrete
coordinates before tracing, so it is a static property of a compiled function.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

from . import inviscid as core


def ensure_cw_static(coords: np.ndarray) -> np.ndarray:
    """Return clockwise-ordered coords (NumPy, done before tracing)."""
    c = np.asarray(coords, dtype=float)
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return c if area < 0.0 else c[::-1].copy()


def _solve_single(coords, alpha_rad, open_te: bool):
    """(cp, cl, cm about (0.25, 0)) for clockwise ``coords``; traceable."""
    ps = core.PanelSet.from_clockwise_nodes(coords, open_te)
    sol = core.solve(ps, jnp.rad2deg(alpha_rad), xp=jnp)
    cl, _, cm = core.pressure_loads(ps, sol.cp, jnp.rad2deg(alpha_rad))
    return sol.cp, cl, cm


@functools.partial(jax.jit, static_argnums=2)
def _cl_cm(coords, alpha_rad, open_te):
    _, cl, cm = _solve_single(coords, alpha_rad, open_te)
    return cl, cm


@functools.partial(jax.jit, static_argnums=2)
def _cp(coords, alpha_rad, open_te):
    return _solve_single(coords, alpha_rad, open_te)[0]


def _cl(coords, alpha_rad, open_te):
    return _solve_single(coords, alpha_rad, open_te)[1]


_batched_cl = jax.jit(jax.vmap(_cl, in_axes=(None, 0, None)), static_argnums=2)
_batched_cm = jax.jit(jax.vmap(lambda c, a, o: _solve_single(c, a, o)[2], in_axes=(None, 0, None)),
                      static_argnums=2)
_dcl_dalpha = jax.jit(jax.grad(_cl, argnums=1), static_argnums=2)
_dcl_dgeom = jax.jit(jax.grad(_cl, argnums=0), static_argnums=2)


def _prep(coords, alpha_rad):
    c = np.asarray(coords, dtype=float)
    return jnp.asarray(c), jnp.asarray(np.asarray(alpha_rad, dtype=float)), core.has_open_te(c)


# --------------------------------------------------------------------------- #
# Public entry points (clockwise coords, alpha in radians unless stated)
# --------------------------------------------------------------------------- #
def solve_cl_cm(coords, alpha_rad):
    """(cl, cm_qc) for clockwise coords and alpha in radians (compiled)."""
    with jax.enable_x64(True):
        c, a, o = _prep(coords, alpha_rad)
        cl, cm = _cl_cm(c, a, o)
        return float(cl), float(cm)


def solve_cp(coords, alpha_rad):
    """Surface Cp at the panel midpoints, solver (clockwise) order (compiled)."""
    with jax.enable_x64(True):
        c, a, o = _prep(coords, alpha_rad)
        return np.asarray(_cp(c, a, o))


def cl_of_alpha(coords, alpha_rad):
    """Scalar Cl."""
    with jax.enable_x64(True):
        c, a, o = _prep(coords, alpha_rad)
        return float(_cl(c, a, o))


def batched_cl(coords, alphas_rad):
    """Cl for an array of angles (radians) in one vectorized call."""
    with jax.enable_x64(True):
        c, a, o = _prep(coords, alphas_rad)
        return np.asarray(_batched_cl(c, a, o))


def dcl_dalpha(coords, alpha_rad):
    """Exact dCl/dalpha (per radian)."""
    with jax.enable_x64(True):
        c, a, o = _prep(coords, alpha_rad)
        return float(_dcl_dalpha(c, a, o))


def dcl_dgeom(coords, alpha_rad):
    """Exact gradient of Cl with respect to every node coordinate, shape (N + 1, 2)."""
    with jax.enable_x64(True):
        c, a, o = _prep(coords, alpha_rad)
        return np.asarray(_dcl_dgeom(c, a, o))


def solve_numpy_compatible(coords_ccw_or_cw, alpha_deg):
    """Any-orientation coords and degrees; returns (cl, cm_qc) floats as solve_flow reports."""
    cw = ensure_cw_static(coords_ccw_or_cw)
    return solve_cl_cm(cw, np.deg2rad(alpha_deg))


def batch_polar(coords, alphas_deg):
    """Cl and Cm polar for one section (any orientation) in one vectorized call.

    Returns ``(cls, cms)``, NumPy arrays with one entry per angle.
    """
    cw = ensure_cw_static(coords)
    a = np.deg2rad(np.asarray(alphas_deg, dtype=float))
    with jax.enable_x64(True):
        c, aj, o = _prep(cw, a)
        return np.asarray(_batched_cl(c, aj, o)), np.asarray(_batched_cm(c, aj, o))


def lift_slope_and_alpha0(coords, alpha_ref_deg=0.0):
    """(dCl/dalpha per radian, zero-lift angle in degrees) from one exact derivative."""
    cw = ensure_cw_static(coords)
    a_ref = np.deg2rad(alpha_ref_deg)
    cl0 = cl_of_alpha(cw, a_ref)
    slope = dcl_dalpha(cw, a_ref)
    return float(slope), float(np.rad2deg(a_ref - cl0 / slope))
