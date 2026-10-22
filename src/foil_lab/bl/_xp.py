"""Array-namespace helpers shared by the closure sets and the discrete equations.

Every closure is written against an array namespace (NumPy or jax.numpy) and accepts complex
arguments (complex-step derivatives). Branch conditions use the real part; each branch of a
piecewise relation is evaluated only on arguments valid for it, so derivatives stay finite.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ns(x: Any):
    """The array namespace of ``x``: jax.numpy for JAX values, NumPy otherwise."""
    if type(x).__module__.split(".")[0] in ("jax", "jaxlib"):
        import jax.numpy as jnp
        return jnp
    return np


def re(x):
    """Real part (branch conditions of complex-step evaluations)."""
    return x.real if hasattr(x, "real") else x


def branch(xp, cond, x, safe_true, safe_false, f_true, f_false):
    """where(cond, f_true(x), f_false(x)), each branch evaluated on safe arguments only."""
    a = f_true(xp.where(cond, x, safe_true))
    b = f_false(xp.where(cond, safe_false, x))
    return xp.where(cond, a, b)


def maximum(xp, a, b):
    """max(a, b) by the real part (complex-step safe)."""
    return xp.where(re(a) >= re(b), a, b)


def minimum(xp, a, b):
    """min(a, b) by the real part (complex-step safe)."""
    return xp.where(re(a) <= re(b), a, b)
