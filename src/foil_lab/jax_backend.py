"""Optional JAX backend access.

JAX is an optional dependency. Import everything JAX-related through this module
so the rest of Foil Lab degrades gracefully when JAX is not installed:

    from foil_lab.jax_backend import HAS_JAX, jax_core

    if HAS_JAX:
        cls, cms = jax_core.batch_polar(coords, alphas)
    else:
        ... fall back to the NumPy path ...
"""

from __future__ import annotations

try:
    import jax  # noqa: F401
    from . import jax_core
    HAS_JAX = True
except Exception:  # ImportError, or JAX present but broken on this platform
    jax_core = None
    HAS_JAX = False


def require_jax() -> None:
    """Raise a clear error if a JAX-only path is used without JAX installed."""
    if not HAS_JAX:
        raise RuntimeError(
            "This feature requires JAX. Install it with:  "
            "pip install 'jax[cpu]'  (or the CUDA/TPU build for GPU/TPU)."
        )
