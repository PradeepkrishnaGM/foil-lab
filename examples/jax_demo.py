"""JAX backend demo: compiled batch polars and exact aerodynamic gradients.

Run from the repo root:  python examples/jax_demo.py

Requires the optional JAX dependency:  pip install 'jax[cpu]'
"""
import time

import numpy as np

from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab.jax_backend import HAS_JAX


def main():
    if not HAS_JAX:
        print("JAX is not installed. Install with: pip install 'jax[cpu]'")
        return

    from foil_lab import jax_core

    af = generate_naca4("2412", 200)

    print("=== 1. NumPy parity ===")
    for a in (0.0, 5.0, 10.0):
        cl_np = solve_flow(af, a).cl
        cl_jx, _ = jax_core.solve_numpy_compatible(af, a)
        print(f"  a={a:4.1f}  NumPy={cl_np:+.6f}  JAX={cl_jx:+.6f}  d={abs(cl_np-cl_jx):.1e}")

    print("\n=== 2. vmap batch speedup (200-alpha sweep) ===")
    alphas = np.linspace(-8, 16, 200)
    t = time.perf_counter()
    _ = [solve_flow(af, a).cl for a in alphas]
    t_np = time.perf_counter() - t
    cw = jax_core.ensure_cw_static(af)
    aj = np.deg2rad(alphas)
    jax_core.batched_cl(cw, aj)                           # warm compile
    t = time.perf_counter()
    jax_core.batched_cl(cw, aj)
    t_jx = time.perf_counter() - t
    print(f"  NumPy loop: {t_np*1000:7.1f} ms")
    print(f"  JAX vmap:   {t_jx*1000:7.1f} ms   ({t_np/t_jx:.0f}x faster)")

    print("\n=== 3. Exact gradients (autodiff) ===")
    slope, a0 = jax_core.lift_slope_and_alpha0(af)
    print(f"  dCl/dalpha = {slope:.4f} /rad   (thin-airfoil 2*pi = {2*np.pi:.4f})")
    print(f"  zero-lift angle = {a0:.2f} deg  (NACA 2412 ~ -2.1 deg)")
    # Geometry sensitivity: which nodes most affect lift?
    g = jax_core.dcl_dgeom(cw, np.deg2rad(5.0))
    k = int(np.argmax(np.abs(g[:, 1])))
    print(f"  most lift-sensitive node: #{k}, dCl/dy = {g[k,1]:+.2f} "
          f"(at x/c={float(cw[k,0]):.3f})")
    print("  -> these gradients are what gradient-based inverse design will use.")


if __name__ == "__main__":
    main()
