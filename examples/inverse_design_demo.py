"""Inverse design / optimization demo.

Three problems:
  1. design_for_cl  — find a shape that produces a target lift coefficient.
  2. match_cp       — recover a shape whose Cp matches a target airfoil.
  3. maximize_ld    — (optional, slow: --ld) maximize L/D with the coupled viscous solver.

Run from the repo root:  python examples/inverse_design_demo.py [--ld]
"""
import sys
import time

import numpy as np

import foil_lab as fl
from foil_lab.generators import generate_naca4
from foil_lab.design import design_for_cl, match_cp, maximize_ld


def inviscid_cl(coords, alpha):
    """Inviscid Cl of a coordinate array (no repanelling)."""
    return fl.analyze(fl.Airfoil.from_coords(coords), alpha, repanel=False).cl


def main():
    print("=== 1. Design for a target Cl ===")
    for target in (0.6, 0.9, 1.2):
        t = time.perf_counter()
        r = design_for_cl(target_cl=target, alpha_deg=5.0, num_panels=120,
                          max_iter=60)
        got = inviscid_cl(r.coords, 5.0)
        tc = r.coords[:, 1].max() - r.coords[:, 1].min()
        print(f"  target Cl={target:.2f} -> achieved {got:.4f}  "
              f"(t/c={tc:.3f}, {r.n_evals} evals, {time.perf_counter()-t:.1f}s)")

    print("\n=== 2. Inverse design: match a target Cp (recover NACA 2412) ===")
    target = generate_naca4("2412", 120)
    tgt_cl = inviscid_cl(target, 5.0)
    t = time.perf_counter()
    r = match_cp(target, alpha_deg=5.0, num_panels=120, max_iter=60)
    got_cl = inviscid_cl(r.coords, 5.0)
    tc = r.coords[:, 1].max() - r.coords[:, 1].min()
    print(f"  target: NACA 2412 (t/c=0.12, Cl={tgt_cl:.3f})")
    print(f"  recovered: t/c={tc:.3f}, Cl={got_cl:.3f}, {r.message}")
    print(f"  ({r.n_evals} evals, {time.perf_counter()-t:.1f}s)")
    print("  PARSEC cannot exactly equal a NACA shape, so this is the closest")
    print("  PARSEC fit — the honest result of an ill-posed inverse problem.")

    if "--ld" not in sys.argv:
        print("\n  (maximize_ld: run with --ld. It calls the coupled viscous solver for every")
        print("   candidate, about 10 s each; the reduced run below takes about 2 minutes.)")
        return

    print("\n=== 3. Maximize L/D at 4 deg, Re 3e6 (coupled viscous solver; reduced: 60 panels, "
          "2 iterations) ===")
    t = time.perf_counter()
    start = fl.analyze(fl.Airfoil.naca("2412"), 4.0, re=3e6)
    print(f"  reference NACA 2412: Cl={start.cl:.3f} Cd={start.cd:.5f} "
          f"L/D={start.cl / start.cd:.1f} converged={start.converged}")
    r = maximize_ld(alpha_deg=4.0, reynolds=3e6, num_panels=60, max_iter=2)
    print(f"  optimised PARSEC: {r.message}, success={r.success}")
    print(f"  ({r.n_evals} evals, {time.perf_counter()-t:.0f}s)")


if __name__ == "__main__":
    main()
