"""Inverse design / optimization demo.

Three problems:
  1. design_for_cl  — find a shape that produces a target lift coefficient.
  2. match_cp       — recover a shape whose Cp matches a target airfoil.
  3. maximize_ld    — (optional, slow) maximize L/D via the VII viscous solver.

Run from the repo root:  python examples/inverse_design_demo.py
"""
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np

from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab.design import design_for_cl, match_cp


def main():
    print("=== 1. Design for a target Cl ===")
    for target in (0.6, 0.9, 1.2):
        t = time.perf_counter()
        r = design_for_cl(target_cl=target, alpha_deg=5.0, num_panels=120,
                          max_iter=60)
        got = solve_flow(r.coords, 5.0).cl
        tc = r.coords[:, 1].max() - r.coords[:, 1].min()
        print(f"  target Cl={target:.2f} -> achieved {got:.4f}  "
              f"(t/c={tc:.3f}, {r.n_evals} evals, {time.perf_counter()-t:.1f}s)")

    print("\n=== 2. Inverse design: match a target Cp (recover NACA 2412) ===")
    target = generate_naca4("2412", 120)
    tgt_cl = solve_flow(target, 5.0).cl
    t = time.perf_counter()
    r = match_cp(target, alpha_deg=5.0, num_panels=120, max_iter=60)
    got_cl = solve_flow(r.coords, 5.0).cl
    tc = r.coords[:, 1].max() - r.coords[:, 1].min()
    print(f"  target: NACA 2412 (t/c=0.12, Cl={tgt_cl:.3f})")
    print(f"  recovered: t/c={tc:.3f}, Cl={got_cl:.3f}, {r.message}")
    print(f"  ({r.n_evals} evals, {time.perf_counter()-t:.1f}s)")
    print("  PARSEC cannot exactly equal a NACA shape, so this is the closest")
    print("  PARSEC fit — the honest result of an ill-posed inverse problem.")

    print("\n  (maximize_ld is available too, but routes through the VII viscous")
    print("   solver for physical drag and is correspondingly slower — see the")
    print("   design.maximize_ld docstring.)")


if __name__ == "__main__":
    main()
