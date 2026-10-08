"""e^n transition demo: physical transition prediction vs the empirical Michel
criterion, and the effect of the environment (N_crit) on drag via VII.

Run from the repo root:  python examples/transition_demo.py
"""
import numpy as np

from foil_lab.generators import generate_naca4
from foil_lab.bl_integral import march_boundary_layer
from foil_lab.vii import solve_vii


def main():
    print("=== 1. Flat-plate transition: e^n vs Michel ===")
    Re = 5e6
    s = np.linspace(1e-4, 1.0, 500)
    ue = np.ones_like(s)
    for model in ("michel", "en"):
        p = march_boundary_layer(s, ue, Re, transition_model=model)
        print(f"  {model:7s}: x_transition = {s[p.transition_idx]:.3f}")

    print("\n=== 2. e^n environment sensitivity (N_crit) on the flat plate ===")
    print("  (N_crit ~9 clean, ~4-7 noisy/rough, ~11-13 very quiet)")
    for nc in (5, 7, 9, 11, 13):
        p = march_boundary_layer(s, ue, Re, transition_model="en", n_crit=nc)
        print(f"  N_crit={nc:2d}: x_transition = {s[p.transition_idx]:.3f}")

    print("\n=== 3. N_crit drives drag through VII (NACA 2412, a=4, Re=3e6) ===")
    af = generate_naca4("2412", 200)
    for nc in (5, 9, 12):
        r = solve_vii(af, 4.0, reynolds=3e6, transition_model="en",
                      n_crit=nc, max_iter=200, tol=1e-3)
        env = {5: "noisy/dirty", 9: "clean", 12: "very quiet"}[nc]
        print(f"  N_crit={nc:2d} ({env:11s}): Cl={r.solution.cl:.3f} "
              f"Cd={r.cd_wake:.5f}  converged={r.converged}")
    print("\n  Earlier transition (low N_crit) wets more turbulent area -> more drag.")
    print("  This environment sensitivity is exactly what Michel cannot provide.")


if __name__ == "__main__":
    main()
