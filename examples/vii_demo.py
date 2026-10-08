"""Viscous-Inviscid Interaction (VII) demo.

Shows how the transpiration coupling improves drag and pressure over the one-way
(inviscid -> BL) path, and where it converges.

Run from the repo root:  python examples/vii_demo.py
"""
from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab.vii import solve_vii


def main():
    af = generate_naca4("2412", 200)
    Re = 3e6

    print(f"NACA 2412, Re={Re:.0e} — VII vs one-way coupling\n")
    print(f"{'alpha':>6} {'Cl_VII':>8} {'Cl_inv':>8} {'Cd_VII':>9} {'Cd_1way':>9} "
          f"{'conv':>6} {'iters':>6}")
    print("-" * 56)
    for a in (0, 2, 4, 6, 8):
        r = solve_vii(af, float(a), reynolds=Re, max_iter=200, tol=1e-3)
        inv = solve_flow(af, float(a), reynolds=Re)
        print(f"{a:>6} {r.solution.cl:>8.4f} {inv.cl:>8.4f} "
              f"{r.cd_wake:>9.5f} {inv.cd_visc:>9.5f} "
              f"{str(r.converged):>6} {r.iterations:>6}")

    print("\nReading the results:")
    print("  * VII Cl sits below the inviscid Cl — the displacement thickness")
    print("    effectively decambers the airfoil (real viscous effect the")
    print("    one-way path misses).")
    print("  * VII Cd RISES with alpha as the boundary layer thickens; the")
    print("    one-way Cd is nearly flat and unrealistically low.")
    print("  * VII Cd ~0.006-0.008 matches NACA 2412 experimental data; the")
    print("    one-way ~0.0011 does not.")
    print("\n  Note: the direct transpiration formulation converges for attached")
    print("  and mildly-separated flow. Past ~8-10 deg it may not fully")
    print("  converge (converged=False) — deep separation needs the inverse")
    print("  formulation, the documented follow-on.")


if __name__ == "__main__":
    main()
