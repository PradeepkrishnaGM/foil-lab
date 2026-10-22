"""Demonstrates the coupled viscous solver, ground effect, Busemann sweep, constant wing
loading, and the advanced generators. (Surface roughness and the empirical post-stall model were
removed with the one-way boundary layer in M3.)

Run from the repo root:  python examples/advanced_features.py
"""
from foil_lab.generators import (generate_naca4, generate_biconvex,
                                  generate_wedge, generate_parsec)
from foil_lab.solver import solve_flow
from foil_lab.advanced import solve_swept_wing, solve_constant_wing_loading
from foil_lab.ground_effect import solve_ground_effect


def main():
    af = generate_naca4("2412", num_panels=200)

    print("=== Coupled viscous solver (Re = 3e6) ===")
    for a in (0, 5, 10):
        s = solve_flow(af, a, reynolds=3e6)
        print(f"  a={a:2d}  Cl={s.cl:+.3f}  Cd={s.cd_total:.4f}  Cm={s.cm_qc:+.3f}  "
              f"converged={s.converged}")

    print("\n=== Ground effect (a=5) ===")
    free = solve_flow(af, 5.0).cl
    print(f"  free air        Cl={free:+.3f}")
    for h in (2.0, 1.0, 0.5, 0.3):
        s = solve_ground_effect(af, 5.0, height=h)
        print(f"  h/c={h:<4}      Cl={s.cl:+.3f}  ({s.cl/free:.3f}x)")

    print("\n=== Busemann sweep (a=5) ===")
    for sweep in (0, 15, 30, 45):
        r = solve_swept_wing(af, 5.0, float(sweep))
        print(f"  sweep={sweep:2d}deg  Cl={r.cl_3d:+.3f}")

    print("\n=== Constant wing loading (W/S=600 N/m2, c=1.2 m) ===")
    for a in (2, 5, 8):
        p = solve_constant_wing_loading(af, float(a), 600.0, 1.2)
        print(f"  a={a}  Cl={p.cl:.3f}  V={p.speed:.1f} m/s  "
              f"Re={p.reynolds:.2e}  converged={p.converged}")

    print("\n=== Advanced generators (Cl at a=0) ===")
    print(f"  biconvex   Cl={solve_flow(generate_biconvex(0.06), 0.0).cl:+.4f}")
    print(f"  wedge      Cl={solve_flow(generate_wedge(0.05), 0.0).cl:+.4f}")
    parsec = generate_parsec(dict(rle=0.01, x_up=0.4, y_up=0.06, yxx_up=-0.5,
                                  x_lo=0.4, y_lo=-0.04, yxx_lo=0.5, y_te=0.0,
                                  dy_te=0.0, ang_te=-8.0, ang_wedge=8.0), 160)
    print(f"  parsec     Cl={solve_flow(parsec, 0.0).cl:+.3f}")


if __name__ == "__main__":
    main()
