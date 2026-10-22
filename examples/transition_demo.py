"""Transition demo: e^n envelope transition and forced trips in the coupled viscous solver.

NACA 2412 at 4 deg, Re 3e6. The critical amplification factor N_crit stands for the
disturbance environment (about 9 for a clean wind tunnel, lower for a noisier one); a forced
trip fixes transition on one surface.

Run from the repo root:  python examples/transition_demo.py   (about a minute)
"""
import foil_lab as fl


def show(label, r):
    status = "converged" if r.converged else "NOT converged"
    print(f"  {label:24s} x_tr upper {r.xtr_upper:.3f}  lower {r.xtr_lower:.3f}  "
          f"Cd {r.cd:.5f} (friction {r.cd_friction:.5f})  Cl {r.cl:.3f}  [{status}]")


def main():
    af = fl.Airfoil.naca("2412")
    print("=== 1. N_crit (e^n envelope method), NACA 2412, 4 deg, Re 3e6 ===")
    for nc in (5.0, 9.0, 12.0):
        show(f"N_crit {nc:g}", fl.analyze(af, 4.0, re=3e6, n_crit=nc))

    print("\n=== 2. Forced trips (x/c) ===")
    show("free", fl.analyze(af, 4.0, re=3e6))
    show("upper tripped at 0.05", fl.analyze(af, 4.0, re=3e6, trip_upper=0.05))
    show("both tripped at 0.05", fl.analyze(af, 4.0, re=3e6, trip_upper=0.05, trip_lower=0.05))
    print("\n  A lower N_crit or an earlier trip moves transition forward and raises friction drag.")
    print("  ('nan' transition: laminar to the trailing edge.)")


if __name__ == "__main__":
    main()
