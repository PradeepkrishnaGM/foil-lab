"""Demonstrates the coupled viscous solver, ground effect, simple sweep theory and the advanced
generators through the ``fl.analyze`` interface.

Run from the repo root:  python examples/advanced_features.py
"""
import foil_lab as fl
from foil_lab.generators import generate_biconvex, generate_parsec, generate_wedge


def main():
    af = fl.Airfoil.naca("2412")

    print("=== Coupled viscous solver (Re = 3e6) ===")
    pol = fl.polar(af, [0.0, 5.0, 10.0], re=3e6)
    for r in pol.results:
        print(f"  a={r.alpha:4.1f}  Cl={r.cl:+.3f}  Cd={r.cd:.4f}  Cm={r.cm:+.3f}  "
              f"converged={r.converged}")

    print("\n=== Ground effect (a=5, pitched about the quarter chord) ===")
    free = fl.analyze(af, 5.0).cl
    print(f"  free air        Cl={free:+.3f}")
    for h in (2.0, 1.0, 0.5, 0.3):
        r = fl.analyze(af, 5.0, ground_height=h)
        print(f"  h/c={h:<4}      Cl={r.cl:+.3f}  ({r.cl / free:.3f}x)")

    print("\n=== Simple sweep theory (a=5) ===")
    for sweep in (0, 15, 30, 45):
        r = fl.analyze(af, 5.0, sweep=float(sweep))
        print(f"  sweep={sweep:2d}deg  Cl={r.cl:+.3f}")

    print("\n=== Advanced generators (Cl at a=0, on the generated points) ===")
    parsec = generate_parsec(dict(rle=0.01, x_up=0.4, y_up=0.06, yxx_up=-0.5,
                                  x_lo=0.4, y_lo=-0.04, yxx_lo=0.5, y_te=0.0,
                                  dy_te=0.0, ang_te=-8.0, ang_wedge=8.0), 160)
    for name, coords in (("biconvex", generate_biconvex(0.06)), ("wedge", generate_wedge(0.05)),
                         ("parsec", parsec)):
        r = fl.analyze(fl.Airfoil.from_coords(coords, name=name), 0.0, repanel=False)
        print(f"  {name:<10} Cl={r.cl:+.4f}")


if __name__ == "__main__":
    main()
