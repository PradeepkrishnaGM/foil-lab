"""Inviscid polar of a section read from a coordinate file, with plots saved as PNG files.

Run from anywhere:  python examples/basic_alpha_sweep.py [path/to/section.dat]
(default: naca2412.dat next to this script). Writes basic_polar.png and basic_cp.png to the
current directory.
"""

import sys
from pathlib import Path

import numpy as np

import foil_lab as fl
from foil_lab import plot


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("naca2412.dat")
    af = fl.Airfoil.from_file(path)
    pol = fl.polar(af, np.arange(-5.0, 10.5, 1.0))
    cl, alpha = pol.cl, pol.alpha
    slope = np.polyfit(alpha, cl, 1)
    print(f"{af.name}: lift slope {np.degrees(slope[0]) / (2 * np.pi):.3f} x 2 pi per rad, "
          f"zero-lift angle {-slope[1] / slope[0]:.2f} deg")
    fig, _ = plot.polars(pol, labels=[af.name], panels=(("alpha", "cl"), ("alpha", "cm")))
    fig.savefig("basic_polar.png", dpi=120)
    fig, _ = plot.cp(fl.analyze(af, 5.0))
    fig.savefig("basic_cp.png", dpi=120)
    print("wrote basic_polar.png and basic_cp.png")


if __name__ == "__main__":
    main()
