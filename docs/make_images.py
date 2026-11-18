"""Regenerate the README images headless (no screen):  python docs/make_images.py

Writes PNG files to docs/images/. The viscous polar takes about a minute on one core.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import foil_lab as fl  # noqa: E402
from foil_lab import plot  # noqa: E402
from foil_lab.benchmark.references import load_reference  # noqa: E402
from foil_lab.benchmark.sections import canonical  # noqa: E402
from foil_lab.flowfield import velocity_field  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    warnings.simplefilter("ignore")
    af = fl.Airfoil.naca("2412")

    # 1. viscous polar against XFOIL on the same coordinate file
    sec = fl.Airfoil.from_coords(canonical("NACA2412").coords, name="NACA 2412")
    pol = fl.polar(sec, np.arange(-4.0, 13.0, 1.0), re=1e6, repanel=False)
    fig, _ = plot.polars(pol, labels=["Foil-Lab"], reference=load_reference("NACA2412", 1e6, 160),
                         reference_label="XFOIL 6.99")
    fig.suptitle("NACA 2412, Re 1e6, N_crit 9 (same coordinate file for both codes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "polar_naca2412_re1e6.png", dpi=110)
    plt.close(fig)

    # 2. plain flap: geometry and Cp at three deflections
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    res = []
    for d in (0.0, 10.0, 20.0):
        f = af.with_flap(0.75, d)
        plot.geometry(f, ax=axes[0])
        res.append(fl.analyze(f, 4.0))
    axes[0].set_title("25 % plain flap at 0, 10, 20 deg")
    axes[0].get_legend() and axes[0].get_legend().remove()
    plot.cp(res, ax=axes[1], labels=[f"{d:g} deg (Ch {r.ch:+.4f})" for d, r in zip((0, 10, 20), res)])
    axes[1].set_title("inviscid Cp at alpha 4 deg")
    fig.tight_layout()
    fig.savefig(OUT / "flap_naca2412.png", dpi=110)
    plt.close(fig)

    # 3. slotted flap: streamlines of the two-element inviscid solution
    cfg = fl.Configuration((fl.Element(af), fl.Element(fl.Airfoil.naca("2412"), scale=0.3, rotation_deg=20.0,
                                                       pivot=(0.0, 0.0), offset=(0.97, -0.04))))
    ff = velocity_field(cfg, 4.0, grid_res=160, margin=0.3)
    fig, ax = plt.subplots(figsize=(9, 4))
    sp = np.hypot(ff.u, ff.v)
    s = ax.streamplot(ff.x, ff.y, np.ma.masked_where(ff.inside, ff.u), np.ma.masked_where(ff.inside, ff.v),
                      color=np.ma.masked_where(ff.inside, sp), cmap="viridis", density=1.8, linewidth=0.9,
                      arrowsize=0.7)
    for b in ff.bodies:
        ax.fill(b[:, 0], b[:, 1], color="0.6", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(ff.x.min(), ff.x.max())
    ax.set_ylim(ff.y.min(), ff.y.max())
    fig.colorbar(s.lines, ax=ax, label="speed / freestream", fraction=0.03)
    r = fl.analyze(cfg, 4.0)
    ax.set_title(f"NACA 2412 with a slotted flap, alpha 4 deg (inviscid): Cl {r.cl:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "slotted_flap_streamlines.png", dpi=110)
    plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
