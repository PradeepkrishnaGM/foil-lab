"""Headless parametric batch: sweep airfoils x alpha x Re, dump to Parquet.

Run from the repo root:  python examples/batch_sweep.py
"""
from pathlib import Path

from foil_lab.generators import generate_naca4, generate_naca5
from foil_lab.batch import build_matrix, run_batch, write_table


def main():
    airfoils = {
        "naca0012": generate_naca4("0012", 200),
        "naca2412": generate_naca4("2412", 200),
        "naca4412": generate_naca4("4412", 200),
        "naca23012": generate_naca5("23012", 200),
    }

    cases = build_matrix(
        airfoils,
        alpha=range(-6, 15),
        reynolds=[1e6, 3e6, 6e6],
        roughness=[0, 2],
        enable_post_stall=True,
    )
    print(f"Solving {len(cases)} cases...")

    rows = run_batch(cases, workers=1, progress=True)
    n_ok = sum(r["converged"] for r in rows)
    print(f"Done: {n_ok}/{len(rows)} converged")

    out = Path(__file__).parent / "sweep_results.parquet"
    write_table(rows, str(out))
    print(f"Wrote {out}")

    # Quick peek: best L/D per airfoil at Re=3e6, clean.
    by_foil = {}
    for r in rows:
        if r["reynolds"] == 3e6 and r["roughness"] == 0 and r["converged"] \
                and r["cd_total"] > 1e-6 and r["cl"] > 0:
            ld = r["cl"] / r["cd_total"]
            best = by_foil.get(r["airfoil"])
            if best is None or ld > best[0]:
                by_foil[r["airfoil"]] = (ld, r["alpha_deg"])
    print("\nBest L/D (Re=3e6, clean):")
    for foil, (ld, a) in sorted(by_foil.items()):
        print(f"  {foil:12s} L/D={ld:6.1f} at alpha={a:+.0f} deg")


if __name__ == "__main__":
    main()
