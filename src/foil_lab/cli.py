"""Command line of Foil-Lab (SPEC.md 4.8), on the public analysis API.

Examples
--------
    foillab analyze --naca 2412 --alpha 5 --re 1e6
    foillab analyze --name e387 --alpha 3 --re 3e5 --output point.json
    foillab analyze --naca 0012 --flap 0.75 10 --alpha 2 --re 3e6
    foillab polar --naca 0012 --start -4 --end 12 --step 1 --re 3e6 --output polar.csv
    foillab polar --naca 2412 --start 0 --end 8 --mach 0.5 --aspect-ratio 8
    foillab polar --naca 2412 --start 0 --end 8 --re 1e6 --db polars.sqlite
    foillab coords --name clarky --output clarky.dat
    foillab benchmark viscous-report --output benchmark_output
    foillab info

Without ``--re`` the result is inviscid. With a Reynolds number the coupled viscous-inviscid solver
runs (Mach 0): transition by the e^n envelope method (``--ncrit``) with optional forced trips
(``--trip-upper``, ``--trip-lower``, x/c). Every line says whether the point converged. Invalid
input gives a one-line "error: ..." message and a non-zero exit status, never a traceback.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


class InputError(Exception):
    """Invalid command-line input (reported as one "error: ..." line)."""


def _airfoil(args):
    import foil_lab as fl
    given = [x for x in (args.naca, args.dat, args.name) if x]
    if len(given) != 1:
        raise InputError("give exactly one of --naca CODE, --dat FILE or --name NAME")
    if args.naca:
        code = args.naca.strip()
        if not (code.isdigit() and len(code) in (4, 5)):
            raise InputError(f"--naca expects 4 or 5 digits, got {args.naca!r}")
        af = fl.Airfoil.naca(code)
    elif args.dat:
        if not Path(args.dat).exists():
            raise InputError(f"coordinate file not found: {args.dat}")
        af = fl.Airfoil.from_file(args.dat)
    else:
        af = fl.Airfoil.from_name(args.name)
    if getattr(args, "flap", None):
        hinge_x, deflection = args.flap
        af = af.with_flap(hinge_x, deflection) if args.flap_y is None else \
            af.with_flap(hinge_x, deflection, args.flap_y)
    return af


def _kw(args) -> dict:
    kw = dict(re=args.re or None, num_panels=args.panels, mach=args.mach, sweep=args.sweep,
              aspect_ratio=args.aspect_ratio, ground_height=args.ground_height)
    if args.re:
        kw.update(n_crit=args.ncrit, trip_upper=args.trip_upper, trip_lower=args.trip_lower)
    elif any(v is not None for v in (args.ncrit, args.trip_upper, args.trip_lower)):
        raise InputError("--ncrit and --trip-upper/--trip-lower need a Reynolds number (--re)")
    return kw


def _f(v, fmt: str) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)


def _status(r) -> str:
    if not r.reynolds:
        tag = "inviscid"
    else:
        tag = f"converged ({r.iterations} it)" if r.converged else "NOT converged"
    return tag + ("; " + "; ".join(r.warnings) if r.warnings else "")


def cmd_analyze(args) -> None:
    import foil_lab as fl
    af = _airfoil(args)
    r = fl.analyze(af, args.alpha, **_kw(args))
    print(f"{af.name}  alpha {r.alpha:g} deg  Re {_f(r.reynolds, 'g') if r.reynolds else 'inviscid'}"
          f"  Mach {r.mach:g}  [{_status(r)}]")
    print(f"  Cl  = {r.cl:+.4f}")
    if r.cd is not None:
        parts = f"friction {_f(r.cd_friction, '.5f')}, pressure {_f(r.cd_pressure, '.5f')}"
        if r.cd_induced is not None:
            parts += f", induced {r.cd_induced:.5f}"
        print(f"  Cd  = {r.cd:.5f}  ({parts})")
        if r.reynolds:
            print(f"  x_tr upper / lower = {_f(r.xtr_upper, '.3f')} / {_f(r.xtr_lower, '.3f')}")
    else:
        print("  Cd  = -  (inviscid: no profile drag)" if r.cd_induced is None
              else f"  Cd  = induced {r.cd_induced:.5f} only (inviscid)")
    print(f"  Cm  = {r.cm:+.4f}  (about {tuple(round(v, 4) for v in r.ref_point)})")
    if r.ch is not None:
        print(f"  Ch  = {r.ch:+.5f}  (hinge moment, trailing edge down positive)")
    print(f"  critical Mach = {_f(r.mach_critical, '.3f')}")
    for c in r.corrections:
        print(f"  correction: {c}")
    if args.output:
        _write(r, args.output, af.name)


def cmd_polar(args) -> None:
    import numpy as np
    import foil_lab as fl
    af = _airfoil(args)
    if args.step <= 0:
        raise InputError("--step must be positive")
    alphas = [round(float(a), 10) for a in np.arange(args.start, args.end + 1e-9, args.step)]
    if not alphas:
        raise InputError("--start is above --end")
    kw = _kw(args)
    if args.db:
        from .database import PolarDatabase
        with PolarDatabase(args.db) as db:
            pol = _RowsPolar(db.polar(af, alphas, **kw))
        print(f"# database {args.db}")
    else:
        pol = fl.polar(af, alphas, **kw)
    print(f"# {af.name}  Re {kw['re'] or 'inviscid'}  Mach {args.mach:g}")
    print(f"# {'alpha':>7} {'Cl':>9} {'Cd':>9} {'Cm':>9}  status")
    for r in pol.results:
        print(f"{r.alpha:>9.2f} {r.cl:>9.4f} {_f(r.cd, '9.5f'):>9} {r.cm:>9.4f}  {_status(r)}")
    if args.output:
        _write(pol, args.output, af.name)


class _RowsPolar:
    """Database rows shaped like ``Polar.results`` for printing and export."""

    def __init__(self, rows):
        import types
        self.results = tuple(types.SimpleNamespace(
            alpha=r["alpha"], cl=r["cl"], cd=r["cd"], cm=r["cm"], reynolds=r["reynolds"] or None,
            converged=bool(r["converged"]), iterations=r["iterations"],
            warnings=tuple(w for w in (r["warnings"] or "").split("; ") if w)) for r in rows)
        self.rows_ = rows


def _write(obj, path: str, name: str) -> None:
    from . import export
    suffix = Path(path).suffix.lower()
    if isinstance(obj, _RowsPolar):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(obj.rows_[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(obj.rows_)
    elif suffix == ".json":
        export.to_json(obj, path)
    elif suffix in (".pol", ".txt"):
        if not hasattr(obj, "results"):
            raise InputError("polar text (.pol, .txt) is for the polar command")
        export.to_polar_text(obj, path, name)
    else:
        export.to_csv(obj, path)
    print(f"wrote {path}")


def cmd_coords(args) -> None:
    from . import export
    af = _airfoil(args)
    if args.panels:
        af = af.repanelled(args.panels)
    if not args.output:
        raise InputError("--output FILE is required")
    export.save_coordinates(af, args.output)
    print(f"wrote {args.output} ({af.num_panels + 1} points; source: {af.source})")


def cmd_benchmark(args) -> None:
    from .benchmark.__main__ import main as bench_main
    code = bench_main(args.bench_args)
    if code:
        raise SystemExit(code)


def cmd_info(args) -> None:
    from . import __version__
    print(f"Foil-Lab {__version__}")
    print("2D panel method for single, flapped and multi-element sections, with a coupled")
    print("viscous-inviscid solver (e^n transition); accuracy against XFOIL in VALIDATION.md.")

    def have(mod):
        try:
            __import__(mod)
            return "yes"
        except Exception:
            return "no"
    for label, mod in (("JAX", "jax"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
                       ("PySide6 (GUI)", "PySide6"), ("pyarrow (Parquet)", "pyarrow"), ("h5py (HDF5)", "h5py")):
        print(f"  {label:<18} {have(mod)}")
    import shutil
    print(f"  {'XFOIL (benchmark)':<18} {'yes' if shutil.which('xfoil') else 'no'}")
    print("Validated: Re 1e5 to 6e6, attached or mildly separated flow, Mach 0 for viscous results.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="foillab", description="Foil-Lab: 2D airfoil analysis.")
    sub = p.add_subparsers(dest="command", required=True)

    def airfoil_args(sp, flap: bool = True):
        g = sp.add_argument_group("section (one of)")
        g.add_argument("--naca", help="NACA 4- or 5-digit code, e.g. 2412")
        g.add_argument("--dat", help="coordinate file (Selig or Lednicer)")
        g.add_argument("--name", help="name in the UIUC database (downloaded on demand), e.g. e387")
        sp.add_argument("--panels", type=int, default=None, help="panels per element (default 160)")
        if flap:
            sp.add_argument("--flap", type=float, nargs=2, metavar=("HINGE_X", "DEFLECTION"),
                            help="plain flap: hinge x/c and deflection in deg (trailing edge down +)")
            sp.add_argument("--flap-y", type=float, default=None,
                            help="hinge height as a fraction of the local thickness (default 0.5)")

    def condition_args(sp):
        g = sp.add_argument_group("conditions")
        g.add_argument("--re", type=float, default=0.0, help="Reynolds number (0 or absent: inviscid)")
        g.add_argument("--ncrit", type=float, default=None, help="e^n critical amplification (default 9)")
        g.add_argument("--trip-upper", type=float, default=None, help="forced transition, upper, x/c")
        g.add_argument("--trip-lower", type=float, default=None, help="forced transition, lower, x/c")
        g.add_argument("--mach", type=float, default=0.0, help="Mach number (Kármán-Tsien; inviscid only)")
        g.add_argument("--sweep", type=float, default=0.0, help="sweep angle in deg (simple sweep theory)")
        g.add_argument("--aspect-ratio", type=float, default=None, help="finite-wing aspect ratio")
        g.add_argument("--ground-height", type=float, default=None,
                       help="height of the moment point above the ground, in chords")
        sp.add_argument("--output", default=None, help="write the result: .csv, .json (polar also .pol)")

    a = sub.add_parser("analyze", help="one angle of attack")
    airfoil_args(a)
    condition_args(a)
    a.add_argument("--alpha", type=float, required=True, help="angle of attack (deg)")
    a.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("polar", help="a range of angles")
    airfoil_args(pl)
    condition_args(pl)
    pl.add_argument("--start", type=float, default=-4.0)
    pl.add_argument("--end", type=float, default=12.0)
    pl.add_argument("--step", type=float, default=1.0)
    pl.add_argument("--db", default=None, help="SQLite polar database: reuse stored points, store new ones")
    pl.set_defaults(func=cmd_polar)

    c = sub.add_parser("coords", help="write a section's coordinate file")
    airfoil_args(c)
    c.add_argument("--output", default=None, help="output .dat file")
    c.set_defaults(func=cmd_coords)

    b = sub.add_parser("benchmark", help="run or report the XFOIL benchmark (python -m foil_lab.benchmark)",
                       add_help=False)
    b.add_argument("bench_args", nargs=argparse.REMAINDER,
                   help="arguments of the benchmark harness, e.g. 'viscous-report' or '--help'")
    b.set_defaults(func=cmd_benchmark)

    i = sub.add_parser("info", help="version, optional components, validated range")
    i.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> None:
    """Entry point of the ``foillab`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "re", 0.0) and args.re < 0:
        parser.error("--re must be 0 (inviscid) or a positive Reynolds number")
    if getattr(args, "panels", None) is not None and args.panels < 20:
        parser.error("--panels must be at least 20")
    try:
        args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
    except Exception as exc:                    # every other failure: one line, no traceback
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
