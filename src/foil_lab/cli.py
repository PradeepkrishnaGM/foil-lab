"""Command-line interface for Foil Lab.

Examples
--------
    foillab analyze --naca 2412 --alpha 5 --re 1e6
    foillab polar --naca 0012 --start -4 --end 12 --step 1 --re 3e6
    foillab polar --naca 2412 --start 0 --end 8 --re 3e6 --trip-upper 0.05 --ncrit 7
    foillab info

Without ``--re`` (or with ``--re 0``) the result is inviscid. With a Reynolds number the coupled
viscous-inviscid solver runs (:mod:`foil_lab.viscous`; one body, Mach 0): transition by the e^n
envelope method (``--ncrit``) with optional forced trips per surface (``--trip-upper``,
``--trip-lower``, x/c). Every viscous line says whether the solver converged; a polar reaches
each angle from a converged neighbour (``solver.solve_flow_sweep``).
"""

from __future__ import annotations

import argparse
import sys


def _load_airfoil(args):
    """Build coordinates from --naca or --dat."""
    if args.naca:
        from .generators import generate_naca4, generate_naca5
        code = args.naca
        if len(code) == 5:
            return generate_naca5(code, args.panels), f"NACA {code}"
        return generate_naca4(code, args.panels), f"NACA {code}"
    if args.dat:
        from .utils.io import load_dat
        coords = load_dat(args.dat)
        return coords, args.dat
    raise SystemExit("error: provide --naca CODE or --dat FILE")


def _status(sol) -> str:
    """'inviscid', or the viscous convergence status with iterations and warnings."""
    if sol.reynolds <= 0:
        return "inviscid"
    tag = f"converged ({sol.iterations} it)" if sol.converged else "NOT converged"
    return tag + ("; " + "; ".join(sol.warnings) if sol.warnings else "")


def _viscous_kw(args) -> dict:
    return dict(trip_x_upper=args.trip_upper, trip_x_lower=args.trip_lower, n_crit=args.ncrit)


def cmd_analyze(args):
    from .solver import solve_flow
    coords, name = _load_airfoil(args)
    kw = _viscous_kw(args) if args.re > 0 else {}
    s = solve_flow(coords, args.alpha, reynolds=args.re, **kw)
    print(f"{name}  alpha={args.alpha:.2f} deg  Re={args.re:g}  [{_status(s)}]")
    print(f"  Cl  = {s.cl:+.4f}")
    if args.re > 0:
        print(f"  Cd  = {s.cd_total:.5f}  (friction {s.cd_friction:.5f}, pressure {s.cd_pressure:.5f})")
        print(f"  x_tr upper / lower = {s.xtr_upper:.3f} / {s.xtr_lower:.3f}")
    else:
        print("  Cd  = 0  (inviscid: no drag)")
    print(f"  Cm  = {s.cm_qc:+.4f}")
    if args.re > 0 and s.converged and s.cd_total > 1e-6:
        print(f"  L/D = {s.cl / s.cd_total:.1f}")


def cmd_polar(args):
    import numpy as np
    from .solver import solve_flow, solve_flow_sweep
    coords, name = _load_airfoil(args)
    alphas = [float(a) for a in np.arange(args.start, args.end + 1e-9, args.step)]
    if args.re > 0:
        sols = solve_flow_sweep(coords, alphas, args.re, **_viscous_kw(args))
    else:
        sols = [solve_flow(coords, a) for a in alphas]
    print(f"# {name}  Re={args.re:g}  " + ("viscous" if args.re > 0 else "inviscid"))
    print(f"# {'alpha':>7} {'Cl':>9} {'Cd':>9} {'Cm':>9}  status")
    for a, s in zip(alphas, sols):
        print(f"{a:>9.2f} {s.cl:>9.4f} {s.cd_total:>9.5f} {s.cm_qc:>9.4f}  {_status(s)}")


def cmd_info(args):
    from . import __version__
    print(f"Foil Lab v{__version__}")
    print("2D airfoil panel method with viscous-inviscid coupling.")
    # Report which optional backends are available.
    try:
        from .jax_backend import HAS_JAX
    except Exception:
        HAS_JAX = False
    def have(mod):
        try:
            __import__(mod); return "yes"
        except Exception:
            return "no"
    print(f"  JAX backend:   {'yes' if HAS_JAX else 'no'}")
    print(f"  matplotlib:    {have('matplotlib')}")
    print(f"  PySide6 (GUI): {have('PySide6')}")
    print(f"  pyarrow:       {have('pyarrow')}")
    print("\nMethods: inviscid (no --re); coupled viscous-inviscid with e^n transition (--re, "
          "one body, Mach 0; validated for Re 5e5 to 6e6).")


def build_parser():
    p = argparse.ArgumentParser(
        prog="foillab",
        description="2D airfoil analysis (panel method + viscous coupling).")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        g = sp.add_argument_group("airfoil")
        g.add_argument("--naca", help="NACA 4- or 5-digit code, e.g. 2412")
        g.add_argument("--dat", help="path to a Selig .dat coordinate file")
        g.add_argument("--panels", type=int, default=160, help="panel count")
        g.add_argument("--re", type=float, default=0.0,
                       help="Reynolds number (0 = inviscid)")
        g.add_argument("--ncrit", type=float, default=None,
                       help="e^n critical amplification factor (default: params.VISCOUS.n_crit)")
        g.add_argument("--trip-upper", type=float, default=None,
                       help="forced transition on the upper surface, x/c (default: free)")
        g.add_argument("--trip-lower", type=float, default=None,
                       help="forced transition on the lower surface, x/c (default: free)")

    a = sub.add_parser("analyze", help="solve a single angle of attack")
    add_common(a)
    a.add_argument("--alpha", type=float, default=5.0, help="angle of attack (deg)")
    a.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("polar", help="sweep a range of angles")
    add_common(pl)
    pl.add_argument("--start", type=float, default=-5.0)
    pl.add_argument("--end", type=float, default=12.0)
    pl.add_argument("--step", type=float, default=1.0)
    pl.set_defaults(func=cmd_polar)

    inf = sub.add_parser("info", help="show version and available backends")
    inf.set_defaults(func=cmd_info)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "re", 0.0) < 0:
        parser.error("--re must be 0 (inviscid) or a positive Reynolds number")
    try:
        args.func(args)
    except (NotImplementedError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
