"""Command line of the benchmark harness.

    python -m foil_lab.benchmark references   # regenerate XFOIL reference polars (needs XFOIL)
    python -m foil_lab.benchmark run          # inviscid rows on the same files (v1.3.1 viscous
                                              # methods deleted in M3 C5)
    python -m foil_lab.benchmark report       # error tables, plots and report.md in the output folder
    python -m foil_lab.benchmark all          # all three, in order
    python -m foil_lab.benchmark viscous      # run the coupled viscous solver (M3, both closure sets)
    python -m foil_lab.benchmark viscous-report  # its SPEC.md 4.1 tables (m3_viscous/report.md)

Generated output goes to ``benchmark_output/`` (gitignored). Reference CSVs are written into
the package data directory so that they can be committed. No command writes the committed before
pictures (``docs/baseline_v1.3.1.md``, ``docs/benchmark_m2.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    p = argparse.ArgumentParser(prog="python -m foil_lab.benchmark", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["references", "run", "report", "all", "viscous", "viscous-report"])
    p.add_argument("--output", default="benchmark_output", help="output folder (default: %(default)s)")
    p.add_argument("--workers", type=int, default=None, help="parallel processes")
    p.add_argument("--sections", default=None,
                   help="comma-separated section names for 'references' and 'viscous' (default: all)")
    p.add_argument("--closure-sets", default=None,
                   help="comma-separated closure sets for 'viscous' (default: fidkowski2022; "
                        "drela1987 is parked and not benchmarked)")
    p.add_argument("--reynolds", default=None, help="comma-separated Reynolds numbers for 'viscous'")
    args = p.parse_args(argv)
    out = Path(args.output)
    if args.command in ("references", "all"):
        from .references import generate_references
        only = tuple(args.sections.split(",")) if args.sections else None
        generate_references(output_root=out, workers=args.workers, only=only)
    if args.command in ("run", "all"):
        from .baseline_v131 import run_baseline
        run_baseline(output_root=out, workers=args.workers)
    if args.command in ("report", "all"):
        from .report import write_report
        write_report(output_root=out)
    if args.command == "viscous":
        from .viscous_run import CLOSURE_SETS, run_matrix
        run_matrix(output_root=out, workers=args.workers,
                   closure_sets=tuple(args.closure_sets.split(",")) if args.closure_sets else CLOSURE_SETS,
                   sections=tuple(args.sections.split(",")) if args.sections else None,
                   reynolds=tuple(float(v) for v in args.reynolds.split(",")) if args.reynolds else None)
    if args.command in ("viscous", "viscous-report"):
        from .viscous_run import write_report as write_viscous_report
        write_viscous_report(output_root=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
