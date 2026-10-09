"""Command line of the benchmark harness.

    python -m foil_lab.benchmark references   # regenerate XFOIL reference polars (needs XFOIL)
    python -m foil_lab.benchmark run          # run the v1.3.1 baseline on the same files
    python -m foil_lab.benchmark report       # error tables, plots and report.md in the output folder
    python -m foil_lab.benchmark all          # all three, in order
    python -m foil_lab.benchmark baseline-doc # rewrite the committed docs/baseline_v1.3.1.md

Generated output goes to ``benchmark_output/`` (gitignored). Reference CSVs are written into
the package data directory so that they can be committed. Only ``baseline-doc`` writes the
committed baseline document; ``report`` and ``all`` never do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    p = argparse.ArgumentParser(prog="python -m foil_lab.benchmark", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["references", "run", "report", "all", "baseline-doc"])
    p.add_argument("--output", default="benchmark_output", help="output folder (default: %(default)s)")
    p.add_argument("--workers", type=int, default=None, help="parallel processes")
    p.add_argument("--sections", default=None,
                   help="comma-separated section names for 'references' (default: all)")
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
    if args.command == "baseline-doc":
        from .report import BASELINE_DOC, write_report
        write_report(output_root=out, docs_path=BASELINE_DOC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
