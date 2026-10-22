"""Generate and load the committed XFOIL reference polars.

Reference files live in the package (``foil_lab/data/reference/xfoil``) so that tests and the
comparison run without XFOIL installed. Regenerating them needs ``xfoil`` and ``xvfb-run``.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import resources
from pathlib import Path

import numpy as np

from ..params import BENCHMARK, SWEEP, XFOIL, BenchmarkParams
from .sections import CanonicalSection, SectionUnavailable, canonical
from .xfoil import XfoilPolar, polar_to_csv_text, read_reference_csv, run_polar, write_coordinate_file

#: Source-tree location of the reference files (used when regenerating; editable installs).
SOURCE_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference" / "xfoil"


def reference_name(section: str, reynolds: float, pane_nodes: int) -> str:
    """File name of one reference polar, e.g. ``NACA2412_Re1000000_N160.csv``."""
    return f"{section}_Re{int(round(reynolds))}_N{pane_nodes}.csv"


def load_reference(section: str, reynolds: float, pane_nodes: int) -> XfoilPolar:
    """Load a committed reference polar from package data."""
    path = resources.files("foil_lab") / "data" / "reference" / "xfoil" / \
        reference_name(section, reynolds, pane_nodes)
    return read_reference_csv(path.read_text())


def write_canonical_file(sec: CanonicalSection, out_dir: Path) -> Path:
    """Write the canonical coordinate file both codes read; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sec.name}.dat"
    write_coordinate_file(path, sec.coords, sec.name)
    return path


def _one(sec: CanonicalSection, reynolds: float, pane_nodes: int, out_dir: str,
         log_dir: str) -> tuple[str, str, float]:
    """Worker: run XFOIL for one case and write its reference CSV. Returns (file, status, s)."""
    import dataclasses

    t0 = time.perf_counter()
    params = dataclasses.replace(XFOIL, pane_nodes=pane_nodes)
    source = sec.source
    if sec.repanel_deviation is not None:
        source += f"; max distance of repanelled points to the original polyline {sec.repanel_deviation:.2e}"
    pol = run_polar(sec.coords, sec.name, reynolds, source, sweep=SWEEP, params=params,
                    log_dir=log_dir, hinge=sec.hinge)
    name = reference_name(sec.name, reynolds, pane_nodes)
    Path(out_dir, name).write_text(polar_to_csv_text(pol))
    return name, pol.provenance["converged_points"], time.perf_counter() - t0


def generate_references(bench: BenchmarkParams = BENCHMARK,
                        out_dir: Path = SOURCE_REFERENCE_DIR,
                        output_root: Path = Path("benchmark_output"),
                        workers: int | None = None,
                        only: tuple[str, ...] | None = None) -> list[str]:
    """Regenerate the reference polars of the benchmark matrix (``only``: these sections).

    Writes the reference CSVs to ``out_dir``, the canonical coordinate files to
    ``output_root/coords`` and every XFOIL console log to ``output_root/xfoil_logs``.
    Sections that cannot be obtained (no cache, no network) are skipped with a message.
    Returns the names of the files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "xfoil_logs"
    sections = []
    for name in bench.sections:
        if only is not None and name not in only:
            continue
        try:
            sec = canonical(name, bench.num_panels)
        except SectionUnavailable as exc:
            print(f"skipped: {exc}")
            continue
        write_canonical_file(sec, output_root / "coords")
        sections.append(sec)
    jobs = [(sec, re, n) for sec in sections
            for re in bench.reynolds_low + bench.reynolds_validated for n in bench.pane_nodes]
    written: list[str] = []
    with ProcessPoolExecutor(max_workers=workers or bench.workers) as pool:
        futs = [pool.submit(_one, sec, re, n, str(out_dir), str(log_dir)) for sec, re, n in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, conv, secs = fut.result()
            written.append(name)
            print(f"[{i}/{len(futs)}] {name}: {conv} converged, {secs:.1f} s", flush=True)
    return sorted(written)


def load_canonical_file(section: str, output_root: Path = Path("benchmark_output")) -> np.ndarray:
    """Read back the canonical coordinate file of a section (what XFOIL also read)."""
    from ..utils.io import load_dat

    return load_dat(output_root / "coords" / f"{section}.dat")
