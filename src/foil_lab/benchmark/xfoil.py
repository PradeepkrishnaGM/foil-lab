"""Run the external XFOIL program and read its polars.

XFOIL is GPL software. It is used here only as an external program (``/usr/bin/xfoil``),
driven through its text command interface in a subprocess with a timeout. Nothing from its
source is used.

Recipe (see ``DECISIONS.md``, M1 Q1-Q5):

* Both codes read the same coordinate file, written by :func:`write_coordinate_file`.
* XFOIL loads that file and redistributes it with ``PANE`` (``XfoilParams.pane_nodes``).
* Each case is run as two fresh XFOIL processes, ``0 -> alpha_max`` and ``0 -> alpha_min``.
* An angle is converged only if XFOIL wrote it to its polar file. Missing angles are recorded
  as not converged and are never interpolated.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..params import SWEEP, XFOIL, SweepParams, XfoilParams

#: Columns of a reference polar CSV, in order.
CSV_COLUMNS = ("alpha", "converged", "retried", "cl", "cd", "cdp", "cm", "xtr_top", "xtr_bot")


class XfoilError(RuntimeError):
    """XFOIL could not be started, timed out, or produced no readable polar."""


@dataclass(frozen=True)
class PolarRow:
    """One converged XFOIL operating point."""

    alpha: float
    cl: float
    cd: float
    cdp: float
    cm: float
    xtr_top: float
    xtr_bot: float


@dataclass
class XfoilPolar:
    """Result of one benchmark case: all requested angles, converged or not.

    ``rows`` maps every requested angle to its :class:`PolarRow`, or to ``None`` if XFOIL did
    not converge there. ``retried`` holds the angles that converged only on the single retry
    from a fresh start. ``provenance`` is written verbatim into the CSV header.
    """

    rows: dict[float, PolarRow | None]
    provenance: dict[str, str]
    runs: list["XfoilRun"] = field(default_factory=list)
    retried: set[float] = field(default_factory=set)

    @property
    def alphas(self) -> list[float]:
        """All requested angles, ascending."""
        return sorted(self.rows)

    @property
    def converged_alphas(self) -> list[float]:
        """Angles at which XFOIL converged, ascending."""
        return [a for a in self.alphas if self.rows[a] is not None]


@dataclass
class XfoilRun:
    """Raw record of one XFOIL process."""

    script: str
    stdout: str
    returncode: int | None
    runtime_s: float
    timed_out: bool
    polar_text: str


# --------------------------------------------------------------------------------------- #
# Coordinate files
# --------------------------------------------------------------------------------------- #
def write_coordinate_file(path: str | Path, coords: np.ndarray, name: str) -> str:
    """Write coordinates in Selig format (name line, then ``x y`` pairs) for both codes.

    Coordinates are written with 10 decimals so that the file, not the writer, defines the
    geometry. Returns the SHA-256 of the file contents.

    Raises
    ------
    ValueError
        If the array is not ``(n, 2)``, contains non-finite values, has consecutive duplicate
        points (XFOIL treats those as a corner), or is not in Selig order (trailing edge,
        upper surface, leading edge, lower surface: counter-clockwise, positive signed area).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 10:
        raise ValueError(f"expected an (n, 2) coordinate array, got shape {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates contain non-finite values")
    seg = np.hypot(*np.diff(coords, axis=0).T)
    if np.any(seg == 0.0):
        raise ValueError(f"consecutive duplicate points at indices {np.flatnonzero(seg == 0.0).tolist()}")
    x, y = coords[:, 0], coords[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area <= 0.0:
        raise ValueError("coordinates are not in Selig order (signed area must be positive)")
    if any(ch.isspace() for ch in name) or not name:
        raise ValueError("name must be a non-empty string without whitespace")
    text = name + "\n" + "".join(f"{px:.10f} {py:.10f}\n" for px, py in coords)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------------------- #
# Script and parsing
# --------------------------------------------------------------------------------------- #
def _fmt(value: float) -> str:
    """Compact, exact-enough number formatting for XFOIL commands."""
    return f"{value:.6g}"


def build_script(coord_name: str, polar_name: str, reynolds: float, alpha_start: float,
                 alpha_end: float | None = None, alpha_step: float | None = None,
                 params: XfoilParams = XFOIL) -> str:
    """Return the XFOIL command script for one fresh XFOIL process.

    ``reynolds`` of 0 gives an inviscid run. With ``alpha_end`` and ``alpha_step`` the script
    sweeps from ``alpha_start`` to ``alpha_end`` (``ASEQ``; negative step for a downward
    sweep). Without them it solves the single angle ``alpha_start`` directly (``ALFA``), which
    is how a skipped angle is retried from a fresh boundary layer.
    """
    if (alpha_end is None) != (alpha_step is None):
        raise ValueError("give both alpha_end and alpha_step, or neither")
    if alpha_end is None:
        solve = f"ALFA {_fmt(alpha_start)}"
    else:
        solve = f"ASEQ {_fmt(alpha_start)} {_fmt(alpha_end)} {_fmt(alpha_step)}"
    lines = [f"LOAD {coord_name}"]
    if params.pane_nodes is not None:
        lines += ["PPAR", f"N {params.pane_nodes}", "", "", "PANE"]
    lines += ["OPER"]
    if params.mach:
        lines += [f"MACH {_fmt(params.mach)}"]
    if reynolds > 0:
        lines += [f"VISC {_fmt(reynolds)}", "VPAR", f"N {_fmt(params.n_crit)}",
                  f"XTR {_fmt(params.xtr_top)} {_fmt(params.xtr_bottom)}", ""]
    lines += [f"ITER {params.max_iter}", "PACC", polar_name, "", solve,
              "PACC", "", "QUIT", ""]
    return "\n".join(lines)


_ROW = re.compile(r"^\s*-?\d+\.\d+(\s+-?\d+\.\d+){6}")


def parse_polar_text(text: str) -> tuple[dict[str, str], list[PolarRow]]:
    """Parse an XFOIL polar save file (``PACC`` output).

    Returns the header values found (``version``, ``re``, ``mach``, ``ncrit``, ``xtrf_top``,
    ``xtrf_bot``) and the data rows in file order.

    Raises
    ------
    XfoilError
        If the column header line is missing.
    """
    header: dict[str, str] = {}
    m = re.search(r"XFOIL\s+Version\s+(\S+)", text)
    if m:
        header["version"] = m.group(1)
    m = re.search(r"Mach\s*=\s*(\S+)\s+Re\s*=\s*(\S+)\s*e\s*(-?\d+)\s+Ncrit\s*=\s*(\S+)", text)
    if m:
        header["mach"] = m.group(1)
        header["re"] = _fmt(float(m.group(2)) * 10.0 ** int(m.group(3)))
        header["ncrit"] = m.group(4)
    m = re.search(r"xtrf\s*=\s*(\S+)\s*\(top\)\s+(\S+)\s*\(bottom\)", text)
    if m:
        header["xtrf_top"], header["xtrf_bot"] = m.group(1), m.group(2)
    if not re.search(r"alpha\s+CL\s+CD\s+CDp\s+CM\s+Top_Xtr\s+Bot_Xtr", text):
        raise XfoilError("polar file has no column header")
    rows = []
    for line in text.splitlines():
        if _ROW.match(line):
            f = [float(v) for v in line.split()[:7]]
            rows.append(PolarRow(*f))
    return header, rows


# --------------------------------------------------------------------------------------- #
# Running XFOIL
# --------------------------------------------------------------------------------------- #
def xfoil_package() -> str:
    """Installed XFOIL package release as reported by ``rpm -q xfoil``, or ``unknown``."""
    if shutil.which("rpm") is None:
        return "unknown"
    out = subprocess.run(["rpm", "-q", "xfoil"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def run_xfoil(script: str, workdir: Path, polar_name: str,
              params: XfoilParams = XFOIL) -> XfoilRun:
    """Run one XFOIL process with ``script`` on stdin inside ``workdir``.

    A crash (non-zero return code) or a timeout is recorded in the returned :class:`XfoilRun`
    rather than raised, so that whatever XFOIL already wrote to its polar file is kept. The
    process runs in its own session; on timeout the whole process group is killed, so neither
    ``xfoil`` nor the virtual X server started by ``xvfb-run`` is left running.

    Raises
    ------
    XfoilError
        If the XFOIL command (or its X wrapper) is not installed.
    """
    if shutil.which(params.command[0]) is None:
        raise XfoilError(f"command not found: {params.command[0]}")
    polar_path = workdir / polar_name
    polar_path.unlink(missing_ok=True)
    start = time.perf_counter()
    proc = subprocess.Popen(list(params.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=workdir,
                            start_new_session=True)
    try:
        stdout, _ = proc.communicate(script, timeout=params.timeout_s)
        returncode, timed_out = proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
        returncode, timed_out = None, True
    runtime = time.perf_counter() - start
    polar_text = polar_path.read_text() if polar_path.exists() else ""
    return XfoilRun(script=script, stdout=stdout, returncode=returncode, runtime_s=runtime,
                    timed_out=timed_out, polar_text=polar_text)


def _requested_alphas(start: float, end: float, step: float) -> list[float]:
    n = int(round((end - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def _write_log(log_dir: Path | None, stem: str, run: XfoilRun) -> None:
    """Keep the full record of one XFOIL process: script, console output and polar file."""
    if log_dir is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    status = "timeout" if run.timed_out else f"exit {run.returncode}"
    (log_dir / f"{stem}.log").write_text(
        f"# {status}, {run.runtime_s:.2f} s\n# ---- script ----\n{run.script}\n"
        f"# ---- console ----\n{run.stdout}\n# ---- polar file ----\n{run.polar_text}")


def run_polar(coords: np.ndarray, name: str, reynolds: float, geometry_source: str,
              sweep: SweepParams = SWEEP, params: XfoilParams = XFOIL,
              workdir: str | Path | None = None,
              log_dir: str | Path | None = None) -> XfoilPolar:
    """Compute an XFOIL reference polar for one section at one Reynolds number.

    The coordinates are written once with :func:`write_coordinate_file`; the same file is what
    Foil-Lab reads in the benchmark. XFOIL is run twice from a fresh start (0 up to
    ``sweep.alpha_max``, then 0 down to ``sweep.alpha_min``). At alpha = 0 the upward run is
    used. Angles XFOIL did not write to its polar file are recorded as not converged.

    ``geometry_source`` is a free-text description recorded in the provenance header, e.g.
    ``"foil_lab.generators.generate_naca4('2412', 160)"``. If ``log_dir`` is given, the
    script, console output and polar file of every XFOIL process are written there.
    """
    log_path = Path(log_dir) if log_dir is not None else None
    pane = "asis" if params.pane_nodes is None else f"pane{params.pane_nodes}"
    stem = f"{name}_re{_fmt(reynolds)}_{pane}"
    own_tmp = None
    if workdir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="foil-lab-xfoil-")
        workdir = own_tmp.name
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    try:
        coord_file = f"{name}.dat"
        sha = write_coordinate_file(wd / coord_file, coords, name)
        up = _requested_alphas(0.0, sweep.alpha_max, sweep.alpha_step)
        down = _requested_alphas(0.0, sweep.alpha_min, -sweep.alpha_step)
        rows: dict[float, PolarRow | None] = {a: None for a in sorted(set(up) | set(down))}
        runs: list[XfoilRun] = []
        version = ""
        for branch, (a0, a1, da) in (("up", (0.0, sweep.alpha_max, sweep.alpha_step)),
                                     ("down", (0.0, sweep.alpha_min, -sweep.alpha_step))):
            polar_name = f"{name}_{branch}.pol"
            script = build_script(coord_file, polar_name, reynolds, a0, a1, da, params)
            run = run_xfoil(script, wd, polar_name, params)
            runs.append(run)
            _write_log(log_path, f"{stem}_{branch}", run)
            if not run.polar_text:
                continue
            header, parsed = parse_polar_text(run.polar_text)
            version = version or header.get("version", "")
            wanted = set(up if branch == "up" else down)
            for r in parsed:
                a = round(r.alpha, 6)
                # The upward branch owns alpha = 0; never overwrite a converged point.
                if a in wanted and rows.get(a) is None:
                    rows[a] = r
        # One retry per skipped angle: a fresh XFOIL process solving that angle directly, so the
        # boundary layer starts from XFOIL's own initialisation (DECISIONS.md, M1 D4).
        retried: set[float] = set()
        labels = ["up", "down"]
        for a in [a for a, r in rows.items() if r is None]:
            polar_name = f"{name}_retry.pol"
            script = build_script(coord_file, polar_name, reynolds, a, params=params)
            run = run_xfoil(script, wd, polar_name, params)
            runs.append(run)
            labels.append(f"retry {_fmt(a)}")
            _write_log(log_path, f"{stem}_retry_{_fmt(a)}", run)
            if run.polar_text:
                for r in parse_polar_text(run.polar_text)[1]:
                    if round(r.alpha, 6) == a:
                        rows[a] = r
                        retried.add(a)
        m = re.search(r"XFOIL\s+Version\s+(\S+)", runs[0].stdout) if runs else None
        version = version or (m.group(1) if m else "unknown")
        status = "; ".join(
            f"{b}: " + ("timeout" if r.timed_out else f"exit {r.returncode}") + f", {r.runtime_s:.2f} s"
            for b, r in zip(labels, runs))
        c = np.asarray(coords, dtype=float)
        te_gap = float(np.hypot(*(c[0] - c[-1])))
        provenance = {
            "reference": "XFOIL polar",
            "xfoil_version": version,
            "xfoil_package": xfoil_package(),
            "xfoil_command": " ".join(params.command),
            "section": name,
            "geometry_source": geometry_source,
            "geometry_sha256": sha,
            "geometry_points": str(len(coords)),
            "trailing_edge_gap": f"{te_gap:.6f}",
            "pane_nodes": "as-is" if params.pane_nodes is None else str(params.pane_nodes),
            "re": _fmt(reynolds),
            "mach": _fmt(params.mach),
            "n_crit": _fmt(params.n_crit),
            "xtr_top": _fmt(params.xtr_top),
            "xtr_bottom": _fmt(params.xtr_bottom),
            "max_iter": str(params.max_iter),
            "sweep": (f"0 to {_fmt(sweep.alpha_max)} and 0 to {_fmt(sweep.alpha_min)}, "
                      f"step {_fmt(sweep.alpha_step)} deg, fresh XFOIL process per branch"),
            "runs": status,
            "converged_points": (f"{sum(r is not None for r in rows.values())} of {len(rows)} "
                                 f"({len(retried)} after one retry)"),
            "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return XfoilPolar(rows=rows, provenance=provenance, runs=runs, retried=retried)
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()


# --------------------------------------------------------------------------------------- #
# Reference CSV files
# --------------------------------------------------------------------------------------- #
def polar_to_csv_text(polar: XfoilPolar) -> str:
    """Serialise a polar as CSV with a ``# key: value`` provenance header.

    Every requested angle has a row; non-converged angles have ``converged = 0`` and empty
    value fields. ``retried = 1`` marks angles that converged only on the single retry.
    """
    buf = io.StringIO()
    for key, value in polar.provenance.items():
        buf.write(f"# {key}: {value}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for a in polar.alphas:
        r = polar.rows[a]
        if r is None:
            w.writerow([_fmt(a), 0, 0, "", "", "", "", "", ""])
        else:
            w.writerow([_fmt(a), 1, int(a in polar.retried), f"{r.cl:.4f}", f"{r.cd:.5f}",
                        f"{r.cdp:.5f}", f"{r.cm:.4f}", f"{r.xtr_top:.4f}", f"{r.xtr_bot:.4f}"])
    return buf.getvalue()


def read_reference_csv(text: str) -> XfoilPolar:
    """Inverse of :func:`polar_to_csv_text` (runs are not stored, so ``runs`` is empty)."""
    provenance: dict[str, str] = {}
    body = []
    for line in text.splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition(":")
            provenance[key.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    reader = csv.DictReader(body)
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        raise ValueError(f"unexpected columns {reader.fieldnames}")
    rows: dict[float, PolarRow | None] = {}
    retried: set[float] = set()
    for rec in reader:
        a = round(float(rec["alpha"]), 6)
        if rec["converged"] == "1":
            rows[a] = PolarRow(a, float(rec["cl"]), float(rec["cd"]), float(rec["cdp"]),
                               float(rec["cm"]), float(rec["xtr_top"]), float(rec["xtr_bot"]))
            if rec["retried"] == "1":
                retried.add(a)
        else:
            rows[a] = None
    return XfoilPolar(rows=rows, provenance=provenance, retried=retried)


def is_nan_free(polar: XfoilPolar) -> bool:
    """True if every converged row has finite values (a basic sanity check)."""
    return all(all(math.isfinite(v) for v in (r.cl, r.cd, r.cdp, r.cm, r.xtr_top, r.xtr_bot))
               for r in polar.rows.values() if r is not None)
