"""Tests for the XFOIL runner (``foil_lab.benchmark.xfoil``).

Fast tests use a short sample of XFOIL's polar-file format and stand-in commands, so they run
without XFOIL. Tests marked ``xfoil`` run the real program and are skipped when it is missing.
"""

import dataclasses

import numpy as np
import pytest

from foil_lab.benchmark.xfoil import (
    PolarRow, XfoilPolar, build_script, parse_polar_text, polar_to_csv_text,
    read_reference_csv, run_polar, run_xfoil, write_coordinate_file, is_nan_free,
)
from foil_lab.generators import generate_naca4
from foil_lab.params import SWEEP, XFOIL, SweepParams
from foil_lab.utils.io import load_dat

# Format sample of an XFOIL 6.99 polar save file (header layout and three data rows).
SAMPLE_POLAR = """\

       XFOIL         Version 6.99

 Calculated polar for: NACA2412

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000  9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr  Top_Itr  Bot_Itr
  ------ -------- --------- --------- -------- -------- -------- -------- --------
   0.000   0.2249   0.00556   0.00104  -0.0479   0.6493   0.6758  33.4176 130.0546
   1.000   0.3171   0.00557   0.00120  -0.0484   0.5910   0.8434  37.1220 139.5262
  -3.000  -0.1189   0.00688   0.00109  -0.0457   0.8380   0.1903  21.6420 112.3911
"""


def test_parse_polar_header_and_rows():
    header, rows = parse_polar_text(SAMPLE_POLAR)
    assert header == {"version": "6.99", "mach": "0.000", "re": "1e+06", "ncrit": "9.000",
                      "xtrf_top": "1.000", "xtrf_bot": "1.000"}
    assert [r.alpha for r in rows] == [0.0, 1.0, -3.0]
    assert rows[1] == PolarRow(1.0, 0.3171, 0.00557, 0.00120, -0.0484, 0.5910, 0.8434)


def test_parse_polar_without_header_raises():
    with pytest.raises(Exception, match="column header"):
        parse_polar_text("not a polar file\n")


def test_script_viscous_and_inviscid():
    s = build_script("a.dat", "a.pol", 1e6, 0.0, 20.0, 1.0)
    lines = s.splitlines()
    assert lines[:6] == ["LOAD a.dat", "PPAR", "N 160", "", "", "PANE"]
    assert "VISC 1e+06" in lines and "N 9" in lines and "XTR 1 1" in lines
    assert "ASEQ 0 20 1" in lines and lines[-1] == "QUIT"
    inv = build_script("a.dat", "a.pol", 0.0, 0.0, -4.0, -1.0,
                       dataclasses.replace(XFOIL, pane_nodes=None))
    assert "VISC" not in inv and "PANE" not in inv and "ASEQ 0 -4 -1" in inv
    single = build_script("a.dat", "a.pol", 1e6, 2.0).splitlines()
    assert "ALFA 2" in single and not any(s.startswith("ASEQ") for s in single)
    with pytest.raises(ValueError):
        build_script("a.dat", "a.pol", 1e6, 0.0, alpha_end=5.0)


def test_coordinate_file_roundtrip(tmp_path):
    af = generate_naca4("2412", 160)
    sha = write_coordinate_file(tmp_path / "n.dat", af, "NACA2412")
    assert len(sha) == 64
    back = load_dat(tmp_path / "n.dat")
    assert back.shape == af.shape
    assert np.max(np.abs(back - af)) < 1e-10


def test_coordinate_file_rejects_bad_input(tmp_path):
    af = generate_naca4("2412", 60)
    dup = np.insert(af, 10, af[10], axis=0)
    with pytest.raises(ValueError, match="duplicate"):
        write_coordinate_file(tmp_path / "d.dat", dup, "dup")
    with pytest.raises(ValueError, match="Selig order"):
        write_coordinate_file(tmp_path / "r.dat", af[::-1], "rev")
    with pytest.raises(ValueError, match="whitespace"):
        write_coordinate_file(tmp_path / "s.dat", af, "NACA 2412")


def test_csv_roundtrip_keeps_missing_angles():
    rows = {-1.0: PolarRow(-1.0, -0.1, 0.006, 0.001, -0.05, 0.9, 0.2), 0.0: None,
            1.0: PolarRow(1.0, 0.3171, 0.00557, 0.0012, -0.0484, 0.591, 0.8434)}
    pol = XfoilPolar(rows=rows, provenance={"reference": "XFOIL polar", "re": "1e+06"},
                     retried={1.0})
    back = read_reference_csv(polar_to_csv_text(pol))
    assert back.provenance == pol.provenance
    assert back.alphas == [-1.0, 0.0, 1.0]
    assert back.rows[0.0] is None
    assert back.rows[1.0] == rows[1.0]
    assert back.retried == {1.0}
    assert is_nan_free(back)


def test_crash_is_recorded_not_raised(tmp_path):
    params = dataclasses.replace(XFOIL, command=("false",))
    run = run_xfoil("QUIT\n", tmp_path, "x.pol", params)
    assert run.returncode == 1 and not run.timed_out and run.polar_text == ""


def test_timeout_is_recorded_not_raised(tmp_path):
    params = dataclasses.replace(XFOIL, command=("sleep", "5"), timeout_s=0.3)
    run = run_xfoil("", tmp_path, "x.pol", params)
    assert run.timed_out and run.returncode is None


def test_timeout_kills_grandchildren(tmp_path):
    """Like xvfb-run, the wrapper starts a child; a timeout must not leave it running."""
    import subprocess
    marker = "37.123"
    params = dataclasses.replace(XFOIL, command=("sh", "-c", f"sleep {marker} & wait"),
                                 timeout_s=0.5)
    run = run_xfoil("", tmp_path, "x.pol", params)
    assert run.timed_out
    import time
    deadline = time.monotonic() + 3.0      # SIGKILL is asynchronous; allow the kernel to finish
    while True:
        left = subprocess.run(["pgrep", "-f", f"^sleep {marker}$"], capture_output=True, text=True)
        if left.stdout.strip() == "" or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert left.stdout.strip() == ""


def test_failed_runs_give_all_angles_unconverged():
    params = dataclasses.replace(XFOIL, command=("false",))
    pol = run_polar(generate_naca4("0012", 60), "N0012", 1e6, "test",
                    sweep=SweepParams(-2.0, 2.0, 1.0), params=params)
    assert pol.alphas == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert pol.converged_alphas == [] and pol.retried == set()
    # two sweeps plus one retry per missing angle, each recorded
    assert len(pol.runs) == 2 + 5
    assert pol.provenance["runs"].count("exit 1") == 7
    assert "retry 2: exit 1" in pol.provenance["runs"]
    assert pol.provenance["trailing_edge_gap"] == "0.000000"


@pytest.mark.slow
@pytest.mark.xfoil
def test_real_xfoil_symmetric_section_and_determinism():
    """NACA 0012 is symmetric: Cl(-a) = -Cl(a), Cl(0) = 0. Two runs give identical data."""
    sweep = SweepParams(-3.0, 3.0, 1.0)
    af = generate_naca4("0012", 160)
    p1 = run_polar(af, "NACA0012", 1e6, "generate_naca4('0012', 160)", sweep=sweep)
    p2 = run_polar(af, "NACA0012", 1e6, "generate_naca4('0012', 160)", sweep=sweep)
    assert p1.converged_alphas == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert abs(p1.rows[0.0].cl) < 1e-3
    for a in (1.0, 2.0, 3.0):
        assert abs(p1.rows[a].cl + p1.rows[-a].cl) < 2e-3
        assert abs(p1.rows[a].cd - p1.rows[-a].cd) < 2e-5
    assert p1.rows == p2.rows
    assert p1.provenance["xfoil_version"] == "6.99"
    assert p1.provenance["xfoil_package"].startswith("xfoil-6.99")
