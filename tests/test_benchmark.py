"""Tests for the benchmark harness: sections, committed references, attached-flow filter.

These run without XFOIL. Tests that need the downloaded sections (Clark Y, E387) skip when they
are neither cached nor downloadable.
"""

import numpy as np
import pytest

from foil_lab.benchmark import sections
from foil_lab.benchmark.references import load_reference, reference_name
from foil_lab.benchmark.report import attached_filter
from foil_lab.benchmark.xfoil import PolarRow, XfoilPolar
from foil_lab.generators import generate_naca4
from foil_lab.params import BENCHMARK, SWEEP


def _as_selig_text(c):
    return "TEST\n" + "".join(f" {x:.7f} {y:.7f}\n" for x, y in c)


def _as_lednicer_text(c):
    le = int(np.argmin(c[:, 0]))
    up, lo = c[:le + 1][::-1], c[le:]
    body = "".join(f" {x:.7f} {y:.7f}\n" for x, y in up) + "\n" + \
        "".join(f" {x:.7f} {y:.7f}\n" for x, y in lo)
    return f"TEST\r\n {len(up)}.0 {len(lo)}.0\r\n\r\n" + body.replace("\n", "\r\n")


def test_parse_selig_and_lednicer_give_the_same_points():
    c = generate_naca4("2412", 60)
    s, ds = sections.parse_coordinates(_as_selig_text(c))
    l, dl = sections.parse_coordinates(_as_lednicer_text(c))
    assert ds == dl == 0
    assert s.shape == l.shape == c.shape
    assert np.allclose(s, c, atol=1e-7) and np.allclose(l, c, atol=1e-7)


def test_parse_removes_consecutive_duplicates():
    c = generate_naca4("0012", 40)
    text = _as_selig_text(np.insert(c, 7, c[7], axis=0))
    out, removed = sections.parse_coordinates(text)
    assert removed == 1 and out.shape == c.shape


def test_downloaded_section_skips_cleanly_without_cache_or_network(tmp_path, monkeypatch):
    monkeypatch.setenv("FOIL_LAB_CACHE", str(tmp_path))
    spec = sections.SectionSpec("NOPE", "uiuc", url="http://127.0.0.1:9/none.dat", sha256="0" * 64)
    with pytest.raises(sections.SectionUnavailable):
        sections.fetch(spec, timeout_s=2.0)


@pytest.mark.parametrize("name", ["CLARKY", "E387"])
def test_downloaded_sections_canonical(name):
    try:
        sec = sections.canonical(name)
    except sections.SectionUnavailable as exc:
        pytest.skip(str(exc))
    assert sec.coords.shape == (BENCHMARK.num_panels + 1, 2)
    x, y = sec.coords[:, 0], sec.coords[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0      # Selig order
    assert np.all(np.hypot(*np.diff(sec.coords, axis=0).T) > 0)          # no duplicates


def _all_cases():
    return [(s, re, n) for s in BENCHMARK.sections
            for re in BENCHMARK.reynolds_low + BENCHMARK.reynolds_validated
            for n in BENCHMARK.pane_nodes]


@pytest.mark.parametrize("section,re,n", _all_cases(),
                         ids=[reference_name(*c)[:-4] for c in _all_cases()])
def test_committed_reference_complete_with_provenance(section, re, n):
    ref = load_reference(section, re, n)
    p = ref.provenance
    for key in ("xfoil_version", "xfoil_package", "xfoil_command", "geometry_source",
                "geometry_sha256", "trailing_edge_gap", "pane_nodes", "re", "mach", "n_crit",
                "max_iter", "date_utc", "runs"):
        assert p.get(key), key
    assert p["xfoil_version"] == "6.99" and p["pane_nodes"] == str(n) and p["n_crit"] == "9"
    assert float(p["re"]) == re and p["mach"] == "0"
    expected = [round(SWEEP.alpha_min + i * SWEEP.alpha_step, 6)
                for i in range(int(round((SWEEP.alpha_max - SWEEP.alpha_min) / SWEEP.alpha_step)) + 1)]
    assert ref.alphas == expected
    assert ref.retried <= set(ref.converged_alphas)


@pytest.mark.parametrize("section", BENCHMARK.sections)
def test_canonical_file_matches_reference_checksum(section, tmp_path):
    """The canonical file produced by the current code must be the one every committed
    reference of that section was computed from (SPEC.md 4.1). If this fails, regenerate the
    references for the section: python -m foil_lab.benchmark references."""
    from foil_lab.benchmark.xfoil import write_coordinate_file
    try:
        sec = sections.canonical(section)
    except sections.SectionUnavailable as exc:
        pytest.skip(str(exc))
    sha = write_coordinate_file(tmp_path / f"{section}.dat", sec.coords, section)
    stale = [reference_name(section, re, n)
             for re in BENCHMARK.reynolds_low + BENCHMARK.reynolds_validated
             for n in BENCHMARK.pane_nodes
             if load_reference(section, re, n).provenance["geometry_sha256"] != sha]
    assert not stale, f"canonical file changed; regenerate: {stale}"


def _polar(cls):
    rows = {a: (None if cl is None else PolarRow(a, cl, 0.01, 0.0, 0.0, 1.0, 1.0)) for a, cl in cls.items()}
    return XfoilPolar(rows=rows, provenance={})


def test_attached_filter_stops_before_ninety_percent_of_max_lift():
    cls = {a: 0.1 * a for a in range(-4, 13)}
    cls.update({13: 1.25, 14: 1.3, 15: 1.2, 2: None})
    filt, cut = attached_filter(_polar({float(a): v for a, v in cls.items()}))
    # max 1.3 at 14 deg; 0.9 * 1.3 = 1.17 is first reached at 12 deg (Cl 1.2)
    assert cut == 12.0
    assert filt == [float(a) for a in range(-4, 12)]      # includes the unconverged 2 deg


def test_attached_filter_empty_without_positive_points():
    filt, cut = attached_filter(_polar({-1.0: -0.1, 0.0: None, 1.0: None}))
    assert filt == [] and cut is None


def test_report_command_never_writes_the_baseline_document(tmp_path, monkeypatch):
    """'report' writes only into the output folder; docs/baseline_v1.3.1.md is left alone."""
    from foil_lab.benchmark.__main__ import main
    monkeypatch.chdir(tmp_path)
    assert main(["report", "--output", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "docs").exists()
