"""The M3 viscous benchmark (``foil_lab.benchmark.viscous_run``): comparison and SPEC.md 4.1 targets
on synthetic results built from a committed reference (no solver run)."""

import math

import pytest

from foil_lab.benchmark import viscous_run as vr
from foil_lab.benchmark.references import load_reference
from foil_lab.benchmark.report import attached_filter


def _rows(section, re, dcl=0.0, cd_fac=1.0, dcm=0.0, unconverged=()):
    ref = load_reference(section, re, 160)
    rows = []
    for a in ref.alphas:
        x = ref.rows[a]
        if x is None:
            continue
        conv = a not in unconverged
        rows.append(dict(method=vr.method_name("fidkowski2022"), alpha=a, converged="1" if conv else "0",
                         cl=x.cl + dcl, cd=x.cd * cd_fac, cd_te=x.cd, cm=x.cm + dcm, xtr_top=x.xtr_top,
                         xtr_bot=x.xtr_bot, iterations=7, iterations_total=7, continuation=0,
                         runtime_s=1.0, reason="converged" if conv else "max_iter", path="", error=""))
    return ref, rows


def test_compare_and_targets(tmp_path):
    """Offsets are recovered exactly; a non-converged point is counted, not used; the verdict
    follows the SPEC.md 4.1 thresholds."""
    ref, rows = _rows("NACA2412", 1e6, dcl=0.02, cd_fac=1.08, dcm=-0.005, unconverged=(3.0,))
    vr._write(vr.case_file(tmp_path, "NACA2412", 1e6), rows)
    c = vr.compare_case("NACA2412", 1e6, tmp_path)
    filt, _ = attached_filter(ref)
    st = c.stats[vr.method_name("fidkowski2022")]
    assert st.n_compared == sum(ref.rows[a] is not None for a in filt)
    assert st.n_unconverged == 1 and len(st.dcl) == st.n_compared - 1
    assert max(st.dcl) == pytest.approx(0.02) and min(st.dcl) == pytest.approx(0.02)
    assert max(st.dcd_rel) == pytest.approx(0.08)
    assert max(st.dcm) == pytest.approx(0.005)
    assert max(c.stats_te[vr.method_name("fidkowski2022")].dcd_rel) == pytest.approx(0.0, abs=1e-12)
    v = vr.verdict(st)
    assert v["converged"][0] == pytest.approx(1 - 1 / st.n_compared)
    assert v["converged"][1] == (1 - 1 / st.n_compared >= 0.95)       # 15 of 16: not met
    assert all(ok for k, (_, ok) in v.items() if k != "converged")
    _, rows = _rows("NACA2412", 1e6, dcl=0.04, cd_fac=1.31)
    vr._write(vr.case_file(tmp_path, "NACA2412", 1e6), rows)
    v = vr.verdict(vr.compare_case("NACA2412", 1e6, tmp_path).stats[vr.method_name("fidkowski2022")])
    assert not v["mean dCl"][1] and v["max dCl"][1] and not v["max rel dCd"][1] and v["mean dCm"][1]


def test_report_writes_tables(tmp_path):
    _, rows = _rows("NACA0012", 3e6)
    vr._write(vr.case_file(tmp_path, "NACA0012", 3e6), rows)
    cases = vr.write_report(tmp_path)
    assert [(c.section, c.reynolds) for c in cases] == [("NACA0012", 3e6)]
    text = (tmp_path / "m3_viscous" / "report.md").read_text()
    assert "fidkowski2022" in text and "Cd TE" in text and "✓" in text
    assert not any(math.isnan(r["cl"]) for r in vr.read_case(vr.case_file(tmp_path, "NACA0012", 3e6)))


def test_drela1987_parked(tmp_path):
    """"drela1987" is parked: not run by the benchmark and its old rows are not reported; the set
    itself stays selectable (DECISIONS.md M3, report 6 decisions)."""
    from foil_lab.bl import closure_sets
    assert vr.CLOSURE_SETS == ("fidkowski2022",)
    assert closure_sets.get("drela1987") is not None
    with pytest.raises(ValueError, match="parked"):
        vr.run_matrix(tmp_path, closure_sets=("drela1987",))
    _, rows = _rows("NACA0012", 3e6)
    rows += [dict(r, method=vr.method_name("drela1987")) for r in rows]
    vr._write(vr.case_file(tmp_path, "NACA0012", 3e6), rows)
    c = vr.compare_case("NACA0012", 3e6, tmp_path)
    assert list(c.stats) == [vr.method_name("fidkowski2022")]


def test_reference_reliability_rule_and_restart_split(tmp_path):
    """The reliability rule drops a quantity where XFOIL at 240 and 160 nodes differs by more than
    the per-point target (E387 with the 0.25 % gap at Re 5e5, -4 deg: Cd 0.00714 against 0.01611);
    points reached by a restart are pooled separately."""
    from foil_lab.benchmark.xfoil import PolarRow
    x = PolarRow(0.0, 0.5, 0.010, 0.004, -0.05, 0.5, 0.6)
    assert vr.unreliable_quantities(x, PolarRow(0.0, 0.61, 0.010, 0.004, -0.05, 0.5, 0.6)) == {"cl"}
    assert vr.unreliable_quantities(x, PolarRow(0.0, 0.5, 0.0131, 0.004, -0.0801, 0.5, 0.6)) == {"cd", "cm"}
    assert vr.unreliable_quantities(x, None) == frozenset()
    _, rows = _rows("E387_GAP025", 5e5, cd_fac=1.05)
    for r in rows:
        if r["alpha"] == -4.0:
            r["cd"], r["path"] = 3.0 * r["cd"], "repair restart"
    vr._write(vr.case_file(tmp_path, "E387_GAP025", 5e5), rows)
    c = vr.compare_case("E387_GAP025", 5e5, tmp_path)
    m = vr.method_name("fidkowski2022")
    assert c.unreliable["cd"] >= 1
    assert max(c.variants[(m, "wake", "all", False)].dcd_rel) == pytest.approx(2.15)
    assert max(c.variants[(m, "wake", "all", True)].dcd_rel) == pytest.approx(0.05)
    assert c.variants[(m, "wake", "restart", False)].n_compared == 1
    assert c.variants[(m, "wake", "direct", False)].n_compared == \
        c.variants[(m, "wake", "all", False)].n_compared - 1
