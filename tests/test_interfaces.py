"""M7: tables, plots, exports, import by name, polar database and command line (SPEC.md 4.8)."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import foil_lab as fl  # noqa: E402
from foil_lab import export, plot  # noqa: E402

AF = fl.Airfoil.naca("2412")
pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def pol():
    return fl.polar(AF, [-2.0, 0.0, 2.0, 4.0])


# --- tables -------------------------------------------------------------------------------
def test_polar_and_result_to_pandas(pol):
    df = pol.to_dataframe()
    assert list(df.columns) == list(fl.api.ROW_COLUMNS) and len(df) == 4
    np.testing.assert_array_equal(df["cl"].to_numpy(), pol.cl)
    r = pol.results[1]
    assert r.to_dataframe().iloc[0]["cl"] == r.cl
    s = r.surface_dataframe()
    assert len(s) == r.elements[0].num_panels and set(s["surface"]) == {"upper", "lower"}
    np.testing.assert_array_equal(s["cp"].to_numpy(), r.elements[0].cp)
    assert r.elements_dataframe().iloc[0]["name"] == "NACA 2412"


def test_flap_polar_set_to_pandas():
    fps = fl.FlapPolarSet.from_tables([0.0, 5.0], [0.0, 2.0], cl=[[0.1, 0.3], [0.5, 0.7]],
                                      cd=np.full((2, 2), np.nan), cm=np.zeros((2, 2)), ch=np.zeros((2, 2)))
    df = fps.to_dataframe()
    assert len(df) == 4 and df.loc[(df.deflection == 5.0) & (df.alpha == 2.0), "cl"].item() == 0.7
    assert len(fps.at(2.5).to_dataframe()) == 2


# --- plots --------------------------------------------------------------------------------
def test_plot_helpers_return_figure_and_axes_and_never_show(monkeypatch, pol):
    import matplotlib.pyplot as plt

    def no_show(*a, **k):
        raise AssertionError("plot helpers must not call show()")
    monkeypatch.setattr(plt, "show", no_show)
    from foil_lab.benchmark.references import load_reference
    ref = load_reference("NACA2412", 1e6, 160)
    for fig, ax in (plot.geometry(AF.with_flap(0.75, 15.0), panels=True),
                    plot.cp([pol.results[0], pol.results[3]]),
                    plot.geometry(fl.Configuration((fl.Element(AF), fl.Element(AF, offset=(1.1, -0.1), scale=0.3))),
                                  ground=-0.5)):
        assert isinstance(fig, matplotlib.figure.Figure) and ax.figure is fig
        plt.close(fig)
    fig, axes = plot.polars([pol, pol], labels=["a", "b"], reference=ref)
    assert len(axes) == 3 and len(axes[0].lines) == 3          # two polars and the reference
    assert len(axes[1].lines) == 1                              # inviscid: no Cd, only the reference
    plt.close(fig)
    fig, ax = plt.subplots()
    out_fig, out_ax = plot.cp(pol.results[0], ax=ax)            # draws into a given axes
    assert out_fig is fig and out_ax is ax
    plt.close(fig)


# --- exports ------------------------------------------------------------------------------
def test_csv_json_and_polar_text_exports(tmp_path, pol):
    from foil_lab.benchmark.xfoil import parse_polar_text
    c = export.to_csv(pol, tmp_path / "p.csv")
    df = pd.read_csv(c)
    assert len(df) == 4 and np.allclose(df["cl"], pol.cl)
    j = json.loads(export.to_json(pol.results[0], tmp_path / "r.json").read_text())
    assert j["foil_lab"] == fl.__version__ and j["rows"][0]["cl"] == pol.results[0].cl
    assert j["rows"][0]["cd"] is None and j["elements"][0]["name"] == "NACA 2412"
    t = export.to_polar_text(pol, tmp_path / "p.pol", "NACA2412")
    header, rows = parse_polar_text(t.read_text())          # readable by an XFOIL polar parser
    assert [r.alpha for r in rows] == [-2.0, 0.0, 2.0, 4.0]
    assert np.allclose([r.cl for r in rows], pol.cl, atol=5e-5) and header["mach"] == "0.000"
    assert "Foil-Lab" in t.read_text().splitlines()[0] and "XFOIL Version" not in t.read_text()


def test_coordinate_export_round_trip(tmp_path):
    f = AF.with_flap(0.75, 10.0)
    p = export.save_coordinates(f, tmp_path / "f.dat")
    back = fl.Airfoil.from_file(p)
    np.testing.assert_allclose(back.coords, f.coords, atol=1e-10)
    assert p.read_text().splitlines()[0] == f.name


# --- import by name -----------------------------------------------------------------------
def test_from_name_uses_the_cache_and_reads_lednicer(tmp_path, monkeypatch):
    monkeypatch.setenv("FOIL_LAB_CACHE", str(tmp_path))
    c = np.asarray(fl.Airfoil.naca("0012", num_panels=40).coords)
    up, lo = c[:21][::-1], c[20:]
    text = "TEST FOIL\n21. 21.\n\n" + "".join(f"{x} {y}\n" for x, y in up) + "\n" + \
           "".join(f"{x} {y}\n" for x, y in lo)
    (tmp_path / "airfoils").mkdir()
    (tmp_path / "airfoils" / "testfoil.dat").write_text(text)

    def no_network(*a, **k):
        raise AssertionError("must read the cache")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    af = fl.Airfoil.from_name("TestFoil")
    assert af.name == "TEST FOIL" and "sha256" in af.source and "m-selig" in af.source
    np.testing.assert_allclose(af.coords, c, atol=1e-12)
    assert fl.Airfoil.from_name("NACA 23012").name == "NACA 23012"
    with pytest.raises(ValueError):
        fl.Airfoil.from_name("../etc/passwd")


def test_from_name_unavailable(tmp_path, monkeypatch):
    from foil_lab.catalog import AirfoilUnavailable
    import urllib.error
    monkeypatch.setenv("FOIL_LAB_CACHE", str(tmp_path))

    def offline(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr("urllib.request.urlopen", offline)
    with pytest.raises(AirfoilUnavailable):
        fl.Airfoil.from_name("e387")


# --- polar database -----------------------------------------------------------------------
def test_geometry_hash():
    h = fl.geometry_hash(AF)
    assert h == fl.geometry_hash(fl.Airfoil.naca("2412")) and len(h) == 64
    assert h != fl.geometry_hash(AF.translated(0.0, 1e-6))
    assert h != fl.geometry_hash(AF.with_flap(0.75, 5.0))
    assert h != fl.geometry_hash(fl.Configuration((fl.Element(AF),), ref_point=(0.3, 0.0)))


def test_database_stores_reuses_and_filters(tmp_path, monkeypatch):
    from foil_lab import api
    db = fl.PolarDatabase(tmp_path / "p.sqlite")
    calls = []
    real = api.polar

    def counting(g, alphas, **kw):
        calls.append(list(alphas))
        return real(g, alphas, **kw)
    monkeypatch.setattr(api, "polar", counting)
    rows = db.polar(AF, [0.0, 2.0], mach=0.3)
    assert calls == [[0.0, 2.0]] and all(r["converged"] == 1 for r in rows)
    rows2 = db.polar(AF, [2.0, 4.0], mach=0.3)                 # 2 deg reused, 4 deg computed
    assert calls[-1] == [4.0] and rows2[0] == rows[1]
    assert rows2[0]["version"] == fl.__version__ and rows2[0]["geometry_hash"] == fl.geometry_hash(AF)
    assert len(db.query(geometry=AF, mach=0.3)) == 3 and db.query(geometry=AF, mach=0.0) == []
    db.polar(AF, [0.0])                                          # other conditions: a new row
    assert len(db.query(geometry=AF)) == 4
    df = db.to_dataframe(name="NACA 2412")
    assert len(df) == 4 and set(df["method"]) == {fl.api.METHOD}
    db.close()
    with fl.PolarDatabase(tmp_path / "p.sqlite") as db2:        # persisted
        assert len(db2.query()) == 4


# --- command line -------------------------------------------------------------------------
def test_cli_outputs(tmp_path, capsys):
    from foil_lab.cli import main
    main(["analyze", "--naca", "2412", "--alpha", "3", "--mach", "0.4", "--output", str(tmp_path / "a.json")])
    out = capsys.readouterr().out
    assert "critical Mach" in out and "Kármán-Tsien" in out
    assert json.loads((tmp_path / "a.json").read_text())["rows"][0]["mach"] == 0.4
    main(["polar", "--naca", "0012", "--start", "0", "--end", "4", "--step", "2",
          "--output", str(tmp_path / "p.pol")])
    assert (tmp_path / "p.pol").read_text().count("\n") >= 13
    main(["polar", "--naca", "0012", "--start", "0", "--end", "2", "--db", str(tmp_path / "d.sqlite"),
          "--output", str(tmp_path / "d.csv")])
    assert len(pd.read_csv(tmp_path / "d.csv")) == 3
    main(["coords", "--naca", "4412", "--panels", "80", "--output", str(tmp_path / "c.dat")])
    assert len(fl.Airfoil.from_file(tmp_path / "c.dat").coords) == 81
    main(["info"])
    assert "Validated" in capsys.readouterr().out


def test_cli_benchmark_command_passes_arguments(monkeypatch):
    from foil_lab.benchmark import __main__ as bm
    from foil_lab.cli import main
    seen = []
    monkeypatch.setattr(bm, "main", lambda argv: seen.append(argv) or 0)
    main(["benchmark", "viscous-report", "--output", "x"])
    assert seen == [["viscous-report", "--output", "x"]]
