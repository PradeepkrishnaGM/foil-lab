"""Geometry model, operations, generators and the analysis API (M2 step 4).

Tolerances were fixed in DECISIONS.md (M2, "Step 4: tolerances fixed before running").
"""

import hashlib
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import foil_lab as fl
from foil_lab import inviscid as core
from foil_lab.generators import (generate_biconvex, generate_naca4, generate_naca5,
                                 generate_parsec, generate_wedge)
from foil_lab.geometry import (repanel_cosine, rotate, scale, scale_thickness_camber,
                               set_trailing_edge_gap, translate, trailing_edge_gap)

# --------------------------------------------------------------------------------------- #
# Closed NACA geometry pinned (SHA-256 recorded before the default changed, M2). The reference
# arrays in tests/data/closed_naca are checked against those hashes bit for bit; every generated
# coordinate is within 4 ulp of the unit chord (8.9e-16) of them, because numpy's sin/cos/power can
# differ in the last bit between CPU types (GitHub runners: at most 3.4e-17 chord, DECISIONS.md).
# --------------------------------------------------------------------------------------- #
_CLOSED_DIR = Path(__file__).parent / "data" / "closed_naca"
_MAX_ULP = 4
_CLOSED_SHA = {
    ("0012", 120): "41e92a82074285fba6204a3bb357d6453ddd5cc7febe1c66c5e7a8ad53f2e3a5",
    ("0012", 160): "b7c515a34c3f1a9990d0ae0d94c04e1cfac6eff9690c7afe501f4c8114a91633",
    ("2412", 120): "2a94fa33037c1b45b3ae4fd4cc749fad8dbc235e893f563dace2da71594a5afa",
    ("2412", 160): "b23fd00e23d159cbd06dc52e654d1315a948e0377a8e5486b57c730c251ee948",
    ("4412", 120): "664befa6b1832b01823d5d72ef8cebce5f0310822cd52e14e4e6426d1ec9a02f",
    ("4412", 160): "f256a1374f8f47cece083b3170c4b369b586ab388e94aeaf495aec0d5c75772a",
    ("23012", 120): "ddbf5ec2424a56b9b4fa9316a4f6e26eabf843ab74269d0febb0b3cc302cb984",
    ("23012", 160): "bd4ee7b1fa4af9b9f651c96a6ce151c290a63fa384be839977ee321c62872ee3",
}


@pytest.mark.parametrize("code,n", list(_CLOSED_SHA))
def test_closed_naca_pinned(code, n):
    ref = np.load(_CLOSED_DIR / f"{code}_{n}.npy")
    assert hashlib.sha256(ref.tobytes()).hexdigest() == _CLOSED_SHA[(code, n)]
    gen = generate_naca5 if len(code) == 5 else generate_naca4
    a = gen(code, n, trailing_edge="closed")
    assert a.shape == ref.shape and a.dtype == ref.dtype
    d = np.abs(a - ref)
    assert d.max() <= _MAX_ULP * np.spacing(1.0), f"max abs diff {d.max():.3e} chord"


def test_naca_default_is_open():
    assert trailing_edge_gap(generate_naca4("2412", 160)) > 0.002
    assert trailing_edge_gap(generate_naca5("23012", 160)) > 0.002


# --------------------------------------------------------------------------------------- #
# NACA analytic properties
# --------------------------------------------------------------------------------------- #
def _surfaces(c):
    le = int(np.argmin(c[:, 0]))
    return c[:le + 1][::-1], c[le:]        # upper and lower, both from the leading edge


def test_naca_symmetric_section_is_symmetric():
    up, lo = _surfaces(generate_naca4("0012", 4000))
    assert np.max(np.abs(up[:, 1] + lo[:, 1])) <= 1e-15
    assert np.max(np.abs(up[:, 0] - lo[:, 0])) <= 1e-15


def _paired(c, num_panels):
    """Upper and lower nodes of a NACA generator output, paired by construction station."""
    n_up = num_panels // 2 + 1
    return c[:n_up][::-1], c[n_up - 1:]


@pytest.mark.parametrize("code,t", [("0012", 0.12), ("2412", 0.12), ("4415", 0.15), ("23012", 0.12)])
def test_naca_maximum_thickness(code, t):
    """Thickness perpendicular to the mean line: the distance between the upper and lower points
    of one construction station (2 y_t). Tolerance 2e-4, symmetric and cambered sections."""
    gen = generate_naca5 if len(code) == 5 else generate_naca4
    up, lo = _paired(gen(code, 4000), 4000)
    assert abs(np.hypot(*(up - lo).T).max() - t) <= 2e-4


def test_naca_open_gap_closed_form():
    c = generate_naca4("0012", 160)
    t = 0.12
    expected = 10 * t * (0.2969 - 0.1260 - 0.3516 + 0.2843 - 0.1015)
    assert abs(trailing_edge_gap(c) - expected) <= 1e-12


@pytest.mark.parametrize("code,m,p", [("2412", 0.02, 0.4), ("4412", 0.04, 0.4), ("2312", 0.02, 0.3)])
def test_naca4_maximum_camber(code, m, p):
    up, lo = _surfaces(generate_naca4(code, 4000))
    x = np.linspace(0.005, 0.995, 3961)
    camber = 0.5 * (np.interp(x, up[:, 0], up[:, 1]) + np.interp(x, lo[:, 0], lo[:, 1]))
    k = int(np.argmax(camber))
    assert abs(camber[k] - m) <= 1e-3 and abs(x[k] - p) <= 1e-3 + 0.02


def test_naca5_230_camber_position():
    up, lo = _surfaces(generate_naca5("23012", 4000))
    x = np.linspace(0.005, 0.995, 3961)
    camber = 0.5 * (np.interp(x, up[:, 0], up[:, 1]) + np.interp(x, lo[:, 0], lo[:, 1]))
    assert abs(x[int(np.argmax(camber))] - 0.15) <= 2e-3 + 0.02


# --------------------------------------------------------------------------------------- #
# Other generators: property tests, and Airfoil constructors
# --------------------------------------------------------------------------------------- #
def test_parsec_meets_its_constraints():
    p = dict(rle=0.012, x_up=0.4, y_up=0.062, yxx_up=-0.55, x_lo=0.4, y_lo=-0.042, yxx_lo=0.55,
             y_te=0.0, dy_te=0.0, ang_te=-9.0, ang_wedge=9.0)
    up, lo = _surfaces(generate_parsec(p, 4000))
    assert abs(up[:, 1].max() - p["y_up"]) <= 1e-3
    assert abs(up[np.argmax(up[:, 1]), 0] - p["x_up"]) <= 1e-2
    assert abs(lo[:, 1].min() - p["y_lo"]) <= 1e-3
    assert abs(lo[np.argmin(lo[:, 1]), 0] - p["x_lo"]) <= 1e-2


@pytest.mark.parametrize("gen", [generate_biconvex, generate_wedge])
def test_supersonic_sections_symmetric_with_thickness(gen):
    up, lo = _surfaces(gen(0.06, 400))
    assert np.max(np.abs(up[:, 1] + np.interp(up[:, 0], lo[:, 0], lo[:, 1]))) <= 1e-12
    assert abs((up[:, 1].max() - lo[:, 1].min()) - 0.06) <= 1e-3


def test_airfoil_constructors_return_airfoils():
    for a in (fl.Airfoil.naca("2412"), fl.Airfoil.naca("23012", trailing_edge="closed"),
              fl.Airfoil.biconvex(0.06), fl.Airfoil.wedge(0.05),
              fl.Airfoil.parsec(dict(rle=0.012, x_up=0.4, y_up=0.062, yxx_up=-0.55, x_lo=0.4,
                                     y_lo=-0.042, yxx_lo=0.55, y_te=0.0, dy_te=0.0,
                                     ang_te=-9.0, ang_wedge=9.0))):
        assert isinstance(a, fl.Airfoil) and not a.coords.flags.writeable


def test_airfoil_from_file_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "naca2412.dat"
        c = generate_naca4("2412", 120)
        path.write_text("NACA 2412\n" + "\n".join(f"{x:.10f} {y:.10f}" for x, y in c))
        a = fl.Airfoil.from_file(path)
        assert a.name == "naca2412" and np.max(np.abs(a.coords - c)) <= 1e-10


# --------------------------------------------------------------------------------------- #
# Operations on scaled, translated and rotated input (1e-9 chord)
# --------------------------------------------------------------------------------------- #
_TRANSFORMS = {
    "scaled": (lambda c: scale(c, 0.5), 0.5),
    "translated": (lambda c: translate(c, 0.2, 0.1), 1.0),
    "rotated": (lambda c: rotate(c, 10.0, pivot=(0.25, 0.0)), 1.0),
}
_OPERATIONS = {
    "repanel": lambda c, chord: repanel_cosine(c, 160),
    "te_gap": lambda c, chord: set_trailing_edge_gap(c, 0.004 * chord),
    "thickness": lambda c, chord: scale_thickness_camber(c, thickness=1.2),
    "camber": lambda c, chord: scale_thickness_camber(c, camber=0.5),
}


@pytest.mark.parametrize("op", list(_OPERATIONS))
@pytest.mark.parametrize("tr", list(_TRANSFORMS))
def test_operation_commutes_with_transform(op, tr):
    base = generate_naca4("2412", 120)
    t, chord = _TRANSFORMS[tr]
    f = _OPERATIONS[op]
    assert np.max(np.abs(f(t(base), chord) - t(f(base, 1.0)))) / chord <= 1e-9


def test_scaling_by_one_is_identity():
    c = generate_naca4("2412", 120)
    assert np.max(np.abs(scale_thickness_camber(c) - c)) <= 1e-12


def test_thickness_scaling_gives_requested_thickness():
    c = generate_naca4("2412", 4000)
    up, lo = _surfaces(scale_thickness_camber(c, thickness=1.5))
    x = np.linspace(0.01, 0.99, 2001)
    thick = np.interp(x, up[:, 0], up[:, 1]) - np.interp(x, lo[:, 0], lo[:, 1])
    assert abs(thick.max() - 1.5 * 0.12) <= 2e-4


def test_symmetric_generator_output_is_symmetric_by_index():
    """Raw NACA 0012 output: upper and lower nodes paired by index, y_u = -y_l to 1e-12."""
    up, lo = _paired(generate_naca4("0012", 200), 200)
    assert np.max(np.abs(up[:, 1] + lo[:, 1])) <= 1e-12
    assert np.max(np.abs(up[:, 0] - lo[:, 0])) <= 1e-12


def test_zero_camber_gives_symmetric_section():
    """NACA 4412 with camber scaled to zero: both surfaces resampled with the arc-length cubic
    spline at common x stations of the section's own chord frame; |y_u + y_l| <= 1e-5 chord for
    x/c 0.01-0.99. (The earlier 1e-12 was attached to an interpolating check; DECISIONS.md.)"""
    from foil_lab.geometry import _ArcSpline, _other_surface_y, _to_chord_frame
    c = scale_thickness_camber(generate_naca4("4412", 200), camber=0.0)
    sp = _ArcSpline(c)
    _, frame = _to_chord_frame(c, sp)
    x = np.linspace(0.01, 0.99, 197)
    yu = np.array([_other_surface_y(sp, frame, xi, upper=False) for xi in x])
    yl = np.array([_other_surface_y(sp, frame, xi, upper=True) for xi in x])
    assert np.all(yu > 0) and np.all(yl < 0)
    assert np.max(np.abs(yu + yl)) <= 1e-5


def test_te_gap_tool():
    c = generate_naca4("2412", 160, trailing_edge="closed")
    g = set_trailing_edge_gap(c, 0.003)
    assert abs(trailing_edge_gap(g) - 0.003) <= 1e-12
    ahead = c[:, 0] < 0.8 - 1e-9
    assert np.max(np.abs(g[ahead] - c[ahead])) <= 1e-12
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        set_trailing_edge_gap(c, 0.02)
    assert any("above" in str(x.message) for x in w)


# --------------------------------------------------------------------------------------- #
# Element, Configuration, analyze, polar
# --------------------------------------------------------------------------------------- #
def test_configuration_reference_defaults_and_override():
    af = fl.Airfoil.naca("2412")
    cfg = fl.Configuration.of(af)
    chord, point = cfg.reference()
    assert abs(chord - af.chord) <= 1e-15 and np.allclose(point, af.quarter_chord, atol=1e-15)
    cfg2 = fl.Configuration((fl.Element(af), fl.Element(af, offset=(0.0, 5.0))),
                            ref_chord=2.0, ref_point=(0.5, 0.0))
    assert cfg2.reference() == (2.0, (0.5, 0.0))


def test_analyze_airfoil_equals_one_element_configuration():
    af = fl.Airfoil.naca("2412")
    a = fl.analyze(af, 4.0)
    b = fl.analyze(fl.Configuration((fl.Element(af),)), 4.0)
    assert a.cl == b.cl and a.cm == b.cm


def test_analyze_records_and_drag_none():
    r = fl.analyze(fl.Airfoil.naca("2412"), 4.0)
    assert r.cd is None and r.elements[0].cd is None
    assert r.alpha == 4.0 and r.num_panels == (160,) and r.repanelled and r.converged
    assert r.version == fl.__version__ and r.reynolds is None
    assert set(r.elements[0].surface) == {"upper", "lower"}
    r2 = fl.analyze(fl.Airfoil.naca("2412", num_panels=120), 4.0, repanel=False)
    assert r2.num_panels == (120,) and not r2.repanelled


def test_analyze_matches_core_on_same_nodes():
    af = fl.Airfoil.naca("2412", num_panels=160)
    r = fl.analyze(af, 4.0, repanel=False)
    ps = core.PanelSet.from_bodies(np.asarray(af.coords))
    sol = core.solve(ps, 4.0)
    chord, point = fl.Configuration.of(af).reference()
    assert abs(r.cl - core.pressure_loads(ps, np.asarray(sol.cp), 4.0, 0, chord, point)[0]) <= 1e-12


def test_element_loads_sum_to_totals():
    main = fl.Airfoil.naca("2412")
    flap = fl.Element(fl.Airfoil.naca("0012"), offset=(1.05, -0.05), scale=0.3, rotation_deg=-10.0)
    r = fl.analyze(fl.Configuration((fl.Element(main), flap)), 4.0)
    assert abs(sum(e.cl for e in r.elements) - r.cl) <= 1e-12
    assert abs(sum(e.cm for e in r.elements) - r.cm) <= 1e-12


def test_viscous_analysis_of_several_elements_passes_every_body(monkeypatch):
    """M5: with re, every element reaches the coupled solver as one list of bodies (solved together)."""
    from foil_lab import viscous
    seen = []

    def fake_single(coords, alpha, reynolds, params=viscous.VISCOUS, trips=None, *a, **kw):
        seen.append(coords)
        raise RuntimeError("stop")
    monkeypatch.setattr(viscous, "solve_single", fake_single)
    main = fl.Airfoil.naca("2412")
    flap = fl.Element(fl.Airfoil.naca("0012"), offset=(1.05, -0.05), scale=0.3, rotation_deg=-10.0)
    config = fl.Configuration((fl.Element(main), flap))
    with pytest.raises(RuntimeError):
        fl.analyze(config, 4.0, re=1e6)
    assert isinstance(seen[0], list) and len(seen[0]) == 2


def test_polar_matches_analyze():
    af = fl.Airfoil.naca("2412")
    pol = fl.polar(af, [0.0, 4.0, 8.0])
    assert np.array_equal(pol.alpha, [0.0, 4.0, 8.0])
    assert abs(pol.cl[1] - fl.analyze(af, 4.0).cl) <= 1e-12


# --------------------------------------------------------------------------------------- #
# Cross-check against XFOIL's own NACA command (slow, xfoil): tolerance 1e-4 chord
# --------------------------------------------------------------------------------------- #
def _distance_to_polyline(points, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.einsum("pij,ij->pi", points[:, None] - a[None], ab)
                / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return np.min(np.linalg.norm(points[:, None] - (a[None] + t[..., None] * ab[None]), axis=2),
                  axis=1)


def _xfoil_naca(code):
    from foil_lab.benchmark.xfoil import run_xfoil
    from foil_lab.params import XFOIL
    from foil_lab.utils.io import load_dat
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        run = run_xfoil(f"NACA {code}\nSAVE naca.dat\n\nQUIT\n", work, "naca.dat", params=XFOIL)
        assert run.polar_text, "XFOIL wrote no coordinate file"
        return load_dat(str(work / "naca.dat"))


@pytest.mark.slow
@pytest.mark.xfoil
def test_open_naca0012_matches_xfoil_naca_command():
    """Symmetric section (identical in both constructions): XFOIL's points within 1e-4 chord of
    our 4000-panel open NACA 0012."""
    assert np.max(_distance_to_polyline(_xfoil_naca("0012"), generate_naca4("0012", 4000))) <= 1e-4


@pytest.mark.slow
@pytest.mark.xfoil
def test_naca2412_functions_match_xfoil_vertical_construction():
    """XFOIL's NACA command adds the thickness vertically (judged from its coordinates). Our
    mean-line and thickness functions combined vertically must match its NACA 2412 within 1e-5
    chord (set after seeing 3.1e-7; DECISIONS.md). Our generator itself keeps the perpendicular
    construction."""
    from foil_lab.generators import _naca4_camber, _naca_thickness
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 2001)))
    yc, _ = _naca4_camber(x, 0.02, 0.4)
    yt = _naca_thickness(x, 0.12, "open")
    vertical = np.vstack([np.column_stack([x[::-1], (yc + yt)[::-1]]),
                          np.column_stack([x[1:], (yc - yt)[1:]])])
    assert np.max(_distance_to_polyline(_xfoil_naca("2412"), vertical)) <= 1e-5


def test_te_gap_tool_keeps_trailing_edge_midpoint():
    """Changing only the gap does not move the trailing-edge midpoint (1e-9 chord), open
    cambered and closed sections."""
    for c in (generate_naca4("4412", 200), generate_naca4("2412", 160, trailing_edge="closed")):
        g = set_trailing_edge_gap(c, 0.004)
        assert abs(trailing_edge_gap(g) - 0.004) <= 1e-12
        assert np.hypot(*(0.5 * (g[0] + g[-1]) - 0.5 * (c[0] + c[-1]))) <= 1e-9


@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_thickness_scaling_keeps_trailing_edge_midpoint(factor):
    """Thickness scaling on an open cambered section: the trailing-edge midpoint stays on the
    original mean line (does not move), 1e-9 chord. (Gap proportionality: see DECISIONS.md, M2
    step 4, pending.)"""
    c = generate_naca4("4412", 200)
    g = scale_thickness_camber(c, thickness=factor)
    assert np.hypot(*(0.5 * (g[0] + g[-1]) - 0.5 * (c[0] + c[-1]))) <= 1e-9


@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_thickness_scaling_gap_components(factor):
    """Thickness scaling on an open cambered section, in the original chord frame: the
    chord-normal part of the trailing-edge gap scales by the factor and the chordwise offset
    between the two trailing-edge points is unchanged, both to 1e-9 chord."""
    from foil_lab.geometry import _ArcSpline, _to_chord_frame
    c = generate_naca4("4412", 200)
    sp = _ArcSpline(c)
    before, frame = _to_chord_frame(c, sp)
    g = scale_thickness_camber(c, thickness=factor)
    le, e, nrm, chord = frame
    after = np.column_stack([(g - le) @ e, (g - le) @ nrm]) / chord
    d0, d1 = before[0] - before[-1], after[0] - after[-1]
    assert abs(d1[1] - factor * d0[1]) <= 1e-9
    assert abs(d1[0] - d0[0]) <= 1e-9
