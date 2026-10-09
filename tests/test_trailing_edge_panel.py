"""Open trailing edge: Drela (1989) trailing-edge panel in the inviscid core.

Tolerances are the fixed values recorded in DECISIONS.md (M2) before the first run. Geometry:
NACA 2412 at alpha = 5 deg, opened from the closed section linearly in x (upper surface
+x g/2, lower surface -x g/2), a construction local to these tests.
"""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest

from foil_lab import inviscid as core
from foil_lab.benchmark.sections import canonical
from foil_lab.benchmark.xfoil import build_script, parse_polar_text, run_xfoil, write_coordinate_file
from foil_lab.generators import generate_naca4
from foil_lab.params import XFOIL

ALPHA = 5.0


def _opened(n, gap):
    c = generate_naca4("2412", n, trailing_edge="closed").copy()
    le = int(np.argmin(c[:, 0]))
    c[:le, 1] += 0.5 * gap * c[:le, 0]
    c[le + 1:, 1] -= 0.5 * gap * c[le + 1:, 0]
    return c


def _run(coords, alpha=ALPHA):
    ps = core.PanelSet.from_bodies(coords)
    sol = core.solve(ps, alpha)
    return ps, sol, core.body_loads(sol, 0)


def test_closed_trailing_edge_has_no_te_panel():
    ps = core.PanelSet.from_bodies(generate_naca4("2412", 160, trailing_edge="closed"))
    assert core.trailing_edge_panels(ps).size == 0


def test_te_panel_signs_match_drela_eq2():
    """sigma_TE and gamma_TE magnitudes are those of Drela (1989) eq. (2); source is outflow."""
    ps, sol, _ = _run(_opened(320, 0.0025))
    te = core.trailing_edge_panels(ps)
    g = np.asarray(sol.gamma)
    q = float((te.coupling @ g)[0])
    first, last = ps.body_nodes[0].start, ps.body_nodes[0].stop - 1
    assert q == pytest.approx(0.5 * (g[first] - g[last]))
    assert q > 0 and te.s_dot_n[0] > 0          # flow leaves through the base: sigma_TE > 0
    assert te.s_dot_n[0] ** 2 + te.s_dot_t[0] ** 2 == pytest.approx(1.0)


def test_gap_to_zero_recovers_closed_section():
    _, _, ref = _run(_opened(320, 0.0))
    dcl, dcm = [], []
    for gap in (0.0025, 0.001, 0.0003, 0.0001):
        _, _, ld = _run(_opened(320, gap))
        dcl.append(abs(ld.cl - ref.cl))
        dcm.append(abs(ld.cm - ref.cm))
    assert all(np.diff(dcl) < 0) and all(np.diff(dcm) < 0), (dcl, dcm)
    assert dcl[-1] <= 0.002 and dcm[-1] <= 0.0005


def test_flow_through_gap_is_small():
    """Mean normal velocity just inside the trailing-edge panel <= 0.02 of the freestream."""
    c = _opened(320, 0.0025)
    ps, sol, _ = _run(c)
    a, b = c[0], c[-1]
    gap = b - a
    length = np.hypot(*gap)
    nrm = np.array([gap[1], -gap[0]]) / length
    nrm = nrm if nrm[0] > 0 else -nrm
    pts = a[None, :] + np.outer(np.linspace(0.1, 0.9, 9), gap) - 1e-3 * length * nrm[None, :]
    u, v = core.induced_velocity(ps, sol.gamma, pts[:, 0], pts[:, 1])
    vn = (u + np.cos(np.deg2rad(ALPHA))) * nrm[0] + (v + np.sin(np.deg2rad(ALPHA))) * nrm[1]
    assert abs(vn.mean()) <= 0.02


def _second_difference_alternates(seg):
    """True if the second difference flips sign twice in a row (+-+ or -+-): a sawtooth."""
    s = np.sign(seg[:-2] - 2 * seg[1:-1] + seg[2:])
    flips = s[1:] * s[:-1] < 0
    return bool(np.any(flips[1:] & flips[:-1]))


@pytest.mark.slow
def test_te_panel_refinement():
    res = {}
    for n in (160, 320, 640, 1280):
        _, sol, ld = _run(_opened(n, 0.0025))
        res[n] = (ld, np.asarray(sol.cp), np.asarray(sol.gamma))
    for n, (_, cp, g) in res.items():
        assert max(abs(cp[0]), abs(cp[-1])) <= 1.1, n
        k = max(4, int(0.05 * len(g)))
        assert not _second_difference_alternates(g[:k]), n
        assert not _second_difference_alternates(g[-k:]), n
    assert max(abs(res[1280][1][0] - res[640][1][0]), abs(res[1280][1][-1] - res[640][1][-1])) <= 0.02
    f = res[1280][0]
    assert abs(res[160][0].cl - f.cl) <= 0.005 and abs(res[160][0].cm - f.cm) <= 0.001
    for n in (320, 640):
        assert abs(res[n][0].cl - f.cl) <= 0.002 and abs(res[n][0].cm - f.cm) <= 0.0005
    pairs = ((160, 320), (320, 640), (640, 1280))
    for attr in ("cl", "cm"):
        succ = [abs(getattr(res[b][0], attr) - getattr(res[a][0], attr)) for a, b in pairs]
        assert all(np.diff(succ) < 0), (attr, succ)
    assert abs(res[1280][2][0]) <= 1.1 * abs(res[160][2][0])


def _xfoil_inviscid(coords, pane, alphas):
    params = dataclasses.replace(XFOIL, pane_nodes=pane)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_coordinate_file(work / "foil.dat", coords, "NACA2412")
        for a in alphas:
            run = run_xfoil(build_script("foil.dat", "polar.txt", 0.0, a, params=params), work,
                            "polar.txt", params=params)
            _, rows = parse_polar_text(run.polar_text)
            assert rows, f"XFOIL gave no inviscid result at {a} deg (pane {pane})"
            out[a] = (rows[-1].cl, rows[-1].cm)
    return out


@pytest.mark.slow
@pytest.mark.xfoil
def test_inviscid_core_matches_inviscid_xfoil_open_naca2412():
    """Open NACA 2412, canonical 160-panel file: XFOIL (no VISC, PANE 160) vs the core at 160.

    Fixed tolerances (DECISIONS.md, M2): |dCl| <= 0.005 and |dCm| <= 0.002 at 0, 5 and 10 deg.
    """
    coords = canonical("NACA2412", 160).coords
    ref = _xfoil_inviscid(coords, 160, (0.0, 5.0, 10.0))
    for a, (cl_x, cm_x) in ref.items():
        _, _, ld = _run(coords, a)
        assert abs(ld.cl - cl_x) <= 0.005, (a, ld.cl, cl_x)
        assert abs(ld.cm - cm_x) <= 0.002, (a, ld.cm, cm_x)
