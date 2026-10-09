"""Two-element exact test case: Williams, ARC R&M 3717 (1973), configurations A and B, alpha 0.

Option (b): the core runs on the tabulated Table 2 geometry (see ``tests/williams_tables.py``).
Fixed tolerances (DECISIONS.md, M2): at 200 panels per element, Cl within 1 % of Table 3 per
element and in total, and max |dCp| <= 0.05 for x/c 0.02-0.98 of each element (Q5); refinement:
the difference to our own 1600-panel solution on the same geometry (max |dCp| at the tabulated
points, x/c 0.02-0.98) falls from 200 to 400 to 800 panels with observed order >= 1.5 on each
element. The Q5 criterion "error against the table falls from 200 to 400 panels" was replaced by
the user because it measured the resolution of the 61-point table (DECISIONS.md, M2).
"""

import numpy as np
import pytest

from foil_lab import inviscid as core
from tests.williams_tables import CONFIG, Element, read_forces, read_table2


def _solve(config, num_panels):
    els = {e: Element.from_table(read_table2(config, e)) for e in ("main", "flap")}
    nodes = {e: el.nodes(num_panels) for e, el in els.items()}
    sol = core.solve(core.PanelSet.from_bodies([nodes["main"][0], nodes["flap"][0]]), 0.0)
    out = {}
    for k, e in enumerate(("main", "flap")):
        el, s = els[e], nodes[e][1]
        cp = np.interp(el.s_table, 0.5 * (s[:-1] + s[1:]), core.to_selig_order(sol, sol.cp, k))
        xc = el.x_over_c(el.table[:, :2])
        win = (xc >= 0.02) & (xc <= 0.98)
        out[e] = (core.body_loads(sol, k).cl, float(np.abs(cp - el.table[:, 2])[win].max()),
                  cp[win])
    out["total"] = -2.0 * float(np.sum(sol.circulation))
    return out


@pytest.mark.parametrize("config", ["A", "B"])
def test_williams_lift_and_pressure(config):
    f = read_forces()
    r = _solve(config, 200)
    for e in ("main", "flap"):
        ref = f[(CONFIG[config], "CL", e)]
        assert abs(r[e][0] / ref - 1.0) <= 0.01, (e, r[e][0], ref)
        assert r[e][1] <= 0.05, (e, r[e][1])
    ref_t = f[(CONFIG[config], "CL_total", "total")]
    assert abs(r["total"] / ref_t - 1.0) <= 0.01, (r["total"], ref_t)


@pytest.mark.slow
@pytest.mark.parametrize("config", ["A", "B"])
def test_williams_converges_on_tabulated_geometry(config):
    ref = _solve(config, 1600)
    runs = {n: _solve(config, n) for n in (200, 400, 800)}
    for e in ("main", "flap"):
        err = [float(np.max(np.abs(runs[n][e][2] - ref[e][2]))) for n in (200, 400, 800)]
        orders = [np.log2(err[0] / err[1]), np.log2(err[1] / err[2])]
        assert err[0] > err[1] > err[2], (e, err)
        assert min(orders) >= 1.5, (e, err, orders)
