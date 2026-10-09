"""Two-element exact test case: Williams, ARC R&M 3717 (1973), configurations A and B, alpha 0.

Option (b): the core runs on the tabulated Table 2 geometry (see ``tests/williams_tables.py``).
Fixed tolerances (DECISIONS.md, M2, Q5): at 200 panels per element, Cl within 1 % of Table 3
per element and in total; max |dCp| <= 0.05 for x/c 0.02-0.98 of each element; the error falls
from 200 to 400 panels.
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
        out[e] = (core.body_loads(sol, k).cl, float(np.abs(cp - el.table[:, 2])[win].max()))
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
def test_williams_error_falls_with_refinement(config):
    r200, r400 = _solve(config, 200), _solve(config, 400)
    for e in ("main", "flap"):
        assert r400[e][1] < r200[e][1], (e, r200[e][1], r400[e][1])
