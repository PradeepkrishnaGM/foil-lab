"""Golden values of the solvers (CLAUDE.md "Golden values": recorded before any speed optimisation,
unchanged after it). Each case returns a dict of floats; the state of every viscous case enters as
a SHA-256 of its bytes, so any change of the converged iterate, however small, is caught.

Record (only when a change of the numbers is intended and recorded in DECISIONS.md):

    python -m tests.golden_cases > tests/data/golden/golden_v21.json
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np

import foil_lab as fl
from foil_lab import viscous as V
from foil_lab.benchmark.sections import canonical


def _viscous(r) -> dict:
    x, _ = r.state
    return dict(converged=bool(r.converged), iterations=int(r.iterations), cl=float(r.cl),
                cd=float(r.cd), cm=float(r.cm), xtr_upper=float(r.xtr_upper),
                xtr_lower=float(r.xtr_lower),
                state=hashlib.sha256(np.ascontiguousarray(x, dtype=float).tobytes()).hexdigest())


def _chain(section, re, alphas, trips=None):
    c = canonical(section).coords
    geo = V.build_geometry(c)
    out, state = {}, None
    for a in alphas:
        r = V.solve_viscous(c, a, re, trips=trips, start=state, geometry=geo)
        out[f"{a:g}"] = _viscous(r)
        state = r.state if r.converged else state
    return out


def _api(result) -> dict:
    keys = ("cl", "cd", "cm", "xtr_upper", "xtr_lower")
    d = {k: float(getattr(result, k)) for k in keys if getattr(result, k, None) is not None}
    d["converged"] = bool(result.converged)
    return d


def cases() -> dict:
    """Every golden case (about a minute on one core)."""
    warnings.simplefilter("ignore")
    out = {
        "naca2412_re1e6": _chain("NACA2412", 1e6, [2.0, 5.0]),
        "naca4412_re5e4_pinned": _chain("NACA4412", 5e4, [4.0, 5.0]),
        "naca2412_f10_re1e6": _chain("NACA2412_F10", 1e6, [2.0]),
        "naca2412_re3e6_trip": _chain("NACA2412", 3e6, [2.0], trips={(0, "upper"): 0.1}),
    }
    af = fl.Airfoil.naca("2412")
    inv = fl.analyze(af, 4.0)
    out["inviscid_naca2412_4deg"] = dict(cl=float(inv.cl), cm=float(inv.cm))
    main = fl.Airfoil.naca("2412")
    conf = fl.Configuration((fl.Element(main),
                             fl.Element(main, scale=0.3, rotation_deg=20.0, pivot=(0.0, 0.0),
                                        offset=(0.97, -0.04))))
    out["slotted_re3e6_4deg"] = _api(fl.analyze(conf, 4.0, re=3e6))
    return out


if __name__ == "__main__":
    print(json.dumps(cases(), indent=1, sort_keys=True))
