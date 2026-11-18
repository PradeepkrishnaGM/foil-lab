"""Tests for v0.6 Tier-1 features: multi-element, trip wires, batch."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.geometry import translate, bodies_overlap
import foil_lab as fl
from tests import section
from foil_lab import inviscid as core


# --- True multi-element ----------------------------------------------------- #
def test_multielement_far_separation_recovers_isolated():
    front = generate_naca4("2412", 160)
    rear = translate(generate_naca4("2412", 160), 20.0, 0.0)
    solo = section.cl(front, 5.0)

    sol = core.solve(core.PanelSet.from_bodies([front, rear]), 5.0)
    cl_front = core.body_loads(sol, 0).cl
    cl_rear = core.body_loads(sol, 1).cl
    # Far apart: each recovers (close to) the isolated value.
    assert abs(cl_front - solo) < 0.05
    assert abs(cl_rear - solo) < 0.05


def test_multielement_independent_circulations():
    front = generate_naca4("2412", 160)
    rear = translate(generate_naca4("0012", 160), 2.0, -0.4)
    sol = core.solve(core.PanelSet.from_bodies([front, rear]), 5.0)
    # Two distinct bodies => one circulation each, both free (not tied together).
    assert sol.circulation.shape == (2,)
    assert abs(sol.circulation[0] - sol.circulation[1]) > 1e-3


def test_overlap_detection():
    a = generate_naca4("2412", 80)
    b = translate(generate_naca4("2412", 80), 0.3, 0.0)  # overlaps
    c = translate(generate_naca4("2412", 80), 5.0, 0.0)  # clear
    assert bodies_overlap([a, b])
    assert not bodies_overlap([a, c])


# --- Trip wires (coupled viscous solver) ------------------------------------ #
@pytest.mark.slow
def test_trip_wire_increases_drag():
    af = fl.Airfoil.from_coords(generate_naca4("2412", 200))
    natural = fl.analyze(af, 5.0, re=3e6, repanel=False).cd
    tripped = fl.analyze(af, 5.0, re=3e6, repanel=False, trip_upper=0.05, trip_lower=0.05).cd
    assert tripped > natural


@pytest.mark.slow
def test_trip_wire_monotonic():
    af = fl.Airfoil.from_coords(generate_naca4("2412", 200))
    cds = [fl.analyze(af, 5.0, re=3e6, repanel=False, trip_upper=t, trip_lower=t).cd
           for t in (0.6, 0.4, 0.2, 0.05)]
    # Earlier trip (smaller x/c, later in list) => more drag.
    assert all(cds[i] <= cds[i + 1] + 1e-6 for i in range(len(cds) - 1))


# --- Batch ------------------------------------------------------------------ #
@pytest.mark.slow
def test_batch_matrix_and_run():
    from foil_lab.batch import build_matrix, run_batch
    afs = {"a": generate_naca4("2412", 120), "b": generate_naca4("0012", 120)}
    cases = build_matrix(afs, alpha=[0, 5], re=[1e6])
    assert len(cases) == 4
    rows = run_batch(cases, afs)
    assert len(rows) == 4 and [r["airfoil"] for r in rows] == ["a", "a", "b", "b"]
    assert all(r["converged"] for r in rows)
    par = run_batch(cases, afs, backend="process", workers=2)
    for r, q in zip(rows, par):          # same results in the same order from the process pool
        assert (r["alpha"], r["cl"], r["cd"]) == (q["alpha"], q["cl"], q["cd"])


def test_batch_export_roundtrip(tmp_path):
    from foil_lab.batch import build_matrix, run_batch, write_table
    afs = {"a": generate_naca4("2412", 120)}
    rows = run_batch(build_matrix(afs, alpha=[0, 5, 10]), afs)
    # CSV always works; parquet/h5 require optional deps — skip if absent.
    fmt_deps = {"csv": None, "parquet": "pyarrow", "h5": "h5py"}
    for ext, dep in fmt_deps.items():
        if dep is not None:
            pytest.importorskip(dep)
        p = tmp_path / f"out.{ext}"
        write_table(rows, str(p))
        assert p.exists() and p.stat().st_size > 0


def test_batch_handles_failures_gracefully():
    """A missing airfoil or a failing group is flagged in its rows, never crashes the batch."""
    from foil_lab.batch import Case, run_batch
    cases = [Case("good", 5.0), Case("missing", 5.0), Case("bad", 5.0, mach=1.5)]
    rows = run_batch(cases, {"good": generate_naca4("2412", 120), "bad": generate_naca4("2412", 120)})
    statuses = {r["airfoil"]: r["converged"] for r in rows}
    assert statuses == {"good": True, "missing": False, "bad": False}
    assert "missing" in rows[1]["error"] and "ValueError" in rows[2]["error"]
