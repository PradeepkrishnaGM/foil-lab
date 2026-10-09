"""Tests for v0.6 Tier-1 features: multi-element, trip wires, batch."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.geometry import translate, bodies_overlap
from foil_lab.solver import solve_flow
from foil_lab import inviscid as core


# --- True multi-element ----------------------------------------------------- #
def test_multielement_far_separation_recovers_isolated():
    front = generate_naca4("2412", 160)
    rear = translate(generate_naca4("2412", 160), 20.0, 0.0)
    solo = solve_flow(front, 5.0).cl

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


# --- Trip wires ------------------------------------------------------------- #
def test_trip_wire_increases_drag():
    af = generate_naca4("2412", 200)
    natural = solve_flow(af, 5.0, reynolds=3e6).cd_visc
    tripped = solve_flow(af, 5.0, reynolds=3e6,
                         trip_x_upper=0.05, trip_x_lower=0.05).cd_visc
    assert tripped > natural


def test_trip_wire_monotonic():
    af = generate_naca4("2412", 200)
    cds = [solve_flow(af, 5.0, reynolds=3e6, trip_x_upper=t, trip_x_lower=t).cd_visc
           for t in (0.6, 0.4, 0.2, 0.05)]
    # Earlier trip (smaller x/c, later in list) => more drag.
    assert all(cds[i] <= cds[i + 1] + 1e-6 for i in range(len(cds) - 1))


# --- Batch ------------------------------------------------------------------ #
def test_batch_matrix_and_run():
    from foil_lab.batch import build_matrix, run_batch
    afs = {"a": generate_naca4("2412", 120), "b": generate_naca4("0012", 120)}
    cases = build_matrix(afs, alpha=[0, 5], reynolds=[1e6])
    assert len(cases) == 4
    rows = run_batch(cases, workers=1)
    assert len(rows) == 4
    assert all(r["converged"] for r in rows)


def test_batch_export_roundtrip(tmp_path):
    from foil_lab.batch import build_matrix, run_batch, write_table
    afs = {"a": generate_naca4("2412", 120)}
    rows = run_batch(build_matrix(afs, alpha=[0, 5, 10]), workers=1)
    # CSV always works; parquet/h5 require optional deps — skip if absent.
    fmt_deps = {"csv": None, "parquet": "pyarrow", "h5": "h5py"}
    for ext, dep in fmt_deps.items():
        if dep is not None:
            pytest.importorskip(dep)
        p = tmp_path / f"out.{ext}"
        write_table(rows, str(p))
        assert p.exists() and p.stat().st_size > 0


def test_batch_handles_failures_gracefully():
    """A bad case should be flagged, not crash the batch."""
    from foil_lab.batch import Case, run_batch, register_airfoils
    register_airfoils({"good": generate_naca4("2412", 120)})
    cases = [Case(airfoil="good", alpha_deg=5.0),
             Case(airfoil="missing", alpha_deg=5.0)]  # KeyError in worker
    rows = run_batch(cases, workers=1)
    statuses = {r["airfoil"]: r["converged"] for r in rows}
    assert statuses["good"] is True
    assert statuses["missing"] is False
