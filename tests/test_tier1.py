"""Tests for v0.6 Tier-1 features: multi-element, kernels, trip wires, batch."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4
from foil_lab.geometry import translate, bodies_overlap
from foil_lab.solver import solve_flow
from foil_lab.panels import build_panels
from foil_lab.influence import LinearVortexModel, ConstantVortexModel
from foil_lab.linear_system import assemble_and_solve
from foil_lab.forces import integrate_loads


# --- Higher-order kernels --------------------------------------------------- #
def test_both_kernels_correct_lift():
    af = generate_naca4("0012", 200)
    for model in (LinearVortexModel(), ConstantVortexModel()):
        cl0 = solve_flow(af, 0.0, influence_model=model).cl
        cl5 = solve_flow(af, 5.0, influence_model=model).cl
        assert abs(cl0) < 1e-6, f"{type(model).__name__} nonzero at a=0"
        assert 0.5 < cl5 < 0.7, f"{type(model).__name__} Cl5={cl5}"


def test_kernels_agree_closely():
    af = generate_naca4("2412", 200)
    lin = solve_flow(af, 5.0, influence_model=LinearVortexModel()).cl
    con = solve_flow(af, 5.0, influence_model=ConstantVortexModel()).cl
    assert abs(lin - con) < 0.05


# --- True multi-element ----------------------------------------------------- #
def test_multielement_far_separation_recovers_isolated():
    front = generate_naca4("2412", 160)
    rear = translate(generate_naca4("2412", 160), 20.0, 0.0)
    solo = solve_flow(front, 5.0).cl

    g = build_panels([front, rear])
    C = LinearVortexModel().compute(g)
    flow = assemble_and_solve(g, C, np.deg2rad(5.0))
    cp = 1 - flow.vt_total ** 2
    cl_front = integrate_loads(g, cp, np.deg2rad(5.0), body_index=0).cl
    cl_rear = integrate_loads(g, cp, np.deg2rad(5.0), body_index=1).cl
    # Far apart: each recovers (close to) the isolated value.
    assert abs(cl_front - solo) < 0.05
    assert abs(cl_rear - solo) < 0.05


def test_multielement_independent_circulations():
    front = generate_naca4("2412", 160)
    rear = translate(generate_naca4("0012", 160), 2.0, -0.4)
    g = build_panels([front, rear])
    C = LinearVortexModel().compute(g)
    flow = assemble_and_solve(g, C, np.deg2rad(5.0))
    # Two distinct bodies => two circulation unknowns.
    assert flow.gammas.shape == (2,)


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
