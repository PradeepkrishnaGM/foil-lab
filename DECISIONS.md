# Decisions

Every design decision and every answer to a planning question, newest milestone last.
Out-of-scope problems found along the way are listed under "Later" at the end.

---

## M1: Benchmark harness and baseline

Planning answers given on 2026-10-08.

### Rules for M1

- M1 changes no solver numerics. Before and after step 1 the full suite must give
  86 passed, 2 xfailed, and `examples/validation_report.py` must reproduce
  `validation_output.txt` byte for byte.
- No bug listed in the v1.3.1 code review is fixed in M1, however easy. The baseline must
  describe v1.3.1 as it is (numerics of commit `be54d0a`).
- Tolerances of the new exact-solution tests are fixed before they are run. If a test that was
  expected to pass fails, work stops and the numbers are shown; the tolerance is not changed and
  the test is not marked expected-failure without an explicit decision.
- At the end of M1, revised accuracy targets and exact wording changes to `SPEC.md` 4.1 (alpha
  range, attached-flow filter, reference recipe) are proposed for approval. `SPEC.md` is not
  edited without approval.

### Q1. Running XFOIL

XFOIL 6.99 as packaged for Fedora (`xfoil-6.99-25.fc44`) crashes with a floating-point
exception as soon as an operating point is solved when graphics are disabled (`PLOP`, `G`),
including inviscid runs. With graphics left on it runs normally.

Decision: run XFOIL as `xvfb-run -a xfoil` with graphics on. The rpm release
(`rpm -q xfoil`) is recorded in every provenance header. Regenerating reference data needs
Xvfb; the tests never do.

### Q2. Reference recipe

Decision: our canonical coordinate file, loaded into XFOIL, then `PANE` at 160 nodes.
Before generating the reference set, the 0.856 outlier (NACA 2412, Re 1e6, alpha 5 deg) is
diagnosed and a table of Cl and Cd at 120/160/200/240 `PANE` nodes and with the file as-is is
shown for review. If the canonical file needs fixing, that is a file-format fix in the
benchmark writer, not a solver change. Diagnosis results: see "Q2 diagnosis" below.

### Q3. Methods in the baseline

- Full matrix: inviscid (`Cl`, `Cm` only), the default one-way viscous path (`solve_flow` with
  Re), and transpiration VII (`vii.solve_vii`).
- Newton VII (`vii_newton.solve_vii_newton`) only on a small set: NACA 0012 and 2412,
  Re 5e4 and 2e5, alpha 0 to 8 deg in 2 deg steps.

### Q4. Operating conditions on our side

160 panels for every method. N_crit 9, free transition on both surfaces, Mach 0.

### Q5. Alpha range

Sweep -4 to +20 deg in 1 deg steps. Every angle where XFOIL converged is kept. A missing angle
is recorded as not converged and is never interpolated.

### Q6. Attached-flow filter

A point is "attached" if its alpha lies between -4 deg and the first angle at which XFOIL's
Cl reaches 0.9 times its maximum Cl, where the maximum is taken over the positive-alpha sweep.
Post-stall points are excluded even when their Cl is lower. The number of points inside the
filter is reported per case. Plots show all points.

### Q7. Non-NACA sections

Clark Y and Eppler E387 from the UIUC Airfoil Coordinates Database, downloaded on demand into
`~/.cache/foil-lab/`. The source URL and a checksum are committed; the coordinates are not.
The downloader converts to Selig order and checks for duplicate points (the UIUC files are not
all in the same format). Tests that need these sections skip cleanly when the cache is empty
and there is no network.

### Q8. What is committed

- The curated XFOIL reference CSVs, with provenance headers.
- `docs/baseline_v1.3.1.md`: tables only, no images, stating the commit the numerics were
  measured on (`be54d0a` numerics) and the date.
- Everything else the benchmark produces (our results, plots, logs) stays out of git.

### Q9. Old reference data

`naca2412_xfoil_re50k.csv` (downloaded third-party data) and `naca0012_thin_airfoil.csv`
(flat-plate formula, not a valid reference for a 12 % thick section) are removed, together with
the tests that use them, but only after the new benchmark and exact-solution tests are in place.
`validation.py` and `examples/validation_report.py` are updated or removed in the same step so
that nothing is left broken. Every removed test and the reason are listed under
"Removed tests" below.

### Q10. Two-element exact solution

Option (c): the published two-element exact solution (Williams, ARC R&M 3717) is added in M2
from the report, supplied by the owner. Values are never typed from memory; when transcribed,
the table is shown with page numbers. Two data-free tests are added now:

- NACA 0012 and NACA 4412, 50 chords apart: each element's Cl is within 0.5 % of its isolated
  value.
- A mirrored pair: lifts equal and opposite by symmetry.

### Q11. Tolerances (fixed now)

| Test | Tolerance |
|---|---|
| Kármán-Trefftz surface pressure | max abs dCp <= 0.02 for x/c in 0.02-0.98 at 200 panels, error decreasing from 160 to 320 panels; section parameters stated in the test; alpha 0 and 4 deg |
| Two bodies 50 chords apart (0012 + 4412) | each element within 0.5 % of its isolated Cl |
| Mirrored pair | lifts equal and opposite |
| Ground effect vs mirrored two-body problem | abs dCl <= 1e-6, plus normal velocity on the ground plane zero to 1e-8 |
| Pressure lift = circulation lift = far-field lift | within 0.5 % |
| Exact transforms without repanelling | 1e-12 |
| Transforms through repanelling | 1e-4 chord, measured as distance to the original curve |
| Corrections applied exactly once | 1e-3 |
| NumPy vs JAX inviscid Cl | <= 1e-6, all four NACA sections at three angles |
| Determinism | bit-identical |

### Q12. Expected-failure style

`pytest.mark.xfail(strict=True, raises=AssertionError, reason="fixed in Mx: ...")`. A test that
unexpectedly passes fails the suite; a crash is not counted as an expected failure.

### Q13. Stray directory

The empty untracked directory `src/foil_lab/{gui,utils}` in the main checkout is removed with
`rmdir` (fails if it is not empty).

### Q14. Supported Python versions

`requires-python = ">=3.10"` is left as it is; see "Later".

### Q15. Transition location for v1.3.1

Captured in the benchmark adapter by wrapping the internal boundary-layer function. No library
change.

### Step 1: packaging and hygiene

- `sys.path` hacks removed from `main.py` and all examples; the package must be installed
  (`pip install -e ".[all]"`). The pytest `pythonpath` setting is removed for the same reason.
- The two legacy reference CSVs moved from `tests/validation/data/` to package data
  `src/foil_lab/data/legacy/` and are loaded through `importlib.resources`; `pyproject.toml`
  declares them as package data. Checked: a built wheel contains them and `load_reference`
  works from the unpacked wheel.
- Markers `slow` and `xfoil` registered with `--strict-markers`. `tests/conftest.py` skips
  `xfoil` tests when `xfoil` or `xvfb-run` is missing.
- `slow` was applied to the 11 tests measured at 2 s or more on this machine (i5-10500T):
  four Newton VII tests, the low-Re VII limitation test, the quasi-simultaneous xfail, both
  `match_cp` tests, `test_vii_drag_rises_with_alpha`, and two JAX gradient tests.
- `test_jax.py::test_lift_slope_and_alpha0` now takes about 2.1 s in the fast suite. Nearly all
  of that is the one-off JAX gradient compilation, which lands on whichever gradient test runs
  first; marking it as well would only move the cost again, so it stays in the fast suite.
- Measured after step 1: full suite 86 passed, 2 xfailed (73.6 s); fast suite 77 passed,
  11 deselected (15.6 s); `validation_output.txt` reproduced byte for byte.

### Step 2: XFOIL runner

- `foil_lab/benchmark/xfoil.py`: coordinate-file writer (Selig order and duplicate-point checks,
  10 decimals, SHA-256), command script, subprocess under `xvfb-run -a` with a timeout, polar
  parser, provenance header, reference CSV read/write.
- Each case is two fresh XFOIL processes (0 up to alpha_max, 0 down to alpha_min). Alpha 0 comes
  from the upward run. Crashes and timeouts are recorded in the provenance, not raised.
- New defaults in `foil_lab/params.py` (start of the single parameters module; solver
  parameters move there in M2): 160 `PANE` nodes, N_crit 9, free transition (`XTR 1 1`),
  Mach 0, `ITER 200`, 300 s timeout per process, sweep -4 to 20 deg in 1 deg steps.
  `ITER 200` and the timeout are proposals awaiting approval.

### Q2 diagnosis (NACA 2412, Re 1e6, alpha 5 deg)

Checked and ruled out:

- Duplicate points: none (161 points, smallest consecutive spacing 3.9e-4).
- Point ordering: Selig order (positive signed area).
- Unconverged or inviscid result: the 0.856 run was viscous and converged; the inviscid value
  of that geometry is 0.867. The same value is reached by a direct `ALFA 5` and by a 0-to-5 deg
  sequence, so it is not hysteresis.
- File resolution and number format: 160 or 320 panels, 6 or 10 decimals give the same result
  after `PANE`.

Cause: the trailing-edge definition. Foil-Lab's `generate_naca4` uses the closed-trailing-edge
thickness coefficient (-0.1036, sharp TE). XFOIL's own `NACA` command and NACA Report 460 use
-0.1015 (blunt TE, gap 0.25 % chord). For a sharp trailing edge, XFOIL's result keeps changing
with the number of `PANE` nodes; for the blunt one it does not.

Measured through the runner (viscous, alpha 5 deg):

| PANE nodes | closed TE (ours): Cl | Cd | Cm | open TE (-0.1015): Cl | Cd | Cm |
|---|---|---|---|---|---|---|
| 120 | 0.8620 | 0.00829 | -0.0640 | 0.8105 | 0.00791 | -0.0531 |
| 160 | 0.8564 | 0.00824 | -0.0629 | 0.8101 | 0.00793 | -0.0531 |
| 200 | 0.8522 | 0.00822 | -0.0620 | 0.8100 | 0.00794 | -0.0530 |
| 240 | 0.8493 | 0.00820 | -0.0614 | 0.8099 | 0.00795 | -0.0530 |
| as-is | 0.8152 | 0.00881 | -0.0548 | 0.8101 | 0.00797 | -0.0531 |

XFOIL's own `NACA 2412` (160 nodes): Cl 0.8094, Cd 0.00775, Cm -0.0540.

Further measurements:

- The drift is a property of sharp trailing edges in XFOIL, not of our file: XFOIL's own
  NACA 2412 closed with `TGAP 0` gives Cl 0.8582 / 0.8524 / 0.8486 / 0.8456 / 0.8424 at
  120 / 160 / 200 / 240 / 300 nodes. Our closed-TE file is still moving at 360 nodes (0.8433).
- XFOIL's "max camber" printout for a loaded file (0.01906) differs from the 0.020 it prints
  for its `NACA` command for both TE variants; this is how XFOIL estimates camber from points,
  not a geometry difference.
- XFOIL sometimes skips single attached angles at Re 1e6 (for example 2 and 3 deg for the
  closed TE at 160 nodes). Those are recorded as not converged (Q5).

### D1. Canonical NACA geometry: open trailing edge

`generate_naca4` and `generate_naca5` take `trailing_edge="closed"` (default, -0.1036, as
before) or `"open"` (-0.1015, NACA Report 460). The default is unchanged and the closed
geometry is bit-identical to v1.3.1 (pinned by `tests/test_generators.py`). Measured after the
change: full suite 102 passed, 2 xfailed (86 existing + 10 runner + 6 generator tests);
`validation_output.txt` reproduced byte for byte. The benchmark uses the open trailing edge for
both codes.

Check of v1.3.1 with the open file before running the matrix (NACA 2412, both geometries
written by the benchmark writer and read back with `load_dat`; VII with the command-line
settings `max_iter=200`, `tol=1e-3`, Re 1e6):

| Method | alpha | Cl closed | Cl open | dCl | dCm | converged (closed / open) |
|---|---|---|---|---|---|---|
| inviscid | 0 | 0.2579 | 0.2589 | +0.0010 | -0.0002 | - |
| inviscid | 5 | 0.8563 | 0.8577 | +0.0015 | -0.0005 | - |
| inviscid | 10 | 1.4476 | 1.4495 | +0.0019 | -0.0007 | - |
| VII | 0 | 0.2183 | 0.2232 | +0.0048 | -0.0011 | yes / yes |
| VII | 5 | 0.7171 | 0.7312 | +0.0141 | -0.0032 | yes / yes |
| VII | 10 | 1.4549 | 1.4819 | **+0.0270** | -0.0057 | **no / no** (200 iterations) |

- Inviscid: no trailing-edge spike. The open TE gives slightly lower, smoother Cp on the last
  panels (for example 0.420 instead of 0.563 at 5 deg on the TE panels).
- VII at 10 deg: the stop criterion (dCl > 0.02) is met, but both runs are unconverged iterates
  at the 200-iteration limit. With `max_iter=400` both converge (closed at 281, open at 240
  iterations) and dCl = -0.0058 (Cl 1.5008 closed, 1.4950 open). At 5 deg the +0.0141 is a
  converged difference and does not change with more iterations.
- VII at 10 deg, both geometries, converged or not: the upper-surface Cp oscillates panel to
  panel over the last six panels (converged closed: 0.563, 0.160, -0.104, 0.053, 0.059, -0.122;
  open: 0.480, 0.246, -0.091, 0.019, 0.110, -0.169). This is a v1.3.1 VII defect, present with
  the closed trailing edge as well; it is not caused by the open file.
- Status: stopped before the matrix, as required by the stop rule. Awaiting a decision.

### D2. Paneling sensitivity

XFOIL is run at 160 and at 240 `PANE` nodes for every section. The difference is reported as
"paneling sensitivity (160 vs 240)", not as uncertainty; the report states that for sharp
trailing edges it is a lower bound. Each section's trailing-edge gap is recorded in the
provenance header (`trailing_edge_gap`). Accuracy targets are judged on sections with an open
trailing edge; sharp-edged sections are reported separately with that caveat.

### D3. XFOIL limits

`ITER 200` and a 300 s timeout per XFOIL process: approved.

### D4. Skipped angles

One retry per skipped angle: a fresh XFOIL process that solves that angle directly (`ALFA`), so
the boundary layer starts from XFOIL's own initialisation. Retried points are flagged in the
`retried` column of the reference CSV. If the retry fails, the angle stays not converged. The
console output of every XFOIL process (sweeps and retries) is kept in the gitignored
`benchmark_output/` folder.

Diagnosis of a skipped angle (NACA 2412 closed TE, Re 1e6, 160 nodes, 2 deg; logs in
`benchmark_output/diag_d4/`): XFOIL itself fails to converge. The polar file it writes has no
2 or 3 deg row, so the runner and parser are not involved. At 2 deg the initial boundary-layer
march fails on the lower surface ("MRCHDU: Convergence failed at 76 side 2"), then the Newton
iteration stalls at rms residual 5.6e-3 (largest change at upper-surface station 55) for all
200 iterations with Cl fixed at 0.4380, and XFOIL prints "VISCAL: Convergence failed". 3 deg
behaves the same (stall at rms 5.05e-3). The single retry from a fresh start stalls at exactly
the same residual (5.593e-3) and also fails, so both angles stay not converged.

### Removed tests

(None yet.)

---

## Later

- `requires-python = ">=3.10"` is declared but only Python 3.14 is available and tested here.
  Decide the supported versions and test them in M7.
- `examples/basic_alpha_sweep.py` opens `naca2412.dat` relative to the current directory, so it
  only works when run from `examples/`. Fix with the M7 example rework.
- README and TESTING.md still describe `PYTHONPATH=src` and older test counts; rewrite in M9.
- M2: decide which NACA trailing edge (open or closed) is the default in v2.0.
- M3: transpiration VII shows a panel-to-panel Cp oscillation over the last panels of the
  upper surface at 10 deg (NACA 2412, Re 1e6), with either trailing edge.
- M3: the command-line VII settings (`max_iter=200`) stop before convergence at 10 deg for
  NACA 2412 at Re 1e6; 240-281 iterations are needed there.
