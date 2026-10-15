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

### E1, E2 and report additions

- E1: the open trailing edge is used for the whole matrix.
- E2: v1.3.1 VII runs with its command-line settings (200 iterations, tol 1e-3). Unconverged
  points are reported as not converged and excluded from the error metrics, with the count
  shown per case.
- The baseline report shows, per case, the number of attached-filter angles where XFOIL did
  not converge at 160 and at 240 nodes, and how many of those the 240-node run recovers.
- Per case and method, the number of v1.3.1 points that did not converge is shown next to the
  error metrics, so a method cannot look better because its bad points dropped out.

### Step 3: reference set

- Sections: NACA 0012, 2412, 4412 (`generate_naca4`, open TE), NACA 23012 (`generate_naca5`,
  open TE), Clark Y and E387 from the UIUC database. Clark Y is in Lednicer format and E387 in
  Selig format; both are converted to Selig order (no duplicate points found). They are
  normalised and repanelled to 160 panels with the v1.3.1 `normalize_airfoil` and
  `repanel_cosine`. Largest distance of the repanelled points to the original point polyline:
  Clark Y 1.6e-4, E387 8.1e-4 chord (E387 has only 61 original points, so part of this is the
  polyline itself). Trailing-edge gaps: NACA 0.00252, Clark Y 0.00120, E387 0 (sharp).
- Classification for the targets (D2): a section is "sharp" if its trailing-edge gap is zero.
  Clark Y (gap 0.0012) counts as open.
- Reference files: `src/foil_lab/data/reference/xfoil/<SECTION>_Re<re>_N<nodes>.csv`,
  84 files (6 sections x 7 Reynolds numbers x 160/240 nodes), package data. XFOIL console logs
  and the canonical coordinate files are written to `benchmark_output/`.
- Download cache: `$FOIL_LAB_CACHE` or `~/.cache/foil-lab/airfoils`; the SHA-256 of each
  downloaded file is checked against the value committed in `benchmark/sections.py`.
- Runner bug found and fixed during the first generation: on a timeout, `subprocess.run` killed
  only `xvfb-run`, so the stuck `xfoil` and its Xvfb server kept running (8 orphans, up to 36
  minutes at about 85 % CPU each). XFOIL now runs in its own session and the whole process group
  is killed on timeout (test: `test_timeout_kills_grandchildren`). The converged XFOIL data was
  not affected, but run times were inflated and some of the 7 timeouts may have been caused by
  the CPU contention, so the whole reference set and the baseline were regenerated with the
  fixed runner.

### Step 4: baseline adapter

- `foil_lab/benchmark/baseline_v131.py` runs v1.3.1 unchanged on the canonical files.
  Transition locations are captured by wrapping `solver.calculate_viscous_drag`,
  `boundary_layer.analyze_boundary_layer`, `vii._split_surfaces`, `vii.march_boundary_layer`
  and their `vii_newton` counterparts inside the worker process (Q15). For the one-way path the
  stagnation-point search of `calculate_viscous_drag` is repeated in the adapter to map
  boundary-layer stations to x/c.
- One-way and inviscid results have no convergence criterion in v1.3.1; they are reported as
  "n/a" and counted as converged unless a value is not finite.
- Runs use one BLAS thread per worker process (6 workers). This changes run time, not the
  method.

### Step 5: comparison rules

- The attached-flow filter excludes the cut-off angle itself (SPEC 4.1: "below 90 % of
  XFOIL's maximum lift").
- Metrics are pooled over points (each used point has equal weight) when aggregating over
  sections.

### Step 6: exact-solution tests, first run (xfail marks ignored)

Expected failures, all confirmed and marked `xfail(strict=True, raises=AssertionError)`:

| Test | Measured | Tolerance | Fixed in |
|---|---|---|---|
| NACA 0012 + NACA 4412, 50 chords apart, stacked | -4.6 % / -5.9 % | 0.5 % | M2 |
| Mirrored pair, Cl1 + Cl2 | 0.116 | 1e-9 | M2 |
| Ground effect vs mirror problem | 4.54 | 1e-6 | M6 |
| Normal velocity on the ground plane | 0.111 | 1e-8 | M6 |
| Far-field lift / pressure lift - 1 | 1.000 | 0.005 | M2 |
| Repanelling scaled / translated / rotated input | 136 / 3.4e8 / 41 chord | 1e-4 | M2 |
| Aspect ratio 1e6 + Mach 0.5 vs 2D | 0.139 | 1e-3 | M6 |

Layout of the two-body test: the bodies are stacked (offset 50 chords across the stream), not
in tandem. Measured with a per-body-closed assembly (scratch, not library code), the physical
2D interference at 50 chords in tandem is +2.30 % on the NACA 0012 (point-vortex estimate
+2.07 %), so a tandem test could never pass at 0.5 % even with a correct solver; stacked it is
-0.18 % (0012) and +0.11 % (4412). The first run used tandem (2.03 %); it was changed before the
test was committed. The tolerance is unchanged. v1.3.1 stacked: -4.6 % / -5.9 %.

Passed as expected: Kármán-Trefftz at 0 deg (0.0111 at 200 panels), pressure lift equals
circulation lift, exact transforms, Kármán-Tsien applied once, NumPy and JAX at 11 of 12
points, determinism.

Expected to pass but failed (stop rule: tolerances unchanged, not marked; awaiting a decision):

- Kármán-Trefftz at 4 deg: max |dCp| 0.0220 > 0.02 at 200 panels. The analytic solution was
  checked: its integrated Cp reproduces the Kutta-Joukowski lift exactly (0.996245). The v1.3.1
  error is first order and sits at x/c = 0.02 on the upper surface: 0.0273 / 0.0220 / 0.0139 /
  0.0068 / 0.0035 at 160 / 200 / 320 / 640 / 1280 panels. At 0 deg: 0.0139 / 0.0111 / 0.0072 /
  0.0037 / 0.0019.
- NumPy vs JAX, NACA 4412 at -4 deg: 1.49e-6 > 1e-6. All 12 points: 0012 8.9e-8, 8.9e-8,
  2.2e-7; 2412 6.5e-7, 2.2e-7, 4.4e-8; 4412 1.49e-6, 3.3e-7, 3.8e-7; 23012 4.2e-7, 2.3e-9,
  3.1e-8 (alpha -4, 4, 10). The two cores assemble differently arranged linear systems
  (NumPy: wrapped nodal strengths plus one circulation; JAX: N+1 nodal strengths).

Decision (after M1 review): both tolerances stay unchanged; both tests are marked
`xfail(strict=True, raises=AssertionError)`, fixed in M2, with the measured value in the reason.
Only the failing parameter combinations are marked (Kármán-Trefftz at 4 deg; NumPy vs JAX for
NACA 4412 at -4 deg); the other combinations remain ordinary tests.

Kármán-Trefftz error of v1.3.1 (max |dCp| for x/c in 0.02-0.98; panels uniform in the circle
angle):

| Panels | alpha 0: max dCp | location | alpha 4: max dCp | location |
|---|---|---|---|---|
| 100 | 0.0210 | x/c 0.028, upper | 0.0379 | x/c 0.028, upper |
| 200 | 0.0111 | x/c 0.021, upper | 0.0220 | x/c 0.021, upper |
| 400 | 0.0060 | x/c 0.021, lower | 0.0112 | x/c 0.020, upper |

Observed order of convergence: 0.92 and 0.90 at 0 deg, 0.78 and 0.98 at 4 deg (100 to 200 and
200 to 400 panels), so first order. The maximum always sits just aft of the leading edge, at the
start of the compared band; aft of x/c = 0.10 the error is about half as large (0.0083 at 4 deg,
200 panels).

### Accuracy targets: approved

The proposed targets are approved as stated under "Proposed accuracy targets" (open trailing
edge, Re 5e5 to 6e6, attached-flow filter, met at every Reynolds number separately). They are
the acceptance criteria for M3.

### Decision 3: SPEC.md 4.1 wording approved and applied

Approved with two changes and applied to SPEC.md 4.1: "Mach 0" added to the reference recipe
after "free transition", and after the canonical-file sentence: "Each reference stores the
checksum of the canonical file it was computed from. A test fails if the file produced by the
current code no longer matches, and the references for that section are then regenerated."
That test is `tests/test_benchmark.py::test_canonical_file_matches_reference_checksum` (fast;
Clark Y and E387 skip when they are neither cached nor downloadable). The introductory sentence
of SPEC.md section 4 about *provisional* targets was not part of the approved wording and is
unchanged.

Plots: the Cd axis of the Cl-Cd panel is limited to 0-0.03 so that the attached range is
readable; the plots were redrawn from the stored results without rerunning the solvers.

### Removed tests

Removed together with the two legacy reference files, `foil_lab/validation.py`,
`examples/validation_report.py` and `validation_output.txt` (Q9), after the benchmark and the
exact-solution tests were in place:

| Test (`tests/validation/test_reference_polars.py`) | Reason |
|---|---|
| `test_inviscid_matches_thin_airfoil_theory` | Compared a 12 % thick section with the flat-plate formula, which is not a valid reference. Replaced by the Kármán-Trefftz exact-solution test. |
| `test_reference_data_loads` | Loaded downloaded third-party data (airfoiltools.com). Replaced by `test_committed_reference_complete_with_provenance` on our own XFOIL references. |
| `test_inviscid_zero_lift_angle_physical` | Used the third-party file only for its angle list. The same check remains in `test_naca_polars.py::test_zero_lift_angles` and `test_solver.py::test_cambered_zero_lift_angle`. |
| `test_documented_low_re_vii_limitation` | Pinned a failure on third-party data at Re 5e4. Low-Re convergence of every v1.3.1 method is now reported by the benchmark. |

The validation sections of README.md, VALIDATION.md and TESTING.md were reduced to pointers to
the benchmark; they are rewritten from benchmark output in M9.

### Baseline result (summary; full tables in `docs/baseline_v1.3.1.md`)

Reference set: 2100 requested points, 1968 converged (37 only on the retry), 132 not converged.
Seven XFOIL processes reached the 300 s timeout in both generations (with and without CPU
contention), so these are genuine hangs.

Validated range (Re 5e5 to 6e6), open-trailing-edge sections, attached-flow filter, pooled:

| | points compared | not converged | mean dCl | max dCl | mean rel dCd | max rel dCd | mean dCm | max dCm |
|---|---|---|---|---|---|---|---|---|
| XFOIL 240 vs 160 | 325 | 0 | 0.0004 | 0.0029 | 0.4 % | 2.6 % | 0.0001 | 0.0005 |
| inviscid | 330 | 0 | 0.0678 | 0.3088 | - | - | 0.0117 | 0.0482 |
| one-way (default) | 330 | 0 | 0.0678 | 0.3088 | 70.3 % | 91.7 % | 0.0117 | 0.0482 |
| transpiration VII | 330 | 64 | 0.0446 | 0.2557 | 7.1 % | 22.3 % | 0.0093 | 0.0483 |

Transpiration VII per Reynolds number (open sections): converged 80.0 / 83.3 / 78.4 / 80.9 % at
Re 5e5 / 1e6 / 3e6 / 6e6; mean dCl 0.055 / 0.045 / 0.042 / 0.039; mean rel dCd 6.9 / 6.3 / 7.0 /
8.2 %; mean dCm 0.0114 / 0.0091 / 0.0088 / 0.0084.

### Proposed accuracy targets (approved, see "Accuracy targets: approved")

For sections with an open trailing edge, Re 5e5 to 6e6, inside the attached-flow filter, met at
every Reynolds number separately (not only pooled):

| Quantity | Target | v1.3.1 transpiration VII, worst Reynolds number |
|---|---|---|
| Converged points (of those where XFOIL converged) | >= 95 % | 78.4 % |
| mean abs dCl | <= 0.03 (unchanged) | 0.055 |
| max abs dCl | <= 0.10 (new) | 0.256 |
| mean rel abs dCd | <= 10 % (unchanged) | 8.2 % |
| max rel abs dCd | <= 30 % (new) | 22.3 % |
| mean abs dCm | <= 0.01 (unchanged) | 0.0114 |
| max abs dCm | <= 0.03 (new) | 0.048 |

Reasons:

- XFOIL's own paneling sensitivity (0.0004 in Cl, 0.4 % in Cd, 0.0001 in Cm) is far below the
  targets, so the targets are not limited by the reference.
- The provisional means are kept: v1.3.1 VII already meets the Cd and nearly the Cm mean, so they
  are reachable; Cl needs a real improvement.
- Maximum limits catch errors near the filter edge that means hide (v1.3.1 max dCl 0.26).
- The convergence requirement stops a method from looking better because bad points dropped out.
- Per Reynolds number, so that a good Reynolds number cannot hide a bad one.
- Transition location is reported without a target in v2.0. Sharp-trailing-edge sections and
  Re below 5e5 are reported without targets.

### Proposed SPEC.md 4.1 wording (awaiting approval; SPEC.md not edited)

Replace "Sections: NACA 0012, 2412, 4412, 23012, plus at least two non-NACA sections imported by
name." with:

> Sections: NACA 0012, 2412, 4412, 23012 (open trailing edge of NACA Report 460, thickness
> coefficient -0.1015), plus Clark Y and Eppler E387, downloaded by name from the UIUC Airfoil
> Coordinates Database.

Replace "Both codes receive the same coordinate file." with:

> Both codes read the same canonical coordinate file: 160 panels, Selig order.
>
> Reference recipe: XFOIL 6.99 runs under `xvfb-run` with graphics enabled. It loads the
> canonical file and redistributes it with `PANE` at 160 nodes (the reference) and at 240 nodes
> (paneling sensitivity), N_crit 9, free transition, `ITER 200`, 300 s timeout per process. Each
> case is two fresh XFOIL processes, 0 to +20 deg and 0 to -4 deg.

Replace the provenance bullet with:

> Each reference polar is stored with its provenance: XFOIL version and package release, `PANE`
> node count, N_crit, Reynolds number, Mach number, geometry source and checksum, trailing-edge
> gap, date, and the status and run time of every XFOIL process.

Replace "Angle of attack from -4° up to the last angle at which XFOIL converges, in 1° steps."
with:

> Angle of attack from -4 deg to +20 deg in 1 deg steps. Every angle where XFOIL converges is
> kept. A skipped angle is retried once from a fresh start directly at that angle and flagged as
> retried; an angle that still fails is recorded as not converged and never interpolated.

Add to "Reported per case":

> the number of angles each code did not converge on, and the paneling sensitivity (XFOIL at 240
> versus 160 nodes).

Add to the error-metrics bullet:

> Points where either code did not converge are excluded from the metrics and counted next to
> them.

Replace the provisional-targets bullet with:

> Attached-flow filter: angles from -4 deg up to, but not including, the first positive angle at
> which XFOIL's Cl reaches 0.9 times its maximum over the positive-angle sweep.
>
> Targets, for sections with an open trailing edge, Re 5e5 to 6e6, inside the attached-flow
> filter, met at every Reynolds number separately: at least 95 % of the points where XFOIL
> converged are converged; mean |dCl| <= 0.03 and max <= 0.10; mean relative |dCd| <= 10 % and
> max <= 30 %; mean |dCm| <= 0.01 and max <= 0.03.
>
> Sections with a sharp trailing edge are reported separately without targets; for them the
> paneling sensitivity is a lower bound, because XFOIL's result keeps changing with the number of
> `PANE` nodes.

---

## M2: Geometry model and one inviscid core

Plan approved on 2026-10-09. Work on branch `m2-inviscid-core` from `main` (`e7432de`).
SPEC.md section 4 introduction: the sentence about provisional targets is replaced by "The
benchmark always reports the measured numbers, whether or not a target is met." (approved).

### Approved plan, by point

1. Vorticity model: linear-strength vortex panels, nodal strengths at the N+1 nodes of every
   body, zero normal velocity at the N panel midpoints, one Kutta condition per body
   (strength at the first node plus strength at the last node = 0), no unknown sources. Each
   body closes on itself; bodies are coupled only through their mutual influence. Sources:
   Kuethe and Chow, Foundations of Aerodynamics, 5th ed. (1998), section 5.10; Katz and
   Plotkin, Low-Speed Aerodynamics, 2nd ed. (2001), section 11.4.2. The panel influence is
   checked against numerical quadrature of the induced velocity.
2. Open trailing edge: the two trailing-edge points are separate nodes with the usual Kutta
   condition and the gap is not panelled.
3. One assembly, written once against an array namespace and run with NumPy or JAX. The
   private assemblies in `vii.py`, `ground_effect.py` and `flowfield.py` and the old
   `influence.py`, `linear_system.py`, `forces.py` and `jax_core` assembly are replaced and
   deleted. The VII solver itself is not changed; its results before and after routing are
   reported from the benchmark.
4. Williams R&M 3717: option (a), see Q4.
5. Geometry model, arc-length repanelling, transforms, trailing-edge gap tool; Clark Y and E387
   references regenerated; E387 with a 0.25 % gap added as a section with targets.
6. Every test marked expected-failure for M2 ends as an ordinary passing test with its tolerance
   unchanged.

One stop in the middle: when the new core passes the Kármán-Trefftz and two-element tests,
show the Kármán-Trefftz errors and orders, the two-body and mirrored-pair numbers, NumPy
versus JAX, the core health checks, the Williams tolerances and the transcribed Williams tables
with page numbers and check results.

### Answers

- Q1. New Kármán-Trefftz assertion: observed order of convergence at least 1.5 at 0 and 4 deg,
  from 100, 200 and 400 panels, on the same error measure as the existing test (max |dCp| for
  x/c in 0.02-0.98). The RMS error and its order are recorded without a tolerance.
- Q2. Gap left unpanelled. The gap-to-zero test and the leak-through-the-gap test get fixed
  tolerances, stated at the stop before they are run on the benchmark sections. The
  trailing-edge gap tool warns above a gap of 1 % chord; the limit is documented. Default
  trailing edge of `generate_naca4`/`generate_naca5` becomes "open" (behaviour change for the
  changelog); the closed geometry stays pinned bit-identical under the explicit option.
- Q3. Ground effect moves onto the new core now (section plus mirror image, stream parallel to
  the ground, section rotated about c/4). If the two M6 ground-effect tests then pass, their
  marks are removed; the pivot option and the remaining M6 items stay in M6. Once ground effect
  uses the same core, the mirror comparison is only a consistency check; the no-penetration
  test on the ground plane is the one that counts, and the test docstrings say so.
- Q4. Option (a): implement Williams' exact method as a test helper from his equations and
  Table 1; the transcribed tables check both the helper and the transcription. Transcription:
  Table 1 in full, the force table on printed page 7, Table 3 "Exact" rows at flap 10 and 30
  deg, alpha 0, and Table 2(a)/(b) in full. Checks: Cp = 1.0000 at the trailing-edge rows;
  linear integration of the tabulated Cp reproduces Table 3 C_N and C_A within 1e-3; no
  outliers in second differences along each surface; the helper reproduces Table 2 Cp within
  1e-3. How the tables were read is stated and uncertain digits are marked. If a transcribed
  value disagrees with the helper, neither is changed: page, row, both values and neighbouring
  rows are shown. The transcribed tables are committed as test data with the full citation.
- Q5. Williams test, configurations A and B at alpha 0, 200 panels per element: Cl within 1 %
  per element and in total of Table 3; max |dCp| <= 0.05 for x/c 0.02-0.98 of each element;
  error decreasing from 200 to 400 panels. Restated in full at the stop before the first run.
- Q6. The approximate 6-series generator is removed. NACA 4/5 are checked by analytic
  properties; no transcribed ordinate tables. One cross-check in the slow xfoil group: our open
  NACA 2412 against coordinates saved from XFOIL's own `NACA` command, as a distance between the
  curves, with the tolerance stated before running. The other generators stay and return an
  `Airfoil`.
- Q7 to Q11: answered on 2026-10-09, see "Answers Q7-Q11" below.

### Additions to the plan

- Core health checks at the stop (NACA 2412 open and closed at 160 and 320 panels, and the
  Kármán-Trefftz section): the vortex strength along the surface must be free of panel-to-panel
  sawtooth, especially near the trailing edge; the matrix condition number is reported. If there
  is a sawtooth, stop and propose alternatives from published sources.
- After routing, the GUI must still start (offscreen smoke test) and the CLI, batch and design
  paths must run. The benchmark adapter must still work (M3 needs it).
- The JAX path must not change global JAX settings on import without documenting it.

### Step 1: the new core (`foil_lab/inviscid.py`)

- Linear-strength vortex panels as planned. The panel velocity is evaluated in closed form
  (derivation in the docstring) and matches numerical quadrature of the Biot-Savart integral to
  about 1e-16 (`tests/test_inviscid_core.py`). Assembly through fixed start-node and end-node
  incidence matrices; the same code runs with NumPy and with `jax.numpy`.
- Surface-velocity read-out: on the Kármán-Trefftz section the induced-velocity sum at the
  panel midpoints converges only at first order (max |dCp| 0.0214 at 200 panels, 4 deg), while
  the nodal vortex strengths converge at second order (circulation lift errors 7.6e-4, 1.8e-4,
  4.5e-5 at 100/200/400 panels). Because the flow inside a closed body is at rest, the surface
  speed equals the local sheet strength (Katz and Plotkin 2001, section 11.4.2); taking the mean
  of the two nodal strengths at each midpoint gives second order. That read-out is used; the
  induced-velocity value is kept as a diagnostic.
- Transpiration: a pure vortex sheet cannot carry a net outflow from a closed body, so a
  prescribed wall-normal velocity enters as a known constant-strength source per panel equal to
  that velocity (equivalent source of Lighthill 1958). The flow inside the body then stays at
  rest and the read-out above remains valid. Source kernel checked against quadrature.
- JAX: the solve runs inside `jax.enable_x64(True)`, a scoped setting; the global setting is not
  changed (tested).

Results on the new core (closed trailing edges; the open edge is the subject of the next item):

| Check | Result | Tolerance |
|---|---|---|
| Kármán-Trefftz 0 deg, max dCp at 100/160/200/320/400 panels | 0.00618 / 0.00266 / 0.00190 / 0.00077 / 0.00052 | 0.02 at 200 |
| Kármán-Trefftz 4 deg, same | 0.00605 / 0.00257 / 0.00187 / 0.00076 / 0.00051 | 0.02 at 200 |
| Observed order of max dCp (100-200 / 200-400), 0 deg and 4 deg | 1.70 / 1.87 and 1.69 / 1.87 | >= 1.5 |
| Observed order of RMS dCp, 0 deg and 4 deg (no tolerance) | 1.85 / 1.94 and 1.85 / 1.94 | - |
| Location of max dCp | x/c 0.976-0.979, upper surface | - |
| NACA 0012 + 4412, 50 c stacked | -0.178 % / +0.111 % | 0.5 % |
| Mirrored pair Cl1 + Cl2 | 5.1e-14 | 1e-9 |
| NumPy vs JAX, 4 sections x 3 angles | max 6.7e-16 | 1e-6 |

Health checks (alpha 5 deg; Kármán-Trefftz at 4 deg):

| Case | Condition number | Trailing-edge node strengths | Sawtooth (alternating second difference) |
|---|---|---|---|
| NACA 2412 closed, 160 | 4.4e4 | -0.544 / +0.544 | none |
| NACA 2412 closed, 320 | 1.2e5 | -0.641 / +0.641 | none |
| Kármán-Trefftz, 200 | 9.3e4 | -0.427 / +0.427 | none |
| NACA 2412 open, 160 | 7.6e2 | +3.44 / -3.44 | none, but singular |
| NACA 2412 open, 320 | 1.0e3 | +6.80 / -6.80 | none, but singular |

- Closed trailing edge: smooth strength everywhere. The two coincident trailing-edge nodes carry
  a single-node value of opposite sign to their neighbours; the last panel on each side reads
  Cp = +0.99 to +1.00, i.e. the stagnation point of a finite-angle trailing edge. Loads converge:
  NACA 2412, 5 deg, Cl 0.86143 / 0.86163 / 0.86168 / 0.86169 and Cm c/4 -0.06272 / -0.06268 /
  -0.06267 / -0.06267 at 160 / 320 / 640 / 1280 panels; pressure and circulation lift agree to
  1e-4 or better.
- Open trailing edge (gap left unpanelled, as approved in Q2): the trailing-edge node strengths
  double with every refinement (3.44, 6.80, 13.56, 27.09 at 160 to 1280 panels), the last-panel
  Cp diverges (-4.2, -17.7, -72.0, -289.0), and the flow crosses the gap at about 0.17 of the
  freestream speed. The integrated Cl still converges (0.86358, 0.86378, 0.86381, 0.86380) only
  because the singular panel shrinks. The approved treatment is therefore not acceptable; work
  stopped here for a decision (rule: stop and propose published alternatives).

Proposed alternatives for the open trailing edge (awaiting decision):

- (A) A straight panel across the gap carrying a constant source and a constant vortex whose
  strengths are tied to the two trailing-edge nodal strengths, so no new unknowns; the Kutta
  condition is unchanged. Published in M. Drela, "XFOIL: An Analysis and Design System for Low
  Reynolds Number Airfoils", in T. J. Mueller (ed.), Low Reynolds Number Aerodynamics, Lecture
  Notes in Engineering 54, Springer (1989), pp. 1-12. Implemented from the paper only, in the
  present velocity formulation.
- (B) A Dirichlet (zero internal potential) formulation, Katz and Plotkin (2001) chapter 11,
  with the base region closed and the Kutta condition through the wake strength; a larger change
  that would need the Kármán-Trefftz order re-checked.
- (C) Close the gap geometrically for the solver only. Not recommended: the benchmark would no
  longer compare the same geometry with XFOIL.

Decision (2026-10-09): option (A), implemented from Drela (1989) only, with the equations and
page numbers recorded here; XFOIL source is not opened. If (A) fails its checks, stop before
trying (B). The plan change (surface speed from the sheet strength; transpiration as equivalent
sources) is accepted. Blocked until the paper is supplied: the equations are not implemented
from memory.

Fixed tolerances for the trailing-edge checks (stated before running; NACA 2412, alpha 5 deg;
gaps made by opening the closed section linearly in x, a test-local construction):

- Gap to zero: at gaps 0.0025 / 0.001 / 0.0003 / 0.0001 c (320 panels) |dCl| and |dCm| to the
  closed section fall monotonically; at 0.0001 c, |dCl| <= 0.002 and |dCm| <= 0.0005.
- Flow through the gap: mean normal velocity across the gap <= 0.02 of the freestream (320
  panels).
- Trailing-edge Cp: |Cp| on the last panel of each side <= 1.1 at every N from 160 to 1280, and
  its change from 640 to 1280 panels <= 0.02.
- Loads: |Cl(N) - Cl(1280)| <= 0.005 at N = 160 and <= 0.002 at N >= 320; |Cm(N) - Cm(1280)|
  <= 0.001 at N = 160 and <= 0.0005 at N >= 320; successive changes decrease.
- No sawtooth: no sign alternation of the second difference of the nodal strength in the 5 % of
  nodes nearest each trailing edge; trailing-edge node strength at 1280 panels <= 1.1 times its
  value at 160.

Inviscid XFOIL cross-check (slow, xfoil group): tighter than proposed, set by the user before the
run: open NACA 2412, canonical 160-panel file, 0 / 5 / 10 deg, XFOIL without VISC at PANE 160
against the core at 160 panels, |dCl| <= 0.005 and |dCm| <= 0.002. Differences at 320 (canonical
320-panel file, PANE 320) are reported without a tolerance.

### Open trailing edge: option (A), implemented

Source: M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils", in
T. J. Mueller (ed.), *Low Reynolds Number Aerodynamics*, Lecture Notes in Engineering 54,
Springer (1989), pp. 1-12 (`review-gui/drela-1989-xfoil.pdf`, a 12-page scan; read from page
images). XFOIL source was not opened.

Equations used, with their place in the paper:

- Eq. (2), page 2, and the trailing-edge detail of Figure 1, page 2: a panel across the gap with
  uniform source and vortex strengths tied to the surface vorticity,
  sigma_TE = 1/2 (gamma_1 - gamma_N) |s x t|, gamma_TE = 1/2 (gamma_1 - gamma_N) |s . t|,
  where s bisects the trailing-edge angle and t is the unit vector along the panel.
- Eq. (3), page 2, last term: the panel enters the influence sum through (gamma_1 - gamma_N)
  only, so it adds no unknowns.
- Eq. (8), page 3: Kutta condition gamma_1 + gamma_N = 0 (unchanged from step 1).

Not used: the streamfunction kernels (eqs. 4-6) and the nodal streamfunction condition (eq. 7),
page 3. The core keeps its midpoint Neumann condition and its own closed-form kernels; only the
trailing-edge panel is taken from the paper.

Our sign convention: nodes run clockwise, so 1/2 (gamma_first - gamma_last) is the mean speed q
of the flow leaving the trailing edge along s (the same expression as eq. 2). The panel runs from
the last node to the first, and with the flow inside the body at rest the jump conditions across
the panel give sigma_TE = q (s . n) and gamma_TE = -q (s . t) in this code's vortex sign, whose
magnitudes are those of eq. (2); a test checks q = 1/2 (gamma_first - gamma_last) and
sigma_TE > 0 (outflow through the base). A body whose gap is below 1e-10 of its size has no
trailing-edge panel, so closed results are unchanged bit for bit. Loads are integrated over the
body panels only (no base pressure); the circulation includes the panel's vortex.

Plan change (accepted 2026-10-09), with citations: the surface speed is the local sheet strength,
because the flow inside the body is at rest (Katz and Plotkin 2001, section 11.4.2); wall
transpiration enters as known equivalent sources, outward normal velocity equal to the source
strength (M. J. Lighthill, "On displacement thickness", J. Fluid Mech. 4 (1958) 383-392).

Results against the fixed tolerances (NACA 2412, alpha 5 deg):

| Check | Result | Tolerance |
|---|---|---|
| Gap to zero, 320 panels: |dCl| at 0.0025 / 0.001 / 0.0003 / 0.0001 c | 1.79e-3 / 6.1e-4 / 7.3e-5 / 3.9e-5 | falling; <= 2e-3 at 1e-4 |
| Gap to zero: |dCm| | 7.2e-4 / 2.7e-4 / 5.5e-5 / 3.7e-6 | falling; <= 5e-4 at 1e-4 |
| Flow through a 0.25 % gap (inner side, 9 points) | mean 0.0044, max 0.0094 | mean <= 0.02 |
| Last-panel Cp, 160-1280 panels | 0.435 to 0.458 | <= 1.1 |
| Last-panel Cp change 640 -> 1280 | 0.0034 | <= 0.02 |
| |Cl(N) - Cl(1280)| at 160 / 320 / 640 | 2.8e-4 / 7.0e-5 / 1.5e-5 | 5e-3 / 2e-3 / 2e-3 |
| |Cm(N) - Cm(1280)| at 160 / 320 / 640 | 4.2e-5 / 9e-6 / 2e-6 | 1e-3 / 5e-4 / 5e-4 |
| Successive changes, Cl / Cm | 2.1e-4, 5.5e-5, 1.5e-5 / 3.3e-5, 7.6e-6, 1.7e-6 | decreasing |
| Trailing-edge node strength 1280 vs 160 | 0.7370 vs 0.7342 | <= 1.1 times |
| Sawtooth (sign alternation of the second difference) | none | none |

Before the gap panel (step 1) the same section had trailing-edge strengths doubling with N,
last-panel Cp down to -289 and a gap flow of about 0.17. Matrix condition number with the panel:
2.8e2-3.8e2.

Reading of "sign alternation": in each 5 % segment the second difference changes sign exactly
once, a single smooth inflection (at 1280 panels: +4.9e-3, +4.5e-4, +1.5e-4, +4.6e-5, +6e-6,
then negative and smooth). The test treats a sawtooth as two consecutive sign flips (+-+ or
-+-), which never occurs. Reported to the user, not silently adopted.

Inviscid XFOIL cross-check, Cl / Cm differences (core minus XFOIL):

| alpha | 160 vs PANE 160 | 320 vs PANE 320 | XFOIL PANE 240 minus PANE 160 |
|---|---|---|---|
| 0 | +0.0007 / -0.0001 | +0.0004 / -0.0001 | +0.0004 / -0.0001 |
| 5 | +0.0009 / -0.0003 | +0.0005 / -0.0002 | +0.0006 / -0.0001 |
| 10 | +0.0010 / -0.0004 | +0.0006 / -0.0002 | +0.0008 / -0.0001 |

All within |dCl| <= 0.005 and |dCm| <= 0.002. The 0.832-0.858 spread of XFOIL's inviscid Cl with
PANE count mentioned at the M2 stop came from an untrusted scratch parse in M1 and is not
reproduced.

### Williams R&M 3717: transcription and exact helper

Transcription (committed as `tests/data/williams/*.csv` with the full citation): Table 1, the
force table of section 4.1, Table 3 "Exact" rows for flap 10 and 30 deg at alpha 0, and Table
2(a)/(b) in full (main A 61 rows, flap A 61, main B 62, flap B 61). Read from 300-dpi renders of
the scanned page images (the PDF has no text layer), every column in close-up; no digit is
uncertain. Table 2 and Table 1 pages are PDF pages 20-24 (printed 19-23), Table 3 PDF page 25
(printed 24), the force table PDF page 8 (printed 7).

Checks on the transcription:

- The trailing-edge row of each element has Cp = 1.0000.
- Second differences of xi and eta along each surface stay within 5 times their median: no
  outlier rows. Cp second differences are large only at the leading-edge suction peaks and the
  trailing-edge jump to 1.0, which are features of the distribution.
- Linear integration of the tabulated Cp around each element against Table 3 (C_N, C_A):
  main A -8.7e-3 / -2.2e-3, flap A -1.0e-3 / -7.8e-4, main B -2.2e-3 / -1.6e-4, flap B -5.7e-4 /
  -3.2e-4. The 1e-3 criterion is not met for the main elements. Integrating the same points with
  a cubic spline in arc length instead gives main A -3.0e-3 / -2.4e-3, flap A +6.6e-4 / +2.4e-4,
  main B +1.6e-3 / +4.4e-4, flap B +9.7e-5 / +1.4e-5: with about 61 points the integral itself
  is uncertain by a few 1e-3 at the sharp leading-edge peaks, so this check cannot confirm or
  rule out the main-element transcription to 1e-3. The flaps agree within 1e-3.
- Frames: the stagnation points of Table 1(c)/(d) are given in the centre-line frame (both
  circle centres on the real axis); rotated by the declination of the second centre (-11.888
  deg) they lie at -beta1 on the first circle and at -(beta2 + flap) on the second, exactly as
  page 6 states. The first-circle centre is (c1 - a cos beta1, +a sin beta1); the page-6 formula
  for the second centre has -a sin beta1, which is inconsistent with Table 1 at the 2e-3 level.

Exact helper (option (a)): the image series (eqs. 1-9, Appendix I), the Kutta condition (eqs.
10-13) and the two Kármán-Trefftz maps (eq. 14) were implemented. Result:

- Stagnation points reproduce Table 1 exactly; circulations are 1/(2 pi) of Table 1 within 2-4 %
  for the main element and 1-2 % for the flap (Table 1 uses 2 pi times the unit circulation of
  eq. 8).
- Geometry does not reproduce Table 2. Section 3.2 does not say how the second transformation's
  radius and axis are defined for the (no longer circular) image of the second circle. Three
  readings were tried (c2 = b/k2 with the flap rotation; image of the centre; least-squares circle
  through the image). Best rms distance of the tabulated main points to the helper curve 2.0e-3
  (B) to 2.4e-3 (A); the flap shape alone fits to 6.5e-3 at best. The tabulated main aerofoil does
  not match the first Kármán-Trefftz profile alone either (rms 2.0e-3 for A, 5.0e-3 for B), and
  the two tabulated mains differ from each other (t/c 0.1538 and 0.1516), so the second
  transformation reshapes the main aerofoil in a way the report does not specify fully.
- Cp therefore differs by order 1 near the trailing edges. Per the agreed rule, option (a) is
  stopped and option (b) is proposed; neither the helper nor the transcription was changed to
  make them agree. The helper is not committed.

Decision (2026-10-09): the exact helper failed for the reasons above and is kept out of the
repository. Option (b) is approved with the Q5 tolerances unchanged: the tabulated Table 2
geometry is panelled directly. Recorded with it: the page 6 sign inconsistency (-a sin beta1 in
the formula for the second centre, +a sin beta1 consistent with Table 1), and that integrating
the tabulated Cp cannot resolve 1e-3 with about 61 points per element. Rules: a single tabulated
point that disagrees while its neighbours agree is shown to the user (file, row, both values,
neighbours, page) and the CSV is not edited; if the 200 -> 400 criterion fails, the numbers and
the evidence for the cause are shown and the criterion is not changed.

Option (b) results (`tests/test_williams.py`, `tests/williams_tables.py`; cubic spline in arc
length through [TE, rows 1..n-1, TE], cosine clustering at both edges, Cp interpolated in arc
length to the tabulated points; Cl per element by pressure integration, total from circulation
against CL**):

| Config | N | Cl main (T3) | Cl flap (T3) | Total (CL**) | max dCp main | max dCp flap |
|---|---|---|---|---|---|---|
| A | 200 | 2.9040 (2.9065), -0.09 % | 0.8288 (0.8302), -0.17 % | 3.7321 (3.7386), -0.17 % | 0.0222 | 0.0108 |
| A | 400 | 2.9039, -0.09 % | 0.8288, -0.17 % | 3.7325, -0.16 % | 0.0247 | 0.0107 |
| A | 800 | 2.9038 | 0.8288 | 3.7326 | 0.0255 | 0.0108 |
| B | 200 | 1.6877 (1.6915), -0.23 % | 0.3350 (0.3366), -0.48 % | 2.0222 (2.0290), -0.33 % | 0.0094 | 0.0076 |
| B | 400 | 1.6876, -0.23 % | 0.3349, -0.50 % | 2.0224, -0.33 % | 0.0121 | 0.0074 |
| B | 800 | 1.6875 | 0.3349 | 2.0224 | 0.0140 | 0.0078 |

Cl (1 %) and max |dCp| (0.05) pass. The criterion "error falls from 200 to 400 panels" FAILS for
both main elements (flaps: 0.0108 -> 0.0107 and 0.0076 -> 0.0074, pass). Not changed; the test is
left uncommitted pending the user's decision. Evidence for the cause:

1. The core converges on this geometry: max |Cp_N - Cp_3200| in the window is 2.1e-2 / 5.7e-3 /
   1.2e-3 / 4.1e-4 at N = 200 / 400 / 800 / 1600 (config A main), about second order. The
   difference to the table levels off at a nonzero floor (0.026 for A, 0.015 for B), so the
   error at 200 panels is lower only where discretisation error happens to offset it.
2. One printed point: Table 2(a), main aerofoil, row 17 (`table2a_main.csv`; printed page 20,
   PDF page 21): xi 0.41297, eta -0.07214, Cp 0.58264. Re-read on the page image: printed
   exactly so. Its eta is 9.0e-4 off a quartic through its eight neighbours (other mid-chord rows
   1e-4 or less; rounding 5e-6); the fit gives -0.07124, the printed value with two digits
   swapped. Our converged Cp at rows 16 / 17 / 18 is 0.59070 / 0.55679 / 0.60222 against the
   tabulated 0.58115 / 0.58264 / 0.59271 (table minus ours -0.0096 / +0.0259 / -0.0095), while
   rows 15 and 19 agree to 1.1e-3: the tabulated Cp is smooth, the bump in ours comes from the
   tabulated geometry. Likely a misprint in the original. Table 2(b) main row 16 (printed page
   22, PDF page 23; eta -0.06936, also printed so) is 2.7e-4 off the same fit (-0.06963), with a
   smaller Cp zigzag (+0.0066 / -0.0114 / +0.0073 at rows 15-17). The CSVs are unchanged.
3. With only those two eta values replaced in memory by the fit (evidence only), max |dCp| for
   the main elements still rises: A 0.0084 -> 0.0130 -> 0.0163, B 0.0073 -> 0.0121 -> 0.0140
   (200 / 400 / 800). The maximum then sits on the upper surface just behind the leading edge
   (A rows 34-35, x/c 0.027-0.046, Cp about -5.7 to -4.7; B row 34, x/c 0.026, Cp -2.77),
   where about 61 points do not fix the shape of the curvature peak.
4. Geometry sensitivity: building the spline through every other tabulated point changes the
   converged Cp by up to 0.059 (A main), 0.037 (B main), 0.017-0.020 (flaps), more than the
   floor in item 1. The floor is set by reconstructing the geometry from the table, which
   refinement of our panels cannot reduce.

### Step 3: routing through the core

Everything that solved a potential flow now goes through `foil_lab.inviscid`:

- `solver.solve_flow`: the core, then the unchanged compressibility, 3D, one-way drag and
  post-stall layers. `SolutionData` keeps its fields; `gamma` now holds the core's nodal
  strengths, `sources` the known panel sources (VII transpiration, else zeros), and a new
  `panels` field carries the core's panel set. `influence_model` is removed (one vorticity
  model). `compute_panel_geometry` keeps its 13-tuple, in the same clockwise order.
- `vii`, `vii_newton`, `vii_qs`: `_inviscid_with_transpiration` calls the core with the
  transpiration as known equivalent sources (outer normal velocity = v_n, the same condition the
  old right-hand side imposed). The matrices are assembled once per geometry
  (`inviscid.assemble`) and reused across iterations; the iteration schemes, relaxation,
  clipping and smoothing are unchanged (VII itself is M3).
- `flowfield`: velocities from the solution's own singularities (`inviscid.induced_velocity`,
  including sources and the trailing-edge panel), evaluated in blocks; points inside a body are
  found by ray casting on the body polygon (was a bounding box).
- `ground_effect`: the section and its mirror image about y = 0, solved as two bodies by the
  core in a stream parallel to the ground (Q3 option (a)). Behaviour change: the section is
  pitched by alpha about its quarter-chord point and raised by `height` (quarter-chord point);
  v1.3.1 inclined the stream instead, so the stream crossed the wall. The returned `alpha_deg`
  is the stream direction in the solution frame (0); `coords_list` holds the pitched, raised
  section; loads are in ground axes, moment about the raised quarter-chord point.
- `jax_core`: thin wrappers over the core with `xp = jax.numpy` (same names and signatures).
  The import-time `jax.config.update("jax_enable_x64", True)` of v1.3.1 is removed: every public
  function runs inside the scoped `jax.enable_x64(True)` and takes NumPy arrays, so JAX's global
  configuration is never changed (test added). Callers (`design`, `examples/jax_demo.py`,
  `tests/test_jax.py`) pass NumPy arrays; a JAX float32 array built outside the scope would lose
  precision.
- Benchmark adapter: panel midpoints from `inviscid.PanelSet` (same order).
- GUI: the streamline plot draws `sol.coords_list` (for ground effect the pitched, raised
  section) instead of the unsolved input.
- Deleted: `influence.py`, `linear_system.py`, `forces.py` (approved), and `panels.py`, which
  nothing used any more (addition).
- Quarter-chord moment: now the exact moment about (0.25, 0); v1.3.1 used cm_le + 0.25 cl,
  which mixes lift and normal force (differs by 0.25 (cl - c_n), about 4e-3 at Cl 1.5 and 10
  deg). Recorded as a behaviour change.

Tests rewritten or removed:

- `test_tier1.py`: `test_both_kernels_correct_lift` and `test_kernels_agree_closely` removed (they
  compared the two removed kernels). The two multi-element tests now use the core; the
  independent-circulation test also checks that the two circulations differ by more than 1e-3.
- `test_solver.py`: Kutta-Joukowski check uses the core's circulation (tolerance unchanged);
  the clockwise check uses `PanelSet`.
- `test_vii_newton.py`: sensitivity-matrix test with the new signature (tolerance unchanged).
- `test_exact_solutions.py`: two-body and circulation helpers use the core. Expected-failure
  marks removed because the tests now pass with unchanged tolerances: Kármán-Trefftz at 4 deg,
  stacked two-body, mirrored pair, far-field lift, NumPy/JAX for NACA 4412 at -4 deg (M2), and
  both ground-effect tests (M6; per Q3 the no-penetration test is the one that counts). Still
  marked: repanelling of scaled/translated/rotated shapes (M2 step 4) and the aspect-ratio
  compressibility test (M6).
- `test_jax.py`: NumPy inputs; new test that `jax_core` leaves the global x64 setting unchanged.

Failing after routing, not changed (decision needed): four VII tests on the closed NACA 2412
(`test_vii.py`: converges_attached, reduces_lift_vs_inviscid, drag_physically_plausible;
`test_transition.py`: vii_en_converges). Cause: on a closed (sharp, finite-angle) trailing edge
the core gives the correct inviscid stagnation point (edge velocity 0.02 on the last panels,
Cp +0.96 to +1.0); v1.3.1's Hess-Smith core gave 0.66 there. The v1.3.1 VII marches the
boundary layer to the trailing edge, meets that deceleration in the last panel and does not
converge (residual 1.5e-3 to 2.1e-3 after 200 iterations at 4 deg), with Cl above inviscid and
Cd about 1e-4. Away from the trailing edge the response of the edge velocity to transpiration
agrees with v1.3.1 to about 1 % (single-panel blowing at mid-chord: -4.73e-3 / 7.0e-5 /
4.87e-3 against -4.72e-3 / 7.0e-5 / 4.86e-3). On the open trailing edge (the benchmark files and
the approved step-4 generator default) VII converges in about 90 iterations: NACA 2412, 4 deg,
Re 1e6, Cl 0.644 (inviscid 0.743), Cd 0.0071. The one-way drag on closed sections is affected in
the same way (CLI, NACA 2412, 5 deg, Re 1e6: Cd 0.00069).

Smoke checks after routing: CLI analyze / polar / info run (VII on the closed default section
shows the problem above); GUI under Xvfb runs single analysis (inviscid, ground effect, VII)
and polar; the streamline view of ground effect keeps the GUI's fixed window (y within 0.5) and
also shows the mirror flow below the wall (cosmetic, GUI follow-up); inverse-design example and
JAX example run (JAX vs NumPy 7e-14).

Benchmark before / after (before: stored M1 run of v1.3.1; after: the same adapter on the routed
code, same canonical files, 6 workers, one BLAS thread each; output in
`benchmark_output/m2_after/`, not committed). Errors against viscous XFOIL at PANE 160, attached
filter as in M1; "not conv." = excluded / compared.

### open trailing edge, by Re

| Re | method | not conv. before | mean dCl before | mean rel dCd before | mean dCm before | max dCl before | not conv. after | mean dCl after | mean rel dCd after | mean dCm after | max dCl after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50000 | inviscid | 0/63 | 0.2119 | - | 0.0236 | 0.5136 | 0/63 | 0.2148 | - | 0.0244 | 0.5227 |
| 50000 | oneway | 0/63 | 0.2119 | 75.4 % | 0.0236 | 0.5136 | 0/63 | 0.2148 | 76.3 % | 0.0244 | 0.5227 |
| 50000 | vii | 28/63 | 0.1563 | 28.8 % | 0.0251 | 0.3057 | 25/63 | 0.1743 | 31.2 % | 0.0267 | 0.4023 |
| 50000 | newton | 8/9 | 0.1765 | 17.7 % | 0.0378 | 0.1765 | 7/9 | 0.2175 | 46.0 % | 0.0350 | 0.2957 |
| 100000 | inviscid | 0/67 | 0.0982 | - | 0.0154 | 0.2477 | 0/67 | 0.1008 | - | 0.0162 | 0.2565 |
| 100000 | oneway | 0/67 | 0.0982 | 69.4 % | 0.0154 | 0.2477 | 0/67 | 0.1008 | 70.6 % | 0.0162 | 0.2565 |
| 100000 | vii | 24/67 | 0.1247 | 20.2 % | 0.0236 | 0.2020 | 22/67 | 0.1239 | 19.7 % | 0.0237 | 0.2074 |
| 200000 | inviscid | 0/70 | 0.0654 | - | 0.0116 | 0.2973 | 0/70 | 0.0680 | - | 0.0123 | 0.3070 |
| 200000 | oneway | 0/70 | 0.0654 | 64.9 % | 0.0116 | 0.2973 | 0/70 | 0.0680 | 66.2 % | 0.0123 | 0.3070 |
| 200000 | vii | 19/70 | 0.0873 | 7.6 % | 0.0173 | 0.1471 | 20/70 | 0.0840 | 7.5 % | 0.0169 | 0.1420 |
| 200000 | newton | 5/10 | 0.0840 | 3.6 % | 0.0158 | 0.1275 | 5/10 | 0.1010 | 6.4 % | 0.0220 | 0.1928 |
| 500000 | inviscid | 0/70 | 0.0555 | - | 0.0108 | 0.2238 | 0/70 | 0.0591 | - | 0.0115 | 0.2335 |
| 500000 | oneway | 0/70 | 0.0555 | 63.1 % | 0.0108 | 0.2238 | 0/70 | 0.0591 | 64.4 % | 0.0115 | 0.2335 |
| 500000 | vii | 14/70 | 0.0552 | 6.9 % | 0.0114 | 0.1311 | 13/70 | 0.0553 | 8.8 % | 0.0117 | 0.2138 |
| 1000000 | inviscid | 0/78 | 0.0625 | - | 0.0113 | 0.2717 | 0/78 | 0.0673 | - | 0.0125 | 0.2834 |
| 1000000 | oneway | 0/78 | 0.0625 | 65.9 % | 0.0113 | 0.2717 | 0/78 | 0.0673 | 67.0 % | 0.0125 | 0.2834 |
| 1000000 | vii | 13/78 | 0.0445 | 6.3 % | 0.0091 | 0.1235 | 16/78 | 0.0385 | 6.0 % | 0.0080 | 0.1162 |
| 3000000 | inviscid | 0/88 | 0.0745 | - | 0.0124 | 0.3048 | 0/88 | 0.0802 | - | 0.0145 | 0.3183 |
| 3000000 | oneway | 0/88 | 0.0745 | 73.1 % | 0.0124 | 0.3048 | 0/88 | 0.0802 | 74.0 % | 0.0145 | 0.3183 |
| 3000000 | vii | 19/88 | 0.0422 | 7.0 % | 0.0088 | 0.2557 | 15/88 | 0.0404 | 8.5 % | 0.0088 | 0.2435 |
| 6000000 | inviscid | 0/94 | 0.0750 | - | 0.0120 | 0.3088 | 0/94 | 0.0811 | - | 0.0146 | 0.3233 |
| 6000000 | oneway | 0/94 | 0.0750 | 76.8 % | 0.0120 | 0.3088 | 0/94 | 0.0811 | 77.6 % | 0.0146 | 0.3233 |
| 6000000 | vii | 18/94 | 0.0390 | 8.2 % | 0.0084 | 0.2434 | 16/94 | 0.0344 | 9.2 % | 0.0076 | 0.2454 |

### open trailing edge, validated range pooled (Re 5e5-6e6)

| Re | method | not conv. before | mean dCl before | mean rel dCd before | mean dCm before | max dCl before | not conv. after | mean dCl after | mean rel dCd after | mean dCm after | max dCl after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | inviscid | 0/330 | 0.0678 | - | 0.0117 | 0.3088 | 0/330 | 0.0729 | - | 0.0134 | 0.3233 |
| pooled | oneway | 0/330 | 0.0678 | 70.3 % | 0.0117 | 0.3088 | 0/330 | 0.0729 | 71.3 % | 0.0134 | 0.3233 |
| pooled | vii | 64/330 | 0.0446 | 7.1 % | 0.0093 | 0.2557 | 60/330 | 0.0414 | 8.2 % | 0.0089 | 0.2454 |

### E387 (sharp), by Re

| Re | method | not conv. before | mean dCl before | mean rel dCd before | mean dCm before | max dCl before | not conv. after | mean dCl after | mean rel dCd after | mean dCm after | max dCl after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50000 | inviscid | 0/11 | 0.2780 | - | 0.0128 | 0.3440 | 0/11 | 0.2132 | - | 0.0229 | 0.3201 |
| 50000 | oneway | 0/11 | 0.2780 | 74.8 % | 0.0128 | 0.3440 | 0/11 | 0.2132 | 395087977.8 % | 0.0229 | 0.3201 |
| 50000 | vii | 3/11 | 0.2581 | 55.8 % | 0.0232 | 0.4130 | 6/11 | 0.1942 | 10743491.9 % | 0.0298 | 0.3130 |
| 100000 | inviscid | 0/12 | 0.0364 | - | 0.0088 | 0.0927 | 0/12 | 0.0206 | - | 0.0206 | 0.0868 |
| 100000 | oneway | 0/12 | 0.0364 | 59.3 % | 0.0088 | 0.0927 | 0/12 | 0.0206 | 474489432.7 % | 0.0206 | 0.0868 |
| 100000 | vii | 2/12 | 0.0687 | 29.1 % | 0.0230 | 0.1586 | 1/12 | 0.0424 | 7877223.4 % | 0.0213 | 0.1175 |
| 200000 | inviscid | 0/12 | 0.0335 | - | 0.0068 | 0.0929 | 0/12 | 0.0253 | - | 0.0135 | 0.0866 |
| 200000 | oneway | 0/12 | 0.0335 | 49.1 % | 0.0068 | 0.0929 | 0/12 | 0.0253 | 622883438.5 % | 0.0135 | 0.0866 |
| 200000 | vii | 1/12 | 0.0510 | 18.1 % | 0.0138 | 0.1530 | 4/12 | 0.0371 | 12392706.9 % | 0.0126 | 0.1313 |
| 500000 | inviscid | 0/12 | 0.0478 | - | 0.0089 | 0.1501 | 0/12 | 0.0288 | - | 0.0132 | 0.0862 |
| 500000 | oneway | 0/12 | 0.0478 | 45.8 % | 0.0089 | 0.1501 | 0/12 | 0.0288 | 744064824.5 % | 0.0132 | 0.0862 |
| 500000 | vii | 3/12 | 0.0177 | 52.6 % | 0.0032 | 0.0458 | 3/12 | 0.0231 | 27656478.1 % | 0.0072 | 0.0850 |
| 1000000 | inviscid | 0/12 | 0.0456 | - | 0.0084 | 0.1285 | 0/12 | 0.0287 | - | 0.0134 | 0.1078 |
| 1000000 | oneway | 0/12 | 0.0456 | 53.5 % | 0.0084 | 0.1285 | 0/12 | 0.0287 | 660931643.7 % | 0.0134 | 0.1078 |
| 1000000 | vii | 1/12 | 0.0205 | 27.0 % | 0.0042 | 0.0592 | 4/12 | 0.0237 | 18119094.0 % | 0.0079 | 0.0628 |
| 3000000 | inviscid | 0/13 | 0.0371 | - | 0.0067 | 0.1032 | 0/13 | 0.0288 | - | 0.0133 | 0.1331 |
| 3000000 | oneway | 0/13 | 0.0371 | 61.7 % | 0.0067 | 0.1032 | 0/13 | 0.0288 | 507711202.9 % | 0.0133 | 0.1331 |
| 3000000 | vii | 0/13 | 0.0085 | 11.6 % | 0.0017 | 0.0201 | 0/13 | 0.0418 | 23923166.5 % | 0.0133 | 0.1290 |
| 6000000 | inviscid | 0/12 | 0.0286 | - | 0.0054 | 0.0742 | 0/12 | 0.0228 | - | 0.0116 | 0.1053 |
| 6000000 | oneway | 0/12 | 0.0286 | 64.9 % | 0.0054 | 0.0742 | 0/12 | 0.0228 | 385915262.1 % | 0.0116 | 0.1053 |
| 6000000 | vii | 0/12 | 0.0050 | 10.2 % | 0.0011 | 0.0135 | 0/12 | 0.0402 | 22140889.7 % | 0.0134 | 0.1074 |

Reading:

- Inviscid and one-way Cl/Cm move slightly away from viscous XFOIL (pooled mean |dCl| 0.0678 ->
  0.0729). Expected: the core is now the correct inviscid solution (NACA 2412, 5 deg: 0.863
  against inviscid XFOIL 0.862; v1.3.1 0.858), and the benchmark compares it with viscous XFOIL
  (0.810).
- Transpiration VII on the open sections: pooled mean |dCl| 0.0446 -> 0.0414, |dCm| 0.0093 ->
  0.0089, not converged 64 -> 60 of 330, mean relative |dCd| 7.1 % -> 8.2 %. Newton VII (small
  subset) is worse: Re 2e5 mean |dCl| 0.084 -> 0.101, relative |dCd| 3.6 % -> 6.4 %.
- E387 (sharp trailing edge): one-way and VII drag are meaningless after routing (Re 1e6:
  one-way Cd 26 837 at 2 deg and 100 743 at 5 deg; VII 1 463 and 4 614, reported converged;
  v1.3.1: 0.0034-0.014). Same cause as the four failing VII tests: the correct inviscid
  stagnation point at a sharp trailing edge breaks the v1.3.1 boundary-layer march there.
  Decision needed (VII and the one-way drag are M3).

### Answers Q7-Q11 (2026-10-09; for step 4)

- Q7. Thickness and camber scaling (SPEC 4.4) is done in M2.
- Q8. API approved with these details:
  - `fl.analyze` and `fl.polar` accept an `Airfoil` directly as well as a `Configuration`.
  - `Result` also records alpha, panel count, whether the geometry was repanelled, and the code
    version.
  - Drag is `None` for an inviscid result, never 0.
  - `solve_flow` keeps its current behaviour when called with a Reynolds number until M3, so
    the GUI, CLI and batch tools do not break. Only the new `fl.analyze` raises
    `NotImplementedError` for a Reynolds number.
- Q9. `analyze` repanels to 160 panels by default; `repanel=False` uses the given points unchanged.
- Q10. Reference chord = the first element's chord; moment reference = its quarter-chord point.
  Both are overridable on the `Configuration` and both are reported in the `Result`.
- Q11. `docs/benchmark_m2.md`, tables only; `docs/baseline_v1.3.1.md` stays unchanged.

### Decisions after the step 3 report (2026-10-09)

Sawtooth reading confirmed by the user: a sawtooth is two sign flips in a row (+-+ or -+-) of
the nodal-strength second difference.

Williams refinement criterion replaced by the user. Original (Q5): "the error against the table
falls from 200 to 400 panels". It failed for both main elements (A 0.0222 -> 0.0247, B 0.0094 ->
0.0121; evidence under "Option (b) results": a likely misprint at Table 2(a) main row 17 and a
geometry-reconstruction floor near the leading edge). The user replaced it because it measured
the resolution of the 61-point table, not the solver. New criterion: the difference to our own
1600-panel solution on the same geometry falls from 200 to 400 to 800 panels with observed order
of at least 1.5 on each main element and flap. Read as: e_N = max |Cp_N - Cp_1600| over the
tabulated points with x/c 0.02-0.98; e_200 > e_400 > e_800 and log2(e_200/e_400),
log2(e_400/e_800) >= 1.5 (stated before running). Max |dCp| against the table <= 0.05 and Cl
within 1 % are unchanged. Result:

| Element | e_200 / e_400 / e_800 | orders |
|---|---|---|
| A main | 2.02e-2 / 5.27e-3 / 9.74e-4 | 1.94, 2.44 |
| A flap | 8.99e-3 / 2.40e-3 / 5.56e-4 | 1.90, 2.11 |
| B main | 7.36e-3 / 2.56e-3 / 6.65e-4 | 1.52, 1.95 |
| B flap | 2.04e-3 / 6.39e-4 / 1.08e-4 | 1.67, 2.56 |

All pass; B main's first order (1.52) is close to the limit. The order of the second step is
inflated because the 1600-panel reference carries its own error. The CSVs stay exactly as
printed; the two suspected misprints are listed in `tests/data/williams/ERRATA_SUSPECTED.md`.

Benchmark report command: `python -m foil_lab.benchmark report` (and `all`) no longer writes
`docs/baseline_v1.3.1.md`; only the explicit `baseline-doc` command does (test added).

Changelog (Unreleased): exact quarter-chord moment; `panels.py` removed.

### Closed (sharp) trailing edge: verdict (ii) and fix P3 (2026-10-09)

Check against the exact solution (the user's concern): on the Kármán-Trefftz section (tau = 10
deg) the exact speed on the panel touching the edge is 0.69-0.75, while the core gave 0.12-0.15
(Cp about +0.98 against +0.44 to +0.53) at 200, 400 and 800 panels and at 0 and 4 deg. The second
panel was 0.03 low at every N. On closed NACA 2412 the last panel dropped to 0.06 / 0.01 (160 /
320 panels), against 0.64-0.72 with gaps of 1e-4 and 1e-3 c. Cause: the two coincident
trailing-edge nodes took the wrong sign (Kármán-Trefftz, 200 panels: -0.427 next to +0.723), so
the Kutta condition held with a flipped pair. Verdict (ii), accepted by the user: the core was
locally wrong at a closed trailing edge. The step 3 explanation (a correct stagnation point
breaking the boundary layer) was wrong.

Fix (approved form, P3): Drela (1989), page 3, eq. (9), (gamma_3 - 2 gamma_2 + gamma_1) -
(gamma_{N-2} - 2 gamma_{N-1} + gamma_N) = 0, for bodies without a trailing-edge panel. Drela
replaces one of two identical nodal streamfunction equations with it. The least-squares
formulation is our own adaptation for the midpoint Neumann method, which has no identical pair:
the Kutta condition and eq. (9) are imposed exactly by eliminating gamma_first and gamma_last
(gamma_last = -gamma_first; gamma_first = (2 gamma_1 - gamma_2 + gamma_{L-2} - 2 gamma_{L-1}) / 2 in
body-local node numbers), and the N Neumann equations are solved in least squares (QR) for the
remaining unknowns (`inviscid.trailing_edge_elimination`). Per body; open bodies keep their
exact Kutta row; with no closed body the exact square system is used as before (results
unchanged). Same assembly for NumPy and JAX.

P3 against P2 (P2: all rows in one least-squares problem):

| Metric | P2 | P3 |
|---|---|---|
| KT edge-panel \|dq\|, 200 / 400 / 800, 0 deg | 0.0218 / 0.0212 / 0.0205 | identical |
| KT next panel \|dq\| | 0.0015-0.0016 | identical |
| KT max \|dCp\| x/c 0.02-0.98, 200 panels, 0 / 4 deg | 0.00116 / 0.00097 | identical |
| NACA 2412 closed, 160 panels: edge \|q\|, Cl (gap 1e-4 c: 0.636, 0.86143) | 0.718, 0.86145 | identical |
| Kutta residual | up to 4.7e-11 | 0 |
| eq. (9) residual | up to 1.6e-9 | up to 6.7e-16 |
| Max Neumann residual | 2.8e-6 (KT 200), 6.6e-6 (NACA 160) | identical |
| All Neumann rows scaled by 0.1 / 10: max change in \|q\| | 1.2e-11 / 1.2e-9 | 3.3e-15 / 2.9e-15 |
| Only the two edge-panel rows scaled by 0.1 / 10 | 1.5e-5 / 1.5e-7 | 1.5e-5 / 1.5e-7 |

Condition number of the closed system: 4.4e4-1.2e5 before, 7.8 after (NACA 2412, 160-640
panels).

Kármán-Trefftz extension (tolerances fixed by the user; 0 and 4 deg; 200, 400, 800 panels; both
surfaces; x/c > 0.98): panel touching the edge |dq| <= 0.03 and |dCp| <= 0.05; every other panel
|dq| <= 0.01 and |dCp| <= 0.02. The edge-panel tolerance was set after the P2 scratch result
(about 0.021) had been seen. Passes. New tests also check the per-body constraints in a
closed + open + closed configuration, invariance to scaling of the Neumann rows, and the JAX
gradient of Cl on a closed section against central differences (relative 1e-4, nodes 30 and 1).

Ordered checks after the fix:

1. Core, gap, Williams, exact-solution and inviscid XFOIL tests, tolerances unchanged: 58 passed,
   4 expected failures (repanelling, step 4; aspect ratio, M6).
2. The six VII tests on closed geometry, unedited: five pass. STOPPED here as instructed:
   `test_vii_newton.py::test_newton_low_re_drag_right_order_of_magnitude` fails on
   `r.converged` (Newton VII, NACA 2412, Re 5e4, 2 deg, 120 panels; residual 2.0e-2 after 60
   iterations; Cd 0.0180, inside the test's band). Newton VII at this point converges or not
   depending on panel count on both trailing edges: closed 100 yes, 120 no, 140 no (2.5e-1), 160
   yes but Cl 0.60; open 100 yes, 120 yes, 140 no (9.5e-2), 160 yes (Cl 0.68).
   Steps 3 (guard) and 4 (benchmark, batch) and the merge were not started.

### Citations made without the source at hand

Added by me in M1 and M2 without the book or paper at hand (the user will check them):

- `src/foil_lab/inviscid.py` module docstring and DECISIONS.md (M2 plan): J. L. Kuethe and
  C.-Y. Chow, *Foundations of Aerodynamics*, 5th ed., Wiley (1998), section 5.10.
- `src/foil_lab/inviscid.py` module docstring: J. Katz and A. Plotkin, *Low-Speed
  Aerodynamics*, 2nd ed., CUP (2001), sections 10.3-10.4 and 11.4.2.
- `src/foil_lab/inviscid.py` (`source_panel_velocity`): Katz and Plotkin (2001), section 10.2.
- `src/foil_lab/inviscid.py` (`solve`) and DECISIONS.md (M2 step 1, plan change): Katz and
  Plotkin (2001), section 11.4.2 (surface speed equals the sheet strength).
- DECISIONS.md (M2 step 1, alternative B): Katz and Plotkin (2001), chapter 11.
- `src/foil_lab/inviscid.py` (`solve`), `src/foil_lab/vii.py` and DECISIONS.md: M. J. Lighthill,
  "On displacement thickness", J. Fluid Mech. 4 (1958) 383-392.
- `tests/test_exact_solutions.py` (M1): L. M. Milne-Thomson, *Theoretical Aerodynamics* (1958),
  chapter 7; Katz and Plotkin (2001), section 6.7.
- `src/foil_lab/generators.py` (M1): Jacobs, Ward and Pinkerton, NACA Report 460 (1933), and the
  open-trailing-edge coefficient -0.1015; Abbott and von Doenhoff. Also cited in
  `benchmark/sections.py`, `benchmark/report.py` and DECISIONS.md (M1 D1).

With the source at hand: Drela (1989) (`review-gui/drela-1989-xfoil.pdf`) and Williams, ARC R&M
3717 (`review-gui/arc-rm-3717.pdf`).

Already in the v1.3.1 code, not added or checked by me: Thwaites (`boundary_layer.py`,
`vii_qs.py`), Head (`vii_qs.py`), Michel, Tollmien-Schlichting and Drela & Giles (1987)
(`transition.py`), Squire-Young (`vii.py`), Kármán-Tsien (`corrections.py`), Busemann
(`gui/main_window.py`).

### Decisions after the P3 report (2026-10-09)

P3 accepted as implemented.

Decision 1: `test_vii_newton.py::test_newton_low_re_drag_right_order_of_magnitude` is marked
`xfail(strict=True, raises=AssertionError)`, fixed in M3, reason "Newton VII convergence at Re
5e4 depends on panel count on both trailing edges". Panel count, tolerance and iteration limit
unchanged. Newton VII, NACA 2412, Re 5e4, 2 deg, `max_iter=60`, `tol=1e-4`:

| Panels | Closed (P3) | Open (trailing-edge panel) |
|---|---|---|
| 100 | converges, 17 iterations; Cl 0.345, Cd 0.0178 | converges, 19; Cl 0.373, Cd 0.0179 |
| 120 | not converged (residual 2.0e-2); Cl 0.343, Cd 0.0180 | converges, 50; Cl 0.348, Cd 0.0174 |
| 140 | not converged (residual 2.5e-1); Cl 0.343, Cd 0.0103 | not converged (9.5e-2); Cl 0.359, Cd 0.0180 |
| 160 | converges, 56; Cl 0.601, Cd 0.0066 | converges, 38; Cl 0.675, Cd 0.0141 |

Decision 2 (option a): within one optimisation the trailing-edge type is fixed. By default a
section that starts closed stays closed (the PARSEC gap `dy_te` is held at zero and is not a
design variable); with `trailing_edge="open"` the gap is a design variable with the lower bound
`params.DESIGN.min_te_gap` (0.001 c; a new default, proposed and awaiting approval), so the
optimiser never reaches the switch. Applies to `design_for_cl`, `maximize_ld` and `match_cp`.
`test_design_for_cl_hits_target` is unedited and passes (all three targets); a new test checks
both cases. Why the optimiser must not cross between the treatments: at zero gap the closed
treatment (Drela eq. 9) and the open one (trailing-edge panel) differ by a discretisation amount,
which a finite-difference step of about 1.5e-8 across zero gap sees as a gradient of order 1e5
(SLSQP then stopped at its start: Cl 0.918 against target 1.1). Cl at 5 deg, closed against a gap
of 1e-8:

| Section | Panels | Closed | Gap 1e-8 | Jump |
|---|---|---|---|---|
| PARSEC start | 100 | 0.917750 | 0.916657 | -1.09e-3 |
| PARSEC start | 200 | 0.918712 | 0.918176 | -5.36e-4 |
| PARSEC start | 400 | 0.918953 | 0.918683 | -2.69e-4 |
| PARSEC start | 800 | 0.919012 | 0.918875 | -1.37e-4 |
| NACA 2412 | 100 | 0.861065 | 0.861014 | -5.1e-5 |
| NACA 2412 | 160 | 0.861448 | 0.861428 | -2.0e-5 |
| NACA 2412 | 320 | 0.861632 | 0.861627 | -4.8e-6 |
| NACA 2412 | 640 | 0.861677 | 0.861676 | -1.6e-6 |

The jump is first order on the PARSEC start section and falls faster on NACA 2412; both
treatments converge to the same limit.

### Step 4: tolerances fixed before running (2026-10-09)

- XFOIL NACA cross-check (slow, xfoil group; Q6): XFOIL's own `NACA 2412` coordinates, saved
  with `SAVE`, against our open NACA 2412: the largest distance from any XFOIL point to our
  4000-panel open NACA 2412 polyline <= 1e-4 chord.
- Closed NACA geometry pinned bit for bit under the explicit `trailing_edge="closed"` option
  (Q2): SHA-256 of the coordinate array for 0012, 2412, 4412 and 23012 at 120 and 160 panels,
  recorded before the default changes.
- NACA 4/5 analytic properties: symmetric 00xx (upper y = -lower y to 1e-15); maximum thickness
  equals t/c within 2e-4 at 4000 panels; open trailing-edge gap equals the closed-form value
  10 t (0.2969 - 0.1260 - 0.3516 + 0.2843 - 0.1015) = 0.021 t within 1e-12 for a symmetric
  section; 4-digit maximum camber m at x = p within 1e-3 (4000 panels);
  5-digit 230xx maximum camber at x = 0.15 within 2e-3 (4000 panels).
- Geometry operations on scaled (0.5), translated (0.2, 0.1) and rotated (10 deg about the
  quarter chord) input: an operation applied to the transformed section equals the transformed
  result of the operation on the original within 1e-9 chord (repanelling, trailing-edge gap,
  thickness and camber scaling). The three repanelling tests of M1 keep their tolerance (1e-4
  chord to a 4000-panel curve).
- Thickness/camber scaling by 1 returns the input within 1e-12; scaled thickness has maximum
  thickness f x the original within 2e-4 chord (NACA 2412, 4000 panels); scaled camber 0 on a
  cambered NACA section gives a symmetric section within 1e-12 about the chord line.
- Trailing-edge gap tool: the new gap equals the request within 1e-12 chord; points ahead of
  the blending start are unchanged within 1e-12; a gap above 1 % chord gives a warning.
- `fl.analyze` on an Airfoil equals `fl.analyze` on the one-element Configuration bit for bit;
  per-element loads sum to the totals within 1e-12; NACA 2412 at 4 deg, 160 panels, open
  trailing edge, Cl within 1e-12 of the core called directly on the same nodes.

### Checks 3 and 4 after P3 (2026-10-09)

Check 3, guard: `solver.drag_is_plausible` (finite, 0 <= Cd <= 1). A viscous result outside it
is never reported as converged: `VIIResult.converged` of all three VII variants (so the CLI and
GUI status follow), batch rows with a Reynolds number, and one-way rows of the benchmark adapter.
Tests in `tests/test_drag_guard.py`.

Check 4, benchmark before / after (before: stored M1 run of v1.3.1; after: the routed core with
P3, same adapter and canonical files, 6 workers, one BLAS thread; output in
`benchmark_output/m2_after/`, not committed):

### open trailing edge, by Re

| Re | method | not conv. before | mean dCl before | mean rel dCd before | mean dCm before | max dCl before | not conv. after | mean dCl after | mean rel dCd after | mean dCm after | max dCl after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50000 | inviscid | 0/63 | 0.2119 | - | 0.0236 | 0.5136 | 0/63 | 0.2148 | - | 0.0244 | 0.5227 |
| 50000 | oneway | 0/63 | 0.2119 | 75.4 % | 0.0236 | 0.5136 | 0/63 | 0.2148 | 76.3 % | 0.0244 | 0.5227 |
| 50000 | vii | 28/63 | 0.1563 | 28.8 % | 0.0251 | 0.3057 | 25/63 | 0.1743 | 31.2 % | 0.0267 | 0.4023 |
| 50000 | newton | 8/9 | 0.1765 | 17.7 % | 0.0378 | 0.1765 | 7/9 | 0.2175 | 46.0 % | 0.0350 | 0.2957 |
| 100000 | inviscid | 0/67 | 0.0982 | - | 0.0154 | 0.2477 | 0/67 | 0.1008 | - | 0.0162 | 0.2565 |
| 100000 | oneway | 0/67 | 0.0982 | 69.4 % | 0.0154 | 0.2477 | 0/67 | 0.1008 | 70.6 % | 0.0162 | 0.2565 |
| 100000 | vii | 24/67 | 0.1247 | 20.2 % | 0.0236 | 0.2020 | 22/67 | 0.1239 | 19.7 % | 0.0237 | 0.2074 |
| 200000 | inviscid | 0/70 | 0.0654 | - | 0.0116 | 0.2973 | 0/70 | 0.0680 | - | 0.0123 | 0.3070 |
| 200000 | oneway | 0/70 | 0.0654 | 64.9 % | 0.0116 | 0.2973 | 0/70 | 0.0680 | 66.2 % | 0.0123 | 0.3070 |
| 200000 | vii | 19/70 | 0.0873 | 7.6 % | 0.0173 | 0.1471 | 20/70 | 0.0840 | 7.5 % | 0.0169 | 0.1420 |
| 200000 | newton | 5/10 | 0.0840 | 3.6 % | 0.0158 | 0.1275 | 5/10 | 0.1010 | 6.4 % | 0.0220 | 0.1928 |
| 500000 | inviscid | 0/70 | 0.0555 | - | 0.0108 | 0.2238 | 0/70 | 0.0591 | - | 0.0115 | 0.2335 |
| 500000 | oneway | 0/70 | 0.0555 | 63.1 % | 0.0108 | 0.2238 | 0/70 | 0.0591 | 64.4 % | 0.0115 | 0.2335 |
| 500000 | vii | 14/70 | 0.0552 | 6.9 % | 0.0114 | 0.1311 | 13/70 | 0.0553 | 8.8 % | 0.0117 | 0.2138 |
| 1000000 | inviscid | 0/78 | 0.0625 | - | 0.0113 | 0.2717 | 0/78 | 0.0673 | - | 0.0125 | 0.2834 |
| 1000000 | oneway | 0/78 | 0.0625 | 65.9 % | 0.0113 | 0.2717 | 0/78 | 0.0673 | 67.0 % | 0.0125 | 0.2834 |
| 1000000 | vii | 13/78 | 0.0445 | 6.3 % | 0.0091 | 0.1235 | 16/78 | 0.0385 | 6.0 % | 0.0080 | 0.1162 |
| 3000000 | inviscid | 0/88 | 0.0745 | - | 0.0124 | 0.3048 | 0/88 | 0.0802 | - | 0.0145 | 0.3183 |
| 3000000 | oneway | 0/88 | 0.0745 | 73.1 % | 0.0124 | 0.3048 | 0/88 | 0.0802 | 74.0 % | 0.0145 | 0.3183 |
| 3000000 | vii | 19/88 | 0.0422 | 7.0 % | 0.0088 | 0.2557 | 15/88 | 0.0404 | 8.5 % | 0.0088 | 0.2435 |
| 6000000 | inviscid | 0/94 | 0.0750 | - | 0.0120 | 0.3088 | 0/94 | 0.0811 | - | 0.0146 | 0.3233 |
| 6000000 | oneway | 0/94 | 0.0750 | 76.8 % | 0.0120 | 0.3088 | 0/94 | 0.0811 | 77.6 % | 0.0146 | 0.3233 |
| 6000000 | vii | 18/94 | 0.0390 | 8.2 % | 0.0084 | 0.2434 | 16/94 | 0.0344 | 9.2 % | 0.0076 | 0.2454 |

### open trailing edge, validated range pooled (Re 5e5-6e6)

| Re | method | not conv. before | mean dCl before | mean rel dCd before | mean dCm before | max dCl before | not conv. after | mean dCl after | mean rel dCd after | mean dCm after | max dCl after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | inviscid | 0/330 | 0.0678 | - | 0.0117 | 0.3088 | 0/330 | 0.0729 | - | 0.0134 | 0.3233 |
| pooled | oneway | 0/330 | 0.0678 | 70.3 % | 0.0117 | 0.3088 | 0/330 | 0.0729 | 71.3 % | 0.0134 | 0.3233 |
| pooled | vii | 64/330 | 0.0446 | 7.1 % | 0.0093 | 0.2557 | 60/330 | 0.0414 | 8.2 % | 0.0089 | 0.2454 |

### E387 (sharp), by Re

| Re | method | not conv. before | mean dCl before | mean rel dCd before | mean dCm before | max dCl before | not conv. after | mean dCl after | mean rel dCd after | mean dCm after | max dCl after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50000 | inviscid | 0/11 | 0.2780 | - | 0.0128 | 0.3440 | 0/11 | 0.2805 | - | 0.0138 | 0.3467 |
| 50000 | oneway | 0/11 | 0.2780 | 74.8 % | 0.0128 | 0.3440 | 0/11 | 0.2805 | 72.8 % | 0.0138 | 0.3467 |
| 50000 | vii | 3/11 | 0.2581 | 55.8 % | 0.0232 | 0.4130 | 1/11 | 0.2087 | 44.0 % | 0.0116 | 0.2892 |
| 100000 | inviscid | 0/12 | 0.0364 | - | 0.0088 | 0.0927 | 0/12 | 0.0376 | - | 0.0090 | 0.0954 |
| 100000 | oneway | 0/12 | 0.0364 | 59.3 % | 0.0088 | 0.0927 | 0/12 | 0.0376 | 55.6 % | 0.0090 | 0.0954 |
| 100000 | vii | 2/12 | 0.0687 | 29.1 % | 0.0230 | 0.1586 | 0/12 | 0.0351 | 22.7 % | 0.0152 | 0.0606 |
| 200000 | inviscid | 0/12 | 0.0335 | - | 0.0068 | 0.0929 | 0/12 | 0.0355 | - | 0.0072 | 0.0956 |
| 200000 | oneway | 0/12 | 0.0335 | 49.1 % | 0.0068 | 0.0929 | 0/12 | 0.0355 | 44.1 % | 0.0072 | 0.0956 |
| 200000 | vii | 1/12 | 0.0510 | 18.1 % | 0.0138 | 0.1530 | 0/12 | 0.0310 | 28.5 % | 0.0084 | 0.0628 |
| 500000 | inviscid | 0/12 | 0.0478 | - | 0.0089 | 0.1501 | 0/12 | 0.0502 | - | 0.0099 | 0.1525 |
| 500000 | oneway | 0/12 | 0.0478 | 45.8 % | 0.0089 | 0.1501 | 0/12 | 0.0502 | 43.0 % | 0.0099 | 0.1525 |
| 500000 | vii | 3/12 | 0.0177 | 52.6 % | 0.0032 | 0.0458 | 0/12 | 0.0235 | 38.0 % | 0.0044 | 0.0678 |
| 1000000 | inviscid | 0/12 | 0.0456 | - | 0.0084 | 0.1285 | 0/12 | 0.0479 | - | 0.0092 | 0.1309 |
| 1000000 | oneway | 0/12 | 0.0456 | 53.5 % | 0.0084 | 0.1285 | 0/12 | 0.0479 | 50.1 % | 0.0092 | 0.1309 |
| 1000000 | vii | 1/12 | 0.0205 | 27.0 % | 0.0042 | 0.0592 | 0/12 | 0.0151 | 19.0 % | 0.0030 | 0.0444 |
| 3000000 | inviscid | 0/13 | 0.0371 | - | 0.0067 | 0.1032 | 0/13 | 0.0394 | - | 0.0075 | 0.1056 |
| 3000000 | oneway | 0/13 | 0.0371 | 61.7 % | 0.0067 | 0.1032 | 0/13 | 0.0394 | 57.2 % | 0.0075 | 0.1056 |
| 3000000 | vii | 0/13 | 0.0085 | 11.6 % | 0.0017 | 0.0201 | 0/13 | 0.0058 | 10.3 % | 0.0009 | 0.0165 |
| 6000000 | inviscid | 0/12 | 0.0286 | - | 0.0054 | 0.0742 | 0/12 | 0.0309 | - | 0.0060 | 0.0769 |
| 6000000 | oneway | 0/12 | 0.0286 | 64.9 % | 0.0054 | 0.0742 | 0/12 | 0.0309 | 60.5 % | 0.0060 | 0.0769 |
| 6000000 | vii | 0/12 | 0.0050 | 10.2 % | 0.0011 | 0.0135 | 0/12 | 0.0048 | 11.4 % | 0.0010 | 0.0130 |

Reading: open sections are unchanged from the step 3 run (P3 only affects closed edges). E387
(sharp) is sane again and mostly better than v1.3.1: transpiration VII at Re 1e6 mean |dCl|
0.0205 -> 0.0151, relative |dCd| 27.0 % -> 19.0 %, 0 of 12 not converged (was 1). Newton VII on
its subset is worse (Re 2e5: |dCl| 0.084 -> 0.101, |dCd| 3.6 % -> 6.4 %), consistent with its
panel-count sensitivity (M3).

Batch example on the idle machine: runs, 504 points. Its "best L/D" (950-1440 at Re 3e6) is the
known v1.3.1 one-way drag defect (NACA 2412 open, 12 deg: one-way Cd 0.00179, v1.3.1 0.00183,
XFOIL 0.0151), not new; the one-way path is removed in M3.

### Step 4: implemented, stopped on failing tests (2026-10-09)

Implemented on branch `m2-step4-wip` (not merged):

- `foil_lab.airfoil`: `Airfoil` (read-only Selig coordinates, name, source; `naca`, `from_file`,
  `from_coords`, `parsec`, `biconvex`, `wedge`; `te_gap`, `chord`, `leading_edge`,
  `quarter_chord`; `scaled`, `translated`, `rotated`, `repanelled`, `with_te_gap`,
  `with_thickness_scaled`, `with_camber_scaled`), `Element` (scale about the origin, pitch
  about a pivot, offset), `Configuration` (reference chord and point per Q10, overridable).
- `foil_lab.api`: `Result` (totals, per-element loads and Cp with upper/lower labels, alpha,
  panel count, repanelled flag, reference chord and point, converged, method, corrections,
  code version; drag `None` when inviscid), `Polar`, `analyze`, `polar` (Airfoil, Element or
  Configuration; repanel to 160 by default; `NotImplementedError` for a Reynolds number).
  `fl.Airfoil`, `fl.Element`, `fl.Configuration`, `fl.analyze`, `fl.polar` at the package
  root. `solve_flow` unchanged. No pandas conversion (pandas is not a dependency; M7).
- `geometry`: `repanel_cosine` is now arc-length repanelling (cubic spline in arc length,
  leading edge farthest from the trailing-edge midpoint by root finding, cosine clustering at
  both edges, end points kept); the x-based PCHIP version is deleted. New
  `scale_thickness_camber`, `set_trailing_edge_gap` (blend aft of 0.8 c, warning above 1 % c),
  `trailing_edge_gap`. Parameters in `params.GEOMETRY` (num_panels 160, te_gap_warn 0.01,
  te_gap_blend_start 0.8, le_search_samples 20; new defaults, proposed).
- Generators: default trailing edge "open" for NACA 4/5 (closed pinned by SHA-256 under the
  explicit option); `generate_naca6` removed (GUI, one test, one example and three README lines
  updated).
- Version `2.0.0.dev0` (package and `__version__`), so that `Result.version` is not "1.3.1".
- Reference chord of a single NACA section by the farthest-point leading edge: open NACA 2412
  chord 1.00008, quarter-chord point (0.24994, 0.00119).
- Existing tests changed because of the approved default change (Q2), to pin "closed"
  explicitly with unchanged comparisons: `test_generators.py::test_default_is_bit_identical_to_v131`
  (renamed `test_closed_option_is_bit_identical_to_v131`), `test_generators.py::test_trailing_edge_gap`,
  `test_geometry.py::test_normalize_unit_chord`, `test_xfoil_runner.py::test_failed_runs_give_all_angles_unconverged`.
  The three M1 repanelling tests pass with their tolerance; their M2 mark is removed.

Failing (full suite: 311 passed, 6 failed, 3 expected failures):

1. `test_open_naca2412_matches_xfoil_naca_command` (tolerance 1e-4 c): max distance 1.13e-3 c,
   on the upper surface at x/c 0.014-0.025. XFOIL's `NACA 2412` output matches a construction
   with the thickness added vertically to the camber line to 3.1e-7 c; our generator adds it
   perpendicular to the camber line. The trailing-edge gap agrees (0.00252). Inferred from the
   coordinates only; XFOIL source was not read. Decision needed: which definition the NACA
   4-digit generator uses (the "perpendicular" reading of NACA Report 460 / Abbott and von
   Doenhoff is from memory, section unverified).
2. `test_naca_maximum_thickness[4415-0.15]`: vertical thickness 0.15023 against 0.15 (2.3e-4 >
   2e-4); the property depends on the same thickness definition.
3. `test_zero_camber_gives_symmetric_section`: 6.4e-3 measured about the x-axis (the test used
   the wrong frame); 6.1e-4 in the section's own chord frame, still far above the 1e-12 fixed
   beforehand, because the check interpolates between unpaired nodes. The 1e-12 was not
   attainable with this measurement.
4. `test_canonical_file_matches_reference_checksum[CLARKY]` and `[E387]`: the canonical files
   changed with the new repanelling; regenerating these references is step 5.
5. `test_vii_newton.py::test_newton_low_re_drag_right_order_of_magnitude`: XPASS(strict). The
   test is unedited and uses the generator default, now open; the open 120-panel case converges
   (table under "Decisions after the P3 report"). The panel-count sensitivity remains.

Fixed during step 4 without changing a tolerance: `set_trailing_edge_gap` searched a bracket
symmetric in the gap and so could not open a closed trailing edge.

### Decisions on the step 4 failures (2026-10-09)

1. NACA definition: thickness stays perpendicular to the mean line, for 4- and 5-digit sections
   (the standard definition). XFOIL's `NACA` command, judged from its saved coordinates, adds the
   thickness vertically (its NACA 2412 matches a vertical construction to 3.1e-7 c). The XFOIL
   cross-check is changed by the user. Original test: XFOIL's NACA 2412 within 1e-4 c of our open
   NACA 2412; it could not pass because the two codes use different constructions (1.13e-3 c at
   x/c 0.014-0.025). New test, two parts: (a) NACA 0012, identical in both constructions, direct
   comparison, 1e-4 c (unchanged); (b) NACA 2412, our mean-line and thickness functions combined
   vertically inside the test, against XFOIL's coordinates, 1e-5 c; this tolerance was set after
   the 3.1e-7 result had been seen. The 4-digit mean line is now a separate function
   (`generators._naca4_camber`, same arithmetic; the closed-geometry SHA-256 pins still pass).
2. Maximum-thickness test: thickness measured perpendicular to the mean line, as the distance
   between the upper and lower points of one construction station (2 y_t); tolerance 2e-4
   unchanged, symmetric and cambered sections. Passes.
3. Zero-camber symmetry: the earlier 1e-12 was attached to the wrong measurement (an
   interpolating check). Replaced by two checks: (a) raw NACA 0012 output, upper and lower nodes
   paired by index, y_u = -y_l to 1e-12 (passes); (b) NACA 4412 with camber scaled to zero, both
   surfaces resampled with the arc-length cubic spline at common x stations of the section's own
   chord frame, |y_u + y_l| <= 1e-5 c for x/c 0.01-0.99. (b) FAILS: 4.42e-5 at x/c 0.99. Where:
   the asymmetry is linear in x (slope 3.80e-5, offset 6.3e-6; 3.6e-7 after removing the linear
   part), i.e. a tilt of 0.0011 deg between the result's own chord line and its symmetry line.
   Measured in the chord frame the operation used, the result is symmetric to 3.9e-8. Cause: the
   two trailing-edge nodes of the open cambered section have different chordwise positions in
   that frame, so they receive different thicknesses; the lower trailing-edge point moves from y
   -0.001249 to -0.001294 while the upper stays at +0.001249, which shifts the trailing-edge
   midpoint and so the recomputed chord line. Stopped here as instructed; not loosened.
4. Newton test: pinned to `trailing_edge="closed"` (the geometry it always tested); strict
   expected-failure mark kept; its reason notes that the open 120-panel case happens to converge.
5. Defaults approved by the user: `DESIGN.min_te_gap` 0.001 c; `GEOMETRY` num_panels 160,
   te_gap_warn 0.01, te_gap_blend_start 0.8, le_search_samples 20; version 2.0.0.dev0.

### Trailing-edge pairing in the shape operations (2026-10-09)

Cause of the zero-camber failure: `scale_thickness_camber` treated the two trailing-edge nodes as
separate stations. On an open cambered section they lie at different chordwise positions in the
chord frame (NACA 4412: offset 0.000325 c), so they received different thicknesses; after camber
removal the trailing-edge midpoint left the chord line and the result's own chord line tilted by
0.0011 deg (check 3b: 4.42e-5).

Fix (option a, approved): the two trailing-edge nodes are one station, the trailing-edge midpoint
(x' = 1). Camber and thickness there come from the pair itself: c_TE = (y_0 + y_N)/2, t_TE = y_0 -
y_N in the chord frame; the nodes move to camber * c_TE +/- thickness * t_TE / 2, keeping their
chordwise positions. Results: zero-camber check 3b passes as specified (|y_u + y_l| <= 1e-5 c, x/c
0.01-0.99); scaling by 1 still returns the input (1e-12); the operations on transformed input still
agree to 1e-9 c.

`set_trailing_edge_gap` had the same issue: the two trailing-edge nodes got different blend
weights, so the midpoint moved 1.2e-6 c when only the gap changed (NACA 4412, gap 0.004). Fixed the
same way (both nodes take the weight of x' = 1, equal and opposite changes); new test: the
midpoint does not move (1e-9 c, open cambered and closed sections) and the gap is exact (1e-12).

Thickness-scaling check requested by the user: (1) the trailing-edge midpoint stays on the
original mean line: holds (moved 2.2e-16 c; test added, 1e-9). (2) the gap scales by the same
factor to 1e-9 c: does NOT hold with the current definition, which scales only the chord-normal
part of the trailing-edge opening and keeps the chordwise offset between the two points. NACA 4412
(gap 0.002520; normal part 0.002498, chordwise part 0.000325): x0.5 gives 0.0012911 (ratio
0.5123, 3.1e-5 from 0.5 x gap), x1.5 gives 0.0037625 (ratio 1.4930, 1.8e-5 off). Stopped before
changing the definition, as instructed; check (2) not added.

Decision (user, 2026-10-09): option (b), keep the current definition (thickness scaled normal to
the chord; the chordwise offset of the trailing-edge points is a property of the edge). Check (2)
is now: in the original chord frame, the chord-normal part of the trailing-edge gap scales by the
factor to 1e-9 c and the chordwise offset between the two trailing-edge points is unchanged to
1e-9 c (`test_thickness_scaling_gap_components`). The original check (gap length proportional)
was over-specified.

Rule from here to the end of M2 (user): definitional choices in geometry operations and tests of
my own that measure the wrong thing are decided by me, recorded here with the reason, and listed
in the final report. Still stop for any solver or accuracy tolerance, any benchmark or
exact-solution test, any pre-existing test that fails, and anything that changes SPEC.md.

### Step 5: references for Clark Y, E387 and E387 with a 0.25 % gap (2026-10-09)

- The canonical Clark Y and E387 files change because `repanel_cosine` is now arc-length
  repanelling (step 4). Their references were regenerated with XFOIL (same settings as M1:
  PANE 160 and 240, N_crit 9, free transition, Mach 0, ITER 200, one retry per skipped angle):
  `python -m foil_lab.benchmark references --sections CLARKY,E387,E387_GAP025` (new option; the
  NACA references are unchanged and were not rerun).
- Largest distance of the repanelled points to the downloaded polyline (recorded in the
  provenance): Clark Y 8.11e-4 (M1) -> 2.42e-4; E387 1.64e-4 (M1) -> 6.63e-4, at the leading edge
  (x/c 0.0025) of a 61-point file. Decision (mine, under the user's rule for geometry choices):
  accepted. Near a sparsely sampled leading edge the arc-length spline passes between the straight
  segments of the polyline, so this distance measures the polyline's corner-cutting rather than an
  error of the new points; the spline does not depend on chord, position or rotation, which the
  x-based interpolation of v1.3.1 did.
- `E387_GAP025` (new section kind "gap"): the canonical E387 (160 panels) with
  `geometry.set_trailing_edge_gap(0.0025)` applied, i.e. thickness added aft of 0.8 c with the
  trailing-edge midpoint fixed. Decision (mine): the gap is applied after repanelling, so both
  codes read the same 161 points and the shape ahead of 0.8 c is identical to E387 (test, 1e-12).
  It is in the open-trailing-edge group of the report, where the accuracy targets are judged;
  `BENCHMARK.sections` gains this entry, as the approved M2 plan (point 5) specifies.

### End of M2 (2026-10-09)

Step 6: full suite 340 passed, 0 failed, 4 expected failures (aspect-ratio compressibility, M6;
Newton low-Re drag on the closed section, M3; two non-strict known limitations of the
experimental VII solvers); fast suite 47.7 s. `docs/benchmark_m2.md` (tables only, Q11) written
from `benchmark_output/m2_final/` (not committed); `docs/baseline_v1.3.1.md` unchanged. GUI
(headless), CLI and `fl.analyze` smoke checks run.

Benchmark after M2, validated range pooled, five open sections (NACA 0012, 2412, 4412, 23012,
Clark Y): transpiration VII mean |dCl| 0.0446 -> 0.0386 (target 0.03 not met), mean relative
|dCd| 7.1 % -> 8.0 % (met), mean |dCm| 0.0093 -> 0.0082 (met), not converged 64 -> 61 of 330;
inviscid / one-way |dCl| 0.0678 -> 0.0738 (now the correct inviscid solution against viscous
XFOIL). E387 with a 0.25 % gap (new): VII |dCl| 0.0078, |dCd| 11.7 %, |dCm| 0.0018, all
converged. E387 sharp: VII |dCl| 0.0124 -> 0.0074, |dCd| 23.2 % -> 10.6 %, not converged 4 -> 0.

Decisions made by me under the user's rule (geometry definitions and my own tests that measured
the wrong thing), each recorded above:

1. Thickness-scaling gap check measured by components (decided by the user, option b; listed for
   completeness).
2. Repanelling deviation of Clark Y and E387 against the downloaded polyline accepted (step 5).
3. `E387_GAP025` built by applying the gap tool after repanelling, so both codes read the same
   points and the shape ahead of 0.8 c is identical to E387 (step 5).

## M3: Viscous solver at high Reynolds number

Branch `m3-viscous` from `main` (0c55826). Plan in progress; nothing implemented yet.

### Diagnosis (before any rewrite)

NACA 2412 (canonical file), Re 1e6, N_crit 9, against XFOIL's boundary-layer `DUMP` (XFOIL run
as an external program). Diagnostic committed in `tests/diagnostics/` with the 0 and 5 deg dumps.

- The inviscid side and transpiration are right: our core driven by XFOIL's delta* gives Cl
  0.2406 / 0.8089 (XFOIL 0.2410 / 0.8101), Cm within 2e-4; XFOIL's wake delta* adds only
  0.0004 / 0.002 in Cl.
- XFOIL's dumped Cf is relative to the freestream dynamic pressure, ours to the local edge value;
  divided by ue^2 the laminar Cf agree (2-8 %). (Report 1 called the laminar Cf "2x low"; that
  was a normalisation artefact.)
- Our coupled state at 5 deg is a solution of our own equations on the upper surface (one
  uncapped march on its edge velocity: trailing-edge delta* within 0.8 %); on the lower surface
  the laminar model has no solution beyond laminar separation (Thwaites lambda < -0.09 from x/c
  0.898) and the package forces transition at H = 3.5. At "convergence" the unrelaxed update
  still differs by 5.1 % of max delta* (the test compares relaxed iterates).
- Started from XFOIL's 5 deg state, the package iteration reaches our thick trailing-edge state
  within 20 iterations (Cl 0.758, upper TE H 2.40); without limiters it diverges. The thick state
  belongs to our model; the edge-velocity-prescribed march is too sensitive near the trailing
  edge to run without limiters.
- At 10 and 14 deg our model cannot separate (H cap 2.8 hit; trailing-edge H collapses to about
  1.2): Cl 1.483 / 1.913 against XFOIL 1.268 / 1.495, Cm -0.086 / -0.090 against -0.037 / -0.012,
  not converged in 300 iterations.

### Answers (2026-10-09)

- Q1: diagnosis extended to 10 and 14 deg (above).
- Q2: the wake is not a requirement up front; the formulation keeps room for it and "add the
  wake" is a named checkpoint, taken if the Cd or Cm targets are not met without it.
- Q3: diagnostics committed under `tests/diagnostics/` (slow; regeneration in the xfoil group)
  with the two XFOIL dumps and provenance headers.

### Plan approved (2026-10-09) with changes

Sources supplied in `review-gui/`: Drela & Giles 1987 (DG87), Drela 1989 AIAA-89-2200 (BTE89),
Green et al. R&M 3791, Squire & Young R&M 1838 (SY38); Drela 1989 LNE 54 (LNE89) already at hand.
All equations were read from page images (the OCR text layers garble formulae).
Fidkowski's mfoil paper may be used as a secondary published source; it has not been supplied
and is not used so far. Neither mfoil's nor XFOIL's code is read.

Answers:
- Q2: Michel's criterion dropped. SPEC.md 4.3 changed (approved): "Transition by the e^n envelope
  method, with forced trip locations per surface."
- Q3: `solve_flow` with a Reynolds number moves to the new solver; `enable_post_stall` and
  `viscous_mods.py` are deleted.
- Q4: viscous results are Mach 0 only in M3.
- Q5: benchmark adapter gets one "viscous" method plus inviscid; the two existing benchmark
  documents stay as the before pictures.
- Q6: convergence defaults approved as starting values (scaling below). If C2 shows points that
  stall just above the tolerance, the residual histories are shown before any change.

Changes to the plan (user): Jacobian from JAX automatic differentiation (JAX path) and from
complex-step of the local station residuals with the chain rule through the edge-velocity
relation (NumPy path), never hand-derived; a test that the two agree and match finite
differences of the full residual. C1 also run in the inverse direction (prescribed delta*,
compare ue). Solver structured for several bodies with two branches each (single body tested in
M3). Stops after C0+C1, after C2, after C3; C4 (wake) taken without asking if C3 misses Cd or Cm.

### Formulation details fixed before C1, with sources

Equations (Mach 0): momentum DG87 (4) = LNE89 (21); shape parameter DG87 (5) = LNE89 (22);
amplification DG87 (35) = LNE89 (24); lag LNE89 (23) (decision below). Closures DG87 (10)-(13)
laminar, (14)-(19) turbulent, (20)-(24) dissipation, slip velocity, delta, C_tau,EQ; DG87
(29)-(30) and (33)-(35) amplification; wake BTE89 (13)-(18) (checkpoint C4 only); drag SY38 (3),
(13)-(14).

1. **Initialisation of the Newton iteration.**
   - A polar point starts from the previous converged angle (DG87 p. 1353: "a good initial guess
     was available for each point from the previous angle of attack").
   - A first point is started from a station-by-station march of the same discrete equations on
     the inviscid edge velocity. At stations where the prescribed-ue march has no solution (near
     separation), the march switches to prescribing a small increase of H_k and solves for ue.
   - No published source at hand describes the cold start: this is my own procedure, and its only
     role is to supply a starting state (the converged answer does not depend on it, which C2
     checks with two different starts).
2. **Transition inside the Newton system.**
   - DG87 pp. 1351-1352 and Fig. 5: the interval containing transition is split into a laminar and
     a turbulent subinterval.
   - The onset xi_tr is defined implicitly by the discrete amplification equation reaching N_crit
     on the laminar subinterval, so it is a smooth function of the neighbouring variables.
   - The interval's equations are weighted averages of the two subintervals.
   - C_tau^(1/2) at xi_tr is 0.7 times its equilibrium value.
   - Forced transition: xi_tr set to the trip location (DG87 p. 1352).
   - Mine: the weights are the subinterval length fractions (xi_tr - xi_1)/(xi_2 - xi_1) and its
     complement; the amplification rate on the laminar subinterval is the trapezoidal average of
     the end values; which interval holds transition is re-identified after each Newton update.
3. **Stagnation point.**
   - On the surface ue = +-gamma (LNE89 eq. 30).
   - The stagnation point is where the nodal sheet strength changes sign on a body, located by
     linear interpolation between the two nodes; xi on each branch is measured from it.
   - It is re-located after each Newton update. A node that changes branch keeps its variables
     (they are small and nearly symmetric there); this is mine.
   - The first station of each branch uses the Falkner-Skan stagnation (Hiemenz) similarity
     relations of DG87: m(H_k) = 1 in eq. (34) gives H_k, and eq. (33) gives
     theta^2 = l(H_k) xi / (Re ue) (DG87 eqs. 33-34).
4. **States outside a closure's valid range.**
   - The DG87 closures contain (H_k - 1) and l(H_k) = (6.54 H_k - 14.07)/H_k^2 (positive only for
     H_k > 2.151) and log10 Re_theta.
   - Mine: amplification is zero where Re_theta < Re_theta0 (DG87 eq. 36 integrates from that
     point) or l(H_k) <= 0.
   - A Newton update is scaled down so that H_k stays above `ViscousParams.hk_min` (laminar and
     turbulent) and Re_theta above `ViscousParams.re_theta_min`; every such scaling is counted in
     the result.
   - A solution that ends on a bound is reported `converged=False` with the reason. No closure is
     clipped silently.

Other decisions (mine, recorded):

- **Stations at the panel nodes**, where the sheet strength is the surface speed (LNE89 eq. 30),
  so the source of each panel is sigma = dm/dxi between its two nodes (LNE89 eq. 32).
  Unknowns per station: theta, m = ue delta*, and n or C_tau^(1/2) (LNE89 eq. 34). ue is
  eliminated by ue = ue_inv + D m (LNE89 eq. 33), with D built exactly from our core.
- **Lag equation:** LNE89 eq. (23), the later form Drela describes as giving better lift and drag
  near stall. DG87 eq. (22) and BTE89 eq. (3) print the original 4.2 form.
- **DG87 eq. (18)** is printed with "H_k" on the left; the text says it is the relation between
  H*, H_k and Re_theta, so it is read as H* = ....
- **Its "log"** has no base. DG87 writes log10 explicitly in eqs. (14) and (30), so plain "log" is
  read as the natural logarithm. (BTE89 Fig. 4's H*-H_k curve is closer to the natural-log
  version.) To verify in M9.
- **Discretisation:** two-point central (trapezoidal) differences (DG87 p. 1351), written in
  logarithmic form for theta, H* and C_tau, so each residual is dimensionless. DG87's downstream
  bias of the shape equation at high Re is a parameter (`shape_eq_bias`, 0.5 = central),
  changed only if C1/C2 show the stiffness DG87 describes.
- **Squire-Young** (before the wake checkpoint): theta_1 = theta_upper + theta_lower (SY38 p. 4),
  H_1 = (delta*_upper + delta*_lower)/theta_1, U_1 = theta-weighted mean of the two trailing-edge
  edge speeds; C_D = 2 theta_1 (U_1)^((H_1 + 5)/2) (SY38 eqs. 3, 13-14, U_0 = 1). Friction drag
  = SY38 eq. (4); pressure drag = total - friction.

Residual scaling (Q6), exact form at every station interval (i-1, i) of each branch:

- momentum: R1 = ln(theta_i/theta_{i-1}) + (2 + H_avg) ln(ue_i/ue_{i-1}) - (xi_i - xi_{i-1})
  (Cf/2)_avg / theta_avg
- shape: R2 = ln(H*_i/H*_{i-1}) + (2H**/H* + 1 - H)_avg ln(ue_i/ue_{i-1}) - (xi_i - xi_{i-1})
  (2C_D/H* - Cf/2)_avg / theta_avg
- lag: R3 = ln(C_tau,i/C_tau,i-1) - (xi_i - xi_{i-1}) [rate of LNE89 eq. 23]_avg
- amplification: R3 = n_i - n_{i-1} - (xi_i - xi_{i-1}) (dn/dxi)_avg
- edge velocity: R4 = ue_i - ue_inv,i - sum_j D_ij m_j, in units of the freestream speed.

All residuals are dimensionless and O(1) per interval. Converged: max |R| <= 1e-8 over all
residuals and the last Newton update <= 1e-6 relative on theta and m, within 50 iterations
(`ViscousParams`).

### C0 and C1 (2026-10-09): small decisions, results, stop

Small numerical definitions (decided under the M2 rule, no tolerance involved):

- **March start.** The first station of a branch takes the local Falkner-Skan similarity state
  (DG87 eqs. 33-34) with m = (xi/ue) due/dxi from the first two stations, clipped to [-0.05, 1]
  (m = 1 at a stagnation point, m = 0 on a flat plate). A fixed Hiemenz start put the flat plate
  6 % above Blasius; with the local m it is 1.1 % above.
- **Transition onset fraction w.** For a provisional Newton state with no root of the onset
  quadratic in [0, 1], w is taken at the nearer end of the interval (1 if N_crit is not reached,
  0 if already exceeded). Without this the first Newton step at onset produced NaN.
- **Inverse march (C1 only).** Started from a given state part-way along a branch
  (`start_state`), because ue = l xi/(Re theta^2) with delta* prescribed is ill-conditioned near
  the stagnation point (ue came out 1e5 at 5 deg).
- **Closure derivative test.** The test points avoid H_0 = 3 + 400/Re_theta, where the turbulent
  H* (DG87 eq. 18) has a continuous slope but an unbounded second derivative, so a central
  difference is inaccurate there. My test was mis-measuring, not the closure.

C0: all closure, Jacobian (JAX = complex step = finite differences) and flat-plate tests pass at
the SPEC 4.2 tolerances (laminar theta, delta* within 1.1 % of Blasius at x = 0.1; turbulent Cf
within 1.4 % of 0.0592 Re_x^-0.2).

C1 misses the targets of the plan (TE delta*, theta within 5 %, transition within 0.02 c, the 5 deg
lower-surface laminar separation). Cause located: on XFOIL's own dumped states, DG87 eq. (11)
(laminar Cf, transcription checked against the page image) gives Cf 1 % above XFOIL's at H 2.2-2.5,
3-4 % at 2.5-2.8, 12-15 % at 2.8-3.1 and 43 % at 3.1-3.5. Laminar H* (eq. 10) agrees within 0.6 %,
turbulent Cf (eq. 14) exactly, turbulent H* within 1 %. Our amplification closure integrated on
XFOIL's states gives XFOIL's transition (0.657 vs 0.657; 0.312 vs 0.293). LNE89 section 3.1 says
XFOIL's formulation is that of ISES [DG87] and gives no other laminar closure. So the installed
XFOIL 6.99 uses a laminar Cf that is not in the supplied sources. Stopped for the user's decision.

### Laminar closure choice on physics (2026-10-09): option B, Fidkowski paper

User's decision: option B with the Fidkowski paper (FID; paper only, never mfoil's or XFOIL's
code); choose on physics: adopt a FID laminar closure only where it is at least as close to the
exact Falkner-Skan relations as DG87 over H 2.2-4.0, otherwise stop. C1 targets unchanged.
FID transcribed from the page images (appendix pp. 17-19; equations pp. 5-8).

Exact reference: `tests/diagnostics/falkner_skan.py` (Hartree form, shooting; reproduces Blasius
f''(0) 0.4696, H 2.591 and Hiemenz 1.2326, 2.216; tested). Comparison:
`tests/diagnostics/laminar_closure_compare.py`. Max |error| against Falkner-Skan:

| quantity | closure | H 2.2-2.6 | 2.6-3.0 | 3.0-3.5 | 3.5-4.0 (absolute) |
|---|---|---|---|---|---|
| Re_theta Cf/2 | DG87 eq. 11 | 0.0106 (2.9 %) | 0.0022 (1.7 %) | 0.0060 (12 %) | 0.0072 |
| | FID eq. 56 | 0.0062 (2.8 %) | 0.0145 (11 %) | 0.0177 (36 %) | 0.0171 |
| | XFOIL dump | 0.0067 (3.1 %) | 0.0150 (12 %) | 0.0178 (31 %) | 0.0169 |
| H* | DG87 eq. 10 | 0.1 % | 0.05 % | 0.03 % | 0.03 % |
| | FID eq. 47 | 0.2 % | 0.4 % | 0.7 % | 0.9 % |
| | XFOIL dump | 0.2 % | 0.5 % | 0.7 % | 0.9 % |
| Re_theta 2C_D/H* | DG87 eq. 12 = FID eq. 61 (identical for H < 4) | 0.1 % | 0.3 % | 0.5 % | 0.4 % |

Result: FID's laminar Cf and H* are further from Falkner-Skan than DG87 above H 2.6 (FID's Cf
crosses zero at H 3.83, DG87's at 4.14, Falkner-Skan at 4.03); XFOIL's dumped values follow FID. By the rule, no
FID laminar closure is adopted; DG87 eqs. (10)-(12) stay. Stopped for the user before items 4-6
(item 6, the M9 README note, is added under Later).

### Two closure sets (user's decision, 2026-10-09)

**The rule of the previous section (adopt a FID closure only where it is at least as close to
Falkner-Skan as DG87) is replaced by the user.** Why: the comparison showed that XFOIL 6.99's
laminar fits (= FID eqs. 47, 56) are further from the Falkner-Skan similarity solutions than DG87
in adverse pressure gradient (Cf error 11-36 % against 2-12 % for H 2.6-3.5; zero Cf at H 3.83
against 4.14 for DG87 and 4.03 exact), yet XFOIL is the reference of SPEC 4.1 and can be compared
station by station only with the closures it uses. Neither physics nor agreement with XFOIL alone
decides, so both sets are kept, named and documented:

- `"fidkowski2022"`: everything as published in FID (laminar, turbulent, lag constants,
  amplification, state at transition, guards; wake at C4). Where FID says a term is the author's
  own (eqs. 58, 74, 76), it is implemented as published and marked. C1 and C2 targets and the
  SPEC 4.1 targets are judged with this set. Implemented and verified first.
- `"drela1987"`: DG87 with LNE89, as published. Run through the same C1, C2 and benchmark; no
  targets against XFOIL; numbers reported side by side.
- The equations, discretisation and Newton solver exist once; a set is a module with a fixed
  interface (`foil_lab.bl.closure_sets.INTERFACE`), selected by name. No formulas are mixed.
- The default set is the user's decision after C3, with both benchmark results. Until then the
  code's default is `"fidkowski2022"` (provisional, the set the targets are judged with).
- The Falkner-Skan table and the laminar closure comparison stay as committed diagnostics and as a
  test (`test_laminar_closures_against_falkner_skan_as_recorded`, errors recorded as measured, not
  targets); they go into the M9 validation document.

**Where the sets need a different form of an equation, and how one solver is kept:**

1. Lag equation. Written once in FID's form (FID eq. 8),
   (delta/C_tau) dC_tau/dxi = K (C_tau,EQ^1/2 - eta_D C_tau^1/2) + 2 delta (u_q - (1/ue) due/dxi).
   LNE89 eq. (23) is exactly the case K = 5.6, eta_D = 1, u_q = (1/(0.75 delta*)) [Cf/2 -
   ((H_k - 1)/(6.7 H_k))^2] (4/3 = 1/G_beta, 6.7 = G_A). Each set supplies K(U_s), eta_D, u_q.
2. Turbulent dissipation: FID's max(wall + outer + stress, laminar) (eqs. 62-65) and DG87 eq. (20)
   behind one function signature `cd_turb(hk, ret, cf, us, ctau, hstar)`.
3. Amplification: FID's rate depends on n (eq. 76). The onset is therefore defined for both sets
   as the root of the amplification residual on the laminar subinterval with the state
   interpolated to xi_t and n = N_crit there (FID eq. 18; DG87 p. 1351 states the same
   definition). This replaces my earlier closed form with the rate interpolated linearly in xi.
   Solved by 40 bisections (state-independent) plus one Newton correction, so derivatives follow
   the implicit function theorem (FID eq. 19) in NumPy, complex step and JAX alike.
4. State at transition: one function `ctau_transition(hk, ctau_eq)` (DG87: C_tau^1/2 = 0.7
   C_tau,EQ^1/2; FID eq. 17: 1.8 exp(-3.3/(H_k - 1)) C_tau,EQ^1/2).
5. Start of a branch: the local similarity form of the momentum and shape equations written with
   the set's own laminar closures, G [(1 - m)/2 + (2 + H) m] = Re_theta Cf/2 and G (1 - H) m =
   Re_theta (2C_D/H* - Cf/2), G = Re ue theta^2/xi, m from the first two stations clipped to
   [-0.05, 1]. At m = 1 these are FID's stagnation equations (p. 7); this replaces the DG87 eqs.
   (33)-(34) start for both sets (each set then starts consistently with its own closures).

Discretisation differences of FID not adopted (the discretisation exists once and stays the
central log form of DG87 p. 1351): the xi-weighted source terms (FID eqs. 9-10), the H_k-dependent
upwinding (eq. 13), the midpoint evaluation of cf xi/theta, the stagnation state extrapolated to
xi = 0 with a quadratic ue fit (p. 7). With the FID set, C1 agrees with XFOIL to about 1 % at the
trailing edge, so these are not needed for agreement; revisit only if C2 shows otherwise.
(Later reversed: eqs. 9-10 in C2, the quadratic ue fit in C2, the upwinding of eq. 13 after report 7.)

**Every difference between the sources (item 4), equation numbers, and evaluation on the dumps**
(`tests/diagnostics/closure_set_compare.py`; ratio closure/XFOIL on the dumped (H, Re_theta),
mean (max deviation)):

| quantity | drela1987 | fidkowski2022 | on XFOIL's dumped states |
|---|---|---|---|
| laminar H* | DG87 eq. 10 | FID eqs. 47-48 | DG87 0.999-0.995 (0.7 %); FID 1.0000 |
| laminar Cf | DG87 eq. 11 | FID eq. 56 | DG87 1.016 / 1.082 / 1.305 (H 2.2-2.6 / 2.6-3.0 / 3.0-3.5); FID 1.000 (0.6 %) |
| laminar C_D | DG87 eq. 12 | FID eq. 61 (same below H_k 4; -0.0016 vs -0.003 above) | not dumped |
| turbulent Cf | DG87 eq. 14 | FID eqs. 57-60 (eq. 14 with guards A_cf (58, author's own), B_cf >= 1.303) | both 1.000 |
| turbulent H* | DG87 eqs. 18-19 | FID eqs. 49-54 (other fit; Re_theta >= 200 inside) | DG87 0.990-0.998 (1.1 %); FID 1.0000 |
| U_s | DG87 eq. 21 | FID eq. 68 (same, max 0.98) | not dumped |
| delta | DG87 eq. 23 | FID eq. 55 (same, max 12 theta) | not dumped |
| C_tau,EQ | DG87 eq. 24 ((H_k-1)^3, 0.015) | FID eqs. 78-79 ((H_k-1) H_kc^2, 1/(2 G_A^2 G_beta) = 0.01485, H_kc = H_k - 1 - 18/Re_theta) | via the march, below |
| turbulent C_D | DG87 eq. 20 | FID eqs. 62-65 (low-Re wall factor, 0.995 - U_s, stress term, max with laminar) | via the march |
| lag | LNE89 eq. 23 (K = 5.6) | FID eqs. 8, 77 (K = 5.6/(0.75 (1 + U_s)): 5.74 / 4.98 / 4.39 at U_s 0.3 / 0.5 / 0.7; H_kc in u_q) | via the march |
| amplification rate | DG87 eqs. 29, 33-35 | FID eqs. 69-71, 75 | theta dn/dxi equal within 4 % for H_k 2.6-4 |
| critical Re_theta | DG87 eq. 30 (step) | FID eqs. 72-74 (smooth ramp over +-0.1 decades; eq. 74 author's own), eq. 76 (author's own) | transition on XFOIL's states: DG87 0.657 / none / 0.312, FID 0.658 / 0.684 / 0.296 (XFOIL 0.657 / 0.681 / 0.293) |
| C_tau at transition | DG87 p. 1351 (0.7 C_tau,EQ^1/2) | FID eq. 17 (0.23 C_tau,EQ^1/2 at H_k 2.6) | via the march |
| H_k limit | none in DG87 | FID p. 18: 1.05 on the airfoil | - |
| wake | BTE89 eqs. 13-18, 20-32 | FID eqs. 14-15, 66-67, eta_D 0.9 | C4 only |

March on XFOIL's ue with transition forced at XFOIL's location (the turbulent part, where the
lag, C_D, C_tau,EQ and transition-state differences act): TE delta*/XFOIL = 1.006, 1.004, 1.009
with fidkowski2022 against 1.056, 1.048, 1.131 with drela1987 (0 deg upper, lower, 5 deg upper):
the 5-13 % excess of report 3 came from the turbulent and transition closures of the DG87 set,
not from the solver.

**C1 with "fidkowski2022"** (targets unchanged): transition 0.657 / 0.682 / 0.294 / laminar (XFOIL
0.657 / 0.681 / 0.293 / laminar); TE delta* ratio 1.006, 1.001, 1.009; theta 1.007, 1.002, 1.007;
H diff -0.002, -0.002, +0.004; inverse max |ue diff| 0.0046, 0.0082, 0.0048, 0.0001. 5 deg lower:
laminar separation reproduced (H 3.98 against 3.96 at x/c 0.99); the direct problem has no root at
the last station (x/c 0.9999; residual floor 3.8e-5 at H 4.12; roots exist for ue 0.05 % higher),
so the direct TE values there (delta* 1.053, H +0.21) are those of a non-converged station; the
inverse check (plan change 3) gives theta 1.000, H -0.001, ue within 0.0001 there. Counted as
passed under the planned inverse check; flagged to the user.

March fix found by C1: a laminar step without a solution now also triggers the transition interval
(accepted only if the onset lies inside the interval). Before, transition at 0 deg lower was
missed when XFOIL's ue was already transitional (0.714 instead of 0.682).

### C2: coupled Newton solver (2026-10-09), decisions and results; stopped

`foil_lab/viscous.py`, `ViscousParams` in `params.py`. Changes against the formulation fixed
before C1, each forced by a C2 failure (histories in the report), each from FID unless marked:

1. **Augmented state** (FID p. 6, eq. 27) instead of my earlier choice of m = ue delta* with ue
   eliminated: unknowns (theta, delta*, n or C_tau, ue) per station, residuals the three BL
   equations and R_u = ue - ue_inv - D (ue delta*). With m as unknown, delta* = m/ue is
   ill-conditioned at the stations next to the stagnation point (ue -> 0): cold starts at 0, -4,
   10 deg stalled at residuals 50-600. With the augmented state they converge.
2. **Stagnation point** from the two adjacent speeds, s_stag = s_k + ds ue_k/(ue_k + ue_k+1)
   (FID p. 6), its derivative in the Jacobian (FID eq. 28). First interval: the stagnation
   equations (ue = K xi) on the state extrapolated linearly to xi = 0, K from a quadratic fit
   (FID p. 7), replacing the similarity relations at the first station (which gave an odd-even
   H oscillation behind the stagnation point, as FID describes for XFOIL).
3. **Source terms integrated in ln xi** (FID eqs. 9-10) for momentum and shape, for both sets,
   replacing dxi times the average. When the stagnation point comes close to a station, Cf and C_D
   grow like 1/xi and the old form became inconsistent on the first interval (a residual jump to
   90 when the stagnation point moved one panel). This reverses the "not adopted" entry of the
   two-sets section for these two terms; the lag and amplification equations keep dxi (FID eqs.
   11-12). C0 and C1 rerun after the change: all pass (C1 TE delta* ratios 1.007, 1.000, 1.009,
   1.011; transition 0.657 / 0.682 / 0.294 / laminar).
4. **Step limits** of FID p. 11 (theta, delta* not below half; n up <= 2, n > 0.2 down <= 80 %;
   C_tau^1/2 up <= 0.05, down <= 80 % where C_tau^1/2 > 0.1 max; ue <= 20 %), then halving for
   positivity, H_k >= hk_min and turbulent Re_theta >= re_theta_min. Mine: the ue limit and the
   H_k check are not applied at the two stations next to a stagnation point where ue changes sign
   (a sign change only moves the stagnation point). Each scaled step and its binding limit is
   counted in the result (`step_limits`).
5. **Branch change** (mine): a station that changes branch keeps its speed (sign with d) and takes
   theta and delta* of its downstream neighbour on the new branch, n = 0 (Hiemenz: theta, delta*
   nearly uniform at a stagnation point). Replaces "keeps its values" of the plan.
6. **Transition re-identification** after each update by marching n (FID p. 8); the implicit
   trapezoid of that march is iterated to round-off. With 8 fixed-point iterations it left n
   1e-7 off the Newton solution and 7 deg stalled at a residual 8.3e-8 (> 1e-8) with full steps
   (history in the report); a bug, fixed, no tolerance changed.
7. **Angle-of-attack continuation** (FID section VIII.C): a warm-started point that fails is
   reached through intermediate angles (increment halved, at most three times); iterations of
   the intermediate solves are counted in the point's iterations.
8. **Starting march** (FID section V.A, eqs. 20-21; `march_init`): direct, inverse with prescribed
   H_k where the direct Newton fails or H_k > 3.8 (laminar) / 2.5 (turbulent), else scaling. Mine:
   the H_k switch is not applied on the transition interval itself (its downstream station still
   carries the laminar H_k; applying it corrupted the start at 5 deg), and the fallback sets
   C_tau from the set's transition value when the station turns turbulent. The first station of
   the starting march uses m = 1 (next to a stagnation point).
9. New parameters (proposed defaults, awaiting approval): `hk_min` 1.02, `re_theta_min` 20 (FID
   eq. 59 guard: log10 Re_theta >= 1.303), `init_hk_max_lam` 3.8, `init_hk_max_turb` 2.5 (FID
   p. 9), `jacobian` "complex_step" (0.08 s per Jacobian against 7.9 s for eager JAX at 161
   stations; both tested). `shape_eq_bias` stays 0.5: 0.6 and 1.0 did not help the stagnation
   oscillation, which item 2 removed.

Tests (`tests/test_viscous.py`): complex-step Jacobian = central finite differences of the full
residual (1e-5 of each column), JAX = complex step (1e-8; measured 2e-9), cold start at 5 deg
converges in under 20 iterations with quadratic tail, 4 deg from the march and from 2 deg give the
same Cl, Cd, Cm, transition (1e-7).

**C2 results, NACA 2412, Re 1e6, "fidkowski2022"** (`tests/diagnostics/c2_polar.py`): converged at
every angle -4 to 15 deg (16 deg not; attached range per the benchmark filter: below 12 deg).
Against the C2 targets: iterations < 20 missed at -2 (180, continuation), -1 (65, continuation),
0 (22, cold start), 14 (116, continuation); |dCl| <= 0.02 at 0-10 deg missed at 1-4 deg (-0.021 to
-0.047) and 8-10 deg (-0.054 to -0.151); Cl_max 1.24 against XFOIL 1.52 (missed); |dCm| <= 0.01
missed at 3 deg and from 8 deg; Cd within 15 % up to 9 deg (-1 to -14 %), missed from 10 deg
(-18 to -24 %).

Cause located: the missing wake. With XFOIL's own delta*, our core gives XFOIL's Cl to 0.002, and
the wake adds only 0.006 (10 deg) and 0.019 (14 deg) directly; but it raises the upper edge
speed near the trailing edge by 0.02 at x/c 0.95 and 0.04-0.08 at 0.995 (5-14 deg). Without it the
trailing-edge deceleration thickens the coupled layer over the last 10 % of chord: at 0 deg
delta* is within 2 % at x/c 0.5 and 4 % at 0.9 but 34-39 % high at the trailing edge, with TE ue
0.838 against 0.887; at 10 deg the upper layer separates at the trailing edge (H 4.6 against
2.7), which lowers circulation and Cl. Proposed: take C4 ("add the wake") before C3.

"drela1987" (no targets): C1 as in the two-sets section; C2 converged -4 to 5 deg, not from 6 deg;
at 0-5 deg dCl -0.014 to -0.070, Cd +1 to +13 %.

### C4: the wake (2026-10-10), decisions and results

**Why the wake is required (user's decision after report 5).** Not for its direct share of lift
(0.002 at 5 deg, 0.006 at 10 deg with XFOIL's delta*) but for its effect on the edge velocity near
the trailing edge: with the same delta*, the wake raises the upper-surface ue by 0.015 at x/c 0.95
and 0.04-0.08 at 0.995 (5-14 deg). Without it the trailing-edge deceleration thickened the coupled
layer over the last 10 % of chord (TE delta* +34-39 % at 0 deg, upper-surface separation at 10 deg).
With the wake the C2 trailing-edge delta* is +7-9 % at 0 deg, +5 % at 5 and 10 deg (upper surface).

**Formulation, with sources** (`foil_lab/wake.py`, `foil_lab/viscous.py`, `foil_lab/bl/march.py`):

- Wake line, node count, offset, length and source distribution: FID p. 3 (Fig. 1, eq. 2), p. 7;
  LNE89 pp. 7-8. Edge velocity over airfoil and wake stations: FID eqs. 23-27, rebuilt per angle.
- First wake station: FID eq. 14 (BTE89 eqs. 16-18 / LNE89 eqs. 28-29 for "drela1987"); a laminar
  trailing edge enters with the set's C_tau at transition (FID p. 7). Dead-air thickness: FID eq. 15
  (dt/dxi clipped to +-3/f_w) and BTE89 eqs. 20-22 (no clip); closures of delta* - h_w, integral
  equations with the full delta* (FID p. 7; BTE89 p. 62). Wake closures in each set's module.
- Starting march along the wake: FID p. 9 (direct; inverse with eq. 22; extrapolation).
- Drag: Squire-Young at the last wake station (FID p. 12); the trailing-edge Squire-Young value is
  kept as `cd_te` and reported as a cross-check.

Our choices (not stated in the sources):

1. The first wake half-panel starts at the trailing-edge midpoint (FID Fig. 1 draws the sheet from
   the trailing edge), not at the first wake node eps_w c downstream: starting it at the node puts
   a logarithmic singularity of the tangential velocity there (with XFOIL's delta* at 5 deg: ue 0.82
   at the first wake node against XFOIL's 0.91; with the midpoint start 0.93; downstream nodes
   unchanged and within 0.001 of XFOIL's from x/c 1.05).
2. First wake spacing = mean length of the two trailing-edge panels (0.0004 c on the canonical
   160-panel files). FID gives the node count and total length only. **Open, see below.**
3. Eq. 14 written as ratios minus one (dimensionless residuals, as the rest of the system); the
   C_tau residual as (C_tau,w/C_tau,start)^1/2 - 1 for both sets.
4. In the wake xi continues the upper surface (FID p. 3: the wake extends the upper surface):
   xi = xi_TE,upper + distance along the wake; it enters only the ln xi source terms (FID eqs. 9-10).
5. Starting march in the wake: inverse mode above `init_hk_max_turb` (FID gives no wake value); the
   extrapolation fallback applied to delta* - h_w.
6. The velocity of one body's wake sources at another body's wake is not included (one body in M3;
   to add in M5).

Changes forced by C4 failures:

- **H_k and C_tau clipped after the update, as FID p. 11 does** (delta* raised so that H_k >=
  `hk_min`, in the wake the closures' H_k; negative C_tau^1/2 set to 0.1 max C_tau^1/2), replacing
  the step halving for these two of C2 item 4. With halving, cold starts with the wake stalled at
  step factors of 1e-15 (2 deg drela1987, 4 deg fidkowski2022: the Newton direction is right but its
  full length drives H_k below the bound at the trailing-edge stations). Halving remains for theta,
  delta* > 0 and Re_theta >= `re_theta_min`. Every clip is counted in `step_limits`.
- **Polar continuation from the last converged state** after a failed point (FID VIII.C: "a solution
  ... initializes ...", "different starting angles of attack and fixed or adaptive increments"),
  instead of a new starting march. Cold starts fail more often with the wake (fidkowski2022 cold:
  converged at -4, -3, -1..6, 10 deg; not at -2, 7-9, 11, 12 deg) because the march on the inviscid
  edge velocity meets the blunt-base deceleration at the finely panelled trailing edge (inviscid ue
  0.86 -> 0.74 over the last 0.6 % of chord). A segregated re-march was tried and rejected (it
  converged -2 and 4 deg but broke 0 deg).

**Approved ViscousParams defaults (user, after report 5), with sources:**

| parameter | default | source |
|---|---|---|
| `hk_min` | 1.02 | FID p. 11: H_k,min 1.02 in the wake (1.00005 on the airfoil) after the update; one value everywhere is our choice |
| `re_theta_min` | 20 | FID eq. 59 (log10 Re_theta limited to 1.303 in the turbulent Cf); applying it as a bound on the Newton step is our choice |
| `init_hk_max_lam` | 3.8 | FID p. 9 |
| `init_hk_max_turb` | 2.5 | FID p. 9 (also used in the wake: our choice) |
| `jacobian` | "complex_step" | our choice (0.08 s per Jacobian against 7.9 s for eager JAX at 161 stations; both tested) |

**Iterations (user's decision):** a point is judged by the Newton iterations of its final solve; the
total including failed attempts and intermediate continuation solves is always reported
(`ViscousResult.iterations`, `iterations_total`).

**C2 rerun with the wake ("fidkowski2022"): passed, targets unchanged.** Converged at every
attached angle (-4..11 deg), at most 16 iterations per final solve (totals up to 286 with
continuation); abs(dCl) max 0.0133 at 0-10 deg; Cl_max 1.502 vs 1.525 at 16 deg; abs(dCm) max
0.0068 (0.0030 attached); Cd -9.9..+9.6 % (Cd TE -11.8..-0.2 %). The 2-5 deg lower-surface
transition now matches XFOIL (0.963 / 0.990 / laminar / laminar against 0.966 / 0.994 / 1.000 /
1.000) and dCl there is -0.004..+0.012 (was -0.023..-0.047). Tables in the M3 report 6.

**Open decisions (for the user; C3 is run with the current choices and reports both drag values):**

1. *First wake spacing and the wake-end drag.* The wake-end Cd depends on the first wake spacing: at
   0 deg, +12.0 / +9.5 / +3.4 / -5.5 / -25 % against XFOIL for 0.0001 / 0.0004 / 0.002 / 0.005 /
   0.01 c, and on panel refinement of the same section (+9.5 / +10.8 / +11.4 % at 160 / 240 / 320
   panels), while `cd_te` stays within -0.5..-2.6 % of XFOIL's Cd throughout. Cause: our edge speed
   at the first wake node differs from that at the trailing-edge nodes (0.899 against 0.874 at 0
   deg); FID eq. 14 carries theta across that jump without a momentum term, and the following
   deceleration adds about 9 % to the far-wake theta. Our wake equations on XFOIL's own wake edge
   speed reproduce XFOIL's wake theta to 0.6 %, and XFOIL's own wake-end Cd exceeds its trailing-edge
   Squire-Young value by only 0.5 / 2.6 / 4 % (0 / 5 / 10 deg).
2. *drela1987 in the wake.* With the DG87 closures the far wake (x > 1.45 c) wants H_k < 1.02 and
   ends on the approved `hk_min` clip; no point of that set converges with the wake (every other
   residual is ~0).

### C3: full benchmark with the wake (2026-10-10), results; stopped

`python -m foil_lab.benchmark viscous` (`foil_lab/benchmark/viscous_run.py`; one "viscous" method per
closure set, Q5); tables with `viscous-report` in `benchmark_output/m3_viscous/report.md`. Sweeps as
the XFOIL reference (0 deg up to 20, 0 deg down to -4, each angle from the last converged one with
continuation), same canonical files, metrics and attached-flow filter as the baseline report; both
the wake-end Cd and the trailing-edge Squire-Young value are compared.

Change made during C3 (a failure of the formulation, not a tolerance):

- **Stagnation point on a node** (FID p. 7, "in the unlikely case that the stagnation point
  coincides, to machine precision, with a panel node, the stagnation equations apply to that node").
  A symmetric section at 0 deg puts the stagnation point on the leading-edge node to round-off; ue = 0
  there, the starting march raised "similarity start failed" and the log-form interval from that
  node is singular. Now (our reading of FID p. 7): when a branch's first node has a sheet strength
  below 1e-9 of the next node's (`STAG_NODE_TOL`), the stagnation equations apply at the next node
  (state extrapolated to xi = 0 as before) and the node on the stagnation point carries that node's
  theta and delta* (Hiemenz flow) with n = 0. NACA 0012 at 0 deg now converges at every Reynolds
  number but 5e5 (8-10 iterations, Cl = 0 exactly, symmetric layers; test added).

Results ("fidkowski2022", the set the targets are judged with; open trailing edges, attached-flow
filter, per Reynolds number; targets: converged >= 95 %, mean |dCl| <= 0.03 / max <= 0.10, mean rel
|dCd| <= 10 % / max <= 30 %, mean |dCm| <= 0.01 / max <= 0.03):

| Re | converged | mean / max dCl | mean / max rel dCd (wake end) | (TE value) | mean / max dCm |
|---|---|---|---|---|---|
| 5e5 | 69/83 = 83.1 % (missed) | 0.0040 / 0.0188 | 4.7 % / 9.6 % | 2.4 % / 5.9 % | 0.0008 / 0.0028 |
| 1e6 | 78/91 = 85.7 % (missed) | 0.0054 / 0.0196 | 5.8 % / 10.5 % | 1.8 % / 9.7 % | 0.0011 / 0.0032 |
| 3e6 | 74/102 = 72.5 % (missed) | 0.0068 / 0.0312 | 7.2 % / 12.4 % | 1.4 % / 12.7 % | 0.0013 / 0.0051 |
| 6e6 | 41/110 = 37.3 % (missed) | 0.0040 / 0.0144 | 8.9 % / 14.2 % | 1.5 % / 6.5 % | 0.0008 / 0.0031 |

All accuracy targets are met on the converged points at every validated Reynolds number; the
convergence target is missed at all four. Where it fails (diagnosed cases): the transition location
moves far between neighbouring angles (XFOIL's too: NACA 4412 lower surface 0.40 / 0.67 / 0.98 at
2 / 3 / 4 deg, Re 6e6; 0.80 / 0.35 at 0 / -1 deg, Re 5e5), and the transition re-identification
inside the Newton iteration does not survive it even with continuation down to 0.125 deg; the
iterate ends with a clipped station (H_k = hk_min, a local ue spike) or oscillates. At Re 3e6-6e6
mid-to-high angles also fail where transition is near the leading edge (NACA 0012 Re 3e6 from 10
deg: the turbulent station after a laminar-separated transition station is clipped at H_k 1.02 at
every update; FID's airfoil value 1.00005 does not change this).

E387 (sharp trailing edge, no targets): converged 9/13, 7/13, 4/14, 1/16 at Re 5e5-6e6. At Re 1e6,
-4 and -3 deg are reported converged (residual criteria met) on a non-physical solution (Cl 0.19 and
0.18 against XFOIL -0.063 and 0.046; wake-end Cd 0.20 with a trailing-edge value of 0.0001), reached
by a down sweep whose 0 to -2 deg points failed. The flag is truthful by its definition; the
solution is not.

"drela1987" (no targets): converged 2/83, 1/91, 1/102, 0/110 at Re 5e5-6e6 (open trailing edges).
With the DG87/BTE89 closures the far wake drives H_k to 1 within one chord behind the trailing
edge; with `hk_min` 1.02 the last wake stations end on the clip; with 1.00005 (tried, not adopted)
only the last station does. Its numbers are not usable until the user decides how this set treats
the far wake.

Stopped here for the user (C3 instruction): the convergence target is missed for "fidkowski2022"
at every validated Reynolds number, with every accuracy target met on the converged points; open
decisions in C4 and above.

### Decisions after report 6 (user, 2026-10-10)

1. **Wake drag.** The drag definition (Squire-Young at the end of the wake) and the first wake
   spacing rule stay unchanged for now. First the airfoil/wake source interface is checked against
   FID (below, "Airfoil/wake source interface").
2. **"drela1987" parked.** "Not converged with the wake" is accepted as this set's current state. The
   set, its laminar Falkner-Skan comparison and its tests are kept; it is excluded from benchmark
   runs (`viscous_run.CLOSURE_SETS = ("fidkowski2022",)`; `run_matrix` refuses a parked set and the
   report ignores old rows of it). Its wake treatment is decided after v2.0 (Later). No more M3 time
   is spent on it.
3. **Convergence of "fidkowski2022".** Non-converged attached-flow points are classified by cause;
   the largest class is compared with XFOIL's boundary-layer dump before anything changes. Measures
   that only change how a solution is reached (limiting how far transition moves per Newton
   iteration, finer or adaptive continuation, restarting from the starting march after a failed
   warm start) may be implemented without asking, provided every point that converged before
   converges to the same Cl, Cd, Cm and transition within 1e-6, tolerances and the definition of
   converged are unchanged, each measure is recorded here with its source or "our choice", and the
   result records how it was reached. Anything that changes a converged solution (closures, clips
   or limits on H_k or C_tau, the transition formulation, the discretisation) needs the user's
   approval first.
4. E387 (sharp trailing edge) at Re 1e6, -4 and -3 deg converges to a non-physical solution (Cl
   0.19 against XFOIL -0.063): looked at at the end of M3, after the convergence work (Later).

### Airfoil/wake source interface against FID (decision 1 after report 6; open)

What FID specifies (page images, pp. 3, 7, 16): (a) the source strength at the first wake node is
the sum of the first and last airfoil panel sources, "in order to avoid a velocity spike at the
trailing edge" (p. 3); the strength is linear over each wake half-panel (eq. 2), i.e. from
sigma_1 + sigma_N-1 at the first wake node to sigma_1^w at the first panel's midpoint (Fig. 1); (b)
the wake nodes, the first eps_w c = 1e-5 c behind the trailing-edge midpoint, are the panel ends,
so the first half-panel starts at the first wake node (p. 7, Fig. 1); (c) when an evaluation point
lies on a panel endpoint, the ln r of that endpoint is set to zero (appendix p. 16; stated for the
streamfunction, and the velocity follows the same paragraph); (d) the wake edge speed sums the
freestream and all vortex and source sheets (eq. 24). The paper says nothing more about the
interface.

What we do (`wake.py`): (a) and (d) as FID. (b) and (c) differ: the first half-panel starts at the
trailing-edge midpoint (C4 our choice 1), so the first wake node lies 1e-5 c inside the sheet, and
velocities are evaluated 1e-9 off the sheet (never on an endpoint).

FID read literally ((b) and (c)) was implemented in a diagnostic (`tmp`, not committed) and is
worse: with XFOIL's own delta* the first-node speed is 1.014 / 1.071 / 1.152 (0 / 5 / 10 deg; ours
0.936 / 0.939 / 0.947; XFOIL 0.887 / 0.907 / 0.936); coupled (NACA 2412, Re 1e6) it does not
converge at 0 and 10 deg and gives Cd -41 % at 5 deg with TE ue 0.944 against 0.907. It fails C2,
so it was not adopted, and ours stays until the user decides.

Cause of the jump at the first wake node, from a decomposition on XFOIL's delta*: the dead-air
closure (FID eq. 15) enters the wake mass flow, so the first wake half-panel carries a sink of
about ue dh_w/dxi ~ -0.25 while the first-node strength (rule (a)) holds only the airfoil
boundary-layer sources; the strength ramps over one half-panel, which raises ue near its start
(+0.05 over the trailing-edge value on XFOIL's delta*), and the finer the first spacing, the more
of that ramp the first nodes see. XFOIL's own dump shows the first wake point (1e-4 c behind the
trailing edge) with exactly the trailing-edge speed and its next point 0.0083 c downstream.

Coupled, ours (NACA 2412, Re 1e6, 1 deg steps; trailing-edge ue / first wake node, XFOIL equal at
both): 0.874 / 0.899 (XFOIL 0.887), 0.904 / 0.917 (0.907), 0.935 / 0.938 (0.936) at 0 / 5 / 10 deg;
TE delta* ratio upper / lower 1.087 / 1.074, 1.051 / 1.274, 1.056 / 1.035; theta ratio 1.057 /
1.053, 1.030 / 1.003, 1.016 / 1.014. Wake-end Cd against XFOIL at 0 / 5 / 10 deg:

| first wake spacing | 0.0001 c | 0.0004 c (rule) | 0.002 c | 0.005 c | 0.01 c |
|---|---|---|---|---|---|
| Cd | +12.0 / +8.8 / +3.5 % | +9.5 / +5.3 / +0.4 % | +3.4 / -1.7 / -5.8 % | -5.5 / -9.8 / -12.6 % | -25.4 / -25.9 / -24.6 % |

| panels (generated, open TE) | 160 | 240 | 320 |
|---|---|---|---|
| Cd | +9.5 / +5.3 / +0.4 % | +10.8 / +7.4 / +2.8 % | +11.4 / +8.7 / +4.2 % |

Cd TE stays within -0.5..-4.9 % in all of these. The sensitivity remains: stopped for the user (the
drag definition and the spacing rule are unchanged).

### Convergence of "fidkowski2022" (decision 3 after report 6)

**3a. Classification** (`tests/diagnostics/c3_failures.py`: replays C3 exactly through
`viscous_run.run_case`, observing, not changing, the clip and transition steps after every Newton
update; classes judged on the last 15 iterations of the failed direct attempt; thresholds are the
diagnostic's own). The first pass reproduced C3's 14 / 13 / 28 / 69 failures at Re 5e5 / 1e6 / 3e6
/ 6e6 exactly. Its "other" class (39 of 124) turned out to be mostly one mechanism, added as a
class: an **isolated H_k clip in the turbulent layer** (one turbulent airfoil station, 6-27
stations behind transition, held at H_k = 1.02 between neighbours at H 1.38-1.48; the iterate is a
fixed point, unchanged to 1e-10 between iterations: the Newton step drives that station below the
clip and the clip puts it back). Counts: report 7.

**3b. Comparison with XFOIL**, NACA 4412, Re 5e5, -1 deg (dump in `tests/diagnostics/data`,
`transition_compare.py`): the failed iterate equals XFOIL's state within about 1 % from the leading
edge to x/c 0.80 (transition 0.356 against 0.354; H 1.465 against 1.466 at x/c 0.60) except one
station at x/c 0.85 (H_k 1.02 against 1.471, theta 4.48e-3 against 2.07e-3, ue 0.693 against 0.933,
ue overshoots to 0.99 on both sides). x/c 0.85 is where the warm start (0 deg) was finishing its
transition (0.796); XFOIL's transition also jumps from 0.80 to 0.35 between 0 and -1 deg. The
defect is a remnant of the previous solution held by the clip, not another physical state.
NACA 0012, Re 3e6, 10 deg (the class "H_k clip after laminar-separated transition"): the laminar
separation bubble is reproduced (H 4.68 against 4.60 at x/c 0.014); our onset falls at a node
(0.0187, node 0.01877), so the whole reattachment from H 4.6 lies in one fully turbulent interval
whose downstream station is clipped at every iteration (XFOIL: H 3.03 and 1.80 on its first two
turbulent stations); transition does not move. Changing this touches the discretisation or the
clip (decision 3d): not done, for the user.

**3c. Measures (only how the solution is reached; `viscous.solve_point`, used by `polar`):** after
the existing warm start with continuation fails,

1. *repair restart* (our choice): restart Newton from the failed iterate with every isolated
   station on the H_k clip (`isolated_clips`: turbulent, both neighbours turbulent and above the
   clip) replaced by the mean of its two neighbours (theta, delta*, C_tau; ue kept), at most twice
   in a row;
2. *restart from the starting march* (user's list; FID section V.A for the march), followed by
   repair restarts of it.

The equations, `hk_min` and the clip, the tolerances and the definition of converged are
unchanged. `ViscousResult.path` records how a point was reached ("warm start", "continuation",
"march", "repair restart", "march restart", "march restart + repair restart"); `iterations` are
the converged attempt's, `iterations_total` all of them; the benchmark CSV has a `path` column and
full-precision floats. Experiments before adopting them (from the stuck iterates): smoothing the
clipped station converged in 4-5 iterations at E387 gap Re 5e5 7 deg, NACA 23012 Re 1e6 -3 deg and
NACA 4412 Re 5e5 -1 deg, to the cold-march solution where that converges (equal Cl, Cd,
transition); NACA 0012 Re 6e6 3 deg did not (a period-2 oscillation of the lag residual of the
transition interval, no clip involved). Tests: `test_isolated_clips_finds_one_station_spikes`,
`test_restart_reaches_the_cold_start_solution` (slow; to 1e-6).

**C3 rerun with the measures** (`c3_failures --write-benchmark`, i.e. `run_case` itself; 49 cases,
about 58 min on 6 workers, 5.04 h summed over the cases against 5.83 h before): all 868 points that
converged before converge again, to the same Cl, Cd, Cm and transition within 2.4e-11 (condition:
1e-6); 162 points converge that did not. SPEC.md 4.1 table (open trailing edges, attached filter):

| Re | converged | mean / max dCl | mean / max rel dCd (wake end) | (Cd TE) | mean / max dCm |
|---|---|---|---|---|---|
| 5e5 | 81/83 = 97.6 % (met) | 0.0037 / 0.0188 | 7.6 % / 147.7 % (max missed) | 5.2 % / 152.0 % | 0.0008 / 0.0028 |
| 1e6 | 87/91 = 95.6 % (met) | 0.0051 / 0.0196 | 5.9 % / 11.9 % | 2.0 % / 9.7 % | 0.0010 / 0.0032 |
| 3e6 | 90/102 = 88.2 % (missed) | 0.0071 / 0.0312 | 7.2 % / 14.3 % | 1.5 % / 12.7 % | 0.0014 / 0.0051 |
| 6e6 | 84/110 = 76.4 % (missed) | 0.0074 / 0.0240 | 8.2 % / 14.6 % | 2.0 % / 10.8 % | 0.0014 / 0.0041 |

The max Cd miss at 5e5 comes from two points that converge for the first time, E387 with the
0.25 % gap at -3 and -4 deg (Cd 0.01107 / 0.01313 against XFOIL 0.00447 / 0.00714; Cl within
0.0035), reached by a plain warm start from -2 deg (now converged by a repair restart). XFOIL's Cd
there depends strongly on its paneling (0.00447 / 0.00714 at 160 PANE nodes, 0.00616 / 0.01611 at
240; -2 deg 0.00965 / 0.00961). Reported as measured; nothing excluded.

Failures after, by class (Re 5e5 / 1e6 / 3e6 / 6e6): cold start 2 / 3 / 2 / 2; H_k clip after
laminar-separated transition 0 / 1 / 10 / 16; transition relocation 0 / 0 / 0 / 1; isolated clip
0 / 0 / 0 / 4; other 0 / 0 / 0 / 3. The remaining main class (27 of 44) is the one 3b showed needs
a discretisation or clip change (3d): for the user.

### Decisions after report 7 (user, 2026-10-10)

1. **Wake drag** stays the wake-end value. First test the explanation (open trailing edge: the
   first wake stations lie in the dead-air region behind the base, where the wake-line velocity is
   not an edge velocity; the edge velocity jumps across the trailing edge, 0.874 to 0.899 at 0 deg,
   with no momentum term for the jump): (a) the same spacing and panel tables on the closed NACA
   2412, where the sensitivity should then be much smaller, with the edge velocity across the
   trailing edge; (b) if confirmed, a rule (continuous edge velocity across the trailing edge, no
   wake station sampling the wake-line velocity inside the dead-air length of FID eq. 15, the
   current treatment for a sharp edge), recorded as our choice; (c) accepted if the wake-end Cd
   changes by at most 2 % over 160-320 panels and reasonable spacings beyond the dead-air region at
   0, 5, 10 deg, with C2 still met; (d) otherwise stop and show; never tune the spacing to XFOIL.
2. **FID eq. 13 upwinding** adopted as published in the shared discretisation; C0, C1, C2 rerun
   (targets unchanged); the movement of the previously converged C3 points and the failure
   classification reported. If the "clip after laminar-separated transition" class stays above 5
   points: FID's airfoil H_k limit for "fidkowski2022" next; anything beyond the paper needs approval.
3. **Repair restart** accepted as our choice; the `path` column stays; the benchmark report shows the
   error metrics separately for points reached directly and by a restart.
4. **Reference reliability**, proposed SPEC.md 4.1 addition, applied in the report now (SPEC.md is
   edited only after the user confirms): "A reference point where XFOIL's own result at 240 nodes
   differs from 160 nodes by more than the target for a quantity is reported but not counted
   towards that quantity's targets. The number of such points is shown per case." Applied to all
   sections; tables with and without the rule.

### H_k-dependent upwinding, FID eq. 13 (decision 2 after report 7)

Implemented in `bl/equations.py` as FID p. 6 states it (page image): eta_up = 1 - (1/2) exp(-(ln
|f_Hu|)^2 C_up/H_k2^2), f_Hu = (H_k2 - 1)/(H_k1 - 1), C_up = 5 on the airfoil and 1 in the wake (taken
from the downstream station), q_u = (1 - eta_up) q1 + eta_up q2; 0.5 (central) for equal H_k, towards 1
(backward Euler) where H_k changes strongly. Which terms (FID p. 6): momentum all averaged; shape
equation: the Cf and C_D source terms upwinded, the coefficient of ln(ue2/ue1) averaged; lag equation:
C_tau^1/2 and C_tau,EQ^1/2 upwinded, delta averaged, u_q from the upwinded Cf and H_k with the averaged
delta* and Re_theta; dn/dxi averaged. Our reading where FID is silent: K_lag and eta_D averaged (FID:
"quantities without a subscript are averaged"); the lag equation is FID eq. 12 divided by the averaged
delta; both subintervals of a transition interval use the same rule. Not part of eq. 13 and not
adopted: FID's additional midpoint evaluation of cf xi/theta in the momentum equation, and its
ln(ue2/ue1) form of the due/dxi term of the lag equation (ours stays 2 dxi (due/dxi)(1/ue)_avg). The
`shape_eq_bias` parameter (0.5, central, never changed) is replaced by eq. 13 and removed. Test
`test_upwind_factor_fid_eq13`; the interval Jacobian tests (JAX = complex step = finite differences)
pass unchanged.

- C0: passes. C1 (targets unchanged), trailing-edge delta* / theta ratio against XFOIL: 0 deg upper
  1.001 / 1.001 (before 1.007 / 1.008), lower 0.997 / 0.997 (1.000 / 1.001), 5 deg upper 1.000 / 1.000
  (1.009 / 1.007), 5 deg lower (laminar separation) 1.001 / 1.000, H +0.002 (1.011, H +0.042);
  transition 0.657 / 0.682 / 0.294 / laminar unchanged.
- C2 (NACA 2412, Re 1e6): converged at every angle -4..16 deg (15 deg now too); final-solve iterations
  <= 18; |dCl| <= 0.0132 at 0-10 deg; Cl_max 1.530 at 16 deg (XFOIL 1.525); |dCm| <= 0.0028; Cd
  -3.1..+8.9 % over the attached range (-4..11 deg), -15.1 % at 16 deg (beyond it; before -9.9 %):
  flagged to the user, as "Cd within 15 %" was judged over all angles in report 6.
- `test_solution_does_not_depend_on_the_start`: the 3 deg start from the march now stalls on an
  isolated H_k clip; the test takes the 3 deg solution as a polar point (`solve_point`, repair
  restart); 5 deg from it converges directly and equals the cold 5 deg solution to 1e-14 (assertion
  1e-7 unchanged).
- C3 replay (`c3_failures --write-benchmark`): previously converged points move (open trailing edge,
  validated Re, attached filter, 335 points converged before and after): Cl mean 0.0019 / max 0.019
  (NACA 4412 Re 6e6 13 deg); Cd (wake end) relative mean 1.7 % / max 7.4 % (NACA 0012 Re 3e6 13
  deg), Cd TE 1.8 % / 7.4 %; Cm mean 0.0003 / max 0.0027; transition mean 0.0006 c / max 0.022 c.
  7 points converged before and not now (NACA 23012 Re 1e6 0, 1 deg; E387 gap Re 1e6 0, 1 deg; Re
  6e6 0, 4, 5 deg), 40 converge that did not.
- Failure classes after (Re 5e5 / 1e6 / 3e6 / 6e6, attached, open): not converged 0 / 5 / 0 / 6
  (report 7: 2 / 4 / 12 / 26); cold start 0 / 5 / 0 / 3; H_k clip after laminar-separated
  transition 0 / 0 / 0 / 0 (report 7: 0 / 1 / 10 / 16); transition relocation 0; isolated clip 0;
  other 0 / 0 / 0 / 3. The class is below 5, so FID's airfoil H_k limit was not tried.

### Wake drag: test of the explanation (decision 1 after report 7); stopped at (a)

Test (a), on the eq. 13 state, NACA 2412, Re 1e6, 0 / 5 / 10 deg; closed edge = `generate_naca4('2412',
N, 'closed')`, its XFOIL reference run with the same recipe for this test (PANE 160: Cd 0.00555,
0.00824, 0.01546; PANE 240: 0.00556, 0.00820, 0.01540; not committed). Wake-end Cd against XFOIL:

| first wake spacing | 0.0001 c | 0.0004 c (rule) | 0.002 c | 0.005 c | 0.01 c |
|---|---|---|---|---|---|
| open | +11.4 / +8.0 / +1.7 % | +8.9 / +4.6 / -1.3 % | +2.8 / -2.3 / -7.2 % | -6.0 / -10.3 / -13.9 % | -25.8 / -26.3 / -25.7 % |
| closed | -42.3 / -39.9 / -6.6 % | -40.7 / -39.5 / -4.5 % | -50.6 / -46.6 / -14.5 % | +292.6* / -57.5 / -34.3 % | +214.5* / -63.3 / -48.6 % |

| panels | 160 | 240 | 320 |
|---|---|---|---|
| open | +8.9 / +4.6 / -1.3 % | +10.6 / +7.1 / +2.0 % | +11.3 / +8.5 / +3.7 % |
| closed | -40.7 / -39.5 / -4.5 % | -29.6 / -34.1 / -4.9 % | -25.1 / -31.7 / -4.6 % |

(*) another solution (Cl 0.13 below XFOIL, TE ue 0.76). Edge velocity, trailing edge (equal on both
sides) / first wake station, 160 panels: open 0.874 / 0.899, 0.904 / 0.917, 0.933 / 0.936; closed
0.902 / 0.772, 0.918 / 0.810, 0.931 / 0.922.

Result: without a dead-air region the sensitivity is larger, not smaller, so (a) does not confirm
the explanation; per (d) no rule was implemented and nothing was tuned. What the data support: in
both cases Cd_wake/Cd_TE = (ue_wake1/ue_TE)^k with k = 3.2-3.6 wherever |jump| >= 0.02 (except open
5 deg, k 4.8-5.1), i.e. Squire-Young's (H + 5)/2 applied to a theta carried unchanged across the
edge-velocity jump at the first wake station (FID eq. 14); spacing and panel count act through that
jump, whose sign differs (open: wake-line speed above the TE value; closed, finite TE angle: below).
Open for the user.

### Reference reliability in the report (decision 4 after report 7; SPEC.md not edited)

`viscous_run.unreliable_quantities`: "differs by more than the target for a quantity" is read as the
per-point (max) target, |dCl| > 0.10, |dCd|/Cd > 30 %, |dCm| > 0.03 (our reading; for the user to
confirm with the SPEC.md wording); no 240-node result at that angle: nothing excluded; the
convergence count is unaffected. The report shows every table with and without the rule, the
number of such points per case and quantity, and the direct/restart split (decision 3).

### Decisions after report 8 (user, 2026-10-10)

1. **Wake drag.** The dead-air explanation of decision 1 after report 7 was wrong (the user's
   statement; test (a) showed the closed edge more sensitive, not less). Accepted finding: the
   wake-end drag error follows the edge-velocity jump between the trailing edge and the first wake
   station, across which FID eq. 14 carries the trailing-edge state without a momentum term. Rule
   to implement on open and closed edges, recorded as our choice: the edge velocity is continuous
   across the trailing edge; the first wake station is at the trailing edge with the trailing-edge
   edge velocity; eq. 14 applies there at equal edge velocity; the change of edge velocity down the
   wake is crossed by ordinary wake intervals (with their momentum term). Acceptance (NACA 2412, Re
   1e6, 0 / 5 / 10 deg, open and closed): wake-end Cd varies by at most 2 % of its value over 160 /
   240 / 320 panels and first wake spacings 0.0001-0.002 c; wake-end Cd, the trailing-edge
   Squire-Young value and their ratio against XFOIL's reported before and after; closed section
   |dCl| <= 0.02 and |dCm| <= 0.01 at 0-10 deg; C1 and C2 still met on the open section. Otherwise
   stop and show; the spacing is never tuned to XFOIL.
2. **C2 drag expectation** ("Cd within 15 %") is judged over the attached range, as in SPEC.md 4.1;
   beyond it, reported only.
3. **"How reached" measure:** a point whose start from the march fails is reached from the nearest
   converged angle of its sweep, under the rules of decision 3 after report 6; recorded in `path`.
   The three "other" failures at Re 6e6 are to be looked at.
4. **Reference reliability confirmed** with the per-point (maximum) reading. SPEC.md 4.1 edited
   (approved wording): "A reference point where XFOIL's own result at 240 nodes differs from 160
   nodes by more than the per-point (maximum) target for a quantity is reported but not counted
   towards that quantity's targets. The number of such points is shown per case."

Then C3; C5 only if every SPEC.md 4.1 target is met at every Reynolds number.

### Continuous edge velocity across the trailing edge (decision 1 after report 8); stopped

Implemented (not committed to `m3-viscous`; kept on branch `m3-te-tie-wip`):

- The first wake station stays FID's first wake point, eps_w c = 1e-5 c behind the trailing-edge
  midpoint (at the trailing edge to 1e-5 c; no geometry change).
- Its edge-velocity relation R_u = ue - ue_inv - D (ue delta*) is replaced by R_u = ue_w1 -
  (ue_TE,lower + ue_TE,upper)/2 (the two trailing-edge speeds are equal by the Kutta condition of the
  core, so this is either of them; the mean keeps the row symmetric). Its rows of ue_inv and D are
  zero; the wake-line velocity at that point is no longer used. The Jacobian (complex step: R_u rows
  exact; JAX: automatic) carries the tie; the Jacobian tests pass.
- Eq. 14 (theta, delta* + h_TE, C_tau) is unchanged and now holds at equal edge velocity; the change
  of ue to the second wake station is crossed by the first ordinary wake interval.
- Wake source strengths: still sigma = d(ue delta*)/ds per wake panel from the station values, so the
  first wake panel's source now uses m_w1 = ue_TE (delta*_u + delta*_l + h_TE) and carries the mass
  flow change from the trailing-edge speed to the second station's speed; the first-node strength
  rule (FID p. 3, sum of the first and last airfoil panel sources) is unchanged.
- The starting march starts the wake at the trailing-edge speed.

Results (scripts and logs in the job scratch `tmp/d3/`): see report 9. In short, open edge: wake-end
Cd within 0.4 / 0.3 / 0.8 % (0 / 5 / 10 deg) over 160-320 panels and spacings 0.0001-0.002 c (before:
spread 2.4-15 points); ratio wake-end / TE value 1.00 / 1.013 / 1.025 against XFOIL's 1.009 / 1.027 /
1.040 (dumps). C2: accuracy met with margin (Cd -4.4..+0.6 % over all angles), but the iteration
target (< 20 per final solve) is missed at 0 deg (49, cold start reached by a repair restart) and 4 deg
(40, warm start). Closed edge: not one point converges at 0 / 5 / 10 deg with any panel count or
spacing. Continuation in the weight of the tie (diagnostic only), from report 8's converged closed
solutions: the solution is lost at weight 0.55 / 0.65 / 0.85 (0 / 5 / 10 deg); along the way the
trailing-edge speed rises and the wake-line speed just behind the sharp edge falls (implied 0.58 at
0 deg, weight 0.5), i.e. with the trailing-edge mass flow carried to the first wake station, the first
wake panel becomes a sink behind the sharp edge that lowers the speed there further. Per decision 1,
stopped and shown; nothing tuned.

### Decisions after report 9 (user, 2026-10-10)

Track A (open trailing edges), then Track B (sharp trailing edge, diagnosis only):

1. **Continuity rule on open trailing edges only: interim state.** The rule of decision 1 after report
   8 is adopted for bodies with an open trailing edge. Bodies with a closed (sharp) edge keep the
   report 8 treatment (the first wake station on the wake-line velocity) for now. Reason: on the sharp
   edge our edge velocity is 0.90 at the trailing edge and 0.77 just behind it, where XFOIL has 0.835
   rising to 0.867; the kinematics at a sharp edge are wrong, and the rule cannot be applied there until
   that is fixed (Track B).
2. **Warning, no substitution.** Until Track B is resolved, a viscous result on a closed trailing edge
   carries a warning that its wake drag is unreliable. No other drag value is substituted.
3. **Decision 3 after report 8** (`reach_from_nearest`): proceed, with tests, under the same "how
   reached" rules; the three "other" failures at Re 6e6 are looked at.
4. **Tests of the side branch:** the change of `test_converges_quadratically_at_5deg` is accepted as
   described in report 9. `test_restart_reaches_the_cold_start_solution` gets another case in which a
   restart is actually exercised, with the same assertion; the change is recorded.
5. **Iterations:** "under 20" was a plan expectation, not a requirement. C3 reports the median and
   maximum final-solve iterations per Reynolds number. If a Reynolds number converges below 95 %,
   residual histories of two failing points are shown before anything changes.
6. Merge into `m3-viscous` when the full suite passes, rerun C3, write the SPEC.md 4.1 table; stop
   before C5.

### Track A, steps 1-2: the rule on open trailing edges; closed-edge drag warning

- `viscous.make_structure`: the first wake station's tie to the trailing-edge speed (and the zeroed
  rows of ue_inv and D) only where the core treats the body's trailing edge as open
  (`PanelSet.open_te`, the same switch as the inviscid trailing-edge panel: gap above
  `inviscid.CLOSED_GAP_FRACTION` of the body size). A closed body keeps R_u = ue - ue_inv - D (ue
  delta*) at that station. The starting march uses the trailing-edge speed there only where the tie
  applies.
- `ViscousResult.warnings`: a result on a body with a closed trailing edge carries
  `viscous.WARN_CLOSED_TE_DRAG` ("wake drag unreliable: closed (sharp) trailing edge; ..."); `cd` stays
  the wake-end Squire-Young value and `cd_te` the trailing-edge cross-check. The benchmark CSV has a
  `warning` column.
- Check (NACA 2412, Re 1e6; the report 9 variants: 0..10 deg at 160 panels, 0 / 5 / 10 deg for first
  wake spacings 0.0001-0.002 c and 240 / 320 panels): every closed point equals report 8 (c773602) and
  every open point equals report 9 (the rule everywhere) bit for bit in Cl, Cd, Cd TE, Cm and
  transition, with the same converged flags.
- Tests: `test_edge_velocity_tie_only_on_an_open_trailing_edge`,
  `test_closed_trailing_edge_result_warns_about_wake_drag` (slow).

### Track A, step 3: reaching failed cold starts from the nearest converged angle

- `viscous.reach_from_nearest` (our choice; decision 3 after report 8; FID section VIII.C), called by
  `viscous_run.run_case` after each sweep: a point whose start from the starting march failed is
  reached from the nearest converged angle of the same sweep, one sweep angle at a time, each angle a
  polar point (`solve_point`) started from the last one reached; the walk stops at the first angle
  that fails. Equations, tolerances and the definition of converged are unchanged. A point reached
  this way has `path` "from <angle> deg: <path>" and its `iterations_total` includes the failed
  attempts; the benchmark report counts it with the restarts ("restart" subset). The C3 diagnostic
  (`c3_failures`) takes such points from the benchmark row.
- Test: `test_reach_from_nearest_walks_from_the_nearest_converged_angle` (walking order, stop at the
  first failure, points that failed from a warm start are not walked to, path and totals).
- In the C3 run on the Track A code it reached no point: it was tried at the three cold-start failures
  at Re 6e6 (Clark Y 0 deg from 1 deg; E387 0.25 % gap 1 deg from 2 deg, which also stops the walk to
  0 deg) and failed there. "Of its sweep" is read literally: E387 gap 0 deg (up sweep) is not reached
  from -1 deg, which converged in the down sweep.
- The three "other" failures at Re 6e6 of report 8, on the Track A code: E387 0.25 % gap at 4 and 5 deg
  now converge directly from the warm start (10 and 12 iterations; the continuity rule changed the
  solution path). NACA 2412 at 1 deg still fails from the warm start (0 deg) and from a cold start:
  the largest residual is the lag equation of one lower-surface station at x/c 0.48, the first
  station behind the transition interval, held on the H_k clip (H = 1.020); the residual falls
  monotonically but slowly (2.0 after 50 iterations, about 10 % per iteration), i.e. a clipped
  station behind transition, not an oscillation. Classified "transition relocation" in this run.

### Track A, step 4: the two side-branch tests

- `test_converges_quadratically_at_5deg`: accepted as described in report 9 (the step into the
  residual tolerance is judged; bounds unchanged).
- `test_restart_reaches_the_cold_start_solution`: case changed, assertion unchanged (restart in
  `path`, `iterations_total > iterations`, Cl, Cd, Cm and transition equal to the cold start within
  1e-6). Old case NACA 4412, Re 5e5, -1 deg from 0 deg: with the continuity rule it converges without
  a restart and its cold start no longer converges. New case E387 with the 0.25 % gap, Re 1e6, -1 deg
  from 0 deg (found in the C3 run): the warm start fails, a repair restart converges (5 iterations, 205
  in total), and the cold start at -1 deg converges independently (27 iterations); they agree to
  1e-15. Candidates checked and not taken: Clark Y Re 6e6 6 and 9 deg (also agree to 1e-15, but need a
  7-10 point polar); NACA 23012 Re 5e5 -3 deg, Re 6e6 -1 deg and E387 gap Re 6e6 -1 deg (their cold
  starts do not converge). The test skips when the UIUC E387 file is not cached and cannot be
  downloaded, as the benchmark tests do.

### Track A, step 5: iterations (C3 on the Track A code)

Final-solve iterations of the converged points (open trailing edges, validated Re, attached filter,
XFOIL converged): median / max 8 / 37 (Re 5e5), 9 / 49 (1e6), 9 / 40 (3e6), 10 / 43 (6e6). Every
Reynolds number converges at 95 % or more, so no residual histories were required.

### Track A, step 6: merged into m3-viscous; C3 and the SPEC.md 4.1 table

`m3-te-tie-wip` merged into `m3-viscous` after the full suite passed (404 passed, 3 xfailed, 1 xpassed;
the xpass is the non-strict xfail of the old VII tests, as in report 8). C3 = `c3_failures
--write-benchmark` (= `viscous_run.run_case`), run on 168343b, whose `src/` and diagnostics equal the
merged commit; repeated on the merged commit as a check (below). SPEC.md 4.1 table ("fidkowski2022",
open trailing edges, attached-flow filter, validated Re; reference-reliability rule of SPEC.md 4.1
applied; Cd at the end of the wake):

| Re | converged | mean / max dCl | mean / max rel dCd | (Cd TE) | mean / max dCm | met |
|---|---|---|---|---|---|---|
| 5e5 | 83/83 = 100 % | 0.0017 / 0.0119 | 1.6 % / 5.7 % | 2.8 % / 7.3 % | 0.0004 / 0.0022 | all |
| 1e6 | 91/91 = 100 % | 0.0020 / 0.0161 | 1.7 % / 7.3 % | 2.7 % / 12.2 % | 0.0004 / 0.0032 | all |
| 3e6 | 100/102 = 98.0 % | 0.0026 / 0.0242 | 1.5 % / 6.1 % | 2.7 % / 16.0 % | 0.0005 / 0.0045 | all |
| 6e6 | 106/110 = 96.4 % | 0.0026 / 0.0240 | 1.6 % / 6.0 % | 2.7 % / 16.9 % | 0.0005 / 0.0045 | all |

The rule excludes 2 points from the Cd metrics (E387 with the 0.25 % gap, Re 5e5, -3 and -4 deg; XFOIL
Cd 0.00447 / 0.00714 at 160 PANE nodes, 0.00616 / 0.01611 at 240). Without the rule, Re 5e5 misses max
rel dCd (150.8 %, those two points; mean 4.6 %); everything else is the same. Sharp E387 (no targets):
converged 13/13, 13/13, 14/14, 15/16 at Re 5e5-6e6; every result carries the wake-drag warning.
Final-solve iterations: step 5. Report text: the reliability paragraph now cites SPEC.md 4.1 (it said
"proposed ... not yet edited"). Stopped before C5 (user).

### Track B: sharp trailing edge, diagnosis only (no package code changed); stopped for the user

Closed NACA 2412 (`generate_naca4('2412', 160, 'closed')`), Re 1e6, 0 and 5 deg. XFOIL's boundary-layer
dumps of report 9 (PANE 160) plus an inviscid DUMP and the PANE nodes (SAVE) of the same file, run here
(external program; job scratch, not committed). Our relation ue = ue_inv + D (ue delta*) (the solver's
own, `make_structure`) driven by XFOIL's delta* on airfoil and wake. Scripts: job scratch
`tmp/e1/trackb.py`, `eq9_check.py`.

1. **Edge velocity driven by XFOIL's delta*** (our 160 panels, rule spacing): within about 0.003 of
   XFOIL's from x/c 0.95 to 0.99 (one odd-even pair of -0.02 / -0.036 at x/c 0.990, where XFOIL's
   coarser delta* is interpolated onto our nodes) and from 0.007 c behind the edge on. Over the last
   0.6 % of chord ours rises to 0.951 / 1.203 at the trailing-edge node (XFOIL 0.835 / 0.899, 0 / 5
   deg); on the first 0.007 c of the wake ours is 0.06-0.29 high.
2. **Parts** (on XFOIL's own PANE nodes, so that geometry and node spacing are identical and the rule
   gives XFOIL's first wake spacing, 0.0083 c):
   - inviscid surface speed at the trailing-edge nodes (eq. 9 extrapolation): ours 0.835 / 0.829,
     XFOIL 0.757 / 0.752 (+0.078); one node upstream they agree to 0.002-0.008. XFOIL 6.99's dump does
     not satisfy LNE89 eq. 9 as printed (residual -0.135 at both angles with the Kutta condition met
     exactly); eq. 9 applied to XFOIL's own neighbouring nodes gives 0.825 / 0.819;
   - source contribution on the surface: at the trailing-edge node ours -0.031 / -0.024, XFOIL
     (viscous minus inviscid) +0.078 / +0.147; upstream ours is 0.01-0.06 below XFOIL's from x/c 0.97;
   - inviscid wake-line velocity: 0.59 at 1e-5 c, 0.851 / 0.846 at 0.0083 c, the same on our panels
     and on XFOIL's nodes. It follows the exact local corner flow behind a sharp edge with the Kutta
     condition, |u| ~ r^k, k = tau/(2 pi - tau): tau = 16.5 deg from the NACA thickness slope gives k =
     0.048; our values give 0.050 (1e-5 to 4e-4 c) and 0.048 (4e-4 to 9e-4 c). XFOIL dumps no
     inviscid wake;
   - source contribution on the wake line: +0.035 / +0.087 at 0.0083 c (total 0.887 / 0.933 against
     XFOIL 0.867 / 0.896); at the first wake station (on the edge) XFOIL carries the trailing-edge
     speed, ours the wake-line value 0.74 / 0.88.
   On our own 160 panels (trailing-edge panels 0.0004 c, 20 times shorter than XFOIL's) the source
   contribution at the trailing-edge node is +0.249 / +0.506, half from airfoil and half from wake
   sources, growing by a nearly constant step each time the distance to the edge halves: a
   logarithmic spike from the jump between the airfoil sources (delta* growing) and the near-wake
   sinks (delta* falling) at the edge, regularised only over the first wake half-panel. Both
   trailing-edge parts therefore depend on the panel size at a sharp edge (eq. 9 node value 0.70 on
   our panels, 0.835 on XFOIL's).
3. **First wake spacing 0.001 c and 0.008 c** (second wake station at that distance; first station at
   eps_w c as always): trailing-edge node 0.932 / 1.165 and 0.832 / 0.966 (rule 0.951 / 1.203; XFOIL
   0.835 / 0.899); first wake station 0.837 / 1.062 and 0.728 / 0.844. With every spacing the wake-line
   disagreement is confined to the first ~0.008 c behind the edge; from 0.011 c on ours is within 0.003
   (0.0004 and 0.001 c spacing) and 0.008 / 0.015 (0.008 c) of XFOIL's.
4. **Published sources:** LNE89 gives eq. 9 (inviscid node extrapolation), eqs. 28-29 (wake initial
   state) and two-piece linear wake sources against the log singularity at wake nodes (pp. 7-8); FID
   p. 3 gives the first-wake-node source strength (sum of the first and last airfoil panel sources, "to
   avoid a velocity spike at the trailing edge") and eq. 9; BTE89 treats the blunt edge (dead air,
   zero for a sharp edge); DG87 nothing on this. None says how the edge velocity is defined at and just
   behind a sharp, finite-angle trailing edge in the coupled problem, and the installed XFOIL departs
   from the printed eq. 9. Options for the user (report 10); none implemented.

### Decisions after report 10 (user, 2026-10-11)

1. **M3 accuracy and convergence criteria met.** Track A met every SPEC.md 4.1 target at Re 5e5, 1e6,
   3e6 and 6e6 (table in "Track A, step 6"). M3's accuracy and convergence criteria are recorded as
   met with closure set "fidkowski2022", which is **confirmed as the default set**
   (`ViscousParams.closure_set`; the docstring no longer says "provisional"). "drela1987" stays parked.
2. **Slow marks.** `test_farfield_lift_equals_pressure_lift` and
   `test_report_command_never_writes_the_baseline_document` are marked `slow`; the fast suite runs in
   42.6 s again (370 passed, 1 xfailed). The full suite runs before every merge.
3. **`reach_from_nearest` across sweeps.** A point whose start from the march failed may be reached from
   the nearest converged angle of either sweep of the same case. Implemented as a second pass in
   `viscous_run.run_case` after the per-sweep walks (which are unchanged, so every earlier result is
   unchanged): over the combined, sorted angles, a still-failed cold point is walked to from a converged
   angle of the other sweep where that angle is at least as near as the nearest converged angle of its
   own sweep (that one has already been tried) (`reach_from_nearest(..., group=...)`; our reading).
   Test: `test_reach_from_nearest_uses_the_other_sweep_when_it_is_nearer`.
4. **Sharp trailing edge.** Option (b) of report 10 (the sharp edge as the small-gap limit of the open
   treatment) is the route for v2.0, with (c) (the interim state: warning, no substitution) as the
   fallback. Option (a), a resolution-independent sharp-edge treatment, is a Later item after v2.0,
   starting from the Track B findings.

### Step 1 after report 10: sharp edge as the small-gap limit; rule not met, interim state kept

User's rule: between gaps 0.001 c and 0.0005 c, if Cd changes by less than 1 % and Cl by less than 0.005
at every point (closed NACA 2412 and sharp E387, Re 1e6 and 3e6, alpha 0 / 5 / 10 deg), a closed trailing
edge gets `DESIGN.min_te_gap` by default in viscous analysis; otherwise the interim state stays.

Study on e9d54fc (no package code changed): `generate_naca4('2412', 160, 'closed')` and the canonical
E387, up sweep 0..10 deg as in the benchmark (march, warm starts, `reach_from_nearest`), at the sharp
original (closed treatment) and with the gap tool at 0.002 / 0.001 / 0.0005 / 0.00025 c (open treatment).
XFOIL 6.99 as the external program on the same coordinate files, PANE 160 and 240. Scripts and tables
are generated output, not committed.

| case (0 / 5 / 10 deg) | max rel dCd, 0.001 -> 0.0005 c | max dCl | met |
|---|---|---|---|
| NACA 2412, Re 1e6 | 0.26 % (5) | 0.0036 (10) | yes |
| NACA 2412, Re 3e6 | 0.23 % (5) | **0.0056 (5)** | no |
| E387, Re 1e6 | 0.42 % (5) | 0.0020 (5) | yes |
| E387, Re 3e6 | 0.37 % (10) | 0.0041 (10) | no: 5 deg does not converge at 0.001 or 0.0005 c (50 iterations; unconverged values differ by 0.81 % / 0.0028) |

- Over all eleven angles the maxima are 0.34 / 0.57 / 0.47 / 0.58 % in Cd and 0.0036 / 0.0056 / 0.0029
  / 0.0041 in Cl. The changes roughly halve with each halving of the gap (NACA 2412 Re 3e6, 5 deg, dCl:
  0.0100, 0.0056, 0.0026 for the steps 0.002 -> 0.001 -> 0.0005 -> 0.00025 c).
- **Decision (by the rule): not met; no policy implemented.** Closed edges keep report 8's treatment
  and `WARN_CLOSED_TE_DRAG`; nothing is substituted.
- Ours against XFOIL (PANE 160) on the same 0.001 c geometry, 0 / 5 / 10 deg, the 10 points where both
  converged (E387 Re 3e6: XFOIL fails at 0 deg, ours at 5 deg): |dCl| <= 0.009, |dCm| <= 0.0015, rel dCd
  -8.9 to +1.4 % (9 of 10 between -2.2 and +1.4 %; E387 Re 1e6 10 deg -8.9 %, where XFOIL itself moves
  6.8 % between PANE 160 and 240).
- XFOIL's sharp result differs from its own 0.001 c result by up to 0.042 in Cl (NACA 2412 Re 1e6
  5 deg: 0.8564 sharp, 0.8141 gapped). Inviscid, the two geometries differ by <= 0.0008 in Cl (XFOIL and
  ours agree to 0.0013), so the jump comes from XFOIL's viscous sharp-edge treatment.
- The missing XFOIL polar (sharp NACA 2412, Re 3e6, PANE 240) was generated again in the foreground:
  25 of 25 converged, data identical to the earlier run.

---

## Later

- End of M3 (user's decision, 2026-10-10): extend the validated range downward. The new solver
  converges on essentially every point at Re 1e5 and 2e5 with mean |dCl| 0.010 and 0.005 and drag
  within 5 to 6 %. Propose targets for Re 1e5 and 2e5 and report Re 5e4 as it is. This moves
  low-Reynolds validation from v2.2 into v2.0 and needs a SPEC.md change (for the user's approval).
- After M3 (candidate for M7; user, 2026-10-10): solver speed. Reuse the coupling between angles, a
  faster starting march, a compiled JAX path. Golden values recorded first; results must not change.
  Time per point reported before and after; the target is set from measurements.
- M3 cleanup (C5; user, 2026-10-10): confirm that `design.maximize_ld` and the batch example use the
  new viscous solver, and rerun the design example to show that its lift-to-drag values are now
  realistic.
- v2.1 (user, 2026-10-10): multi-point optimisation and CST or Bezier shapes, as in SPEC.md. Also
  evaluate whether a target-velocity (inverse) design mode is worth adding on top of the optimiser,
  or whether Cp matching covers it.
- M3 (after report 8): convergence at Re 1e6 and 6e6 is 94.5 % (target 95 %); 8 of the 11
  non-converged attached points are cold starts at 0 or 1 deg, the first angles of an up sweep
  started from the march (the H_k clip after a laminar-separated transition, report 7's main
  class, is gone with the eq. 13 upwinding). A "how reached" measure (e.g. reaching such points
  from the nearest converged angle of the sweep) is for the user to decide.
- M3 (after report 8): wake drag. The wake-end Cd follows the edge-velocity jump between the
  trailing edge and the first wake station on open and closed trailing edges (decision 1 after
  report 7, test (a)); open for the user.
- M3/M9 (after report 8): with a closed (sharp) trailing edge, NACA 2412 at Re 1e6 shows dCl -0.053
  and dCm +0.011 at 5 deg and +0.048 / -0.010 at 10 deg (open edge within 0.01 / 0.002), with the
  same edge-velocity jump; sharp sections are reported without targets (SPEC 4.1).
- M5: the velocity of one body's wake sources at another body's wake (not included in M3).
- M3 (end of M3, after the convergence work; user's decision after report 6): E387 sharp at Re 1e6,
  -4 and -3 deg converges to a non-physical solution (Cl 0.19 against XFOIL -0.063).
- After v2.0: decide the wake treatment of the parked closure set "drela1987" (its far wake drives
  H_k below the 1.02 clip; not converged with the wake; excluded from benchmark runs until then).
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
- M2: add the Williams two-element exact solution (ARC R&M 3717) once the report is supplied.
- M2: find the cause of the NumPy/JAX difference (up to 1.49e-6 in Cl for NACA 4412 at -4 deg).
  Two double-precision solves of the same system should agree far more closely; the two cores
  currently assemble differently arranged systems.
- M2: Kármán-Trefftz pressure error is first order and largest just aft of the leading edge
  (0.0220 at 200 panels, 4 deg); the test is marked xfail for M2.
- M3: E387 (sharp trailing edge): XFOIL converges at only part of the sweep and seven XFOIL
  processes over the matrix hang until the timeout. Consider whether E387 stays in the matrix.
- M3: Newton VII takes 25-60 s per point and converged at 1 of 9 points at Re 5e4 and 5 of 10 at
  Re 2e5 (subset).
- M7: the benchmark adapter `baseline_v131.py` is removed when the v1.3.1 solvers are replaced.
- M2: once the trailing-edge gap tool exists, add an E387 variant with a 0.25 % gap as a section
  with targets, and regenerate the Clark Y and E387 references after the new repanelling.
- M3: on NACA 2412 at Re 1e6 the transpiration VII lift is too low in the linear range (0.73 at
  5 deg against XFOIL 0.81 and inviscid 0.86, so about twice the correct viscous loss), it shows
  no stall above 9 deg, and Cm drifts the wrong way with angle.
- Option if the trailing-edge gap tests show sensitivity: a trailing-edge panel treatment from
  a published source (decided in M2, Q2).
- v2.1 or later: a correct NACA 6-series generator (Ladson's method); not in v2.0.
- M9 (THIRD_PARTY.md): Williams, B. R., "An Exact Test Case for the Plane Potential Flow About
  Two Adjacent Lifting Aerofoils", ARC R&M 3717, HMSO, 1973; transcribed values used as test
  data.
- M9: check every citation listed under "Citations made without the source at hand" (M2)
  against the books and papers; each is marked "section unverified" until then: Kuethe & Chow
  (1998) section 5.10; Katz & Plotkin (2001) sections 6.7, 10.2, 10.3-10.4, 11.4.2 and chapter
  11; Lighthill (1958); Milne-Thomson (1958) chapter 7; NACA Report 460 (1933) and its -0.1015
  coefficient; Abbott and von Doenhoff. Also the pre-existing v1.3.1 citations listed there.
- M3: Newton VII at Re 5e4 (NACA 2412, 2 deg) reaches a different solution at 160 panels (Cl
  0.60 closed, 0.68 open, against about 0.34-0.37 at 100-120 panels); convergence depends on
  panel count on both trailing edges (table in M2, "Decisions after the P3 report").
- After v2.0: a closed (sharp) trailing-edge treatment that matches the open (trailing-edge
  panel) limit more closely at zero gap. The jump is first order on the PARSEC start section
  (1.1e-3 at 100 panels, 1.4e-4 at 800) and falls faster on NACA 2412 (5e-5 at 100, 1.6e-6 at
  640).
- M9: verify against the source: NACA Report 460 (Jacobs, Ward and Pinkerton, 1933) and Abbott
  and von Doenhoff, for the NACA 4- and 5-digit construction with thickness perpendicular to the
  mean line (in addition to the coefficient -0.1015 listed above).
- M7: the farthest-point leading edge gives a generated NACA 2412 a chord of 1.00008, so its
  coefficients differ from the nominal unit chord by about 8e-5. Decide then whether generated
  sections keep their nominal chord.

- M9: the README states that the viscous model follows the published formulation of Drela (1989)
  and Drela & Giles (1987).
- M9: verify the reading of DG87 eq. (18) (left side H*, natural logarithm) against the source.
- M9: the README names both viscous sources, Drela (1989) / Drela & Giles (1987) and Fidkowski
  ("A Coupled Inviscid-Viscous Airfoil Analysis Solver, Revisited"), and says that the closure set
  "fidkowski2022" follows the later published forms (close to later XFOIL versions).
- After v2.0: compare both closure sets ("fidkowski2022", "drela1987") with wind-tunnel data,
  downloaded on demand (not committed).
- After v2.0 (user, after report 10): option (a) for the sharp trailing edge, a resolution-independent
  treatment of the edge velocity at and just behind a sharp edge in the coupled problem (our choice, not
  published). Starting point, the Track B findings (M3, "Track B"): with XFOIL's delta* our edge velocity
  agrees to about 0.003 except over the last ~0.6 % of chord and the first ~0.008 c of wake; two parts
  depend on the panel size there, the LNE89 eq. 9 trailing-edge node value (0.835 on XFOIL's nodes, 0.70
  on ours) and a logarithmic source spike at the airfoil/wake junction (+0.04 per halving of the
  distance to the edge; +0.249 / +0.506 at 0 / 5 deg on our 160 panels); the inviscid wake-line velocity
  follows the exact corner flow (|u| ~ r^k, k = tau/(2 pi - tau) = 0.048). Candidate forms: the TE speed
  extrapolated from the attached flow upstream (as eq. 9 does for gamma, applied to the total ue), and
  a source strength ramped across the edge over a length set by the layer (a few delta*), each chosen
  by convergence under panel refinement (160 / 320 / 640), never tuned to XFOIL.
- After v2.0: a converged but wrong result on a sharp edge. Closed NACA 2412, Re 3e6, 0 deg converges
  (march, 13 iterations) at Cl 0.059, Cd 0.0272 against XFOIL 0.234, 0.00537 (step 1 after report 10);
  the 0.001 c gapped section gives 0.243, 0.00530 there. The closed-edge warning is the only guard;
  option (a) must remove it.
