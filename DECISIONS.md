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
