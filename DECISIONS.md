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
- Q7 to Q11: pending (asked again on 2026-10-09).

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
