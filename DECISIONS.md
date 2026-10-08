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

### Q2 diagnosis

(To be filled in after step 2.)

### Removed tests

(None yet.)

---

## Later

- `requires-python = ">=3.10"` is declared but only Python 3.14 is available and tested here.
  Decide the supported versions and test them in M7.
- `examples/basic_alpha_sweep.py` opens `naca2412.dat` relative to the current directory, so it
  only works when run from `examples/`. Fix with the M7 example rework.
- README and TESTING.md still describe `PYTHONPATH=src` and older test counts; rewrite in M9.
