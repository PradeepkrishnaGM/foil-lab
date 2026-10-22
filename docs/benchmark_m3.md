# Benchmark after M3: coupled viscous solver against XFOIL

Tables only. Measured 2026-10-15 with `python -m foil_lab.benchmark viscous` on the M3 C5 code
(branch `m3-viscous`; the commit that adds this file), closure set "fidkowski2022" (the default).
The before pictures are `docs/baseline_v1.3.1.md` (M1) and `docs/benchmark_m2.md` (M2).

**Rules** (SPEC.md 4.1):
- Reference: XFOIL 6.99, 160 `PANE` nodes, N_crit 9, free transition, Mach 0. Both codes read the
  same canonical 160-panel coordinate file.
- Angles from -4 to 20 deg in 1 deg steps. Each case is two sweeps from 0 deg (up and down) with warm
  starts. A point whose start from the march failed is then reached from the nearest converged angle
  of either sweep. Every point says how it was reached (`path`).
- Errors are taken inside the attached-flow filter, at points where both codes converged. A point
  counts as converged only if the residual criteria of `params.ViscousParams` were met.
- Cd is Squire-Young at the end of the wake; "Cd TE" is the Squire-Young value at the trailing edge
  (a cross-check).
- Reference-reliability rule: a reference point where XFOIL's own result at 240 nodes differs from
  160 nodes by more than the per-point target is not counted for that quantity.
- ✓ / ✗: SPEC.md 4.1 target met / not met, at each validated Reynolds number separately. The validated
  range is Re 1e5 to 6e6 (SPEC.md 4.1, extended from 5e5 by the user after report 11, proposal A);
  Re 5e4 is reported without a claim.
- Run times are per point on one core of an i5-10500T, with 5 worker processes running at once.

## Open trailing edges (NACA 0012, 2412, 4412, 23012, Clark Y, E387 with a 0.25 % gap)

With the reference-reliability rule (the SPEC.md 4.1 measure):

| Re | converged / XFOIL converged | mean / max dCl | mean / max rel dCd | mean / max rel dCd (Cd TE) | mean / max dCm | iterations median / max | run time per point (s) |
|---|---|---|---|---|---|---|---|
| 1e5 | 81/81 (100.0 %) ✓ | 0.0062 ✓ / 0.0275 ✓ | 3.0 % ✓ / 26.3 % ✓ | 6.8 % / 26.7 % | 0.0011 ✓ / 0.0040 ✓ | 9 / 40 | 10.3 |
| 2e5 | 83/83 (100.0 %) ✓ | 0.0029 ✓ / 0.0146 ✓ | 2.0 % ✓ / 19.9 % ✓ | 4.1 % / 20.1 % | 0.0006 ✓ / 0.0026 ✓ | 9 / 22 | 7.3 |
| 5e5 | 83/83 (100.0 %) ✓ | 0.0017 ✓ / 0.0119 ✓ | 1.6 % ✓ / 5.7 % ✓ | 2.8 % / 7.3 % | 0.0004 ✓ / 0.0022 ✓ | 8 / 37 | 11.6 |
| 1e6 | 91/91 (100.0 %) ✓ | 0.0020 ✓ / 0.0161 ✓ | 1.7 % ✓ / 7.3 % ✓ | 2.7 % / 12.2 % | 0.0004 ✓ / 0.0032 ✓ | 9 / 49 | 9.4 |
| 3e6 | 100/102 (98.0 %) ✓ | 0.0026 ✓ / 0.0242 ✓ | 1.5 % ✓ / 6.1 % ✓ | 2.7 % / 16.0 % | 0.0005 ✓ / 0.0045 ✓ | 9 / 40 | 14.5 |
| 6e6 | 106/110 (96.4 %) ✓ | 0.0026 ✓ / 0.0240 ✓ | 1.6 % ✓ / 6.0 % ✓ | 2.7 % / 16.9 % | 0.0005 ✓ / 0.0045 ✓ | 10 / 43 | 15.3 |
| 5e4 (not validated) | 72/78 (92.3 %) | 0.0205 / 0.1378 | 5.8 % / 35.9 % | 12.5 % / 37.6 % | 0.0031 / 0.0557 | 10 / 35 | 20.5 |

**Every SPEC.md 4.1 target is met at Re 1e5, 2e5, 5e5, 1e6, 3e6 and 6e6.** At Re 1e5 the closest
margin is max rel dCd, 26.3 % against 30 %: one point (E387 with the 0.25 % gap, -4 deg), where
XFOIL itself moves 17.7 % between 160 and 240 nodes. The next largest is 10.0 %.

Without the rule, only one figure changes: at Re 5e5, mean / max rel dCd is 4.6 % / 150.8 % ✗
(Cd TE 5.7 % / 148.5 %). The two points the rule excludes are E387 with the 0.25 % gap at -3 and
-4 deg. There, XFOIL's own Cd changes from 0.00447 / 0.00714 at 160 nodes to 0.00616 / 0.01611 at
240 nodes. At every other Reynolds number the rule excludes nothing.

## Sharp trailing edge (E387), no targets

Every closed-edge result carries the warning "wake drag unreliable: closed (sharp) trailing edge"
(DECISIONS.md M3, Track A step 2; step 1 after report 10).

| Re | converged / XFOIL converged | mean / max dCl | mean / max rel dCd | mean / max rel dCd (Cd TE) | mean / max dCm | iterations median / max | run time per point (s) |
|---|---|---|---|---|---|---|---|
| 5e5 | 13/13 (100.0 %) | 0.0055 / 0.0370 | 6.4 % / 20.7 % | 2.8 % / 17.3 % | 0.0012 / 0.0076 | 10 / 26 | 11.8 |
| 1e6 | 13/13 (100.0 %) | 0.0054 / 0.0271 | 5.8 % / 19.9 % | 2.3 % / 7.0 % | 0.0011 / 0.0057 | 9 / 32 | 11.9 |
| 3e6 | 14/14 (100.0 %) | 0.0071 / 0.0434 | 8.4 % / 26.7 % | 2.1 % / 5.9 % | 0.0015 / 0.0091 | 13 / 32 | 9.2 |
| 6e6 | 15/16 (93.8 %) | 0.0155 / 0.1097 | 11.1 % / 34.6 % | 3.3 % / 14.5 % | 0.0033 / 0.0226 | 9 / 34 | 16.1 |
| 2e5 | 13/13 (100.0 %) | 0.0042 / 0.0145 | 6.5 % / 35.2 % | 4.1 % / 33.3 % | 0.0012 / 0.0049 | 11 / 21 | 5.4 |
| 1e5 | 12/12 (100.0 %) | 0.0091 / 0.0430 | 9.6 % / 47.9 % | 4.7 % / 28.0 % | 0.0017 / 0.0056 | 10.5 / 18 | 16.3 |
| 5e4 (not validated) | 8/13 (61.5 %) | 0.0703 / 0.1361 | 13.6 % / 37.9 % | 18.5 % / 34.8 % | 0.0164 / 0.0448 | 14.5 / 26 | 54.8 |

## Re 2e5 and 1e5, per section (rule applied)

| section | Re 2e5: converged | mean / max dCl | mean / max rel dCd | mean / max dCm | Re 1e5: converged | mean / max dCl | mean / max rel dCd | mean / max dCm |
|---|---|---|---|---|---|---|---|---|
| NACA 0012 | 14/14 | 0.0049 / 0.0132 | 2.3 % / 3.3 % | 0.0010 / 0.0024 | 13/13 | 0.0090 / 0.0228 | 2.8 % / 4.0 % | 0.0014 / 0.0034 |
| NACA 2412 | 15/15 | 0.0043 / 0.0146 | 2.2 % / 3.0 % | 0.0008 / 0.0026 | 14/14 | 0.0065 / 0.0258 | 2.5 % / 4.2 % | 0.0011 / 0.0040 |
| NACA 4412 | 13/13 | 0.0017 / 0.0059 | 1.5 % / 3.0 % | 0.0003 / 0.0008 | 13/13 | 0.0035 / 0.0175 | 2.6 % / 6.1 % | 0.0007 / 0.0024 |
| NACA 23012 | 15/15 | 0.0032 / 0.0111 | 2.0 % / 4.6 % | 0.0006 / 0.0020 | 14/14 | 0.0066 / 0.0193 | 2.3 % / 3.9 % | 0.0011 / 0.0030 |
| Clark Y | 13/13 | 0.0014 / 0.0037 | 1.6 % / 2.7 % | 0.0003 / 0.0006 | 13/13 | 0.0087 / 0.0275 | 3.2 % / 10.0 % | 0.0012 / 0.0037 |
| E387, 0.25 % gap | 13/13 | 0.0013 / 0.0019 | 2.3 % / 19.9 % | 0.0003 / 0.0007 | 14/14 | 0.0028 / 0.0120 | 4.6 % / 26.3 % | 0.0008 / 0.0028 |
| E387 (sharp) | 13/13 | 0.0042 / 0.0145 | 6.5 % / 35.2 % | 0.0012 / 0.0049 | 12/12 | 0.0091 / 0.0430 | 9.6 % / 47.9 % | 0.0017 / 0.0056 |

## Points not converged (attached-flow filter, XFOIL converged)

- **Validated range, open edges (6 points):**
  - NACA 23012, Re 3e6, -1 deg (warm start)
  - Clark Y, Re 3e6, 9 deg (warm start)
  - NACA 2412, Re 6e6, 1 deg (warm start)
  - Clark Y, Re 6e6, 0 deg (start from the march)
  - E387 with the 0.25 % gap, Re 6e6, 0 and 1 deg (start from the march)

  Each stops at the iteration limit. A start from the march that fails is then tried from the
  nearest converged angle of either sweep; that did not reach these points either.
- **Validated range, sharp E387:** Re 6e6, 2 deg.
- **Re 5e4 (reported, no claim):**
  - NACA 4412: 5 to 9 deg
  - NACA 23012: -3 deg
  - sharp E387: 3, 5, 6, 7 and 8 deg

## Change from the C3 run of report 10

Run again on the C5 code. All 1167 points that converged in both runs are bit-identical (Cl, Cd,
Cm), and no point was gained or lost. That covers both changes since report 10:
- the C5 rerouting;
- the second pass of `reach_from_nearest` across sweeps (`e9d54fc`), which reached no further
  point.

Run once more after report 11: `run_case` now calls `viscous.sweep`, and the label of points
reached by a walk ("from <angle> deg: ...") is fixed. All 1225 rows are identical (converged, Cl,
Cd, Cd TE, Cm, transition); only 5 path labels changed.

Closed trailing edges: the benchmark solves the canonical files unchanged (SPEC.md 4.1), so sharp
E387 stays sharp above. Through `fl.analyze`, `fl.polar` and `solver.solve_flow`, a closed edge is
opened to 0.0005 c by default; that rerun is in DECISIONS.md (M3, decisions after report 11).
