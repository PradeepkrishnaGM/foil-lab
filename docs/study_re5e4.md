# Study: why Re 5e4 misses the targets (v2.2, 2026-11-18)

Scope and answers: DECISIONS.md, "v2.2 study: Re 5e4". Diagnostic only; no solver code changed.
Agreement is with XFOIL 6.99, not with experiment. Reproduce with
`tests/diagnostics/re5e4_study.py` (`run`, `breakdown`, `bl`, `trace`); output in
`benchmark_output/study_re5e4` (not in git).

## Result in one paragraph

At Re 5e4 the targets are missed for two separate reasons. **Convergence** (92.3 % against 95 %):
5 of the 6 failures are one solver defect, the transition interval switching back and forth between
two neighbouring nodes on every Newton iteration (NACA 4412, 5-9 deg). That is fixable in the solver.
**Per-point maxima** (dCl, dCd, dCm): all three misses at 5e4 come from one point, and at 7e4 both
misses come from two points; each sits where the laminar separation bubble changes regime
(a bubble that jumps forward, or bursts). There both codes, or XFOIL alone, find a solution of a
different bubble type. These are model differences, and with no experiment at hand we cannot
tell which code is right. **Recommendation:** fix the transition-interval cycle (plan below, for
approval); do not claim Re 5e4 or 7e4, because the per-point maxima would still be missed.

## Pooled SPEC.md 4.1 verdict (open trailing edges, attached-flow filter)

| Re | converged | mean / max dCl | mean / max rel dCd | mean / max dCm |
|---|---|---|---|---|
| 5e4 (release run) | 72/78, 92.3 % **miss** | 0.021 / 0.138 **miss** | 5.9 % / 35.9 % **miss** | 0.003 / 0.056 **miss** |
| 7e4 (study run) | 76/76, 100 % | 0.014 / 0.141 **miss** | 4.4 % / 35.5 % **miss** | 0.002 / 0.025 |

Targets: converged >= 95 %; dCl 0.03 / 0.10; rel dCd 10 % / 30 %; dCm 0.01 / 0.03. Every mean target
is met at both Reynolds numbers. The Re 7e4 run took 37 min on 5 workers (the sharp E387, which has
no targets, 28 min of it).

## Every failing point

| Re | section | alpha | failure | mechanism | kind |
|---|---|---|---|---|---|
| 5e4 | NACA 4412 | 5, 6, 7, 8, 9 | not converged | transition interval cycles between two nodes | solver |
| 5e4 | NACA 23012 | -3 | not converged | bubble regime boundary (forward jump) | model |
| 5e4 | E387 0.25 % gap | -4 | dCl 0.138, dCd 35.9 %, dCm 0.056 | bubble burst (XFOIL) vs short bubble (ours) | model |
| 7e4 | NACA 23012 | -2 | dCl 0.141 | likely the same regime boundary (not traced) | model? |
| 7e4 | E387 0.25 % gap | -4 | dCd 35.5 % | as at 5e4, smaller | model |

None of these points is flagged by the reference-reliability rule (XFOIL 160 vs 240 nodes).

### NACA 4412, Re 5e4, 5-9 deg: transition interval cycle (solver)

Both codes have the same flow at 2-6 deg: laminar separation on the upper surface at x/c 0.31-0.40,
transition at 0.65-0.80, laminar H up to 17 (ours) and 22 (XFOIL). At 2-4 deg we converge and agree
(Cl 0.647 against 0.614 at 4 deg; transition 0.713 against 0.738). From 4.25 deg on Newton fails:

- Started from the converged 4 deg state, the transition interval moves between x/c 0.694 and 0.712
  on every iteration, a clean two-cycle; the largest residual alternates between those two nodes
  (0.06 and 0.29) for all 50 iterations (`trace --section NACA4412 --alphas 4 4.25`).
- More iterations do not help (200: the residual grows to 1e6), nor do 0.25 deg steps (4.25 fails),
  nor 240 panels (5 deg fails).

In `viscous.solve_viscous` the onset position w inside the transition interval is part of the
coupled solution, but which interval holds the transition (`Branch.p_tr`) is chosen again after
every Newton step (`update_transition`), outside the Newton system. With a long laminar bubble the
choice at one node implies the other, so the iteration cannot settle.

### NACA 23012, Re 5e4 -3 deg and Re 7e4 -2 deg: bubble regime boundary (model)

At -2 deg (5e4) the lower surface has a long laminar separation from x/c 0.56 to the trailing edge;
at -3 deg XFOIL has a short leading-edge bubble instead (separation 0.03-0.26, transition 0.35) and
its Cl jumps from 0.007 to -0.310. At -4 deg we find the same leading-edge bubble and agree (Cl
-0.402 against -0.392, transition 0.167 against 0.180). Approached from -4 deg, we still converge at
-3.5 deg (Cl -0.360) but not at -3 deg; approached from -2 deg, -3 deg fails too. Our model's regime
boundary lies a little before XFOIL's, and at -3 deg it has no steady solution on either branch.
At Re 7e4 -2 deg converges but misses dCl (0.141; lower-surface transition 0.961 against XFOIL's
0.992); that it is the same boundary is likely, but this point was not traced.

### E387 with 0.25 % gap, -4 deg: bubble burst (model)

Both codes converge. XFOIL: long lower-surface laminar separation from x/c 0.003 to 0.71, H up to
63, transition 0.385, Cl -0.397. Ours: short leading-edge bubble (separation to 0.17), transition
0.063, Cl -0.259. At -2 and -3 deg the two codes agree on a short bubble (transition 0.130 / 0.127
and 0.076 / 0.115). So XFOIL's bubble bursts between -3 and -4 deg and ours does not. Which is
physical cannot be decided without experiment; XFOIL's 160- and 240-node results agree within the
per-point targets.

## Related open item: flap points at Re 6e6 (different cause)

The flap points that do not converge at Re 6e6 (DECISIONS.md, Later) are **not** this mechanism.
NACA 0012, 15 deg flap, 0 deg (`trace --re 6e6 --section NACA0012_F15 --alphas 0`): from the
starting march the Newton steps are cut to 0-4 % by the step limits from the second iteration on,
and the largest residual stays near the leading edge and on the flap's lower surface. A poor
initial state is the likelier cause; it needs its own study.

## Recommendation

1. **Fix the transition-interval cycle** (solver; for approval as its own step). Two published-method
   routes:
   - (a) let the transition position move continuously across node boundaries inside the Newton
     system, as in the transition-interval formulation of M. Drela, "XFOIL: An Analysis and Design
     System for Low Reynolds Number Airfoils" (1989), where the transition location is solved
     together with the boundary layer;
   - (b) detect the two-cycle, freeze the interval, converge, and report `converged` only if the
     converged amplification reaches N_crit inside the frozen interval. This keeps the residual
     criteria; it adds a consistency condition and would need your approval as a definition.

   Expected gain (not measured): the five NACA 4412 points at Re 5e4, so 77/78 (98.7 %)
   convergence there. The per-point maxima would still be missed at 5e4 and 7e4. Every committed
   golden value and benchmark result must be rerun and stay unchanged where no cycle occurred.
2. **No claim at Re 5e4 or 7e4.** The remaining misses are bubble-regime differences against XFOIL
   with no method known to be right, which is the SPEC.md v2.2 condition for not claiming. No change
   to SPEC.md is proposed.
3. **Flap points at Re 6e6:** a separate study of the starting state.
