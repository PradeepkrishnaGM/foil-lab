# Validation

This document records how Foil Lab has been checked against references, what is
validated, and — just as importantly — what is **not**. Everything here is
reproducible from `examples/validation_report.py` and the sourced data under
`tests/validation/data/`. No numbers are hand-entered into prose from memory.

## Reference data

| File | Airfoil | Source | Conditions |
|------|---------|--------|-----------|
| `naca0012_thin_airfoil.csv` | NACA 0012 | Thin-airfoil theory (analytical) | inviscid limit |
| `naca2412_xfoil_re50k.csv` | NACA 2412 | XFOIL 6.96 via airfoiltools.com | Re=50k, Ncrit=9, M=0 |

XFOIL is itself a viscous panel/BL code, not wind-tunnel truth — but it is the
de-facto reference the target users already trust, so matching it is the most
direct statement of "drop-in alternative". Wind-tunnel validation (e.g. Abbott &
von Doenhoff) remains future work; that experimental data is not freely
available in clean machine-readable form.

## What is validated

**Inviscid core — strong.** Against analytical thin-airfoil theory (NACA 0012),
the inviscid lift matches to MAE ≈ 0.04 (the small residual is finite-thickness
physics the flat-plate theory omits, so the panel method is arguably the more
accurate of the two). Zero-lift angle, lift-curve slope (~2π/rad), and
aerodynamic center (~0.25c) all match theory — see `tests/` and
`tests/validation/`.

**Lift-curve character vs XFOIL.** Foil Lab's inviscid NACA 2412 zero-lift angle
(≈ −2.15°) matches thin-airfoil theory. XFOIL at Re=50k reports ≈ −0.3°, shifted
by the strong low-Re separation bubble — the divergence there is the reference's
viscous physics, not a Foil Lab error.

**VII at high Re.** The transpiration viscous-inviscid coupling was validated at
Re ≈ 3×10⁶ against analytical trends (drag rises with incidence, viscous
decambering reduces lift, drag magnitude ≈ 0.006–0.008 consistent with
experiment for NACA 2412). See `tests/test_vii.py`.

## What is NOT validated (known limits)

* **Low Reynolds number (≲ 200k): VII does not converge.** At Re=50k the
  transpiration VII fails to converge at every tested angle (0/11). This regime
  is dominated by laminar separation bubbles, which the *direct* transpiration
  formulation cannot resolve — it needs the inverse/simultaneous-Newton
  formulation (in progress; see below). `tests/validation` pins this limit so a
  future fix shows up as a test change.

  *Progress (v1.3):* the new simultaneous-Newton solver (`vii_newton`) plus a
  laminar-separation-bubble closure achieves the **first low-Re convergence** in
  the project: at Re=50k, NACA 2412 converges at α=0–2° with Cd ~0.017 (XFOIL
  ~0.024–0.031 — same order, ~30–45% low, versus the old one-way ~0.001). Higher
  angles do not yet converge (longer bubbles). This is experimental and reported
  via the `converged` flag; see CHANGELOG.md. The headline envelope statement is
  unchanged for now — low-Re is partially working, not yet a full XFOIL
  replacement — but the wall has genuinely started to move.
* **Point-by-point drag-polar match vs experiment:** not yet established across
  the envelope. Foil Lab matches XFOIL lift trends where the flow is attached;
  a full experimental drag-polar match is future work.
* **Deep stall / post-stall:** the empirical post-stall model is a plausibility
  clamp, not a validated prediction.

## Honest bottom line

Foil Lab is **trustworthy for attached, higher-Reynolds 2D analysis** and for
inviscid/differentiable workflows (the JAX core, batch sweeps, inverse design).
It is **not yet a low-Re XFOIL replacement** — that requires the inverse-VII
work and experimental validation. The project states this openly rather than
implying full parity.

## Reproducing

```bash
python examples/validation_report.py        # full report
pytest tests/validation/ -q                 # automated envelope checks
```
