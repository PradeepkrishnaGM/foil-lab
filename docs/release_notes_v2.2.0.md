# Foil-Lab 2.2.0

Foil-Lab 2.2 is the first published release of the rework of v1.3.1. Its aim is "correct and
proven": every result path is checked against an exact solution or against XFOIL 6.99, and the
comparison is reproducible with one command. Flaps and multi-element sections are the headline
capability. It contains the prepared 2.0.0 (never released on its own) and the 2.1 and 2.2 work
on its known issues; CHANGELOG.md lists both.

## Highlights

- One coupled viscous-inviscid solver (integral boundary layer and wake in one Newton system, e^n
  transition, a truthful `converged` flag). Against XFOIL on identical coordinate files it meets
  the published targets at every Reynolds number from 1e5 to 6e6 (VALIDATION.md): 1206 of 1225
  benchmark points converge.
- Plain flaps from -30 to +30 deg with hinge moment and flap polar sets; validated up to 5 deg,
  reported without a claim up to 20 deg (1428 of 1500 flap benchmark points converge).
- Several elements, inviscid and viscous, with forces per element and per-element trips.
- Corrections applied once, in a fixed recorded order: ground effect (image method), simple sweep,
  Kármán-Tsien with the critical Mach number, finite aspect ratio (Helmbold lift; profile drag and
  moments of the section at the effective angle).
- A single Python API (`fl.analyze`, `fl.polar`), pandas tables, plot helpers, exports, airfoils by
  name from the UIUC database, a polar database, a batch runner, a command line and a repaired GUI.

## New since the 2.0.0 draft

- Fewer failed viscous points: transition-interval cycles are pinned (NACA 4412 at Re 5e4: 9 -> 24
  of 25 angles), the starting march is guarded, and a last sweep pass walks to failed points from
  both sides. Points that converged before are unchanged.
- Benchmark run time 3.84 h -> 1.2 h with unchanged golden values.
- Finite-wing profile drag and moments at the effective angle.

## Breaking changes from 1.3.1

- The v1.3.1 viscous paths (one-way drag, transpiration VII and its variants, roughness, post-stall
  model) are removed; with a Reynolds number every interface runs the coupled solver.
- Ground effect pitches the section about a pivot in a stream parallel to the ground.
- The aspect-ratio correction no longer applies compressibility a second time; the sweep correction
  uses the normal-plane angle (lift slope proportional to cos(sweep)).
- The approximate NACA 6-series generator is removed. `plotting.py` is replaced by `foil_lab.plot`.
- The v1.3.1 interface (`solver.solve_flow`, `SolutionData`, `analysis.run_alpha_sweep`,
  `advanced`, `ground_effect.solve_ground_effect`, `flowfield.compute_velocity_field`) is removed;
  `fl.analyze` and `fl.polar` replace it. The constant-wing-loading polar is not carried over.

## Limits

Subsonic, attached or mildly separated flow; viscous results at Mach 0; validated Re 1e5 to 6e6;
maximum lift and post-stall approximate; multi-element results without wake merging or slot
separation and without a viscous reference. Details in README.md and VALIDATION.md.

## Known issues

- Some viscous points do not converge and are returned with `converged=False` (never as a silent
  result), mostly low angles of flapped sections at Re 6e6 and post-stall angles at low Reynolds
  numbers. Neighbouring angles of a polar are usually unaffected.
- Plain-flap lift is below XFOIL's at large deflection (mean 0.02 to 0.04 at 15 to 20 deg); at 20
  deg it is traced to the separated boundary layer on the flap (VALIDATION.md, Limits).
- E387 at Re 5e4 (reported, not validated) can converge to a separated solution where XFOIL keeps
  the flow attached.
- A single `fl.analyze` call on such a hard point can take a few minutes while it tries other
  starting angles; `fl.polar` over a range of angles is faster there.

## Install

`pip install foil_lab-2.2.0-py3-none-any.whl` (from the release assets), Python 3.10 or newer.
