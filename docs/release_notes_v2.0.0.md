# Foil-Lab 2.0.0 (draft release notes)

Foil-Lab 2.0 is a rework of v1.3.1 whose aim is "correct and proven": every result path is
checked against an exact solution or against XFOIL 6.99, and the comparison is reproducible with
one command. Flaps and multi-element sections are the headline capability.

## Highlights

- One coupled viscous-inviscid solver (integral boundary layer and wake in one Newton system, e^n
  transition). Against XFOIL on identical coordinate files it meets the published targets at every
  Reynolds number from 1e5 to 6e6 for sections with an open trailing edge (VALIDATION.md).
- Plain flaps from -30 to +30 deg with hinge moment and flap polar sets; validated at 0 deg,
  reported without a claim up to 20 deg.
- Several elements, inviscid and viscous, with forces per element.
- Corrections applied once, in a fixed recorded order: ground effect (image method), simple sweep,
  Kármán-Tsien with the critical Mach number, finite aspect ratio.
- A single Python API (`fl.analyze`, `fl.polar`), pandas tables, plot helpers, exports, airfoils by
  name from the UIUC database, a polar database, a batch runner, a command line and a repaired GUI.

## Breaking changes from 1.3.1

- The v1.3.1 viscous paths (one-way drag, transpiration VII and its variants, roughness, post-stall
  model) are removed; with a Reynolds number every interface runs the coupled solver.
- Ground effect pitches the section about a pivot in a stream parallel to the ground.
- The aspect-ratio correction no longer applies compressibility a second time; the sweep correction
  uses the normal-plane angle (lift slope proportional to cos(sweep)).
- The approximate NACA 6-series generator is removed. `plotting.py` is replaced by `foil_lab.plot`.

## Limits

Subsonic, attached or mildly separated flow; viscous results at Mach 0; validated Re 1e5 to 6e6;
maximum lift and post-stall approximate; multi-element results without wake merging or slot
separation and without a viscous reference. Details in README.md and VALIDATION.md.

## Install

`pip install foil-lab-2.0.0-py3-none-any.whl` (from the release assets), Python 3.10 or newer.
