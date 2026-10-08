# Foil-Lab v2.0 specification

Status: agreed scope, 8 October 2026. Starting point: v1.3.1 (baseline commit `be54d0a`).

## 1. Purpose

Foil-Lab is an MIT-licensed, pip-installable 2D panel-method solver for single, flapped and
multi-element airfoil sections, with its accuracy against XFOIL published for every case.

v2.0 has two aims, in this order:

1. **Correct and proven.** Every result path is checked against an exact solution or against
   XFOIL, and the comparison is reproducible with one command.
2. **Flaps and multi-element as the headline capability**, reachable from a clean Python API,
   the command line and the GUI.

A claim may appear in the README only if a test or a benchmark figure backs it.

## 2. Scope

**In v2.0**

- Reproducible benchmark against a locally installed XFOIL, plus exact-solution tests.
- One inviscid solver and one coupled viscous solver, valid at Re from 5e5 to 6e6.
- Geometry model with elements, arc-length repanelling and working transforms.
- Plain flaps with hinge moment, and polars across flap angle.
- Multi-element sections: inviscid, then a boundary layer per element.
- Corrected compressibility (with critical Mach), ground effect, aspect-ratio and sweep corrections.
- Python API, command line, batch runner, polar database, airfoil import by name, polar overlays.
- GUI repaired so that everything it shows is correct. No new GUI features.
- Release: user-facing README with images, validation section, downloadable package.

**Later**

- v2.1: CST and Bezier parameterisation, multi-point optimisation, 360° polar extrapolation,
  export to QBlade and XFLR5 formats, new GUI features.
- v2.2: low-Reynolds-number accuracy (laminar separation bubbles) and stall prediction, only if
  the benchmark shows a method that works.

**Not planned**

- Transonic flow, finite-wing solvers (lifting line, vortex lattice), propeller or turbine blade
  design, machine-learning surrogates, a web GUI.

## 3. Conventions

- Coordinates are chord-normalised, Selig order: trailing edge, upper surface, leading edge,
  lower surface, trailing edge. Internal ordering is the solver's business and must never leak
  into labels or output.
- Angle of attack in degrees. Reynolds number based on the reference chord and freestream speed.
- `Cl`, `Cd`, `Cm` are per unit span, normalised by the reference chord. `Cm` is about the
  quarter-chord point of the reference chord, nose-up positive.
- For multi-element sections the reference chord is stated in the result. Forces are reported
  per element and in total.
- Hinge moment is reported as a coefficient normalised by dynamic pressure and the reference
  chord squared, so it can be compared with the value XFOIL reports. The sign convention is
  documented and covered by a test.
- Drag is reported in total and split into friction and pressure parts where the method allows.

## 4. Requirements

Accuracy targets marked *provisional* are confirmed or revised after the first baseline run
(milestone M1). The benchmark always reports the measured numbers, whether or not a target is met.

### 4.1 Benchmark against XFOIL

- XFOIL is used only as an external program, run as a subprocess with a timeout. No XFOIL
  source is copied, translated, linked or vendored.
- Both codes receive the same coordinate file.
- Each reference polar is stored with its provenance: XFOIL version, panel count, N_crit,
  Reynolds number, Mach number, geometry source and date.
- Benchmark matrix:
  - Sections: NACA 0012, 2412, 4412, 23012, plus at least two non-NACA sections imported by name.
  - Validated range: Re = 5e5, 1e6, 3e6, 6e6. N_crit = 9. Mach = 0.
  - Reported but outside the validated range: Re = 5e4, 1e5, 2e5.
  - Angle of attack from -4° up to the last angle at which XFOIL converges, in 1° steps.
  - Flap cases: NACA 0012 and 2412 with a 25 % chord flap at 0°, 5°, 10°, 15°, 20°.
- Reported per case: `Cl`, `Cd`, `Cm`, transition location on each surface, hinge moment for
  flap cases, convergence status of both codes, and run time.
- Error metrics per case and per Reynolds number: mean and maximum absolute difference in `Cl`
  and `Cm`, mean and maximum relative difference in `Cd`.
- *Provisional* targets in the validated range, for attached flow (below 90 % of XFOIL's
  maximum lift): mean |ΔCl| ≤ 0.03, mean relative `Cd` difference ≤ 10 %, mean |ΔCm| ≤ 0.01.
- One command regenerates all reference data, runs the comparison, and writes the tables and
  plots used in the validation document.
- Generated data stays out of git. Small curated reference files with provenance headers are
  committed so that tests run without XFOIL installed.

### 4.2 Exact-solution and consistency tests

- Surface pressure on a Joukowski or Kármán-Trefftz airfoil against the analytic solution.
- A two-element case with different bodies against a published exact potential-flow solution.
- Ground effect against the equivalent two-body mirror problem solved in free air.
- Lift from surface pressure equals lift from circulation, and equals lift from a far-field
  momentum contour.
- Laminar flat plate against Blasius; turbulent flat plate against a standard skin-friction law.
- Geometry operations on scaled, translated and rotated shapes: the shape is preserved.
- Compressibility and aspect-ratio corrections are each applied exactly once.
- NumPy and JAX inviscid solvers agree to 1e-6 in `Cl`.

### 4.3 Solver

- One assembly of the inviscid linear system, used by every mode. Each body closes on itself.
- The vorticity model is what the documentation says it is. If a higher-order option exists,
  a test shows that it differs from the low-order one and converges faster.
- One boundary-layer implementation and one coupled viscous method. Experimental solvers that
  do not converge are removed from the public interface.
- Every result carries a `converged` flag that is true only if the stated residual criteria
  were met. There are no silent fallbacks: if a fallback is used, the result says which.
- Transition by e^n (default) or Michel, with forced trip locations per surface.
- All tunable numbers live in one parameters module with documented defaults.

### 4.4 Geometry

- `Airfoil`: coordinates, name, source, trailing-edge gap. Created from NACA 4 and 5 digit
  generators, a coordinate file, or a database name.
- `Element` and `Configuration`: one or more airfoils with position, scale, rotation and pivot.
- Repanelling by arc length with leading- and trailing-edge refinement, independent of chord
  length and position.
- Operations: scale, translate, rotate, set trailing-edge gap, scale thickness and camber, flap.
- Generators that are known to be wrong (the approximate 6-series) are either corrected against
  published ordinates or removed.
- Import by name downloads coordinates on demand into a cache outside the repository. Third-party
  coordinate or performance data is not bundled unless its licence clearly allows it.

### 4.5 Flaps

- Plain flap defined by hinge position (chordwise and through the thickness) and deflection.
  The hinge region is closed with a smooth arc, with no gap and no kink.
- Deflections from -30° to +30° produce valid geometry. Viscous accuracy is benchmarked up to 20°.
- Hinge moment coefficient is computed from surface pressure and friction on the flap.
- A flap polar set covers several deflections and supports interpolation in deflection angle.

### 4.6 Multi-element

- Inviscid analysis of two or more elements with arbitrary relative position, validated by 4.2.
- Viscous analysis with a boundary layer on each element, driven by the coupled inviscid flow.
- Stated limits: no merging of an upstream wake with a downstream boundary layer, no slot-flow
  separation model, no maximum-lift prediction for high-lift configurations.
- A slotted-flap example ships with the release, with its limits printed next to the result.

### 4.7 Corrections

- Kármán-Tsien compressibility, applied once. Critical Mach number is reported. A result above
  the critical Mach number carries a warning.
- Ground effect: the freestream stays parallel to the ground and the section rotates about a
  pivot. Validated by 4.2.
- Aspect-ratio and simple-sweep corrections are applied in a fixed, documented order, and the
  result records which corrections were applied.

### 4.8 Interfaces

- Python API importable from the package root. Illustrative shape, exact names to be decided:

  ```python
  import foil_lab as fl

  af = fl.Airfoil.naca("2412")
  res = fl.analyze(af, alpha=4.0, re=3e6)             # one operating point
  pol = fl.polar(af, alphas=range(-4, 15), re=3e6)    # a polar
  flapped = af.with_flap(hinge_x=0.75, deflection=10.0)
  ```

- Results convert to pandas tables. Plot helpers return the figure and axes and never block.
- Polar overlays: several sections, Reynolds numbers, flap angles or reference polars on one plot.
- Command line: analyse a point, compute a polar, run the benchmark, show package information.
  Invalid input gives a clear message, not a traceback.
- Batch runner with serial and parallel backends that work on all supported Python versions.
- Polar database: rows keyed by geometry hash, conditions, method, code version and converged flag.
- Export: CSV and JSON for results, XFOIL-style polar text files, coordinate files.

### 4.9 GUI

Repair only. After v2.0 the GUI:

- labels upper and lower surfaces correctly,
- shows a usable flow field that agrees with the solver,
- redraws after every geometry change,
- runs long solves in a worker thread with progress and cancel,
- calls the same analysis API as the command line, including flaps and multi-element,
- reports drag once.

### 4.10 Quality

- Tests are marked fast or slow. The fast suite runs in under 60 seconds on a 6-core desktop CPU.
- Same input gives the same output, bit for bit, on one machine.
- Reference values are recorded before any speed optimisation and must not change after it.
- Supported Python versions are declared in `pyproject.toml` and tested.
- No path hacks; data files are loaded through package resources.

### 4.11 Licence and provenance

- Foil-Lab is MIT licensed.
- Methods are implemented from published papers and textbooks, cited in the code.
- `THIRD_PARTY.md` lists dependencies, data sources and their licences.

### 4.12 Release

- README: what it is, install, a 30-second example, images, capabilities, validation summary
  with links to the full tables, known limits, citation of methods.
- `VALIDATION.md` generated from the benchmark output.
- `CHANGELOG.md` holds the version history that is currently in the README.
- GitHub release with source distribution and wheel, built by GitHub Actions on tags and
  manual runs only.

## 5. Limits to state in the README

- Subsonic, attached or mildly separated flow. Not for transonic flow.
- Validated at Re from 5e5 to 6e6. Lower Reynolds numbers are reported in the benchmark but
  not validated.
- Maximum lift and post-stall behaviour are approximate.
- Multi-element results exclude wake merging and slot separation.
