# Changelog

## v2.0.0 (prepared; not yet released)

Foil-Lab v2.0 is a rework of v1.3.1 to SPEC.md: every result path is checked against an exact
solution or against XFOIL, and flaps and several elements are the headline capability. Details of
every decision: DECISIONS.md; accuracy: VALIDATION.md.

* **Benchmark harness** (M1): XFOIL 6.99 as an external program on the same coordinate file,
  committed reference polars with provenance, error tables against the SPEC.md 4.1 targets.
* **Geometry and one inviscid core** (M2): `Airfoil`, `Element`, `Configuration`; arc-length
  repanelling; one linear-vortex panel assembly for every mode, each body closed on itself; the
  approximate 6-series generator removed; JAX core in agreement to 1e-6.
* **One coupled viscous-inviscid solver** (M3): integral boundary layer and wake in one Newton system,
  e^n transition, a truthful `converged` flag; the v1.3.1 one-way drag path and the non-converging
  VII solvers removed. Meets the SPEC.md 4.1 targets at Re 1e5 to 6e6.
* **Plain flaps and hinge moment** (M4): smooth hinge geometry from -30 to +30 deg, hinge moment with
  a documented sign, flap polar sets with interpolation in deflection; flap cases in the benchmark.
* **Several elements** (M5): a boundary layer and wake per element, forces per element and in total.
* **Corrections** (M6): Kármán-Tsien applied once with the critical Mach number, ground effect by the
  image method (validated by the mirror problem), simple sweep and Helmbold's finite-wing factor, in
  a fixed, recorded order. v1.3.1 applied compressibility twice with an aspect ratio.
* **Interfaces** (M7): `fl.analyze`, `fl.polar`, pandas tables, plot helpers returning figure and axes,
  exports, airfoils by name from the UIUC database, a SQLite polar database, the batch runner and the
  command line on one API; an example notebook.
* **GUI repair** (M8): a thin client of the API with solves in a worker thread.
* **Removed: the v1.3.1-shaped interface.** `foil_lab.solver` (`solve_flow`, `solve_flow_sweep`,
  `SolutionData`, `compute_panel_geometry`), `foil_lab.analysis` (`run_alpha_sweep` and its
  helpers), `foil_lab.advanced` (`solve_swept_wing`, `solve_constant_wing_loading`),
  `ground_effect.solve_ground_effect` and `flowfield.compute_velocity_field`. Use `fl.analyze` and
  `fl.polar` (with `sweep`, `ground_height`, `aspect_ratio`, `mach`, `re`) and
  `flowfield.velocity_field`. The constant-wing-loading polar has no replacement in v2.0.

## Development notes from M2 (kept for the record)

### Behaviour changes
* The pitching moment about the quarter chord (`cm_qc`) is now the exact moment of the surface
  pressure about (0.25, 0). v1.3.1 used `cm_le + 0.25 * cl`, which mixes lift and normal force
  (about 4e-3 different at Cl 1.5 and 10 deg).

* All potential-flow solves go through one inviscid core (`foil_lab.inviscid`, linear-strength
  vortex panels): `solve_flow`, the three VII variants, the flow field, ground effect and the
  JAX entry points. Results change accordingly (see DECISIONS.md, M2).
* Open trailing edges carry a trailing-edge panel (Drela 1989, eq. 2); closed (sharp) trailing
  edges use Drela's eq. (9) with the Kutta condition imposed exactly.
* Ground effect: the section is pitched by alpha about its quarter chord and raised by `height`
  in a stream parallel to the ground (v1.3.1 inclined the stream, which crossed the wall). The
  returned `alpha_deg` is 0 (the stream direction) and `coords_list` holds the pitched, raised
  section.
* `SolutionData.gamma` holds the nodal vortex strengths, `sources` the known panel sources
  (transpiration), and a new field `panels` the panel set.
* `compute_velocity_field` uses the solution's own singularities and a true inside-body test.
* `jax_core` no longer switches JAX's global float64 setting on import; each call runs in a
  scoped float64 context and takes NumPy arrays.
* `python -m foil_lab.benchmark report` no longer rewrites `docs/baseline_v1.3.1.md`; use the
  new `baseline-doc` command.

* NACA 4/5-digit generators default to the open trailing edge (NACA Report 460); the closed
  geometry is unchanged under `trailing_edge="closed"`.
* `geometry.repanel_cosine` repanels by arc length (independent of chord, position and
  rotation) instead of interpolating y(x) per surface.
* New public interface: `fl.Airfoil`, `fl.Element`, `fl.Configuration`, `fl.analyze`,
  `fl.polar` (inviscid). Version 2.0.0.dev0.

### Removed
* `generate_naca6` (approximate 6-series; a correct generator is planned after v2.0).
* `foil_lab.panels` (`build_panels`, `PanelGeometry`, `ensure_cw`): replaced by
  `foil_lab.inviscid.PanelSet`.
* `foil_lab.influence`, `foil_lab.linear_system`, `foil_lab.forces` (the Hess-Smith kernels,
  the private assembly and the old load integration), and the `influence_model` argument of
  `solve_flow`.

## v1.3 — Low-Re VII: simultaneous Newton coupling (partial success)

**Goal:** make VII converge at low Reynolds number, where every prior method gave
0 converged points.

**Outcome: real, partial success.** For the first time, VII converges at Re=50k —
at low-to-moderate angles — with drag of the right order of magnitude vs XFOIL.
Higher low-Re angles do not yet converge. Shipped as an experimental path
(`vii_newton`), labeled honestly; the validated high-Re `vii.solve_vii` is
unchanged.

### What worked
* **Simultaneous-Newton coupling** (`foil_lab.vii_newton.solve_vii_newton`).
  Instead of iterating inviscid<->BL with under-relaxation (which limit-cycled),
  it solves the coupled fixed point `v_n = T(u_e0 + D v_n)` by Newton's method.
  The inviscid sensitivity `D` (transpiration -> edge velocity) is **exact**
  because the panel system is linear (`vt = vt0 + D v_n` to machine precision);
  the viscous Jacobian is finite-differenced; a line search handles the
  near-singular coupled Jacobian. Validated to converge at Re=1e6 and match the
  transpiration VII there to <0.02 in Cl.
* **Laminar-separation-bubble closure** in `bl_integral`. At low Re the laminar
  layer separates before transition; the marcher now forces transition at the
  laminar-separation point (short-bubble reattachment model). This is what made
  low-Re convergence possible — without it `T(u_e)` is ill-posed and Newton
  cannot reach a root. The high-Re path is unaffected (verified: NACA 2412 at
  Re=3e6 still Cl=0.633, Cd=0.0061, unchanged).

### Honest current state (Re=50k, NACA 2412)
* Converges at alpha = 0, 2 deg. Cd ~0.017 vs XFOIL ~0.024-0.031 — same order,
  ~30-45% low, versus the old one-way path's ~0.001 (20x too low).
* Does NOT yet converge at alpha >= ~4 deg, where the bubble is longer and the
  integral closure is weakest. Reported by the `converged` flag; pinned by an
  xfail test so a future fix shows up as a change.
* Drag at low Re is approximate, not yet a quantitative XFOIL match.

### Why this matters
This is the first genuine movement on the low-Re wall that blocked v0.8-v1.2.
The coupling is now correct (the hard part); the remaining gap is the
boundary-layer *bubble model*, a more localized problem than the coupling
instability we started with. The diagnosis is precise: extend the bubble closure
(transitional separation length, longer bubbles) to reach higher angles.

---

## v1.2 — Low-Re VII groundwork (experimental, did not converge)
Veldman quasi-simultaneous machinery; stopped divergence but limit-cycled at the
trailing edge. Superseded by the v1.3 Newton approach.

## v1.1 — Validation harness & honest envelope
Reproducible validation vs XFOIL 6.96 and thin-airfoil theory.

## v1.0 — Inverse design & optimization (PARSEC + SLSQP on the JAX core).
## v0.9 — e^n envelope transition prediction.
## v0.8 — Transpiration VII + validated integral boundary-layer marcher.
## v0.7 — Differentiable JAX inviscid core (jit/vmap/grad).
## v0.6 — Multi-element, higher-order kernels, trip wires, batch runner.
## v0.5 — Corrected core physics; modular refactor.

## Release notes of v0.5 to v1.0 (archived from the v1.3.1 README)

These notes describe v1.x as it was; several of its features were replaced or removed in v2.0.

### What's new in v1.0 — inverse design

Until now the workflow was forward-only: you supplied a shape and got its
aerodynamics. v1.0 closes the loop the other way — you specify a *target* and the
tool finds a shape. The new `design.py` couples the **PARSEC** airfoil
parameterization (11 physically-meaningful parameters) to a gradient-based
optimizer (SLSQP), with three problem types:

* **`design_for_cl(target_cl, alpha)`** — find a shape that produces a target
  lift coefficient. Fast and reliable: hits the target to <0.01 in a couple of
  seconds (e.g. ask for Cl=0.9, get 0.901).
* **`match_cp(target_coords, alpha)`** — the classic inverse problem: recover a
  shape whose pressure distribution matches a target airfoil. Validated by
  recovering a NACA 2412 from its own Cp (t/c 0.123 vs 0.12, Cp MSE ~6e-3).
* **`maximize_ld(alpha, Re, target_cl=...)`** — maximize lift-to-drag, routed
  through the **VII** viscous solver so the drag is physical (essential — the
  one-way drag is too optimistic to optimize against).

Why this is fast: the expensive inner solve runs through the **pre-compiled JAX
inviscid core** from v0.7. Because the panel count is fixed during an
optimization, the JAX function compiles once and is reused for every candidate
geometry (~3 ms/eval instead of ~55 ms with recompilation). This is the payoff
of having built the differentiable core — inverse design is where it cashes in.

```python
from foil_lab.design import design_for_cl, match_cp
from foil_lab.generators import generate_naca4

# Design an airfoil for a target lift coefficient.
r = design_for_cl(target_cl=0.9, alpha_deg=5.0)
print(r.message, "->", r.coords.shape)

# Inverse-design: recover a shape matching a target's pressure distribution.
r2 = match_cp(generate_naca4("2412", 120), alpha_deg=5.0)
```

The GUI gains a **"Design for Cl"** control in the Geometry tab: enter a target
Cl, get an airfoil, then analyze it like any other shape.

### Honesty
* PARSEC spans a broad but not universal family; some targets aren't exactly
  representable, and the optimizer returns the closest PARSEC fit (the honest
  outcome of an ill-posed inverse problem). `match_cp` regularizes lightly
  toward the start to stay well-behaved.
* `maximize_ld` is the slow option (VII in the loop) and guards against the
  optimizer exploiting non-converged viscous cases; constrain it with
  `target_cl` for a meaningful result.

---

### What's new in v0.9 — e^n transition

The boundary-layer transition point was previously fixed by **Michel's
criterion** — a single empirical correlation that says "transition happens
around here" with no sensitivity to the flow environment. v0.9 adds the
physically-grounded **e^n envelope method** (Drela & Giles), now the default for
the viscous path.

Instead of a correlation, e^n tracks the amplification of Tollmien–Schlichting
instability waves: a log-amplification factor `n` accumulates along the surface
at a rate `dn/dRe_θ(H)` (a closed-form function of the local shape factor), and
transition is declared when `n` reaches a critical value `N_crit`. That gives
two things Michel can't:

* **A physical environment knob (`N_crit`).** `~9` for a clean tunnel / free
  flight, `~4–7` for a noisy stream or a rough/dirty surface, `~11–13` for a
  very quiet facility. Lower `N_crit` → earlier transition → more turbulent
  wetted area → more drag.
* **Correct pressure-gradient sensitivity.** Because the amplification rate and
  neutral point depend on the shape factor `H`, adverse gradients destabilize
  the boundary layer and advance transition automatically.

Validated against the textbook **flat-plate transition Reynolds number**: at
`N_crit=9`, e^n predicts transition at `Re_x ≈ 2.8×10⁶` (classic value
≈2.5–3×10⁶), and `N_crit` sweeps the location monotonically and smoothly.

```python
from foil_lab.generators import generate_naca4
from foil_lab.vii import solve_vii

# Dirty surface (low N_crit) vs a clean one — note the drag difference.
dirty = solve_vii(generate_naca4("2412", 200), 4.0, reynolds=3e6, n_crit=5)
clean = solve_vii(generate_naca4("2412", 200), 4.0, reynolds=3e6, n_crit=12)
print(dirty.cd_wake, clean.cd_wake)   # dirty > clean
```

The transition model is selectable per call (`transition_model="en"` default, or
`"michel"` for the legacy correlation), and an explicit trip still overrides
either. e^n drives the VII viscous path; the fast one-way drag estimate keeps
Michel (with its roughness coupling) for speed.

---

### What's new in v0.8 — VII coupling

Until now the boundary layer was one-way: the inviscid solver fed it edge
velocities, but it never fed back. The new `vii.py` closes that loop with the
**transpiration-velocity method**. The boundary-layer displacement thickness
δ* is modelled as an equivalent wall-blowing velocity `v_n = d(u_e·δ*)/ds` that
is injected into the inviscid flow-tangency condition; the inviscid and viscous
systems are then iterated to a converged fixed point with adaptive
under-relaxation.

This is backed by a new, separately-validated **integral boundary-layer
marcher** (`bl_integral.py`): Thwaites' method in the laminar region, Michel (or
explicit trip) transition, and Head's entrainment method in the turbulent
region. It is validated against the **Blasius** laminar similarity solution
(θ, δ* to ~1%) and the **turbulent flat-plate** skin-friction law
(`cf ≈ 0.0592·Re_x^-0.2` to ~2%).

What the coupling buys you, validated:

* **Realistic drag.** VII profile drag (Squire–Young from the wake momentum
  thickness) for NACA 2412 at Re=3e6 comes out ≈0.006–0.008, matching
  experiment; the old one-way path gave ≈0.0011 — unphysically low and nearly
  flat with α.
* **Drag rises with incidence** as the boundary layer thickens — the
  characteristic viscous behaviour the one-way path misses entirely.
* **Lift is reduced vs. inviscid** because δ* effectively decambers the
  airfoil — a real viscous effect, now captured.

```python
from foil_lab.generators import generate_naca4
from foil_lab.vii import solve_vii

r = solve_vii(generate_naca4("2412", 200), alpha_deg=4.0, reynolds=3e6)
print(r.solution.cl, r.cd_wake, r.converged, r.iterations)
```

### Honest scope of the VII solver
This is the **direct (transpiration) formulation** with under-relaxation. It is
real, converges, and clearly improves the physics — but with limits stated
plainly:

* Converges for **attached and mildly-separated** flow (roughly up to ~8–10° on
  a typical section at moderate Re). The `converged` flag reports honestly; past
  the limit it returns `converged=False` and those numbers should not be
  trusted.
* It does **not** robustly handle deep separation. Under prescribed edge
  velocity the integral BL approaches the Goldstein singularity near separation
  and the iteration stalls. Capturing large separation bubbles needs the
  **inverse / simultaneous-Newton** formulation (XFOIL-grade) — the documented
  follow-on. The JAX core's autodiff is intended to make that Newton Jacobian
  tractable.
* Single body, inviscid coupling only (no wake/confluent-BL model for
  multi-element).

---

### What's new in v0.7 — the JAX core

The inviscid forward solver now has a **numerically identical JAX twin**
(`jax_core.py`): the same panel geometry, linear-vortex influence, Kutta-closed
linear solve, and pressure-integrated loads, written as a single pure traceable
function. Validated to match the NumPy core to ~1e-6 across NACA cases. This
unlocks three things the pure-NumPy core could not do:

* **Compiled batch sweeps (`jax.vmap` + `jax.jit`).** A 200-angle polar runs in
  one vectorized, compiled call — measured **~100–150× faster** than the NumPy
  Python loop (≈1700 ms → ≈17 ms), bit-for-bit agreeing to 1e-6. This is the
  genuine parallel batch speedup the pure-Python `run_batch` honestly could not
  deliver; `batch.run_batch_jax` uses it and still exports to Parquet/HDF5/CSV.
* **Exact gradients (`jax.grad` / `jax.jacobian`).** `dCl/dα` matches finite
  differences to ~1e-10, and `dCl/d(geometry)` gives the lift sensitivity to all
  node coordinates at once. These are the foundation for gradient-based inverse
  design / shape optimization and (later) the VII Newton solve.
* **Device-agnostic.** Written against `jax.numpy`, so it runs on GPU/TPU with
  the appropriate JAX build. *Validated on CPU here; GPU/TPU is supported by JAX
  but untested in this environment — treat device acceleration as "should work,
  unverified".*

### Honest scope of the JAX port
* Covers the **inviscid single-body** path only. The viscous integral boundary
  layer has data-dependent transition searches (value-dependent branching) that
  do not trace cleanly under JAX, so it stays on NumPy. Use the JAX core for
  fast/differentiable inviscid sweeps and sensitivities; use the NumPy
  `solve_flow` for viscous drag, multi-element, ground effect, and post-stall.
* JAX is an **optional dependency** (`pip install 'jax[cpu]'`). Everything else
  works without it; JAX-only paths fall back or raise a clear message via
  `foil_lab.jax_backend`.

Quick start:
```python
import numpy as np
from foil_lab.generators import generate_naca4
from foil_lab import jax_core

af = generate_naca4("2412", 200)
cls, cms = jax_core.batch_polar(af, np.linspace(-8, 16, 200))   # compiled sweep
slope, alpha0 = jax_core.lift_slope_and_alpha0(af)              # exact, autodiff
```

---

### What's new in v0.6

Five validated features built on the v0.5 corrected core:

1. **True multi-element analysis.** Each body now gets its *own* bound-circulation
   unknown and its *own* Kutta condition, instead of a single shared
   circulation. Validated: two tandem airfoils far apart each recover the
   isolated single-airfoil lift; close together they show the correct mutual
   interference (front upwash raises its lift, rear sits in the front's downwash
   and loses lift). A bounding-box overlap check (`geometry.bodies_overlap`)
   warns when bodies intersect (panel methods require non-intersecting bodies).

2. **Higher-order singularity kernels.** The kernel is now a clean swappable
   `InfluenceModel`. `LinearVortexModel` (default) uses linearly varying,
   node-continuous vorticity — smooth surface-velocity gradients and no
   panel-to-panel vorticity jump. `ConstantVortexModel` provides the lower-order
   constant-strength element for comparison. Swap via
   `solve_flow(..., influence_model=...)`.

3. **Trip wires (localized transition).** Force boundary-layer transition at an
   explicit x/c on either surface via `trip_x_upper` / `trip_x_lower`. Validated:
   moving the trip forward monotonically increases drag (longer turbulent run),
   and a trip never delays transition past the natural point.

4. **Headless batch runner** (`batch.py`). Build a Cartesian matrix of
   airfoils × alpha × Re × Mach × roughness × trips, solve it with robust
   per-case error handling (one failure never kills the run), and export to
   **Parquet, HDF5, or CSV** by file extension.

5. **GUI overhaul.** The sidebar is now organized into logical tabs
   (Geometry / Conditions / Run), wrapped in a scroll area so it never
   overflows, with a draggable splitter between the controls and the plots.
   Trip-wire controls added.

### An honest note on batch parallelism
The batch runner exposes `workers`/`backend` knobs but **defaults to serial**.
With the current pure-Python/NumPy core, neither threads nor processes reliably
speed it up: per-case time is dominated by GIL-bound Python assembly (geometry
build, the nodal-vortex loop), not the GIL-releasing dense solve. The runner's
real value is matrix organization, robustness, and multi-format export. Genuine
parallel speedup needs the planned JAX core (vmap over the whole matrix) — see
"Not yet implemented" below. The knobs are wired so that lands cleanly later.

---

---

### What changed in v0.5

### The headline fix: correct lift
The previous core under-predicted lift by roughly 5–8× and could even return the
wrong sign (a symmetric airfoil reported non-zero lift at zero incidence). The
root cause was a single lumped constant-vortex closure that cannot represent an
airfoil's bound circulation. v0.5 replaces it with the standard **Hess–Smith
constant-source + constant-vortex panel method**. The solver now reproduces
textbook results:

| Case | Foil Lab v0.5 | Theory / reference |
|------|---------------|--------------------|
| NACA 0012, α = 0° | Cl = 0.000 | 0 (symmetry) |
| NACA 0012 lift slope | ≈ 6.0 /rad | 2π ≈ 6.28 /rad |
| NACA 2412 zero-lift angle | −2.15° | ≈ −2.1° |
| NACA 2412 aerodynamic center | 0.259c | 0.25c |
| Stagnation Cp | 0.999 | 1.000 |

Near-field (pressure-integral) lift and far-field (Kutta–Joukowski) lift agree,
and results converge cleanly with panel count — both signs of a correctly posed
panel solve.

### Architecture refactor
The monolithic solver was decomposed into single-responsibility modules so the
numerical method, boundary conditions, and post-processing are independently
testable and swappable:

```
panels.py         Geometry/discretization (the "mesh" layer). Forces clockwise
                  orientation and computes outward normals.
influence.py      The swappable singularity kernel. HessSmithModel implements
                  the InfluenceModel protocol; drop in a new class to try a
                  different element (e.g. higher-order vortex) without touching
                  anything else.
linear_system.py  Boundary-condition + Kutta assembly and the dense solve.
forces.py         Pressure integration -> Cl, Cm (per-body capable).
solver.py         Thin orchestrator wiring the stages together. Keeps the
                  legacy solve_flow / SolutionData / compute_panel_geometry API.
```

### Performance
The influence-coefficient build was an O(N²) Python double loop. It is now fully
vectorized with NumPy broadcasting over contiguous arrays:

* **~20.8× faster** influence assembly (≈174 ms → ≈8.4 ms at N = 160).
* Cache-friendly array operations; the off-body flowfield reconstruction is
  vectorized over grid points as well.

Numerically the refactor is faithful: the linear system, Kutta condition, and
conservation behaviour of a panel method (a closed body integrates zero net
force under a uniform pressure field) are preserved.

---

### New features

### Advanced flight conditions (`advanced.py`)
* **Constant wing-loading polar** — evaluate a sweep at fixed W/S instead of
  fixed Reynolds number, iterating speed ⇄ Re to a self-consistent trimmed
  flight point: `V = sqrt(2 (W/S) / (ρ Cl))`, `Re = ρ V c / μ`.
* **Busemann simple-sweep** — estimate swept-wing behaviour by solving in the
  plane normal to the leading edge; coefficients on the full freestream scale by
  `cos²Λ` (verified exact to 1e-3 at Λ = 30°).

### Viscous modifiers (`viscous_mods.py`, `boundary_layer.py`)
* **Surface roughness (0–3)** — lowers the Michel transition threshold (earlier
  turbulence) and linearly scales up profile drag (dirt/bugs/paint).
* **Empirical post-stall** — once the inviscid suction peak passes an onset
  value, lift is shed and a drag increment added, producing a rounded post-stall
  Cl curve. Calibrated to give a plausible Cl_max (~1.6–1.7 for NACA 24xx).
  *This is an engineering approximation, not a separated-flow prediction.*
* Also fixed: the boundary-layer routine now distinguishes a (reattaching)
  laminar separation bubble from genuine turbulent trailing-edge separation,
  instead of flagging the whole upper surface as separated at moderate α.

### Ground effect (`ground_effect.py`)
Rigid-wall **Method of Images**, done correctly. The feature spec proposed
appending a mirror airfoil as a second body — but this solver gives every body
its own free circulation, so an appended mirror is solved as a second free wing,
not a wall. v0.5 instead *slaves* the image singularities to the real ones
(image source +, image vortex −, reflected across y = 0), keeping y = 0 a
streamline and the unknown count equal to the real body. Result: lift **rises**
approaching the ground (the real ground-effect cushion), converging back to
free-air at large height. The deformable free-surface (high-Froude) case is
intentionally out of scope and raises `NotImplementedError`.

### Geometry generators (`generators.py`)
* **PARSEC** 11-parameter parameterization (for optimization studies).
* **Biconvex** (double-circular-arc) and **double-wedge** supersonic sections.

### Geometry processing (`geometry.py`)
* **Coordinate smoothing** — fast vectorized weighted-average (Laplacian) filter
  to damp numerical bumps in imported `.dat` data; preserves the trailing edge.

---
