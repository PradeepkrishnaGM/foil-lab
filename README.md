# Foil Lab v1.3 — First Low-Re VII Convergence (experimental)

A Python-native airfoil analysis and design tool built from first principles: a
2D **panel-method** aerodynamics solver with an interactive PySide6 GUI, a clean
scripting API, an optional **JAX backend** for differentiable inviscid analysis,
**viscous-inviscid interaction (VII)** coupling, **e^n transition prediction**,
and **inverse design**.

> **What this is (and isn't).** Foil Lab is a 2D incompressible **panel method**
> (source + vortex) with integral boundary-layer coupling — in the spirit of
> XFOIL, *not* a mesh-based CFD / RANS solver.

## Quickstart

```bash
# from the project directory, in a fresh virtualenv:
pip install -e .              # core (numpy, scipy, matplotlib)
pip install -e ".[all]"       # + JAX, GUI, batch formats, pytest

foillab info                                    # show version + backends
foillab analyze --naca 2412 --alpha 5 --re 1e6  # single point
foillab polar --naca 2412 --start -4 --end 12 --re 3e6 --vii transpiration
foillab polar --naca 2412 --start 0 --end 6 --re 50000 --vii newton   # low-Re (experimental)
python main.py                                  # launch the GUI
```

Testing on your own machine? See **[TESTING.md](TESTING.md)** for a full
step-by-step protocol and what output to capture.

> **v1.3 highlight.** The new simultaneous-**Newton VII** (`foil_lab.vii_newton`)
> plus a laminar-separation-bubble closure achieves the project's **first
> convergence at low Reynolds number** (Re=50k), where every prior method failed
> at every angle. It converges at low-to-moderate α with drag of the right order
> vs XFOIL; higher α and quantitative drag accuracy are still WIP. It is
> experimental (honest `converged` flag, runtime warning); the validated high-Re
> path remains `vii.solve_vii`. See [CHANGELOG.md](CHANGELOG.md).

## Validation

The earlier validation harness (`foil_lab.validation`, `examples/validation_report.py`) and its
two reference files were removed in milestone M1. Accuracy is now measured by a benchmark against
a locally installed XFOIL (`python -m foil_lab.benchmark all`, see `foil_lab/benchmark/`). The
measured state of this version is in [`docs/baseline_v1.3.1.md`](docs/baseline_v1.3.1.md); it
shows large errors in several result paths. This section will be rewritten from the benchmark
output in a later milestone.

---

## What's new in v1.0 — inverse design

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

## What's new in v0.9 — e^n transition

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

## What's new in v0.8 — VII coupling

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

## What's new in v0.7 — the JAX core

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

## What's new in v0.6

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

## What changed in v0.5

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

## New features

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
* **NACA 6-series** laminar-flow approximation (thickness form + `a`-mean line).
* **Biconvex** (double-circular-arc) and **double-wedge** supersonic sections.

### Geometry processing (`geometry.py`)
* **Coordinate smoothing** — fast vectorized weighted-average (Laplacian) filter
  to damp numerical bumps in imported `.dat` data; preserves the trailing edge.

---

## Installation

```bash
cd foil-lab
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install numpy scipy PySide6 matplotlib
```

Requires Python 3.10+, NumPy, SciPy. The GUI additionally needs PySide6 and
Matplotlib.

## Usage

### GUI
```bash
python main.py
```
Load a `.dat` file or generate a shape, set flight conditions (including
roughness, sweep, ground clearance), then run a single case or an alpha sweep.

Generator input box syntax:
| Input | Produces |
|-------|----------|
| `2412` | NACA 4-digit |
| `23012` | NACA 5-digit |
| `6:63,0.12,0.4` | NACA 6-series (series, t/c, Cl_design) |
| `bc:0.06` | Biconvex, t/c = 0.06 |
| `wedge:0.05` | Double-wedge, t/c = 0.05 |

### Scripting
```python
from foil_lab.generators import generate_naca4
from foil_lab.solver import solve_flow
from foil_lab.analysis import run_alpha_sweep

af = generate_naca4("2412", num_panels=200)

sol = solve_flow(af, alpha_deg=5.0, reynolds=3e6)
print(sol.cl, sol.cd_total, sol.cm_qc)

polar = run_alpha_sweep(af, -4, 16, 1.0, reynolds=3e6,
                        roughness=1, enable_post_stall=True)
```

Ground effect and sweep:
```python
from foil_lab.ground_effect import solve_ground_effect
from foil_lab.advanced import solve_swept_wing, solve_constant_wing_loading

ge = solve_ground_effect(af, 5.0, height=0.5)          # h/c = 0.5
sw = solve_swept_wing(af, 5.0, sweep_deg=30.0)
wl = solve_constant_wing_loading(af, 5.0, wing_loading=600, chord=1.2)
```

### Swapping the numerical kernel
```python
from foil_lab.influence import InfluenceModel
from foil_lab.solver import solve_flow

class MyModel:                      # implement compute(geom) -> InfluenceCoeffs
    def compute(self, geom): ...

solve_flow(af, 5.0, influence_model=MyModel())
```

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

74 tests cover inverse design (target-Cl convergence, Cp recovery, thickness constraints), e^n transition (flat-plate Re_x validation, N_crit sensitivity), VII (Blasius/turbulent flat-plate BL validation, drag-rises-with-alpha, lift reduction vs inviscid), the JAX backend (NumPy parity, autodiff vs finite-difference, vmap), and the physics fix (symmetry, ~2π slope, orientation independence,
near-field = circulation), every feature (multi-element, both kernels, trip
wires, batch export, ground effect, sweep, post-stall, roughness, generators),
geometry/IO, and validation against published NACA zero-lift angles and
aerodynamic-center location.

---

## Limitations & honesty notes

* **Method.** Inviscid panel method + integral BL + empirical corrections. Good
  for attached, subsonic, preliminary 2D work. Not valid for strongly separated,
  transonic-with-shocks, or supersonic flow (the biconvex/wedge generators build
  the geometry, but the incompressible panel solver does not model supersonic
  physics — use them as inputs to an appropriate solver).
* **Drag at low alpha** is under-predicted, as in any inviscid + integral-BL
  method without full viscous–inviscid interaction (so absolute L/D figures run
  optimistic; trends are meaningful).
* **Post-stall** is an empirical clamp, not a prediction; calibrate per airfoil
  family before trusting Cl_max.
* **Multi-element** now uses true per-body circulation and Kutta conditions
  (v0.6). Bodies must not overlap (`bodies_overlap` flags this). There is still
  no wake/confluent-boundary-layer model, so close-coupled slotted systems are
  inviscid-accurate only.
* **NACA 6-series** is a practical approximation, not the exact tabulated
  ordinates.
* **Compressibility** is a Kármán–Tsien Cp correction only (subsonic).

## Not yet implemented (scoped, deliberately deferred)

These were requested and are real, substantial projects rather than features;
they are intentionally **not** stubbed in, to keep every shipped capability
trustworthy:

* **Full inverse-formulation VII** — the VII solver is the direct/transpiration
  method (attached + mild separation). The XFOIL-grade inverse/simultaneous
  Newton solve for deep separation bubbles is the next step.
* **3D VLM / 3D panel method** — a separate solver (finite wings).
* **Transonic Euler / full-potential** — a different governing equation needing
  an actual mesh; outside the panel-method paradigm.
* **Unsteady discrete-vortex method** — time-marching with free-wake shedding.

*Done in v0.7:* JAX core port. *v0.8:* transpiration VII + integral BL marcher.
*v0.9:* e^n envelope transition. *v1.0:* inverse design & optimization
(PARSEC + SLSQP on the pre-compiled JAX core).

## Tech stack
Python · NumPy · SciPy · JAX · PySide6 · Matplotlib · pytest · pyarrow · h5py

