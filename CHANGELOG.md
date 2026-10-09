# Changelog

## Unreleased (M2, in progress)

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

### Removed
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

