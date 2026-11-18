# Foil-Lab API reference (v2.0)

Everything below is importable from the package root (`import foil_lab as fl`) unless a module is
named. Conventions (SPEC.md section 3): coordinates in Selig order; angles in degrees; Cl, Cd, Cm
per unit span on the reference chord; Cm about the moment reference point, nose up positive; hinge
moment Ch positive when it tends to deflect the flap trailing edge down.

## Geometry

| Name | What it does |
|---|---|
| `fl.Airfoil.naca(code, num_panels=None, trailing_edge="open")` | NACA 4- or 5-digit section |
| `fl.Airfoil.from_file(path)` | Selig or Lednicer coordinate file |
| `fl.Airfoil.from_name(name)` | "NACA 2412", or a UIUC database name (downloaded once into `~/.cache/foil-lab/airfoils`, or `$FOIL_LAB_CACHE/airfoils`) |
| `fl.Airfoil.from_coords(coords, name)` | from an (n, 2) array |
| `af.scaled(f)`, `af.translated(dx, dy)`, `af.rotated(deg, pivot)` | rigid operations (shape preserved) |
| `af.repanelled(n)` | arc-length repanelling with leading- and trailing-edge clustering |
| `af.with_te_gap(gap)`, `af.with_thickness_scaled(f)`, `af.with_camber_scaled(f)` | shape operations |
| `af.with_flap(hinge_x, deflection, hinge_y=0.5)` | plain flap, -30 to +30 deg, smooth hinge with fairings |
| `fl.Element(airfoil, offset, scale, rotation_deg, pivot)` | an airfoil placed in a configuration |
| `fl.Configuration(elements, ref_chord=None, ref_point=None, name=...)` | one or more elements; reference chord and moment point default to the first element's chord and quarter-chord point |

## Analysis

```python
fl.analyze(geometry, alpha, re=None, repanel=True, num_panels=None, n_crit=None,
           trip_upper=None, trip_lower=None, params=None, mach=0.0, sweep=0.0,
           aspect_ratio=None, ground_height=None, pivot=None) -> Result
fl.polar(geometry, alphas, re=None, ...same options...) -> Polar
fl.flap_polars(airfoil, hinge_x, deflections, alphas, re=None, hinge_y=0.5, **polar_kwargs) -> FlapPolarSet
```

- `geometry`: `Airfoil`, `Element` or `Configuration`.
- `re=None`: inviscid. With `re` the coupled viscous-inviscid solver runs (Mach 0) for one or
  several elements, with e^n transition (`n_crit`, default 9) and optional trips on element 0.
- Corrections, each applied once in this order and listed in `Result.corrections`: ground effect
  (`ground_height` in reference chords above the ground, section pitched about `pivot`), sweep
  (simple sweep theory), compressibility (`mach`, Kármán-Tsien, inviscid only), finite aspect ratio
  (Helmbold lift factor, induced drag in `cd` and `cd_induced`).

### `Result`

`alpha, cl, cd, cm, ch, converged, reynolds, mach, sweep, aspect_ratio, ground_height,
mach_critical, cd_friction, cd_pressure, cd_induced, xtr_upper, xtr_lower, n_crit, trips,
iterations, iterations_total, residual, path, method, corrections, warnings, ref_chord, ref_point,
elements, version`.

- `converged` is true only when the solver's residual criteria were met; `path` says how a viscous
  point was reached; `warnings` lists anything the user must know (closed trailing edge drag,
  Mach above critical, ...).
- `elements`: one `ElementResult` per element (`name, cl, cm, cd, ch, cd_friction, xtr_upper,
  xtr_lower, x, y, cp, surface`).
- `to_dict()`, `row()` (flat), `to_dataframe()`, `elements_dataframe()`, `surface_dataframe()`
  (pandas).

### `Polar`

`results`, and arrays `alpha, cl, cd, cm, ch, converged`; `rows()`, `to_dicts()`,
`to_dataframe()`.

### `FlapPolarSet`

`deflections, alphas, tables` (cl, cd, cm, ch; NaN where not converged); `at(deflection)` (solved
or PCHIP-interpolated `FlapSlice`), `value(quantity, alpha, deflection)`, `leave_one_out(...)`,
`to_dataframe()`.

## Plots (`foil_lab.plot`)

Every helper returns `(figure, axes)`, accepts an existing `ax`, and never calls `show()`.

- `plot.geometry(geometry, ax=None, panels=False, ground=None)`
- `plot.cp(result_or_list, ax=None, labels=None)`
- `plot.polars(polars, labels=None, panels=(("alpha","cl"), ("cd","cl"), ("alpha","cm")), reference=None)`:
  overlays of several polars (sections, Reynolds numbers, flap angles) and reference polars
  (`XfoilPolar` from `foil_lab.benchmark.references.load_reference`).
- `plot.flap_set(flap_polar_set, quantity="cl", deflections=None)`

## Files (`foil_lab.export`)

- `export.to_csv(obj, path)`, `export.to_json(obj, path)` for a `Result`, `Polar`, `FlapPolarSet`
  or `FlapSlice`.
- `export.to_polar_text(polar, path, name)`: the column layout of an XFOIL polar file, written by
  Foil-Lab.
- `export.save_coordinates(airfoil, path)`: Selig coordinate file.

## Polar database

```python
db = fl.PolarDatabase("polars.sqlite")
rows = db.polar(geometry, alphas, re=1e6)      # stored rows reused, missing angles computed and stored
db.add(result_or_polar, geometry)
db.query(geometry=None, name=None, method=None, version=None, converged=None, **conditions)
db.to_dataframe(**filters)
fl.geometry_hash(geometry)
```

Rows are keyed by geometry hash, conditions (Reynolds and Mach number, sweep, aspect ratio,
ground height, N_crit, trips, panel count), angle, method and code version.

## Batch runner (`foil_lab.batch`)

```python
cases = batch.build_matrix(airfoils, alpha=range(-4, 11), re=[1e6, 3e6], mach=[0.0], ...)
rows = batch.run_batch(cases, airfoils, backend="serial" | "process", workers=None)
batch.write_table(rows, "out.parquet")          # .parquet, .h5 / .hdf5, else CSV
batch.to_dataframe(rows)
```

Cases that differ only in alpha are solved as one polar. Tested on Python 3.10, 3.12 and 3.14.

## Command line (`foillab`)

```
foillab analyze  (--naca CODE | --dat FILE | --name NAME) --alpha DEG [--re RE] [--flap X DEG]
                 [--mach M] [--sweep DEG] [--aspect-ratio AR] [--ground-height H] [--output F]
foillab polar    ... --start --end --step [--db polars.sqlite] [--output F.csv|.json|.pol]
foillab coords   (--naca | --dat | --name) [--panels N] --output F.dat
foillab benchmark <arguments of python -m foil_lab.benchmark>
foillab info
```

Invalid input gives one "error: ..." line and a non-zero exit status.
