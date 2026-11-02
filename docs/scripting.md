# Scripting Aerolume

Everything the desktop app does with designs, simulations and results can be done from
Python: build geometry, plan and run 2D and 3D CFD, airfoil polars and finite-wing analyses,
add heat transfer, fans, jets and smoke emitters, run reference cases and grid-convergence
studies, sweep parameters, compare designs, make figures (including 3D streamlines, smoke and
vortex pictures), export data and write reports. Scripts run in three places and behave the
same in each:

| where | how | `project` is |
|---|---|---|
| the app | **View ▸ Script** dock (Alt+4, or **Script** on the toolbar): editor, Run (Ctrl+Enter), output with inline figures | the open project |
| the command line | `aerolume-cli script study.py --project "Study.aerolume"` | the `--project` folder |
| Python, Jupyter | `import aerolume as al` | what you open or create |

Claude Desktop, Claude Code and other MCP clients can drive the same API through the
`aerolume-mcp` server, with no API key inside Aerolume: see docs/mcp.md.

Whatever a script does is recorded in the project exactly as if it had been done in the app:
its designs appear in the Project tree, its simulations run through the app's job queue (with
the live view on the Simulate page), and its results open on the Results, Compare and Report
pages.

The API lives in `aerolume.api` and is re-exported by the package: `import aerolume as al`
is fast and loads neither PyTorch, VTK nor Qt until a solver, a 3D picture or a figure is
needed.

---

## 1. Five lines

```python
import aerolume as al

p = al.new_project("Airfoil study")              # a folder "Airfoil study.aerolume"
d = p.add_naca("2412", chord=1.0)
polar = p.run_polar(d, re=1e6, alpha=(-4, 14, 1))
print(polar)                                     # Cl,max, best L/D, the table of points
```

In the app's Script dock the first two lines are not needed: `project` and `al` are already
bound.

## 2. Projects

```python
p = al.open_project("C:/Studies/Wing study.aerolume")      # ".aerolume" may be left out
p = al.open_project("Wing study", create=True)              # create it when missing
p = al.new_project("Wing study")                             # refuses an existing project
p = al.new_project()                                         # temporary, deleted when Python exits
with al.open_project("Wing study") as p:                     # saved on exit (changes are saved anyway)
    ...
```

`p.designs` (list of `Design`), `p.simulations`, `p.design(name_or_id)`,
`p.simulation(name_or_id)`, `p.result(name_or_id)` (the stored result or None),
`p.results(design=None)` and `p.summary()` (plain data: designs, simulations, status, key
numbers) read the project. Names match without regard to case; ids and unique id prefixes work
too. `p.remove_design(d)` and `p.remove_simulation(s)` delete (with results, immediately).

**Re-running a script is cheap.** Adding a design that already exists with the same name and
geometry returns the existing one. Running a simulation whose design and settings equal an
existing simulation that has a complete result returns the stored result (`r.reused` is True)
instead of running again; `rerun=True` runs that same simulation again. `print(r)` says so ("Stored
result of an identical run, reused"), and the Script console's end line counts the reused results
("Finished in 1.3 s · 1 stored result of an identical run reused, not run again"), so a result
computed earlier is never mistaken for a 1.3 s run. A stopped or failed
simulation with the same settings is run again in place. A script that is edited and run again
therefore only computes what changed, and the project does not fill up with duplicates.

## 3. Designs

```python
d = p.add_naca("2412", chord=0.3)                      # NACA 4/5 digit, built like Geometry ▸ NACA
d = p.add_airfoil("e387.dat", chord=0.2)                 # Selig / Lednicer file, or an (N, 2) array
d = p.add_template("sedan")                              # the template gallery (2D)
d = p.add_template("ahmed", slant_angle=30)              # named "Ahmed body (2D) · slant angle 30°"
d = p.import_geometry("side.svg", kind="vehicle", length=4.6)   # SVG, DXF, image, CSV outline
d = p.import_mesh("car.stl", units="mm", kind="vehicle")        # STL / OBJ / PLY, holes tolerated
d = p.add_sphere(0.2)                                    # also add_primitive("box"|"cylinder", ...)
w = p.add_wing(span=10, root_chord=1.2, tip_chord=0.6, sweep_deg=5, dihedral_deg=3,
               twist_deg=-2, airfoil="2412")             # finite wing, symmetric about y = 0
```

`import_mesh` takes `units` ("auto" guesses from the size, or "m", "cm", "mm", "in", "ft", or
metres per unit), `kind` ("vehicle": nose at x = 0, on the road; "generic": free flight),
`length` (scale to this length), `clearance` and `on_road`. References (frontal area, length,
width, mirror symmetry) are measured as the import dialog does. Every file import (`import_mesh`,
`import_geometry`, `add_airfoil` from a file) records its provenance in `design.raw.source`: the
file name, format, full path, size in bytes and the SHA-256 of the bytes read
(`al.file_source(path)` gives the same dictionary), so the Mesh inspector's Source group
names the file and a result can be traced to it after the file moves or changes. `add_wing` takes the
quarter-chord sweep (the aerodynamic one), the tip twist (negative = washout) and NACA codes or
2D airfoil designs of the project for `airfoil` / `tip_airfoil`; `sections=` takes any planform
as `aerolume.model.WingSection` stations.

A `Design` has `name`, `kind`, `dimension`, `id`, `references`, `analyses` (what it can run,
default first), `summary()`, `simulations()`, `results()`, `check()` (the Geometry page's
diagnostics), `plot(path=None)` (outline drawing, or iso/side/front/top views in 3D),
`export(path)` (.svg .dxf .dat in 2D; .stl .obj in 3D, a wing is lofted), `rename(name)`,
`copy(name=None)` and, for meshes, `transform(scale=, length=, rotate_deg=(rx, ry, rz),
clearance=)` (the design is then placed and its references re-derived as after an edit on the
Geometry page). `d.raw` is the `aerolume.model.Design`.

## 4. Running

| call | analysis | typical time |
|---|---|---|
| `p.run_cfd(d, speed=30, alpha=4)` | Navier–Stokes, 2D or 3D by the design | 10 s (2D airfoil, Draft, GPU) to minutes (3D car) |
| `p.run_polar(d, re=1e6, alpha=(-4, 14, 1))` | viscous–inviscid airfoil polar (XFOIL-class) | ~1.5 s per angle (+10 s the first time) |
| `p.run_wing(w, speed=30, alpha=(-4, 12, 1), method="vlm")` | vortex lattice / lifting line on viscous section polars | 1 s (25 s the first time per airfoil) |
| `p.run(name_or_id)` | a simulation already in the project | as above |

`run_cfd` keywords (every one optional):

| keyword | meaning |
|---|---|
| `speed` or `re` | free-stream speed [m/s] (default 30), or the Reynolds number on the design's reference length |
| `alpha`, `yaw` | angle of attack / pitch and sideslip [°] (yaw: 3D only; positive = wind from the left) |
| `quality` | "draft" (the app's start), "standard", "fine" |
| `ground`, `moving_ground`, `rotate_wheels` | road on/off (default: vehicles on a moving road), belt, spinning wheels |
| `fluid`, `altitude` | "air", "water" or an `aerolume.model.Fluid`; or ISA air at an altitude [m] |
| `device`, `precision` | "auto", "cuda", "cpu"; "auto", "float32", "float64" |
| `sensitivity` | "cd", "cl", "ld" or "cm": also compute the shape-sensitivity map |
| `symmetry` | 3D: "auto" (half domain for a symmetric body at zero yaw) or "full" |
| `reynolds`, `re_cap` | "real" or "capped" and the cap |
| `duration`, `average_from`, `cells_per_ref`, `turbulence` | run length and averaging start [L/U], grid, "auto" / "laminar" / "les" |
| any other `aerolume.model.CFDSettings` field | passed through (`domain_upstream=8`, ...) |

New simulations start from the same defaults as in the app: Draft quality, and for 2D airfoils
above Re 10⁵ the capped Reynolds mode (no affordable 2D grid resolves an airfoil boundary layer
at Re 10⁶, and an under-resolved run looks authoritative while being wrong). `r.re_sim` and
`r.re_real` say what was simulated; the polar gives the answer at the real Reynolds number.

`run_polar` takes `re` (default 10⁶) or `speed`, `alpha=(start, end, step)`, `ncrit` (default
9), `xtr` or `xtr_upper` / `xtr_lower` (forced transition, 1 = free), `mach`, `fluid`.
`run_wing` takes `speed`, `alpha`, `method` ("vlm" any planform, "llt" straight wings),
`viscous` (False: induced drag only, fast), `yaw`, `ncrit`, `mach`, `fluid`.

Every run also takes `name=` (default: the app's style, "CFD · 30 m/s · α 4°"), `rerun=`,
`progress=`, `cancel=` and `verbose=`.

## 5. Results

A `Result` holds the numbers people ask for first:

| | CFD (2D and 3D): one operating point | polar and wing: curves |
|---|---|---|
| `cd`, `cl`, `cm` | mean coefficients (3D also `cs`, `c_roll`, `cn`) | arrays over the converged angles |
| `ld` | Cl / Cd (printed and tabled only when `lifting`: airfoils and wings) | array |
| `alpha` | the angle | array |
| `stats[name]` | `Stat`: mean, stderr (± of the mean), rms, min, max | — |
| `stderr(name)` | standard error of the mean | — |
| `forces` | drag, lift, moment, drag area, power (2D per metre of span; 3D N, N·m) | — |
| `converged` | the runner's statistical verdict | every angle converged |
| `metrics` | — | Cl,max (and whether it was reached), best L/D, lift slope, Oswald factor, ... |
| `best()`, `at(alpha)` | — | the point of best L/D; a point interpolated at an angle |
| `re_sim`, `re_real` | simulated and real Reynolds numbers | the Reynolds number (MAC for wings) |

Also `name`, `kind` ("cfd2d", "cfd3d", "polar", "wing"), `status`, `cancelled`, `reused`,
`warnings` and `notes` (always read the warnings), `conditions`, `design`, `simulation`,
`wall_seconds`, `raw` (the model object), `summary()` (flat dict), `table()` (the Summary table,
or one row per angle) and `show()` (select it in the app). `print(r)` prints the table with
each mean rounded to its standard error as the Results page rounds it (the error to one or two
significant figures by the Particle Data Group rule, the mean to the same decimal place; "steady"
for a settled series, "≈0" for a mean inside its own error bar), in plain notation (`-`,
`1.2e-06`); `table()` keeps the unrounded numbers for your own arithmetic.

The tables follow the Results page. L/D appears only for lifting designs (`r.lifting` is true
for airfoils and wings): a car's or a sphere's L/D is arithmetic nobody optimises, and
`sweep.best()` of such runs picks the lowest drag instead. On a half-domain 3D run (a
mirror-symmetric body at zero yaw) the side force, rolling and yawing moments are zero by
construction; the table marks them `by_symmetry` and `print(r)` writes `0 (symmetry)`, as
the report does, rather than a measured-looking `0 ± 0`.

### Figures

`r.plot(kind=None, path=None, style="print", **options)` returns a matplotlib figure (no
pyplot, safe on any thread), or writes it when `path` is given (PNG, SVG or PDF).

| result | kinds (the first is the default) |
|---|---|
| 2D CFD | "history", "cp", "field" (`field="speed"`, "cp", "vorticity", "temperature", "dye", ...; `streamlines=True`), "geometry", "sensitivity", "diagnostics" |
| 3D CFD | "history", "surface" (`field="cp"`, "cf", "sensitivity"), "streamlines" (3D streamlines through the mean flow around the body), "smoke" (the instantaneous smoke cloud), "vortices" (Q-criterion iso-surface), "slice" (`index=`, `field=`), "geometry", "diagnostics" |
| polar | "polar" (Cl–α, drag polar, Cm, L/D), "cp" (`alpha=` one or several) |
| wing | "polar", "span" (span loading, `alphas=`), "planform" |

3D pictures are the 3D viewer's renders (VTK, off-screen) with the app's colour maps; they need
an OpenGL context. "streamlines" start on a rake upstream of the body and run through the
time-averaged flow; "smoke" is the dye the run released upstream, as it was at the end of the
run (a snapshot, so the filaments stay sharp), drawn as a volume around the grey body:
`opacity=` scales it, `color=` replaces the theme's grey, `threshold=` hides thin smoke.
Results of runs made before smoke was recorded say so and suggest `rerun=True`.
`al.show(fig)` displays a figure: inline in the Script dock, a PNG file on the command line and
in plain Python.

### Exports

`r.export(path, what=None)` chooses the format by the suffix:

| suffix | what is written |
|---|---|
| `.csv` | `what`: "summary" (default), "history", "surface" (2D), "polar" / "points", "span" (wings), "slice" (3D), "diagnostics" (CFD: divergence and change norms over time) |
| `.vtp` | 3D surface with Cp, Cf, wall shear, sensitivity (ParaView) |
| `.vti` / `.npz` | flow fields; 3D: the mean volume, or `what="smoke"` the smoke snapshot (mirrored to the full body) |
| `.json` | `summary()` |
| `.png` / `.svg` | `plot(what)` |
| `.pdf` / `.html` | a report of the result |

## 6. Sweeps

```python
sw = p.sweep(d, alpha=[0, 2, 4, 6], speed=30)                 # one simulation per point
sw = p.sweep(d, alpha=[0, 4, 8], speed=[20, 30])               # a full factorial: 6 points
pols = p.sweep(d, analysis="polar", re=[2e5, 5e5, 1e6], alpha=(-4, 14, 1))
cars = p.sweep(lambda slant: p.add_template("ahmed", slant_angle=slant),
               slant=[15, 25, 35], speed=40)                   # a geometry study
```

Lists, tuples, ranges and arrays are swept; scalars go to every run. For polars and wing
sweeps `alpha` is the curve's own range, never a sweep parameter. When the first argument is a
function, its own keyword arguments are swept too: name them explicitly (`lambda slant: ...`).

A `Sweep` has `params`, `values`, `results` (iterate it, index it), `table()` (one row per
point: the swept values and the key numbers), `best(key, minimize=None)` (lower is better for
drag-like keys), `plot(x, y, by=None, path=None)` and `export("sweep.csv")`. CFD points carry
±1 standard-error bars; unconverged points are drawn hollow. For a sweep of curves, `plot` can
overlay the curves themselves: `pols.plot("alpha", "cl")` (lift curves),
`pols.plot("cd", "cl")` (drag polars).

## 7. Heat, fans, jets and smoke

CFD runs take placed physics as keywords of `run_cfd` (and of `sweep`, `plan` and `grid_study`).
Each one is checked before anything is recorded: a wrong position, a vent with a temperature, a
fifth smoke colour or a heat source without heat transfer is an `al.InputError` that says why,
and the project is left as it was.

```python
d = p.add_template("ahmed")
hot = d.copy("Heated Ahmed").set_thermal("temperature", 60)      # bodies at 60 °C (or "heat_flux", W/m²)

r = p.run_cfd(hot, speed=2,
              thermal={"ambient": 20, "buoyancy": True},          # or thermal=True
              walls={"bottom": ("heat_flux", 50), "top": "adiabatic"},
              heat_sources=[al.HeatSource(kind="rect", position=(1.2, 0.1), size=(0.2, 0.1), power=200)],
              fans=[al.Fan(position=(-1.0, 0.4), direction=(1, 0), diameter=0.3, value=5)],      # 5 N/m
              ports=[al.Port(kind="jet", position=(0.0, 0.8), direction=(0, -1), size=0.05, speed=3,
                             temperature=40)],
              emitters=[{"kind": "rake", "position": (-1.5, 0.1), "end": (-1.5, 1.0), "count": 8}])
print(r.nusselt, r.heat_rate, r.fans, r.diagnostics)

room = p.add_case(al.fan_room_case())                           # an empty 4 m x 3 m room (its design)
r2 = p.run_cfd(room, domain="closed", domain_box=(0, 4, 0, 3), thermal=True,     # a radiator in the room
               heat_sources=[{"kind": "rect", "position": (0.5, 0.3), "size": (0.4, 0.2), "power": 300}])
```

| keyword | what it adds | values |
|---|---|---|
| `thermal` | the energy equation with Boussinesq buoyancy along −y (2D) / −z (3D) | `True`, a dict of `al.ThermalSettings` fields (`ambient` / `t_ambient` °C, `buoyancy`, `prandtl`, `beta`, `gravity`, `conductivity`, `cp`) or the object. Without an ambient temperature it is the fluid's own (15 °C for the default ISA sea-level air); with one and no `fluid=`, the run uses air at that temperature, so the density and viscosity match it |
| `walls` | conditions on the domain sides | `{"bottom": 40.0}` (°C), `("heat_flux", 50)` (W/m²), `"adiabatic"`; in a tunnel the left side is the inlet and the right the outlet, so only bottom and top (3D: front, back) take them |
| `heat_sources` | volumetric heaters | `al.HeatSource`: `kind` "rect" / "disc" (2D), "box" / "sphere" (3D), `position`, `size`, `power` (W in 3D, W/m in 2D; negative cools) |
| `fans` | actuator discs (momentum sources, not resolved blades) | `al.Fan`: `position`, `direction`, `diameter`, `mode` "thrust" (N, N/m in 2D), "pressure_jump" (Pa) or "velocity" (target axial speed, m/s), `swirl` (3D) |
| `ports` (or `jets`) | jets that blow in, vents that suck out | `al.Port`: `kind` "jet" / "vent", `position`, `direction`, `size`, `speed`, `temperature` (jets, needs `thermal`), `dye` |
| `emitters` | smoke (dye) or tracer particles, for pictures only | `al.Emitter` or dicts: `kind` "point", "line", "rake" (`end`, `count`), "rect" (`size`), 3D "disc" / "plane"; `species` "dye" / "particles", `color` (up to four smoke colours), `rate`, `pulse_period`, `start`, `stop` |
| `domain`, `domain_box` | "tunnel" (free stream from the left, default) or "closed" (walls on every side: cavities, rooms) and its extent in metres | `domain_box=(xmin, xmax, ymin, ymax[, zmin, zmax])` |
| `diagnostics`, `adaptive_dt` | the solver's divergence and change norms; the time-step controller | `True` / `False`; "cfl" or "pid" |

A simulation that is already in the project (set up on the Setup page, or with
`p.add_simulation`) takes placed physics one object at a time:

```python
sim = p.add_simulation(room, domain="closed", domain_box=(0, 4, 0, 3), speed=1.0)
sim = p.add_emitter(sim, kind="rake", position=(0.4, 0.4), end=(0.4, 2.6), count=8)   # fields as keywords
sim = p.add_fan(sim, al.Fan(position=(1.3, 1.5), direction=(1, 0), diameter=0.5, value=1.0))
sim = p.edit_simulation(sim, thermal=True)                       # a heater needs heat transfer first
sim = p.add_heat_source(sim, {"kind": "rect", "position": (2, 0.3), "size": (0.4, 0.2), "power": 300})
sim = p.add_port(sim, kind="jet", position=(3.5, 2.5), direction=(0, -1), size=0.1, speed=2)
r = p.run(sim)
```

Each call runs the `run_cfd` keyword's checks on the simulation's existing objects plus the new
one (the four-colour smoke limit counts the emitters already there; a fan outside the domain box
is refused) and records nothing when one fails. Like `edit_simulation`, a simulation that has
a result is copied rather than edited (pass `copy=False` to insist, which is refused), and an
automatically named simulation is renamed after its new physics.

Bodies are heated on the design (`d.set_thermal(mode, value, body=None)`, recorded in the
project; `d.copy()` first keeps an unheated original). A design with a heated body refuses a run
without `thermal=`, since the heating would silently do nothing. A closed domain without `speed`
takes the speed that drives it (the free-fall velocity √(gβΔT H) the planner states, a fan's or a
jet's speed) as its reference, so its coefficients and Reynolds number mean something. ΔT is the
largest set temperature difference or, for heat given as a power or a flux (a heater, a heat-flux
body or wall), the rise it produces, q″L/(k·Nu): a closed room heated only by a 300 W/m heater
runs on buoyancy alone (ΔT ≈ 58 K estimated at the heater, Ra 1.8×10¹¹ in a 4 m × 3 m room). Runs are named after
their physics ("CFD · 2 m/s · α 0° · heated · 1 fan"), so two runs that differ only in a fan do
not share a name.

**What this build can run.** `al.supported_physics(2)` and `al.supported_physics(3)` list the
features the planner and solver implement ("emitters", "thermal", "heat_sources", "fans",
"ports", "closed_domain", "adaptive_dt"). A run that asks for one that is missing is refused,
naming it ("This version's 2D solver does not implement heat transfer yet: ... nothing was
run"), rather than run without it; `p.plan()` says the same.

Results of these runs add `r.nusselt` (mean Nusselt number Nu = Q L / (k ΔT A) of the heated
surfaces) with `r.nusselt_stat` (its standard error), `r.heat_rate` (W, or W/m in 2D),
`r.fans` (per fan: delivered thrust with its standard error, volume flow, mean induced velocity, ideal
power; `print(r)` lists them with their units and their standard errors: the thrust's own, the volume
flow's and the induced velocity's from the recorded `fan<i>_flow` / `fan<i>_velocity` histories),
`r.physics` (the features used), `r.diagnostics` and `r.diagnostics_history()`;
`r.plot("diagnostics")` draws the solver's own health over time and
`r.export("diag.csv", what="diagnostics")` writes it. A 2D field plot takes
`field="temperature"` (°C), `"theta"` (θ = (T − T_amb)/ΔT_ref) and `"dye"`, `"dye1"`... for smoke.

The models and their limits are in docs/theory_ns2d.md and docs/theory_ns3d.md, and what is
validated (and how closely) in docs/validation.md. In short: heat transfer is a Boussinesq
model with constant properties; fans are actuator discs, so their delivered thrust and volume
flow are meaningful but blade loading and noise are not; jets are penalised velocity regions,
not resolved nozzles; smoke and tracers never change the flow.

## 8. Reference cases and grid studies

A **case** is a ready set-up with the published number it should reproduce:

```python
case = al.cavity_case(ra=1e4)              # differentially heated square cavity, air, de Vahl Davis (1983)
print(case, case.reference, *case.notes)   # ΔT that gives Ra exactly, the free-fall velocity
r = p.run_case(case, quality="standard")   # or p.run_cfd(case, ...); keywords override the case
print(case.check(r))                       # measured ± standard error, reference, difference in %
```

| case | set-up | reference |
|---|---|---|
| `al.cavity_case(ra=1e4, pr=0.71, size=0.1, dimension=2)` | closed square (cube) of air, hot left and cold right wall, the others adiabatic; ΔT chosen so Ra = g β ΔT L³ / (ν κ) is exactly `ra` | mean Nu of the hot wall, the quantity the run measures: de Vahl Davis' (1983) Nu₀ for Ra 10³–10⁶ (1.117, 2.238, 4.509, 8.817; the paper's cavity-average Nū 1.118, 2.243, 4.519, 8.800 is a different quantity); 3D: Tric et al. (2000), Ra 10⁴–10⁶ (2.0542, 4.3370, 8.6407) |
| `al.heated_cylinder_case(re=40, d=0.01, delta_t=10)` | a cylinder `delta_t` above the stream, buoyancy off (forced convection) | Lange, Durst & Breuer (1998) correlation at Pr 0.7, valid for 10⁻⁴ ≤ Re ≤ 200 (Nu 3.28 at Re 40) |
| `al.fan_room_case(thrust=1.0, room=(4, 3), diameter=0.5)` | one thrust fan in a closed room | a **consistency** check, not a validation (`case.kind == "consistency"`): thrust mode imposes the set thrust, so agreement shows the disc delivers it. Froude momentum theory does not apply in a closed room (the fan recirculates its own wake, and the induced velocity reads several times below the free-air value); the fan model is validated in a channel against confined momentum theory (docs/validation.md) |

A case has `name`, `design` (not yet in a project; `p.add_case(case)` adds it), `settings` (the
`run_cfd` keywords), `reference` (None off the published parameters, for instance Pr ≠ 0.71),
`kind` ("validation" for an independent published value, "consistency" when the set-up imposes
it) and `check(result)`. `check` returns the measured value, its standard error, the reference,
the difference in %, `kind`, `within_stderr` (the reference lies within two standard
errors) and, where docs/validation.md states one, `band_pct`, `band_source` and `within_band`
(the difference lies inside the tolerance band that reference is checked with: ± 2 % for the
cavity at Ra 10⁴, ± 4 % for the heated cylinder at Re 20-40). That is the statistical part only: the grid error needs a grid study.

A refused case run records nothing: the run is built and checked (keywords, physics this
build's solver implements) before the case's design is added, so a refusal leaves no empty
"Heated cavity" design behind. `p.add_simulation(case)` prepares the run without that physics
check, for a build that will run it later.

A stored result finds its reference again, in a later session, in the app or for the
assistant: `r.reference_check()` matches the run's design and settings to the case it
reproduces (`al.reference_case(design, simulation)`) and returns `case.check(r)` with
`"applies": True`. When the design is a reference set-up whose settings left the published
conditions (buoyancy switched on for the forced-convection cylinder, the cavity's walls changed,
Re outside the correlation's 10⁻⁴–200) it returns `{"applies": False, "why": ...}` rather than
a comparison under conditions the reference was not published for; a design that is no
reference set-up gives None. A heated 2D cylinder template run in forced convection is matched
to the Lange correlation too, with a note when its Pr is 0.71 rather than the correlation's 0.7.

A **grid study** runs the same case on three systematically refined grids and estimates the
discretisation error by the procedure of Celik et al. (2008), as the Results page and the
report do (`aerolume.analysis.gci`, one implementation):

```python
gs = p.grid_study(case, quality="draft", quantity="nusselt")    # grids 1, 1.5 and 2.25 x the planner's
print(gs)                       # the table, then per quantity: fine value ± GCI band, order, extrapolation
gs.gci["nusselt"].p             # observed order; .phi_ext, .gci_fine, .u_num, .noise_limited, .asymptotic
gs.plot("nusselt", "gci.png"); gs.export("grid_study.csv")

# your own three grids: Celik et al. (2008) Table 1 gives p 1.53, φ_ext 6.1685, GCI 2.2 %
g = al.grid_convergence(h=[18000 ** -0.5, 8000 ** -0.5, 4500 ** -0.5], phi=[6.063, 5.972, 5.863])
print(g.summary())
```

`p.grid_study_of(simulation)` studies a CFD simulation that is already in the project (one set up
on the Setup page, a case's, a run with fans or heating) exactly as it is set up: the simulation
itself is the coarsest grid (its stored result is reused) and two copies that differ only in
`cells_per_ref` are the finer ones ("… · 18 cells/ref").

`grid_study` takes `cells_per_ref=[...]` (three resolutions) or `refinement=` (default 1.5;
Celik et al. recommend at least 1.3), `quantity` ("cd", "cl", "cm", "nusselt" or a summary key;
default Nu for heat-transfer runs without forces, else Cd and, for lifting designs, Cl) and the
`run_cfd` keywords. Each grid is an ordinary recorded simulation, so a re-run reuses them.
Because CFD values are time averages, every grid's standard error goes into the analysis: when
a grid-to-grid change is not larger than twice the combined standard errors, the study is
*noise-limited* and no order or GCI is claimed (average longer or refine further). The
asymptotic verdict compares the observed order with the scheme's formal order (2) within 10 %.
Cost grows with the cell count, about refinement³ per level in 2D and refinement⁴ in 3D.

## 9. Comparisons, reports, plans

```python
cmp = p.compare([r1, r2, r3], baseline=0)      # or p.compare(sweep)
print(cmp)                                     # the Compare page's table, Δ % and significance
cmp.table(); cmp.warnings; cmp.plot(["cd", "cl"], "bars.png")
p.report([r1, r2], "study.pdf", title="Rear slant study")     # or .html; default: reports folder
pl = p.plan(d, speed=30, quality="standard")   # nothing is run or recorded
print(pl.cells, pl.grid, pl.estimate, pl.re_sim, pl.warnings)
```

A difference is marked *significant* when it exceeds twice the combined standard error of the
two means; the comparison's warnings say when runs differ in speed, angle, fluid or simulated
Reynolds number, or mix 2D with 3D (which are never compared in one table). `al.plan(design,
...)` plans a design that is not in a project.

## 10. Progress, cancel, verbose, errors

Every call is synchronous and prints nothing unless `verbose=True` (or `al.set_verbose()`).
Calls that run solvers take `progress(fraction, message)` and `cancel` (a
`threading.Event`): a cancelled run keeps its partial result in the project, marked
"cancelled", and the call raises `al.Cancelled` (`exc.result` is the partial result).
`al.context(cancel=, progress=, verbose=)` sets defaults for every call made on the current
thread inside a `with` block; the Script dock and the command line use it, so Stop and Ctrl+C
reach every run of a script.

Errors are `al.AerolumeError` subclasses with messages written for people:

| class | when | also a |
|---|---|---|
| `al.InputError` | a bad argument (an unknown NACA code, a negative speed, an analysis the design cannot have; unknown keywords get a "Did you mean" hint) | `ValueError` |
| `al.NotFoundError` | an unknown design, simulation, template or file (the message lists what exists) | `LookupError` |
| `al.RunError` | a run failed (planner refused the set-up, solver diverged, GPU out of memory) | `RuntimeError` |
| `al.ProjectFileError` | the project folder could not be read or written | `OSError` |
| `al.Cancelled` | stopped | — |

`al.plain(obj)` turns `summary()` / `table()` output into strict JSON data (NaN becomes None).

## 11. The Script dock

**View ▸ Script** (Alt+4) opens an editor (Python highlighting, line numbers, auto-indent)
beside an output pane; **Tools ▸ Run script…** opens a `.py` file in it and runs it. Names
bound in a script: `project` (the open project, or None), `al`, `show(obj)` (figures appear
inline in the output; anything else is printed) and `log(text, level="info")` (a line in the
output and in the Messages dock). Inline figures are as wide as the pane allows and stay
readable in a short dock (it scrolls instead of shrinking them); click one to open it at full
size, where **Save image…** writes the PNG (to the project's `exports` folder by default) and
**Copy** puts it on the clipboard. The output follows the newest text until a run shows its
first figure; it then stays at that figure's top while the run prints and plots more below it
(scroll to the end to follow again; a failed run jumps to its traceback). While a script runs,
the line beside the status reports the simulation it started (*Simulation: Averaging forces…*)
and clears when the script ends. The dock opens at about 40 % of the window height and keeps
that share (or the height you drag it to) when the window is resized.

| key (console focused) | action |
|---|---|
| Ctrl+Enter, F5 | run the script |
| Ctrl+Shift+Enter | run the selection, or the current line |
| Shift+F5 | stop |
| Ctrl+S | save the script in the project |
| Ctrl+/ | comment / uncomment lines |
| Tab, Shift+Tab | indent, dedent |
| Ctrl+Tab, Ctrl+Shift+Tab | leave the editor: focus moves to the next / previous control |
| Esc, then Tab (or Shift+Tab) | leave the editor the same way (one Tab after Esc) |

Because Tab indents, the editor has explicit ways out for keyboard-only use (WCAG 2.1.2, no
keyboard trap): Ctrl+Tab always moves focus on to the Run button and the rest of the window,
and so does a Tab pressed right after Esc. **⋯ ▸ Tab key moves focus** makes plain Tab and
Shift+Tab move focus for as long as it is ticked (Tab then no longer indents; untick it to indent
with Tab again). The editor's tooltip states the same.

Scripts are saved in the project's `scripts` folder and listed in the script box; a script with
a file is saved before each run. **Examples** inserts ready scripts (section 13). Variables are
kept between runs, as in an interactive console, until **⋯ ▸ Reset variables**.

Scripts run on a worker thread, so the window stays responsive. Their simulations go through
the same queue as the app's own runs: a script waits behind a running simulation instead of
competing for the GPU, and the Simulate page shows its live view. **Stop** first stops the
running simulation (it keeps its partial result) and ends the script; plain Python that never
calls Aerolume (a long loop) is interrupted a moment later. An error marks the failing line in
the editor; the traceback shows the script's own lines (click "line N" to jump there) and
counts the frames inside Aerolume instead of listing them.

## 12. Command line

```
aerolume-cli script FILE [--project P [--create]] [--figures DIR] [-v] [-- ARGS...]
```

`project` is bound to `--project` (`--create` makes it when missing), `args` (and
`sys.argv[1:]`) to what follows `--`, `show()` saves `<script>-figure-<n>.png` in `--figures`
(default: the current folder), `log()` prints. The first Ctrl+C stops the running simulation,
which keeps its partial result, and ends the script; a second one aborts. Exit code 0 on
success, 1 for an error or a stop (the script's lines of the traceback are printed), 2 when the
file is missing; `sys.exit(n)` returns `n`.

```
aerolume-cli script examples/scripts/wing_taper.py --project "Wing study" --create
```

## 13. Examples

`examples/scripts/` holds the scripts the Examples menu inserts. Each runs in the app, with
`aerolume-cli script` and on its own (then in a temporary project):

| file | what it does | time (RTX 4060 Ti, measured) |
|---|---|---|
| `polar_sweep.py` | NACA 2412 polars at Re 2×10⁵, 5×10⁵, 10⁶; lift curves and drag polars overlaid | ~65 s |
| `alpha_sweep.py` | 2D CFD of a NACA 0012 at four angles, lift and drag with error bars | ~50 s |
| `compare_designs.py` | sedan against fastback, 2D CFD on a moving road, Compare table and bars | ~3.5 min |
| `mesh_3d.py` | import a mesh (without one, a sphere at Re 100: CD 1.16 on the Draft grid, 1.09 measured), plan, Draft 3D run, surface Cp, 3D streamlines, smoke | ~35 s (GPU to itself) |
| `wing_taper.py` | equal-area wings with taper 0.3–1.0: span efficiency and lift slope | ~5 s |
| `flow_pictures.py` | streamlines, smoke and vortices of the newest 3D result, surface .vtp for ParaView | ~5 s |
| `report.py` | a PDF report of the newest results | ~20 s |
| `heated_cavity.py` | the de Vahl Davis cavity at Ra 10⁴: plan, Draft run, hot-wall mean Nu against 2.238, temperature field | needs heat transfer and closed domains in the 2D solver |
| `heated_cylinder.py` | a cylinder 10 K above the stream at Re 40: Nu against the Lange et al. correlation (3.28); `GRID_STUDY = True` adds two grids and the GCI | needs heat transfer in the 2D solver |
| `fan_room.py` | a 0.5 m fan in a closed 4 m × 3 m room with a smoke rake: delivered thrust against the set 1 N/m (a consistency check: the thrust is imposed), the circulation | needs fans, emitters and closed domains in the 2D solver |
| `smoke_3d.py` | a sphere at Re 300 (its wake sheds) with two coloured smoke rakes: smoke, streamlines, vortex cores, the smoke volume as .vti. Draft read CD 0.751 ± 0.009 (not converged at 40 L/U) against 0.656 (Johnson & Patel 1999): a picture grid | 116 s (GPU shared with other work) |

The physics examples check nothing in advance: when this build's solver does not implement
what they need (`al.supported_physics(2)`), the run is refused with that reason and nothing is
recorded. `smoke_3d.py` falls back to the solver's default smoke rake when placed emitters are
not available in 3D, and says so.

Times are for a first run; running a script again takes seconds, since every unchanged
simulation returns its stored result (`compare_designs.py`: 5.5 s the second time). A GPU shared
with another program can take twice as long (the 3D example took 6.5 min on a busy GPU).

## 14. For integrators: hosts

A `Project` makes every change through a small `al.Host` object. The default host edits the
project store on the calling thread and runs simulations there. Inside the app,
`aerolume.app.scripting.AppHost(state, jobs)` performs each change on the GUI thread through
`AppState` and runs simulations through the job queue; the Script dock and the AI assistant
bind it with `al.Project(state.project, host=AppHost(state, jobs))`.
`aerolume.app.scripting.ScriptEngine(state, jobs)` runs source code on a worker thread with
the console's names bound, captures its output and figures and reports a `ScriptOutcome`
(status, error, script line, traceback, output, figures, simulations touched); the assistant's
`run_python` tool uses it. A custom host (a remote runner, a test double) overrides
`add_design`, `update_design`, `add_simulation`, `update_simulation`, `result_saved`,
`run_simulation` (return `(result, "done" | "cancelled")`, or None to let the API run it) and
friends; see the `Host` docstring.

## 15. Reference

```python
# module (aerolume, aerolume.api)
open_project(path, *, create=False) -> Project
new_project(path=None, name=None, *, exist_ok=False) -> Project
plan(design, analysis=None, **settings) -> Plan
context(*, cancel=None, progress=None, verbose=None, show=None, project=None)   # with-block
set_verbose(on=True); show(obj); plain(obj); version()
AerolumeError, InputError, NotFoundError, RunError, ProjectFileError, Cancelled(result)
Host; script_traceback(exc, filename) -> (line, lineno, text)

supported_physics(dimension=None) -> frozenset; physics_features(model_simulation) -> list
cavity_case(ra=1e4, pr=0.71, *, size=0.1, dimension=2, t_ambient=15) -> Case
heated_cylinder_case(re=40, *, d=0.01, delta_t=10, pr=0.7, t_ambient=15) -> Case
fan_room_case(thrust=1.0, *, room=None, diameter=0.5, dimension=2, position=None) -> Case
reference_case(design, simulation, *, re=None) -> (Case | None, why_not)
file_source(path, fmt=None) -> dict    # {type, file, path, format, bytes, sha256}
grid_convergence(h, phi, *, stderr=None, quantity="", formal_order=2, method="celik", safety=None)
    -> analysis.gci.GCIResult
Emitter, Fan, Port, HeatSource, ThermalSettings, WallThermal        # aerolume.model types

# Project(store_or_path, *, host=None)
path, name, raw, host, designs, simulations
design(ref) -> Design; simulation(ref); result(ref) -> Result | None; results(design=None); summary()
add_naca(code, chord=1.0, *, name=None, sharp_te=True, reuse=True) -> Design
add_airfoil(path_or_coords, chord=1.0, *, name=None) -> Design
add_template(template, *, name=None, **params) -> Design
import_geometry(path, *, kind=None, length=None, name=None) -> Design
import_mesh(path, *, units="auto", kind="vehicle", on_road=None, length=None, clearance=None, name=None) -> Design
add_primitive(shape, *, name=None, **size); add_sphere(d=0.2) -> Design
add_wing(span, root_chord, tip_chord=None, *, sweep_deg=0, dihedral_deg=0, twist_deg=0, airfoil="2412",
         tip_airfoil=None, sections=None, name=None) -> Design
add_design(model_design) -> Design; remove_design(ref); remove_simulation(ref)
add_simulation(design, analysis=None, *, name=None, **settings) -> model.Simulation   # not run
plan(design, analysis=None, **settings) -> Plan
run_cfd(design, *, speed=None, alpha=0, yaw=0, quality=None, ground=None, name=None, rerun=False,
        progress=None, cancel=None, verbose=None, **settings) -> Result
run_polar(design, *, re=None, alpha=(-4, 14, 1), ...) -> Result
run_wing(design, *, speed=None, alpha=(-4, 12, 1), method="vlm", viscous=True, yaw=0, ...) -> Result
run(simulation, *, rerun=False, ...) -> Result
add_case(case) -> Design; run_case(case, *, rerun=False, ..., **overrides) -> Result
grid_study(design_or_case, *, cells_per_ref=None, refinement=1.5, levels=3, quantity=None, ...,
           **settings) -> GridStudy
grid_study_of(simulation, *, cells_per_ref=None, refinement=1.5, quantity=None, rerun=False, ...) -> GridStudy
edit_simulation(simulation, **settings) -> model.Simulation      # checked; a run with a result is copied
add_emitter(simulation, emitter=None, *, copy=None, **fields) -> model.Simulation
add_fan(...); add_heat_source(...); add_port(...)    # same signature; checked like run_cfd's keywords
sweep(design_or_factory, *, analysis=None, name=None, rerun=False, ..., **params) -> Sweep
compare(results_or_sweep, *, baseline=0) -> Comparison
report(results, path=None, *, sections=None, title=None, author=None, fmt=None) -> Path
save()

# Design: id, name, kind, dimension, is_3d, analyses, references, raw
summary(); simulations(); results(); check(); plot(path=None, *, style, alpha); export(path, *, units)
rename(name); copy(name=None); transform(*, scale, length, rotate_deg, clearance)
set_thermal(mode, value=0, *, body=None)      # "temperature" °C | "heat_flux" W/m² | "adiabatic"

# Result: sim_id, name, kind, analysis, dimension, design, simulation, status, cancelled, reused,
#   cd, cl, cm, cs, c_roll, cn, ld, lifting, alpha, stats, forces, units, re_sim, re_real, converged,
#   n_converged, metrics, warnings, notes, conditions, wall_seconds, raw
#   physics, nusselt, nusselt_stat, heat_rate, fans, diagnostics
stderr(key); best(); at(alpha); summary(); table(); plot_kinds(); plot(kind=None, path=None, *, style, **o)
diagnostics_history()
export(path, what=None); report(path, **o); show()

# Sweep: params, values, results, analysis; len / iter / [i]
table(); best(key=None, *, minimize=None); plot(x=None, y=None, *, by=None, path=None, style, errorbars,
title); export(path)

# Comparison: results, baseline, rows, warnings, names
table(); plot(metric="cd", path=None, *, style)

# Plan: analysis, est_seconds, estimate, cells, grid, quality, device, re_real, re_sim, notes, warnings, raw
summary()

# Case: name, design, settings, reference, notes, kind ("validation" | "consistency"); check(result) -> dict
# Result.reference_check() -> dict | None   (the stored result against the reference its set-up reproduces)
# GridStudy: results, cells_per_ref, h, quantities, gci {quantity: GCIResult}, study, notes
table(); summary(); plot(quantity=None, path=None, *, style); export(path.csv)
```
