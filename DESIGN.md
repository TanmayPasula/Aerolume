# Aerolume — design and contracts

Aerolume is an open-source (Apache-2.0) desktop studio for 2D and 3D aerodynamics (3D: section 7).
It combines an AeroJAX-style **real-time, differentiable Navier–Stokes solver**
(PyTorch, GPU or CPU) with an AirShaper-style **workflow**: import a shape, set
speed and attitude, press run, get coefficients, pressure maps, sensitivity
maps, comparisons and a report. Its first users are airfoil designers and
vehicle (car / bike / truck side-profile) designers. The first public release, **0.1.0**, holds
all of it: the 2D core (sections 1–6), 3D, finite wings, scripting and the AI assistant
(sections 7–9), and placed physics — emitters, heat transfer, fans, jets, closed rooms, live
editing and diagnostics — optional accounts, the Windows installer and the UI v3 shell
(sections 10–13; 10.5 records the contracts as integrated). The labels v0.1, v0.2 and v0.3 in
this file and in code comments name internal development stages before that release (v0.1 the
2D core, v0.2 the 3D / wings / scripting / assistant stage, v0.3 the placed-physics stage), not
published versions. Planned for 0.2: local grid refinement with a sharper immersed wall, a wall
model with transition, and finite-width 2D vehicles (README, *Limits*).

This file is the contract every module is written against. If you change a
public signature described here, update this file in the same change.

---

## 1. Stack

| Concern            | Choice                                                     |
|--------------------|------------------------------------------------------------|
| Language           | Python 3.11+                                               |
| Compute            | PyTorch ≥ 2.4 (CUDA when available, else CPU; MPS best-effort) |
| GUI                | PySide6 (LGPL — keeps the app Apache-2.0), pyqtgraph for plots |
| Static figures     | matplotlib (Agg backend only — never pyplot GUI backends)  |
| Airfoil VII solver | vendored mfoil (MIT, K. Fidkowski) in `third_party/mfoil`  |
| Geometry import    | own SVG parser, ezdxf (DXF), Pillow + contourpy (images)    |
| Tests              | pytest; GPU tests auto-skip without CUDA                   |

Dev interpreter: a virtual environment outside any synced folder (docs/install.md), for example
`%USERPROFILE%\venvs\aerolume\Scripts\python.exe`; the reference machine runs torch
2.14.1+cu130 on an RTX 4060 Ti. Install: `pip install -e .[dev]`.

## 2. Package layout

```
src/aerolume/
  __init__.py            version
  model.py               SHARED DATA MODEL (dataclasses, SI units)  <- contract
  core/units.py          unit parsing / formatting
  core/fluid.py          air (ISA / temperature), water, custom fluids
  core/device.py         device + dtype selection, hardware description
  geometry/              shape creation, import, polygon ops, SDF, parametrisation
  solvers/ns2d/          incompressible Navier–Stokes (PyTorch), immersed boundary
  solvers/airfoil/       viscous–inviscid airfoil analysis (mfoil wrapper)
  simulation/            plan (auto-setup), runner (production runs), live session
  analysis/              statistics, Strouhal, convergence, coefficient post-processing
  optimize/              autograd sensitivity maps, shape optimisation
  viz/                   colormaps, field rendering (torch), streamlines, particles
  project/               project persistence (folder format)
  export/                exports: CSV, JSON, NPZ, VTK
  report/                PDF + HTML report generation
  app/                   PySide6 desktop application
  cli.py                 headless command line
tests/                   unit + verification tests (fast; CPU by default; `-m slow` for GPU validation)
validation/              benchmark scripts -> docs/theory_ns2d.md, docs/validation*.md
examples/                example geometries; examples/make_example_projects.py builds the
                         example projects shipped in src/aerolume/examples/projects/
website/                 static project site (website/build.py, standard library only)
third_party/mfoil/       vendored mfoil + its MIT licence
docs/                    user guide, theory, validation
```

Rules:
* `model.py`, `core/*` and `analysis/*` import only numpy / stdlib.
* torch is imported only in `solvers/ns2d`, `geometry/sdf.py`, `optimize`, `viz/render.py`,
  `simulation/*`, `core/device.py` (lazily).
* Qt is imported only under `app/`.
* matplotlib only in `report/` and `viz/figures.py`, always `matplotlib.use("Agg")`-safe
  (use `matplotlib.figure.Figure` directly, never `pyplot`).
* All public functions have type hints and a one-paragraph docstring.

## 3. Coordinates, units, conventions

* Model data is **SI**: metres, m/s, Pa, N, kg/m³, Pa·s. Forces in 2D are **per unit span** (N/m).
* Design frame: x to the right (downstream), y up. Airfoils: leading edge at (0,0),
  trailing edge at (chord, 0). Vehicles: road at y = 0, vehicle above it, nose at the left.
* Flow always comes from the left along +x in the solver. Angle of attack / pitch `alpha_deg`
  is applied by **rotating the geometry** about `Design.moment_ref` by **−alpha** (nose up
  for positive alpha). Hence solver Fx is drag and Fy is lift directly (wind axes).
* Polygons: `np.ndarray` shape (N, 2), float64, **closed implicitly** (last point ≠ first),
  **counter-clockwise** orientation for solid outlines. Multiple solids = multiple `Body`s.
* Coefficients: `C = F / (0.5 ρ U² · Design.force_ref)`; moments use `force_ref * length_ref`.
  Airfoils: `force_ref = length_ref = chord`. Vehicles: `length_ref` = overall length (for Re),
  `force_ref` = frontal height (2D analogue of frontal area); "drag area" `CdA = F/q` [m²/m].
* Reynolds number: `Re = ρ U length_ref / μ`.
* Pitching moment positive nose-up, about `Design.moment_ref` (quarter chord for airfoils).
* Solver internals are non-dimensional: length by `length_ref`, velocity by `U`,
  time by `length_ref/U`, pressure by `ρU²`.

## 4. Shared data model (`aerolume.model`)

See the module for the exact fields. Summary:

* `Fluid(name, density, viscosity)`; the defaults are ISA sea-level air from
  `model.isa_sea_level_air()`, identical to `core.fluid.air_isa(0.0)`, so GUI-made and
  script-made simulations store the same numbers
* `Body(name, points, role="body"|"wheel"|"element", omega=0.0 rad/s, center=None)`
* `Design(id, name, kind="airfoil"|"vehicle"|"generic", bodies, length_ref, force_ref,
   moment_ref, source)`
* `FlowConditions(fluid, speed, alpha_deg)`
* `GroundConfig(enabled, moving, rotate_wheels)`
* `CFDSettings(quality, cells_per_ref, domain_*, duration, turbulence, reynolds_mode,
   precision, device)`
* `AirfoilSettings(alpha_start, alpha_end, alpha_step, ncrit, xtr_upper, xtr_lower, mach)`
* `Simulation(id, name, design_id, analysis="cfd"|"polar", conditions, ground, cfd, airfoil,
   status, created, notes)`
* Results: `Stat(mean, rms, min, max, stderr, stderr_reliable)`, `SurfaceData`, `FieldData`,
  `CFDResult` (… `strouhal`, `strouhal_length` [m] …), `PolarPoint`, `PolarResult`.
  `Stat.stderr_reliable` is False when the averaging window was too short for batch means:
  `stderr` is then a conservative estimate, never used for a significance test.
* `from_dict` raises `TypeError` for a non-object (a list or string where a dict belongs), so the
  project store can skip a damaged file with a message.
* 3D (section 7): `Design.dimension` (2 | 3) with `meshes: list[Mesh3D]`, `wing: WingSpec`
  (`WingSection` stations), `ref_area`, `span_ref`, `moment_ref3`, `symmetric`;
  `FlowConditions.yaw_deg` (7.2 for its sign); `CFDSettings.symmetry` / `domain_width`;
  `Simulation.analysis == "wing"` with `WingSettings`; results `CFD3DResult` (`SurfaceData3D`,
  `SliceData`, `VolumeData`; `volume` is the mean block, `extra_volumes` further blocks by
  `label` such as the "instant" smoke snapshot of 7.13, `volumes()` / `find_volume(field,
  label=None)` pick one) and `WingResult` (`WingPoint`, `stall_alpha` / `stall_cl`).
  `KINDS_2D` are the kinds a 2D design may have ("wing" is 3D only). `SCHEMA_VERSION` (2) and
  `SCHEMA_VERSION_2D` (1) drive the project schema policy (7.10). numpy scalars, `np.bool_`
  included, serialise as plain JSON numbers / booleans.

Every dataclass has `to_dict()` / `from_dict()` (JSON-safe; numpy arrays become lists only in
small objects — large arrays are stored by `project/` in `.npz`).

## 5. Module contracts

### 5.1 geometry
```python
naca4(code: str, n: int = 160, chord: float = 1.0, closed_te: bool = True) -> np.ndarray
naca5(code: str, ...) -> np.ndarray
naca(code: str, ...) -> np.ndarray                      # dispatch 4/5 digit
read_airfoil_dat(path_or_text) -> tuple[str, np.ndarray] # Selig + Lednicer formats
read_svg(path, *, tolerance=...) -> list[np.ndarray]     # every closed/open path -> polygon
read_dxf(path) -> list[np.ndarray]
read_image_silhouette(path, *, threshold=None, invert=False, simplify=...) -> list[np.ndarray]
read_points_csv(path) -> list[np.ndarray]
templates.list_templates() -> list[TemplateInfo]; templates.make(name, **params) -> Design
design_from_airfoil(coords, name, chord) -> Design
design_from_polygons(polys, name, kind, *, length=None, ground_clearance=None) -> Design
polygon ops: orient_ccw, signed_area, centroid, bbox, resample(poly, n|spacing),
  smooth, close_gaps, remove_duplicates, is_self_intersecting, frontal_height,
  transform(poly, scale, rotate_deg, about, translate), point_in_polygon
sdf.signed_distance(polys: list[Tensor|ndarray], x: Tensor, y: Tensor) -> Tensor  # torch, <0 inside
  - differentiable w.r.t. polygon vertices; chunked so memory stays bounded
params: cst_fit(coords, order) -> CSTParams; cst_eval(params, n) -> coords;
  normal_basis(poly, n_modes, width) -> Tensor  (smooth normal-displacement modes)
import_file(path, **options) -> ImportResult      # dispatch on SUPPORTED_EXTENSIONS (GUI import)
check_design(design, cells_per_ref=None) -> list[str]   # human-readable diagnostics (thin
  trailing edges, gaps under 2 cells, self-intersections); the planner adds the grid-dependent ones
also: naca4/naca5, normalize_airfoil, format_airfoil_dat / write_airfoil_dat, perimeter,
  simplify_rdp, repair_self_intersections, volume_fraction, vertex_normals, apply_normal_displacement
```
The readers take an optional ``notes: list[str]`` that collects what was repaired or assumed
(closed gaps, chained open paths, units), which the import dialog shows.

### 5.2 solvers.ns2d (non-dimensional)
```python
@dataclass
class ImmersedBody:  # nondim polygon, CCW
    points: np.ndarray; name: str = ""; role: str = "body"
    omega: float = 0.0; center: tuple[float, float] = (0, 0)   # rigid rotation (wheels)

@dataclass
class SolverConfig:
    nx: int; ny: int; h: float; x0: float; y0: float          # domain [x0, x0+nx h] x [y0, y0+ny h]
    re: float                                                  # U L / nu
    bodies: list[ImmersedBody]
    inlet: str = "uniform"           # "uniform" | "parabolic" (channel benchmarks)
    bc_top: str = "slip"             # "slip" | "noslip"
    bc_bottom: str = "slip"          # "slip" | "noslip" | "moving" (u = ground_speed)
    ground_speed: float = 1.0
    periodic: bool = False           # verification mode: periodic in x and y, no inlet/outlet
    advection: str = "upwind3"       # "upwind3" | "central"
    turbulence: str = "none"         # "none" | "smagorinsky" | "wale" | "vreman"
    cfl: float = 0.5
    dt: float | None = None          # fixed dt, else from cfl + viscous limit
    force_ref: float = 1.0           # nondim normalisation length for coefficients
    moment_ref: tuple[float, float] = (0.25, 0.0)
    dtype: str = "float32"; device: str = "auto"
    use_cuda_graphs: bool = True
    dye: bool = True                 # passive smoke scalar for visualisation
    perturb: bool = True             # tiny asymmetric kick so shedding starts promptly

class FlowSolver:
    def __init__(self, cfg: SolverConfig): ...
    cfg; device; dtype; t: float; steps: int; dt: float
    def advance(self, n: int) -> None                # fast path, no autograd
    def coefficients(self) -> np.ndarray             # latest [Cd, Cl, Cm] (all bodies)
    def body_coefficients(self) -> np.ndarray        # per body, shape (nbody, 3)
    def history(self) -> dict[str, np.ndarray]       # t, cd, cl, cm (since last reset)
    def field(self, name: str) -> torch.Tensor       # cell-centred (ny, nx):
        # "u","v","speed","p","cp","vorticity","mask","dye","nut","divergence"
    def surface(self, n_per_body: int = 400, mean: bool = False) -> list[dict]
        # per body: s, x, y, nx, ny, cp, cf, name (instantaneous or time-averaged)
    def set_inflow(self, *, re: float | None = None) -> None # live change
    def set_bodies(self, bodies: list[ImmersedBody]) -> None # live change, rebuild masks
    def start_averaging(self) -> None; def mean_fields(self) -> dict[str, torch.Tensor]
    def reset(self) -> None
    def state_dict(self) -> dict; def load_state_dict(self, d) -> None
# functional, differentiable core (used by optimize/):
make_operators(cfg) -> Ops; init_state(cfg, ops) -> State
step(state: State, geom: Geometry, ops: Ops, dt) -> tuple[State, forces Tensor[3]]
build_geometry(bodies_tensors, ops) -> Geometry   # masks from SDF, differentiable
```

### 5.3 solvers.airfoil
```python
analyze(coords: np.ndarray, re: float, alpha_deg: float, *, mach=0.0, ncrit=9.0,
        xtr_upper=1.0, xtr_lower=1.0, viscous=True) -> PolarPoint
polar(coords, re, alphas, ..., progress=None, cancel=None) -> PolarResult   # warm-started sweep
```

### 5.4 simulation
```python
# simulation/plan.py
plan_simulation(design: Design, sim: Simulation, *, device: str | None = None,
                airfoil_coords=None) -> SimPlan       # SimPlan3D for 3D designs (7.7)
plan_live(design, sim, *, device=None, airfoil_coords=None) -> SimPlan     # explore mode: small
                                                      # grid, t_end = inf, live=True
focus_window(polys, extent, *, ground=False) -> (x0, x1, y0, y1)   # default view window, metres
estimate_steps_per_second(device, cells, *, advection, turbulence, dtype) -> float
record_throughput(device, cells, steps_per_s, ...)  # calibrates the estimate after each run
# simulation/defaults.py (data model only, no torch: the GUI calls it on the event thread)
default_cfd_settings(design, conditions=None, *, device, precision) -> CFDSettings
    # what EVERY new simulation starts from (new study, Setup, project tree, CLI): Draft,
    # and the capped Reynolds mode for airfoils above AIRFOIL_CAP_RE = 1e5 with the cap FIXED
    # at AIRFOIL_DEFAULT_CAP_RE = 2,500 (what every quality resolves on every device), so a
    # quality or device change refines the grid of the same flow; re-exported by plan
default_reynolds_cap(design) -> float | None      # 2,500 for airfoils, None otherwise
geometry_class(design) -> "airfoil" | "vehicle" | "generic"   # layout rules; STARTER_QUALITY
                                                  # (3D: "vehicle" | "generic" | "wing")
default_analysis(design); default_wing_settings(design); default_simulation(design, name,
    conditions, *, device, precision)             # the whole new Simulation (7.7)
simulation_name(design, sim, *, by_reynolds=None) -> str
    # the one naming rule (API, AI tools, example projects): "CFD 3D · 30 m/s · α 0°",
    # "Polar · Re 1.0×10⁶", "Wing polar · 25 m/s · LLT". A flow set by its Reynolds number
    # is named by it ("CFD 3D · Re 100 · α 0°"): by_reynolds=True from a caller given re=,
    # None = by Re below NAME_BY_RE_BELOW_SPEED = 0.1 m/s (no vehicle or wing runs that slowly)

@dataclass
class SimPlan:
    # --- the contract -----------------------------------------------------------
    solver_config: SolverConfig        # non-dimensional, ready for FlowSolver
    design: Design; sim: Simulation
    length_ref: float; speed: float; fluid: Fluid
    re_real: float; re_sim: float; turbulence: str      # "laminar" | "les-<model>"
    t_end: float; t_avg_start: float   # convective times L/U
    dt: float; steps_estimate: int; est_seconds: float; cells: int; cells_per_ref: float
    bodies_wind: list[Body]            # rotated geometry in metres (wind frame)
    extent: tuple[float, float, float, float]   # domain in metres (wind frame)
    notes: list[str]; warnings: list[str]       # human-readable rationale, shown in Setup
    # --- extras used by the runner, the live session, the GUI and the CLI ----------
    device: str; device_name: str; precision: str; advection: str; quality: str
    ground: bool; ground_moving: bool
    strouhal_length: float             # St reference length [m] (chord / frontal height)
    blockage: float                    # frontal height (with a road: height above it) / domain height
    ref_label: str                     # "chord", "vehicle length", ...
    live: bool; capped: bool           # explore plan; Re capped below the real value
    geometry_class: str                # layout rules used: "airfoil" | "vehicle" | "generic"
    conv_rtol: float                   # early-stop target for stderr(mean Cd) / |mean Cd|
    blockage_limit: float              # the blockage the domain was sized for
    est_seconds_headless: float        # estimate without the live view (CLI, scripts)
    bl_resolved: bool                  # grid resolves the laminar boundary layer at re_sim
    time_unit -> float                 # property: seconds per L/U
    grid -> (nx, ny)                   # property
    focus_extent -> (x0, x1, y0, y1)   # property: focus_window() of this plan
    estimate_label -> str              # property: "~40 s" for the Run button ("" when live)
    def summary(self) -> dict[str, str]  # label -> value for the Setup page, shown verbatim;
                                         # includes "Blockage" and "Stops early when"
```

Planner rules (constants in `plan.py`; docs/user_guide_simulation.md explains them):

| rule | Draft | Standard | Fine |
|---|---|---|---|
| `CONVERGENCE_RTOL`: standard error of mean Cd / mean Cd | 5 % | 3 % | 2 % |
| mean Cl target (`CL_RTOL_FACTOR` = 2) | 10 % | 6 % | 4 % |
| `BLOCKAGE_MAX`, vehicle (airfoil and generic) | 8 % (5 %) | 5 % (5 %) | 5 % (5 %) |
| vehicle cells per vehicle height, GPU (CPU) | 36 (24) | 48 (32) | 64 (40) |
| thin-feature target, cells across | 4 | 6 | 8 |
| vehicle duration / averaging from, L/U | 30 / 6 | 36 / 8 | 40 / 10 |
| airfoil duration / averaging from, L/U | 40 / 8 | 50 / 12 | 60 / 16 |

* With a road the domain height comes from the blockage limit alone and is never traded for
  cells; after rounding to FFT-friendly sizes the grid is shrunk back into the cell budget.
  Budget and thin-feature warnings appear only more than 10 % below target. A vehicle whose
  `force_ref` equals its length (a wing on a road) is laid out like an airfoil. Bluff bodies
  get 8–12 sizes of domain upstream and 16–24 downstream.
* Explore plans (`plan_live`) allow 10 % blockage (`BLOCKAGE_LIVE`).
* `reynolds_mode == "capped"` uses `cfd.reynolds_cap`; without one (old projects, a capped car)
  it falls back to the grid's resolvable Re `(5 / (BL_CELLS_MIN h))²` with a warning that the
  Re then depends on the grid. A cap the grid does not resolve gets the LES model and a warning.
* Airfoil domains: 4 chords ahead, 5–6 behind, 6 tall at Draft and Standard (budget permitting);
  every free-flight airfoil plan carries `AIRFOIL_DOMAIN_NOTE` (the remaining domain bias,
  measured: +2 % Cd, +5 % Cl, +6 % Cm at Draft). Airfoil run lengths 40 / 50 / 60 L/U (most
  stop early).
* A grid that needs more than `MEMORY_FRACTION` (80 %) of the device's memory (GPU memory, or the
  computer's RAM for a CPU run; `BYTES_PER_CELL` per precision) raises `ValueError`.
* An unavailable device (including a CUDA index this machine does not have) or an unsupported
  precision falls back with a warning. The CPU path runs
  `CPU_SOLVER_THREADS` solver threads (`device_name` says how many; `live.CPU_THREADS` aliases it).
* When `sim.cfd.compute_sensitivity` is set, `est_seconds` includes
  `optimize.sensitivity.estimate_seconds(plan)` and a note says so.

```python
# simulation/live.py
@dataclass
class ViewSettings:                   # mutable; the GUI edits it while a run is going
    field: str = "speed"              # "speed","u","v","cp","p","vorticity","dye","nut","mask"
    cmap: str | None = None           # None -> default per field (viz.DEFAULT_CMAP)
    range_mode: str = "auto"          # "auto" (robust, per frame, smoothed) | "fixed"
    vmin: float | None = None; vmax: float | None = None   # display units (SI)
    particles: bool = False; streamlines: bool = False
    fps: float = 20.0                 # target frame rate of LiveFrame callbacks

@dataclass
class LiveFrame:
    rgba: np.ndarray                  # (H, W, 4) uint8, row 0 = top (max y)
    extent: tuple[float, float, float, float]   # metres, wind frame
    field: str; cmap: str; vmin: float; vmax: float; units: str   # vmin/vmax in display units
    t: float; step: int; steps_per_s: float     # t in convective times
    coeffs: dict[str, float]          # latest {"cd","cl","cm"}
    history: dict[str, np.ndarray]    # {"t","cd","cl","cm"} (downsampled to <= 2000 pts)
    progress: float; eta_s: float | None
    phase: str                        # "transient" | "averaging" | "done" | "live"
    bodies: list[np.ndarray]          # wind-frame polygons, metres
    particles: np.ndarray | None = None          # (N, 2) metres
    streamlines: list[np.ndarray] | None = None  # polylines, metres
    focus: tuple[float, float, float, float] | None = None   # suggested view window, metres

class FrameRenderer: ...              # shared by the runner and LiveSession; takes focus=
class LiveSession:
    def __init__(self, design: Design, sim: Simulation, *, device=None, view=None): ...
    view: ViewSettings
    def post(self, **changes) -> None   # thread-safe: alpha_deg, speed, ground_moving,
                                        # paused, reset, field, cmap, particles, streamlines
    def tick(self, max_ms: float = 33.0) -> LiveFrame   # run as many steps as fit, render once
    def close(self) -> None
```
Explore ticks are paced on wall time: the warm-up tick and the first render are not timed, and
CUDA ticks run whole 10-step graph blocks. A spinning body without a centre spins about its
pitched centroid. Switching the road between moving and fixed keeps the old solver if the
rebuild fails.

```python
# simulation/runner.py
run_cfd(plan, *, progress=None, frame=None, cancel=None, view=None) -> CFDResult
    # progress(fraction: float, message: str); frame(LiveFrame) at about view.fps (render
    # time is subtracted from the pacing and capped at 20 % of wall time; the final frame is
    # rendered before the sensitivity sweep and delivered last);
    # cancel: threading.Event: stop between chunks and return the partial result (its
    # warnings say so). Raises ValueError for a live plan; RuntimeError for a blow-up
    # (including a collapsing time step) or a device failure (readable CUDA messages).
run_polar(design, sim, *, progress=None, cancel=None) -> PolarResult
run_simulation(design, sim, *, project=None, airfoil_coords=None, device=None, progress=None,
               frame=None, cancel=None, view=None, planned=None) -> any result
    # THE dispatcher: "polar" -> run_polar, "wing" -> run_wing, "cfd" -> plan_simulation (2D or
    # 3D) + run_cfd; "design:<id>" wing airfoils from project / airfoil_coords; planned(plan)
    # sees the plan first. The job manager, the CLI and the scripting API (section 8) use it.
```

`CFDResult.converged` is true only when all of these hold:

1. the run is past the start-up transient, found with MSER and never earlier than
   `plan.t_avg_start`;
2. the standard error of the mean Cd (batch means) is at most `conv_rtol × |Cd|`;
3. the standard error of the mean Cl is at most `CL_RTOL_FACTOR × conv_rtol × |Cl|`, with an
   absolute floor of `conv_rtol × |Cd|` (so the near-zero lift of a symmetric body can converge);
4. nothing drifts: a difference between the two halves of the averaging window counts only if
   it exceeds both the tolerance and 3 standard errors;
5. the amplitude has settled: within a ±25 % band for laminar flows; for LES a change must
   also exceed 3 standard errors of log-rms;
6. periodic flows have at least 4 averaged cycles.

The run stops early once they hold; a run that ends without them carries a warning that states
the uncertainty it did reach, and says the error bars are rough estimates (not measurements) when
batch means flagged them unreliable (`batch_means(...)["reliable"]`). After an averaging restart the stale verdict is cleared and the
progress keeps moving. When `sim.cfd.compute_sensitivity` is set, the runner calls
`optimize.compute_sensitivity` with its own developed solver and then `attach_sensitivity`.
The import is guarded: a missing module, a failure, running out of GPU memory or a cancel each
become a warning on the result, and the forces are always kept.

### 5.4b optimize
```python
compute_sensitivity(design, sim, *, objective=None, plan=None, state=None, window=None,
                    n_per_body=None, device=None, progress=None, cancel=None,
                    ensemble=None, smooth_cells=1.5) -> SensitivityMaps
attach_sensitivity(result: CFDResult, maps) -> CFDResult   # fills SurfaceData.sensitivity
optimize.sensitivity.estimate_seconds(plan) -> float       # used by the planner's estimate
optimize_airfoil(coords, re, *, alpha=None, target_cl=None, objective="ld"|"cd"|"cl",
                 min_thickness=None, keep_area=False, n_iter=8, order=4, max_change=0.06,
                 ncrit=9, mach=0, xtr_upper=1, xtr_lower=1, progress=None, cancel=None,
                 analyze_fn=None) -> OptimizationResult
optimize_shape_cfd(design, sim, *, objective=None, n_modes=12, n_iter=4, step_cells=0.75,
                   max_total_cells=4, keep_area=True, window=None, plan=None, device=None,
                   progress=None, cancel=None) -> OptimizationResult            # experimental
OptimizationResult(method, objective, initial, best, improvement, best_coords, history, cst, x,
                   converged, message, evaluations, failed_evaluations, wall_seconds, notes,
                   warnings, extra)   # .to_dict()
Cancelled      # exception
# also: WindowedObjective, finite_difference_check, normal_modes, airfoil_metrics, OBJECTIVES
```

* `SensitivityMaps` is a `list[np.ndarray]` (one array per body, at exactly the points
  `FlowSolver.surface()` returns) with metadata attributes (objective, value, coefficients,
  window, dt, steps, wall_seconds, peak_memory_bytes, device, cuda_graphs, period, ensemble,
  notes, warnings, …) and a `summary()` method.
* Units: `sensitivity = L_ref × g(s)` [1/m], where `dJ = ∫ g δn ds` for an outward
  displacement δn. Positive means that pushing the surface outward there increases J. Values
  are smoothed over `smooth_cells` cells along the outline (per-vertex derivatives alias with
  the grid). The pattern is reliable, the magnitude indicative (about 2× the physical slope on
  a Re 40 cylinder: window truncation and the masks' sub-cell response; theory_sensitivity.md
  §5): every map carries `MAGNITUDE_WARNING`, and the GUI and the report show it relative to its
  98th-percentile magnitude.
* Shedding is detected with the runner's Strouhal rules (`shedding_period`), so a decaying
  start-up ripple of a steady flow gets the steady window.
* J is the mean of cd, cl, ld (mean Cl / mean Cd) or cm over a window of the developed flow;
  `window` is the averaging time or `(lead, average)` in L/U. Defaults: steady flows 6 + 2 body
  time units on a GPU, 3 + 1 on a CPU; shedding flows a 2-unit lead and whole periods covering
  at least 4 units (2 on a CPU), averaged over `ensemble` start phases (default 3), with a
  warning that such maps are indicative.
* The reverse sweep keeps about √N checkpoints (moved to host memory when the GPU is short) and
  replays one forward + backward step as a CUDA graph. The cell-centre solid fraction χ_c (it
  only switches the eddy viscosity off inside solids) is not differentiated; everything that
  places the wall is.
* `attach_sensitivity` matches bodies by name, resamples by arc length when the lengths
  differ, and merges the map's notes and warnings into the result without duplicates.
* Cancellation: `compute_sensitivity` raises `Cancelled`; the two optimisers return the best
  result so far with `extra["cancelled"] = True` and a warning (`optimize_shape_cfd` raises
  only when cancelled before its first gradient).
* `optimize` imports no torch until a CFD function is used (lazy attributes), so the GUI can
  import the package at start-up.

### 5.5 analysis
```python
stats(series, t, t_start=None) -> Stat; coefficient_stats(t, x, t_start=None) -> Stat
batch_means(x, batch_size=None) -> dict   # stderr, stderr_batches, ..., reliable (see the docstring)
batch_means_stderr(x, batch_size=None) -> float
effective_sample_size(x); integrated_autocorr_time(x); autocorrelation(x); auto_batch_size(x)
strouhal(t, cl, length=1.0, speed=1.0, *, t_start=None) -> float | None
dominant_frequency(t, x, *, t_start=None, min_fraction=0.2) -> float | None
detect_steady(t, x, window=None, *, rtol, atol) -> bool
convergence_state(t, x, t_start=None, *, rtol, atol) -> dict
transient_end(t, x, *, floor=0.1, t_min=None, batch=5) -> float     # MSER
summarize_history(history, *, length, speed, t_start, keys) -> dict
compare_rows(results, baseline_id=None) -> list[dict]   # the Compare page AND the report use it,
                                                        # so screen and PDF never disagree; cells carry
                                                        # stderr_reliable and lower_bound; significant is
                                                        # None for a missing or unreliable error bar
comparison_warnings(results, conditions) -> list[str]   # different speeds / angles / fluids / Re_sim
conditions_differ(conditions) -> list[str]; reynolds_mismatch(results) -> {sim_id: re_sim}
                                                        # fluids differ beyond FLUID_REL_TOL = 1e-3
                                                        # in density or viscosity (rounding is not
                                                        # a change of fluid)
polar_metrics(polar) -> dict[str, float]                 # cl_max, alpha at cl_max, cl_max_reached
                                                         # (0: lift still rising at the last angle,
                                                         # cl_max is a lower bound), best L/D, ...
METRIC_KEYS; Metric
```

### 5.6 viz
```python
COLORMAPS: dict[str, np.ndarray (256,3) uint8]; DEFAULT_CMAP: dict[field, cmap]
FIELD_INFO: dict[field, (label, units_kind, diverging: bool)]
render_field(field: Tensor|ndarray, cmap: str, vmin, vmax, *, mask=None, solid_rgba=..., log=False)
    -> np.ndarray (H, W, 4) uint8, row 0 = top
auto_range(field, *, diverging=False, mask=None, prev=None, smoothing=0.8) -> (vmin, vmax)
streamlines(u, v, extent, *, density=1.0, mask=None) -> list[np.ndarray]   # evenly spaced
class Particles: torch tracers; seed(), advect(u, v, dt), positions() -> ndarray (N,2)
lic_texture(u, v, *, extent, shape, length, ...) -> line-integral-convolution texture
colorbar_image(name, length, thickness, *, vertical) -> (H, W, 4) uint8; field_info(name) -> FieldInfo;
default_cmap(name); get_lut / lut_rgba(name); to_si(name, values, *, speed, density, length)
# viz/figures.py (matplotlib Figure builders, no pyplot) — used by report and GUI exports:
fig_geometry, fig_cp, fig_history, fig_polar, fig_field, fig_compare_bars, fig_sensitivity
# 3D and wings (report 7.11): images rendered off-screen by viz3d, composed by matplotlib
fig_views3d(images, names, titles=None, extents=None, *, st, figsize, color=None, ground=None,
            layout="auto")             # several camera views, one colour bar drawn by matplotlib
fig_history3d(history, t_avg_start=None, stats=None, *, st, figsize, symmetry_plane=False)
fig_wing_polar(wings, *, st, figsize, labels=None)      # CL–α, CL–CD, Cm–α of WingResults
fig_span_loading(wing, *, st, figsize, alphas=None)     # c·cl / c̄ and cl across the span
fig_wing_planform(wing, *, st, figsize, metrics=None)   # planform with MAC and quarter-chord line
SurfaceCut                             # a section of a 3D surface for fig_cp(kind="body")
COEFF_LABELS_3D                        # cd, cs, cl, c_roll, cm, cn -> (symbol, name)
```
`fig_field` takes `xlabel` / `ylabel` / `lines` so it can draw a 3D slice in its own plane;
`fig_cp(kind="body")` plots Cp along a cut of a 3D surface.

### 5.6b report / export
```python
report.SECTIONS = ("summary","geometry","setup","results","surface","fields","convergence",
                   "sensitivity","polar","comparison","methodology")
report.generate(project, sim_ids: list[str], path, *, fmt="pdf"|"html", sections=None,
                title=None, author=None) -> Path
report.build_document(project, sim_ids, *, sections, title, author, ...) -> Document  # format-neutral
export.result_csv(result, path); export.history_csv(result, path); export.surface_csv(result, path)
export.polar_csv(polar, path); export.fields_vtk(field_data, path)  # .vti XML ImageData
export.fields_npz(field_data, path); export.design_svg/design_dxf/design_dat(design, path)
# (3D exports: 7.10)
```
* **3D and wing results** go through the same `SECTIONS`: `report/content3d.py` writes the 3D
  and wing blocks of each section, `report/common.py` holds what 2D and 3D share (`Entry`,
  the ± / Δ formatting; `content.py` re-exports the old names), `report/render3d.py` draws
  the pictures: `render_views(build, views=("iso", "front", "side", "top"), *, size, theme,
  scale, ground, focus, margin, orbit) -> Views` (one scene, several camera presets, a shared
  colour range handed back as a `ColorSpec`, orthographic views cropped to the body with
  their world extent) and `plane_section(vertices, faces, axis, position, values=None)` (exact
  mesh sections for slice outlines and centre-line Cp). Without an OpenGL context it raises
  `RenderUnavailable` and the writer prints the reason in place of that figure; the rest of
  the report renders. Wing sweeps go into the "polar" section, whose title follows its content
  ("Airfoil polars", "Wing polars", "Airfoil and wing polars"); `SECTION_TITLES["polar"]` is
  "Polars".
* Number rules (`report/fmt.py`, 2D and 3D): a mean below `ZERO_COEFF` = 10⁻⁶ prints "≈ 0"; a
  mean is quoted to at most `MAX_SIG` = 5 significant figures (its error then as a power of
  ten); standard errors below 10⁻⁵ print as powers of ten; unsigned percentages below the
  shown precision print as an upper bound ("< 0.01 %").

### 5.7 project
Folder format `<Name>.aerolume/`:
```
project.json                    name, schema, version, created, modified, design ids, sim ids, comparisons
designs/<id>.json               Design.to_dict()
designs/<id>.npz                3D designs: mesh vertex / face arrays (7.10)
sims/<id>/simulation.json       Simulation.to_dict()
sims/<id>/result.json           CFDResult / PolarResult / CFD3DResult / WingResult scalars
sims/<id>/arrays.npz            histories, surface data, fields (float16 for fields)
reports/<name>.pdf|.html
scripts/*.py                    the user's scripts (Script console, section 8)
experiments/<id>/               AI assistant experiment logs (section 9)
```
`Project.create(path, name)`, `Project.open(path)`, `.save()`, `.add_design()`, `.update_design()`,
`.add_simulation()`, `.update_simulation()`, `.designs: dict[id, Design]`, `.simulations: dict[id, Simulation]`,
`.save_result(sim_id, result)`, `.load_result(sim_id, *, arrays=True)`, `.has_result()`,
`.delete_result()`, `.result_summary(sim_id)` (scalars only, no arrays), `.simulations_of(design_id)`,
`.remove_design()` (also removes its simulations), `.remove_simulation()`, `.duplicate_simulation()`,
`.sim_dir()`, `.reports_dir`, `.scripts_dir`, `.experiments_dir` (the three folders are created on
demand and belong to their tools: `save()` never rewrites or prunes them), `.schema` (7.10).
Writes are atomic (temp file + replace); `project/recent.py` keeps the
recent-projects list. Opening skips any damaged design or simulation file (and resets unreadable
`settings` / `comparisons`) with a message in `load_warnings`, reads JSON with or without a
byte-order mark, and pins the cap of a capped simulation saved without one to its result's `re_sim`. Fields are stored as float16 with a per-field scale, so a Draft result is a
few MB.

Example projects ship as package data in `src/aerolume/examples/projects/<Name>.aerolume/` (real
Draft results, regenerated by `examples/make_example_projects.py`). Two are 3D, so a first look
finds a 3D result without a GPU run: "Sedan in 3D" (`geometry3d.sedan()`, Draft CFD at 30 m/s on
a moving road, with surface cp, the mean volume and the "instant" smoke snapshot; its volumes are
block-averaged to fit the 15 MB package-data budget, and its notes say so) and "Tapered wing" (a
viscous vortex-lattice polar). The builder creates every simulation with `default_simulation`
and runs it with `run_simulation`, so the examples are exactly what the Run button produces.

### 5.8 app (PySide6)
The binding visual and interaction spec is docs/ui_guidelines.md; this section is the code
contract.

* **Window** (`app/main_window.py`). Menus (File, Edit, View, Simulation, Tools, Help), one
  toolbar (New, Open, Save | Import geometry | Run, Stop | Report), the **step bar**
  (`widgets/step_bar.py`: flat tabs Geometry → Setup → Simulate → Results → Compare → Report
  with an accent underline and a check on completed steps), the **Project** dock
  (`widgets/project_tree.py`: designs with their simulations nested, a status icon per
  simulation, context menu: new simulation, run, duplicate, rename, delete), the **Messages**
  dock (`widgets/messages.py`: planner notes, warnings, errors, run log), and the status bar
  (message, run progress, device, units). The Project tree is the only selector: there are no
  design / simulation combo boxes. On the Start page the docks and the step bar step aside.
  Pages are imported lazily; a page whose module fails is replaced by `PlaceholderPage` (the
  error and a Retry button), so one broken page never stops the app from starting.
* `app/state.py`: `AppState(QObject)` is the single source of truth: project, current design
  id, current simulation id. Signals `projectChanged`, `designsChanged`, `simulationsChanged`,
  `currentDesignChanged(str)`, `currentSimulationChanged(str)`, `resultChanged(str)`,
  `reportsChanged()` (a report was written: the step bar ticks Report),
  `navigateRequested(str)`, `statusMessage(str, int)` and `messageLogged(level, source, text)`
  (feeds the Messages dock). `navigate("page")` or `navigate("page:action")` shows a page,
  calls its `on_shown()` and then `handle_action(action)` (`"simulate:run"`, `"simulate:live"`,
  `"results:fields"`, `"geometry:import"`, …). Shell actions that belong to no page go through
  `request(command)` and the `commandRequested(str)` signal, which the main window carries out
  (`AppState.COMMANDS`: `"import-mesh"` or `"import-mesh:<path>"`, `"new-wing"`).
  `new_simulation(design_id, analysis=None, *, select=True, device=None, precision=None)` adds
  a simulation with every default of `simulation.defaults.default_simulation` (at the
  conditions of the design's latest simulation) and refuses an analysis the design cannot have
  (`ValueError`); `unique_design_name(base)` / `unique_simulation_name(design_id, base)` append
  " 2", " 3", …. Module helpers shared by pages and the tree: `is_3d`, `is_wing`,
  `design_tag` ("2D" | "3D" | "Wing"), `design_icon`, `design_description`, `analysis_label`,
  `analysis_description`, `analyses_for(design)` (the analyses a design may have, the default
  first).
* **3D in the shell.** The step bar's `set_context(text, tag)` shows the selected design with a
  plain-text "3D" / "Wing" tag before a thin rule; the status bar's run status reads
  `jobPlanned` / `jobFrame` (grid, cells, device). The toolbar's Import geometry is a dropdown
  (2D outline… Ctrl+I, 3D mesh… Ctrl+Shift+I). Mesh import goes through
  `main_window.MESH_IMPORT_HOOK` = `"aerolume.app.dialogs.mesh_import:import_mesh_dialog"`
  (then `MESH_IMPORT_HOOK_ALTERNATIVES`) with the contract `import_mesh_dialog(parent, state,
  path=None, *, add=None) -> list[Design] | None` (`add` is the shell's adder, which keeps
  names unique; a dialog that adds designs itself is not added to twice; with no dialog the
  shell imports with automatic units as a generic body and warns). `pages.start.STUDY_TYPES`
  lists the study kinds (airfoil, vehicle (2D), 3D body from a mesh, wing, empty) for the
  Start page and the New study dialog; `MESH_EXTENSIONS`, `starter_simulations(kind, design)`
  and `create_study(kind, name, folder, *, designs=None)` (a "mesh" study starts empty and the
  import fills it; a cancelled import leaves a valid empty study).
* `app/jobs.py`: `JobManager(QObject)` runs CFD / polar jobs on worker threads, one at a time,
  FIFO queue for the rest:
  ```python
  class JobManager(QObject):
      jobQueued = Signal(str); jobStarted = Signal(str)
      jobProgress = Signal(str, float, str)      # sim id, fraction, message
      jobFrame = Signal(str, object)             # sim id, LiveFrame (coalesced: newest only)
      jobFinished = Signal(str, object)          # sim id, CFDResult | PolarResult (already saved)
      jobFailed = Signal(str, str); jobCancelled = Signal(str)
      jobPlanned = Signal(str, object)           # sim id, the SimPlan the run uses
      queueChanged = Signal()
      view: ViewSettings                         # shared with the running job
      runner                                     # swappable run function (tests use fakes)
      def __init__(self, state: AppState): ...
      def submit(self, sim_id: str) -> None; def cancel(self, sim_id: str) -> None
      def cancel_all(self) -> None; def wait(self, timeout=30.0) -> bool
      def running_id(self) -> str | None; def queued_ids(self) -> list[str]
      def is_busy(self, sim_id) -> bool; def plan_of(self, sim_id); def progress_of(self, sim_id)
      def result(self, sim_id); def cached_result(self, sim_id); def result_error(self, sim_id) -> str
      def shutdown(self, timeout=10.0) -> None    # cancel all, join threads (window close)
      def unsaved_ids(self) -> list[str]; def retry_saves(self) -> None
  ```
  Workers receive deep copies of the design and simulation; frames go through a one-slot
  mailbox so a slow GUI never builds a backlog; `project.save_result` runs on the worker
  thread. On success the result is saved, `sim.status` set, `state.update_simulation` and
  `state.notify_result` called. A cancelled run keeps its partial result, marked "cancelled" —
  except a stopped re-run of a simulation that had a complete result, which keeps that result
  (status "done"). A queued job that is removed (or dropped at a project switch or shutdown) gets
  back its status from before it was queued. Saving a status never raises (a failure is logged and
  the queue moves on); a result whose save fails is still delivered, held in memory
  (`unsaved_ids`) and saved again in the background with a growing delay (`SAVE_RETRY_DELAYS_S`).
  `LiveWorker` runs a `LiveSession` (explore mode) on its own thread with `start`, `post`,
  `stop` (waits at most `STOP_WAIT_S`; a session still being built is detached and closes
  itself, never taking a newer session's place), `shutdown` and the signals `frameReady`,
  `started`, `failed`, `stopped`.
  `run_in_background(fn, ..., on_done, on_error)` runs one-off work (exports, reports).
* **Pages** (each a `QWidget` with `__init__(self, state: AppState, jobs: JobManager)`):
  `StartPage`, `GeometryPage`, `SetupPage`, `SimulatePage`, `ResultsPage`, `ComparePage`,
  `ReportPage` in `app/pages/<key>.py`; optional `on_shown()` and `handle_action(action)`.
  Every page tolerates `state.project is None` or no selection with an empty state that names
  the next action. Each page owns its right-hand property grid (`widgets/common.PropertyGrid`).
* **Window rules.** Closing asks before stopping running or queued simulations or losing an
  unsaved result (title bar, Alt+F4, File ▸ Exit; any close of a visible window) and saves the
  project before anything is stopped. Switching projects asks only once the target is known and
  opened, and stops the jobs just before the switch (a cancelled dialog, a damaged project or the
  open project stop nothing). Geometry files dropped anywhere on the window are imported.
* **Pages.** Setup's `setup:new` creates exactly one simulation; a new simulation copies the
  conditions, quality and Reynolds settings of the project's latest comparable CFD run
  (`setup.comparable_template`). The Simulate Run panel always describes the selected simulation
  (Stop acts on it; a queued one is removed from the queue). Compare never takes a polar as the
  baseline of CFD runs and shows `comparison_warnings` above its table.
* **Examples.** `pages/start.example_projects()` lists `aerolume/examples/projects/*` (package
  data, so it works in an installed package) and then `<repo>/examples/projects/*`; the Start
  page shows 3D examples first, then wings, then 2D (5.7 lists what ships);
  `open_example()` copies the chosen one to `~/Documents/Aerolume/Examples` before opening it,
  so the shipped copy is never modified.
* `app/theme.py`: light (default) and graphite dark palettes, one global QSS, one accent.
  Pages style themselves only through dynamic properties; the full hook list is the module
  docstring of `theme.py` (`QPushButton[variant="primary"|"ghost"|"danger"|"link"]`,
  `QToolButton[variant="icon"|"toggle"]`, `QFrame[card|panel|surface|chrome|divider]`,
  `QLabel[role="h1"|"h2"|"h3"|"secondary"|"muted"|"small"|"mono"|"kpi-value"|"kpi-label"|
  "status-good"|"status-warn"|"status-bad"|...]`, `QLineEdit[invalid="true"]`).
  `theme.tokens()` returns the active colour dict (the guideline table: bg, panel, surface,
  border, border_strong, text, text2, muted, accent, accent_soft, good, warn, bad, series, plus
  derived keys such as grid and plot_bg); `theme.changed` fires on a theme switch;
  `theme.apply(app, name)` installs palette, style sheet, fonts (`ensure_fonts`) and the
  pyqtgraph defaults. Offscreen rendering needs `theme.apply` first, or text renders as boxes.
* `widgets/plots.py` patches `pyqtgraph.AxisItem.paint` (`_guard_empty_axis_paint`): pyqtgraph
  0.14 records axes into a `QPicture`, which crashes PySide6 6.11 on the offscreen platform; the
  patch paints straight onto the widget painter (same pixels). Do not remove it.

### 5.9 cli
`aerolume-cli <command>` (`cli.py`); exit code 0 on success, 1 when a check failed or a run
could not be done or was stopped, 2 for bad arguments. Heavy modules load only in the commands
that need them.

| command | what it does |
|---|---|
| `doctor [--no-solver]` | versions, devices, CUDA, a short solver check |
| `bench [--device D] [--quick]` | solver throughput on this machine |
| `polar (--naca C \| --dat F) --re R [--alpha a:b:s] [--ncrit N] [--xtr X] [--mach M] [--csv F]` | viscous–inviscid polar |
| `quick (--template T \| --naca C \| --dat F) [--speed V] [--alpha A] [--quality Q] [--device D] [--re-mode M] [--re-cap R] [--duration T] [--cells-per-ref N] [--no-ground \| --fixed-ground] [--sensitivity cd\|cl\|ld\|cm] [--report F] [--save DIR] [--plan-only] [-v]` | plan and run one CFD simulation; with `--save` the project is created before the run, so a bad path fails at once |
| `quick3d (--stl F \| --sphere D) [--kind vehicle\|generic] [--units U] [--length L] [--clearance C] [--speed V] [--alpha A] [--yaw B] [--quality Q] [--device D] [--re-mode M] [--re-cap R] [--duration T] [--cells-per-ref N] [--no-ground \| --fixed-ground] [--full-domain] [--sensitivity cd\|cl\|ld] [--save DIR] [--plan-only] [-v]` | plan and run one 3D CFD simulation of a mesh file (STL / OBJ / PLY) or a sphere |
| `wing --naca C --span B --chord C [--taper T] [--sweep S] [--dihedral D] [--twist W] [--alpha a:b:s] [--method vlm\|llt] [--inviscid] [--speed V] [--yaw B] [--ncrit N] [--mach M] [--csv F] [--save DIR] [-v]` | finite-wing polar of a straight-tapered wing (sweep at the quarter chord, tip twist: `geometry3d.tapered_wing`'s definitions) |
| `run PROJECT SIM [--device D] [-v]` | run a saved simulation (2D, 3D or wing) and store its result |
| `cavity [--ra R] [--pr P] [--size L] [--3d] [--quality Q] [--device D] [--duration T] [--cells-per-ref N] [--save DIR] [--plan-only] [-v]` | the differentially heated cavity through the planner; prints the hot wall's mean Nu next to de Vahl Davis (2D) or Tric et al. (3D) (v0.3) |
| `script FILE [--project P [--create]] [--figures DIR] [-v] [-- ARGS]` | run a Python script against the scripting API (section 8) |
| `validate [--quick] [--full] [--device D]` | quick verification / validation subset (`--quick`, the default; `--full`, exclusive with it, adds the Re 100 shedding case and the heated cavity) |

`quick` and `quick3d` take the placed-physics flags of section 10 (v0.3): `--heated T` (every
body held at T °C, heat transfer on), `--ambient T`, `--no-buoyancy`, `--fan X,Y[,Z]:D:VALUE[:MODE[:DIR]]`
(repeatable) and `--closed X0,X1,Y0,Y1[,Z0,Z1]` (a room instead of a tunnel). Results print the
Nusselt number, the heat rate, a fan table and the solver diagnostics. The MCP server is a separate
console script, `aerolume-mcp` (section 13, docs/mcp.md).

During `quick` and `run` the first Ctrl+C stops the run at the next chunk and keeps the partial
result (`run` saves it marked cancelled, exit code 1); a second Ctrl+C aborts. Results print
relative errors and whether a sensitivity map was computed.

## 6. Quality bars

* Solver verification: discrete Poisson exact to round-off; projected divergence < 1e-10 (f64),
  < 1e-5 (f32); Taylor–Green spatial order ≈ 2.
* Solver validation (docs/validation.md): cylinder Re 20/40 (steady Cd, wake length),
  Re 100 (Cd, Cl', St), Schäfer–Turek 2D-1 and 2D-2, square cylinder Re 100,
  NACA 0012 Re 1000; airfoil VII solver vs. published NACA 0012 / 2412 / 4412 data.
* Performance: GPU f32 1024×512 ≥ 300 steps/s (CUDA graphs); CPU f32 512×256 ≥ 15 steps/s.
* The UI never blocks: all solves on worker threads; the window stays responsive at 60 fps.
* Every number shown to the user carries its validity context (Re_sim vs Re_real, grid,
  statistical uncertainty, convergence state).


---

## 7. 3D (development stage v0.2)

v0.2 adds three things, on top of everything in sections 1–6
(2D keeps working unchanged; a design is 2D or 3D by `Design.dimension`):

1. **3D CFD for any shape** — import STL/OBJ/PLY (cars, drones, riders, wings), no
   watertight mesh needed (AirShaper behaviour), GPU incompressible Navier–Stokes on a
   uniform Cartesian grid with Brinkman immersed boundaries, forces and moments with
   statistics, surface pressure, slices, 3D streamlines and vortex iso-surfaces.
2. **Finite wings** (XFLR5-like) — a planform editor (`WingSpec`) on top of the 2D
   airfoil polars: vortex-lattice (VLM) and nonlinear lifting-line (LLT) analysis with
   viscous strip drag; CL, CDi, CDv, Cm, span efficiency and span loading in seconds.
   A wing can also be lofted to a mesh and sent to 3D CFD.
3. **3D sensitivity maps** — dJ/dn on the surface mesh by autograd through the 3D
   solver (AirShaper's adjoint map), on coarse grids that fit 8 GB.

The 3D viewer is **VTK** (`vtk` 9.x, BSD) used as an off-screen renderer whose image
is painted in a Qt widget (section 7.9), so it behaves the same on screen, offscreen
(tests, screenshots) and in reports.

### 7.1 Budget (RTX 4060 Ti, 8 GB)

Uniform grids only (the exact pressure solve needs them). Practical sizes: Draft ≈ 1–2 M
cells, Standard ≈ 4–6 M, Fine ≈ 10–14 M (float32; measured ~200 B/cell with running means,
2.4 GB at 12 M cells). A symmetric body at zero yaw uses a **symmetry plane at y = 0** and
simulates half the domain. Sensitivity maps run on grids up to 2.1 M cells on a GPU and 0.4 M
on a CPU (`optimize.sensitivity3d.DEFAULT_MAX_CELLS`; the planner's 1.5 M is only the fallback
when that module cannot be imported). CPU fallback works but only Draft is practical. The
planner states all of this.

### 7.2 Frame and conventions (3D)

* Right-handed **x downstream, y to the left of an observer looking downstream** — which is
  the **right** of a driver or pilot facing forward (noses point to −x) — **z up**. Road plane
  z = 0 (vehicles sit on it). Wings: root leading edge at the origin, span along ±y
  (`WingSection.y ≥ 0` is the right half).
* Flow from −x to +x. Angle of attack / pitch `alpha_deg` rotates the geometry about the +y
  axis through `moment_ref3` by **+alpha** (positive rotation about +y lifts the nose, which
  points to −x). Sideslip `yaw_deg` rotates the geometry about +z through `moment_ref3` by
  **−yaw**: positive yaw turns the nose toward +y, the free stream in the vehicle's own frame
  is `U (cos yaw, sin yaw, 0)`, and the **wind comes from the vehicle's left, the −y side**.
  Rotation order: yaw, then pitch. On a road the wheels only yaw. The finite-wing solver uses
  the same rotation: free stream in the wing frame `(cos α cos β, cos α sin β, sin α)` with
  β = yaw. (`plan3d.rotation3`, `ns3d.geometry.rotation_matrix` and
  `solvers.wing.freestream_direction` agree; tests/test_frame3d_conventions.py.)
* Forces in wind axes: Fx drag, Fy side force (+y), Fz lift (negative = downforce), N.
  Moments about `moment_ref3`: Mx roll, My pitch (+ nose-up), Mz yaw, N·m (right-hand rule
  about +x, +y, +z; with x pointing aft, a negative roll lowers the right wing).
* Coefficients: `CD, CS, CL = F / (q·ref_area)`; `Cm = My / (q·ref_area·length_ref)`;
  `C_roll, Cn = M / (q·ref_area·span_ref)`. `Re = ρ U length_ref / μ`.
  Vehicles: ref_area = frontal area (projected on the y–z plane), length_ref = length,
  span_ref = width. Wings: ref_area = planform area, length_ref = MAC, span_ref = span.
  Bluff bodies (sphere…): ref_area = frontal area, length_ref = diameter/height.
* Meshes: `Mesh3D(vertices (N,3) m, faces (M,3))`, any orientation/winding, not necessarily
  closed. Solver internals non-dimensional exactly as in 2D (length_ref, U, ρU²).
* **Wheels** spin at `omega` about `axis` through `center` by the right-hand rule (surface
  velocity `ω axis × (x − center)`). A wheel rolling on a road moving at +U with its axle along
  +y spins at **ω = −U/r about +y** (the contact point moves at +U, the top upstream); the
  planner sets this when `GroundConfig.rotate_wheels` is on. (The 2D `Body` uses +U/r: its
  counter-clockwise rotation is seen from the −y side.)
* Symmetry plane: when used, the solver domain is y ∈ [0, W/2]; reported forces are for the
  full body (Fx, Fz doubled from the half; Fy, Mx, Mz are zero by symmetry and reported as 0),
  surfaces/volumes are mirrored for display (`mirrored=True`).

### 7.3 Package layout additions

```
src/aerolume/
  geometry3d/            mesh I/O (STL ascii/binary, OBJ, PLY), mesh ops, units guess,
                         orientation, symmetry detection, frontal area, part gaps, robust
                         voxelisation and narrow-band signed distance (torch), primitives
                         (sphere, box, cylinder), wing lofting + planform metrics
  solvers/ns3d/          3D incompressible Navier–Stokes (PyTorch), functional core + FlowSolver3D
  solvers/wing/          VLM + nonlinear LLT, viscous strip coupling to solvers.airfoil
  solvers/fpmode.py      subnormal flushing switched per block (CPU solver loops, k-d trees)
  simulation/plan3d.py, runner3d.py, live3d.py, wing.py   (plan_simulation / run_cfd /
                         LiveSession dispatch on design.dimension; run_wing)
  optimize/sensitivity3d.py
  viz3d/                 VTK scene building + off-screen rendering (no Qt)
  app/widgets/view3d.py  Qt widget painting viz3d renders, mouse camera control
  app/pages/wing.py      (wave 2) planform editor page content inside Geometry
```
Import rules: `vtk` only in `viz3d/` and `app/widgets/view3d.py` (lazily: importing either
loads no VTK); torch as before plus `geometry3d/voxel.py` (a lazy attribute of `geometry3d`,
whose other modules need only numpy and scipy), `solvers/ns3d`, `optimize/sensitivity3d.py`,
`simulation/*3d*`. `solvers.wing` imports numpy, scipy and `solvers.airfoil` only.

**Subnormals.** CPU solver loops flush subnormals to zero (decaying tails are ~100× slower
otherwise) through `solvers.fpmode.flush_denormal`, which restores the thread's previous mode
afterwards; left on, it crashes scipy's `cKDTree` (native stack overflow). Geometry code that
uses k-d trees runs inside `fpmode.ieee_subnormals()`, so a caller that left flushing on cannot
crash it either.

### 7.4 geometry3d
```python
# I/O (meshio.py)
read_mesh(path, *, units=None, notes=None, name=None) -> list[Mesh3D]
    # .stl (ascii incl. multi-solid / binary), .obj (groups/objects -> meshes, polygons fan-
    # triangulated, negative indices), .ply (ascii, binary LE/BE). units: None = keep the file's
    # numbers; "auto" = guess_units; "m" | "cm" | "mm" | "in" | "ft" or a float (metres per unit).
    # notes (a list) collects importer messages. Binary STL meshes are named after the file.
read_stl / read_obj / read_ply (same keywords); SUPPORTED_EXTENSIONS_3D; UNIT_SCALES; unit_scale(units)
write_stl(meshes, path, binary=True); write_obj(meshes, path); write_ply(meshes, path, binary=True, double=False)
guess_units(meshes) -> (scale_to_m, reason)               # the GUI confirms: mm vs cm/inch is ambiguous
# mesh ops (mesh_ops.py, numpy; scipy inside a few functions)
transform(meshes, scale=1, rotate_deg=(rx, ry, rz), about=None, translate=None, mirror_axis=None)
    # rotations right-handed, applied rx -> ry -> rz about `about`; mirrors / negative scales keep
    # outward winding
rotation_matrix(rotate_deg); translate; mirror(meshes, axis="y", about=0); merge; weld(mesh, tol)
remove_degenerate; compact; face_normals; vertex_normals; normals(mesh, per="face"|"vertex");
face_areas; area; volume (signed); is_watertight; edge_counts; boundary_edges; inconsistent_edges;
components (edge-connected); mesh_info(mesh) -> dict (triangles, watertight, holes, boundary_edges,
  nonmanifold_edges, inconsistent_edges, degenerate_faces, ...); bbox
ORIENT_PRESETS: dict name -> rotate_deg ("y-up->z-up", ...); orient_preset(meshes, name)
auto_orient(meshes, notes=None)    # longest axis -> x, mirror symmetry decides span vs up, flat side down
orient_outward(mesh)               # consistent outward winding (faces only, vertices untouched)
place_on_road(meshes, clearance=0); centre_y(meshes); place_nose(meshes, x=0)
symmetry_error(meshes, plane_y=0, n_samples=20000) -> RMS mirror distance [m]
frontal_area(meshes, axis="x", resolution=1024) -> m²; projection_mask(meshes, axis, resolution)
sample_surface(meshes, n, seed); wheel_params(mesh) -> (centre, axis, radius); decimate (clustering, previews)
# designs and checks (design3d.py)
design_from_meshes(meshes, name, kind="vehicle"|"generic", *, length=None, units=None, on_road=None,
    clearance=None, fix_winding=True, source=None, notes=None) -> Design
check_design3d(design, cells_per_ref=None) -> list[str]
part_gap(a, b, margin, *, size=None, samples=3000) -> float   # inf: far, touching or interpenetrating
find_wheels3d(meshes) -> [(centre, axis, radius) | None]
# primitives
sphere(d, n, center); box(lx, ly, lz, center); cylinder(d, length, axis, n, center, capped)
# wings (wing.py)
loft(wing, *, n_chord=40, n_span_per_panel=None, closed_tips=True, airfoil_coords=None, name="Wing") -> Mesh3D
planform(wing) -> dict    # the wing solver's planform_metrics (area, span, aspect_ratio, mac, mac_le_x,
                          # mac_y, taper, sweep_c4_deg, mean_chord) + mac_le_z, sweep_le_deg,
                          # dihedral_deg, root_chord, tip_chord
design_from_wing(wing, name, *, moment_ref=None, source=None) -> Design (kind="wing", meshes=[])
tapered_wing(span, root_chord, tip_chord=None, *, sweep_deg=0, dihedral_deg=0, twist_deg=0,
             airfoil="2412", tip_airfoil=None, symmetric=True) -> WingSpec
    # one root-to-tip panel; sweep at the quarter chord (planform()["sweep_c4_deg"] returns it),
    # twist = tip twist (negative = washout); the CLI, the scripting API and the editor share it
section_outline(airfoil, n_chord=40, *, airfoil_coords=None); naca_code(name)
# voxel (torch; lazy attributes of the package)
Grid(nx, ny, nz, h, x0, y0, z0)    # NamedTuple: shape (nz, ny, nx), cells, extent, centres(axis),
                                   # faces(axis) = the MAC face grid normal to axis; grid_around(meshes, h, pad)
inside_mask(meshes, grid, *, device=None, budget_mb=384, info=None) -> bool (nz, ny, nx)
sdf_band(meshes_t, grid, band, *, inside=None, sheet_thickness=None, device=None, dtype=None,
         budget_mb=384, info=None) -> (nz, ny, nx), -band / +band outside the band
sdf_bands(meshes_t, [g, g.faces("x"), g.faces("y"), g.faces("z")], band, ...) -> list  # one preparation
solid_fraction(sdf, h, width=1.0)  # = geometry.sdf.volume_fraction (the 2D C1 ramp)
voxelize(meshes, grid, *, width=1.0, smooth=True, sheet_thickness=None, device=None,
         dtype=float32, budget_mb=384, info=None) -> fraction (nz, ny, nx)
```
* **`design_from_meshes`** puts vehicles nose at x = 0, centred on y = 0, lowest point at
  `clearance` above the road; generic bodies centred (or on the road with `on_road`). It
  re-winds faces outward (`fix_winding`; vertices and their order untouched, so per-vertex
  results map 1:1 onto the user's mesh), detects wheels (role "wheel", axis, centre, ω = 0 until
  the planner sets it), and fills the references: `ref_area` = frontal area of the union;
  `length_ref` = length, or for a **compact** bluff body (length ≤ 1.5 × its smaller
  cross-stream size) that cross-stream size; `span_ref` = width; `force_ref` = frontal height
  (informational in 3D); `moment_ref3` = (x-centre, 0, road) for vehicles, the bounding-box
  centre otherwise. `symmetric` when the mirror error across y = 0 is ≤ 0.5 % of the **largest**
  dimension. `Design.source` gets `scale_to_m`, `frontal_area` [m²], `symmetry_error` [m],
  `compact`, `on_road`, `import_notes` (messages `check_design3d` repeats).
* **`design_from_wing`**: `ref_area` = planform area, `length_ref` = `force_ref` = MAC,
  `span_ref` = span, `moment_ref3` = the quarter-MAC point (mac_le_x + mac/4, 0, mac_le_z);
  `meshes` stays empty (lofted on demand). The planform numbers are the wing solver's, so the
  design's references equal its defaults.
* **`check_design3d`**: importer notes, units, holes / non-manifold / flipped faces,
  references, road contact, a `symmetric` flag the geometry contradicts; with
  `cells_per_ref` also resolution, thin parts (< 2 cells) and gaps between parts (< 2 cells,
  `part_gap`: only pairs whose bounding boxes come within 2 cells are sampled, both ways; 0.55
  s for a 9-part car, was 4.4 s). Wings get planform checks.
* **Voxel conventions.** Cell (k, j, i) is centred at `(x0 + (i + ½) h, y0 + (j + ½) h, z0 +
  (k + ½) h)`; `Grid.faces("x")` centres are `(x0 + i h, …)`, i = 0..nx. Vertices and grid share
  units (pass non-dimensional vertices for a non-dimensional grid). Outputs are on the vertex
  tensors' device and dtype; gradients flow to vertex tensors with `requires_grad`, also
  through the surface (a centre on a face gets the face's signed plane distance, and exact ties
  between faces the mean of their derivatives). `info` receives counts and timings (layers,
  odd_rays, undecided_cells, buried_triangles, sheet_faces, band_cells, cut_padding, seconds).
  Open sheets (pieces of open components with no inside on either side) are thickened to
  `sheet_thickness` (None = one cell, 0 = off); closed parts never are. **The grid may cut the
  surface** (symmetry plane, road, solver window): the classification field grows past the cut
  by itself, and the result equals that of a grid holding the whole surface.
* For a staggered solver, `sdf_band` needs `band ≥ (mask_width + |mask_shift|) h` for an exact
  ramp; `sdf_bands(meshes, [g, g.faces("x"), g.faces("y"), g.faces("z")], band)` gives the
  centre and MAC-face fields from one preparation. Target met: 256 × 128 × 128 cells × 200 k
  triangles in 0.65–0.76 s on the GPU (< 2 s).

### 7.5 solvers.ns3d (non-dimensional; mirrors 5.2)
```python
@dataclass
class ImmersedMesh: vertices (N,3); faces (M,3); name=""; role="body"; omega=0.0;
                    center=(0,0,0); axis=(0,1,0)          # omega = -U/r about +y rolls on a +U road
@dataclass
class SolverConfig3D:
    nx, ny, nz: int; h: float; x0, y0, z0: float; re: float; bodies: list[ImmersedMesh]
    inlet: str = "uniform"
    bc_y: str = "slip"            # "slip" | "symmetry_ymin" (half domain, plane y = y0 = 0) | "periodic"
    bc_zmin: str = "slip"         # "slip" | "noslip" | "moving" (road at ground_speed)
    bc_zmax: str = "slip"         # "slip" | "noslip"
    ground_speed: float = 1.0
    periodic: bool = False        # verification: periodic in x, y, z
    advection: str = "upwind3"; turbulence: str = "none"   # "none"|"smagorinsky"|"wale"|"vreman"
    cfl: float = 0.5; dt: float | None = None
    ref_area: float = 1.0; length_ref: float = 1.0; span_ref: float = 1.0   # nondim
    moment_ref: tuple[float, float, float] = (0, 0, 0)
    dtype: str = "float32"; device: str = "auto"; use_cuda_graphs: bool = True
    perturb: bool = True
    # advanced (defaults validated in docs/theory_ns3d.md)
    eta: float | None = None          # Brinkman permeability (None = the calibrated 2D rule)
    mask_width: float = 1.0           # half-width of the diffuse interface, cells
    les_constant: float | None = None
    fused_kernels: bool = True        # fused CUDA kernels (jiterator) on the fast path
    window_pitch_deg: float = 0.0     # size the mask window for live attitude changes up to these
    window_yaw_deg: float = 0.0       #   angles, so set_bodies keeps the CUDA graphs
class FlowSolver3D:
    advance(n); coefficients() -> np.ndarray [CD, CS, CL, C_roll, Cm, Cn] (full body);
    body_coefficients(); history() -> dict t, cd, cs, cl, c_roll, cm, cn   (HISTORY_KEYS)
    field(name) -> Tensor (nz, ny, nx): FIELDS3D = "u","v","w","speed","p","cp",
                  "vorticity" (magnitude),"q" (Q-criterion),"mask","nut","divergence"
    slice(axis, position, name, *, mean=False) -> Tensor (2D, SliceData orientation)
    surface(mean=False) -> list[dict]: per mesh name, vertices, faces, normals, cp, cf, tau
                  (solver units; every vertex of the full mesh, also with a symmetry plane)
    start_averaging(); stop_averaging(); mean_fields(names=None) -> dict (MEAN_FIELDS3D: u, v,
                  w, speed, p, cp, vorticity, q, u_rms, v_rms, w_rms, p_rms, cp_rms, mask)
    set_inflow(re=); set_bodies(bodies); reset(); state_dict(); load_state_dict(d)
    uses_cuda_graphs, uses_fused_kernels, fused_error, graph_error, t, dt, cells
estimate_memory_bytes(nx, ny, nz, *, dtype="float32", turbulence="none", mask_window_cells=0) -> int
functional core (differentiable): make_operators3d, init_state3d, step3d, step3d_full,
  build_geometry3d(bodies, ops, *, window=None, use_shared=True), rollout3d (checkpointed),
  mesh_signed_distance (the private exact fallback), COEFF_NAMES
```
Numerics as in 2D (MAC grid, SSP-RK3 with projection per stage, exact Poisson solve in 3D —
dense per-axis transforms as cuBLAS products, IEEE float32 forced during the solve —,
upwind3/central, WALE/Vreman/Smagorinsky on the traceless gradient, implicit Brinkman
penalisation with the calibrated wall offset, penalisation-impulse forces cross-checked by a
control volume, CUDA graphs), float32 on GPU. On CUDA the fast path runs fused kernels
compiled at run time by PyTorch's NVRTC jiterator (a private PyTorch API; on failure the solver
falls back to the plain-torch core, ~3.5× slower, with `fused_error` set). The viscosity is a
kernel launch argument, so on the fused path **`set_inflow(re=)` re-captures the CUDA graphs**
on the next advance. Running means are float32 weighted Welford updates sampled every 10
steps. The masks come from `geometry3d.voxel.sdf_band` evaluated directly on the solver's
window (also where the domain cuts the body); without `geometry3d` a private exact distance
with a winding-number sign runs instead.

### 7.6 solvers.wing
```python
analyze_wing(wing, *, alpha_deg, speed=30.0, fluid=None, method="vlm"|"llt", viscous=True, ncrit=9,
             mach=0, beta_deg=0, moment_ref=None, polar_provider=None, coords_provider=None,
             ref_area=None, ref_chord=None, ref_span=None, cancel=None) -> WingPoint
wing_polar(wing, alphas, *, (the same keywords), sim_id="", progress=None, cancel=None,
           details=None) -> WingResult
SectionPolars(ncrit=9, mach=0, xtr_upper=1, xtr_lower=1, panels=140, alpha_step=1, re_ratio=2,
              coords_provider=None)          # cached 2D viscous section polars per (airfoil, Re)
    .ensure(demands); .table(airfoil, re); .node_res(airfoil); .state(); .load_state(d)
                                             # Re nodes on a 48-per-decade log grid (re_grid_value)
section_points_needed(wing, alphas, *, speed, fluid, ncrit, mach, beta_deg, polar_provider,
                      coords_provider) -> int   # first-pass section points not cached yet
planform_metrics(wing) -> dict(area, span, aspect_ratio, mac, mac_le_x, mac_y, taper, sweep_c4_deg,
                                mean_chord)   # projected on x-y; span tip to tip; MAC = ∫c²dy/∫c dy
freestream_direction(alpha_deg, beta_deg) -> (cos a cos b, cos a sin b, sin a)
```
`fluid=None` is ISA air; `moment_ref=None` the MAC quarter chord at y = z = 0;
`ref_area` / `ref_chord` / `ref_span` default to the planform values. `coords_provider(name)`
resolves "design:<id>" airfoils (simulation.wing passes the project's). `details` (a dict) is
filled with `stall`, `strip_y`, `strip_re`, `polar_nodes`, `reference`. Bad input raises
`ValueError` (`WingGeometryError`) before any work; numerical failures give `converged=False`
points; cancel returns the finished points with a warning "Cancelled after N of M angles".
**Stall onset**: the critical-section estimate (first strip away from the tips reaching its
section cl_max, interpolated between angles) is `WingResult.stall_alpha` / `stall_cl` (NaN when
not reached or inviscid) and the note "Stall onset (critical-section estimate)".
VLM: ring vortices on the cambered mean surface with Multhopp spanwise collocation, wake along
the free stream, Kutta–Joukowski forces on all bound segments, Trefftz-plane induced drag;
LLT: Phillips–Snyder nonlinear with section lift curves from the polars. Viscous drag = strip
integration of section cd at the local effective angle. Compressibility: Prandtl–Glauert for
M < 0.6 (warning above). Theory: docs/theory_wing.md; validation: docs/validation_wing.md.

### 7.7 simulation (3D dispatch)
* `plan_simulation(design, sim, *, device=None, airfoil_coords=None)` returns a `SimPlan3D`
  when `design.dimension == 3`, and `plan_live(...)` the explore plan; `airfoil_coords(name)`
  resolves "design:<id>" airfoils when a wing has to be lofted for CFD. `analysis == "wing"`
  raises `ValueError` (that is `run_wing`'s job).
* **`SimPlan3D`** (its own dataclass, `dimension = 3`) has the SimPlan members where they
  apply (`solver_config` a `SolverConfig3D`, `length_ref`, `speed`, `fluid`, `re_real`,
  `re_sim`, `turbulence`, `t_end`, `t_avg_start`, `dt`, `steps_estimate`, `est_seconds`,
  `est_seconds_headless`, `cells`, `cells_per_ref`, `notes`, `warnings`, `device`, `precision`,
  `advection`, `quality`, `ground`, `ground_moving`, `strouhal_length`, `blockage`,
  `geometry_class`, `summary()`) plus `grid` = `grid3` = (nx, ny, nz), `symmetry_plane`,
  `extent3` (solved domain) / `extent3_full` (mirrored), `focus_extent3`, `focus_window(axis)`,
  `slice_positions`, `bodies_wind3` (wind-frame meshes in metres, the solver bodies' vertex
  order), `meshes_design`, `alpha_deg`, `yaw_deg`, `ref_area`, `span_ref`, `moment_ref3`,
  `blockage_y`, `blockage_z`, `body_bounds`; `extent` is the side view (x0, x1, z0, z1) and
  `bodies_wind` is []. `summary()` adds "Cells (3D)", "Symmetry plane", "Attitude" and the
  height/width shares of the blockage.
* **Planner tables** (`plan3d.py`, each justified there). Draft / Standard / Fine:

  | | vehicle | bluff body | wing |
  |---|---|---|---|
  | blockage (frontal / tunnel area) | 5 / 4 / 3 % | 3 / 2 / 2 % | 3 / 2 / 2 % |
  | share of tunnel width and of height | 50 / 45 / 40 % | 40 / 35 / 30 % | 60 / 55 / 50 % |
  | cells per size, GPU (CPU) | 24 / 32 / 40 per height (12 / 16 / 20) | 20 / 24 / 32 per frontal size (10 / 12 / 16) | 16 / 24 / 32 per MAC (8 / 10 / 12) |
  | domain ahead / behind | 1–1.5 / 2–3 lengths | 5–6 / 10–12 sizes | 3–5 / 5–8 chords |
  | run length (t_end, average from) | 20, 25, 30 L/U | 40, 60, 80 sizes/U | 20, 25, 30 L/U |

  Cell budgets: GPU 2 / 6 / 14 M, CPU 0.15 / 0.3 / 0.5 M (explore: GPU 0.4 M, CPU 40 k, up to 8 %
  blockage). Thin features (3 / 4 / 6 cells across) raise the preset by at most 1.5× (wings
  1.25×). The budget fit shortens the domain first, then narrows bluff bodies and wings, then
  coarsens. Reynolds handling, LES switch and early stop follow 2D; the estimate comes from a
  measured 3D throughput table corrected by this machine's runs (separate 3D calibration).
  The symmetry plane is used when the design is symmetric (re-measured), yaw is 0 and
  `cfd.symmetry == "auto"`. Gaps, mirror error, part thickness and frontal area are cached per
  shape (a re-plan costs ~16 ms on a 200 k-triangle mesh). With `compute_sensitivity` the map's
  time (`optimize.sensitivity3d.estimate_seconds`) is added to the estimate, and a grid above the
  map's cell budget is warned about at plan time.
* `run_cfd(plan, ...)` dispatches to `runner3d.run_cfd3d` and returns `CFD3DResult` (six `Stat`
  coefficients, forces [N] and moments [N m]; per-vertex cp/cf/tau on the user's meshes in their
  vertex order (the missing half of a symmetry-plane run probed at its mirror image); three mean
  slices — y = 0 (or the symmetry plane), z at mid-height, x just behind the body — mirrored to
  full width; a block-averaged mean `VolumeData` ≤ ~2 M cells with u, v, w, cp, q, mask and
  `mirrored`; per-part coefficients; Strouhal number; histories; a partial result on cancel).
  With `sim.cfd.compute_sensitivity` the runner calls `compute_sensitivity3d` while the solver
  is alive and `attach_sensitivity3d`; a missing module, the cell or memory budget, cancel, OOM
  or any failure becomes a warning on the result.
* Frames: `LiveFrame.rgba` is a slice image; `ViewSettings.slice_axis` ("y" | "z" | "x") and
  `slice_pos` (metres, None = plan default) choose it, and `LiveFrame.slice_axis` /
  `slice_pos` report it (None in 2D). The 3D frame extent follows the `SliceData` orientation
  (y-slice → (x, z), z-slice → (x, y), x-slice → (y, z)); `bodies` are cross-section polylines
  (closed loops repeat their first point, sheets are open chains); coeffs carry cd, cs, cl,
  c_roll, cm, cn. `LiveSession(design, sim, ...)` returns a `LiveSession3D` for 3D designs
  (always a full domain: yaw needs both halves); post keys: alpha_deg, yaw_deg, speed,
  ground_moving, paused, reset, field, cmap, streamlines, slice_axis, slice_pos, range_mode,
  vmin, vmax, fps, particles (ignored in 3D).
* `run_wing(design, sim, *, progress=None, cancel=None, project=None, airfoil_coords=None,
  polar_provider=None) -> WingResult` for `analysis == "wing"` (also `runner.run_wing`);
  `wing.airfoil_resolver(project, airfoil_coords)` resolves project airfoils and
  `wing.clear_polar_cache(*, disk=True)` drops the section polars kept between runs and
  (with `disk`) between sessions: they are saved to `<config>/cache/section_polars` after a
  run that computed section points and read back by the next session
  (`AEROLUME_NO_POLAR_CACHE=1` turns that off). `wing.estimate_wing_seconds(design, sim, *,
  project=None, airfoil_coords=None)` is the expected wall time from the section points still
  missing (`solvers.wing.section_points_needed`): about a second when cached, 20-60 s for a
  cold viscous polar. Section-polar Reynolds nodes lie on a fixed log grid (48 per decade) so
  a later run at another speed adds only the nodes it needs. `runner.run_simulation`
  (5.4) dispatches any analysis of any dimension, with the project's airfoils for wing
  analyses and lofted-wing CFD alike.
* `defaults.py`: `default_analysis(design)` ("wing" for wings), `default_wing_settings` (VLM,
  viscous), `default_cfd_settings` (3D: Draft, real Re, symmetry "auto"), and
  `default_simulation(design, name, conditions, *, device, precision)` (vehicles on a moving road
  with spinning wheels), and `simulation_name` (5.4: Re-specified runs are named by Re).
* `live.cpu_threads(device)` restores the thread count and the subnormal mode on exit.

### 7.8 optimize (3D)
```python
compute_sensitivity3d(design, sim, *, objective=None ("cd"|"cl"|"ld"|"cm"), plan=None, solver=None,
                      window=None, device=None, progress=None, cancel=None, max_cells=None,
                      ensemble=None, smooth_cells=1.5) -> SensitivityMaps3D
attach_sensitivity3d(result, maps) -> CFD3DResult
estimate_seconds(plan) -> float
estimate_memory_bytes(cells, *, steps, window_cells=0, dtype="float32", offload=False) -> int
SensitivityBudgetError(ValueError); DEFAULT_MAX_CELLS = {"gpu": 2_100_000, "cpu": 400_000}
WindowedObjective3D, normal_modes3d, finite_difference_check3d    # verification tools
```
`SensitivityMaps3D` is a list with one float64 array per solver body, at that body's vertices
in order, with metadata: objective, value, coefficients (six), window, dt, steps, wall_seconds,
peak / extra / estimated memory, offloaded, cells, device, cuda_graphs, bodies, points,
normals, symmetry, symmetry_y, period, ensemble, notes, warnings and `summary()`. Units:
`L_ref² · g` in 1/m (ΔJ ≈ Σ sens · δn · dA / L_ref²; `SurfaceData3D.sensitivity`), positive
where pushing the surface outward increases J. The sweep is the 2D method on the 3D solver:
a short window from the developed flow, one reverse pass with ~√N checkpoints (on the host
when the GPU is short), each step replayed as a CUDA graph; meshes refined to edges ≤ ½ cell,
faces turned toward the fluid, values smoothed by ~1.5 cells of surface Laplacian; symmetry
plane runs halve the derivatives and mirror. `attach_sensitivity3d` matches surfaces by name;
when vertex counts differ it uses the nearest map vertex (mirrored for symmetry runs).
`run_cfd` calls it when `sim.cfd.compute_sensitivity` (memory-guarded; warns and skips above
budget). Theory and verification: docs/theory_sensitivity.md section 9.

### 7.9 viz3d and the 3D widget
```python
viz3d.scene.Scene(theme="light")    # VTK off-screen; one OpenGL context per Scene (one Scene per thread)
  add_mesh(mesh | SurfaceData3D | (V, F), scalars=None, *, cmap, range, edges, opacity, ...) -> handle
  add_meshes(meshes, scalars, ...)  # one shared colour range
  add_ground(z=0, extent=None, *, spacing=None); add_slice(SliceData | VolumeData, axis, position,
  field="speed", ...); add_streamlines(VolumeData, seeds="rake"|"plane"|"line"|points, n, color_by=
  "speed", ...); add_isosurface(VolumeData, field="q", value=None, *, color_by=None);
  add_colorbar(title=None, units=None, layer=None); remove_colorbar(); colorbar_spec(); add_axes()
  set_caption(text); set_top_inset(px); set_edges(on); set_theme(theme, *, gradient=None);
  set_focus(bounds); set_camera("iso"|"front"|"side"|"top"|"back"|"bottom", fit=True);
  camera / set_camera_state; fit(); set_parallel(on); orbit(dx, dy); pan(dx, dy);
  zoom(f, at=None) (dolly; `dolly` alias); project(points); layers / remove / clear /
  set_visible / set_scalar_range / set_cmap / scalar_info / layer_points / layer_lines
  render(width, height, *, scale=dpr) -> np.ndarray (H, W, 3) uint8; save_png(path, w, h, *, scale)
  close(); available / error   # no OpenGL context: render() returns the background, no crash
viz3d.backend_info(); viz3d.colors: FIELD_INFO_3D, ZERO_BASED, display_range, colorize,
  lookup_table (the 2D LUTs: a value has the same colour in 2D, 3D and reports), scene_theme
app/widgets/view3d.py: View3D(parent=None, *, scene=None, overlay_toolbar=False) owns a Scene, paints its render()
  at device resolution, re-renders only on change (4 ms batching); left drag orbits
  (turntable, no roll), right/middle or Shift+left drag pans (cursor-exact), wheel zooms toward
  the cursor, double-click or F fits; toolbar() gives a page strip, set_overlay_toolbar(True) a
  floating one (presets, Fit, Edges); refresh(), set_scene, set_view (use this, not
  scene.set_camera: it repaints and emits), fit, set_edges, set_placeholder(title, text),
  render_image; signals cameraChanged, rendered; theme-aware.
```
Conventions: the colour bar shifts the camera's window centre (`vtkCamera.SetWindowCenter`)
so the model is framed left of the bar while the floor continues under it — there is no side
viewport. Default view angle 25°, fit margin 12 %, fit on sampled mesh points. Symmetry-plane
volumes (`mirrored=True`) are reflected for display with v sign-flipped; surfaces arrive
mirrored from the runner; `SliceData` is drawn as given. Slices are unlit so their colours
match the colour bar; solid cells are neutral grey. `AEROLUME_VTK_BACKEND` = "auto" (default)
or "osmesa" (only used when the library is actually present: VTK aborts the process
otherwise). Colour-map defaults are in docs/ui_guidelines.md section 5.

### 7.10 project / export / analysis (3D)
* **Designs.** `designs/<id>.json` + `designs/<id>.npz` (mesh arrays via
  `Design.mesh_arrays()` / `attach_mesh_arrays()`; vertices float64, faces int32 when they fit,
  byte-shuffled, deflate level 1). The JSON records `"mesh_file": {"format", "digest"}` (SHA-1
  of the canonical arrays) and the npz stores the same digest; the npz is written first and
  only when the arrays changed. On open a design whose npz is missing, unreadable or does not
  fit is **skipped with a load warning and its files left untouched**; a pair from different
  saves that still fits loads with a warning and is re-paired by the next save. A plain
  `np.savez(**design.mesh_arrays())` is accepted. 2D designs are unchanged.
* **Schema policy** (`model.SCHEMA_VERSION = 2`, `SCHEMA_VERSION_2D = 1`). A project is
  written as schema 1 while it holds 2D designs only (byte for byte what Aerolume 0.1 wrote)
  and as schema 2 as soon as it holds a 3D design; `Project.schema` is what was opened or last
  written. A 3D design file leaves out the `bodies` key Aerolume 0.1 requires, so 0.1 skips it
  (with a load warning) instead of reading it as an empty 2D design and later writing it back
  without its dimension and meshes; this version recovers design files missing from the index,
  so the design comes back even after 0.1 has saved the project. A newer schema opens with a
  warning that saving may drop what this version does not know.
* **Results**: `result.json` type "cfd3d" | "wing" besides "cfd" | "polar"; `load_result`
  returns `CFD3DResult` / `WingResult`. 3D arrays in arrays.npz are byte-shuffled at deflate
  level 1 (histories float64; surfaces float32 with exact int faces; slices, the volume, the
  extra volumes of 7.13 and their masks float16 with a per-field scale); wing span
  distributions float64. 2D results keep the
  v0.1 `savez_compressed` layout. An unknown result type or encoding raises
  `ProjectCorruptError`.
* **export**: `mesh_stl(meshes, path, *, binary=True, units="m")`, `mesh_obj(meshes, path, *,
  units="m")`, `surface_vtp(result, path)` (point data cp, cf, tau, sensitivity; cell data
  body_index), `volume_vti(volume, path, *, mirror=None)` (mirrors symmetry-plane runs by
  default), `volume_npz`, `mirror_volume`, `cell_centres3`, `slice_csv(result | slice, path, *,
  index=0, axis=None)`, `wing_csv(wing_result, path)`, `wing_span_csv(wing_result, path, *,
  converged_only=False)`; `result_csv` / `history_csv` accept 3D and wing results (a wing's
  history is its sweep table; its result table includes `stall_alpha` / `stall_cl`).
* **analysis**: `wing_metrics(WingResult)` → `WING_METRIC_KEYS` (lift slope, zero-lift angle,
  CL_max with the reached flag, CD_min, best L/D, CD0 and Oswald factor from a parabolic fit
  over the linear attached polar only -- CL(CD_min) to 0.8 CL_max, below the stall onset --
  with the fitted range `oswald_cl_lo` / `oswald_cl_hi` and `oswald_r2` (NaN when R^2 < 0.98),
  e at cruise, Cm0, dCm/dα; NaN where an inviscid sweep cannot say);
  `summarize_history3d(history, length=, speed=)` over `HISTORY_KEYS_3D`;
  `result_dimension`, `comparison_groups`; `compare_rows` builds rows from the baseline's
  dimension and marks other cells `comparable: False`; `conditions_differ` can return "yaw".

### 7.11 app (wave 2)
Geometry page in 3D mode: `View3D` canvas; Import STL/OBJ/PLY (units dialog with the guess),
transform (scale to length, rotate about x/y/z, orientation presets, place on road, centre),
references (kind, length_ref, ref_area with Auto = frontal area, span_ref, moment_ref3,
symmetric with the measured symmetry error), mesh info (triangles, watertight, holes), wheels
(mark a mesh as a wheel; centre/axis/radius auto). New wing: planform editor (section table:
y, chord, x_le, z_le, twist, airfoil) with live 3D preview and planform metrics. Setup in 3D:
yaw, ground, quality, symmetry; wing analysis settings. Simulate: live slice (FieldView) with
slice axis/position + a small 3D overview. Results: summary table with six coefficients,
forces history, Surface (3D Cp / cf / sensitivity), Fields (slices, streamlines, Q iso-surface
in View3D), Wing tab (CL–α, CL–CD, Cm–α, span loading, e, stall-onset estimate). Compare and
Report handle 3D and wing results (report figures from viz3d off-screen renders through
`report.render3d`, section 5.6b: multi-view surface Cp / cf / sensitivity, iso views of
streamlines and Q iso-surfaces, slices with the exact body section, centre-line Cp, 3D force
histories, wing polars, span loading and the planform; the stall-onset estimate is a wing
figure of merit). 3D surface cf of a wall-unresolved LES is labelled qualitative (it uses the
molecular viscosity only). The 2D kind combos offer `model.KINDS_2D` ("wing" is a 3D kind).

### 7.12 Quality bars (3D)
Results and evidence: docs/validation.md section 8 (3D summary), docs/theory_ns3d.md
section 10, docs/validation_wing.md, docs/theory_sensitivity.md section 9.
* ns3d verification: 3D Poisson exact to round-off; projected divergence < 1e-5 (f32);
  3D Taylor–Green order ≈ 2 (central).
* ns3d validation: sphere Re 20/50/100 steady Cd (Re 100: 1.05–1.15 at ≥ 24 cells/D, ≤ 3 %
  blockage; wake length L/D 0.8–0.95), Re 300 mean Cd ≈ 0.66 and St ≈ 0.137 (Johnson & Patel
  1999) as a slow/stretch case; symmetry-plane run equals the full-domain run within 1 %.
* Wing: LLT elliptic CLα within 2 % of 2πAR/(AR+2); VLM within 2.5 % of published
  lifting-surface values (Multhopp circular wing, Falkner rectangular and swept wings); e ≥ 0.98
  for both; viscous polars vs NACA TN 1270 within the bands in docs/validation_wing.md.
* Sensitivity 3D: autograd vs central finite differences on smooth modes (sphere, coarse),
  sign on all modes, < 15 % on dominant ones.
* Performance: GPU f32 256×128×128 ≥ 40 steps/s with CUDA graphs; memory ≤ 6.5 GB at 12 M cells.
* Draft 3D vehicle run (half domain) completes in ≤ ~5 minutes on this GPU.
* Not yet validated: absolute vehicle drag. The wall-unresolved LES reads an Ahmed-type body
  about 2.1–2.2× its wind-tunnel drag at Draft and Standard (0.64 / 0.61 vs 0.29); differences
  between designs are not validated either. A wall treatment for high-Re immersed walls, or a
  validated vehicle case, is needed before absolute 3D vehicle drag can be quoted.


### 7.13 Flow visualisation in the volume (3D smoke, streamlines)

Users asked for 3D smoke and 3D streamlines **in the space around the body**, not only
colour on the surface. Both are first-class, live during runs and in results.

* **Smoke** is a passive scalar `dye` transported by the 3D solver (the 2D `dye` carried to
  3D): `SolverConfig3D.dye: bool = True` and `dye_rake: DyeRake | None` — emitters upstream of
  the body (`kind="rake"` a row of nozzles across y at chosen heights, `"grid"` a y–z grid of
  nozzles, `"plane"` a full inlet plane band), `positions` in metres (wind frame), nozzle
  radius, release on/off pulses optional. Bounded (TVD/upwind) advection, no diffusion beyond
  numerics, clipped to [0, 1], zero inside solids. `FlowSolver3D.field("dye")`.
* **Streamlines** are traced through the 3D velocity (mean for results, instantaneous during a
  run) from a seed rake (line or plane, user-positioned) or seed points; optional surface
  streak lines from `SurfaceData3D.tau`. Tracing runs in VTK (`vtkStreamTracer`) on a
  down-sampled velocity volume.
* `LiveFrame` gains `volume: VolumeData | None` — every few frames a down-sampled block
  (≤ ~128³ cells, float16) with "dye" and "u","v","w" for the live 3D view; `ViewSettings`
  gains `view3d: bool`, `smoke: bool`, `streamlines3d: bool`, `seed_rake` (dict), so the GUI
  can ask for them. Frame payloads stay under ~20 MB.
* `CFD3DResult.volume` stores the mean velocity/cp/q **and** `CFD3DResult.extra_volumes`
  an instantaneous snapshot `VolumeData(label="instant")` with "dye" and "u", "v", "w" (same
  block grid and `mirrored` flag as the mean volume); results can show smoke and streamlines
  without re-running. `result.find_volume("dye")` returns the smoke block,
  `find_volume("u", "mean")` the mean flow. The project store keeps extra blocks like the
  volume (float16, byte-shuffled, `volumes/<i>/` in arrays.npz, layout key
  `"extra_volumes"`); results written before them load with an empty list, and a reader that
  predates them ignores the key. Smoke is a snapshot, not a time average: an averaged smoke
  cloud is a blur without the filaments smoke is looked at for.
* `viz3d.Scene.add_smoke(volume, field="dye", opacity_curve=..., color="auto")` renders smoke
  as a GPU volume (`vtkSmartVolumeMapper`), white-grey smoke on dark backgrounds and a
  dark-grey, low-opacity smoke on light ones; `add_streamlines(volume, seeds=..., color_by=
  "speed"|None, tubes=bool)`; both work off-screen (tests, reports).

## 8. Scripting API ("codable")

A stable, documented public Python API that people, scripts, notebooks, the CLI, the in-app
Script console **and the AI assistant** all use. Lives in `src/aerolume/api.py` and is
re-exported from `aerolume` (lazy imports; `import aerolume as al` must stay fast and
torch-free until a solver is needed).

```python
import aerolume as al
p = al.open_project(path, create=False) | al.new_project(path=None, name=None)   # None: temporary
d = p.add_naca("2412", chord=1.0) | p.add_airfoil("e387.dat") | p.add_template("ahmed", slant_angle=30)
d = p.import_geometry("side.svg", kind="vehicle", length=4.6)                    # 2D outline
d = p.import_mesh("car.stl", units="mm", kind="vehicle", on_road=None)          # 3D mesh
d = p.add_sphere(0.2) | p.add_primitive("box", lx=..)
w = p.add_wing(span=10, root_chord=1.2, tip_chord=0.6, sweep_deg=5, dihedral_deg=0, twist_deg=0,
               airfoil="2412")                                                    # quarter-chord sweep
r = p.run_cfd(d, speed=30 | re=.., alpha=4, yaw=0, quality="draft", ground=None, **settings) -> Result
pol = p.run_polar(d, re=1e6, alpha=(-4, 14, 1), ncrit=9, xtr=1, mach=0)          # airfoil polar
wr = p.run_wing(w, speed=30, alpha=(-4, 12, 1), method="vlm", viscous=True)      # finite wing
r = p.run(sim_name_or_id)                                                        # a saved simulation
sw = p.sweep(d | factory, alpha=[0, 4, 8], speed=[20, 30]) -> Sweep   # full factorial; a factory's own
                                                     # keywords are swept too (geometry studies)
r.cd, r.cl, r.cm, r.ld (arrays for polars / wings), r.stats, r.forces, r.converged, r.re_sim,
r.re_real, r.warnings, r.metrics, r.best(), r.at(alpha), r.summary(), r.table()
r.plot(kind=None, path=None)   # 2D: history cp field geometry sensitivity; 3D: history surface
                               # streamlines smoke vortices slice geometry; polar/wing: polar cp span planform
r.export(path, what=None)      # by suffix: .csv .vtp .vti .npz .json .png .svg .pdf .html
sw.table(), sw.best(key), sw.plot(x, y, by=None, path=None), sw.export("sweep.csv")
p.compare([r1, r2] | sweep, baseline=0) -> Comparison (rows, warnings, table(), plot())
p.report(results, path=None, sections=None, title=None) -> Path      # default: reports folder
p.plan(d, **settings) | al.plan(model_design, **settings) -> Plan (cells, grid, est_seconds, estimate,
                                                                    re_sim, notes, warnings, summary())
al.context(cancel=, progress=, verbose=, show=, project=); al.set_verbose(); al.show(fig); al.plain(obj)
```

* **Contract of every call.** Synchronous; prints nothing unless `verbose=True`; calls that run
  solvers take `progress(fraction, message)` and `cancel` (`threading.Event`, default: the
  `al.context` one, so Stop / Ctrl+C reach every run of a script). A cancelled run keeps its
  partial result, marked "cancelled", and the call raises `al.Cancelled` (`.result`). Errors are
  `al.AerolumeError` subclasses with readable messages: `InputError` (also `ValueError`; unknown
  keywords get a "did you mean"), `NotFoundError` (also `LookupError`; lists what exists),
  `RunError` (also `RuntimeError`), `ProjectFileError` (also `OSError`), `Cancelled`.
* **Recorded like the GUI.** New simulations start from `simulation.defaults.default_simulation`
  (the GUI's defaults: Draft, capped Re for 2D airfoils above 10⁵, vehicles on a moving road) and
  get the GUI's names ("CFD · 30 m/s · α 4°", "Polar · Re 1.0×10⁶", "Wing polar · 20 m/s"),
  NACA designs are the "naca" template. **Reuse:** a design with the same name and geometry is
  returned instead of added again; a simulation with the same design and run settings (design,
  analysis, conditions, ground and CFD / polar / wing settings) that has a complete result is
  returned with `r.reused` (`rerun=True` runs that same simulation again; a stopped or failed
  one is re-run in place). Re-running an edited script computes only what changed.
* **Printed like the pages.** `r.table()`, `r.summary()`, `print(r)`, sweep tables and
  `Comparison` follow the Results / Compare pages: L/D only where `r.lifting` (airfoils,
  wings; a sweep of cars or bluff bodies `best()`s the lowest drag), and on a half-domain 3D run
  the side force, rolling and yawing moments carry `by_symmetry` and print "0 (symmetry)" as the
  report does. The `mesh_3d` example's fallback is a sphere at Re 100 (validated range Re
  20–300; Draft reads CD 1.16 against 1.09 measured), so the first 3D script run gives a
  checkable number.
* **Hosts.** A `Project(store, host=None)` makes every change through an `al.Host`: `call`,
  `add_design`, `update_design`, `remove_design`, `add_simulation`, `update_simulation`,
  `remove_simulation`, `result_saved`, `report_written`, `show`, `defaults`, `check` and
  `run_simulation(store, sim_id, *, progress, cancel) -> (result, "done" | "cancelled") | None`
  (None: the API runs it on the calling thread through
  `simulation.runner.run_simulation(design, sim, project=store, ...)`, the dispatcher the app
  and the CLI share, and records status and result as `jobs.py` does). The default host edits
  the store on the calling thread. `add_wing` builds its planform with
  `geometry3d.tapered_wing` (the CLI's and the planform editor's builder), so a wing quoted by
  span, chords and quarter-chord sweep is the same wing everywhere. In the app,
  `app/scripting.AppHost(state, jobs)` runs every change on the GUI thread through `AppState`
  (pages refresh at once) and every run through the JobManager (one GPU queue, live view, the
  JobManager's save / retry / cancel rules); `open_project` of the bound project's folder
  returns the bound project (`al.context(project=)`), never a second store on the same folder.
* **Engine** (`app/scripting.py`, no widgets): `ScriptEngine(state, jobs=None)` runs source on a
  worker thread with `al`, `project` (an `AppHost`-bound `Project` or None), `log(text,
  level)`, `show(obj)` and `args` bound; the namespace persists between runs until `reset()`.
  Signals `started`, `output(text, stream)`, `progress(fraction, message)`, `figure(png)`,
  `finished(ScriptOutcome)` (status "finished" | "failed" | "stopped", error, script line,
  traceback of the script's own frames, output, figures, sim ids), `stateChanged(bool)`.
  stdout / stderr written on the script's thread are routed to it (others untouched). `stop()`
  sets the cancel event, then after `STOP_GRACE_S` (1.5 s; at once on a second Stop) raises
  `KeyboardInterrupt` in the script's thread (`PyThreadState_SetAsyncExc`, only while user code
  runs, repeated every 2 s). `SNIPPETS` (examples/scripts/*.py hold the same text, a test
  keeps them equal), `scripts_dir(project)` = `<project>/scripts`, `list_scripts(project)`.
* **Console** (`app/widgets/script_console.py`): `ScriptConsole(state, jobs=None, parent=None)`,
  a dock widget (the main window docks it as "Script"): tool strip (Run, Stop, script box of the
  project's scripts, New, Open, Save, Examples menu, ⋯ menu: run selection, save as, reset
  variables, clear output; status and progress at the right), `CodeEditor` (Python highlighting
  from theme tokens, line numbers, auto-indent, error line) and `OutputView` (stdout, stderr,
  log, `line N` links to the editor, inline figures as framed cards at most 640 px wide and
  never shrunk below 400 px to fit a short dock; a click opens `FigureViewer` with Save image…
  (default `<project>/exports`) and Copy). Keys while it has focus: Ctrl+Enter / F5
  run, Ctrl+Shift+Enter run selection, Shift+F5 stop, Ctrl+S save, Ctrl+/ comment. Signals
  `runStarted`, `runFinished(outcome)`, `runningChanged(bool)`, `scriptSaved(path)`; call
  `shutdown()` when the window closes.
* **CLI.** `aerolume-cli script FILE [--project P [--create]] [--figures DIR] [-v] [-- ARGS]`
  binds `project`, `args`, `log`, `show` (PNG files); first Ctrl+C stops the running simulation
  and ends the script, a second aborts; exit 0 / 1 (error or stopped, the script's traceback
  lines printed) / 2 (no file) / `sys.exit(n)`.

docs/scripting.md is the tutorial and reference; examples/scripts/*.py run in all three places.

## 9. AI experiment assistant

An **Assistant** dock where the user states a goal in plain language ("find the angle of
attack with the best L/D", "how does rear slant angle 15–35° change the drag of this car",
"reduce the drag of my wing tip") and the assistant **plans, runs and analyses experiments**
through the scripting API, then reports findings with numbers and caveats. docs/ai_assistant.md
is the user documentation.

* **Package** `src/aerolume/ai/` (no Qt; lazy imports, `import aerolume.ai` is free):
  `config` (`AssistantConfig`: provider "auto" | "claude" | "ollama", models, effort, Ollama
  endpoint / think / num_ctx, `approval_threshold_s` = 120, `max_steps` = 40; `assistant.json` in
  `project.recent.config_dir()` (AEROLUME_CONFIG_DIR aware); the API key in a separate
  `assistant_key` file, 0600 on POSIX, shown only masked, never logged or put in a project),
  `providers` (neutral `ChatSession.send(user_text= | tool_results=, on_event, cancel) -> Turn`,
  `ProviderError(kind, retryable)`), `tools`, `schema`, `agent`, `experiment`, `prompts`,
  `workspace`, `cli` (`python -m aerolume.ai --project P [--provider] [--model] [--approve
  ask|runs|none] GOAL`; `run_experiment(project, goal, ...)` for scripts).
* **Claude** via the official `anthropic` SDK (optional extra `aerolume[ai]`):
  `client.beta.messages.stream(...)` read to `get_final_message()`; model `claude-opus-5-5`,
  `thinking={"type": "adaptive", "display": "summarized"}`, `output_config={"effort": "high"}`,
  `fallbacks="default"` with beta `server-side-fallback-2026-07-01`; one cached system block
  (`cache_control` breakpoint) plus top-level automatic caching, tools sorted by name, constant
  prompts (byte-identical prefix); client tools with `eager_input_streaming: true`. Rules:
  `refusal` -> nothing of the turn is appended or run, a plain message with the category;
  `max_tokens` with a tool call -> never run, the turn is re-issued with twice the budget;
  unparseable tool JSON (`ValueError` from the stream) -> the turn is re-issued, at most twice;
  `pause_turn` -> appended and re-sent; history append-only and assistant content echoed
  unmodified (preserved thinking), except the documented post-fallback omissions (thinking /
  tool_use before the last `fallback` block; only calls after it run); typed SDK exceptions most
  specific first -> `ProviderError`. Behind a custom `ANTHROPIC_BASE_URL` (a gateway) the
  `fallbacks` and `eager_input_streaming` fields are left off. Credentials: the SDK chain
  (`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / `ant auth login`) or the Preferences key.
* **Local models via Ollama** (`http://localhost:11434`, `urllib` only): `POST /api/chat`
  streaming NDJSON with `tools`, `think` (off by default), `options.num_ctx` = 8192 (an 8 B Q4
  model fits an 8 GB GPU), tool results as `role: "tool"` messages in call order; `/api/tags`
  lists installed models whose capabilities include "tools" (Preferences); one retry when the
  runner dies before answering; a context-nearly-full notice; `keep_alive: 0` unloads the model
  before a GPU CFD run; "localhost" is replaced by 127.0.0.1 when Ollama answers there (Windows
  tries IPv6 first: 2 s per request otherwise). Local models get a compact prompt and a focused
  tool subset.
* **Tools** (JSON schemas in the strict-mode subset: every object `additionalProperties: false`,
  no numeric limits — handlers check ranges and say why; `schema.check_schema` enforces it in
  the tests) wrap `aerolume.api`: `project_overview`, `describe_design`, `create_naca_airfoil`,
  `create_template`, `create_wing`, `derive_design` (new designs only; a user's design is never
  edited in place), `estimate_cost`, `run_polar`, `run_wing`, `run_cfd`, `run_sweep`,
  `get_results`, `compare_results`, `make_plot` (PNG in the experiment folder, shown inline),
  `write_report`, `run_python` (kind "code"), `delete_item` (kind "delete"), `record_plan`,
  `record_conclusion`. Results are compact JSON wrapped as `{"data": ...}` (errors
  `{"error": ...}`) with their validity context: Re real and simulated, convergence, standard
  errors and `stderr_reliable`, warnings, `validity` sentences from 6 / 7.12. Polar and wing
  results also carry their confidence facts as data: `optimum_resolution_deg` (half the angle
  step), `validated_quantities` / `not_validated` (wing: CL_alpha, e and CDi validated; profile
  drag, CD, L/D and CL_max not), `cl_max_is_lower_bound` and the angles whose strips went beyond
  their section cl_max. Wing metrics are named for what they are (`span_efficiency_at_best_LD`,
  `oswald_e_parabolic_fit`, `CD0_parabolic_fit`; a local model called e at cruise "the Oswald
  factor"). CFD means are sent rounded to their standard error (two significant figures, as the
  Results page shows them) with `stderr_meaning` ("statistical averaging uncertainty of this run
  only; modelling error ... not included") and CFD `validated_quantities` / `not_validated`.
  Local models get `get_results` compact: metrics, conditions and confidence facts, validity as
  short codes (the prose is in their prompt), the per-angle table only on request.
  `run_polar` / `run_wing` answer from a stored run of the same settings whose angle range covers
  the request (no run, no approval); a new run that differs from a stored one only in its range is
  named after the range. `record_plan` refuses a control that is also a variable, or a speed
  control when the goal compares speeds. `record_conclusion` checks every finding against the
  simulation it cites (value at its written precision, derived quantities such as a change in %
  excepted) and refuses, listing that run's actual values ("ld_max 29.52 at α 3°"); it also
  refuses a ± on a deterministic polar / wing value and a degree uncertainty on a quantity that
  is not an angle. Its own inputs are never number-check sources.
* **Loop** (`agent.Agent`, one experiment = one conversation and one log, on a worker thread):
  every input is coerced losslessly (numeric strings, scalar -> list, null = not given) and
  validated before anything runs; an invalid one goes back as an error result (`{"INVALID_JSON":
  raw}` when it is not even an object); all results of a turn go back together in call order; at
  most `max_steps` tool rounds per request; an answer with numbers before any tool ran gets one
  reminder; an answer to a "how confident / how much to trust" goal without a ±, standard error
  or angle resolution gets one follow-up. Local models (no forced tool choice in Ollama): the loop
  runs `project_overview` itself and passes the result with the goal; the first run call waits
  (answered "record the plan first") until `record_plan`; an answer after results without
  `record_conclusion` gets one follow-up. Each answer is checked, each check asking at most once
  per request: a claimed action ("has been deleted", "I ran") with no successful tool call of
  that kind is sent back, and repeated after that the answer is replaced by a fixed sentence and
  the experiment ends **unverified** (warn glyph, never the green check); a (design, speed / α /
  Re) pair named in the goal that no run or read matched gets "call run_wing on Tapered wing at
  15 m/s"; a number given for one simulation that is a headline value of another (clauses naming
  a speed or design) is sent back and recorded as an attribution check; "confident within ±" a
  standard error next to a not-validated quantity gets one follow-up. A read call that repeats an
  earlier one exactly (nothing ran, was created or deleted since) is answered with a pointer to
  the earlier result instead of the payload. **Approvals**: `run_python` and `delete_item` always,
  runs whose planner estimate exceeds `approval_threshold_s`, and every run whose estimate fails
  (fail closed: "cost unknown"; only a call whose own input is wrong, which fails before anything
  starts, is not asked about). The run card names what will run from the validated input
  ("15 m/s · α −4° to 14°, step 1° (19 angles) · VLM, viscous strips"; CFD: quality, cells,
  Reynolds numbers, device). A declined call is answered "declined by the user". **Stop**
  sets one cancel event watched by the model stream (an Ollama response is closed), the running
  solver (the API's `cancel`; the partial result is kept, marked cancelled) and any approval
  wait. Events to the UI: status, text, thinking, turn, tool_start, tool_progress, approval,
  tool_end, notice, error, done.
* **Experiment log** `<project>/experiments/<YYYYmmdd-HHMMSS-slug>/experiment.json` +
  `experiment.md` (+ `plots/`): goal, provider and model, plan, actions (tool, input, status,
  summary, seconds, simulation ids, files), key results, conclusions and findings, limitations,
  conversation, token usage, and the **number check** — numbers in the conclusions that match no
  number of a tool result, tool input or the goal at their written precision — with the design,
  attribution and action checks under it. Status running | done | unverified | stopped | failed.
  The title is the goal elided to 80 characters (the full goal under "Goal"); a plan without
  variables lists its "Settings", not "Controls (held fixed)". Markdown tables: real header rows,
  values at 4 significant figures, no empty cell (an em dash), and an uncertainty or qualifier
  for every deterministic result. Rewritten atomically after every step; listed in the Project
  tree ("Experiments": open / delete log; unverified with the warn glyph). The log viewer styles
  the Markdown from theme tokens: headings 11 / 10 / 9 pt semibold, tables with a 1 px collapsed
  border in the border token, 22 px rows and a panel-tinted header row.
* **GUI** `app/widgets/assistant_panel.py`: `AssistantPanel(state, jobs)` docked right as
  "Assistant" (toolbar toggle, View ▸ Assistant panel Alt+3, Tools ▸ Experiment assistant
  Ctrl+Shift+A): provider line with New experiment and settings; transcript of plain blocks
  ("You" / "Assistant" captions, markdown with theme-coloured tables, collapsible Reasoning,
  one-line tool rows with status glyph, progress and fold-out input / result, the plan under its
  row, inline figure thumbnails, approval cards — the only framed block — with Run / Delete and
  Decline); goal box with example goals (the first one worded for the selected design), Run /
  Stop, status and elapsed time; a final answer that repeats an earlier one is shown once; an
  answer withheld by the action check is replaced by a warning line, with the original folded
  away under "Withheld answer". In the window,
  project edits go through `app.scripting.AppHost` and runs through the JobManager; approved
  code runs on a `ScriptEngine`. `AssistantSettingsPage` is Preferences ▸ Assistant. The Script
  console is docked bottom as "Script" (Alt+4, Tools ▸ Run script…); the Start page's Help list
  links both guides.
* Honesty: the system prompt makes the assistant plan first (variables, ranges, controls,
  quality, expected cost), report Re_sim vs Re_real, convergence and uncertainty, never invent
  numbers (all numbers must come from tool results), call differences significant only when
  `compare_results` does, and say when a question is outside the solvers' validity (DESIGN 6 /
  7.12, validation docs).
* Prompt-injection hygiene: file contents, mesh names, notes are data in tool results, not
  instructions (the prompt says so and asks the model to point such text out); tools write only
  inside the project and never use the network (besides the model provider).
* Tests: `tests/test_ai_core.py` (schemas, config, logs, tools), `test_ai_providers.py` (Claude
  through the real SDK over a scripted transport; Ollama against an in-process HTTP server),
  `test_ai_agent.py` (the loop on a real project), `test_ai_checks.py` (each honesty check
  against the live failure it came from), `test_ai_panel.py` (the dock in the window),
  `test_ai_live.py` (slow: qwen3:8b runs the NACA 2412 best-L/D experiment end to end).


---

## 10. Placed physics: emitters, heat transfer, fans, jets (development stage v0.3, AeroJAX parity)

Model types: `Emitter`, `ThermalSettings` + `WallThermal`, `HeatSource`, `Fan`, `Port`, on
`Simulation` (`emitters`, `thermal`, `heat_sources`, `fans`, `ports`), body thermal conditions
on `Body` / `Mesh3D` (`thermal`, `thermal_value`), `CFDSettings.domain_mode` ("tunnel" |
"closed") + `domain_box`, `record_diagnostics`, `adaptive_dt`; results gain `nusselt`,
`heat_rate`, `fans_report`, `diagnostics`. Everything defaults to off, so v0.1/v0.2 projects
and results are unchanged. Works in 2D (ns2d) and 3D (ns3d) with the same semantics.

### 10.1 Emitters (flow visualisation)
* Dye: up to 4 independent smoke channels (`dye0..dye3`; emitters map to channels by colour,
  the first is the existing `dye`), bounded TVD advection, zero in solids, released inside the
  emitter footprint at `rate` with pulses (`pulse_period`, `pulse_duty`) and `start`/`stop`.
* Particles: GPU tracer particles released from the emitter footprint (streaklines; pulsed =
  timelines), advected with RK2 + trilinear/bilinear interpolation, recycled; drawn as points
  or streak lines. `frame="wind"` emitters stay in the tunnel when alpha/yaw rotate the body.
* Fields: `dye`, `dye1..3`; frames/results carry them (2D FieldData, 3D VolumeData instant).

### 10.2 Heat transfer
* Energy equation for θ = (T − T_amb)/ΔT_ref: ∂θ/∂t + u·∇θ = ∇·((1/Pe + ν_t/Pr_t) ∇θ) + q,
  Pe = Re·Pr, turbulent Prandtl 0.85 with LES, same advection scheme family as momentum
  (bounded for θ). Boussinesq buoyancy f = Ri·θ·ĝ with Ri = gβΔT_ref L/U² (2D −y, 3D −z).
* ΔT_ref from the largest |T_body − T_amb|, |T_wall − T_amb|, jet temperature, or the heat
  source scale q″L/k; natural convection in a closed domain uses the free-fall velocity
  U_ref = √(gβΔT L) (Re_ref = √Gr) or, for Ra < 10³ conduction-dominated cases, κ/L —
  the planner chooses and states Ra, Pr, Gr, Ri, Pe in `summary()`.
* Bodies: "temperature" = implicit penalisation θ ← (θ + (dt/η) χ θ_s)/(1 + (dt/η) χ) with
  the same calibrated wall offset as momentum; "heat_flux" = a source in the wall skin;
  "adiabatic" = zero flux (penalise toward the local fluid value). Heat-transfer rate from the
  penalisation impulse (as forces), Nusselt Nu = Q L / (k ΔT A) per body and mean.
* Domain walls: `WallThermal` (adiabatic / temperature / heat flux) on each side; inlet at
  t_ambient; outlet zero-gradient. Heat sources: volumetric q over a smooth region.
* Validation (research grade, docs/validation.md §thermal): de Vahl Davis (1983)
  differentially heated square cavity Ra 10³–10⁶ (mean Nu 1.118, 2.243, 4.519, 8.800; u_max,
  v_max and their positions); forced convection from a heated cylinder, Re 20–40 (Nu vs
  Lange et al. 1998 / Dennis et al. 1973, Pr 0.7); Rayleigh–Bénard onset Ra_c ≈ 1708
  (bracketed); 3D: differentially heated cubic cavity (Fusegi et al. 1991 / Tric et al. 2000,
  Ra 10⁴–10⁶ mean Nu) and a heated sphere (Ranz–Marshall / Whitaker correlation).

### 10.3 Fans and ports
* Fan = actuator disc: body force f = (ΔP / thickness) n̂ (+ swirl) over a smooth disc
  (3D) / strip (2D); "thrust" sets ΔP = T/A, "velocity" regulates ΔP to reach the target
  disc-averaged axial speed (PI controller in the solver, no host sync). `fans_report`:
  delivered thrust, volume flow, mean induced velocity, ideal power.
* Ports: a jet drives the velocity in its footprint to `speed` along `direction` (penalised
  like a moving solid with open faces), carries temperature/dye; a vent sucks fluid out with a
  matching sink; closed domains must balance (planner warns, solver corrects the projection's
  compatibility).
* Validation: actuator-disc momentum theory (Froude: induced velocity a, T = 2ρAU²a(1−a), and
  the far-wake contraction) in a tunnel; fan in a closed box conserving momentum; a free jet's
  momentum flux conservation and centreline decay vs the classical self-similar slope.

### 10.4 Live editing and diagnostics (AeroJAX)
* Explore mode: draw / erase obstacles with the mouse (2D brush and polygon; 3D: place
  primitives), move emitters, fans and heaters, change Re with two of (U, ν, Re) locked, pause /
  step / frame-skip / target FPS; changes apply without restarting (`set_bodies`, `set_inflow`,
  `set_emitters`, `set_sources`).
* Diagnostics (`record_diagnostics`): per-step-block RMS divergence, L2 / RMS / max change and
  relative change per field, CFL, dt, steps/s, pressure-solve residual; exportable CSV; a
  **solver trace** that captures one step's stages (advection–diffusion, divergence, pressure
  solve, correction) as fields for inspection; PID-controlled adaptive dt (`adaptive_dt="pid"`).
* Not in this release (deferred, documented): Lattice-Boltzmann solver, neural-operator
  surrogates.

### 10.5 As built (v0.3 integration; these points supersede the text above where they differ)
* **Model → solver mapping** (`simulation/physics.py`, `plan.py`, `plan3d.py`, `physics_run.py`,
  `runner.py`, `runner3d.py`): positions and sizes / L in the frame of the bodies (`frame="body"`
  items turn with α / yaw, `"wind"` items stay in the tunnel); times × U/L; ΔT_ref is the largest of
  |T_body − T_amb|, |T_wall − T_amb|, |T_jet − T_amb| and the flux scale q″L/k; θ = (T − T_amb)/ΔT_ref,
  wall / body flux G = q″L/(k ΔT_ref); Ri = gβΔT_ref L/U² (0 without buoyancy); heat sources
  Q/(ρ c_p U ΔT_ref L) in 2D, P/(k ΔT_ref L) in 3D; fan thrust T/(ρU²L) (2D) or T/(ρU²L²) (3D),
  pressure jump Δp/(ρU²), velocity v/U; one smoke channel (0–3) per distinct emitter colour, more
  than four colours is a planner warning. A closed box with natural convection uses the free-fall
  speed √(gβΔT L) (Re = √(Ra/Pr)). Results are dimensionalised back: heat rate × ρ c_p U ΔT_ref L
  (W/m, 2D) or k ΔT_ref L (W, 3D); fan thrust ρU²L² and flow U L² (3D). *Conducted* 2D heat (bodies,
  walls) converts with `PhysicsScales.conduction_unit(pe_solver)` = k ΔT_ref Pe_solver, which equals
  ρ c_p U ΔT_ref L uncapped; in a capped-Reynolds run it gives Q = Nu k ΔT P / L with the capped
  flow's Nusselt number (the forces' convention), heaters and advected heat keep ρ c_p U ΔT_ref L,
  and the heat balance stays in that one unit so it closes (a note gives the ratio; critic round 5).
  Body-frame box heaters turn with the body: `HeatSourceSpec.direction` (2D rect axis) and
  `HeatSource3D.axes` (3D box rotation, columns = the box axes).
* **Jets and vents** are mass sources carrying their momentum, temperature and smoke, and vents the
  matching sinks (not "penalised like a moving solid"): their mass and momentum fluxes are exact by
  construction (docs/theory_ns3d.md section 17, theory_ns2d.md Part II).
* **Validation as run** (`validation/run_all.py`, docs/validation.md): the actuator disc / strip is
  checked both ways round, as a propeller (pushing the flow, C_T = 4a(1 + a)) and as a turbine
  (holding it back, 4a(1 − a)), each against the confined momentum theory of Garrett & Cummins
  (2007); `run_actuator_disc` defaults to the propeller in both case modules. The free jets run in a
  closed box (the tunnel always has a U = 1 inflow): the plane jet against Bickley (1937) and the
  round jet (case `jet3d`) against Schlichting (1933). Their conservation statement is the momentum
  budget of the jet core, whose residual is held below 1 % (2D) / 0.5 % (3D) of K; the raw drift
  of K (6 % in 2D) is the box's return-flow shear and pressure gradient, which the budget splits out.
  The heated cylinder is compared with Biswas & Sarkar (2009), Jafroudi & Yang (1986) and Apelt &
  Ledwich (1979) as tabulated by Singhal et al. (2023) in the solver validation, and with the
  Lange, Durst & Breuer (1998) correlation in the API's `heated_cylinder_case` (Dennis et al.'s
  values were not accessible); the cubic cavity with Tric et al. (2000), Fusegi et al. (1991) and
  Wakashima & Saitoh (2004).
* **Explore mode** (`simulation/live.py`, `live3d.py`): `LiveFrame.notices` (list of str or None)
  carries the edits since the previous frame that the session could not carry out (a speed change
  in a closed box, an item moved out of the domain, heat added to a session started without it);
  `LiveSession.notices` keeps the last 50. The Simulate page logs them as warnings in Messages.
  `ViewSettings.temperature3d` and `particles3d` (default on) ask 3D frames for the temperature
  volume and the tracer positions. 3D obstacles are placed primitives:
  `post(obstacles={"ops": [{"kind": "sphere" | "box", "center", "radius", "size", "erase"}]})`.
  2D drawing keeps paint-layer checkpoints (`FlowSolver.paint_snapshot` / `restore_paint`, at most
  `live.PAINT_CHECKPOINT_BYTES`), so Undo or an edited stroke restores the last shared checkpoint and
  replays only the rest (`apply_obstacles_2d(..., checkpoints=)`).
* **Long sessions**: `FlowSolver3D.trim_history(keep)` mirrors the 2D solver's and cuts the force,
  placed-physics and diagnostics host logs (with the diagnostics' step list) by time together.
* **Closed boxes are never judged on forces**: their convergence monitors are Nu (heated bodies or
  walls), the rise of the warmest and of the mean fluid above ambient (`t_rise_max`, `t_rise_mean`
  [K], when heat goes in at a set rate and can leave), fan flow and kinetic energy. The run-time
  calibration is keyed by case signature (`closed-fan+jet+heat@64`, ...), each run moves its key by
  at most 1.5x and the planner uses a weighted median per key; explore sessions never record. The
  app loads the calibration file at start-up (`app.main._use_calibration`), as the CLI does.
* **Heaters at rest** (2D solver, `physics_dt_limit`): with heat sources the step also obeys
  `dt <= 1/S_max` and `dt <= (6 cfl h / (|Ri| S_max))^(1/3)` (a parcel heated from rest moves at
  most cfl cells in a step), so a sealed heater room no longer takes an unbounded first step;
  `runner.start_from_rest` stays as the warm start of heat-flux boxes. The 3D solver has the same
  bounds (`Physics3D.dt_rule`, critic round 5) with S_max = κ max|q| over the heaters and heat-flux
  skins summed where they overlap (`ThermalOps.q_max`, `heat_max` refreshed in place by
  `set_reynolds`).
* **Unbalanced closed boxes**: the pressure solve removes the ports' net inflow evenly, and the
  removed fluid takes its heat and smoke along (2D: `src_neg` carries −q/A on every cell; 3D:
  `PortSet.leak` = Q/V in `scalar_sources`); the heat balance books it as heat out of the ports.
* **Buoyancy order (2D and 3D)**: the Boussinesq impulse is added after the body and port
  penalisation, everywhere, and the force on a body is the penalisation impulse minus
  Ri ∫_body θ (2D `core._add_buoyancy` with `ThermalGeometry.vol_u` / `vol_v`; 3D
  `ThermalOps.add_buoyancy` with `vol_w`).
* **Measured ΔT**: when the plan's ΔT is an estimate (a heater or a heat flux), the result states
  ΔT, Gr, Ra and Ri on the measured rise (`physics.result_scales`,
  `physics_run.apply_measured_thermal`); Results, the report and the Setup page (for a stored result
  of the same configuration hash, `physics_run.warnings_after_run`) label the plan's values
  "(planned estimate)". Natural or mixed convection around a heated body carries "regime not
  validated (docs/validation.md §10)" in the plan and the result.
* **Diagnostics history**: every run (app, API, CLI) stores the solver's diagnostics table decimated
  to 200 rows as `result.diagnostics_history` (`runner.attach_diagnostics_history`).
* **Thermal settling**: `physics_run.storage_share(diag)` is the share of the heat put into the
  fluid that the heat balance's storage term still absorbs at the end of a run. Above
  `STORAGE_UNSETTLED` (0.1) the run warns and marks its Nusselt standard error unreliable. The
  convergence verdict judges the monitors only, so the Results header (2D and 3D,
  `pages.results.stored_heat_fact`) states the share from 5 % up next to the verdict
  ("9 % of the heat still being stored") and draws a "Converged" header in the warning colour
  above 10 %.
* **One number format** (`aerolume.core.format`, Qt-free): the PDG round-to-error rule, "steady"
  below 10⁻⁵ of the mean, "≈0" inside the error bar. The Results tables (`app.widgets.kpi`
  re-exports it), the assistant, the MCP server and `api.Result.__str__` (in plain notation,
  `format.plain`) share it. Fluid names are shown through `core.fluid.display_name`
  ("Air, 20 °C"); the stored name keeps the plain "C" that saved projects and PRESETS use.
* **Imported-file provenance**: GUI, CLI and API imports record `api.file_source(path)` (file,
  format, bytes, SHA-256) in `Design.source`; the Mesh inspector shows the format and the checksum.
* **2D rooms with fans**: speeds, disc velocity and volume flow are qualitative and grid-dependent
  (theory_ns2d.md §18.4); the planner, the result, the Fans tab, the API's fan-room note and the
  assistant's validity table say so (`physics.ROOM_2D_FAN_NOTE`).
* **Thread affinity in the GUI**: code that may run on a worker thread never calls
  `QObject.thread()` (shiboken ties the main QThread's wrapper to the caller's, and collecting that
  object deleted the application's main QThread). `AppHost` records the GUI thread with
  `threading.get_ident()`.

## 11. Accounts (optional, Supabase-ready)

* `aerolume.accounts`: provider-agnostic `AccountService` with a **Supabase** backend over its
  REST APIs (Auth/GoTrue, PostgREST, Storage) using the standard library HTTP client — no new
  dependency. Email + password sign-up / sign-in, magic link, password reset, OAuth (GitHub,
  Google) through the system browser with PKCE and a 127.0.0.1 loopback redirect on a one-time
  path `/callback/<nonce>` (a request to any other path is a 404 and never ends the wait, so another
  web page cannot abort a pending sign-in through the predictable port); session
  refresh; sign-out. Profile: display name, affiliation, avatar, units / theme preferences.
* Cloud: upload / download projects (zipped folder → Storage bucket `projects/<user_id>/…`,
  metadata row in `projects` table), list, sync status, conflict detection by content hash.
  An archive larger than one part (90 % of `max_upload_mb`, 45 MiB by default) is stored as
  `<hash>.zip.part001…NNN`, with `projects.part_count` from migration
  `20261030120000_aerolume_project_parts.sql` (schema version 2); every part is uploaded before
  the row changes, the old parts are deleted only after it points at the new ones, and a download
  reassembles and checks the content hash. The project limit is `max_project_mb` (default 1024,
  the free plan's total storage), not the 50 MB per-file limit; the Cloud projects dialog shows the
  exact packed size, parts and largest runs before uploading (`AccountService.upload_preview`);
  share a result as a read-only link (row in `shares`, public storage object with the report
  HTML / renders). Row-level security on every table (owner-only, shares readable by token).
* The Account button (menu-bar corner, `app/widgets/account_menu.py`) is one fixed height in
  every state, at least `theme.MIN_TARGET` (24 px, WCAG 2.2 SC 2.5.8) by 24 px, signed in or
  out, so the menu bar never changes height when the user signs in.
* The app works fully offline with no account (default). Configuration: Supabase URL + anon
  key in Preferences > Account or `AEROLUME_SUPABASE_URL` / `AEROLUME_SUPABASE_ANON_KEY`;
  nothing is configured by default (no project exists yet — the user connects one later).
  Tokens are stored in the user config dir with restrictive permissions, never logged.
* `supabase/migrations/*.sql` (tables, RLS policies, storage buckets and policies) and
  `supabase/README.md` (how to create and link a project) ship in the repo, ready to apply.
* Tests use an in-process mock Supabase server (http.server on 127.0.0.1) implementing the
  subset of endpoints used.

## 12. Packaging ("a proper app")

* `packaging/`: PyInstaller spec (onedir; resources, fonts, examples, VTK, Qt plugins, torch
  CUDA libs collected; console hidden), `build.py` (builds OUTSIDE OneDrive in
  `%USERPROFILE%\aerolume-build\<variant>` from a snapshot of the sources, smoke-tests the built
  folder headless before compiling the installer and verifies the installer file by file;
  `--status` says whether a build matches the tree (CURRENT, STALE, MISSING, and FAILED / UNTESTED
  for a folder whose smoke test failed or was skipped: neither is ever current, and a failed smoke
  test deletes every installer in `<variant>/installer`); `--refresh-docs` re-copies only the files
  copied next to the executable (README, CHANGELOG, docs, validation scripts), re-runs the smoke
  test and recompiles the installer, refusing if any bundled input changed, and records each refresh
  in `build-info.json` under `docs_refreshed`; `--write-measurements` fills the marked blocks of
  docs/install.md and the CHANGELOG from `build-report.json` (`packaging/release_notes.py`), and
  every compile writes `Aerolume-<ver>-<variant>-release-notes.md` (SHA-256, timings) beside the
  installer. Release order: full build, `--write-measurements`, `--refresh-docs`; the whole
  release sequence (tests, `validation/run_all.py --check-docs`, examples, screenshots, build) is
  the checklist in CONTRIBUTING.md.
  `%LOCALAPPDATA%` was dropped because MSIX hosts redirect it; variant `cpu` or `cuda`), Inno Setup
  script (`aerolume.iss`:
  per-user install, Start-menu + optional desktop shortcut, uninstaller, `.aerolume` project
  association via a `project.aerolume` marker file, app icon, version info) compiled with
  Inno Setup 6's `ISCC.exe` (`AEROLUME_ISCC`, else the per-user default
  `%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe`, Program Files or PATH; `buildlib.find_iscc`).
* App polish for a real desktop program: splash screen while torch/VTK load, single instance
  (second launch opens the project in the first), Windows AppUserModelID + taskbar icon,
  crash dialog with a copyable report and log file location, first-run welcome tour, recent
  projects jump list, "Check for updates" (configurable URL, off until a repository exists).

## 13. UI v3 direction (design references + design skills)

Design references: two earlier projects by the maintainer, ClearNotice and Lumen —
ClearNotice (a web app whose stylesheet defines semantic tokens with WCAG contrast enforced
by a unit test in both themes, Atkinson Hyperlegible type, no gradients or glows, status colours
on soft backgrounds) and Lumen (a desktop editor whose layout, assistant and flicker rules give:
dockable panels that collapse, reorder, move between rails, hide and run in solo mode, with a
saved workspace; on/off switches on panel headers for A/B; nothing applied by the assistant
without a click; an MCP bridge so Claude Desktop / Claude Code can drive the app). Design work
is reviewed in structured passes: design critique, accessibility (WCAG 2.1 AA), UX copy,
design-system consistency and a palette contrast audit.
* Tokens: a semantic token set (bg, bg-subtle, surface, surface-2, fg, muted, faint, line,
  line-strong, control, primary(+hover/fg/soft), accent, success/warn/danger(+fg/soft), focus
  ring, plot series, field-view chrome) with a test asserting WCAG contrast (text ≥ 4.5:1,
  controls ≥ 3:1) in both themes, like ClearNotice's `tokens.test.ts`.
* Type: Atkinson Hyperlegible Next (UI) + Atkinson Hyperlegible Mono (numbers, code), bundled
  as OFL fonts in app resources when the files are available; Segoe UI fallback.
* Layout: Lumen-style dock system for every side panel (Project, Properties, Messages,
  Assistant, Script, Flow-vis, Diagnostics), saved workspaces (Default / Analysis / Live /
  Scripting), panel header switches where a panel's effect can be toggled.
* Assistant rules (Lumen): suggestions as cards listing every change old → new, applied only on
  click, values re-validated, one undo step; data sent to a model only with consent; plus an
  `aerolume-mcp` stdio server exposing the scripting API so Claude Desktop / Claude Code can run
  experiments without an API key in the app.

### 13.1 As built (0.1.0 release integration)
* **Honest minimums.** A page's `minimumSizeHint` is what its content needs, because the dock
  manager narrows the side rails to protect it (`DockManager.page_guard_caps`, never below a
  panel's declared minimum). A scroll area reports a small minimum whatever it holds, so the Setup
  inspector sets its panel's minimum to its form's minimum width plus the scroll bar
  (`pages.setup._FormMinWidth`, re-measured on every layout request; floor 320 px, cap 560 px), and
  the placed-item buttons are never narrower than their text.
* **Narrow strips stack.** Setup's Estimate / Planner strip lays its two columns side by side from
  `INFO_STACK_WIDTH` (520 px) and one above the other below it, with a 20 px hysteresis band; at
  1280 x 800 in the Analysis workspace (Project and Assistant open) it is about 370 px wide.
* **Names for assistive technology.** `ToolStrip.add_button` names every button after its text or
  tooltip; `common.unfocus_scroll_areas` takes scroll areas whose content takes focus out of the Tab
  chain and names the ones that stay; `common.name_unnamed_views` gives tables the heading above them
  and tab bars their tab names (run by the main window after each page switch).
* **Script panel share.** `ScriptConsole.set_dock_share(share | None)` is the public way the main
  window hands the Script dock its height share (None: fall back to `SCRIPT_SHARE`).
* **One run-time calibration file.** The app loads and saves the planner's learned run-time
  calibration at start-up (`simulation.plan.use_calibration_file(config_dir() / CALIBRATION_FILE)`,
  in `app.main`), the same file the CLI uses, so "each run refines the estimate" holds across
  sessions and front ends.
* **One heater estimate.** The Setup page's thermal numbers come from
  `simulation.physics.thermal_estimate(design, sim)` (the planner's own scaling) and, once a result
  exists, its warnings from `simulation.physics_run.warnings_after_run(plan, result)`, restated on the
  measured temperature rise; the page keeps no second formula.
* **Heat balance scale.** `analysis.physics.heat_balance(...)["relative"]` is the residual over the
  gross throughput, max(gross in, gross out, |heat rate|), so a closed cavity whose net wall term is
  about zero does not read as a 100 % imbalance.
* **Packaged metadata.** `packaging/buildlib.prune_own_metadata(dist, version, generated=...)`
  replaces the app's `aerolume-<version>.dist-info` with the folder `write_dist_info` generated, so a
  development venv installed at the same version cannot merge its `RECORD` / `direct_url.json` into
  the shipped app.
* **Closed boxes are framed whole.** The Simulate page frames a closed domain by its `domain_box`,
  letterboxed (`pages.results.closed_rect` through `FieldView.set_focus_rect(..., contain=True)`), in
  the preview, a run and Explore alike, as Results does; a tunnel keeps the bodies' near field.
* **One rounding rule for printed results.** The line a finished run writes to Messages
  (`pages.simulate.mean_error_text`, `physics_summary`: Nu, the heat rate and fan thrust first; a closed
  box has no forces) and the first line of a printed `aerolume.api.Result` round each mean to its
  standard error by `core.format.value_error_texts`, as the tables beside them do.
* **A slow local model is not an unreachable one.** The Ollama provider reports a request that timed
  out (`READ_TIMEOUT_S`, 300 s) as "did not answer within", naming model loading and GPU memory, and
  keeps "not reachable" for a refused connection.
