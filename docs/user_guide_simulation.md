# Running simulations

This chapter covers what happens between pressing **Run** and reading a result:
how Aerolume sets up a Navier–Stokes run, what the quality presets mean, how the
Reynolds number is handled, when a run stops, and how far to trust the numbers.
Sections 1-9 describe 2D runs; 3D CFD (imported meshes) is section 10 and finite-wing
analysis section 11 -- they follow the same rules wherever the text does not say
otherwise. Section 13 covers what a CFD run can carry besides the flow: heat transfer and
buoyancy, heated bodies and heaters, smoke and tracer particles, fans, jets and closed
rooms, the solver diagnostics, and editing all of it live in explore mode. The theory behind the solvers is in `theory_ns2d.md` and `theory_ns3d.md`; the
airfoil polar solver is described in `theory_airfoil.md`, the wing solver in
`theory_wing.md`.

## 1. The analyses

| analysis | what it solves | use it for |
|----------|----------------|------------|
| **CFD** (Navier–Stokes) | the full unsteady 2D flow around any set of bodies, on a uniform grid, GPU or CPU | vehicles, bluff bodies, wings in ground effect, separated flow, wakes, flow pictures, sensitivity maps |
| **Polar** (viscous–inviscid) | an XFOIL-class panel method with an integral boundary layer and e^N transition | single airfoils at real flight Reynolds numbers: Cl, Cd, Cm and transition over a range of angles |
| **3D CFD** (section 10) | the unsteady 3D flow around imported meshes (STL / OBJ / PLY, holes allowed) on a uniform grid, GPU (CPU: Draft only) | cars, bikes and riders, drones, any shape: six force and moment coefficients, surface pressure, slices, a flow volume, 3D sensitivity maps |
| **Wing** (section 11) | a finite wing: vortex lattice or nonlinear lifting line on the 2D viscous polars of its sections | lift, induced and profile drag, pitching moment, span loading and efficiency of a planform in seconds |

For an airfoil at Re above about 10⁴ the Polar analysis gives the better drag and
lift: it models the boundary layer and transition, which a 2D grid at that Reynolds
number cannot resolve. CFD is the right tool when the flow separates, when there is
more than one body, or when the ground matters.

## 2. What the planner decides

Every CFD run is planned automatically from the design and its simulation settings.
The plan is shown in **Setup → Estimate** (the table is `SimPlan.summary()`, the lines
under it are the planner's notes and warnings), and again at the top of the run log.
The planner decides, in this order:

1. **Wind frame.** Flow always comes from the left. The angle of attack (or a
   vehicle's pitch) rotates the geometry about the moment reference point, nose up for
   positive angles. On a road the wheels stay on the road and only the body pitches; a
   pitch that would push the body into the road is refused with a message.
2. **Domain.** Its size depends on the kind of design:

   | design | upstream | downstream | height |
   |--------|----------|------------|--------|
   | airfoil | 2–3 chords | 5–8 chords | 4–6 chords, and at most 5 % blockage |
   | vehicle on a road | 1.25–2 lengths | 3–5 lengths | vehicle height above the road ÷ domain height ≤ 8 % (Draft) or 5 % (Standard, Fine) |
   | vehicle in free air | as above | as above | frontal height ÷ domain height ≤ 8 % / 5 % |
   | wing on a road (front-wing template) | laid out like an airfoil, road kept | | |
   | other bodies | 8–12 frontal sizes | 16–24 | 10–16, and at most 5 % blockage |

   *Blockage* matters more in 2D than in a wind tunnel: the air cannot escape around
   the sides, so a low ceiling speeds up the flow over the body and raises both drag
   and lift. A 4.7 m sedan therefore sits under a 17 m (Draft) or 27 m (Standard)
   ceiling. The grid is uniform, so those cells cost the same as the ones near the
   car; that is the price of honest blockage. Bluff bodies get long domains because
   they are cheap and it matters: the Draft cylinder at Re 100 read Cd 1.53 with the
   inlet 4 diameters upstream and 1.43 with it 10 diameters upstream.
3. **Resolution.** Each quality has a preset: cells per chord for airfoils, cells per
   vehicle height for vehicles, cells per frontal size for other bodies. It is then
   raised until thin features get enough cells (the airfoil thickness, a car's ground
   clearance: 4 cells at Draft, 6 at Standard, 8 at Fine), and finally fitted into a
   cell budget for the device by first shortening the domain upstream and downstream
   (never below the blockage limit) and only then coarsening the grid. Grid sizes are
   rounded up to numbers the pressure solver's FFTs handle well.
4. **Reynolds number and flow model** (section 4).
5. **Duration and averaging window** (section 5).
6. **Run-time estimate**: the number of time steps (the duration over the stability
   time step, which depends on how fast the flow around this kind of shape gets) times
   the solver's speed on a grid of this size, plus the cost of the live view (about a
   fifth of the time in the application; the CLI shows the estimate without it) and of
   building the solver and collecting the outputs. Every run measures all three on this
   machine and the next estimates use the measurements made *on grids of a similar
   size* and at a similar resolution, so a small airfoil run does not set the estimate
   for a large car grid. The application and `aerolume-cli` share the measurements in
   `run_time_calibration.json` in the user configuration folder (`%APPDATA%\Aerolume` on
   Windows), and `aerolume-cli bench` adds to it too (set `AEROLUME_CALIBRATION=off` to
   keep an app or CLI session from using it). Explore sessions never record. Scripts and
   other programs opt in with
   `aerolume.simulation.use_calibration_file(path)`.
   With placed physics the time step is sized for the fastest air the run should reach,
   not for the reference speed (section 13.7), and once a run has timed its first blocks
   the **Remaining** time and `plan.est_seconds` follow the measured rate
   (`plan.est_seconds_planned` keeps the planner's own number).

The budgets are fixed per device class and do not depend on measured speed, so the
same design always gets the same grid; comparisons between designs rely on that.

**New simulations start at Draft.** A simulation created from Setup, from the project
tree or with a new study starts at Draft quality, so the first press of **Run** comes
back in seconds to about a minute on a mid-range GPU. Standard and Fine are one click
away in Setup; the estimate next to **Run** shows what they cost before you start.
Airfoils above Re 10⁵ also start in the **capped** Reynolds mode, at Re 2,500 (section 4).
A simulation added to a design that has none copies the fluid, speed, angle, quality and
Reynolds settings of the study's latest CFD run (a design with another chord gets the
Reynolds mode that suits its own Reynolds number), so a second design is compared at the
conditions the study already uses.

## 3. Quality presets

| | Draft | Standard | Fine |
|---|---|---|---|
| Airfoil, cells per chord (GPU / CPU) | 64 / 40 | 128 / 64 | 256 / 96 |
| Vehicle, cells per vehicle height (GPU / CPU) | 36 / 24 | 48 / 32 | 64 / 40 |
| Other bodies, cells per frontal size (GPU / CPU) | 24 / 16 | 40 / 24 | 64 / 32 |
| Cells across a thin feature (clearance, thickness) | 4 | 6 | 8 |
| Cell budget, GPU | 0.25 M (vehicles 0.4 M) | 1 M | 2 M |
| Cell budget, CPU | 0.1 M | 0.2 M | 0.3 M |
| Vehicle blockage limit | 8 % | 5 % | 5 % |
| Airfoil run length (L/U; most runs stop early) | 40, averaging from 8 | 50, from 12 | 60, from 16 |
| Airfoil domain, chords ahead / behind / tall (before the budget) | 4 / 5 / 6 | 4 / 6 / 6 | 5 / 8 / 6 |
| Vehicle run length (convective times L/U) | 30, averaging from 6 | 36, from 8 | 40, from 10 |
| Other bodies, run length (frontal sizes / U) | 60, from 30 (at most 60 L/U) | 80, from 40 | 120, from 50 |
| Early-stop target: mean Cd / mean Cl known to | ±5 % / ±10 % | ±3 % / ±6 % | ±2 % / ±4 % |

Measured on the development machine (RTX 4060 Ti 8 GB, float32). The GPU was shared
with other programs during most of these runs, so wall times are upper bounds:

| case | grid | wall time | result |
|------|------|-----------|--------|
| NACA 0012, Re 1 000, 5°, Draft | 640 × 384 | 6.5 s (stopped early at t = 18.4 of 40; steady) | Cd 0.137, Cl 0.179 |
| same, with the Cd sensitivity map | | + 14.9 s | map on 1 body |
| NACA 2412, Re 10⁶, 4°, Draft, capped (Re 2 500) | 640 × 384 | 10 s (stopped early at t = 30.6 of 40), plus about 2 s for the polar cross-check | Cd 0.106, Cl 0.150 (viscous polar at Re 10⁶: 0.0070, 0.72) |
| same, real Re (LES) | 640 × 384 | 13 s | Cd 0.093, Cl 0.16 |
| Cylinder, Re 100, Draft | 540 × 450 (23 per D) | 8.4 s (stopped early, converged, at t = 54 of 60) | Cd 1.467, St 0.170 (validation at 24 per D: 1.429, 0.169) |
| Cylinder 1 m, 5 m/s (Re 3.4 × 10⁵, LES), Draft | 540 × 450 | 24 s (60 D/U) | Cd 1.85 ± 0.02, St 0.21; Cl not converged (rms 1.36 about a zero mean). Standard (1 250 × 800) is estimated at 5 min |
| Sedan template, 30 m/s, moving road, Draft | 720 × 450 (126 per length, 7.7 % blockage, 4 cells under the car) | 95–190 s on a shared GPU; ~70 s of solver time on an idle one | four runs: Cd 0.57–0.63 (± 0.02–0.05), Cl 2.5–2.9 (± 0.12–0.29) |
| same, Standard | 972 × 1 080 (188 per length, 4.8 % blockage, 6 cells under the car) | 10.7 min on a shared GPU | Cd 0.716 ± 0.017, Cl 2.16 ± 0.12 (run extended to t = 45 after a late transient) |

The solver alone runs the Draft sedan grid at ~430 steps per second on an idle GPU
(about 29 000 steps for the 30 L/U run). In the application the live view costs up
to a fifth of that. Any other program using the GPU (games, video encoders, other
simulations) slows a run down, sometimes by a factor of two to three; the run-time
estimate adapts after the first run. Measured against the estimate on an RTX 4060
Ti (October 2026): the Draft sedan was estimated at 98 s without and 121 s with a
live view and took 95–102 s and 106–126 s; the Standard cylinder (1.0 M cells) is
estimated at 5.8 min with the live view, close to the ~5.7 min it took in the
application. Runs that stop early because they converged (most airfoil runs) finish
well before the estimate, which is for the full duration.

Note how much the sedan's numbers move between Draft and Standard (Cd +20 %, Cl
−25 %): finer cells, a better resolved underbody channel and less blockage all change
a 2D LES of a car. Compare designs at the same quality. The domain matters as much:
with the road ahead and the wake behind shortened by 0.6 L to fit 0.25 M cells, two
Draft sedan runs read Cd 0.91–0.95 and Cl 1.4–1.5 instead of 0.6 and 2.5. That is
why vehicles keep a 0.4 M-cell Draft budget even though it costs them about 75 s.

On a CPU, vehicles are slow (Draft sedan: about 8 minutes) and the budget forces a
coarser grid; the plan says so in its warnings.

## 4. Reynolds number: real or capped

`Re = ρ U L / μ` with L the chord (airfoils) or the vehicle length. Two modes:

* **Real** (the default, except for new airfoil simulations above Re 10⁵): the
  simulation runs at the real Reynolds number. When the grid
  resolves the laminar boundary layer (at least 4 cells across its thickness at the
  trailing edge) the flow is computed without a turbulence model (*laminar*, a direct
  simulation of the 2D flow). When it does not — a car at 30 m/s has Re ≈ 10⁷ and a
  boundary layer about 0.2 cells thick — the planner switches to a large-eddy
  simulation: the WALE sub-grid model adds dissipation where the grid cannot follow
  the smallest eddies, and third-order upwind advection keeps the run stable. There is
  no wall model and no transition model, so friction drag and anything that depends on
  the exact separation point of a thin boundary layer are not quantitative.
* **Capped**: the simulation runs at the cap (or at the real Reynolds number when that is
  lower), as if the fluid were that many times more viscous. Flow and coefficients are
  those of the capped flow; forces are scaled with the real speed and density. Use it for
  resolved, laminar flow pictures and for studies where you want every run on the same
  physics regardless of speed. Heat transfer follows the same rule: the Nusselt number is
  the capped flow's (its Péclet number is Re_cap × Pr), and heat rates are that Nusselt
  number with the real conductivity, Q = Nu k ΔT P / L. (Until the round-5 fix a capped 2D
  run reported the more conductive simulated fluid's heat rate, Re_real/Re_cap times too
  large: 1814 instead of 34.15 W/m for a NACA 0012 at Re 1.3 × 10⁵ capped at 2,500.) The
  heat balance stays the simulated flow's own, so it closes; its body and wall terms are
  Re_real/Re_cap times the heat rates, and the result's notes say so.

  The cap belongs to the simulation and is set when the simulation is made: Re 2,500 for an
  airfoil, the highest Reynolds number that *every* quality resolves on every device (Draft
  on a CPU has 40 cells per chord: (5 × 40 / 4)² = 2,500). So stepping from Draft to Standard
  refines the grid of the same flow, and a CPU and a GPU simulate the same case. (A cap left
  to the grid — *Auto* in Setup, or projects saved before the cap was stored — followed the
  grid: the same NACA 2412 case simulated Re 6,400 at Draft and 25,600 at Standard on a
  GPU and 2,500 on a CPU, so a "grid study" compared different flows. Opening such a project
  pins the cap to the Reynolds number of its stored result, and a plan without a cap says that
  its Reynolds number now depends on the grid.) A cap above what the grid resolves gets the
  large-eddy model and a warning.

Why airfoils above Re 10⁵ start capped: no grid this tool can afford resolves an
airfoil boundary layer at Re 10⁶, and without a wall or transition model the "real" run
is not closer to reality — NACA 2412 at Re 10⁶ and 4° read Cl 0.16 and Cd 0.097 at
Draft, where the viscous polar (and the wind tunnel) give about 0.72 and 0.007. A capped
run is at least a resolved, repeatable flow that says what it is.

**The polar cross-check.** For a single airfoil at Re 5 × 10⁴ or more, every CFD result
is also analysed with the viscous polar solver at the same real Reynolds number and
angle (about 2 s at the end of the run). When the two disagree (Cl by more than 20 %,
Cd by more than a factor 1.5) the result's first warning puts them side by side —
"Viscous polar at Re 1.00×10⁶, α 4°, gives Cl 0.718, Cd 0.00697 (this CFD run: Cl
0.150, Cd 0.106)" — and says which to quote. When they agree it is a note.

The Setup table shows both numbers ("Real Reynolds number", "Simulated Reynolds
number … (capped)") and the result keeps them.

The flow model can be forced in Setup (**Laminar** or **LES**); the planner warns when
laminar flow is forced on a grid that cannot resolve it.

## 5. When a run stops, and what "converged" means

A run has two phases. **Developing the flow** lets the start-up transient (the
impulsive start, the first vortices) wash out. **Averaging** accumulates mean fields
and the force statistics. About every 1.5 seconds the runner judges the force
history:

* The averaging window starts where the start-up transient really ends, found from the
  data (MSER-5 on Cd and Cl), never before the planned averaging start.
* The mean of Cd must be known to the quality's target: its **standard error** — from
  batch means over whole shedding periods or integral time scales, so autocorrelation
  is accounted for — must be below 5 / 3 / 2 % of the mean. Cl must be known to twice
  that relative precision, or to the same absolute precision as Cd when the body
  makes no lift.
* The two halves of the window must agree: a difference that is larger than the target
  *and* larger than three standard errors is a drift, not noise.
* The fluctuation amplitude must have settled. In a laminar flow (deterministic) that
  means within ±25 % between the halves; a large-eddy simulation is chaotic and its rms
  has its own sampling error, so there the change must also exceed three standard
  errors. Two refinements keep this rule honest at both ends. A fluctuation that is
  *decaying* and whose peak (√2 × rms) fits inside the target cannot move the mean by more
  than the target, so its standard error alone decides and it does not count as shedding:
  the heated cylinder at Re 40 (steady, below Re ≈ 47) used to end "not converged" over a
  start-up wiggle on its lift of rms 0.0099 → 0.0063 against a ±0.083 target. A *growing*
  fluctuation is never waved through, however small, because that is how vortex shedding
  sets in.
* A periodic flow must have at least four averaged cycles.

When all of this holds, the run stops early and the result is marked **converged**.
If the drag is still drifting early in the window, the field averages restart once and
the run is extended by up to 50 % (not when you set the duration yourself). A run that
reaches its end without meeting the target is reported as **not converged** with the
uncertainty it did reach. When the window held too few independent samples to measure
that uncertainty (fewer than about 20, or under 50 for noise-like signals), the warning says
the error bars are rough estimates: they are then a deliberately cautious figure (it also
covers start-up transient left in the window), the report marks them with ~ and Compare
does not call a difference significant on them. A Draft NACA 4412 at Re 1 000 stopped at
20 L/U used to be reported as "usable with the error bars shown" while its means were 4.2 of
those error bars from the converged values.

What to expect: laminar flows (low-Re airfoils, the cylinder) settle quickly and
usually stop early. 2D large-eddy simulations of cars do not: the Draft sedan's drag
fluctuates by ±40 % of its mean with an integral time scale of about 0.26 L/U, and
big vortex events modulate the lift for the first 15 convective times. Its mean drag
is known to about ±3–8 % after 30 L/U, so a Draft car often ends just short of its
±5 % target and is reported as not converged — with its error bars. Getting to ±1 %
would need several hundred L/U; set a longer **Duration** if a comparison needs it.

The lift target is twice the drag target (±10 / 6 / 4 %): in separated 2D flows the
lift fluctuates far more than the drag, and the early stop is about the drag. When the mean
lift is near zero (a symmetric body), the target is the absolute floor tied to the drag
instead, and the result says so: "targets ±5 % and ±0.083 (absolute: the mean is near
zero)" rather than a relative error of a zero mean.

**Stopping a run.** **Stop** (or Ctrl+C in the command line) ends the run at the next
block of steps. The partial result is kept and saved, with a warning that it is
averaged over a shorter window (or, if averaging had not started, that it is a rough
indication from the last half of the record). A stopped *Run again* of a simulation that
already had a complete result keeps that result instead (the same when the window closes
or another project opens during the re-run). If the project folder refuses to save a result
(OneDrive or a virus scanner holding the file, a full disk), the result is still shown,
marked *not saved yet*, and saved again in the background; Simulation ▸ Save unsaved results
tries again at once, and quitting asks first.

## 6. Reading results honestly

* **Error bars.** Every mean is shown with its standard error. Two designs differ
  meaningfully when their difference is larger than about twice the combined error.
* **2D is not 3D.** A 2D vehicle is an infinitely wide side profile. Its drag and lift
  coefficients (per frontal height) are well above wind-tunnel values for the same
  car: the sedan template gives Cd ≈ 0.6 and Cl ≈ 2.8 at Draft (0.72 and 2.2 at
  Standard), where a real saloon measures about 0.3 and 0.1–0.4. The wake sheds as a regular 2D vortex street, and the
  underbody is a channel rather than a gap between four wheels. Rank and compare
  designs simulated the same way; do not quote the numbers as a car's Cd.
* **Wheels.** In 2D a wheel is an infinitely long cylinder. A wheel touching the road
  seals the underbody completely and makes the lift meaningless; the planner warns.
  Leave wheels off, or lift them by a small wheel gap. On a moving road, rotating wheels
  roll with the belt (ω = U / r, counter-clockwise for flow from the left).
* **Ground clearance.** At Draft the gap under a car is 4 cells: air passes (on a
  moving road the Draft sedan's underbody air moves at about 0.9 U on average), but
  half of those cells are the diffuse wall, so the flow under the car and the suction
  that sets the lift are under-resolved. Below about 8 cells (Fine) the plan and the
  result lead with **"Underbody gap under-resolved"**: compare lift between designs run
  at the same quality, and use Fine before quoting it.
* **Blockage.** Shown in the Setup table. Above the preset's limit (only possible with
  a hand-set domain height) drag and lift read high; the plan warns.
* **Sharp edges.** Bodies are immersed in a uniform grid through a wall about one cell
  thick, and sharp corners converge more slowly than rounded shapes: the square cylinder
  at Re 100 reads C<sub>d</sub> 5.9 % high at 32 cells per side and 3.7 % at 48, while its
  Strouhal number is right at both ([validation report](validation.md), section 3).
  Compare sharp-edged designs at the same quality.
* **Strouhal number** is reported when the lift oscillates with a clear dominant
  frequency, based on the chord (airfoils) or the frontal height (vehicles, bodies). In
  a chaotic LES wake it describes the dominant shedding and varies between runs. It is
  shown as "—" when there is nothing to measure: the oscillation is negligible (under
  2 % of the lift or drag) or has no clear spectral line, fewer than four cycles were
  averaged, or the run is an airfoil whose boundary layers the grid does not resolve
  (the "Lift and drag are not reliable" warning). In that last case the lift does
  oscillate -- a NACA 2412 at Re 10⁶ and 4° at Draft rings at a chord-based 1.66 --
  but the oscillation belongs to the under-resolved flow, not to the attached airfoil;
  the result's notes say which reason applied.
* **Surface cp and cf** come from probes 1.5 and 2.5 cells off the wall. cp is reliable
  wherever the flow is resolved; cf is a lower bound when the boundary layer is thinner
  than about three cells (every high-Re run), so read its sign (separation,
  reattachment), not its size. Forces never come from these curves; they come from the
  solver's momentum balance.
* **Per-body coefficients** (several bodies) are means of samples taken through the
  averaging window; the totals use every time step.
* **Sensitivity map** (Setup → Compute sensitivity): after the run, a reverse sweep
  through a short window of the developed flow gives dJ/dn for J = Cd, Cl, L/D or Cm on
  every surface point. Positive values: pushing the surface outward there increases J.
  In an unsteady flow it is the sensitivity of a short time average; read it as a
  direction, not a prediction. If the map cannot be computed (memory, a stopped run),
  the forces are kept and a warning says why.

## 7. Explore mode

**Simulate → Explore** runs a smaller grid in real time while you change the angle,
the speed, the road and the displayed field. It covers every pitch the slider can
reach (±25°, ±5° on a road, where the pitch is also limited so the body stays above
the road), allows up to 10 % blockage and uses float32. Speed changes only change the
Reynolds number, so the flow keeps developing instead of restarting; switching the
road between moving and fixed rebuilds the solver with the current flow. Explore
results are for looking, not for numbers: use **Run** for those. With placed physics
explore also moves emitters, fans, heaters and jets, draws obstacles, holds the Reynolds
number while ν or U changes, pauses and steps, and captures a solver step (section 13.8).

## 8. Command line

```
aerolume-cli doctor                          versions, devices and a 1-second solver check
aerolume-cli bench --device cuda             solver throughput on this machine
aerolume-cli polar --naca 2412 --re 1e6 --alpha -4:14:1 --csv polar.csv
aerolume-cli quick --template sedan --speed "108 km/h" --quality draft
aerolume-cli quick --naca 0012 --speed 0.0146 --alpha 5 --plan-only -v
aerolume-cli quick --template sedan --save MyCar --report car.pdf --sensitivity cd
aerolume-cli run MyCar.aerolume "Sedan quick"  run a saved simulation, store its result
aerolume-cli validate --full                 verification and validation subset
```

`quick` defaults to Draft and `--re-mode auto` (capped for airfoils above Re 10⁵, real
otherwise, the same choice the app makes); it also accepts `--cells-per-ref`,
`--duration`, `--re-mode real|capped`, `--re-cap 1e4`,
`--no-ground` and `--fixed-ground`, and the placed-physics options `--heated`, `--ambient`,
`--no-buoyancy`, `--fan` and `--closed` (section 13.10); `aerolume-cli cavity` runs the
heated-cavity benchmark. Exit codes: 0 success, 1 failed or stopped, 2 bad arguments.

## 9. When something goes wrong

| message | meaning and remedy |
|---------|--------------------|
| *The GPU ran out of memory for the N-cell grid* | choose a lower quality or close other programs using the GPU |
| *The GPU could not run the solver (…)* | the CUDA driver failed; close other GPU programs, restart Aerolume, or choose the CPU under Setup → Device |
| *The flow solution became unstable at t = …* | the grid is too coarse for the flow: a higher quality, LES, or the capped Reynolds mode |
| *At N° the geometry reaches … below the road* | reduce the pitch or raise the vehicle |
| *The … gap … is only N cells wide: the solver treats it as closed* | a finer quality or a wider gap |
| *Not converged to the ±N % target* | the averages carry the error bars shown (rough estimates when the warning says so); run longer for finer comparisons |
| *This grid of N cells needs about X GB* | the cells per reference length set by hand do not fit the device's memory: lower them |
| *No Reynolds cap is set* | the capped mode follows the grid: set the cap in Setup ▸ Solver so every quality simulates the same flow |
| the run is much slower than estimated | another program is using the GPU; the estimate adapts after the first run, and **Remaining** follows the measured rate from the second block on |
| *Buoyancy dominates this run (Ri = …)* | the plume outruns the free stream: read the forces in newtons, or the coefficients on the buoyant scale in the result notes (section 13.1) |
| *The design is marked mirror-symmetric, but its mirror image … is off by …* (3D) | the shape was moved or edited after import: both halves are simulated; centre it on y = 0 to get the symmetry plane back |
| *The wing uses an airfoil from the project … needs the project* (3D) | lofting a wing with project airfoils for CFD needs the open project: run it from the application |
| *The shape-sensitivity map was skipped: …* (3D) | the grid is above the map's budget or the GPU memory is short: Draft quality or a symmetry plane |
| *Airfoil 'design:…': there is no design with id …* (wing) | the airfoil design of a section was deleted: choose another airfoil for it |
| *Nothing drives the flow in this closed box* | a room needs a fan, a jet, a heated wall or body (heat transfer on) or a moving floor; or switch the domain back to a wind tunnel |
| *A closed domain without bodies needs its box* | set Domain ▸ Extent to the room's walls |
| *The jets supply … and the vents remove …: a closed box must balance* | match the openings or speeds; otherwise the solver removes the difference evenly over the whole room |
| *The grid gives N cells across the thermal boundary layer* | heat rates and Nusselt numbers are under-resolved and can be off in either direction: a finer quality |
| *Custom fluid: its thermal properties are unknown* | type the Prandtl number, conductivity, specific heat and expansion coefficient under Heat transfer |
| *β·ΔT = …: … the Boussinesq approximation loses accuracy* | the temperature differences are too large for constant-density buoyancy; the buoyancy is approximate |
| *Fan “…” spans N grid cells* | the thrust is delivered, its induced velocity and flow are indicative: a finer grid or a larger fan |
| *“…” is outside the domain, so it has no effect* | move the item inside the dashed frame in Setup |

## 10. 3D CFD

A design is 3D when it was made from meshes (Geometry ▸ Import STL / OBJ / PLY, or
`aerolume-cli quick3d`) or from a wing planform. The same planner, runner and live view
handle it; only the geometry, the domain and the outputs change. Frame (as in a wind
tunnel): x downstream, y to the left, z up, the road at z = 0.

### 10.1 What the planner decides

1. **Attitude.** Yaw turns the body about the vertical axis through the moment reference
   (positive yaw: the wind comes from the vehicle's left), then pitch turns it nose up.
   On a road the wheels only yaw; a pitch that pushes the body into the road is refused.
2. **Symmetry plane.** A design whose mirror image across y = 0 matches it (measured at
   import, and again by the planner) is simulated in the half domain y ≥ 0 at zero yaw:
   half the cells, the same answer (measured on a sphere at Re 100 on the same grid: Cd
   1.1562851 with the plane, 1.1562861 without, in 15.5 s instead of 32.6 s). Forces and
   moments are reported for the whole body; side force, roll and yaw are zero by
   construction, and wake motions that break the symmetry are suppressed. Yaw, a design
   that is not symmetric, or Setup ▸ Symmetry ▸ Full simulates both halves.
3. **Domain.** Vehicles get 1–1.5 lengths of road ahead and 2–3 behind, bluff bodies
   5–6 frontal sizes ahead and 10–12 behind, wings 3–5 mean chords ahead and 5–8 behind.
   The cross-section keeps the same clearance above and beside the body, sized so that the
   frontal area is at most a set share of the tunnel cross-section (*blockage*) and the
   body fills at most a set share of the tunnel's width and of its height:

   | | Draft | Standard | Fine |
   |---|---|---|---|
   | blockage, vehicles | 5 % | 4 % | 3 % |
   | blockage, bluff bodies and wings | 3 % | 2 % | 2 % |
   | body width or height / tunnel, vehicles | 50 % | 45 % | 40 % |
   | ... bluff bodies | 40 % | 35 % | 30 % |
   | ... wings (span / tunnel width) | 60 % | 55 % | 50 % |

   3D blockage costs far less than 2D blockage (the air goes around the sides), but at low
   Reynolds numbers it is still measurable: a sphere at Re 100 reads about 3–4 % more drag in
   a 2.85 % box than unconfined (theory_ns3d.md 10). The Setup table shows the blockage and
   both shares.
4. **Resolution and budget.** Cells per vehicle height (24 / 32 / 40 on a GPU), per
   frontal size of a bluff body (20 / 24 / 32) or per mean chord of a wing (16 / 24 / 32),
   raised (by at most 1.5×, 1.25× for wings) so a thin feature -- the ground clearance, a
   wing's thickness, a thin part -- spans 3 / 4 / 6 cells, then fitted into the cell budget
   by first shortening the domain ahead and behind (bluff bodies and wings may also narrow
   it, up to 1.6 times the blockage limit, with a warning) and only then coarsening:

   | budget | Draft | Standard | Fine |
   |---|---|---|---|
   | GPU (8 GB) | 2 M cells | 6 M | 14 M |
   | CPU | 0.15 M | 0.3 M | 0.5 M |

   The grid is uniform, so cells far from the body cost as much as those next to it; that is
   why the budgets bind and why a CPU is practical for Draft only (its grids are coarse:
   about 9 cells per diameter for a sphere). A grid that would not fit the device's memory
   is refused with a message.
5. **Reynolds number and flow model**: the 2D rules (section 4). A car at 30 m/s has
   Re ≈ 10⁷: the boundary layers are far thinner than a cell, so the run is a
   wall-unresolved large-eddy simulation (WALE) with no wall or transition model.
6. **Duration**: vehicles 20 / 25 / 30 L/U (averaging from 5 / 6 / 8), bluff bodies
   40 / 60 / 80 frontal sizes / U (from half of that), wings 20 / 25 / 30 L/U, with the
   2D early-stop rules (section 5); the estimate uses throughput measured for the 3D solver
   on the reference GPU and is corrected by every run on this machine.

### 10.2 Measured on the development machine

RTX 4060 Ti 8 GB, float32. Other programs using the GPU slow every run down.

| case | grid | wall time | result |
|------|------|-----------|--------|
| Sphere, Re 100, Draft (half domain, 20 cells/D, 2.9 % blockage) | 320 × 52 × 104 (1.73 M) | 15.9 s, converged, stopped at t = 26 of 40 | Cd 1.156 (the solver's grid study at 2.85 % blockage: 1.172 at 16 cells/D, 1.148 at 24; unconfined sphere 1.087) |
| same, full domain, same grid | 320 × 104 × 104 (3.46 M) | 32.6 s | Cd 1.156 (identical to 9 × 10⁻⁷) |
| Ahmed-type body (25° slant, 1.044 m, moving road, 40 m/s, Re 2.9 × 10⁶), Draft, half domain | 300 × 76 × 84 (1.92 M, 24 cells per height) | 206 s on a GPU other programs were using (estimate for an idle GPU: 117 s), converged at t = 19 of 20 | Cd 0.637 ± 0.005, Cl 0.075 ± 0.019 |
| same, Standard, 12 L/U window | 452 × 108 × 120 (5.86 M, 32 cells per height) | 359 s on a GPU other programs were using (estimate for an idle GPU: 322 s), converged at t = 8.6 | Cd 0.611 ± 0.006, Cl −0.059 ± 0.011 |

**How far to trust a 3D car's numbers.** The Ahmed body's wind-tunnel drag at 25° is
about Cd 0.29 (Ahmed et al. 1984: a fixed floor, stilts, Re 4.3 × 10⁶). The Draft run's
0.64 is more than twice that, and Standard (a third more cells per vehicle height) moves
it only to 0.61, while the small lift changes sign (Draft +0.08, Standard −0.06). The
differences in set-up (moving road, no stilts, Re 2.9 × 10⁶, 4.5 % blockage) are far too
small to explain a factor of two. The likely cause -- not verified here -- is the wall: the
boundary layers are far thinner than a cell and there is no wall model, so the diffuse
wall cannot hold attached flow over the rounded front and the slant the way a real
turbulent boundary layer does, separation is exaggerated and the pressure drag is too
high; refining the grid by a third barely changes it. Until a wall model or a validated
car case exists, read a 3D car's absolute Cd and Cl as indicative only (the planner says
so on every vehicle run) and use the runs to compare designs simulated at the same quality
and conditions -- with the caution that those differences have not been validated against
measurements either.

### 10.3 What a 3D result holds

* **Six coefficients** with error bars: CD (drag), CS (side force), CL (lift, negative =
  downforce), C_roll, Cm (pitch, + nose up) and Cn (yaw), on the reference area (frontal
  area for vehicles and bodies, planform area for wings), the reference length for pitch
  and the span / width for roll and yaw, moments about the moment reference; plus forces
  [N] and moments [N m].
* **Surface** cp, cf and the wall-shear vector at every vertex of *your* mesh (vertex
  order kept, so exports and the 3D view line up with the imported file). With a symmetry
  plane the vertices of the missing half take their mirror image's values. As in 2D, cp is
  usable, cf a lower bound wherever the boundary layer is thinner than a few cells.
* **Three mean slices**: the symmetry (or centre) plane, a horizontal plane at mid-height
  and a cross-flow plane just behind the body, mirrored to the full width.
* **A flow volume** (u, v, w, cp, Q-criterion and the solid fraction, at most 2 M cells,
  block-averaged from the solver grid when it is larger) for streamlines and vortex
  iso-surfaces in the 3D viewer.
* **Per-part coefficients** when the design has several parts. With a symmetry plane a
  part that lies wholly on one side and its mirror image share their force half and half.
* **Sensitivity map** (Setup ▸ Compute sensitivity): dJ/dn per vertex, on grids up to the
  sensitivity module's budget (2.1 M cells on a GPU, 0.4 M on a CPU); larger grids skip it
  with a warning and keep the forces.

The live view during a run, and Explore, show a **slice** (Simulate ▸ Slice: x, y or z and
its position) with the mesh's cross-section drawn on it.

### 10.4 Explore in 3D

Explore plans a small full-domain grid (about 0.4 M cells on a GPU: yaw needs both halves)
that covers pitch ±10° (±3° on a road) and yaw ±15°; changing either rebuilds the solid
masks in place (measured: one 0.3 s frame for the Ahmed-type body) and the flow keeps
developing. On the RTX 4060 Ti it ran at 30 frames per second, about 550 solver steps per
second. Speed changes change the Reynolds number only.

## 11. Finite wings

A wing design is a planform: sections (span station, chord, leading-edge position, twist,
airfoil) from the root to the tip, mirrored for a symmetric wing. The airfoil is a NACA
code or a 2D airfoil design of the same project ("design:<id>", resolved from the open
project). A new simulation of a wing starts as a **Wing** analysis:

* **Vortex lattice** (default): ring vortices on the cambered mean surface; any planform,
  sweep, dihedral and sideslip; induced drag in the Trefftz plane. Its lift is linear.
* **Lifting line** (nonlinear, Phillips–Snyder): for straight or moderately swept wings of
  aspect ratio 4 or more; the section lift curves come from the viscous polars, so it
  follows them to cl max.
* **Viscous** (default): every spanwise strip takes its profile drag from its section's
  2D viscous polar at its own Reynolds number (XFOIL-class, Ncrit and Mach from Setup).
  Those polars are what a first run spends its time on (about 30 s for one airfoil); they
  are kept for the session, so a second run with the same airfoils takes well under a
  second (measured: 30 s, then 0.26 s, for a tapered NACA 2412 wing at six angles).
  Inviscid sweeps take a second or two.

The result has CL, CD = CDi + CDv, Cm, side force, roll and yaw per angle, the span
efficiency e and the span loading, plus a stall-onset estimate (the first section to
reach its cl max). It is potential flow plus strip theory: no fuselage, no stall beyond
what the section polars contain (docs/theory_wing.md). For the flow itself, the wing can
also be run as 3D CFD (lofted to a mesh); at flight Reynolds numbers that CFD cannot
resolve a wing's boundary layers, and the plan says to use the Wing analysis for lift and
drag.

## 12. Command line (3D and wings)

```
aerolume-cli quick3d --stl car.stl --speed "120 km/h" --quality draft      units guessed (mm/cm/m/in)
aerolume-cli quick3d --stl car.obj --kind vehicle --yaw 5 --save MyCar      yawed: full domain
aerolume-cli quick3d --sphere 0.2 --speed 0.0073 --plan-only -v             a sphere at Re 100
aerolume-cli wing --naca 2412 --span 10 --chord 1 --taper 0.5 --sweep 10 --alpha -4:12:1 --csv wing.csv
aerolume-cli wing --naca 0012 --span 6 --chord 1 --method llt --inviscid
aerolume-cli run MyCar.aerolume "car.stl 3D"                                3D and wing simulations too
aerolume-cli doctor                                                         also a 3D solver check and VTK
```

`quick3d` also accepts `--length` (scale to this length), `--clearance`, `--alpha`,
`--full-domain`, `--no-ground`, `--fixed-ground`, `--cells-per-ref`, `--duration`,
`--re-mode`, `--re-cap` and `--sensitivity cd|cl|ld`. Binary STL files have no parts:
export OBJ (groups) or PLY per part to have wheels recognised and spun.

## 13. Placed physics: heat, smoke, particles, fans, jets and closed rooms

A CFD simulation can carry more than the flow around its bodies (DESIGN.md section 10).
Everything below is off by default, so a simulation without it plans and runs exactly as
before (the only addition is the solver diagnostics, recorded by default at about 2 % of a
step in 2D and 3 % in 3D; switch them off with `CFDSettings.record_diagnostics`). It works
in 2D and 3D with the same meaning; where the dimensions differ the text says so. The
solver side (equations, discretisation, its own validation) is in `theory_ns2d.md` Part II
and `theory_ns3d.md` sections 15-18.

| item | set in | what it does |
|------|--------|--------------|
| **Heat transfer** | Setup ▸ Heat transfer (`Simulation.thermal`) | solves for the temperature; Boussinesq buoyancy optional; domain walls adiabatic, held at a temperature or given a heat flux |
| **Heated body** | the body's thermal condition (`Body` / `Mesh3D.thermal`) | holds the surface at a temperature or releases a heat flux; insulated by default |
| **Heater** | `Simulation.heat_sources` | releases a power (W, or W per metre of span in 2D) over a box / rectangle or a sphere / disc; negative cools |
| **Smoke** | `Simulation.emitters`, species "dye" | releases a concentration (0 to 1) from a point, line, rake, box, disc or plane, steady or pulsed |
| **Particles** | `Simulation.emitters`, species "particles" | releases tracer particles (per second) that the flow carries: streaklines, or timelines when pulsed |
| **Fan** | `Simulation.fans` | an actuator disc: a set thrust, a pressure jump, or a speed through it that the solver holds |
| **Jet / vent** | `Simulation.ports` | blows fluid in at a speed (with its own temperature and smoke) or sucks it out |
| **Closed room** | Setup ▸ Domain ▸ Closed box (`CFDSettings.domain_mode`, `domain_box`) | walls on every side instead of an inlet and an outlet |

**What this installation runs.** `aerolume.simulation.SUPPORTED_PHYSICS` lists, per
dimension, the features the planners and solvers carry into a run: `emitters`, `thermal`,
`heat_sources`, `fans`, `ports`, `closed_domain` and `adaptive_dt`, all seven in 2D and in
3D. Scripts read it through `al.supported_physics(2)` / `al.supported_physics(3)`, and the
API, the assistant's run tools and the Setup page refuse or flag anything missing from it
rather than run a case without its fan or heater. `tests/test_simulation_supported_physics.py`
plans a case with every feature, in a room and in a tunnel, in both dimensions, and checks
each one reaches the solver.

**Where things are.** Positions are metres in the design frame. An item placed with
`frame="body"` turns with the geometry when the angle of attack (and in 3D the yaw)
changes, about the same moment point; `frame="wind"` keeps it where it is in the tunnel.
The planner applies that rule once, so the Setup preview, the run and explore agree. An
item outside the planned domain is left out with a warning (it would act on nothing). In
3D an item off the centre plane y = 0 turns the symmetry plane off (it would mirror the
item into a twin), and so do a swirling fan, a fan or jet blowing across the plane, and
different thermal conditions on the front and back walls. Items that are their own mirror
images stay: a smoke rake along y = 0, a smoke rake or line whose two ends mirror each
other across it (a spanwise rake from y = −0.08 to +0.08 m: its nozzles map onto each
other, so the solved half carries the nozzles at y ≥ 0 and the mirror shows the rest), a
heater, fan or jet centred on it and blowing along it. The solver puts exactly half of
each into the solved half and reports fans, jets and heaters for the whole, so a hot
sphere with a smoke rake on its centre line gets the same half-domain grid as without the
rake. A particle rake must lie in the plane: the solver reflects a particle released at
y < 0 back into the solved half, where it would double its mirror nozzle's streak.

### 13.1 Heat transfer: the scales the planner chooses

The solver works with θ = (T − T_ambient) / ΔT. The planner picks the scales and states
them, with the dimensionless groups, in the Estimate (Setup) and in `plan.summary()`:

* **ΔT, the temperature scale**: the spread of every temperature the case sets (the
  ambient, bodies and walls held at a temperature, hot jets) -- T_hot − T_cold, the
  definition the cavity benchmarks use -- or, when larger, the rise a heat flux or a heater
  produces. A heat flux: q L / (k Nu) with Nu from a laminar correlation (forced flow over a
  plate, Pohlhausen; natural convection off a constant-flux plate, Sparrow & Gregg; at least
  1). A heater is a source in the air, not a wall, so its rise is set by the air that
  crosses it (`physics.heater_rise`): P / (ρ c_p u A), with u A the flow its own plume
  carries through it (u_p = (g β P / ρ c_p)^(1/3) for a 2D line source, the point plume's
  speed at one heater width in 3D) or a tunnel's free stream across it, whichever is larger;
  in a closed box plus the box's warming over one turnover, P t / (ρ c_p V); never more than
  q L / (k Nu) of a solid of the heater's size. It is a scale, not a prediction: the run
  computes the temperatures. With nothing heated it is 1 K and the plan says the
  temperature will stay uniform.
* **U, the speed scale**: the free stream in a tunnel. A closed room has none, so it is the
  fastest thing that drives the air: a moving floor, a jet, a speed-controlled fan, a thrust
  fan's momentum-theory disc velocity √(T / 2ρA) (√(Δp / 2ρ) for a pressure fan); failing
  those, the free-fall velocity √(g β ΔT H) when Ra ≥ 10³ (so Re = √Gr), else κ / H (the
  box is conduction-dominated).
* **H, the length of Gr and Ra**: the height of a closed box heated only at its walls or by
  heaters (the cavity convention), otherwise the design's reference length.

From these: Re = U L / ν, Pr = μ c_p / k, Pe = Re Pr, Gr = g β ΔT H³ / ν², Ra = Gr Pr, and
the solver's buoyancy coefficient Ri = g β ΔT L / U². With a forced flow the Estimate also
shows the Richardson number Gr / Re² on H and calls the regime *Forced* below 0.1, *Mixed*
up to 10 and *Natural* above; a closed box without forcing is *Conduction* below Ra 10³
and *Natural* above. Gravity is along −y in 2D and −z in 3D. The two Richardson numbers
differ only in their length, and the plan names it: the solver's coefficient is on L, the
Estimate's on H (row "Length of Gr, Ra and Ri"). The 4 × 3 m fan room with a 0.5 m fan
reads Ri 0.884 on the reference length 0.5 m and 5.30 on the box height 3 m.

**Fluid properties** at the ambient temperature: air from Sutherland's law for the
conductivity (White, Table 1-3), c_p 1006 J/(kg K), β = 1/T; water from Incropera &
DeWitt Table A.6 (0-100 °C). Typed values win. A custom fluid without typed values gets
air's as placeholders, and the plan warns: a run never quietly uses air's conductivity
for another fluid.

**One set of numbers.** ΔT, β·ΔT, Gr, Ra and Pe are derived once
(`simulation.physics.thermal_estimate`, the planner's own derivation, cheap enough for the
Setup page to call on every change), and every line that quotes them reads that one result.

**Checks in the plan.** β ΔT above 0.1 warns that the Boussinesq approximation (density
constant except in the buoyancy force) loses accuracy, and the line names the ΔT it used;
Ra above 10⁸ warns that the thermal boundary layers turn unsteady and then turbulent. A
preset fluid made at another temperature than the ambient (ISA sea-level air is 15 °C) is
a warning too: the run would take its density and viscosity at one temperature and its
conductivity, c_p and β at the other. Set **Flow ▸ Fluid** to the ambient temperature
(`al.core.fluid.air(20)`) to clear it; `simulation.physics.default_ambient(fluid)` is the
ambient that agrees with a preset fluid (15 °C for ISA sea-level air). A plan holds one
warning per quantity (`physics.warning_key`; `physics.merge_warnings` merges a page's own
lines into the plan's without a second β·ΔT or Ra).

**Buoyancy-dominated tunnel runs.** Above Ri = Gr / Re² = 10 the plume, not the free
stream, sets the speeds, and the force coefficients, normalised by the free stream's
q = ½ ρ U², become mostly buoyant force over a small q. The Draft hot sphere (D 0.1 m,
Re 100, 60 °C in 20 °C air, Ri 627) read Cd 6.4 and Cl 119 on q = 0.00013 Pa. The plan
warns, and the result adds the coefficients on the buoyant dynamic pressure ½ ρ u_b²,
u_b = √(g β ΔT L) (Cd 0.010 and Cl 0.19 for the sphere: the coefficients times U² / u_b² =
1 / Ri). Compare such runs by their forces in newtons. The tunnel is sized for blockage,
not for a rising plume: the sphere's plume reached the frictionless ceiling 0.21 m above it
and spread under it, which is where the run's fastest air was; the plan says how far the
ceiling is.

**Draft runs of buoyancy-dominated cases are shortened.** The time step of such a run
follows the plume, so the free stream's duration costs u_peak / U times the steps: the Draft
hot sphere was planned at 40 L/U, 189,682 steps and about 40 minutes on the RTX 4060 Ti. At
Draft the planner now keeps the step count of the same case without its plume, t_end ×
U / u_peak, within 2 to 10 L/U (`physics.draft_buoyant_duration`); the hot sphere runs
2 L/U (the plume travels about 49 sphere diameters in that time), and the plan says so before
Run, with the estimate for the shortened run. That is enough for pictures, not for converged
forces: Standard quality or a Duration typed in Setup runs the full length, and a closed box
suits a heated body in nearly still air better, because its time scale is the plume's own.

**Planned estimate, then measured.** A heater's or heat flux's ΔT is an estimate of the
rise. Planned as q L / (k Nu), the Draft fan room with a 300 W/m heater, a supply jet and a
vent read ΔT 44.1 K (β·ΔT 0.15, a Boussinesq warning; Ra 1.3 × 10¹¹), and its warmest fluid
was 3.31 K above the ambient on the time average (3.97 K at the hottest instant; measured
during the review before the 0.1.0 release). The through-flow rule above plans it at 4.1 K, and a 500 W/m room at 5.6 K
against 5.6 K measured. When ΔT was estimated, a run still re-states the numbers on what it
measured (`physics_run.apply_measured_thermal`):

* ΔT becomes the warmest fluid's rise above the ambient (or the coldest fluid's drop below
  it, whichever is larger; `t_fluid_max` / `t_fluid_min`, time-averaged), and Gr, Ra, the
  Richardson number Gr / Re² and the regime are derived from it (`physics.result_scales`,
  which says "measured" or "planned estimate"). U, Re and Pe stay those of the run.
* The Rayleigh, thermal-resolution and buoyancy-dominated warnings are re-derived on the
  plan's grid from the measured scales and replace the planned ones. The Boussinesq bound
  is checked against the largest difference seen at any instant (mean field and last step).
  A run that exceeds the bound without having been warned gets the warning.
* The plan's temperature-scale and buoyancy notes stay in the result's notes, prefixed
  "Planned estimate:", and one note gives ΔT, Gr and Ra measured next to the planned values.

A ΔT the case sets (bodies, walls or jets at fixed temperatures: the cavity's T_hot − T_cold)
is held exactly and is not re-stated.

**Thermal resolution.** The grid is checked against the thermal boundary layer: in a
forced flow the laminar momentum layer thinned by Pr^(−1/3) (Pr^(−1/2) below Pr 1); in a
naturally convecting box H / (2 Nu) with Nu from the cavity correlation of Berkovsky &
Polevikov (Incropera eq. 9.50). When buoyancy outruns the forced flow (Ri = Gr / Re² > 1)
the layer is the plume's (`physics.buoyant_layer`, Bejan's scalings on the length of Ra):
L·Ra^(−1/4) for natural convection (Ri > 10) and L·(Re_b·Pr)^(−1/2) with Re_b on the
buoyant velocity for mixed convection. Sized on the free stream, the Draft hot sphere
(D 0.1 m, Re 100, Ri 590, Ra 4.2 × 10⁶) was told "about 12 cells across the thermal
boundary layers"; its layer is about 2 mm, 0.4 of its 5 mm cells, and the plan now says so.
Below about 4 cells across the layer the plan warns that heat rates and Nusselt numbers are
under-resolved. A case in which no body or wall exchanges heat (a room warmed only by
heaters or hot jets) has no such layer and no Nusselt number, and the heat input is imposed,
so it gets a note instead of the warning, and its grid is the box's own (64 cells across the
smaller side at Draft), not one sized for a boundary layer it does not have.

**Unvalidated regimes are named.** Natural or mixed convection around a heated or cooled
body is not validated (docs/validation.md §10: the validated heat transfer is laminar
natural convection in cavities heated at their walls and forced convection past a cylinder
and a sphere). Such a plan and its result carry the warning "… around a heated body: regime
not validated (docs/validation.md §10)". The Draft hot sphere above read Nu 53.5 where the
natural-convection correlation of Churchill (1983) gives 22.6 at its Ra and Pr 0.71.

### 13.2 Heat transfer: what a result holds

* **Nusselt number** (`result.nusselt`, mean ± standard error over the averaging window).
  With heated bodies: the mean over them of each body's Nu = Q L / (k (T_wall − T_ambient) A)
  on the reference length (a heat-flux body uses its mean surface temperature). In a
  closed box without heated bodies: the Nusselt number of its hottest wall,
  Nu = (Q / A) H / (k ΔT), the definition of de Vahl Davis (1983) and of the
  Rayleigh–Bénard literature. A steady value is reported as steady, not "± 0".
* **Heat rate** (`result.heat_rate`): W in 3D, W per metre of span in 2D, from the bodies
  into the fluid (from the hot wall in a closed box without heated bodies). When nothing
  is held at a temperature -- a room warmed by a heater, a tunnel with a heat source -- it is
  the heat the heaters and heat-flux walls put into the fluid, measured from the solver's
  balance, and there is honestly no Nusselt number (the result's notes say so). Per body:
  `per_body[name]["nu"]`, `["heat_rate"]`, `["t_wall"]` (°C).
* **Fluid temperatures** (`result.diagnostics`): `t_fluid_mean` (weighted by the fluid
  fraction), `t_fluid_min`, `t_fluid_max` in °C on the solver's full grid (time-averaged
  when the run averaged), `t_fluid_dev_max` (K, the largest difference from the ambient at
  any recorded instant) and, with heaters, `heat_sources_set` (their set power, to compare
  with the measured `heat_in_sources`).
* **Fields**: `temperature` in °C (the mean and the end-of-run fields; 3D slices and the
  stored volume too), `temperature_rms` in K.
* **Heat balance** (`result.diagnostics`): heat in through bodies, walls, heaters, jets and
  the inlet, out through the outlet and vents, the storage rate, and their residual, in W
  or W/m. The command line prints the residual as a share of the largest heat flow. When
  more than 10 % of the heat put in is still being stored at the end (the fluid is still
  warming up, `physics_run.storage_share`), the result warns that the temperatures, the
  heat rate and the Nusselt number were still changing, and the Nusselt number's error bar
  is marked a rough estimate: a standard error over a window that is still drifting is not
  the uncertainty of the mean (the Draft hot sphere stored 69 % of its heat and once
  reported Nu 53.51 ± 0.05).
* **History**: `nu` and `heat_rate` per step, plotted on the Results page and judged by
  the convergence verdict (13.6).

### 13.3 Smoke and particles

Smoke goes into up to four independent channels **by colour**: emitters of one colour
share a channel, a jet that carries smoke uses the first (default) colour, and colours
past the fourth fold into the last channel (the plan warns, naming the colour they are
drawn in). The live view and the stored frames composite the channels in their colours
over a dark background. Placed smoke replaces the automatic inlet comb (2D) or seed rake
(3D). `rate` is the released concentration (clipped to 0 … 1, with a note), pulses
(`pulse_period`, `pulse_duty`) and `start` / `stop` are seconds of flow time.

Particles are released at `rate` per second from the emitter's nozzles and carried by the
flow (RK2, bilinear / trilinear); a steady emitter draws streaklines, a pulsed one
timelines. Frames carry them as points with their emitter's colour and, for point and
rake emitters, as streak lines. A result stores the smoke (the end-of-run channels) but
not the particles.

### 13.4 Fans and jets

A fan is an actuator disc (a strip in 2D) of `diameter` and `thickness` (default an
eighth of the diameter, at least two cells): `"thrust"` (N, or N/m in 2D), `"pressure_jump"`
(Pa) or `"velocity"` (the disc-averaged speed through it in m/s, which the solver holds with
a PI controller on the thrust, inside the CUDA graph). `swirl` (3D) adds a tangential
velocity. The plan warns below 8 cells across the disc: the thrust is still delivered,
the induced velocity and the flow rate are then indicative. `result.fans_report` gives,
per fan, the delivered thrust (with its standard error), pressure jump, volume flow, disc
and induced velocity and ideal power, all in SI.

A jet blows fluid in at `speed` along `direction` through an opening of `size`, with its
own `temperature` (heat transfer on) and smoke; a vent sucks the same flow out. Mass and
momentum fluxes are exact. In a closed room the jets must balance the vents: the plan
compares the volume flows (2 % tolerance) and warns, because the solver otherwise removes
the difference evenly over the whole box to keep the pressure problem solvable. The
opening wants at least 4 cells; fewer keep the flow exact and the profile approximate.

### 13.5 Closed rooms

With Domain ▸ Closed box the tunnel's inlet and outlet become no-slip walls (a moving
floor is a driven cavity). The box is `domain_box`, or the bodies and placed items with one
size of room on every side. The grid fits the walls exactly in height and to the nearest
cell in width (2D; FFT-friendly sizes where the shape allows) or in multiples of 4 cells
(3D), and has 64 / 96 / 128 cells across the smaller side at Draft / Standard / Fine on a
GPU (48 / 64 / 80 on a CPU; 3D: 32 / 48 / 64, CPU 16 / 24 / 32), refined to 4 / 6 / 8 cells
across the thermal boundary layer of a naturally convecting box. The run length follows
the physics: (45 + 0.225 Pe_H) free-fall times for natural convection (the validation
cavities settled in 12-144, so 1.5 times that, capped at 1500), 1.5 diffusion times H²/κ
for conduction, and 30 / 40 / 50 box turnovers for fans and jets; the run stops earlier
once its monitors settle. That steady-state rule does not fit rooms: from Ra 10⁸ on the
flow is unsteady and never reaches the steady state it waits for, and the Draft heater-only
4 × 3 m room was planned at 1500 free-fall times, 9000 L/U and 49 minutes. Draft therefore
runs such boxes for 40 free-fall times (the large-scale circulation is established within a
few tens), with a note; Standard and Fine, or a Duration typed in Setup, keep the long rule.
The Duration note is built from the duration actually used ("6 L/U (1.00 free-fall times
H/U, set in Setup, …)" for the heater-only room, which had read "1500 free-fall times"). Coefficients, where bodies exist, use q = ½ ρ U² with the
reference velocity. A closed box that nothing drives is refused with a message. A box heated
at a set rate with no way for the heat to leave (no vent, no wall or body at a set
temperature) warms by P / (ρ c_p V) per second for as long as it runs; the plan says so, with
the rate, and its temperatures are not convergence monitors (they never settle).

**2D rooms with fans are qualitative.** In 2D the fan's power can only leave the room
through wall friction and numerical dissipation (2D flow has no vortex stretching, so
energy gathers in room-sized vortices), and the room's speeds have no grid-converged
answer: for the 1 N/m fan in the 4 × 3 m room the mean speed is 2.4 m/s at 6 and 12 cells
per fan width and 3.4 m/s at 24, and the room needs 500-900 L/U to settle against the
240 L/U a Draft run plans (theory_ns2d.md §18.4). The thrust is exact and the flow pattern
is meaningful, but speeds, the disc velocity, volume flow and convective heat transfer are
not results. Every 2D closed domain with a fan therefore carries that warning, first, in the
plan and in the result (so the Results banner and the report show it), with a sentence when
the planned duration ends inside the start-up transient. Use 3D for rooms: a 3D fan's jet
breaks down into 3D turbulence, which dissipates its power inside the room.

### 13.6 When a run with placed physics stops

Besides the forces (if there are bodies in a tunnel), the verdict judges **monitors**: the
Nusselt number (to half the quality's tolerance, since the references are quoted to
0.1-1 %; it stands for the bodies' heat rate, or their wall temperature for heat-flux
bodies), each fan's disc velocity (the thrust a speed-controlled fan needs), and in a closed
box the mean kinetic energy and, when heat goes in at a set rate and can leave, the rise of
the warmest and of the mean fluid above the ambient (`t_rise_max`, `t_rise_mean`, in K: a
rise, so a relative tolerance means something). A *steady* monitor must also stop drifting:
its change between the halves of the averaging window has to be under a tenth of its
tolerance, because a value settling exponentially sits within the tolerance of its current
value long before it reaches the asymptote. The Estimate's "Stops early when" lists them,
and the run log says which monitor held a run back.

**A closed box is never judged on its forces.** Force coefficients on a box's own reference
velocity mean nothing inside it, and the Summary hides them; the Electronics enclosure once
ended "Not converged" because its circuit board's Cd was known to only 12 %, with no visible
reason on screen. Closed boxes are judged on their monitors alone, and a not-converged
message names the one that held the run back and every other one the box is judged on.

**Diagnostics** (`record_diagnostics`, on by default): one row every 10 steps with the RMS
and maximum divergence, the pressure-solve residual, the L2 / RMS / maximum / relative
change of each field, the CFL number, the step and the throughput. Explore streams them
to the Diagnostics panel (exportable as CSV); a result keeps the last row and the heat
balance. `adaptive_dt="pid"` replaces the per-step CFL rule by a PID-controlled step
(smoother, at most +0.2 % change per step, theory_ns2d.md section 20).

### 13.7 Time step and run-time estimate with placed physics

The time step is set by the fastest air, so the planner estimates the **peak speed**
(`simulation.physics.peak_speed`) as the largest of the reference velocity, the buoyant
velocity √(g β ΔT H) (√Gr times that below Gr 1), twice each fan's disc velocity (the
slipstream of momentum theory), the jets and a moving floor, and plans the step with the
measured speed factor (max|u| + max|v| (+ max|w|)) relative to it. The coefficients and the
Reynolds number keep the reference velocity. Each kind of case has its own measured factor
(`plan.calibration_kind`), refined by every run of that kind:

| Kind | Factor (of the peak speed) | Measured on |
|---|---|---|
| `closed-fan` (2D) | 3.1 laminar, 5.1 LES | the Draft electronics enclosure (4.4-6.5 times its 2 m/s fan speed over the run); the Draft 2D fan room (10.2 times its thrust fan's disc velocity) |
| `closed-fan+jet+heat` (2D) | 3.9 LES | the Draft 2D fan room with a supply jet, a vent and a 300 W/m heater (3.87) |
| `closed-fan+jet+heat+body` (2D) | 2.7 laminar | the Draft electronics enclosure (2.68) |
| `closed-natural+heater` (2D) | 2.4 | the sealed Draft heater-only room (2.35 of its free-fall velocity; laminar taken the same) |
| `closed-fan` (3D) | 3.4 | the Draft 3D room (6.7 times its 1 m/s fan speed) |
| `closed-natural+heater` (3D) | 3.5 | not measured: the 2D value scaled for the sum of three maxima |
| `<class>-plume` (2D) | 2.7 laminar | the Draft heated cylinder at Ri 30 (2.0 rising to 3.0 of its buoyant velocity over 10 L/U) |
| `<class>-plume` (3D) | 4.7 | the Draft hot sphere at Ri 627 (117-119 U from t = 0.4 to 3) |
| `<class>-jet` | 1.5 (2D), 2.0 (3D) | not measured yet: replaced by the first run of the class |

A box without fans keeps `closed-forced` / `closed-natural` (relative to its reference
velocity, which there is the peak), and a tunnel without a faster source keeps its class
factor.

**Closed boxes are calibrated per case signature.** How fast a box recirculates depends on
what is in it: the plain Draft 2D fan room measured a factor of 5.09 of its peak, the same
room with a supply jet, a vent and a 300 W/m heater 3.87 (the jet and the vent bleed the
circulation), the electronics enclosure 2.68 (laminar), and a sealed heater-only room 2.35 of
its free-fall velocity, where the cavities measured 0.5. One shared "closed-fan" factor let
the other rooms' runs move the furnished room (fan, heater, supply jet and vent) from 52 s
planned (after its own 49 s run) to 164 s. The key is now the class plus the signature and the grid class:
`closed-fan+jet+heat@64` (jets, buoyant heat besides the fan), `+heater` (a natural box
warmed by heaters), `+body` (bodies inside), `@n` (cells across the smaller side, rounded to
a power of two). Each signature starts from the most specific default measured for it
(the values above), each run moves its key by at most 1.5 times (`plan.clamp_update`), and
the planner takes the weighted median of the key's runs (`plan.speed_median`: weighted by
resolution and age, with the default as one more voice), so one unusual run cannot swing
the next estimate. Explore sessions never record: their obstacles, sliders and placed items
change while they run. Measured end to end on the reference machine on 31 October 2026
(`validation/records/planner-timing-2026-10-31.json`, which also states how): the furnished room ran in 67.2 s on a
shared GPU and was then planned at 58.9 s, at 64.9 s after a plain fan room had run and at
65.4 s after the enclosure had (the throughput the other runs measured moves it a little;
the speed factor does not).

**A heated box at rest.** At rest the 2D solver's first step has no bound but the viscous
one when nothing sets a source speed (no jet, vent, or speed- or pressure-controlled fan):
the sealed Draft heater-only room took a first step of 56 L/U and diverged. Such a run (and
an explore session of it) takes its first 10 steps at the planned step
(`runner.start_from_rest`), after which the solver's own buoyant-parcel bound holds; the
room then runs its 240 L/U in about 18 s.

Measured effect on the reference machine (RTX 4060 Ti): the Draft enclosure was planned at
28,321 steps and 1 min 05 s and took 121,433 steps and 5 min 06 s on a shared GPU (an
earlier development build); it is now planned at about 101,000 steps and 2 min 14 s headless
and runs in 2 min 32 s with the GPU otherwise idle (the 2D GPU throughput table was
re-measured on an idle GPU on 31 October 2026: 1.4-1.6 times the earlier, shared-GPU rows,
which had made first 2D estimates about 1.6 times too long; both measurements and the
procedure are in `validation/records/planner-timing-2026-10-31.json`). Planned before any run, the furnished room
reads 74 s headless against 62-67 s measured, the plain fan room 56 s against 51 s. The 3D room
(estimated 6 min 31 s, observed about 20 min) is now estimated at 21 min, and the hot
sphere's step is planned at the 0.0002 L/U it runs at. Once a run has timed two blocks,
**Remaining** is the measured time: the wall time so far plus the remaining flow time at
the measured rate (`simulation.runner.refine_estimate`).

**Cost.** The estimate multiplies the time per step by the measured cost of each feature
in use (2D, on a shared GPU, so indicative: heat transfer +55 %, buoyancy +9 %, smoke
+21 %, particles +19 %, fans +9 %, jets +14 %; 3D, idle GPU: heat transfer +30 %, each
smoke channel after the first +6 %, particles +11 %, fans and jets +3 % each). The plan
note gives the total; each run feeds its measured throughput back into the estimate.

### 13.8 Explore with placed physics

Explore (section 7) edits the placed physics while the flow runs, without a restart:

* **Move and retune** emitters, fans, heaters and jets (drag them in the 2D view, or edit
  them in Setup). The solver rewrites its sources in place and keeps its CUDA graphs
  when only positions or values change; adding the first fan or a new smoke colour
  re-captures them once.
* **Draw obstacles** (2D): brush strokes and polygons add or erase solid; they are
  penalised to rest with the bodies' calibrated wall, insulate, clear smoke, and are
  replayed exactly after an undo, an erase or a road switch. In 3D the session places
  balls and boxes (`post(obstacles={"ops": [{"kind": "sphere", "center": …, "radius":
  …}]})`).
* **The Reynolds lock**: hold two of U, ν and Re and edit the third. A new ν changes Re
  (and Pe, at the same Prandtl number); a new speed re-expresses everything scaled by U
  (fan thrust, jet speed, heater power, buoyancy), and the SI readouts follow.
* **Pause, step and frame skip**: step a given number of steps while paused, or fix the
  steps per frame; the target frame rate paces the rest.
* **Solver trace**: capture one step's stages (advection-diffusion, divergence, pressure
  solve, correction; in 3D cut on the slice shown) into the next frame.
* **Readouts**: frames carry the Nusselt number, the heat rate, the temperature range and
  each fan's thrust and disc velocity in SI (refreshed twice a second), the new
  diagnostics rows, and, when an edit could not be carried out, a notice in words (a
  closed box has no speed slider; an item moved outside the domain).

Two things need a restart, and explore says so: switching heat transfer on (the solver
then carries a temperature field it did not allocate) and, in 3D, adding smoke to a
session that started without any.

### 13.9 Measured on the development machine

Every case below went through the whole chain a user's run takes (the model in SI, the
planner, the solver, the outputs in SI), on the RTX 4060 Ti in float32 with nothing else on
the GPU, on 19 October 2026 (`tests/test_simulation_physics_run.py` repeats five of them
as slow tests). Wall times include building the solver; the planned duration is the
planner's, and "stopped" is where the monitors had settled.

| case | quality | grid | result | reference | difference | stopped at (planned) | wall time |
|------|---------|------|--------|-----------|------------|----------------------|-----------|
| square cavity, Ra 10³ | Draft | 64 × 64 | Nu 1.118 | Nu₀ 1.117 (de Vahl Davis 1983) | +0.09 % | 33.6 (51.0) | 20 s |
| | Standard | 96 × 96 | Nu 1.118 | 1.117 | +0.09 % | 33.2 (51.0) | 43 s |
| square cavity, Ra 10⁴ | Draft | 64 × 64 | Nu 2.251 | 2.238 | +0.58 % | 42.1 (64.0) | 9 s |
| | Standard | 96 × 96 | Nu 2.247 | 2.238 | +0.40 % | 41.8 (64.0) | 20 s |
| square cavity, Ra 10⁵ | Draft | 64 × 64 | Nu 4.562 | 4.509 | +1.18 % | 70.3 (105) | 8 s |
| | Standard | 96 × 96 | Nu 4.539 | 4.509 | +0.67 % | 68.2 (105) | 15 s |
| square cavity, Ra 10⁶ | Draft | 74 × 74 | Nu 9.030 | 8.817 | +2.42 % | 153 (235) | 13 s |
| | Standard | 112 × 112 | Nu 8.911 | 8.817 | +1.07 % | 154 (235) | 22 s |
| cubic cavity, Ra 10⁴ | Draft | 32³ | Nu 2.079 | 2.054 (Tric et al. 2000) | +1.18 % | 43.2 (64.0) | 5 s |
| cubic cavity, Ra 10⁵ | Draft | 32³ | Nu 4.499 | 4.337 | +3.74 % | 69.1 (105) | 4 s |
| heated cylinder, Re 20, Pr 0.7 | Draft | 540 × 450 | Nu 2.502 | 2.409 (Lange et al. 1998) | +3.86 % | 39.0 (60) | 20 s |
| | Standard | 1250 × 800 | Nu 2.482 | 2.409 | +3.06 % | 52.0 (80) | 264 s |
| heated cylinder, Re 40, Pr 0.7 | Draft | 540 × 450 | Nu 3.341 | 3.281 | +1.83 % | 53.4 (60), converged | 44 s (GPU shared with other work) |
| | Standard | 1250 × 800 | Nu 3.302 | 3.281 | +0.67 % | 55.0 (80) | 213 s |
| heated sphere, Re 50, Pr 0.71 | Draft | 320 × 52 × 104 (half) | Nu 5.508 | 5.18 (Whitaker) / 5.78 (Ranz–Marshall) | between the two | 26.1 (40) | 49 s |
| fan in a 4 × 3 m room, 1 N/m | Draft | 96 × 72 | thrust 1.000 N/m | the set 1 N/m | 0 (momentum balance) | 277 (240), not converged | 88 s |
| speed-controlled fan, 1 m/s (2D) | Draft | 96 × 72 | disc velocity 0.994 m/s | the set 1 m/s | −0.6 % | 298 (240), not converged | 57 s |
| speed-controlled fan, 1 m/s (3D, 4 × 3 × 2.5 m) | Draft | 52 × 40 × 32 | disc velocity 0.9998 m/s | the set 1 m/s | −0.02 % | 276 (240), not converged | 27 s |

Reading the table:

* The solver measures the mean Nusselt number of the hot wall, so the reference is de Vahl
  Davis' Nu₀ (Table V; `api.CAVITY_NU_2D`, `aerolume-cli validate --full`), not the paper's
  cavity average Nū (1.118, 2.243, 4.519, 8.800), which differs by 0.1-0.4 %. The cavities
  converge to it as the grid is refined (Ra 10⁶: +2.4 % at Draft, +1.1 % at Standard), and
  the Ra 10³ case, which is conduction-like, is within 0.1 % at both. The 3D Draft cavity at Ra 10⁵ has 3.4 cells across its thermal boundary layer,
  and its plan says so; +3.7 % is what that costs.
* The heated cylinder reads 1-4 % above Lange's correlation, more at Re 20 than at 40;
  the solver's own grid study finds about 2 % above the published values even on a
  converged grid with 1.25 % blockage (theory_ns2d.md 21.2), and these runs add the
  Draft / Standard domains' 5 % blockage.
* The two sphere correlations differ by 11 % at Re 50; the run lies between them, as the
  slow test requires. It is not a precision check.
* The Re 40 cylinder used to run to its end and read "not converged (lift)" over a decaying
  start-up wiggle on a lift of −0.0006 (section 5); it now stops at t = 53.4 with the same
  Nusselt number.
* A room with a fan, a 500 W/m heater, a supply jet and a vent (Draft, 96 × 72, 240 L/U,
  116 s on a shared GPU): heater 500.0 W/m measured against 500 set, 299.5 W/m out through
  the vent, −9.0 W/m enthalpy at the jet, 191.5 W/m still being stored; the balance closes
  to 0.0039 W/m (8 × 10⁻⁶ of the throughput). Mean air temperature 22.1 °C, warmest
  (time-averaged) 25.6 °C.
* The fan rooms are unsteady (one large, slowly wandering vortex), so their monitors never
  settled within 1.15-1.24 times the planned duration and the runs say "not converged";
  the thrust of a thrust fan is exact by construction, and the speed controller holds its
  target to 0.6 % (2D) and 0.02 % (3D) over the averaging window. The 2D rooms' speeds
  are qualitative (section 13.5): only the imposed quantities in these rows are results.
* Every run's divergence stayed below 1.2 × 10⁻⁵ U/L, and the heat balance of the last step
  closed to 3 × 10⁻⁴ of the heat transferred or better (the cavities to 10⁻⁷).

**Estimates.** The run-time estimate covers the planned duration; the cavities stopped at
about two thirds of it. A plan's first estimate on a new machine comes from the reference
throughput table, and each run refines it. With placed physics the refinement reads the
real flow speeds from the diagnostics: the time step of a heat-transfer run is often set
by the temperature's stability bound rather than the flow, and the earlier rule (the CFL
number times the cell over the mean step) took that bound for a fast flow -- 27 times too
fast after the Ra 10³ cavity -- and multiplied every later cavity estimate by 2.6 (60 s for a
9 s run). With the fix the Ra 10⁴ cavity is estimated at 15 s after a Ra 10³ run, against 9 s
for two thirds of its planned duration. A thrust fan in a closed room is the case the
estimate gets wrong first: the circulation it drives is much faster than the momentum-
theory reference (max |u| + max |v| reached 12.6 times it), so the first run took 8.6
times its estimate, and the plan says so.

### 13.10 Command line

```
aerolume-cli cavity --ra 1e4                               the square cavity against de Vahl Davis (1983)
aerolume-cli cavity --ra 1e5 --3d --quality standard       the cubic cavity against Tric et al. (2000)
aerolume-cli quick --template cylinder --param diameter=0.01 --speed 0.0584 --heated 25 --no-buoyancy
aerolume-cli quick --template cylinder --param diameter=0.2 --closed -1,3,-1,1 --fan -0.5,0:0.4:0.5
aerolume-cli quick3d --sphere 0.02 --speed 0.0365 --heated 25 --fan 0.06,0,0:0.01:0.05:velocity
aerolume-cli validate --full                               also the heated cavity through the planner
```

`--heated T` holds every body at T °C (heat transfer on, buoyancy on unless `--no-buoyancy`;
`--ambient` sets the air temperature, default 15 °C, the standard sea-level air of every
command). `--fan X,Y[,Z]:D:VALUE[:MODE[:DIR]]` (repeatable) places a fan in the tunnel frame:
MODE `thrust` (default), `pressure` or `velocity`, DIR its direction (default +x).
`--closed X0,X1,Y0,Y1[,Z0,Z1]` puts the case in a room. The result then lists the Nusselt
number, the heat rate, a table of the fans, and the divergence, pressure residual, CFL
number and heat-balance residual of the last step.

### 13.11 Limits

* Boussinesq buoyancy only (constant density elsewhere); no radiation; no conduction
  inside solids (a body is held at a temperature or releases a flux at its surface).
* One fluid; smoke and particles are passive (they do not change the flow).
* 2D results are per metre of span: a heated cylinder in 2D is an infinitely long one.
* The heated-cylinder Nusselt number reads about 2 % above the published values even on a
  converged grid, the size of the spread between those references (theory_ns2d.md 21.2);
  the turbine-mode fan's induction is 4-5 % below inviscid momentum theory at Re 500
  (attributed to viscosity, not proven).
* The scales are estimates from the inputs: a heat-flux case's ΔT and a thrust fan's U
  come from correlations and momentum theory; the run, not the plan, gives the
  temperatures and speeds. In a closed room a thrust fan can drive the air several times
  faster than its own disc velocity, so the step shrinks and the run takes longer than the
  first estimate (the estimate learns from each run).
* Lattice-Boltzmann solvers and neural-operator surrogates are not in this release.
