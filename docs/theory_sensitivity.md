# Shape sensitivity maps and shape optimisation

Aerolume's CFD solver is differentiable (docs/theory_ns2d.md §13), so a single reverse sweep
through the simulation tells you, for every point of the outline at once, **how much the drag
(or lift, L/D, moment) changes if the surface moves outward there**. That is the *sensitivity
map* shown on the Results page and in reports, and it is the gradient that the CFD shape
optimiser follows (the airfoil optimiser of section 7.1 uses the polar solver instead). This page explains what the map is, how it is computed, how it was verified, and where
it stops being trustworthy. Sections 1–8 describe the 2D maps; section 9 the 3D maps of any
imported mesh, which use the same method on the 3D solver. Code: `src/aerolume/optimize/`
(`sensitivity.py`, `sensitivity3d.py`, `airfoil.py`, `shape.py`); tests: `tests/test_optimize_*.py`,
`tests/test_optimize3d*.py`.

## 1. Reading the map

* **Colour = dJ/dn**, J being the chosen objective (Cd, Cl, L/D or Cm; Setup → *Sensitivity map*).
  **Positive (red): pushing the surface outward here — adding material — increases J.** Negative
  (blue): adding material here decreases J. So to *reduce drag*, push blue regions out and pull
  red regions in; to *raise lift or L/D*, do the opposite. The arrows in the figure point the
  improving way.
* **Units: 1/m.** The value is `L_ref · g(s)`, where `g` is the shape-derivative density defined
  by `dJ = ∮ g(s) δn(s) ds`. In words: the change of J per metre of outward displacement applied
  over a stretch of outline one reference length (chord, vehicle length) long. For a small change
  `δn_i` [m] at map points spaced `ds_i` [m]:

  ```
  ΔJ ≈ (1 / L_ref) · Σ_i sens_i · δn_i · ds_i
  ```

  Example: NACA 0012, Re 1000, 5°, L/D map, trailing-edge region peak ≈ 300 1/m: a 1 mm bump
  over 2 cm of the lower surface there changes L/D by about 300 × 0.001 × 0.02 / 1 = +0.006
  (out of 1.35) — as an upper estimate, see the next point.
* **Read the pattern, not the size.** Signs, peaks and the places that matter most are right;
  the magnitude can be about **twice** the physical derivative (§5: the finite window adds about
  20 % at the default window, the immersed boundary's response to sub-cell shape changes about
  1.5×). The Results page and the report therefore show the map relative to its 98th-percentile
  magnitude; the values in 1/m stay in the exports and in `SensitivityMaps`.
* **It is local and linear.** It predicts the effect of *small* changes (a few per cent of the
  local thickness, a fraction of a grid cell to a few cells). Large changes move separation
  points and shedding, and the map of the new shape is different. Re-simulate any modified design.
* **It belongs to the simulated flow**, with the same validity as the forces (Re_sim, grid,
  2D; docs/theory_ns2d.md §12). For high-Re airfoils the CFD map is qualitative; the airfoil
  optimiser (§6) uses the viscous panel solver instead.

## 2. What exactly is differentiated

The developed flow `(u₀, v₀)` (the state at the end of the run) is held fixed. From it the solver
runs `N = N_lead + N_avg` steps at a fixed time step with the shape displaced by `d_i` along the
outward normal of every outline vertex, and

```
J(d) = F( mean over the last N_avg steps of [Cd, Cl, Cm] )          F = Cd, Cl, Cl/Cd or Cm
```

The map is `∂J/∂d` at `d = 0`. Three choices define it:

* **Fixed starting state.** The flow does not "know" the new shape until it is simulated with
  it. The *lead* time lets it respond before forces are counted: right after a shape change the
  penalisation removes the momentum of fluid that suddenly finds itself inside the body, an
  impulsive transient that is not part of any steady sensitivity.
* **Fixed time step.** With the CFL rule choosing `dt` inside the window, `dt` would be a
  non-smooth function of the shape through `max |u|` (theory_ns2d §13). The window uses 0.9 × the
  CFL step of the starting state.
* **Fixed mask window.** Masks live on a fixed cell window around the bodies (with 4 spare cells),
  so displaced shapes are evaluated on exactly the same discrete function. That is also what
  makes finite differences a valid check (§4).

**Parametrisation.** Each outline is refined so that vertices are at most half a cell apart. The
inserted points lie on the original edges, so the shape — and every mask — is unchanged; only the
displacement description gets finer. Vertex `i` moves along its normal (the central-difference
tangent rotated outward), which displaces the outline by a hat function of the arc length.

**From vertices to a density.** `∂J/∂d_i ≈ g(s_i) · Δs_i` with `Δs_i` the vertex's dual length
(half its two edges), so the density is the ratio of the two, smoothed along the outline with a
Gaussian of σ = 1.5 cells:

```
g(s) = Σ_i K(s − s_i) ∂J/∂d_i  /  Σ_i K(s − s_i) Δs_i
```

The smoothing is necessary. The mask is a one-cell-wide band, so each sub-cell-spaced vertex is
felt by only the one or two staggered faces in its part of the band: per-vertex values alias with
the grid (spikes of several times the local mean next to near-zeros). Their *integral* against any
smooth displacement is accurate (§4); the smoothed density is the readable version of the same
information. The kernel-ratio form stays well defined where vertices cluster (an airfoil nose with
cosine spacing) and reduces to `∂J/∂d_i / Δs_i` as σ → 0. The density is evaluated at the points
`FlowSolver.surface()` returns, so it lines up with the cp and cf distributions.

## 3. The reverse sweep

The gradient flows `d → polygon → signed distance → C¹ masks χ_u, χ_v (+ solid velocity, force
weights) → implicit penalisation → projection → force impulse`. Everything in that chain is pure
torch (theory_ns2d §13); upwind `|V|` and the van Leer dye limiter are the only kinks, and the dye
is switched off in the window.

* **Memory.** The forward pass keeps the velocity every ⌈√N⌉ steps; the reverse pass recomputes
  one segment at a time and pulls the adjoint `(λ_u, λ_v)` back one step at a time with a
  vector-Jacobian product of a single solver step. Memory is about `2√N` velocity fields plus one
  step's intermediates. The mask tensors are leaves of this sweep; their gradients are summed over
  all steps and pushed through the geometry build (SDF) once at the end.
* **Speed.** A step launches a few hundred small kernels, so eager autograd on a GPU is launch
  bound (~24 ms forward + ~50 ms backward per step on a 50 k-cell grid). The step — forward *and*
  backward — is therefore captured as a CUDA graph (`torch.cuda.make_graphed_callables`): 3.4 ms
  per reverse step and 0.8 ms per forward step on the same grid, results bit-identical to eager.
  On a CPU the sweep runs eagerly.
* **LES.** The cell-centre solid fraction `χ_c` only switches the eddy viscosity off *inside*
  the body, so the map holds it fixed; everything that places the wall is differentiated. With
  WALE (the planner's model) the gradients are finite (checked on NACA 0012 at Re 10⁵). When
  this module was written, two details of the solver made other choices impossible: the
  switch-off was an in-place update that autograd could not differentiate with respect to
  `χ_c`, and Vreman (and Smagorinsky wherever the strain is exactly zero) took a square root at
  zero, whose derivative is infinite — measured NaN for Vreman. The released solver
  forms the switched-off viscosity before the in-place write and adds 10⁻³⁰ under both square
  roots (forward values change by about 10⁻¹⁵ relative), so both are now differentiable; the
  maps of Smagorinsky and Vreman runs have not yet been checked against finite differences,
  and the driver still stops with an explanation if a gradient comes out non-finite.

Measured cost (RTX 4060 Ti, float32, CUDA graphs):

| case | grid | steps (lead + avg) | time | peak GPU memory |
|------|------|-------------------:|-----:|----------------:|
| cylinder Re 30, 16 cells/D, float64 | 320 × 160 | 264 | 4.3 s (including graph capture) | 72 MB (process) |
| cylinder Re 30, 48 cells/D | 960 × 480 | 2732 | 30 s | — |
| NACA 0012 Re 1000 5°, Draft plan, L/D | 512 × 256 | 2019 (6 + 2 c/U) | 27 s | 154–176 MB (process), +111 MB for the sweep |

On a CPU (4 threads, float64) a reverse step costs ~120 ms on a 51 k-cell grid; the CPU default
window is therefore shorter (§5). NACA 0012 Draft on the CPU (320 × 160), developing the flow
included: 67 s for the 3 + 1 c/U window (602 steps).

## 4. Verification against finite differences

`finite_difference_check` compares `Σ_i ∂J/∂d_i φ_i` from the reverse sweep with the central
difference `(J(+εφ) − J(−εφ)) / 2ε` of *the same* windowed objective, along smooth displacement
modes φ (periodic Gaussian bumps, `normal_basis`, 8 per outline, mode 0 centred on vertex 0).
float64, ε = 10⁻⁴ L.

**Cylinder, Re 30, 16 cells/D, steady** (domain −5…15 × −5…5 D, window 2 + 2 D/U, 264 steps):

| mode (centre) | 0 (rear) | 1 | 2 (top) | 3 | 4 (front) | 5 | 6 (bottom) | 7 |
|---|---|---|---|---|---|---|---|---|
| autograd | 0.98936 | 2.26776 | 3.73012 | 2.31388 | 0.91026 | 2.31388 | 3.73012 | 2.26776 |
| central FD | 0.98936 | 2.26773 | 3.73002 | 2.31386 | 0.91026 | 2.31386 | 3.73002 | 2.26773 |
| rel. error | 4e-6 | 1.3e-5 | 2.7e-5 | 9e-6 | 5e-6 | 9e-6 | 2.7e-5 | 1.3e-5 |

Signs agree on every mode and the map is symmetric top/bottom (modes 1/7, 2/6, 3/5 identical to
round-off). float32 autograd reproduces the float64 values to 1e-6.

**NACA 0012, Re 500, 4°, Draft plan** (512 × 256, 64 cells/c, window 2 + 2 c/U, 972 steps):
Cd sensitivity within 1 % on all eight modes (largest error 0.98 % on the mode centred on the
trailing edge, others 0.01–0.4 %); Cl sensitivity within 0.1 % except the three modes touching the
trailing edge (3.2, 4.6, 5.4 %). Signs agree everywhere. The trailing-edge discrepancy is a
property of the discrete function, not of the gradient: inside the thin trailing edge some grid
points switch between the upper and the lower surface as their nearest one, so the signed
distance has a kink along the camber line and a central difference straddling it averages two
one-sided slopes. The same mechanism makes per-vertex values on a symmetric body symmetric only to
~10⁻³ of the peak (points exactly on the symmetry line are equidistant from two mirrored segments),
while smooth modes are symmetric to 3·10⁻⁵.

The fast test suite repeats these checks on a 48 × 32 grid (Cd, Cl and L/D, agreement 10⁻⁴),
checks the mirror symmetry, CUDA graph vs eager, determinism and cancellation; the slow suite runs
the two cases above.

## 5. How close is the map to the physical derivative?

Two things separate the map from the derivative a designer wants — the change of the
*converged* force when the real shape changes: the averaging **window** (this section's first
part) and the **immersed boundary** (its second part). The first is a few to twenty per cent at
the defaults; the second about 1.5×. Together the map's magnitude can be about twice the
physical slope, which is why it is presented as a pattern (§1).

**Window length.** The windowed derivative approaches the relaxed one as the lead grows: the
state at the end of the lead has partially relaxed to the new shape's steady flow. Measured on
the steady cylinder (Re 30, 16 cells/D) against central differences of fully re-converged steady
drag (120 D/U per evaluation, drift < 10⁻¹¹), with sub-cell steps (ε = 10⁻⁴ L, as in §4) — so this
table measures the window's convergence, not the error against the physical slope:

| window (lead + average, D/U) | 0.5 + 0.5 | 1 + 1 | 2 + 2 | 3 + 1 | 4 + 4 | 6 + 2 | 7 + 1 | 8 + 4 | 8 + 8 | 10 + 2 | 16 + 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed / steady (rear, top, front modes) | 1.27–1.72 | 1.22–1.26 | 1.15–1.17 | 1.13–1.15 | 1.07–1.08 | 1.05–1.06 | 1.04–1.06 | 1.02–1.04 | 1.02–1.03 | 1.02–1.03 | 1.00–1.02 |

* The windowed value over-estimates the steady one by a factor that is **nearly the same all
  around the outline** once the lead is ≥ 2 body time units, so the *pattern* of the map is right
  early; the magnitude converges with the lead.
* For the same cost, a long lead and a short average is better (6 + 2 beats 4 + 4).
* The relaxation time grows with the Reynolds number and the resolution. On the steady cylinder
  at Re 40 and 40 cells/D (domain 10 D up, 20 D down, 20 D tall; 1250 × 800 cells) the window's
  share is about **+23 % at 6 + 2** and +9 % at 16 + 4 (fit through 6 + 2, 16 + 4 and a 60 + 4
  window: relaxation time ≈ 12 D/U).
* **Defaults:** steady flows 6 + 2 body time units on a GPU, 3 + 1 on a CPU, where the body time
  unit is chord / U (airfoils) or frontal height / U (vehicles, bluff bodies); `window=`
  overrides. The magnitude they give is indicative only (next paragraph).

**The immersed boundary.** The masks place the wall to a fraction of a cell, but the discrete
objective J(R) is not smooth on the cell scale: it has a ripple with the cell size, so its
*local* derivative — the one the adjoint returns, and the one a sub-cell finite difference
measures — is larger than its slope over whole cells, which is what a real (resolved) shape
change produces. Measured on the same Re 40 cylinder, uniform dilation dCd/dR:

| quantity | dCd/dR |
|---|---|
| three `run_cfd` runs, R = 0.475 / 0.5 / 0.525 (whole-cell steps) | 2.33 |
| relaxed windowed J (60 + 4), central difference over 1 cell | 2.23 |
| same, over 0.5 cell | 2.24 |
| same, over 0.25 cell | 2.75 |
| same, over 0.08 cell | 3.53 |
| adjoint, window 16 + 4 | 3.82 |
| adjoint, window 6 + 2 (the default) | 4.32 |

The adjoint agrees with a finite difference of its own objective (4.14 at 6 + 2, ε = 0.08 cell),
so the reverse sweep is right; 4.32 / 2.33 ≈ 1.85 splits into ≈ 1.23 from the window and
≈ 1.5–1.6 from the sub-cell response. The map's pattern (sharp peaks at the top and bottom,
slightly negative at the front stagnation point) is unaffected. Whether the ripple factor is the
same for localised modes as for uniform dilation, and at other Reynolds numbers and grids, has
not been measured; smoothing the mask's sub-cell response is the open fix.

Grid dependence (cylinder, window 4 + 4 D/U, density averaged over 10°-wide bumps): at 16, 32 and
48 cells/D the map has the same shape — sharp peaks at the top and bottom of the cylinder, slightly
negative at the front stagnation point, ~0 in the rear — and its peak value falls by ~8–13 % per
refinement (16.7, 15.5, 13.5 in non-dimensional units) while Cd changes by 0.6 %. The shape is
settled on Draft grids; the magnitude is good to about ten per cent there.

## 6. Unsteady flows: what the map means there

For a shedding flow there is no steady state; the derivative of a short time average also
contains how the shape shifts the phase of the shedding, and for chaotic flows the adjoint of a
long window grows exponentially (Lea, Allen & Haine 2000). Aerolume therefore:

* detects shedding from the lift history (dominant frequency of the run);
* uses a short lead (2 body time units, at most one period: it only has to let the start-up
  impulse of the shape change pass) and **whole periods** of averaging covering ≥ 4 body time
  units (2 on a CPU), so the average does not depend on where in the cycle it ends;
* averages the maps of **3 windows started at evenly spaced phases** of one period (an
  "ensemble adjoint"), costing 3×;
* attaches a warning to the result saying the map is indicative.

Measured on NACA 0012 at Re 10⁵, 4°, Draft (WALE LES, shedding at St ≈ 1.23 based on the chord):
with single windows of arbitrary length the dominant mode values changed sign between windows;
with whole-period windows started at six phases, the scatter over phases is 50–80 % of the mean
on the dominant modes for 3 periods and 20–30 % for 6 periods, and the mean pattern (largest drag
sensitivity on the nose and the forward upper surface) is the same for both. Read such maps as
"where the shape matters", not as numbers.

## 7. Shape optimisation

### 7.1 Airfoils at flight Reynolds numbers: `optimize_airfoil`

CST (Kulfan 2008) perturbations of order 4 per surface (10 variables) are *added to the original
coordinates*, `y += sqrt(x)(1 − x) Σ Δa_i B_i(x)`, so the baseline is reproduced exactly and the
leading edge, trailing edge and chord line (the angle-of-attack reference) never move. Aerodynamics
from the viscous panel solver (mfoil, docs/theory_airfoil.md); gradients by forward differences
(Δa = 10⁻³, a few 10⁻⁴ c of surface movement, where measured Cd is smooth); optimiser SLSQP with
the objective scaled so that its first step is 0.01 in a coefficient. Constraints, all linear in
the variables: max thickness ≥ `min_thickness` (default: the original's), local thickness ≥ 50 %
of the original between 2 and 95 % chord, area ≥ original (`keep_area`), each coefficient change
within ±0.06. `target_cl` makes the angle a variable with Cl = target as an equality constraint and
minimises drag. Analyses that do not converge are infeasible: the objective returns a penalty bowl
centred on the best shape so far, SLSQP is restarted from the best shape if it stalls there, and
only converged, feasible shapes are ever returned.

Measured (NACA 0012, Re 10⁶, 4°, maximise L/D, default constraints): L/D 57.9 → 60.7 → 75.1 →
86.8 after 1, 2, 3 iterations (44 analyses, 154 s on the development machine), Cl 0.419 → 0.553,
Cd 0.00723 → 0.00637, max thickness 12.0 → 12.8 %, 1.0 % camber; the upper-surface transition
moved from 26 % to 42 % chord. The test suite runs the optimiser against an analytic stand-in
(constraints, failure handling, `target_cl`, cancellation, determinism) and the real solver in the
slow suite.

### 7.2 Any shape with the CFD gradient (experimental): `optimize_shape_cfd`

Design variables are amplitudes of 12 smooth normal-displacement modes per body (wheels excluded);
corners sharper than 30° (an airfoil's trailing edge, a car's sharp edges) are held fixed, because
the normal is undefined there and moving a sharp trailing edge along its bisector splits it into a
fishtail. Each iteration: one reverse sweep (§2–3) gives the gradient of the windowed objective
with respect to all amplitudes; with `keep_area` the step is projected so that the area does not
change to first order and a uniform offset then restores the exact area; the step moves the
surface by at most 0.75 cells, is accepted if the windowed objective of the displaced shape is
better (halved up to twice otherwise), and the flow is then advanced on the new shape for one lead
time so the next gradient starts from an adapted flow. Total movement is capped at 4 cells.

Measured (NACA 0012, Re 500, 4°, Draft, maximise L/D, area held): three steps in 43 s on the GPU;
windowed L/D 1.049 → 1.540; an independent full `run_cfd` of both shapes gives **1.035 → 1.518
(+47 %)**, area change 3·10⁻⁶. The optimiser added aft camber and thickened the region just ahead
of the (pinned) trailing edge — at Re 500 a blunt, cambered tail is worth more lift than it costs
in drag. There are no thickness or curvature constraints beyond the area and the pinned corners,
so results can be unconventional: always confirm with a normal run (`result.extra["design"]` is a
ready-to-simulate design).

## 8. Limits, in one place

* The map is the derivative of a short-window time average from a fixed developed flow, taken
  through a diffuse wall: its pattern is reliable, its size can be about twice the physical
  derivative for steady flows (§5) and it is indicative only for shedding flows (§6).
* Shape changes act through a diffuse one-cell interface: sub-cell features and gaps under ~2
  cells cannot be resolved; per-vertex values alias with the grid (hence the 1.5-cell smoothing).
* Near sharp trailing edges thinner than ~2 cells the discrete objective has kinks (§4): the map
  there is a one-sided derivative, verified to ~5 %.
* LES: the sub-grid model is differentiated except for its switch-off inside solids; only WALE
  (the planner's choice) gives finite gradients everywhere.
* Everything in docs/theory_ns2d.md §12 (2D, Re, grid, blockage) applies unchanged.

## 9. Three dimensions: maps on any imported mesh

`optimize.sensitivity3d` applies sections 2–6 to the 3D solver (docs/theory_ns3d.md §11): the
AirShaper-style adjoint map of a car, a drone, a rider or a wing, one value per vertex of the
user's mesh. Reading it is as in §1: **red (positive) means pushing the surface outward there
increases the objective** (CD, CL, L/D or Cm of the full body), blue means it decreases it.

### 9.1 What is differentiated

The developed flow `(u₀, v₀, w₀)` of the run is held fixed; from it the solver runs
`N = N_lead + N_avg` steps at a fixed time step (0.9 × the CFL step of that state) with the
masks on a fixed window (the bodies, their mask band and 2 spare cells), and J is the objective of the window-mean
full-body coefficients — the same windowed objective and default windows as in 2D (§2, §5, §6):
6 + 2 body time units on a GPU, 3 + 1 on a CPU, whole periods from 3 phases for shedding flows.

**Parametrisation.** Imported meshes come at any resolution — a box of 12 triangles, a scan of a
million. Each body mesh is refined by *conforming edge bisection* until no edge is longer than
half a cell: every edge that is too long gets its midpoint, shared by the faces on both sides
(a welded, closed mesh stays welded and closed), each face is split into 2, 3 or 4 children with
its winding kept, and this repeats until all edges are short. Every new vertex lies on an
original edge, so the shape — and every mask — is exactly unchanged, and the user's vertices keep
their indices. Each refined vertex moves along its outward normal, the area-weighted mean of
its faces' normals. The faces are first turned towards the fluid one by one (the solid fraction is
compared one cell outside and one cell inside each face), because imported meshes are often
wound inconsistently and averaging opposite face normals would give meaningless vertex normals
(test: a sphere with 40 % of its faces flipped gets the same normals as the clean one).

The chain `vertex displacements → sdf_band (the shared voxeliser: nearest triangle searched
without autograd, its distance recomputed with autograd) → C¹ masks at the u, v, w faces →
implicit penalisation → projection → force impulse` is pure torch. The fused CUDA kernels of the
production solver have no autograd, so the window runs the reference core (`step3d_full`).

### 9.2 From vertices to a map

`∂J/∂d_i = ∫ g φ_i dA` for the hat function `φ_i` of vertex `i`, so the shape-derivative
density `g` (`dJ = ∬ g δn dA`) is the derivative divided by the vertex's area. As in 2D the
per-vertex values alias with the grid, so derivatives and areas are both smoothed and their
ratio taken:

```
g_i = (S^k ∂J/∂d)_i / (S^k A)_i ,      S = (1 − λ) I + λ D⁻¹ W
```

`S` is a lazy random walk on the refined mesh (`W`: shared edges, plus vertices at the same
position, so meshes that repeat vertices along hard edges stay connected), i.e. `k` explicit
steps of the surface Laplacian. After `k` steps the walk has spread by `σ² = kλ⟨ℓ²⟩/2` per
surface direction, so `k` is chosen for σ = 1.5 cells (the 2D smoothing; 44–99 steps at λ = ½ in
the cases below, a fraction of a second). The
operator is row-stochastic, so the ratio is a kernel-weighted average of `∂J/∂d_i / A_i`, which
reproduces a constant density exactly (test) — the 3D form of the 2D kernel ratio. It follows the
surface: it never mixes the two sides of a thin plate or two bodies across a gap.

**Units:** `sens = L_ref² · g` in 1/m (2D: `L_ref · g`; both are the non-dimensional density
divided by `L_ref`). For a displacement `δn_i` [m] of vertices with areas `dA_i` [m²]:

```
ΔJ ≈ (1 / L_ref²) · Σ_i sens_i · δn_i · dA_i
```

**Symmetry plane.** A half-domain run sees only `y ≥ 0`, and its J is the full body's: moving a
visible vertex moves its mirror image too, so the half-domain derivative is that of a
mirror-symmetric pair — twice the one-sided value — and is halved. Three details make the
half-domain map equal to the full-domain one: the area `A_i` is the *visible* part of the hat
function (faces crossing the plane are clipped exactly — the cells feel only their visible part,
but its gradient reaches all three corners); before smoothing, every vertex receives its mirror
partner's derivative and visible area, so the smoothing kernel straddles the plane as in a full
run instead of being cut off by it; and vertices on the missing half take their mirror image's
value. Measured (tiny sphere, float64): J equal to 10⁻¹⁰, the derivatives along mirror-symmetric
modes equal to 10⁻⁹, the maps equal to 1.4 × 10⁻⁵.

### 9.3 The reverse sweep, memory and cost

The sweep is §3 in 3D: the velocity every ⌈√N⌉ steps (moved to host memory when the GPU is
short), one recomputed segment, and a vector-Jacobian product of one step at a time. It is not
`ns3d.rollout3d`: that one's `torch.utils.checkpoint` chunks run eagerly and re-record a whole
chunk's activations at once. One step — forward and backward — is captured as a CUDA graph:
8.3 ms per VJP at 194 k cells against 120 ms eager, identical results (float32).

Measured (RTX 4060 Ti, float32 unless stated; "extra" is the peak above the memory in use before
the sweep, the flow solver included in "before"):

| case | cells | steps (window) | wall time | extra GPU memory |
|---|---:|---:|---:|---:|
| sphere 12 cells/D, float64 | 194 k | 102 (1 + 1) | 5.4 s | 250 MB |
| sphere 12 cells/D | 194 k | 102 (1 + 1) | 4.0 s | 143 MB |
| sphere 24 cells/D | 1.56 M | 258 (1 + 1) | 25.8 s | 886 MB |
| sphere 24 cells/D, default window, GPU shared with another job | 1.56 M | 1026 (6 + 2) | 183 s | 1276 MB |
| the same in the slow test, GPU less loaded (flow development included) | 1.56 M | 1026 (6 + 2) | ≤ 104 s | — |

That is about 130 bytes per cell for the differentiated step plus 12 bytes per cell for each
velocity state held (`estimate_memory_bytes` uses 200 + 12 per state, plus the mask window and a
fixed 256 MB, so it over-estimates by ~1.5–2×). A sweep step costs ~55 ns per cell on an idle GPU
(60–115 ns while another job shared it). On the CPU (4 threads, eager) it costs 5.2 µs per cell:
34 steps of a 58 k-cell sphere took 10 s, so a 0.4 M-cell Draft map with the CPU window takes
several minutes.

**Budget.** DESIGN 7.1 plans sensitivity runs of up to ~1.5 M cells. `compute_sensitivity3d`
refuses — before any costly work, with `SensitivityBudgetError` (a `ValueError` whose message says
what to change), which the 3D runner turns into a warning on an otherwise complete result — when
the grid has more than 2.1 M cells on a GPU (0.4 M on a CPU; `max_cells=` overrides) or when the
estimated memory exceeds 85 % of the free GPU memory even with the checkpoints on the host. Every
Draft grid passes; at 2.1 M cells and a 1200-step window the estimate is 2.4 GB.

### 9.4 Verification against finite differences

`finite_difference_check3d` compares `Σ ∂J/∂d_i φ_i` with central differences of the same
windowed J along smooth modes φ: Gaussian bumps (`normal_modes3d`, width 0.15 D, mirror-symmetric
about the plane) and a uniform dilation.

**Sphere, Re 30, 12 cells/D, half domain** (box −3…6 × 0…2.5 × −2.5…2.5 D, icosphere of 1280
faces refined to 5120, window 1 + 1 D/U, float64, ε = 10⁻⁴ D):

| mode | front | rear | top | bottom | side | front shoulder (top) | front shoulder (side) | aft shoulder | dilation |
|---|---|---|---|---|---|---|---|---|---|
| autograd | −0.057526 | 0.0089821 | 0.80046 | 0.80046 | 1.51113 | 0.20951 | 0.41436 | 0.18623 | 8.9365 |
| central FD | −0.057526 | 0.0089821 | 0.80045 | 0.80045 | 1.51111 | 0.20950 | 0.41436 | 0.18623 | 8.9364 |
| rel. error | 7e-7 | 9e-8 | 1.3e-5 | 1.3e-5 | 1.3e-5 | 1.2e-5 | 4e-7 | 1.2e-5 | 1.2e-5 |

(The side mode is a mirror pair of bumps, hence about twice the top one.) float32 autograd gives
the float64 values to 10⁻⁶; float32 finite differences are noisy (up to 3 % on the tiny rear
mode). At 24 cells/D (1.56 M cells, float32, ε = 10⁻³ D) the dominant modes agree to 0.3–0.9 %,
the rear mode (1.5 % of the largest) to 24 % — float32 differences, not the gradient.

**The drag map looks axisymmetric**, as the flow is: mean density in 15° bands of the polar
angle θ from the front stagnation point, at the top (+z) and the side (+y) of the sphere
(12 cells/D):

| θ | 0–15° | 15–30° | 45–60° | 60–75° | 75–90° | 90–105° | 105–120° | 120–135° | 165–180° |
|---|---|---|---|---|---|---|---|---|---|
| top | −0.466 | −0.275 | 2.205 | 4.314 | 5.693 | 5.519 | 3.707 | 1.804 | 0.032 |
| side | −0.462 | −0.245 | 2.198 | 4.051 | 5.486 | 5.263 | 3.447 | 1.784 | 0.029 |

Where the map is large the top and side bands differ by up to 7 %: the faceted icosphere meets
the flow differently there (its edges run streamwise at ±y and spanwise at ±z), and a derivative
feels that more than the drag does. A full-domain run gives exactly the same bands as the
half-domain run (to the three printed decimals); −y equals +y exactly, and the bottom equals the
top where the map is large (1.3 % apart in the small front band, where mirrored faces were split
along different diagonals during the refinement).
The pattern is the 2D cylinder's: slightly negative at the front stagnation point (a fuller
nose lowers the drag), the largest values around the equator (a wider sphere has more drag),
almost nothing at the rear.

**Short box with sharp edges** (0.5 × 1 × 1 D, pitched 8° nose-up, Re 30, 12 cells/D, same domain
and window, float64, ε = 10⁻⁴ D; CD and CL, bumps centred on faces, edges and a corner). The side
faces lie *exactly* on a plane of v-faces of the grid (y = ±0.5 D at 12 cells/D), the hard case for
an immersed-boundary derivative:

| mode | front | rear | top | bottom | side | front-top edge | front-bottom edge | front-side edge | corner | rear-top edge | dilation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CD rel. error | 1.5e-6 | 4.3e-6 | 5.8e-6 | 6.0e-6 | 3.4e-6 | 9.7e-7 | 2.1e-5 | 1.9e-7 | 2.1e-5 | 6.5e-6 | 3.4e-6 |
| CL rel. error | 7.8e-6 | 8.0e-6 | 3.3e-6 | 3.6e-6 | 2.4e-5 | 9.9e-7 | 1.2e-5 | 4.9e-4 | 3.2e-4 | 3.5e-5 | 6.2e-4 |

Signs agree on every mode (e.g. CL front-side edge −0.067798 vs −0.067831, corner 0.61912 vs
0.61892; 101 s for both objectives on the GPU). Before the voxeliser was changed the same case read
up to 1.5 % (CD) and 14 % (CL, front-side edge; 7.5 % dilation, 3.2 % corner), all on modes touching
the side faces: the shared signed distance was `±√(d²)` with the sign from the voxeliser's
ray-parity test, so a grid point exactly on a face lost its derivative (`√` of a clamped zero) and a
point within the parity test's ray jitter (0.002 cells) could take the other side's sign. Moving
the box 0.014 cells off the grid planes removed the effect (≤ 7 × 10⁻⁵ on every mode), which located
it. The voxeliser now makes the distance differentiable *through* the surface
(`geometry3d.voxel`, DESIGN.md 7.4): a cell whose nearest feature is a face interior within
0.02 cells of its plane gets the face's **signed plane distance** `s n̂·(p − a)`, with the
side `s` voted by the band cells clearly off that face (parity sign times plane side), and a cell
exactly as far from several faces (a medial plane, an edge) takes the mean of their derivatives,
which is what a central difference across the kink measures. The remaining 3–6 × 10⁻⁴ (CL: the
front-side edge, the corner and the dilation) all involve the corners, where three faces meet and
no single derivative matches a difference in every direction; that is the likely cause, not a
measured one. The slow test checks every mode of this case at 3 × 10⁻³, and
`tests/test_geometry3d_voxel.py` checks the distance's derivative for centres on and within the ray
jitter of a face; the fast sensitivity tests keep their bodies off the grid lattice.

The fast test suite repeats the checks on a 5-cells-per-diameter sphere on the CPU (CD on full
and half domains, CL, Cm and L/D on a pitched box, two bodies with a spinning wheel; required
agreement 10⁻⁵, measured ~10⁻⁸ on the sphere), and checks the mesh refinement, the smoothing, the symmetry plane, the winding
independence, the driver, the budget guard and CUDA graphs against eager; the slow suite
(`tests/test_optimize3d_validation.py`) runs the sphere and box cases above and the driver at
1.56 M cells.

### 9.5 How close is the 3D map to the physical derivative?

The §5 study on the sphere (half domain, 16 cells/D, 0.46 M cells, float32): uniform dilation
dCD/dR (fixed reference area) from the adjoint at three windows, from central differences of the
same windowed J over sub-cell to whole-cell steps, and from separate steady runs (40 D/U each,
mean of the last 5) with R ± ½ cell and ± 1 cell:

| case | adjoint 6 + 2 | adjoint 16 + 4 | adjoint 30 + 4 | windowed J, central difference over 0.08 / 0.25 / 0.5 / 1 cell | steady runs, R ± ½ / ± 1 cell | map ÷ physical |
|---|---|---|---|---|---|---|
| Re 30, 16 cells/D | 7.403 | 7.385 | 7.385 | 7.28 / 7.00 / 7.01 / 7.01 | 6.99 / 6.99 | 1.06 |
| Re 30, 24 cells/D (1.56 M cells) | 6.942 | 6.925 | — | — / 6.92 / — / 6.96 | 7.00 / 6.94 | 0.99–1.00 |
| Re 100, 16 cells/D | 4.825 | 4.697 | 4.694 | 4.65 / 4.11 / 3.88 / 3.90 | 3.77 / 3.79 | 1.28 |

"map ÷ physical" is the default (6 + 2) adjoint over the steady runs' whole-cell slope. A rough
check of the steady slopes: with a fixed reference area `dCD/dR = CD (2 + d ln CD / d ln Re)/R`;
the slope of the standard drag curve (Clift et al.) and the runs' own CD give 6.4 at Re 30 and
3.7 at Re 100, against 6.99 and 3.78 measured (the confined box may account for the
difference; not checked).

* **The window** matters much less than on the 2D cylinder: the 3D sphere flow relaxes within a
  few D/U (6 + 2 is within 0.3 % of 30 + 4 at Re 30 and 2.8 % at Re 100; 2D: +23 % at Re 40).
* **The immersed boundary's sub-cell response** is the larger share, as in 2D, and it grows with
  the Reynolds number and shrinks with the resolution: none at Re 30 and 24 cells/D, 6 % at
  16 cells/D, 24 % at Re 100 and 16 cells/D (sub-cell central differences of the windowed J rise
  from 3.9 to 4.65 as the step shrinks from 1 to 0.08 cells — the cell-scale ripple of §5).
* So the 3D map's size is closer to the physical derivative than the 2D one (1.0–1.3× here, 2D
  up to ~2×), but it was measured on one shape, one mode and two Reynolds numbers; every map keeps
  the warning, and the GUI and the report show it relative to its 98th-percentile magnitude.

**Grid dependence** of the same windowed derivative (sphere, Re 30, 1 + 1 window): from 12 to
24 cells/D the dilation value falls from 8.94 to 7.84 and the top-bump mode from 0.80 to 0.69
(−12 %), with the same pattern; at the default window the dilation value falls 6 % from 16 to
24 cells/D (7.40 → 6.94) while the steady slope stays at 6.99 → 6.94–7.00 — the map's size
converges to the physical slope as the grid is refined.

### 9.6 Limits (3D)

* Everything in §8 and docs/theory_ns3d.md §12 applies: the grid resolves only bodies ≥ ~24
  cells across well, the Re of vehicles needs wall-unresolved LES, and the map belongs to the
  simulated flow.
* Faces lying exactly on grid planes (§9.4) are handled since the voxeliser's distance became
  differentiable through the surface (box case ≤ 6 × 10⁻⁴); where three faces meet at a corner on
  the grid lattice a few 10⁻⁴ remain.
* Open sheets are thickened to one cell by the voxeliser: their map is that of the thickened
  sheet, and both sides of a sheet are one surface.
* Not verified in 3D: maps of shedding or LES flows (the 2D findings of §6 are the guide), maps
  on car-sized meshes, wheels on a moving road (the rotating solid velocity is differentiated and
  passes the fast finite-difference test, nothing more), and the CPU cost above 58 k cells.

## 10. API

```python
from aerolume.optimize import compute_sensitivity, attach_sensitivity, optimize_airfoil, optimize_shape_cfd
maps = compute_sensitivity(design, sim, objective="cd", plan=plan, state=solver)   # list[np.ndarray] + metadata
attach_sensitivity(result, maps)                     # fills SurfaceData.sensitivity, adds notes / warnings
r = optimize_airfoil(coords, 1e6, alpha=4.0, objective="ld", n_iter=8)          # OptimizationResult
r = optimize_shape_cfd(design, sim, n_iter=4)                                    # experimental

# 3D (section 9)
from aerolume.optimize import compute_sensitivity3d, attach_sensitivity3d, SensitivityBudgetError
maps = compute_sensitivity3d(design, sim, objective="cd", plan=plan3d, solver=solver3d)
                                                     # one array per mesh, per vertex, 1/m (L_ref^2 g)
attach_sensitivity3d(result3d, maps)                 # fills SurfaceData3D.sensitivity (by name,
                                                     # nearest vertex / mirror image if counts differ)
```

`run_cfd` calls `compute_sensitivity` with its own finished solver when
`sim.cfd.compute_sensitivity` is set (no second flow development). Lower-level pieces:
`WindowedObjective` (J(d), its gradient, `advance`, `rebase`), `finite_difference_check`,
`normal_modes`, `density_on_outline`, `choose_window`. In 3D: `WindowedObjective3D` (J(d),
`gradient`, `evaluate`, `advance`, `density`), `finite_difference_check3d`, `normal_modes3d`,
`refine_mesh`, `surface_smoother`, `visible_dual_area`, `estimate_memory_bytes`,
`estimate_seconds` (for the planner's run-time estimate) and `SensitivityBudgetError` (a
`ValueError`: the grid or the free GPU memory is over budget; raised before any costly work).

## 11. References

* Christianson (1994), Reverse accumulation and attractive fixed points, Optim. Methods Softw. 3.
* Drela (1989), XFOIL: an analysis and design system for low Reynolds number airfoils, Springer LNE 54.
* Griewank & Walther (2000), Algorithm 799: revolve — an implementation of checkpointing for the
  reverse or adjoint mode of computational differentiation, ACM TOMS 26.
* Jameson (1988), Aerodynamic design via control theory, J. Sci. Comput. 3.
* Kochkov et al. (2021), Machine learning–accelerated computational fluid dynamics, PNAS 118.
* Kraft (1988), A software package for sequential quadratic programming, DFVLR-FB 88-28 (SLSQP).
* Kulfan (2008), Universal parametric geometry representation method, J. Aircraft 45 (CST).
* Lea, Allen & Haine (2000), Sensitivity analysis of the climate of a chaotic system, Tellus 52A.
* Othmer (2008), A continuous adjoint formulation for the computation of topological and surface
  sensitivities of ducted flows, Int. J. Numer. Meth. Fluids 58.
