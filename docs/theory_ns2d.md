# The Aerolume 2D Navier–Stokes solver — theory, numerics and verification

Module: `aerolume.solvers.ns2d` (`config.py`, `core.py`, `poisson.py`, `geometry.py`,
`solver.py`, `rollout.py`; the placed physics of sections 15–23 in `physics.py`, `scalar.py`,
`particles.py`, `diagnostics.py`). Verification and validation scripts:
`validation/ns2d_cases.py`, `validation/thermal2d_cases.py`, `validation/bench_ns2d.py`; tests:
`tests/test_ns2d_core.py`, `tests/test_ns2d_solver.py`, `tests/test_ns2d_physics.py`,
`tests/test_ns2d_validation.py` and `tests/test_ns2d_physics_validation.py` (`-m slow`).

Part I (sections 1–14) is the validated flow solver. Part II (sections 15–23) adds the placed
physics of DESIGN.md section 10 — smoke emitters and tracer particles, heat transfer with
buoyancy, fans, jets and vents, painted obstacles, live editing, diagnostics — without
changing a Part I number.

This document explains every numerical choice, why it was made, and what was measured to
back it. All numbers quoted were produced on the development machine (RTX 4060 Ti 8 GB,
i7-13700F, torch 2.14.1+cu130, Windows 11) by the scripts above.

---

## 1. Equations and scaling

The solver integrates the incompressible Navier–Stokes equations with a Brinkman
penalisation term that represents solid bodies, in non-dimensional form

```
∂u/∂t + ∇·(u u) = −∇p + (1/Re) ∇²u + ∇·(2 ν_t S) − (χ/η)(u − u_s)
∇·u = 0
```

Lengths are scaled by the reference length `L` (chord, cylinder diameter, vehicle
length), velocities by the free-stream speed `U`, time by `L/U`, pressure by `ρU²`, so
`Re = UL/ν`. `χ ∈ [0, 1]` is the solid fraction, `u_s` the velocity of the solid (zero, or
rigid rotation for wheels), `η` the permeability of the penalisation, `ν_t` an optional
subgrid eddy viscosity and `S` the strain-rate tensor.

Flow always enters from the left along +x; angle of attack is applied by rotating the
geometry (DESIGN.md §3), so the x force is drag and the y force is lift.

Pressure coefficient: `cp = 2 (p − p_ref)` where `p_ref` is the mean pressure over the two
inlet columns (in periodic mode: the domain mean). Force coefficients use the reference
length `force_ref` (in units of L): `Cd = Fx / (½ force_ref)`, `Cl = Fy / (½ force_ref)`.
The pitching moment is about `moment_ref` and **positive nose-up**. For flow from the left
a nose-up rotation is *clockwise* in the x-right/y-up frame (the leading edge moves up),
so `Cm = −Mz / (½ force_ref)` where `Mz` is the usual counter-clockwise moment. With this
convention a positively cambered airfoil has a negative `Cm` about c/4 (verified, §11).

## 2. Grid and boundary conditions

Uniform staggered (MAC, Harlow & Welch 1965) grid of `nx × ny` cells of size `h` covering
`[x0, x0 + nx h] × [y0, y0 + ny h]`:

| quantity | location                         | shape         |
|----------|----------------------------------|---------------|
| `u`      | x-faces `(x0 + i h, y0 + (j+½) h)` | `(ny, nx+1)`  |
| `v`      | y-faces `(x0 + (i+½) h, y0 + j h)` | `(ny+1, nx)`  |
| `p`, `χ_c`, dye, ν_t | cell centres           | `(ny, nx)`    |

The staggered arrangement couples pressure and velocity without checkerboard modes and
makes the discrete divergence of the discrete gradient exactly the 5-point Laplacian, which
is what makes the projection exact (§4).

Boundary conditions are imposed through two layers of ghost values built by `pad_u` /
`pad_v` before every right-hand-side evaluation, so the interior stencils never branch:

* **Inlet** (x = x0): `u = 1` (or a parabola with discrete mean exactly 1 and peak 1.5 for
  channel benchmarks; Re is then based on the bulk velocity, as in Schäfer–Turek), `v = 0`
  (odd reflection of v). The inlet column is re-imposed from `ops.u_inlet` at every stage, so
  the solution is differentiable with respect to the inflow.
* **Outlet** (x = x0 + nx h): convective condition `∂u/∂t + U_c ∂u/∂x = 0` with `U_c = 1`,
  integrated by the same Runge–Kutta stages as the interior; v has zero gradient. After each
  stage the outlet column is shifted uniformly so that the outflow equals the inflow
  exactly (a device-side reduction, no host sync). Walls are impermeable, so the net
  boundary flux — and hence the sum of the Poisson right-hand side — is zero to round-off,
  which the all-Neumann pressure problem requires.
* **Top / bottom walls**: `v = 0` on the wall face; `u` ghosts mirror the first interior row
  so that the interpolated wall value is: `slip` → `∂u/∂y = 0`; `noslip` → `u = 0`;
  `moving` (bottom only) → `u = ground_speed` (the rolling road of vehicle studies).
* **Periodic** (verification mode): periodic in x and y, no inlet/outlet; the duplicate
  faces (`u[:, nx] = u[:, 0]`, `v[ny] = v[0]`) are kept bit-identical.

## 3. Spatial discretisation

**Advection** is written in conservative (divergence) form `∇·(u u)`; momentum fluxes are
evaluated at cell centres (normal fluxes) and cell corners (cross fluxes) with the
advecting velocity interpolated to the flux point, then differenced back to the faces.

* `central`: 2nd-order average of the transported velocity. On a staggered grid with a
  discretely divergence-free velocity this divergence form conserves kinetic energy
  (Morinishi et al. 1998), so the scheme adds no numerical dissipation — the right
  choice for DNS and verification. Verified: inviscid periodic random flow keeps its
  energy to 1 part in 10⁴ over 50 steps (the residual is the O(Δt³) RK3 error).
* `upwind3` (default): the transported value at a face between `b` and `c` (neighbours
  `a b | c d`) is the 3rd-order upwind-biased reconstruction `(−a + 5b + 2c)/6` for flow
  from the left (mirror for flow from the right). Written branch-free:

  ```
  φ_face = [7(b + c) − (a + d)]/12  +  sign(V) [3(b − c) − (a − d)]/12
  flux   = V · central4  +  |V| · dissipation
  ```

  i.e. a 4th-order central flux plus a 4th-difference dissipation proportional to |V|
  (the same structure as Kawamura–Kuwahara, with a 3× smaller coefficient). It is robust at
  cell Reynolds numbers far above 2, which makes under-resolved high-Re runs stable without
  smearing resolved scales the way 1st-order upwind would. Being smooth (|V| is the only
  non-smooth element) it is also differentiable almost everywhere.

**Diffusion**: 2nd-order 5-point Laplacian for the molecular part; the subgrid part is the
divergence of `2 ν_t S` with `ν_t` at cell centres (diagonal stresses) and averaged to cell
corners (shear stress), which conserves momentum and is dissipative.

Observed order (Taylor–Green vortex, periodic, Re = 10, t = 1, max velocity error):

| N   | central    | order | upwind3    | order |
|-----|------------|-------|------------|-------|
| 16  | 2.06e-3    |       | 7.53e-4    |       |
| 32  | 5.23e-4    | 1.97  | 2.77e-4    | 1.44  |
| 64  | 1.31e-4    | 1.99  | 9.50e-5    | 1.54  |
| 128 | 3.30e-5    | 2.00  | 2.80e-5    | 1.76  |

`central` is cleanly 2nd order. `upwind3` has two error terms of opposite sign — the
2nd-order diffusion/interpolation error and its 3rd-order upwind dissipation — which
partly cancel on coarse grids: its error is *smaller* than the central one at every
resolution and its observed order rises towards 2 as the 3rd-order term dies out. The
measured energy-decay rate at N = 64 is 0.9992 × the exact `4/Re` (the discrete Laplacian
eigenvalue `(2 − 2 cos h)/h²` is `1 − h²/12` of the exact one).

## 4. Pressure: exact projection with a direct DCT Poisson solver

Each Runge–Kutta stage ends with a projection: solve `L ψ = ∇_h·û` and set
`u = û − ∇_h ψ`, where `L = ∇_h·∇_h` is the 5-point Laplacian with homogeneous Neumann
conditions on all four sides. (Every boundary-normal velocity is prescribed, so the
correction must not change it: zero normal gradient of ψ.)

With Neumann conditions on a cell-centred grid the eigenvectors of `L` are exactly the
DCT-II basis functions, with eigenvalues `λ_k = (2 cos(πk/n) − 2)/h²` per direction. A 2D
forward DCT, a division by `λ_kx + λ_ky` and an inverse DCT therefore solve the *discrete*
problem to round-off: no iterations, no tolerance, identical cost on every call — ideal for
CUDA-graph capture and for autograd. The constant mode (null space of `L`) is set to zero;
the right-hand side has zero sum by construction (§2).

Implementation (`poisson.py`), all in differentiable `torch.fft`:

* A length-N DCT-II is computed with Makhoul's reordering (evens ascending, odds
  descending — one `index_select`), one **real** FFT of length N, and a quarter-wave
  twiddle `Z_k = e^{−iπk/2N} V_k`, k = 0..N/2. For real input that half spectrum holds every
  coefficient: `X_k = 2 Re Z_k`, `X_{N−k} = −2 Im Z_k`.
* The coefficients are never put back in natural order: the solver keeps the *packed*
  layout `[Re Z_0..Re Z_{N/2}, Im Z_1..Im Z_m]` and permutes the eigenvalues once at
  construction. The inverse rebuilds Z from the packed array (including the Nyquist
  identity `Im Z_{N/2} = −Re Z_{N/2}`), untwiddles, runs `irfft` and undoes the reordering
  with one gather. Cost: two half-length complex FFTs per direction and a few elementwise
  passes — about 90 µs for a 1024×512 forward 2D transform inside a CUDA graph.
* Periodic mode uses `rfft2`/`irfft2` with the periodic eigenvalues.

Verified (`test_poisson_is_exact`, `bench_ns2d.py verify`): `max|L ψ − f| / max|f| ≤ 4.1e-14`
in float64 for even and odd sizes, Neumann and periodic (tests: < 1e-10 on random right-hand
sides). Projected divergence of real solver states after full steps (cylinder; the fast tests
also use a moving ground and a no-slip top wall): `max|∇·u| h / U = 3.9e-16` (float64), `1.8e-7`
(float32, GPU) — the quality bars are 1e-10 and 1e-5.

The pressure written to the state is the **step-averaged** pressure
`p = (⅙ ψ₁ + ⅔ ψ₂ + ψ₃) / Δt`. Expanding the stages (§6.4) shows that the stage
gradients reach `u^{n+1}` with exactly these weights, so this `p` is the pressure of the
discrete momentum balance: a control-volume balance evaluated with it reproduces the
penalisation force to 0.1 % (§6.4). The last stage alone, `ψ₃ / (b₃ Δt)` (the first
implementation), is *not* consistent: with stiff penalisation every stage also removes the
velocity that the previous projection left inside the solid, so the stage potentials are not
proportional to `b_s Δt`. The difference is a harmonic, dipole-like field around each body;
it is invisible in the far field but made the control-volume drag of the Re = 40 cylinder 6 %
low and its stagnation cp 0.13 low. `p` is output only (fields, cp, surface data) and does
not feed back into the dynamics.

## 5. Time integration and time step

Three-stage strong-stability-preserving Runge–Kutta (Shu & Osher 1988) with a projection
after every stage, written in the convex-combination form

```
û_s = lerp(u_{s−1}, u_0, a_s) + b_s Δt L(u_{s−1}),   (a, b) = (0, 1), (¾, ¼), (⅓, ⅔)
û_s ← implicit penalisation (§6),   u_s = P(û_s)   (projection, §4)
```

so every stage value is discretely divergence-free. SSP-RK3 is 3rd-order in time and its
stability region contains a segment of the imaginary axis (|z| ≤ √3), which the purely
central advection operator needs (forward Euler and SSP-RK2 do not have that property).

**Time step** — `Δt = min(cfl · h / (max|u| + max|v|), 0.25 h² / (1/Re + max ν_t))` with
`cfl = 0.5` by default. The advective limit uses the sum of the component maxima, a safe
bound for the 2D CFL number; the viscous limit 0.25 is below the SSP-RK3 limit for the 2D
5-point Laplacian (≈0.31). The reduction is computed **on the device at every step** and
the step size stays a device tensor: there is no GPU→CPU synchronisation anywhere in a
step, which is what allows CUDA-graph capture. (The specification suggested refreshing
`umax` every K steps; doing it every step costs two reductions, ~1 % of a step, and is
strictly safer.) A fixed `dt` can be configured instead.

## 6. Immersed boundaries: Brinkman volume penalisation

### 6.1 Masks

Bodies enter only through the solid fraction χ, evaluated from the signed distance `d`
(negative inside) of each polygon **directly at the u-faces and v-faces** (and at centres
for display and LES switch-off):

```
χ(d) = 1                              d + δ ≤ −w
     = ½ (1 − sin(π (d + δ) / 2w))     |d + δ| < w
     = 0                              d + δ ≥ w
```

with half-width `w = mask_width · h` (default 1 cell) and an inward shift `δ = 0.32 h`
(§6.3). The profile is C¹ and has **compact support**: with stiff penalisation any non-zero
tail of χ acts as a wall (a `tanh` profile would move the effective surface outwards by
several cells). The signed distance is exact (distance to the nearest edge, sign from an
even-odd crossing test), differentiable with respect to the vertices, and is taken from
`aerolume.geometry.sdf.signed_distance` when that module is importable (private fallback
otherwise). Masks are stored only on a **window** of cells covering the bodies plus a margin,
so penalisation, force sums and SDF evaluations cost in proportion to the body, not the
domain. The live solver sizes the window for ±25° of pitch about `moment_ref`, so angle
changes from the GUI rebuild masks in place (captured CUDA graphs stay valid).

Rigid rotation (wheels, `omega` about `center`) enters as `χ u_s` with
`u_s = ω × (x − c)`; it is pre-multiplied into the window arrays so it costs nothing per step.

### 6.2 Implicit penalisation

After the explicit part of each stage, the penalisation term is integrated implicitly over
that stage's effective step `b_s Δt`:

```
u ← (u + k χ u_s) / (1 + k χ),     k = b_s Δt / η
```

which is unconditionally stable — η → 0 simply pins u to u_s wherever χ > 0 — and exactly
solves the stage's backward-Euler penalisation equation. The projection follows, so the
final field is divergence-free; the small velocity the projection re-introduces inside the
solid is removed again at the next stage.

### 6.3 Choosing η and placing the effective wall

A smoothed mask with stiff penalisation does not put the no-slip wall at `d = 0`: the wall
sits wherever the penalisation rate `χ/η` overtakes the rate at which momentum is supplied
into a cell. In a wall-bounded shear flow that supply is viscous (`ν/h²`), so the effective
wall position depends on the **stiffness** `S = h²/(ν η)` alone. We measured it with an exact
reference: body-force-driven plane Poiseuille flow between penalised slabs in a periodic
box (flow rate → effective gap width → wall offset), averaging four sub-cell positions of the
wall (`scratch` study, reproduced in this section's table):

| w (cells) | S = 12 | S = 31 | S = 60 | S = 125 | S = 250 | S = 1000 |
|-----------|--------|--------|--------|---------|---------|----------|
| 1.0       | 0.00   | −0.16  | −0.24  | −0.30   | −0.33   | −0.36    |
| 1.5       | −0.31  | −0.52  | −0.63  |         | −0.78   | −0.83    |
| 2.0       | −0.64  |        | −1.03  |         | −1.23   | −1.31    |

(offset of the effective wall in cells, unshifted mask; negative = wall protrudes into the
fluid). The offset saturates as S grows. Wider masks do not reduce the sub-cell scatter
once S is large (≈0.45 cells peak-to-peak for every width) and make the offset more
Re-dependent, so **w = 1 cell**. The permeability is then chosen to keep the stiffness on the
flat part of the curve at every Reynolds number:

```
η = min(0.05 h / U,  h² / (ν · 60))
```

The first bound keeps the inertial penetration `U η` below 5 % of a cell (high Re); the
second keeps `S ≥ 60` (low Re). Over `S ∈ [60, ∞)` the offset spans −0.24 … −0.38 cells, so
the mask is shifted inwards by **δ = 0.32 h**, which centres that band on the true surface:
measured offset +0.05 h at S = 60 and −0.11 h at S = 1000. The penetration length
`√(ν η) ≤ h/√60 ≈ 0.13 h` and both bounds vanish as h → 0, so the method converges to the
no-slip solution. η is a device tensor updated by `set_inflow(re=…)` without breaking
captured CUDA graphs. This calibration uses only the Poiseuille problem; the cylinder
benchmarks below are independent validation.

Remaining geometric uncertainty: for straight walls aligned with grid lines the effective
wall position scatters by about ±0.2 h with the sub-cell position of the wall; on curved
bodies this averages out around the perimeter.

### 6.4 Force evaluation

The force on the bodies is the momentum the penalisation removes. Writing the stage
increments of the penalisation as Δ₁, Δ₂, Δ₃ and expanding the Shu–Osher stages,

```
u^{n+1} = u^n + Δt (⅙ L₁ + ⅙ L₂ + ⅔ L₃) + ⅙ Δ₁ + ⅔ Δ₂ + Δ₃  (+ gradient terms)
```

so the force on body b over the step is

```
F_b = −(h² / Δt) Σ_faces w_b (⅙ Δ₁ + ⅔ Δ₂ + Δ₃)
```

(`w_b` = body b's share of χ at a face; 1 on its own support when bodies do not overlap).
For a steady flow the three weighted increments sum to exactly `−Δt` times the stage
residual, i.e. this is the discrete momentum balance of the whole domain: the projection
adds momentum only through the boundary pressure, so a control-volume balance evaluated
consistently with the discretisation returns the same number (plus the rate of change of
momentum inside the control volume). The moment uses the same increments with lever arms
at the face positions. The method is cheap (two small einsums on the window per step),
per-body, and differentiable.

**Cross-check by a control-volume balance.** For the steady cylinder at Re = 40 (32 cells/D)
the drag from the momentum balance

```
Cd_cv(r) = 2 · ∮_{|x| = r} [ −p n_x − u (u·n) + ν (2 ∂u/∂x n_x + (∂u/∂y + ∂v/∂x) n_y) ] ds
```

with every quantity interpolated bilinearly from its own staggered position, is 1.6205,
1.6206, 1.6205, 1.6202, 1.6198, 1.6199 on circles 1.5, 2, 3, 4, 10 and 20 cells outside the
surface, against **1.6190** from the impulse (0.1 %). The parts change a lot between those
radii (pressure 1.14 → 1.99, viscous 0.49 → 0.12, momentum flux −0.01 → −0.49); only their
sum is constant. The
fast test `test_pressure_closes_the_momentum_balance` checks the same identity (to 0.3 %) on a
coarse Re = 10 case. So the impulse is the force of the discrete solution — not an estimate
of it — and its grid convergence (§11) is the force's grid convergence. Of the candidate
evaluations named in the specification (penalisation impulse, control-volume balance,
surface integration) the impulse is the cheapest, exact per body and differentiable, so it
is the solver's force output; the other two agree with it once the output pressure is the
step-averaged one (§4) and the surface sampling is second order (§6.5). A sharper
interpolated direct-forcing IB was not needed: the calibrated penalisation meets every
cylinder target at 32 cells/D (§11).

### 6.5 Surface distributions (cp, cf)

`FlowSolver.surface(n, mean=False)` returns, per body, `n` points equally spaced in arc
length with outward normals, cp and cf (instantaneous, or from the running means). The wall
is diffuse — the penalised layer reaches `(w − δ) h = 0.68 h` into the fluid — so wall
values are **extrapolated from two probes** along the outward normal at `d₁ = 1.5 h` and
`d₂ = 2.5 h`, just outside the layer:

* **cp**: linear extrapolation of the probe pressures to `d = 0`. This removes the
  centrifugal gradient `∂p/∂n ≈ u_t²/R` across the probe distance (up to 0.07 in cp at the
  cylinder shoulders at 32 cells/D).
* **cf** `= (2/Re) ∂u_t/∂n` at the wall, from the quadratic through the wall velocity at
  `d = 0` (zero, or `(ω × r)·t` on a spinning wheel) and the tangential velocity at both
  probes, with u and v interpolated from their own faces (no extra averaging pass). This is
  second order. The first implementation's one-sided `u_t(d)/d` at a single probe
  under-predicted the cylinder's friction drag by 15–20 % and put separation 5–7° too far aft.
  Only the molecular viscosity enters (WALE and Vreman ν_t vanish at walls anyway).

Measured on the cylinder (§11 table, columns `Cd_surf`, `Cp_front`, `θ_sep`): the integral
`∮ (−cp n_x + cf t_x) ds` reproduces the impulse drag to 0.0–0.1 % at Re 40, and to within
2.1 / 1.1 / 0.7 % (32 / 48 / 64 cells/D) at Re 100, where the boundary layers are thinner
(fewer cells across them) and the deficit converges away with resolution. At Re 40 the
stagnation cp is 1.12–1.15 (viscous flows have `cp₀ > 1` at moderate Re) and separation is
at 126.0–126.5° from the front, i.e. 53.5–54.0° from the rear (Dennis & Chang 1970: 53.8°;
Coutanceau & Bouard 1977: 53.5°). The split between pressure and friction depends on the
probe placement by ±1–2 % each (probes at 1/2 h or 2/3 h instead of 1.5/2.5 h moved Cd_f by
+2.7 % / −2.0 % at Re 40, 32 cells/D); the total is more robust than either part.

**Reference pressure.** cp is relative to the mean pressure of the two inlet columns
(DESIGN.md). In a confined domain the static pressure near the body is lower than at the
inlet: in the 5 %-blockage cylinder domain of §11 the slip-wall pressure abeam the cylinder
is cp = −0.09 (Re 40) and −0.05 (Re 100) relative to the inlet, with a local wall velocity
of 1.04 and 1.02 U. Blockage and upstream length (the planner's domain choice) therefore
shift cp near the body almost uniformly; forces are unaffected.

## 7. Subgrid models (2D LES)

`turbulence = "smagorinsky" | "wale" | "vreman"` adds `∇·(2 ν_t S)`, with `ν_t` computed at
cell centres from the velocity-gradient tensor (corner derivatives averaged to the centre)
and the filter width Δ = h:

* Smagorinsky (Lilly): `ν_t = (C_s Δ)² √(2 S:S)`, `C_s = 0.17`.
* WALE (Nicoud & Ducros 1999): `ν_t = (C_w Δ)² (S^d:S^d)^{3/2} / ((S:S)^{5/2} + (S^d:S^d)^{5/4})`,
  `C_w = 0.5`.
* Vreman (2004): `ν_t = c √(B_β / α:α)`, `c = 0.07` (≈ 2.5 C_s²); in 2D `B_β = det(β)`.

The 2D field is treated as a 3D field without a third component, so the textbook formulas
apply unchanged. WALE and Vreman vanish in pure shear (no eddy viscosity in laminar
boundary layers); Smagorinsky does not (tests). ν_t is multiplied by `1 − χ_c`, i.e. it is
zero inside solids, and its maximum enters the viscous time-step limit.

**Caveat**: 2D turbulence is not 3D turbulence (inverse energy cascade, no vortex
stretching). In Aerolume these models are a *stabilising subgrid dissipation* for
under-resolved high-Re runs, not a turbulence model in the physical sense; results at
Re ≫ 10⁴ are qualitative (see §12).

## 8. Passive dye

A cell-centred smoke scalar is injected at the inlet as a comb of thin streaks (about 40
over the domain height) and advected once per step (forward Euler, final velocity) with a
MUSCL flux limited by van Leer's harmonic mean slope, then clamped to [0, 1].
Concentrations below 10⁻⁶ are set to zero: the limiter otherwise lets far tails decay into
subnormal floats, which are ~100× slower on x86 CPUs (measured: the CPU solver fell from
60 to under 10 steps/s as the dye tails spread). The dye is purely visual and does not
affect the flow. It is set to zero inside solids (cells with χ > ½) after each update, so
smoke never appears to pass through a body; placed emitters, further channels and tracer
particles are in §17.

## 9. Start-up perturbation

`perturb = True` adds a weak Lamb–Oseen vortex (peak swirl 5 % of U, core radius 0.75 ×
the body height) centred on the bodies to the impulsive-start initial field, then projects.
The bodies start with a small circulation, the near wake is asymmetric from the first step,
and periodic shedding locks in after a few tens of convective times instead of waiting for
round-off (which takes 100+ time units and differs between CPU and GPU). It depends only on
the body bounding box, so runs are deterministic and reproducible.

## 10. Performance: CUDA graphs, CPU path

A step launches a few hundred small kernels (three stages of padding, fluxes, Poisson
transforms, penalisation, projection, plus dye and bookkeeping). Under Windows WDDM a
kernel launch costs ~27 µs, so eager execution is launch-bound (59 steps/s at 1024×512).
`FlowSolver` keeps the state in static buffers and captures blocks of 10 steps (plus a
1-step graph for remainders) in `torch.cuda.CUDAGraph`s after a warm-up on a side stream
(the warm-up steps are undone, so capturing never changes the trajectory). Nothing in a
step synchronises with the host: dt is a device tensor, the history row is written into a
device ring buffer with `index_copy_`, running means are accumulated in float64 device
buffers. Live edits write into existing tensors (`nu`, `η`, masks), so graphs survive
them; a different body count or a window that must grow triggers a lazy re-capture; any
capture/replay error falls back to eager execution with a warning. Graph and eager paths
run the same kernels and agree to float32 round-off (test: 1e-6 after 34 steps).

Measured throughput (float32, `FlowSolver.advance`, cylinder case, best of 3):

| device | grid | options | steps/s | target |
|--------|------|---------|--------:|-------:|
| CUDA, RTX 4060 Ti | 1024×512 | default (upwind3, laminar, CUDA graphs) | **493** | ≥ 300 |
| CUDA | 1024×512 | eager (no graphs) | 59 | |
| CUDA | 1024×512 | central advection | 619 | |
| CUDA | 1024×512 | upwind3 + WALE | 317 | |
| CUDA | 1024×512 | upwind3 + Smagorinsky | 373 | |
| CUDA | 2048×1024 | default | 75 | |
| CPU, i7-13700F | 512×256 | 1 thread | 48 | |
| CPU | 512×256 | 4 threads | **61** | ≥ 15 |
| CPU | 512×256 | 16 threads (torch default here) | 36 | |

(`bench_ns2d.py perf`, cylinder in a 24 × 12 domain, dye on, 400 GPU / 60 CPU steps per
repeat after a warm-up that includes the graph capture; repeated runs on the shared machine
scatter by ±5 %.) Graphs give an 8× speed-up at 1024×512. At 2048×1024 (4× the cells) a
step takes 6.6× longer: the large grid is memory-bound, and the extra factor is probably cache
behaviour (not investigated).

CPU notes. (1) `FlowSolver.advance` enables flush-to-zero on CPU
(`torch.set_flush_denormal(True)`), see §8. (2) On a busy hybrid-core CPU, many intra-op
threads can be *slower* than a few: each elementwise op on a 512×256 field takes ~10 µs,
comparable to OpenMP fork/join and to the delay of a descheduled worker thread. 4 threads
was the sweet spot on the development machine; the application may want to cap
`torch.set_num_threads` for the solver process.

## 11. Verification and validation

### Verification (exact answers)

| check | result | bar |
|-------|--------|-----|
| Poisson `‖Lψ − f‖/‖f‖`, float64 | 4.1e-14 | round-off |
| projected divergence `max|∇·u| h/U`, float64 | 3.9e-16 | < 1e-10 |
| projected divergence, float32 (GPU) | 1.8e-7 | < 1e-5 |
| Taylor–Green, central, observed order | 1.97, 1.99, 2.00 | ≈ 2 |
| Taylor–Green energy decay rate / exact (N = 64) | 0.9992 | 1 |
| inviscid energy drift, central, 50 steps | < 1e-4 | conservative |
| outlet mass balance after 30 steps | < 1e-10 | exact |
| gradient d(mean Cd over 20 steps)/d(shape), autograd vs central FD (float64) | 5.030 vs 5.030, rel. err. 2.3e-7 | < 5 % |
| control-volume drag (circles r = 0.75…2.5) / impulse drag, steady Re 10 | within 0.3 % (test); Re 40: 0.1 % (§6.4) | consistent |
| CUDA graph vs eager (34 steps, float32) | agree to 1e-6 | same kernels |

### Validation: circular cylinder

Wide domain `[−10, 20] × [−10, 10]` D (blockage 5 %, slip top and bottom), uniform inflow,
upwind3, float32 on the GPU, perturbation on. Forces averaged over t ∈ [50, 60] (Re = 40) and
[100, 200] (Re = 100). `L/D` is the recirculation length behind the cylinder (from the
rear stagnation point). Reference values (unconfined): Re = 40: Cd 1.50–1.55, L/D 2.13–2.35;
Re = 100: Cd 1.33–1.38, Cl′ ±0.32–0.34, St 0.164–0.167 (Dennis & Chang 1970, Fornberg 1980,
Coutanceau & Bouard 1977, Williamson 1989, Liu et al. 1998, Linnick & Fasel 2005,
Taira & Colonius 2007). The 5 % blockage raises Cd and St by a few per cent relative to
unconfined values; the acceptance bands of the specification allow for that.

Columns: `Cd` mean drag (penalisation impulse), `Cl′` half the peak-to-peak lift, `St` from
lift zero crossings, `L/D` recirculation length of the (mean) flow (in brackets for Re 100,
where it is the mean-flow bubble), `Cd_surf` the drag integrated from `surface(720)` data
(mean fields) and its deviation from `Cd`, `Cp_front` / `Cp_rear` the surface cp at the front
and rear stagnation points (relative to the inlet pressure), `θ_sep` the separation angle
measured from the front (first cf sign change on the upper half). Wall time is the GPU time of
`FlowSolver.advance` for the whole run (RTX 4060 Ti, float32, CUDA graphs, shared machine).

| case | cells/D | grid | steps | wall [s] | Cd | Cl′ | St | L/D | Cd_surf (vs Cd) | Cp_front | Cp_rear | θ_sep |
|------|--------:|------|------:|---------:|---:|----:|---:|----:|----------------:|---------:|--------:|------:|
| Re40 | 16 | 480×320 | 3600 | 3 | 1.6296 | 0.010 | — | 2.320 | 1.5619 (-4.2 %) | 1.092 | −0.615 | 128.0° |
| Re40 | 24 | 720×480 | 5200 | 7 | 1.6216 | 0.009 | — | 2.286 | 1.5925 (-1.8 %) | 1.109 | −0.606 | 127.5° |
| Re40 | 32 | 960×640 | 7000 | 16 | 1.6190 | 0.009 | — | 2.280 | 1.6190 (+0.0 %) | 1.122 | −0.603 | 126.5° |
| Re40 | 48 | 1440×960 | 14000 | 90 | 1.6155 | 0.009 | — | 2.263 | 1.6165 (+0.1 %) | 1.139 | −0.598 | 126.0° |
| Re40 | 64 | 1920×1280 | 24600 | 399 | 1.6141 | 0.009 | — | 2.256 | 1.6142 (+0.0 %) | 1.148 | −0.596 | 126.0° |
| Re100 | 16 | 480×320 | 13000 | 30 | 1.4798 | 0.377 | 0.1670 | (1.39) | 1.3533 (-8.5 %) | 1.053 | −0.836 | 119.5° |
| Re100 | 24 | 720×480 | 19400 | 77 | 1.4291 | 0.355 | 0.1689 | (1.39) | 1.3577 (-5.0 %) | 1.052 | −0.810 | 119.0° |
| Re100 | 32 | 960×640 | 25600 | 59 | 1.4123 | 0.350 | 0.1697 | (1.39) | 1.3829 (-2.1 %) | 1.052 | −0.802 | 117.5° |
| Re100 | 48 | 1440×960 | 38600 | 262 | 1.3993 | 0.344 | 0.1698 | (1.40) | 1.3841 (-1.1 %) | 1.058 | −0.792 | 117.0° |
| Re100 | 64 | 1920×1280 | 51400 | 876 | 1.3947 | 0.342 | 0.1698 | (1.40) | 1.3852 (-0.7 %) | 1.063 | −0.788 | 117.0° |

Reading the table:

* **Grid convergence.** Re 40: Cd 1.6190 → 1.6155 → 1.6141 and L/D 2.280 → 2.263 → 2.256 from
  32 to 64 cells/D — monotone, changes of 0.2 % and 0.1 % in Cd. Re 100: Cd 1.4123 → 1.3993 →
  1.3947, Cl′ 0.350 → 0.344 → 0.342, St 0.1697 → 0.1698 → 0.1698 (changes of 0.9 % and 0.3 %
  in Cd). 32 cells/D is converged to well within the scatter of the reference data, which is
  why the slow validation tests use it. Below that the drag and lift amplitude rise: at 16
  cells/D Re 100 gives Cd +6 % and Cl′ +10 % relative to 64 cells/D.
* **Accuracy.** At 32, 48 and 64 cells/D every value is inside the specification band (Re 40:
  Cd 1.50–1.62, L/D 2.1–2.3; Re 100: Cd 1.30–1.42, Cl′ 0.30–0.36, St 0.160–0.172). At 16 and
  24 cells/D drag (and at 16 also Cl′ and the Re 40 L/D) are slightly above it. Against
  unconfined references the drag is ~5 % high (Re 40: 1.61 vs 1.50–1.54; Re 100: 1.40 vs
  1.33–1.35) and St ~2 % high. That is the expected blockage effect: the 2D wake-blockage
  estimate ε ≈ (Cd/4)(D/H) ≈ 0.02 (Allen & Vincenti 1944) raises the effective dynamic
  pressure by ≈ 4 %, and the measured slip-wall velocity abeam the cylinder is 1.04 U
  (Re 40) / 1.02 U (Re 100).
* **Surface data.** `Cd_surf` matches the impulse to 0.1 % at Re 40 and is within
  2.1 / 1.1 / 0.7 % (32 / 48 / 64 cells/D) at Re 100, where the boundary layers are thinner.
  The front stagnation cp
  is 1.12–1.15 at Re 40 (> 1 by viscous stagnation-flow effects) and 1.05–1.06 at Re 100, and
  separation sits at 126.0–126.5° from the front (53.5–54.0° from the rear; Dennis & Chang
  1970: 53.8°; Coutanceau & Bouard 1977: 53.5°). Base cp is −0.60 (Re 40) and −0.79 to
  −0.80 (Re 100) relative to the inlet; relative to the local free-stream pressure abeam
  the cylinder (§6.5) that is −0.51 and −0.74 to −0.75.

### Channel and sharp-edged benchmarks

The Schäfer–Turek channel benchmarks 2D-1 (Re 20) and 2D-2 (Re 100) — no-slip walls, parabolic
inflow, an off-centre cylinder — and the square cylinder at Re 100 are run by
`bench_ns2d.py channel` and `bench_ns2d.py square`; their results at two resolutions are in
[the validation report](validation.md), section 3.

### Airfoil sanity check (not a quantitative validation)

NACA 0012, Re = 1000, 80/160 cells per chord: Cl(5°) = 0.20/0.23, Cd ≈ 0.13; Cl(10°) = 0.36
(80 cells), in line with published low-Re DNS. NACA 4412 at 4°: Cl 0.21, Cm(c/4) −0.033 vs
0012: Cl 0.18, Cm +0.006 — camber adds a nose-down moment, confirming the sign convention.

### Moving ground

`moving_ground_config(256)` (`validation/ns2d_cases.py`): a rounded box 1.0 × 0.3 (corner
radius 0.08) 0.08 above a road moving at U, two wheels of radius 0.07 that touch the road,
overlap the body and roll with the road (`omega = +U/r`, counter-clockwise), Re = 10⁶,
upwind3 + WALE, 1024 × 307 cells (256 per length), float32 on the GPU, t = 0 … 6.

Result (`bench_ns2d.py ground`, slow test `test_moving_ground_les_re1e6_stable`): the run
completes without any instability. The step stays at the advective limit (Δt ≈ 3 × 10⁻⁴,
19 000 steps, 43 s on an idle GPU), divergence stays at float32 round-off, ν_t peaks at
3–4 × 10⁻⁴ (≈ 0.1 U h, i.e. the model is active only at the grid scale) and the instantaneous
peak speed fluctuates between 3 and 4.7 U. Those peaks sit in intense 2D vortices shed from
the roof leading edge and in the wake: 2D flow at this Re has no vortex stretching, so shear
layers roll up into compact, long-lived vortices (without wheels the peaks are ~3 U). Mean
Cd (t > 3, reference height 0.3) is 2.3–2.9 between runs with different wheel spin; mean Cl
is large and positive (5–7). The lift is *not* meaningful for this geometry: in 2D the
touching wheels seal the underbody between them into a closed pocket whose pressure is set
only by seepage through the diffuse walls. That is a property of 2D side-profile models, not
of the solver; the planner should keep a gap of a few cells between wheels and road or body
(or omit wheels) when it wants underbody flow. The early version of this case had the wheels
spinning against the road (`omega = −U/r`); the sign convention is now documented on
`ImmersedBody`.

## 12. Limitations — what to trust

Summary of the validity range (details below):

| regime | what to trust | evidence |
|--------|---------------|----------|
| Re ≤ ~200, ≥ 32 cells across the body | forces, Strouhal number, wake length, cp / cf distributions: quantitative (a few per cent, plus domain blockage) | cylinder Re 40 / 100, §11 |
| Re ≤ ~200, 16–24 cells across | Cd within 1–6 %, Cl′ within 4–10 %, St within 2 % of the converged values; surface-integrated drag 2–9 % low (cf under-resolved) | cylinder rows 16 and 24 cells/D, §11 |
| 200 < Re ≲ 10⁴ | 2D laminar physics (real flows are 3D above Re ≈ 190 for bluff bodies); needs cells/ref ∝ √Re for the boundary layers | airfoil check at Re 1000 |
| Re ≳ 10⁴ (vehicles, airfoils at flight Re) | qualitative: flow patterns, separation regions, *differences* between designs; not absolute drag. cf magnitudes meaningless | moving-ground run at Re 10⁶ is stable |

* **Resolution.** Forces of bluff bodies are within a few per cent of reference data from
  ~24–32 cells across the body (cylinder: 32 cells/D meets every target). Thin features
  (airfoil trailing edges, splitters) need at least ~3 cells across their thickness to be
  solid; airfoils need ≥ 128 cells per chord for useful drag, more as Re rises.
* **Wall position.** The diffuse wall is placed to ±0.1 h on average and scatters by
  ±0.2 h on grid-aligned straight walls (§6.3). At 32 cells/D that is < 1 % of the diameter.
* **Surface cp and cf.** `surface()` extrapolates from probes 1.5 and 2.5 cells off the wall
  (§6.5). Where the boundary layer spans ≳ 4 cells (cylinder Re ≤ 100 at ≥ 32 cells/D) the
  distributions integrate to the solver force within 0–3 %, the stagnation cp is above 1 as
  it must be at moderate Re, and the separation angle matches published data. Where it is thinner than ~3 cells (high Re, coarse grids) cf
  under-resolves the wall gradient — read its sign (separation, reattachment) and trends, not
  its magnitude; cp stays usable. cp is relative to the inlet pressure, which in a confined
  domain sits a few hundredths above the local free-stream pressure (§6.5). Probes that enter
  another body or leave the domain (wheel–road contact, gaps under ~3 cells) are undefined.
  Reported forces always come from the penalisation impulse.
* **Reynolds number.** Laminar 2D results are physical up to Re ≈ 10³–10⁴ (bluff bodies
  become 3D at Re ≈ 190 — 2D results above that are a 2D idealisation). Above ~10⁴ the
  boundary layers are not resolved at affordable grids; with upwind3 + LES runs stay
  stable (moving-ground test at Re = 10⁶) and give qualitative flow patterns and
  comparative trends between designs, not absolute drag. There is no transition model; a
  Werner–Wengle wall model for the domain walls exists but is opt-in and off in every
  planned run (§18.4).
* **2D.** No spanwise effects, tip vortices or 3D turbulence; LES in 2D is a subgrid
  dissipation, not a turbulence model.
* **Uniform grid.** No stretching or refinement; domain size versus resolution is the
  planner's trade-off. Blockage and outlet proximity affect results at the per-cent level.
* **Compressibility.** None (Mach → 0).
* **Placed physics** (heat transfer, fans, ports, emitters): §22.

## 13. Differentiability

`step(state, geom, ops, dt)` is pure torch with static control flow. Everything between the
polygon vertices and the force coefficients is differentiable: SDF (exact distance to edges)
→ C¹ mask → penalisation → projection (linear, FFT-based) → force impulse. The upwind
dissipation `|V|` and the van Leer limiter are the only non-smooth elements (measure-zero
kinks). `rollout(state, geom, ops, n, dt, checkpoint_every=k)` advances n steps with
`torch.utils.checkpoint` over chunks of k steps (memory O(n/k + k) fields), with a fixed dt
(letting the CFL rule choose dt inside a differentiated rollout would make dt a non-smooth
function of the design). `step` returns the coefficients `[Cd, Cl, Cm]` summed over the bodies
(`step_full`: per body); multiply by `½ force_ref` for the non-dimensional force. Gradients
with respect to body vertices (checked against central finite differences in float64:
relative error 2.3e-7), the inflow profile (`ops.u_inlet`) and the initial state are tested.
`build_geometry(..., window=…)` keeps the window fixed so finite-difference comparisons
evaluate the same discrete function.

## 14. References

* Allen & Vincenti (1944), Wall interference in a two-dimensional-flow wind tunnel, with
  consideration of the effect of compressibility, NACA Report 782.
* Angot, Bruneau & Fabrie (1999), A penalization method to take into account obstacles in
  incompressible viscous flows, Numer. Math. 81.
* Coutanceau & Bouard (1977), Experimental determination of the main features of the viscous
  flow in the wake of a circular cylinder in uniform translation, JFM 79.
* Dennis & Chang (1970), Numerical solutions for steady flow past a circular cylinder at
  Reynolds numbers up to 100, JFM 42.
* Engels, Kolomenskiy, Schneider & Sesterhenn (2016), FluSI: a novel parallel simulation tool
  for flapping insect flight using a Fourier method with volume penalization, SIAM J. Sci. Comput.
* Fornberg (1980), A numerical study of steady viscous flow past a circular cylinder, JFM 98.
* Harlow & Welch (1965), Numerical calculation of time-dependent viscous incompressible flow
  of fluid with free surface, Phys. Fluids 8.
* Kolomenskiy & Schneider (2009), A Fourier spectral method for the Navier–Stokes equations
  with volume penalization for moving solid obstacles, J. Comput. Phys. 228.
* Linnick & Fasel (2005), A high-order immersed interface method for simulating unsteady
  incompressible flows on irregular domains, J. Comput. Phys. 204.
* Liu, Zheng & Sung (1998), Preconditioned multigrid methods for unsteady incompressible
  flows, J. Comput. Phys. 139.
* Makhoul (1980), A fast cosine transform in one and two dimensions, IEEE Trans. ASSP 28.
* Morinishi, Lund, Vasilyev & Moin (1998), Fully conservative higher order finite difference
  schemes for incompressible flow, J. Comput. Phys. 143.
* Nicoud & Ducros (1999), Subgrid-scale stress modelling based on the square of the velocity
  gradient tensor, Flow Turb. Combust. 62.
* Ohtake, Nakae & Motohashi (2007), Nonlinearity of the aerodynamic characteristics of NACA0012
  aerofoil at low Reynolds numbers, J. Japan Soc. Aero. Space Sci. 55.
* Shu & Osher (1988), Efficient implementation of essentially non-oscillatory shock-capturing
  schemes, J. Comput. Phys. 77.
* Taira & Colonius (2007), The immersed boundary method: a projection approach, J. Comput.
  Phys. 225.
* Vreman (2004), An eddy-viscosity subgrid-scale model for turbulent shear flow, Phys. Fluids 16.
* Williamson (1989), Oblique and parallel modes of vortex shedding in the wake of a circular
  cylinder at low Reynolds numbers, JFM 206.
* Kochkov, Smith, Alieva, Wang, Brenner & Hoyer (2021), Machine learning–accelerated
  computational fluid dynamics, PNAS 118 (JAX-CFD, the differentiable-solver lineage AeroJAX
  builds on).

---

# Part II — placed physics

## 15. Placed physics: design rules

DESIGN.md §10 adds AeroJAX-style placed physics to the validated flow solver: smoke
emitters and tracer particles, heat transfer with Boussinesq buoyancy, heated bodies and
walls, volumetric heaters, fans (actuator strips), jets and vents, mouse-painted
obstacles, live editing, diagnostics and a PID-controlled time step. Module
`physics.py` (with `scalar.py`, `particles.py`, `diagnostics.py`); solver-unit specs in
`config.py` (`EmitterSpec`, `ThermalConfig` + `WallBC`, `HeatSourceSpec`, `FanSpec`,
`PortSpec`, `ImmersedBody.thermal`). Four rules keep the validated solver intact:

* **Off means absent.** `ops.phys` is `None` unless a config asks for placed physics,
  and every branch in the step tests a static Python flag, so the default path runs the
  kernels it ran before. Measured with `torch.profiler` (cylinder, 1024×512, eager, median
  of three interleaved rounds): 524 kernels per step against 521 without the one addition
  to the default path — the default smoke is now cleared inside solids (§8; a few small
  window kernels) — and 3048 µs of GPU kernel time per step against 3034 µs (+0.5 %; an
  earlier session measured −0.3 %, i.e. the difference is within the noise of a GPU shared
  with other jobs). Re-measured with the final code, after the wall model of §18.4 joined
  the step behind its own static flag: still 524 kernels per step, and the clean-up's own
  kernels take 0.16–0.18 % of the step's GPU kernel time (medians of two runs of five
  interleaved rounds, all rounds within 0.14–0.23 %; `python validation/bench_ns2d.py
  default-cost`). That ratio is the robust figure: two whole-step kernel times measured
  separately differed by up to ±15 % between rounds on the shared GPU. Every test and
  validation number of the base solver is unchanged (the slow base-solver and placed-physics validations were re-run
  and pass).
* **Static structure, live values.** Which features exist (heat transfer on, buoyancy on,
  the number of fans, smoke channels and emitter timing groups, whether ports or paint
  exist) is static and baked into a captured CUDA graph. Every *value* — footprints,
  positions, thrusts, wall temperatures, Ri, Pe — lives in a device tensor that
  `FlowSolver` rewrites in place, so dragging a heater or an emitter never re-captures a
  graph. A change of structure (a first fan, another smoke channel) rebuilds what it must
  and re-captures lazily on the next `advance`.
* **Exact totals.** Every distributed source (heater power, fan thrust, port volume flow,
  a heat-flux body's heat) is normalised by the discrete integral of its footprint, so the
  released total is exactly the requested one on any grid and at any sub-cell position.
  Footprints use the bodies' C¹ step (`smooth_step`, half-width one cell for heaters and
  fans, half a cell for nozzles and ports), so they move continuously when an object moves
  by a fraction of a cell.
* **No host synchronisation** inside a step: controllers, pulse clocks, particle release
  counters and diagnostics are device arithmetic, so all of it runs inside the CUDA graphs
  (graph and eager agree to 1e-5 with every feature on:
  `test_cuda_graphs_match_eager_with_all_physics`).

## 16. Heat transfer

### 16.1 Energy equation and its discretisation

With `θ = (T − T_amb)/ΔT_ref` the energy equation in solver units is

```
∂θ/∂t + ∇·(u θ) = ∇·((1/Pe + ν_t/Pr_t) ∇θ) + q,      Pe = Re Pr,  Pr_t = 0.85
```

θ lives at cell centres, so the face velocities of the MAC grid are exactly the
advecting velocities of a finite-volume flux. Advection is the MUSCL / van Leer flux of
the smoke (§8) with boundary-aware ghosts (`scalar.tvd_fluxes`); diffusion is in flux
form with the domain-wall conditions applied on the boundary faces themselves
(`scalar.diffusive_fluxes`: Dirichlet over the half cell between wall and first centre,
prescribed flux, or zero), so the heat rates the solver reports are the fluxes the scheme
used. θ is integrated with the **same SSP-RK3 stages as the momentum** (each stage:
explicit update, implicit penalisation toward hot bodies and jets, §16.2), and the
Boussinesq force uses the stage temperature, so momentum and energy are coupled at third
order in time.

**Time step: a discrete maximum principle.** A forward-Euler stage of the limited flux is a
convex combination of neighbouring values when its coefficients sum to at most one: the
MUSCL / van Leer flux contributes at most `2 c` (face Courant numbers summed over both
directions; the limited slope ratio is ≤ 2) and the explicit diffusion `c_d κ Δt / h²`,
where `c_d` is the largest diagonal coefficient of the discrete Laplacian with its wall
conditions — 4 in the interior, 5 next to a Dirichlet wall (conductance 2 over the half
cell), 6 in a corner between two Dirichlet walls (`physics.diffusion_coefficient`). The
temperature step is therefore

```
Δt ≤ 1 / ( 2 (max|u| + max|v|) / h  +  c_d (κ + max ν_t / Pr_t) / h² )
```

and SSP-RK3, a convex combination of such stages, keeps θ within the bounds of its
initial, boundary and body values. The momentum limits alone are not enough: each of them
(advective `0.5 h / (|u| + |v|)` and viscous `0.25 h² / κ`) can be satisfied while their
sum exceeds one. The first implementation did exactly that, and the heated cylinder at
Re 40, 32 cells/D blew up (θ reached 4 × 10³² after 410 steps). With the bound, θ stays in
[0, 1] (`test_temperature_stays_bounded_in_a_tunnel`); the cost is a smaller step where
advection and diffusion are both significant (heated cylinder Re 40, 32 cells/D:
3.2 × 10⁻³ against the 6.9 × 10⁻³ the separate limits allowed; the Ra 10⁶ cavity on 128²
cells needs 1.9 times as many steps per unit time, §21.1). A buoyant fluid parcel at rest
accelerates at `Ri |θ|`, so the step is also kept below `cfl √(h / (Ri max|θ|))`.

### 16.2 Temperature bodies: implicit penalisation and the thermal wall

A body held at θ_s ("temperature") is penalised exactly like the velocity:

```
θ ← (θ + k χ θ_s) / (1 + k χ),     k = b_s Δt / η_θ
```

on the same window and masks as the momentum (cell-centred χ, the calibrated inward
shift δ = 0.32 h of §6.3). The penalised conduction problem is the penalised Stokes
problem of §6.3 with Pe in place of Re, so the thermal permeability uses the momentum rule
with Pe: `η_θ = min(0.05 h, h² Pe / 60)`. That this places the *thermal* wall where the
momentum wall is was measured, not assumed: steady conduction between two penalised slabs
(gap 1 = 16 cells, `thermal2d_cases.run_conduction_slabs`, four sub-cell positions of the
walls) gives the effective-wall offset (cells; positive = inside the solid; float64)

| stiffness S = h²/(κ η_θ) | offsets at sub-cell shift 0, ¼, ½, ¾ | mean | momentum (§6.3) |
|--------------------------|--------------------------------------|-----:|----------------:|
| 60 (the default floor)   | −0.038, +0.084, +0.060, +0.084       | +0.048 | +0.05 |
| 250                      | −0.332, +0.022, +0.015, +0.022       | −0.068 | |
| 1000                     | −0.452, +0.006, +0.004, +0.006       | −0.109 | −0.11 |

— the same band as the momentum calibration. The heat a body puts into the fluid is the
θ the penalisation adds, combined over the stages with the weights (⅙, ⅔, 1) of §6.4 and
split between temperature bodies by their share of χ: like the force, it is the heat
rate of the discrete solution, and in a steady state it equals the heat leaving through
the walls to 10⁻⁶ (`test_hot_body_heat_rate_balances_the_walls`).

### 16.3 Heat-flux and adiabatic bodies

Every body that is not held at a temperature is **insulated**: the diffusivity on each face
is multiplied by `1 − χ` of those bodies (the Neumann penalisation of Kadoch et al. 2012),
so no heat crosses an adiabatic body or a painted obstacle. A "heat_flux" body releases
`G = q″L/(k ΔT_ref)` through its surface as a volumetric source in its skin, with the weight
`−dχ/dd` (whose integral across the interface is exactly one) normalised per body so that
the released heat is exactly `G P / Pe` (P the outline length inside the domain). Measured:
in an adiabatic box the heat content grows at exactly that rate (1e-9,
`test_heat_flux_body_releases_its_heat`). Its wall temperature is the skin-weighted mean of
θ; against the exact conduction solution of a flux slab it reads high by 0.26 G h
(16 and 32 cells per gap; 0.42 G h when the wall sits half a cell off a face) — a
first-order bias of a quarter cell, 0.8 % of G D at 32 cells per D.

### 16.4 Domain walls, heat rates and Nusselt numbers

`WallBC` sets each side to adiabatic, temperature or heat flux; in a tunnel the inlet is
held at `theta_inlet` and the outlet has zero diffusive flux. The wall heat rates are the
boundary-face fluxes of the scheme (`2 κ (θ_w − θ_1)` per cell for a held side, `κ G` per
unit length for a flux side), so a steady state balances to round-off: conduction between
two held walls reproduces the exact linear profile and `Nu = 1` to 1e-10
(`test_conduction_between_walls_is_exactly_linear`), a flux wall the exact slope
(`test_heat_flux_wall_profile_and_balance`). This conservative wall flux is also what makes
the cavity Nusselt numbers converge at second order (§21.1).

`FlowSolver.heat_transfer()` reports per body the heat rate (units ρ c_p U ΔT_ref L per
span), the wall temperature and the mean Nusselt number based on the reference length,
`Nu = Pe Q / ((θ_w − θ_ref) P)` (DESIGN.md §10.2: `Nu = Q L / (k ΔT A)`); `surface()` adds
the local Nusselt number from the same two-probe quadratic as cf (§6.5).

### 16.5 Buoyancy and scaling

The Boussinesq force `f = −Ri θ ĝ` (`Ri = g β ΔT_ref L / U²`, ĝ = −y by default, any
direction allowed) is evaluated at the faces from the face average of θ; boundary faces
carry no force. A temperature that depends only on the height is balanced exactly by the
pressure: the force is then a discrete gradient that the projection removes, and a stably
stratified box stays at rest to 1e-12 (`test_stable_stratification_stays_at_rest`).

**Buoyancy and bodies (changed in review round 5 before the release, as the 3D solver does, theory_ns3d.md
§15.3).** The impulse `b Δt Ri θ` is added to each stage value *after* the body and port
penalisation and *everywhere*, also inside solids (`core._add_buoyancy`). Until round 5 it
was part of the explicit tendency, before the penalisation. The penalisation then removed
the buoyancy of the fictitious fluid inside each body and booked it as a force on the body,
and the forcing was no longer a gradient across the interface. Measured on a cylinder
(D = 1, 32 cells/D, Ri 1, Re 100, closed 8D box):

| case | true force | before | after |
|---|---|---|---|
| θ = 1 everywhere (hydrostatic rest), \|u\|max after 20 steps | 0 | 1.0 × 10⁻² | 4.4 × 10⁻⁷ (float32 round-off) |
| same, Cl | −Ri A / 0.5 = −1.5708 | +0.0146 | −1.5718 |
| hot body (θ_s = 1) in still cold fluid, Cl after 20 steps | ≈ 0 at first | +1.766 | +0.195 (a real plume forming, §16.5 note) |

The force on a body is the penalisation impulse minus `Ri ∫_body θ dA`: each stage books the
share `w b Δt Ri vol θ_f` into the impulse accumulators, with `vol` the smoothed indicator
*without* the calibrated wall offset (`ThermalGeometry.vol_u` / `vol_v`), whose integral is
the true area to second order: +1.2 %, +0.27 % and +0.08 % at 8, 16 and 32 cells/D, where the
momentum mask (0.32 h inside the surface) under-counts by 14 %, 7.5 % and 3.9 %. Painted
obstacles get the same share. A hot body in still cold fluid shows a short start-up transient:
the penalisation removes the interface fluid's buoyancy over a few stages while the share is
booked at once, so Cl reads −0.99, −0.66, −0.26, −0.08 over the first four steps and +0.005 at
step 5 (default step; at a quarter of the step the transient takes more steps but a shorter
time, 0.03 instead of 0.07 L/U). The slow rise after it, to +0.14 by step 12 (t = 0.20), is the
rising warm boundary layer pulling on the cylinder. Tests: a uniformly warm fluid around
a temperature or adiabatic body stays at rest to 10⁻¹⁰ (float64) with Cl = −Ri A_vol / 0.5 to
10⁻⁶; a linear stratification around an insulated body stays at rest; the hot body's Cl is
within ±0.3 from step 5 (`tests/test_physics_round5.py`). The body-free validation rows
(differentially heated cavity, Rayleigh–Bénard) see the same arithmetic as before: buoyancy
was a uniform gradient-free addition there either way, and the round-5 validation run
(docs/validation.md §9) reproduces them. Natural convection in a closed domain uses
the free-fall velocity `U = √(g β ΔT L)` as the reference (Ri = 1, `Re = √(Ra/Pr)`,
`Pe = √(Ra Pr)`); the planner states Ra, Pr, Gr, Ri and Pe. Velocities convert to the
benchmark's `κ/L` by multiplying by Pe, heat rates per unit wall length to Nusselt numbers
by Pe.

### 16.6 Heaters and the energy balance

A heater is a volumetric source of exact total power over a smooth rect or disc footprint
(in an adiabatic box the heat content grows at exactly that power: 1e-10,
`test_heater_energy_budget_is_exact`). `FlowSolver.heat_balance()` reports the terms of the
energy budget with the keys `aerolume.analysis.physics` reads — heat in from bodies, walls,
heaters, jets (the enthalpy their penalisation adds) and the inlet (advection +
conduction), heat out through the outlet (advection) and vents, the storage rate, and the
residual in − out − storage. The terms are the scheme's own fluxes at the end of the step:
in a steady run with every heat path at once (a hot and a heat-flux body, a heater, a hot
jet, a vent, a held and a cooled wall, inflow, outflow and buoyancy) the residual is
1.8 × 10⁻⁶ of the heat input (64 × 32 cells, re-measured with the final code;
`test_energy_balance_closes_in_a_tunnel` checks 10⁻⁵ on a coarser grid); during transients
it is O(Δt) because the Runge–Kutta stages see intermediate fields.

## 17. Emitters: smoke channels and tracer particles

**Smoke.** Up to four channels (`dye`, `dye1..dye3`) share the bounded advection of §8 (one
batched flux evaluation for all channels). Emitters — point, line (capsule), rake (nozzles),
rect or disc footprints with an anti-aliased edge — relax their footprint toward their
concentration `rate` while on; emitters of one channel with equal timing share one
weight / target map, so many nozzles cost one fused update. The release clock
(`start`, `stop`, `pulse_period`, `pulse_duty`) is device arithmetic on the simulated time,
so pulsed puffs stay inside the CUDA graph. Smoke is zero inside solids (χ > ½) and painted
obstacles, vents remove it with their fluid, and jets can release it on a channel. With
emitters placed, the inlet comb on channel 0 is optional (`dye_inlet`).

**Tracer particles** (`particles.py`). Each nozzle owns a fixed ring of slots in one pool
(`particle_capacity`, default 32 768), so a release recycles the nozzle's oldest particle,
memory is fixed and every update — advection, kill, release — has static shapes. Particles
advance by the midpoint rule (RK2) with the end-of-step velocity sampled bilinearly from the
staggered faces where u and v live; they die when they leave the domain or enter a solid. A
continuous emitter releases one particle per nozzle each time its accumulated `rate · Δt /
nozzles` passes an integer — up to four per nozzle and step, each placed along the local
velocity by the time since its threshold crossing — so the released number is `rate · t`
to within one particle and a fast streakline stays evenly spaced. A pulsed emitter releases
one batch per period from all its nozzles: a timeline. In uniform flow the paths are exact
(`x = x_e + age` to 1e-9, `test_particles_follow_uniform_flow_exactly`).
`particle_lines()` returns streaklines (per nozzle, ordered by age) and timelines (per
batch, ordered along the nozzles) as polylines. The planner keeps `frame="wind"` emitters
fixed in the tunnel when it rotates the bodies for incidence; the solver itself has one frame.

## 18. Fans: actuator strips

### 18.1 Momentum source

A fan is an actuator disc — in 2D a strip of `width` across and `thickness` along its
`direction` (default width/8, at least two cells) — that adds the body force
`f = T φ n̂ / ∫φ dA` with φ the smooth box footprint evaluated on the u- and v-faces. The
discrete integral of the force is exactly `T n̂`: in a periodic box (no walls, no outlet) the
total momentum grows at exactly the thrust (1e-10,
`test_fan_adds_exactly_its_thrust_in_a_periodic_box`). "thrust" sets T (per unit span),
"pressure_jump" sets `ΔP` with `T = ΔP · width`.

### 18.2 Velocity mode: an in-graph PI controller

"velocity" regulates ΔP so that the footprint-averaged speed along the axis, u_d, reaches
the target. The plant's quasi-steady slope follows from momentum theory: with the
free-stream component U along the axis, `ΔP = T/D = 2 u_d (u_d − U)`, so
`dΔP/du_d = 2 |2 u_d − U|` (U = 0 in a closed room). The PI gains make the loop gain one
(`k_p` = that slope) and the integral loop decay at `ω = |target| / (2 D)`
(`k_i = 2 ω k_p`), with anti-windup by clamping the integral so ΔP stays within
`50 max(target², 1)`. The controller state lives in the solver state (`ctrl`), so it runs
inside the CUDA graph. Measured (tunnel, target 1.5, width 1, Re 100): 2.0 % short of the
target at t = 3, 0.15 % at t = 7, 0.006 % at t = 12, no overshoot. The first gains were
sized for the fan's own fluid mass only and ignored that most of an increment of thrust
accelerates the far wake; that loop was four times too slow (2 % short of the target at
t = 25).

### 18.3 Fan report

`fans_report()` gives per fan the delivered thrust, the pressure jump, the volume flow
(u_d × width), the disc and induced velocity (u_d minus the free stream along the axis) and
the ideal power T u_d (instantaneous or time-averaged), plus the per-step history.

### 18.4 Fans in closed 2D rooms: a grid study, and the wall model

**What is measured.** The app's fan room (`fan_room_case`: a 1 N/m thrust fan 0.5 m across,
a third of the way along a 4 × 3 m air room with no-slip walls on every side; the planner
picks WALE LES and upwind3 at Re = U L/ν = 3.1 × 10⁴ with U = 0.904 m/s, the fan's static
momentum-theory velocity, and L = 0.5 m) was spun up from rest to 1200 L/U on three grids,
with and without the wall model below (`run_fan_room`, `validation/bench_ns2d.py fan-room`;
the configuration matches `api.plan` field by field). The room-mean speed is the mean of
|u| over all cells; each value is the average of samples every 2 L/U over t ∈ [800, 1200],
"spread" the range of the 100-L/U window means over [600, 1200], "drift" the change between
the halves of [800, 1200]. RTX 4060 Ti, float32, GPU shared with other jobs.

| cells/L (h) | wall model | mean speed [m/s] | spread of 100-L/U means | max speed [m/s] | disc velocity [m/s] | drift |
|---|---|---:|---:|---:|---:|---:|
| 6 (83 mm) | none | 2.37 | 2.27–2.37 | 8.3 | 5.3 | −0 % |
| 6 (83 mm) | Werner–Wengle | 2.30 | 2.29–2.30 | 8.2 | 5.2 | +0 % |
| 12 (42 mm, Draft) | none | 2.42 | 1.96–3.76 | 5.9 | −0.0 | −25 % |
| 12 (42 mm, Draft) | Werner–Wengle | 4.18 | 3.99–4.30 | 13.5 | 2.7 | −4 % |
| 24 (21 mm, Fine) | none | 3.42 | 3.01–3.70 | 12.2 | 1.5 | −4 % |
| 24 (21 mm, Fine) | Werner–Wengle | 3.35 | 2.72–4.39 | 15.9 | 4.5 | +39 % |

Without the model the settled mean speed rises with the resolution (2.37 → 2.42 → 3.42 m/s,
+44 % from 6 to 24 cells/L, no sign of levelling off): a finer grid has less numerical
dissipation, and that dissipation is one of the room's two brakes. 4.6 × 10⁶ steps and
103 min of GPU time in all for the six runs.

Over the Draft run's own averaging window (t ∈ [120, 240] L/U) the rooms without the model
average 1.65, 1.81 and 2.34 m/s on the three grids (20-L/U window means 1.36–2.31 on the
Draft grid) — the 1.72 m/s a Draft run of the example reports is one sample of a transient.

**What it shows.** The 2D closed fan room has no grid-converged answer. Its mean speed
changes by up to 82 % between grids and by nearly a factor of two within one run, the
rooms take 500–900 L/U to leave their start-up transient (the Draft plan runs 240), only
the coarsest rooms reach a steady state — on the finer grids the 100-L/U means keep
wandering by 8–92 % over the last 600 L/U, switching between flow states (one room-filling
vortex with the fan in its fast outer ring, or several cells with the fan in a slow core,
which is why the disc velocity ranges from 0 to 5.3 m/s) — and every room runs at 2.5–4.6
times the fan's own static velocity, with peaks of 6–16 m/s. A 1 N/m fan does not do that to a real room.
The reason is the dimension, not the grid: the fan's power T u_d can only leave a 2D room
through wall friction and the subgrid and upwind dissipation of the resolved eddies. 2D
flow has no vortex stretching, so energy cascades to *large* scales (Kraichnan 1967) and
gathers in a few room-sized vortices that dissipate little; in a 3D room the fan's jet
breaks down into 3D turbulence, which dissipates the power in the jet and the room interior.
The 2D numbers are therefore set by under-resolved wall friction and numerical
dissipation, which is grid-dependent by construction.

**The wall model, and why it is off.** With no-slip ghost cells the wall stress is
ν u₁ / y at y = h/2. On the Draft grid (h = 42 mm) the first face sits at y⁺ ≈ 170 for
2 m/s near the wall, where the turbulent friction of the law below is 9.6 times the
resolved one (5.7 times at 1 m/s, 16 times at 4 m/s), and the resolved one scales with 1/h — the obvious suspect for the grid dependence. `wall_model="werner_wengle"`
adds the Werner & Wengle (1991) law: inverting u⁺ = 8.3 (y⁺)^(1/7) for the friction velocity
gives the explicit `τ = (|u₁| (ν/y)^(1/7) / 8.3)^(7/4)`, which equals ν|u₁|/y at
y⁺ = 8.3^(7/6) = 11.81 and exceeds it above; the solver adds the excess
`max(τ − ν|u₁|/y, 0)` as a force τ/h on the face row next to each no-slip domain wall (u
next to the floor and ceiling — relative to a moving road — and v next to the side walls of
a closed box). So a resolved wall (first face in the viscous sublayer) is left exactly as
it was. Against the log law u⁺ = ln(y⁺)/0.41 + 5.2 the power law's stress is within +4.5 /
−1.7 % for 30 ≤ y⁺ ≤ 1000 (−8.7 % at 3000, −8.4 % at y⁺ = 12). The model is exact in its own
terms (tests: the stress is the power-law u_τ² to round-off above the crossover and zero
below, it only removes kinetic energy, it reaches only the wall faces, it is
differentiable; `tests/test_ns2d_wallmodel.py`).

It does not rescue the 2D room. With the turbulent friction in place the rooms organise into
one strong vortex sooner and the 12-cell room *speeds up* (4.18 m/s against 2.42): once the
wall is the sink, the steady speed follows from T u_d ≈ τ(U) U × perimeter, which for
τ ∝ U^(7/4) gives several m/s, and the 24-cell room still drifts by 39 %. Its three grids
(2.30, 4.18, 3.35 m/s) are no closer to agreement than the three without it. Correct wall
friction cannot replace the missing 3D interior dissipation. The model therefore stays
opt-in (`SolverConfig.wall_model`, default "none"), so every validated case — and every
planned run — is unchanged; it is the right tool for a 2D flow whose losses really are at
the walls (a channel or a lid-driven cavity at high Re), not for a room.

**What to tell the user.** In a 2D closed room with fans, the delivered thrust is exact
(§18.1) and the *pattern* (where the jet goes, where it recirculates) is meaningful, but
speeds, the disc velocity and anything derived from them (volume flow, ideal power,
convective heat transfer) are qualitative and grid-dependent, and Froude momentum theory
does not apply (the fan recirculates its own wake). Use the 3D solver for rooms.

## 19. Ports (jets and vents) and painted obstacles

### 19.1 Jets and vents

A jet creates fluid and blows it out of its open face; a vent sucks fluid in and removes it.
The port body — a region `depth` deep (default three cells) behind the face of a jet, beyond
it for a vent — carries a uniform mass source of exactly `speed × size` per unit span, and the
projection solves `∇·u = source` instead of zero (`L ψ = ∇·û − s`). The velocity inside the
body is penalised (with the momentum η) toward the field of that source: zero at the closed
back, `speed` along the direction at the open face (a linear ramp, the continuity solution of
a uniform source). Measured: the flux through the first face line past a jet's open face is
`speed × size` to round-off (10⁻¹⁶ relative in float64; the test checks 10⁻⁹,
`test_jet_and_vent_in_a_closed_box_conserve_mass`), the velocity there is 0.9987 `speed`,
and the momentum flux two widths downstream is 1.023 × that of a top-hat (§21.5). A jet can
carry a temperature (penalised in its body) and smoke; a vent removes smoke and heat with
its fluid.

### 19.2 Mass balance and the Neumann compatibility

A tunnel's outlet is corrected so that outflow = inflow + the ports' net volume flow (to
1e-10, `test_jet_in_a_tunnel_balances_at_the_outlet`). In a closed (or periodic) domain the
all-Neumann pressure problem needs a zero-sum right-hand side; a balanced set of ports
satisfies that exactly, and an unbalanced one is made compatible by spreading the excess as
a uniform sink over the domain (the planner warns: physically the box would pressurise). The
projection then stays exact (residual and divergence below 1e-10 in float64,
`test_unbalanced_port_in_a_closed_box_keeps_the_projection_exact`).

The fluid that uniform sink removes takes its heat and smoke with it (review round 5 before the release): when
the ports' net flow q is positive, the scalar sink `src_neg` (the vents' own `m < 0`) also
carries `−q/A` on every cell, so θ and smoke lose `q θ / A` locally and the reported vent heat
includes the leak. Before, the conservative scalar update kept them — a source `+θ q/A`
everywhere — and a 64² closed adiabatic box fed by a θ = 1 jet (q = 0.1) with no vent climbed
to mean θ 2.05 and max 2.77 by t = 21.4 while its heat balance still "closed" (heat out of the
ports 0). Now max θ stays at 0.988, the mean reaches 0.910 (the well-mixed estimate
1 − exp(−q t / V) is 0.882; the jet's penalised footprint puts in a little more than q·1),
heat out of the ports is 0.091 against 0.100 in, and the residual is 1.6 × 10⁻⁶. A net sink
(only vents) needs nothing: the fluid the solve creates evenly arrives at θ = 0 and dilutes
through the conservative flux.

### 19.3 Painted obstacles (explore mode)

`paint_disc(center, radius, erase=False)` and `paint_polygon(points)` add or remove solid in a
brush footprint; `clear_paint()` removes all. The paint layer is penalised to rest with the
momentum η, insulates heat and clears smoke, and stays outside the differentiable body
geometry (sensitivities never see it). Its solid fraction is evaluated on each grid's own
points (centres, u-faces, v-faces) with the bodies' calibrated interface (one-cell C¹ step
shifted 0.32 h inside the outline), and its force is the penalisation impulse weighted by the
paint's share of the penalisation at each face — exactly the bodies' force rule. A painted disc
is therefore the same obstacle as an `ImmersedBody` of that outline: drag 2.1338 against
2.1337 (+0.006 %), the same mask to 10⁻⁴ and the same residual velocity inside
(`test_paint_erase_and_clear`, which allows 1 % and 0.02). (A first version averaged the
face fractions from the centres and weighted the impulse by the solid fraction itself, which
discounted the interface where most momentum is removed: half the body's drag.) Brush
strokes rewrite tensors in place, so after the first stroke no graph is re-captured.

## 20. Live editing, diagnostics, solver trace and the PID time step

### 20.1 Live edits

`set_inflow(re=…)` (ν, η, and with heat transfer κ = 1/(Re Pr) and η_θ), `set_bodies`,
`set_emitters`, `set_sources(heat_sources=, fans=, ports=)` and `set_thermal(richardson=,
prandtl=, walls=, theta_inlet=)` apply without reallocating the grid or the flow: an edit that
keeps the structure rewrites the existing tensors (tests check the `Physics` object is the same
one and, on CUDA, that no graph is re-captured); a structural change rebuilds the placed
physics, keeps u, v, p, θ and the smoke already in the flow, and re-captures lazily.

### 20.2 PID-controlled time step

`adaptive_dt="pid"` replaces the jumpy CFL rule (it follows every change of max|u|, e.g. a
vortex passing the location of the maximum) by a PID controller in log space that steers the
step toward 0.9 × the stability limit: error `e = log(0.9 Δt_lim / Δt_prev)`, update
`Δt = Δt_prev exp(k_I e + k_P (e − e₁) + k_D (e − 2e₁ + e₂))` with (k_P, k_I, k_D) =
(0.10, 0.25, 0.02), growth at most 3 % per step and never above the instantaneous limit
(including the temperature bound of §16.1). Measured (cylinder, 300 steps): largest growth
1.0019 per step, largest CFL number 0.459 (`test_pid_time_step_is_smooth_and_stable` holds
1.03 and 0.5).

### 20.3 Diagnostics and the solver trace

`SolverConfig.diagnostics=True` records, on the last step of every block of 10 steps (the
CUDA-graph block) and without a host sync: t, Δt, the CFL number, RMS and max divergence
(minus the port sources), the relative residual `‖Lψ − f‖/‖f‖` of the last pressure solve,
and the L2, RMS, max and relative change per step of u, v, p and θ, into a float64 device ring
copied to the host once per `advance`; throughput (steps/s per `advance`) is measured on the
host. The pressure solve is direct (exact DCT), so its residual is a round-off number: it
proves the solve exact on this grid and would expose a broken compatibility condition at once
(below 1e-10 in every float64 test). `diagnostics()` returns the table,
`latest_diagnostics()` the last row (for `CFDResult.diagnostics`), `diagnostics_csv(path)`
writes it.

Cost of each feature (`bench_ns2d.py physics-perf`: the cylinder case of §10 at 1024 × 512,
float32, CUDA graphs, variants interleaved over three rounds, median; measured while another
process kept the GPU at 100 % utilisation, so the absolute rates are about 4× below the
idle-machine figures of §10 and the relative costs are indicative):

| variant | steps/s | vs default |
|---------|--------:|-----------:|
| default (no placed physics) | 113.2 | — |
| `diagnostics=True` | 111.1 | −1.8 % |
| `adaptive_dt="pid"` | 111.9 | −1.1 % |
| two smoke channels (point + 8-nozzle rake) | 93.5 | −17 % |
| tracer particles, 8 nozzles | 95.0 | −16 % |
| heat transfer, forced (hot cylinder) | 73.2 | −35 % |
| heat transfer + buoyancy | 69.2 | −39 % |
| fan, velocity mode | 104.2 | −8 % |
| jet + vent | 99.7 | −12 % |
| everything at once | 47.7 | −58 % |

Temperature costs most: a second transported field with its own TVD fluxes, wall
conditions, penalisation and heat-rate bookkeeping in each of the three Runge–Kutta stages.

`trace_step(commit=False)` runs one step eagerly and returns every stage as cell-centred fields:
the explicit update (advection, diffusion, forces), the penalised field, the Poisson right-hand
side (the divergence to remove), the pressure potential, the correction (−∇ψ) and the projected
field, with θ before and after its penalisation, and the final u, v, p, θ. By default the run is
untouched (the step is computed on the side); the stages satisfy projected = penalised +
correction exactly (`test_trace_step_stages_are_consistent`).

## 21. Validation of the placed physics

The cases live in `validation/thermal2d_cases.py`, whose docstring names the source every
reference number was read from; `python validation/bench_ns2d.py thermal` runs the grid
studies below and the slow tests `tests/test_ns2d_physics_validation.py` (`-m slow`) hold
one resolution of each to a tolerance band. Float32 on the GPU unless stated, upwind3 advection,
CUDA graphs. Exact-balance checks (answers known to round-off) are in §21.6.

### 21.1 Differentially heated square cavity (de Vahl Davis 1983)

Unit square, no-slip walls, hot left wall θ = +½, cold right wall θ = −½, adiabatic top and
bottom, Pr = 0.71, Boussinesq buoyancy with the free-fall scaling of §16.5 (`cavity_config`).
Each run is integrated until the hot-wall Nusselt number changes by less than 10⁻⁶ per
free-fall time over two consecutive checks (`run_cavity`). Quantities, in de Vahl Davis'
units (velocities in κ/L): `Nu` the hot-wall mean from the scheme's own boundary fluxes (the
cold wall gives the same value to ≤ 4 × 10⁻⁷ relative in every run: the energy balance of a
steady state), `Nu½` the horizontal heat flux uθ − ∂θ/∂x averaged over the vertical mid-plane
(the benchmark's Nu₁/₂), `u_max` on the vertical mid-plane with its height y, `v_max` on the
horizontal mid-plane with its position x (peaks refined by a parabola through the three
largest samples), `ψ_mid` the stream function at the centre. Reference: de Vahl Davis (1983),
Table V (the bench-mark solution), in the last row of each block; `ext.` is the Richardson
extrapolation of the three grids above it with the observed order p (`richardson`).

| Ra | grid | Nu | Nu½ | u_max (y) | v_max (x) | ψ_mid |
|----|-----:|---:|----:|----------:|----------:|------:|
| 10³ | 32² | 1.1201 (+0.19 %) | 1.1196 | 3.663 (0.8135) | 3.711 (0.1785) | 1.183 |
| | 64² | 1.1184 (+0.03 %) | 1.1182 | 3.653 (0.8133) | 3.700 (0.1784) | 1.177 |
| | 128² | 1.1179 (−0.01 %) | 1.1179 | 3.650 (0.8132) | 3.698 (0.1783) | 1.175 |
| | ext. (p) | 1.1177 (1.90) | | 3.6497 (2.19) | 3.6978 (2.30) | 1.1747 (2.02) |
| | dVD | 1.118 | 1.118 | 3.649 (0.813) | 3.697 (0.178) | 1.174 |
| 10⁴ | 32² | 2.2690 (+1.16 %) | 2.2691 | 16.239 (0.8230) | 19.710 (0.1186) | 5.108 |
| | 64² | 2.2507 (+0.34 %) | 2.2506 | 16.195 (0.8232) | 19.641 (0.1189) | 5.082 |
| | 128² | 2.2462 (+0.14 %) | 2.2462 | 16.187 (0.8232) | 19.631 (0.1188) | 5.076 |
| | ext. (p) | 2.2448 (2.05) | | 16.185 (2.49) | 19.629 (2.76) | 5.074 (2.10) |
| | dVD | 2.243 | 2.243 | 16.178 (0.823) | 19.617 (0.119) | 5.071 |
| 10⁵ | 64² | 4.5624 (+0.96 %) | 4.5615 | 34.936 (0.8546) | 68.713 (0.0657) | 9.165 |
| | 128² | 4.5315 (+0.28 %) | 4.5313 | 34.790 (0.8546) | 68.672 (0.0659) | 9.128 |
| | 256² | 4.5238 (+0.11 %) | 4.5242 | 34.760 (0.8546) | 68.649 (0.0658) | 9.120 |
| | ext. (p) | 4.5213 (2.01) | | 34.752 (2.26) | 68.620 (0.86) | 9.118 (2.24) |
| | dVD | 4.519 | 4.519 | 34.73 (0.855) | 68.59 (0.066) | 9.111 |
| 10⁶ | 64² | 9.1051 (+3.47 %) | 9.1035 | 66.585 (0.8499) | 222.47 (0.0379) | 16.737 |
| | 128² | 8.8903 (+1.03 %) | 8.8904 | 65.255 (0.8499) | 221.33 (0.0377) | 16.473 |
| | 256² | 8.8409 (+0.46 %) | 8.8410 | 64.941 (0.8499) | 220.72 (0.0378) | 16.408 |
| | ext. (p) | 8.8262 (2.12) | | 64.844 (2.08) | 220.00 (0.89) | 16.386 (2.00) |
| | dVD | 8.800 | 8.799 | 64.63 (0.850) | 219.36 (0.0379) | 16.32 |

Reading the table:

* **Order.** Nu, u_max and ψ_mid converge monotonically at the formal second order (observed
  p = 1.9–2.5). The conservative wall flux of §16.4 matters here: the Nusselt number is the
  flux the scheme used, not a one-sided gradient re-computed afterwards. v_max at Ra 10⁵ and
  10⁶ converges irregularly (p ≈ 0.9): its peak sits 0.066 and 0.038 L from the wall, 4–17
  cells on these grids, where the parabolic peak refinement is itself first order; read the
  extrapolation of v_max there as indicative only.
* **Accuracy.** The extrapolated values are within 0.08 % of de Vahl Davis for Ra 10³–10⁵
  (Nu, u_max, v_max, ψ_mid) and within 0.3–0.4 % at Ra 10⁶; the fine-grid values are within
  0.15 % (Ra ≤ 10⁵, 128²–256²) and 0.6 % (Ra 10⁶, 256²). The peak positions agree to the
  third decimal everywhere. de Vahl Davis' values are themselves extrapolated from finite-
  difference solutions and quoted to three or four significant figures, so differences of a
  few 10⁻⁴ are below what the reference resolves.
* **Grid study of the uncertainty machinery.** `validation/run_all.py` (case `gci_cavity`)
  feeds the Ra 10³ runs at 16², 32², 64² and 128² to `aerolume.analysis.gci` and checks that
  the GCI band of the 64² value holds the 128² value — the cavity is the case where that can
  be tested against a known answer.
* **Cost and reproducibility.** The temperature step bound of §16.1 (the discrete maximum
  principle) is the binding limit in these cavities: Ra 10⁶ on 128² cells takes 27 000 steps
  to t = 130 (85 s, GPU shared with other jobs); with the momentum limits alone the same run
  took 14 000 steps to t = 125, 1.9 times fewer per unit time. The table above was
  produced before that bound was added; re-running Ra 10⁴ on 64² and Ra 10⁶ on 128² with the
  final code (smaller steps, the same steady state) reproduces it to 2 × 10⁻⁵ relative
  (Nu 2.2507 and 8.8901 against 2.2507 and 8.8903).

### 21.2 Forced convection from a heated cylinder

The momentum validation's cylinder (§11: domain [−10, 20] × [−10, 10] D, slip top and bottom,
5 % blockage, uniform inflow) with the cylinder held at θ = 1 in a θ = 0 stream, Pr = 0.71,
no buoyancy, averaged over t ∈ [50, 60] (`heated_cylinder_config`, `run_heated_cylinder`).
`Nu` is the mean Nusselt number from the penalisation heat rate (§16.4), `Nu_surf` the
integral of the local Nusselt number of `surface()` (the two-probe wall gradient), `Nu_front`
/ `Nu_rear` the local values at the stagnation points. References (unconfined, Pr 0.71, as
tabulated by Singhal, Dutta & Kalita 2023, Table 2): Re 20: 2.4483 (Biswas & Sarkar 2009),
2.433 (Jafroudi & Yang 1986); Re 40: 3.2351 (Biswas & Sarkar), 3.2 (Jafroudi & Yang), 3.255
(Apelt & Ledwich 1979).

| Re | cells/D | grid | Nu | Nu_surf | Cd | Nu_front | Nu_rear | vs Biswas & Sarkar |
|----|--------:|------|---:|--------:|---:|---------:|--------:|-------------------:|
| 20 | 16 | 480×320 | 2.4998 | 2.5993 | 2.1664 | 4.633 | 1.084 | +2.10 % |
| 20 | 32 | 960×640 | 2.5034 | 2.5206 | 2.1663 | 4.456 | 1.067 | +2.25 % |
| 40 | 16 | 480×320 | 3.3439 | 3.4874 | 1.6203 | 6.943 | 1.319 | +3.36 % |
| 40 | 32 | 960×640 | 3.3355 | 3.3600 | 1.6120 | 6.448 | 1.300 | +3.10 % |

Domain study at Re 40, 16 cells/D: half-height 20 D (blockage 2.5 %, upstream 15 D,
downstream 25 D) gives Nu 3.3129 (−0.93 %), half-height 40 D (blockage 1.25 %, upstream 20 D,
downstream 30 D) 3.3014 (−1.27 %); Cd falls from 1.620 to 1.563 and 1.543.

* **Grid.** Nu changes by +0.14 % (Re 20) and −0.25 % (Re 40) from 16 to 32 cells/D, so the
  discrete heat rate is converged to a few tenths of a per cent at 16 cells/D. The local wall-
  gradient value `Nu_surf` converges more slowly (4.0–4.3 % above the heat-rate value at 16
  cells/D, 0.7 % at 32): like cf (§6.5) it differentiates a field across the diffuse interface. The
  reported Nu always comes from the heat rate.
* **Accuracy.** At 5 % blockage the heat rate is 2.1–3.4 % above the unconfined references —
  most of it the same blockage that raises the drag (§11). Shrinking the blockage to 1.25 %
  lowers Nu by 1.3 %, leaving +2.0 % against Biswas & Sarkar and +1.4 % against Apelt &
  Ledwich at Re 40. The published values themselves spread by that much (3.20–3.255 at
  Pr 0.71, and up to 3.28 at Pr 0.70, below); the Pr 0.70 study below places the
  extrapolated unconfined value inside that spread. The slow tests hold Nu to ± 4 % of
  Biswas & Sarkar at 16 cells/D and 5 % blockage, and at Pr 0.70 on the 2.5 % domain to
  ± 1.5 % of Bharti et al. (Re 20) and of Lange et al.'s correlation (Re 40)
  (`test_heated_cylinder_pr070_references`).
* **Physics.** The local Nusselt number (`surface()["nu_local"]`) at the front stagnation
  point is 4.2–5.3 times that at the rear (the thin front boundary layer against the
  recirculating wake).

**At Pr 0.70, against Lange et al. and the Pr 0.70 solutions.** The same case at Pr 0.70, the
Prandtl number of Lange, Durst & Breuer's (1998) correlation `Nu = 0.082 Re^0.5 + 0.734 Re^x`,
`x = 0.05 + 0.226 Re^0.085` (2.4087, 2.8774 and 3.2805 at Re 20, 30, 40), at 16 cells/D on
the three domains of the study above (`CYLINDER_DOMAINS`; `bench_ns2d.py thermal`). The
other Pr 0.70 references are secondary tabulations (the originals were not accessible):
Bharti, Chhabra & Eswaran (2007) 2.4653 and 3.2825, and Paramane & Sharma (2009) 2.4189 and
3.2465 at Re 20 and 40, as tabulated by Sharma & Dhiman (2012, Tables 1–2); Dennis, Hudson &
Smith (1968) 2.5216 and Lange et al. 2.4087 at Re 20, as tabulated by the Notus CFD validation
pages. The correlation as implemented reproduces Notus' value for Lange et al. to its four
decimals (2.40868; `test_cylinder_reference_data_and_lange_correlation`), which is the check
that its form is theirs. "drift" is the relative change of the heat rate between the halves
of the averaging window t ∈ [50, 60]. RTX 4060 Ti, float32, GPU shared with other jobs.

| Re | blockage | grid | Nu | vs Lange | vs Bharti | vs Paramane & Sharma | vs Dennis | Cd | drift | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 % | 480×320 | 2.4879 | +3.29 % | +0.92 % | +2.85 % | −1.34 % | 2.166 | −5 × 10⁻⁶ | 41 s |
| | 2.5 % | 640×640 | 2.4635 | +2.28 % | −0.07 % | +1.84 % | −2.30 % | 2.085 | −3 × 10⁻⁵ | 69 s |
| | 1.25 % | 800×1280 | 2.4542 | +1.89 % | −0.45 % | +1.46 % | −2.67 % | 2.055 | −6 × 10⁻⁵ | 112 s |
| | → 0 | extrapolated | **2.4484** | **+1.65 %** | **−0.68 %** | **+1.22 %** | **−2.90 %** | | | |
| 30 | 5 % | 480×320 | 2.9430 | +2.28 % | | | | 1.818 | −1 × 10⁻⁵ | 33 s |
| 40 | 5 % | 480×320 | 3.3270 | +1.42 % | +1.36 % | +2.48 % | | 1.620 | −3 × 10⁻⁵ | 26 s |
| | 2.5 % | 640×640 | 3.2962 | +0.48 % | +0.42 % | +1.53 % | | 1.563 | −9 × 10⁻⁵ | 44 s |
| | 1.25 % | 800×1280 | 3.2848 | +0.13 % | +0.07 % | +1.18 % | | 1.543 | −1.4 × 10⁻⁴ | 64 s |
| | → 0 | extrapolated | **3.2781** | **−0.07 %** | **−0.13 %** | **+0.97 %** | | | | |

* **Domain.** Halving the blockage twice lowers Nu by 0.0244 then 0.0093 (Re 20) and 0.0308
  then 0.0114 (Re 40): an observed order of 1.39 and 1.44 in the blockage. Richardson-
  extrapolating the three domains to zero blockage (as a proxy for the unconfined cylinder;
  the inflow and outflow move out with the walls) gives 2.4484 and 3.2781, 1.6 % and 1.5 %
  below the 5 % domain's values and 0.2 % below the 1.25 % domain's. With the grid effect
  measured at Pr 0.71 above (+0.14 % and −0.25 % from 16 to 32 cells/D), the discretisation
  and domain uncertainty of the extrapolated values is about ± 0.3 %. The largest domain is
  also the slowest to settle (its heat rate still falls by 1.4 × 10⁻⁴ across the averaging
  window), which is two orders below the differences discussed here.
* **Agreement.** The extrapolated unconfined Nusselt numbers lie inside the spread of the
  published Pr 0.70 values (Re 20: 2.4087–2.5216, 4.7 %; Re 40: 3.2465–3.2825, 1.1 %): at
  Re 40 within 0.13 % of both Bharti et al. and Lange et al.'s correlation; at Re 20 0.7 %
  below Bharti et al., 1.7 % above the correlation (a fit to Lange et al.'s own solutions)
  and 2.9 % below Dennis et al., which bracket it. Against the correlation the difference
  falls steadily with Re: +3.29, +2.28 and +1.42 % at 5 % blockage for Re 20, 30, 40.
* **Prandtl number.** At 5 % blockage Nu(Pr 0.70) / Nu(Pr 0.71) = 0.9952 (Re 20) and 0.9949
  (Re 40), i.e. Nu ∝ Pr^0.34 and Pr^0.36 — the Pr^(1/3) of a thin thermal boundary layer.
  The Pr 0.71 table and the Pr 0.70 table are therefore consistent with each other.

### 21.3 Rayleigh–Bénard onset

Linear stability between rigid isothermal plates predicts onset at Ra_c = 1707.76 with wave
number k_c = 3.117 (Chandrasekhar 1961; Reid & Harris 1958), independent of Pr. The case
(`rb_config`, `run_rb_growth`) is one convection roll: a box of width π/k_c with stress-free,
adiabatic side walls (which hold exactly that mode), no-slip plates at θ = ±½, Pr 0.71,
float64. It starts from the conduction profile plus a 10⁻⁴ roll perturbation, fits the
exponential growth of the kinetic energy over the second half of 120 free-fall times, and
interpolates the growth rates at Ra 1600 and 1800 to zero (`rb_onset`).

| cells across the layer | σ(1600) [κ/L²] | σ(1800) [κ/L²] | Ra_c | error |
|-----------------------:|---------------:|---------------:|-----:|------:|
| 16 | −0.503 | +0.850 | 1674.4 | −1.96 % |
| 32 | −0.678 | +0.671 | 1700.6 | −0.42 % |

The error falls by 4.7× per halving of h, consistent with second order; assuming p = 2 the
two grids extrapolate to 1709.3 (+0.09 %). (The validation runner's quick case, 24 cells
and growth rates at Ra 1500 and 2000, gives 1701.)

### 21.4 Actuator strip in a channel (momentum theory)

A fan of width D = 1 and thickness D/8 normal to the flow in a slip-walled tunnel of height
10 D (blockage B = 0.1), "thrust" mode with T = C_T · ½ρU²D, Re = 500, averaged over
t ∈ [30, 60] (`run_actuator_disc`). The disc velocity u_t is the fan report's footprint-
averaged axial speed. Reference: quasi-inviscid confined momentum theory (the streamtube
model of Garrett & Cummins 2007: continuity, Bernoulli in the core and the bypass, channel
momentum balance; `channel_disc_theory`, cross-checked against the independent
implementation in `validation/run_all.py` to 10⁻⁶), which for B → 0 recovers Froude's
C_T = 4a(1 + a) (propeller) and 4a(1 − a) (turbine).

| fan | C_T | cells/D | u_t | theory | error | induction a | theory a | error a | Froude a (B = 0) |
|-----|----:|--------:|----:|-------:|------:|------------:|---------:|--------:|-----------------:|
| propeller (blows with the stream) | 0.5 | 16 | 1.0998 | 1.1030 | −0.29 % | 0.0998 | 0.1030 | −3.1 % | 0.1124 |
| | 0.5 | 32 | 1.1013 | 1.1030 | −0.15 % | 0.1013 | 0.1030 | −1.6 % | |
| | 0.8 | 16 | 1.1530 | 1.1576 | −0.40 % | 0.1530 | 0.1576 | −2.9 % | 0.1708 |
| | 0.8 | 32 | 1.1553 | 1.1576 | −0.20 % | 0.1553 | 0.1576 | −1.4 % | |
| turbine (blows against it) | 0.5 | 16 | 0.8794 | 0.8736 | +0.66 % | 0.1206 | 0.1264 | −4.6 % | 0.1464 |
| | 0.5 | 32 | 0.8775 | 0.8736 | +0.45 % | 0.1225 | 0.1264 | −3.1 % | |
| | 0.8 | 16 | 0.7901 | 0.7782 | +1.54 % | 0.2099 | 0.2218 | −5.4 % | 0.2764 |
| | 0.8 | 32 | 0.7871 | 0.7782 | +1.15 % | 0.2129 | 0.2218 | −4.0 % | |

The delivered thrust equals the set thrust to 10⁻⁷ in every run (§18.1). The disc velocity
is within 0.4 % (propeller) and 1.5 % (turbine) of the inviscid theory; the induction, a
small difference u_t − 1, is within 1.4–5.4 % and converges toward the theory with the
grid. The remaining shortfall is expected from what the theory leaves out: viscosity at
Re 500 (the wake mixes with the bypass and the core never reaches its inviscid far-wake
speed), the finite thickness and smooth edges of the strip. The confinement matters — the
unconfined Froude values are 8–25 % larger — so the comparison is made with the confined
model.

### 21.5 Plane jet from a port

A jet port of width b = 1 at speed 1 (Re_j = 25) blows along +x into a closed box 80 b
long and 80 b tall, 8 cells per b; a vent across the far wall removes the jet's volume
flow as a uniform drift. Mean fields over t ∈ [300, 400] (`run_plane_jet`, `jet_measures`).
Reference: Bickley's (1937) similarity solution of the plane laminar jet, u_c =
0.4543 (K²/(νx))^(1/3) with the kinematic momentum flux K = ∫u² dy conserved, the half-width
b½ = (0.8814/0.2752)(ν²/K)^(1/3) x^(2/3) and a sech² profile.

| quantity (8 ≤ x/b ≤ 24) | measured |
|-------------------------|---------:|
| K at x/b = 4, 12, 24 | 1.023, 0.970, 0.929 |
| change of K over the span | −6.0 % |
| — shear of the return flow on the jet core, ∫ν[∂u/∂y]edges dx | −3.6 % |
| — the box's streamwise pressure gradient on the core, −∫W dp_edge/dx dx | −2.5 % |
| **momentum-budget residual** of the core | **−0.27 %** |
| centreline decay slope of u_c⁻³ / Bickley | −4.7 % |
| half-width b½ at x/b = 20 (30) / Bickley, fitted virtual origin 1.61 b | −3.9 % (−5.4 %) |
| rms deviation of u/u_c from sech² at x/b = 20 | 0.016 |

* **Momentum is conserved; the jet is confined.** The flux through the first face line past
  a port is speed × width to round-off and the momentum flux two widths downstream is 1.023 × a
  top-hat's (§19.1). Downstream, K = ∫u² dy over the jet's forward-flow core falls by 6 %
  between 8 and 24 port widths. Integrating the steady x-momentum equation across the core
  (`jet_measures`, docstring) splits that change into the shear of the counter-flow with
  which the closed box returns the entrained fluid (3.6 %) and the adverse pressure gradient
  that drives that counter-flow (2.5 %); the remaining 0.27 % is the budget's residual — the
  scheme conserves momentum, the drift is the box. A first version of the case integrated K
  over the whole box height and over 12 ≤ x/b ≤ 40 and reported a 17.5 % drift. A box twice
  as tall and 100 b long (800 × 1280 cells, averaged over t ∈ [350, 500], 515 s on a shared
  GPU) reduces the drift to 5.2 % with the same budget residual (−0.20 %).
* **Similarity.** The centreline decay and the half-width follow Bickley within 4–5 %, and
  the profile is sech² to 0.016 rms. The jet is slightly narrower and decays slightly faster
  than the free jet, both consistent with the momentum it loses to the confinement.
* The validation runner (`validation/run_all.py`, case `plane_jet`) and the slow test
  (`test_plane_jet_momentum_flux`) hold the budget residual below 1 % of K — the
  conservation statement that can be held tight — and the raw drift of K below 8 % over the
  same span. (An earlier runner held the raw drift to 5 % and failed at 6.2 % although the
  budget closed to −0.27 %.)

### 21.6 Exact balances (tests, `tests/test_ns2d_physics.py`)

| check | result |
|-------|--------|
| conduction between two held walls: linear profile, Nu = 1 | 1e-10 |
| heat-flux wall: exact slope, heat in = heat out | to round-off |
| heater in an adiabatic box: d(heat content)/dt = power | 1e-10 |
| heat-flux body: released heat = G P / Pe | 1e-9 |
| hot body in a box: body heat rate = wall heat rate (steady) | 1e-6 |
| energy budget, every heat path at once, steady (tunnel, 64 × 32) | residual 1.8e-6 of the input |
| temperature bounded by its boundary and body values (tunnel, heated cylinder) | θ ∈ [0, 1] |
| stably stratified box stays at rest | 1e-12 |
| fan in a periodic box: d(momentum)/dt = thrust | 1e-10 |
| velocity-mode fan: footprint speed vs target (tunnel, target 1.5) | 0.15 % at t = 7, 0.006 % at t = 12 |
| jet and vent in a closed box: flux past the open face = speed × size | 1e-16 |
| tunnel with a jet: outflow = inflow + ports | 1e-10 |
| unbalanced port in a closed box: projection residual and divergence | < 1e-10 (float64) |
| painted disc vs the same `ImmersedBody`: drag | 2.1338 vs 2.1337 (+0.006 %) |
| particles in uniform flow: x = x_e + age | 1e-9 |
| CUDA graphs vs eager with every feature on | 1e-5 |
| trace stages: projected = penalised + correction | to round-off |
| live edits keep the `Physics` object and do not re-capture graphs | identity / capture count |

## 22. Limitations of the placed physics

* **Boussinesq, constant properties.** Density differences enter only the buoyancy term
  (valid for β ΔT ≪ 1, i.e. temperature differences of tens of kelvin in air); viscosity,
  conductivity and heat capacity do not depend on temperature. No radiation, no
  compressibility, no conjugate conduction inside solids: a body is isothermal, releases a
  prescribed flux, or is insulated.
* **Thermal wall.** The penalised thermal wall sits within ±0.1 cell of the momentum wall
  (§16.2), but the wall temperature of a heat-flux body reads 0.26 G h high (§16.3, first
  order in h). Heat rates are second order; local wall gradients (`nu_local`) need ≳ 32 cells
  across the body to be within 1 % of the heat-rate Nusselt number.
* **Forced convection.** Extrapolated to zero blockage, the cylinder's mean Nusselt number
  lies inside the spread of the published values (Re 40: within 0.13 % of Bharti et al. and
  of Lange et al.'s correlation; Re 20: −0.7 % and +1.7 %; §21.2). A finite domain raises it:
  +1.5 to +1.6 % at 5 % blockage, +0.2 % at 1.25 %. The published values themselves spread
  by 1–5 %, so quote Nusselt numbers with ± 2 % plus the blockage effect of the domain used.
* **Natural convection** is validated for steady laminar cavities up to Ra 10⁶ (§21.1).
  Above that, flows become unsteady and the boundary layers thin as Ra^(−1/4): 256² cells
  at Ra 10⁶ already need 40 000 explicit steps, and turbulent natural convection (Ra ≳ 10⁸)
  is outside what a uniform explicit 2D grid resolves.
* **Fans** are actuator strips: thrust and the momentum-theory induction are reproduced
  (§21.4); there are no blades, no swirl (2D), no fan curve beyond the "velocity" mode's
  regulated pressure jump. Induction is within 1.4–5.4 % of inviscid theory at Re 500.
* **Fans in closed 2D rooms** are qualitative. The thrust is exact, but the room's speeds
  are not grid-converged (settled mean speed 2.4 → 3.4 m/s from 6 to 24 cells per fan width
  for a 1 N/m fan in a 4 × 3 m room, 2.3–4.2 m/s with the wall model, still drifting by up
  to 39 % after 1200 L/U; §18.4): with no vortex
  stretching the fan's power can only leave through under-resolved wall friction and
  numerical dissipation, so a 2D room spins up into room-sized vortices. Momentum theory
  does not apply in a closed room. Use 3D for rooms.
* **Ports** impose a uniform source over the port body and a top-hat velocity at the open
  face (momentum flux 1.023 × the top-hat value two widths downstream). An unbalanced set of
  ports in a closed domain is made solvable by a uniform sink (the planner warns); the real
  box would pressurise. Jets in closed boxes lose momentum to the return flow (§21.5).
* **Emitters** are visualisation: smoke is a passive scalar with the numerical diffusion of
  the TVD scheme (streaks narrower than ~2.5 cells smear); tracer particles are massless,
  follow the bilinearly interpolated velocity with RK2, and carry no diffusion or inertia —
  use them for streaklines and timelines, not for particle statistics.
* **Painted obstacles** are outside the differentiable geometry by design: sensitivities and
  optimisation see only `ImmersedBody` outlines.
* **PID time step.** `adaptive_dt="pid"` smooths the step for live viewing; every
  validation run above uses the CFL rule, and the PID step has been checked for stability
  (CFL ≤ 0.5), not for accuracy of transients.

## 23. References (placed physics)

* Apelt & Ledwich (1979), Heat transfer in transient and unsteady flows past a circular
  cylinder in the range 1 < R < 40, JFM 95, 761–777.
* Bharti, Chhabra & Eswaran (2007), A numerical study of the steady forced convection heat
  transfer from an unconfined circular cylinder, Heat Mass Transfer 43, 639–648 (Pr 0.7 values
  as tabulated by Sharma & Dhiman 2012, §21.2).
* Bickley (1937), The plane jet, Phil. Mag. (7) 23, 727–731.
* Biswas & Sarkar (2009), Effect of thermal buoyancy on vortex shedding past a circular
  cylinder in cross-flow at low Reynolds numbers, Int. J. Heat Mass Transfer 52, 1897–1912.
* Celik, Ghia, Roache, Freitas, Coleman & Raad (2008), Procedure for estimation and reporting
  of uncertainty due to discretization in CFD applications, J. Fluids Eng. 130, 078001.
* Chandrasekhar (1961), Hydrodynamic and Hydromagnetic Stability, Oxford University Press.
* de Vahl Davis (1983), Natural convection of air in a square cavity: a bench mark numerical
  solution, Int. J. Numer. Meth. Fluids 3, 249–264.
* Dennis, Hudson & Smith (1968), Steady laminar forced convection from a circular cylinder at
  low Reynolds numbers, Phys. Fluids 11, 933–940 (Re 20 value as tabulated by the Notus CFD
  validation, §21.2).
* Garrett & Cummins (2007), The efficiency of a turbine in a tidal channel, JFM 588, 243–251.
* Jafroudi & Yang (1986), Steady laminar forced convection from a circular cylinder,
  J. Comput. Phys. 65, 46–56.
* Kadoch, Kolomenskiy, Angot & Schneider (2012), A volume penalization method for
  incompressible flows and scalar advection–diffusion with moving obstacles, J. Comput.
  Phys. 231, 4365–4383.
* Kraichnan (1967), Inertial ranges in two-dimensional turbulence, Phys. Fluids 10,
  1417–1423 (the inverse energy cascade of 2D flow, §18.4).
* Lange, Durst & Breuer (1998), Momentum and heat transfer from cylinders in laminar
  crossflow at 10⁻⁴ ≤ Re ≤ 200, Int. J. Heat Mass Transfer 41, 3409–3430 (the correlation of
  §21.2; original not accessible, its form checked against the value the Notus CFD validation
  tabulates for it at Re 20).
* Notus CFD, validation documentation, "Heated cylinder flow" (Re 20, Pr 0.70; table of mean
  Nusselt numbers with Dennis et al. 1968 and Lange et al. 1998),
  https://doc.notus-cfd.org/d2/d1e/group__doc__validation__heated__cylinder__flow.html.
* Paramane & Sharma (2009), Numerical investigation of heat and fluid flow across a rotating
  circular cylinder maintained at constant temperature in 2-D laminar flow regime, Int. J. Heat
  Mass Transfer 52, 3205–3216 (rotation rate 0, as tabulated by Sharma & Dhiman 2012).
* Reid & Harris (1958), Some further results on the Bénard problem, Phys. Fluids 1, 102–110.
* Roache (1998), Verification and Validation in Computational Science and Engineering,
  Hermosa.
* Sharma & Dhiman (2012), Heat transfer from a rotating circular cylinder in the steady regime:
  effects of Prandtl number, Thermal Science 16(1), 79–91, Tables 1 and 2 (the Pr 0.7 values of
  Bharti et al. and of Paramane & Sharma used in §21.2).
* Singhal, Dutta & Kalita (2023), Int. J. Thermal Sciences 194, 108588 (arXiv:2303.17562),
  Table 2: the compilation of cylinder Nusselt numbers used in §21.2.
* Valli, Carey & Coutinho (2005), Control strategies for timestep selection in finite element
  simulation of incompressible flows and coupled reaction–convection–diffusion processes,
  Int. J. Numer. Meth. Fluids 47, 201–231 (the PID step-size controller family of §20.2).
* van Leer (1979), Towards the ultimate conservative difference scheme V. A second-order
  sequel to Godunov's method, J. Comput. Phys. 32, 101–136 (MUSCL).
* Werner & Wengle (1993), Large-eddy simulation of turbulent flow over and around a cube in
  a plate channel, in Turbulent Shear Flows 8 (Durst et al., eds.), Springer, 155–168
  (presented 1991; the power-law wall model of §18.4).
