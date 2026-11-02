# The Aerolume 3D Navier–Stokes solver — theory, numerics and verification

Module: `aerolume.solvers.ns3d` (`config.py`, `core.py`, `poisson.py`, `geometry.py`, `fused.py`,
`fields.py`, `solver.py`, `rollout.py`, `dye.py`, and for the placed physics of §15–18 `physics.py`
and `emitters.py`). Verification and validation: `validation/ns3d_cases.py`,
`validation/thermal3d_cases.py`, `validation/bench_ns3d.py`; tests: `tests/test_ns3d_core.py`,
`tests/test_ns3d_solver.py`, `tests/test_ns3d_validation.py` (`-m slow`, GPU), `tests/test_ns3d_dye.py`
(smoke, §14), `tests/test_ns3d_physics.py` (heat transfer, fans, ports, §15 and §17),
`tests/test_ns3d_emitters.py` (§16), `tests/test_ns3d_live.py` (§18) and
`tests/test_ns3d_physics_validation.py` (`-m slow`).

The 3D solver is the validated 2D solver ([theory_ns2d.md](theory_ns2d.md)) carried to three
dimensions: the same equations, staggered grid, time integrator, exact projection, immersed
boundary with the same calibrated wall offset, the same force evaluation and the same
differentiable functional core. This document states what is new in 3D, why, and what was
measured. Every number was produced on the development machine (RTX 4060 Ti 8 GB, i7-13700F,
torch 2.14.1+cu130, Windows 11) by the scripts above; the GPU was shared with other work
during most runs, so throughputs and wall times are lower bounds on what an idle machine does
(the exception is the cost table of §18, measured on an idle GPU and checked for contention).

---

## 1. Equations, frame and coefficients

```
∂u/∂t + ∇·(u u) = −∇p + (1/Re) ∇²u + ∇·(2 ν_t S) − (χ/η)(u − u_s)
∇·u = 0
```

non-dimensional with the reference length `L` (sphere diameter, vehicle length, mean
aerodynamic chord), the free-stream speed `U`, time `L/U` and pressure `ρU²`; `Re = UL/ν`.
Frame (DESIGN 7.2): right-handed, **x downstream, y to the left, z up**; the flow enters at
`x = x0` along +x. Attitude is applied to the geometry by the planner, so the solver's forces
are wind-axis forces: Fx drag, Fy side force, Fz lift. Moments about `moment_ref` are the
components of `M = Σ (r − r_ref) × F`: Mx roll, My pitch (positive nose-up: a positive rotation
about +y lifts the nose, which points to −x), Mz yaw. Coefficients:

```
CD, CS, CL = (Fx, Fy, Fz) / (½ ref_area)          Cm = My / (½ ref_area length_ref)
C_roll, Cn = (Mx, Mz) / (½ ref_area span_ref)
```

(`ρU² = 1` in solver units). A lift force behind the moment reference gives a negative
(nose-down) Cm, as in aeronautics; a test pitches a plate nose-up by +10° about +y and checks
that CL > 0.

**Symmetry plane.** With `bc_y = "symmetry_ymin"` the domain is the half `y ≥ y0` of a
mirror-symmetric problem and the plane `y = y0` is a free-slip wall — for incompressible flow a
free-slip wall *is* a symmetry plane (`v = 0`, `∂u/∂y = ∂w/∂y = 0`, `∂p/∂y = 0`), so the
numerics are those of a slip wall. The solver reports **full-body** values: summing the
half-body forces with their mirror images (`y → 2y0 − y`, `Fy → −Fy`) gives exactly

```
Fx = 2 Fx_half,  Fz = 2 Fz_half,  My = 2 My_half,  Fy = 0,
Mx = −(yr − y0) Fz,  Mz = (yr − y0) Fx       (zero when the moment reference lies on the plane)
```

## 2. Grid and boundary conditions

Uniform staggered (MAC) grid of `nx × ny × nz` cells of size `h`:

| quantity | location | shape |
|---|---|---|
| `u` | x-faces `(x0 + i h, y0 + (j+½) h, z0 + (k+½) h)` | `(nz, ny, nx+1)` |
| `v` | y-faces `(x0 + (i+½) h, y0 + j h, z0 + (k+½) h)` | `(nz, ny+1, nx)` |
| `w` | z-faces `(x0 + (i+½) h, y0 + (j+½) h, z0 + k h)` | `(nz+1, ny, nx)` |
| `p`, `χ_c`, `ν_t` | cell centres | `(nz, ny, nx)` |

Boundary conditions enter through two ghost layers per side (`pad3`), filled axis by axis (x,
then y on the x-padded rows, then z on everything, so edge and corner ghosts compose the rules
of the axes involved). The rule of every component on every boundary:

| boundary | u | v | w |
|---|---|---|---|
| inlet x = x0 | repeat inlet value (u = 1) | odd (v = 0 on the plane) | odd |
| outlet | zero gradient; convective `u_t + u_x = 0` + mass-flux correction | zero gradient | zero gradient |
| y walls, `slip` / symmetry plane | even (`∂u/∂y = 0`) | on the face: v = 0, odd | even |
| road z = z0: `slip` / `noslip` / `moving` | even / odd / `2U_g − u` | even / odd / odd | w = 0 on the face |
| top: `slip` / `noslip` | even / odd | even / odd | w = 0 on the face |
| periodic axis | wrap (face arrays keep a bit-identical duplicate face) | wrap | wrap |

After each stage the outlet plane is shifted uniformly so that outflow equals inflow (a device
reduction, no host sync): every other boundary is impermeable or periodic, so the Poisson
right-hand side sums to zero exactly, as the all-Neumann problem requires. `bc_y = "periodic"`
makes y periodic with inlet/outlet in x; `periodic = True` is the triply periodic verification
mode. The fast tests check every ghost rule (wall values interpolate to 0 / `U_g` exactly), the
mass balance to 10⁻¹⁰, and that periodic duplicates stay bit-identical.

## 3. Spatial discretisation

Advection is the 2D conservative form in three directions: for each momentum component the
flux along its own direction lives at cell centres (advecting velocity = the component itself)
and the two cross fluxes on the cell edges, with the advecting velocity averaged to the edge
(e.g. the y-flux of u-momentum at the x–y edges carries `(v_{i−½} + v_{i+½})/2`). Each flux is
the 2D one — central (kinetic-energy conserving with a discretely divergence-free velocity) or
the default `upwind3` (4th-order central part plus a |V|-weighted 4th-difference dissipation, the
3rd-order upwind-biased reconstruction). Diffusion is the 7-point Laplacian (`ν ∇²u`, valid since
`∇·u = 0`).

**Verification — Taylor–Green.** The 2D Taylor–Green vortex is an exact Navier–Stokes solution
in 3D as well; it is set up in each coordinate plane in turn (so every pair of velocity
components and every pair of axes is exercised), periodic, Re = 10, t = 0.5, max velocity
error:

| N | central | order | upwind3 | order |
|---|---|---|---|---|
| 16 | 1.135e-3 | | 5.73e-4 | |
| 32 | 2.890e-4 | 1.97 | 1.74e-4 | 1.72 |
| 64 | 7.257e-5 | 1.99 | 5.28e-5 | 1.72 |
| 128 | 1.816e-5 | 2.00 | 1.54e-5 | 1.77 |

(identical to the printed digits in the x–y, x–z and y–z planes: the discretisation is
isotropic; `bench_ns3d.py verify`)

`central` is cleanly second order in all three planes. As in 2D, `upwind3` has two error terms of
opposite sign whose partial cancellation makes it *more* accurate than central on coarse grids,
with an observed order rising towards 2. The energy-decay rate at N = 64 is
0.9992 × the exact `4/Re` (the discrete Laplacian eigenvalue is `1 − h²/12` of the exact one).
An inviscid periodic random field keeps its energy to 10⁻⁴ over 30 central steps (the residual
is the O(Δt³) Runge–Kutta error).

## 4. Pressure: exact projection with orthonormal transforms

Each Runge–Kutta stage ends with `L ψ = ∇_h·û`, `u = û − ∇_h ψ`, `L` the 7-point Laplacian
with homogeneous Neumann conditions (or periodic) per axis. `L` is a Kronecker sum
`Lx ⊕ Ly ⊕ Lz` of 1D operators whose eigenvectors are known: the DCT-II basis for
cell-centred Neumann, and for periodic axes the **discrete Hartley basis**
`cas(2πkj/n)/√n` (`cas = cos + sin`) — real, symmetric, orthonormal, and an eigenbasis because
the periodic Laplacian is a symmetric circulant (`λ_k = λ_{n−k}`). With the orthonormal matrices
`Q` (rows = eigenvectors) a solve is three forward transforms, a division by
`λx + λy + λz` and three inverse transforms with `Qᵀ`. The constant mode is set to zero.

**Why matrix products instead of FFT-based DCTs** (the 2D solver uses Makhoul's packed DCT):
on the grids this solver runs (≤ ~600 cells per axis) a transform along one axis is a single
cuBLAS GEMM that reads and writes the field once — along x `(nz·ny, nx) @ Qᵀ`, along y a
batched `(ny × ny) @ (ny × nx)`, along z `Q @ (nz, ny·nx)` — with no transposes, no complex
arithmetic and nothing to repack. Measured at 256 × 128 × 128 in float32: 0.9 ms for a forward +
inverse 3D transform, against 0.8 ms for a bare `rfftn` + `irfftn` (which would still need the
DCT pre- and post-processing). The x transform is fed the pre-transposed matrix: cuBLAS's
transposed-operand kernel was up to 4× slower there. The cost per element grows like `n`, so
very long axes would favour FFTs again; at 11.8 M cells (384 × 192 × 160) a solve takes 3.7 ms.

**TF32 must never be used here**: its 10-bit mantissa would leave a divergence of ~10⁻³. Every
solve runs inside `ieee_matmul()`, which forces IEEE float32 for cuBLAS (and oneDNN on the CPU)
through PyTorch's per-backend `fp32_precision` switch and restores the previous setting; a GPU
test enables TF32 process-wide and checks that the projected divergence stays below 10⁻⁵.

Verification: `max|Lψ − f|/max|f|` 4.9 × 10⁻¹⁴ in float64 (Neumann, periodic and mixed, even
and odd sizes), 2.2 × 10⁻⁵ in float32 on the GPU (256 × 128 × 128). Projected divergence of
real solver states (sphere, moving road, no-slip top): 7.3 × 10⁻¹⁶ (float64), 4.0 × 10⁻⁷ (float32,
GPU); the bars are 10⁻¹⁰ and 10⁻⁵. As in 2D the pressure written to the state is the
step-averaged `p = (⅙ψ₁ + ⅔ψ₂ + ψ₃)/Δt`, the pressure of the discrete momentum balance
(theory_ns2d.md §4).

## 5. Time integration and time step

SSP-RK3 with a projection after every stage, exactly as in 2D. The time step is
`Δt = min(cfl·h/(max|u| + max|v| + max|w|), h²/(6(1/Re + max ν_t)))`, `cfl = 0.5`: the sum of the
three component maxima bounds the 3D advective CFL number, and the viscous limit keeps the 2D
solver's 80 % margin (SSP-RK3 is stable to ≈ 2.51 on the negative real axis and the 7-point
Laplacian reaches `12ν/h²`, so the limit is `0.209 h²/ν`). Δt is a device scalar computed
every step; nothing in a step synchronises with the host.

## 6. Immersed bodies

### 6.1 Masks from triangle meshes

Bodies are triangle soups (`ImmersedMesh`), not necessarily closed or consistently wound. The
solid fraction is the 2D compact C¹ half-sine step of the signed distance `d` (negative
inside), shifted `δ = 0.32 h` inwards, half-width `w = h`, evaluated **directly at the u-, v-
and w-face positions** and at cell centres, on a window of cells around the bodies (plus a
margin of `w + 3 + δ` cells, clipped to the domain; boundary faces are never penalised).

The signed distance comes from `aerolume.geometry3d.voxel.sdf_band` when that module is
importable: triangles binned to cells, the exact point–triangle distance within the band
(2.32 h here, enough for the ramp), the sign from ray-parity voting along three axes, which is
robust to holes, randomly flipped triangles and overlapping parts, and a differentiable
recompute of the distance from the selected triangle. The solver evaluates it directly on its
window, which the domain clips — always at a symmetry plane, often at the road. The voxeliser
once classified a face whose probes left its grid as an open sheet and thickened it by half a
cell (distances up to 0.45 h too small next to the plane), so this bridge evaluated a grid
covering the whole body; the voxeliser now grows its own classification field past such a cut,
and a cut window equals a covering grid to rounding (tests: half domains, windows across an
open mesh, open tyres through the road, a sheet across the plane). A grid point exactly on a
face — every row of centres on an axis-aligned face at round coordinates — takes the face's
signed plane distance, so the mask keeps its derivative there (§11).

Without the shared module a private exact implementation runs: the nearest triangle by chunked
brute force (no autograd), the distance recomputed from it with autograd, and the sign from the
generalised winding number (|w| > ½; van Oosterom & Strackee solid angles). It tolerates small
holes and a globally flipped mesh but not inconsistent winding, and its cost is points ×
triangles; it agrees with the shared one to 2 × 10⁻⁸ on a sphere (test). It also uses the signed
plane distance within 0.02 cells of a face.

**The wall calibration carries over unchanged.** The 2D values `η = min(0.05 h, h² Re/60)` and
`δ = 0.32 h` were calibrated on plane Poiseuille flow between penalised slabs — a 1D property of
the discretisation normal to a grid-aligned wall, identical in 3D. The sphere results of §10
(control-volume closure to 0.01–0.03 %, drag converging to the reference values once the confinement is accounted for) are its independent
3D validation. The mask volume of a sphere equals the radial integral of the shifted ramp to
0.5 % (the ramp's first moment adds `≈ 0.19 · 4π r w²` to the volume of its 0.5 iso-surface;
test).

**Rigid rotation** (wheels) enters as `χ u_s` with `u_s = ω a × (x − c)` about an arbitrary axis
`a`, pre-multiplied into the window arrays. Sign: with `a = +y` the point below the axle moves at
`−ωr` in x, so a wheel rolling on a road moving at `+U` has `ω = −U/r` (the 2D counter-clockwise
`+U/r` seen from the −y side); a test checks the solid velocity at the contact point.

### 6.2 Penalisation, forces and moments

Implicit Brinkman penalisation per stage, `u ← (u + kχu_s)/(1 + kχ)`, `k = b_s Δt/η` (2D §6.2).
The force on each body is the momentum the penalisation removes,
`F = −(h³/Δt) Σ w_b (⅙Δ₁ + ⅔Δ₂ + Δ₃)` (`w_b` the body's share of χ), with the moment lever arms
taken at each face's own position: u-faces carry Fx and contribute `(z − zr)Fx` to My and
`−(y − yr)Fx` to Mz, v-faces Fy (`−(z − zr)Fy` to Mx, `(x − xr)Fy` to Mz), w-faces Fz
(`(y − yr)Fz` to Mx, `−(x − xr)Fz` to My). Tests: a sphere has zero side force, lift and moments
to 10⁻³ of its drag; moments about two references differ exactly by `(r₁ − r₂) × F`
(10⁻¹⁰); several bodies sum; a half-domain run returns the full-domain coefficients to 10⁻⁹
(float64, a box on the symmetry plane).

**Control-volume cross-check.** For a steady flow the drag also follows from the momentum
balance on a box around the body,
`Fx = −∮ [p n_x + u (u·n) − ν(∂u/∂x_j + ∂u_j/∂x) n_j] dA`, every quantity taken at second order
from its own staggered position (`ns3d_cases.control_volume_drag`; on the symmetry plane the box
has no face — `v = 0` and `∂u/∂y = 0` there — and the result is doubled). On the sphere it
matches the penalisation impulse to 0.01–0.03 % at every Reynolds number and resolution of §10,
while its pressure, momentum-flux and viscous parts change by tens of per cent between a half-
and a full-domain box: the impulse *is* the force of the discrete solution.

### 6.3 Surface pressure and friction

`FlowSolver3D.surface(mean)` returns, per mesh vertex, cp, the wall-shear vector `tau` (scaled
as a skin-friction coefficient, `tau = (2/Re) ∂u_t/∂n`) and `cf = |tau|`, sampled exactly as in
2D: along the vertex normal (area-weighted) at probes `d₁ = 1.5 h` and `d₂ = 2.5 h` off the
wall, cp extrapolated linearly to the wall, `∂u_t/∂n` from the quadratic through the wall
velocity (zero, or `ω a × r` on a wheel) and the tangential velocity at both probes, every
quantity trilinear from its own staggered grid. Normals are oriented **per vertex** towards the
fluid (the side where the solid fraction at the first probe is lower), so any winding works;
on a sheet with fluid on both sides the mesh's own orientation is kept. In a half-domain run
the probes of vertices on the missing half are sampled at their mirror image (v reversed), so
the full mesh gets mirror-symmetric data (test: identical to 10⁻¹² at mirror vertices).

As in 2D the surface integral of `−cp n + tau` is a *consistency check*, not the force: the
probes under-resolve the wall gradient when the boundary layer spans few cells. On the sphere
it is 13 / 7.6 / 4.0 % low at Re 100 and 16 / 24 / 32 cells/D (§10, almost all of it friction).

## 7. Subgrid models (3D LES)

`turbulence = "smagorinsky" | "wale" | "vreman"` adds `∇·(2ν_t S)` with ν_t at cell centres
(with a ghost ring) from the velocity-gradient tensor — diagonal components from face
differences, off-diagonal ones the average of the four surrounding edge differences — diagonal
stresses at centres, shear stresses on edges with ν_t averaged from the four cells around each
edge: conservative and dissipative. Constants as in 2D (`C_s = 0.17`, `C_w = 0.5`, `c = 0.07`,
filter width h). The models are evaluated on the **traceless** gradient (`g₃₃ := −g₁₁ − g₂₂`):
identical in the interior, where the discrete divergence is zero to round-off after every
projection, and it leaves eight independent components, which lets the fused path evaluate a
model in one kernel. ν_t is zero inside solids (`× (1 − χ_c)`) and its maximum enters the viscous
time-step limit. WALE and Vreman vanish in pure shear (laminar boundary layers); Smagorinsky
does not (tests). Unlike 2D, these are genuine 3D LES models — but on the grids an 8 GB GPU
affords, wall-bounded flows at vehicle Reynolds numbers are wall-unresolved LES (§12).

## 8. Start-up perturbation

`perturb = True` adds a Lamb–Oseen vortex with its **axis along y** (peak swirl 5 % of U, core
radius 0.75 × the body height), confined spanwise by a Gaussian, centred on the bodies and
projected. It is divergence-free for any spanwise envelope (`∂u/∂x + ∂w/∂z = 0` and v = 0), it
breaks the up–down symmetry so an unsteady wake locks in promptly, and it is mirror-symmetric
about the y = y0 plane when that is a symmetry plane, so half- and full-domain runs evolve
identically. Steady validation cases run with `perturb = False`.

## 9. Performance

A 3D step written as plain torch operations is bound by memory traffic: each upwind3 flux costs
about nine elementwise passes over a field and a step evaluates 27 fluxes, plus padding,
diffusion, projection and bookkeeping. That reference core (`core.py`, kept as the
differentiable path and the CPU path) runs 256 × 128 × 128 at 10.7–13.1 steps/s on the GPU. There is
no Triton and no CUDA compiler on the development machine, so `torch.compile` and custom
extensions are out; PyTorch's **jiterator** (`torch.cuda.jiterator`) compiles small elementwise
CUDA kernels at run time through NVRTC, which ships with PyTorch, and accepts *strided views* as
inputs. A whole stencil — five shifted views of the padded velocity and the edge velocity sums —
is then one kernel that reads each array once (neighbouring reads hit the L2 cache) and writes
once: a 7-input stencil costs about one contiguous elementwise pass (0.16 ms at 4.2 M cells).
The fused path (`fused.py`) uses, per component and stage:

* `crossd0` / `crossd`: advection along each cross direction (both edges, precomputed edge
  velocity sums) plus the diffusion along that direction;
* `updself`: advection and diffusion along the component's own direction **fused with the
  Runge–Kutta update** `lerp(u_prev, u0, a) + bΔt L`, so the stage value never needs its own
  pass;
* `div` and `gsub` for the projection (ψ padded once with replicated or wrapped ghosts, which
  makes the gradient on Neumann boundary faces exactly zero without slice writes);
* for LES, 18 kernels: eight gradient components, the eddy viscosity of the model in one kernel
  (eight inputs thanks to the traceless gradient), edge and diagonal stresses, and their
  divergence accumulated into the tendencies.

jiterator takes at most eight tensor inputs, which is what shapes the split; boundary faces are
fixed after the fused update. The viscosity is a kernel launch argument, so a captured CUDA
graph bakes it in and `set_inflow` re-captures (the reference core reads the `nu` tensor). The
fused step equals the reference to 3 × 10⁻¹⁴ after 20 steps (moving road, WALE) and < 10⁻¹¹ in the tests in float64 for every boundary-condition and
subgrid combination (tests), and FlowSolver3D runs the same CUDA-graph machinery as 2D (blocks
of 10 steps captured after a side-stream warm-up, device-side CFL, history ring, live edits into
existing tensors); graph and eager runs are bit-identical (test). Any compile, capture or replay
failure falls back to the reference core or eager execution with a warning.

**Running means** use a weighted Welford update in float32 (mean and the sum of squared
deviations, which keeps the variance accurate where a sum of squares would cancel), half the
memory of float64 accumulators. They sample the flow once every 10 steps with the elapsed time
as the weight — a rectangle rule over ~0.1 L/U, far below any flow time scale — on the same
steps in graph and eager runs, and reading them folds in the time since the last sample. This
cut the averaging overhead from 20 % to ~4 % of a step.

Measured throughput (`bench_ns3d.py perf`, float32, best of 3 × 200 steps after a warm-up that
includes the NVRTC compile and the graph capture; sphere in a 12 × 6 × 6 box, shared GPU):

| configuration | steps/s | peak memory |
|---|---:|---:|
| 256×128×128 (4.2 M cells), default: upwind3, laminar, fused kernels, CUDA graphs | **44.9–49.2** (target ≥ 40) | 0.56 GB |
| same + running means | 43.0 | 0.84 GB |
| same, eager (no graphs) | 42.1 | 0.49 GB |
| same, reference core (no fused kernels) | 10.7–13.1 | 0.68 GB |
| central advection | 56.4 | 0.58 GB |
| upwind3 + WALE (fused LES; the reference LES ran 6.6) | 33.8 | 0.75 GB |
| 400×176×172 (12.1 M cells) + running means | 17.2 | **2.37 GB** (bar ≤ 6.5) |
| CPU, 96×48×48, 4 threads (reference core) | 4–26 | — |

The range on the default row spans three runs with different background load (the profile
of one step: 23 ms of GPU time at 4.2 M cells, dominated by the stencil kernels, the stage
update and the 3D transforms). Graphs gain little here (eager 42 vs 45–49): with fused kernels a
step launches only ~300 kernels, so the 3D step is bandwidth-bound, not launch-bound.

Memory: 2.37 GB peak at 12.1 M cells with running means (bar 6.5 GB), i.e. ~200 bytes
per cell in float32; `estimate_memory_bytes()` gives planners this figure (plus the mask window).
The CPU numbers vary by a factor of several with the load other processes put on the shared
machine (the same 96 × 48 × 48 case measured 4–26 steps/s); with four threads the reference core
is practical for Draft-size grids only, as DESIGN 7.1 says.

## 10. Verification and validation

### Verification (exact answers)

| check | result | bar |
|---|---|---|
| Poisson `max|Lψ − f| / max|f|`, float64, Neumann / periodic / mixed | 4.9 × 10⁻¹⁴ | round-off |
| same, float32 on the GPU, 256 × 128 × 128 | 2.2 × 10⁻⁵ | (float32 GEMMs) |
| projected divergence `max|∇·u| h/U`, float64 (sphere, moving road) | 7.3 × 10⁻¹⁶ | < 10⁻¹⁰ |
| projected divergence, float32 (GPU) | 4.0 × 10⁻⁷ | < 10⁻⁵ |
| Taylor–Green, central, observed order (three planes) | 1.97, 1.99, 2.00 | ≈ 2 |
| Taylor–Green energy decay rate / exact (N = 64) | 0.9992 | 1 |
| inviscid energy drift, central, 30 steps | < 10⁻⁴ | conservative |
| outlet mass balance | < 10⁻¹⁰ | exact |
| fused kernels vs reference core, float64, 20 steps | 3 × 10⁻¹⁴ | round-off |
| CUDA graphs vs eager, float32, 34 steps incl. running means | bit-identical | same kernels |
| half domain + symmetry plane vs full domain, float64 box, 10 steps | 10⁻⁹ (all six coefficients) | identical |
| moments about two references vs `(r₁ − r₂) × F` | 10⁻¹⁰ | exact |
| gradient d(mean CD)/d(vertex mode), autograd vs central FD, float64 | 9.1521 vs 9.1521, rel. error 2.9 × 10⁻¹⁰ (shared SDF: 2.8 × 10⁻¹⁰) | < 5 % |
| control-volume drag / impulse, steady sphere, all §10 runs | 1.0001–1.0003 | consistent |

### Validation: sphere

Sphere of diameter 1 (geodesic mesh, 5120 triangles, facets 0.04 % of D inside the sphere) at
the origin of the box `[−5, 10] × [−2.625, 2.625]²` D (blockage 2.85 %, slip side walls), half
domain with the symmetry plane at y = 0, upwind3, float32 on the GPU, `perturb = False`,
forces averaged over the last quarter of the run (Re ≤ 50: t = 30, Re 100: t = 40; all runs
steady to < 10⁻⁴ over the averaging window). `L/D` is the recirculation length behind the
sphere on the axis, `Cd_cv` the control-volume drag, `Cd_surf` the surface integral of the
probe data, `θ_sep` the separation angle from the front stagnation point (first sign change of
the meridional wall shear). References (unconfined): Clift, Grace & Weber (1978) correlation
`Cd = 24/Re (1 + 0.1935 Re^0.6305)`: 2.735 (Re 20), 1.574 (Re 50), 1.087 (Re 100);
Johnson & Patel (1999): Cd 1.57 / 1.087, L/D ≈ 0.4 (Re 50) and 0.88 (Re 100).

| Re | cells/D | grid (half domain) | steps | wall [s] | Cd | Cd_cv (vs Cd) | Cd_surf (vs Cd) | L/D | θ_sep | cp front / rear |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 16 | 240×42×84 | 2400 | 19 | 2.8631 | 2.8639 (+0.03 %) | 2.6659 (−6.9 %) | 0.07 | — | 1.282 / −0.437 |
| 20 | 24 | 360×63×126 | 5200 | 189 | 2.8663 | 2.8666 (+0.01 %) | 2.7569 (−3.8 %) | 0.05 | — | 1.341 / −0.426 |
| 50 | 16 | 240×42×84 | 2000 | 15 | 1.6579 | 1.6582 (+0.02 %) | 1.5328 (−7.5 %) | 0.427 | 143.2° | 1.114 / −0.273 |
| 50 | 24 | 360×63×126 | 3000 | 107 | 1.6508 | 1.6509 (+0.01 %) | 1.5801 (−4.3 %) | 0.420 | 140.6° | 1.132 / −0.259 |
| 100 | 16 | 240×42×84 | 2800 | 21 | 1.1719 | 1.1721 (+0.02 %) | 1.0193 (−13.0 %) | 0.923 | 129.9° | 1.078 / −0.226 |
| 100 | 24 | 360×63×126 | 4200 | 111 | **1.1479** | 1.1480 (+0.01 %) | 1.0602 (−7.6 %) | **0.894** | 128.2° | 1.077 / −0.215 |
| 100 | 32 | 480×84×168 | 5400 | 177 | 1.1376 | 1.1377 (+0.01 %) | 1.0924 (−4.0 %) | 0.880 | 128.0° | 1.075 / −0.209 |

(`bench_ns3d.py sphere --re 20 50 100 --cpd 16 24` and `--re 100 --cpd 32`; the wall times are
from a GPU running other heavy jobs at the same time — the 24-cell runs advanced at 28–38
steps/s against 72–85 measured on the same grid when the GPU was less loaded.)

Confinement study at 16 cells/D (`bench_ns3d.py domain`): the same sphere in wider boxes.

| Re | half-width (D) | blockage | Cd | L/D |
|---:|---:|---:|---:|---:|
| 20 | 2.625 | 2.85 % | 2.8631 | 0.069 |
| 20 | 4.0 | 1.23 % | 2.7789 | 0.069 |
| 20 | 5.25 | 0.71 % | 2.7610 | 0.069 |
| 100 | 2.625 | 2.85 % | 1.1719 | 0.923 |
| 100 | 4.0 | 1.23 % | 1.1453 | 0.930 |
| 100 | 5.25 | 0.71 % | 1.1401 | 0.932 |

Reading the tables:

* **Acceptance (DESIGN 7.12).** Re 100 at 24 cells/D and 2.85 % blockage: Cd **1.148** (band
  1.05–1.15) and L/D **0.894** (band 0.80–0.95); at 32 cells/D 1.138 and 0.880. Separation at
  128.0–128.2° from the front against 126.5° from the fit
  `θ = 180° − 42.5° (ln(Re/20))^0.483` to Taneda's measurements (Clift et al. 1978), and
  140.6° against 139.3° at Re 50. L/D 0.42 at Re 50 (Taneda, Johnson & Patel: about 0.4). At
  Re 20 a bubble of 0.05–0.07 D appears, at the threshold where separation begins (Re ≈ 20–24).
* **The control volume equals the impulse** to 0.01–0.03 % in every run, while its pressure,
  momentum-flux and viscous parts change by up to 30 % between boxes: the penalisation
  impulse is the force of the discrete solution, as in 2D.
* **Grid convergence.** At Re 20 and 50 the drag changes by +0.1 % and −0.4 % from 16 to 24
  cells/D: converged. At Re 100 it falls 1.1719 → 1.1479 → 1.1376 (16, 24, 32 cells/D; observed
  order ≈ 1.45, Richardson limit ≈ 1.118); L/D 0.923 → 0.894 → 0.880.
* **The remaining excess is confinement, and it is large at low Re.** Against the unconfined
  references the drag is ~5 % high at every Re. At Re 20, where the grid is converged, widening
  the box takes it from 2.863 to 2.779 and 2.761; a linear extrapolation in blockage to zero
  gives 2.737 against 2.735 (Clift). At Re 100 the box study at 16 cells/D extrapolates to 1.133,
  i.e. the 2.85 % box adds 3.4 %; applied to the Richardson limit that gives ≈ 1.081 for an
  unconfined, grid-converged sphere against 1.087 (−0.5 %). (These two extrapolations multiply
  separately measured factors: an estimate, not a measurement.) Slip side walls 2.6 D from the
  axis confine a low-Re flow much more than the classical high-Re blockage estimates suggest,
  because its velocity disturbance decays slowly; planners should prefer wide domains at low Re.
* **Surface data.** The probe-based surface integral is 13 / 7.6 / 4.0 % below the force at
  16 / 24 / 32 cells/D at Re 100 (friction under-resolved, as in 2D) and converges towards it.
  The front stagnation cp is above 1 at moderate Re, as it must be.

### Symmetry plane

Sphere, Re 100, 16 cells/D (`bench_ns3d.py symmetry`): the half domain (240×42×84, symmetry
plane at y = 0) and the full domain (240×84×84, slip walls on both sides) give **identical**
results to the printed digits — Cd 1.1719, L/D 0.9233, Cd_surf 1.0193, separation 129.86° —
in 6.5 s against 14.3 s (bar: within 1 %). The perturbation is symmetric about the plane, so
the full-domain flow stays mirror-symmetric to round-off and the two runs solve the same
discrete problem. A fast test checks the same identity to 10⁻⁹ in float64 for a box whose
lift and pitching moment are non-zero, and that the half-domain run reports CS = C_roll = Cn
= 0.

### Moving road, rolling wheels, LES at Re 10⁶

`moving_road_config(64)`: a box "car" (length 1, width 0.4, height 0.3) 0.08 above a road
moving at U, two wheels on the y > 0 side (radius 0.07, width 0.06) touching the road and rolling
with it (ω = −U/r about +y), half domain (256×51×58 cells, symmetry plane), upwind3 + WALE,
Re = 10⁶, t = 0 … 6.6 (`bench_ns3d.py ground`, slow test
`test_moving_road_wheels_wale_re1e6_is_stable`). The run completes without any instability:
3000 steps at the advective limit (Δt ≈ 2.2 × 10⁻³), projected divergence 5.8 × 10⁻⁷ (×h/U),
ν_t up to 1.4 × 10⁻³ (≈ 0.09 U h: active at the grid scale only), peak speed 1.60 U, mean
CD 1.30 (frontal area 0.12, including the wheels' mirror images) and CL 0.05 over the second
half. Unlike the 2D side profile (theory_ns2d.md §11), the wheels cannot seal the underbody:
air passes beside them, and the lift is no longer an artefact of the geometry. This is a
robustness test, not a validation: the boundary layers are not resolved at Re 10⁶.

### Sphere at Re 300 (stretch case)

Johnson & Patel (1999) find planar-symmetric hairpin shedding at Re 300 with mean Cd 0.656,
mean lift 0.069 and St 0.137. Half domain (the symmetry plane is the plane of the shedding,
which the y-axis start-up vortex selects), perturbation on, the 2.85 % box of the steady cases,
t = 0 … 150, averages over t ≥ 80 (`bench_ns3d.py unsteady`):

| cells/D | grid (half) | steps | wall [s] | mean Cd | Cd amplitude | mean CL | CL amplitude | St |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 360×63×126 | 16 500 | 194 | 0.730 (+11.3 %) | 0.0075 | 0.0715 | 0.032 | 0.131 (−4.3 %) |
| 32 | 480×84×168 | 22 000 | 778 | 0.710 (+8.2 %) | 0.0048 | 0.0718 | 0.022 | 0.135 (−1.5 %) |

The solver reproduces the regime — periodic shedding in the symmetry plane with a steady mean
lift of the published size (0.072 vs 0.069) — and St is within 1.5 % at 32 cells/D. The drag
converges from above (−2.7 % from 24 to 32 cells/D); about 3 % of the remaining excess is the
confinement measured at Re 100 (§10), the rest is resolution: at Re 300 the boundary layer is
only ~1.5 (24) to 2 (32) cells thick. Read this case as a stretch: Re ≳ 300 needs ≥ 32 cells
across a bluff body for drag within ~5 %. The slow test checks the 24-cell run against bands
of 15 % (Cd), 8 % (St) and 20 % (mean lift). Wall times: shared GPU, 28 steps/s on the 32-cell
grid where an idle GPU would run ~40.

## 11. Differentiability

`step3d(state, geom, ops, dt)` is pure torch with static control flow; everything between the
mesh vertices and the force coefficients is differentiable: the signed distance (recomputed from
the nearest triangle with autograd, in both the shared and the private implementation) → C¹ mask
→ penalisation → projection (matrix products) → force impulse. `rollout3d` advances n steps with
`torch.utils.checkpoint` over chunks of k steps and a fixed Δt. Gradient check (float64, CPU,
20 × 12 × 12 cells, an 80-triangle sphere at 5 cells/D, Re 20, 10 steps, J = mean CD, a smooth
vertex-displacement mode — stretch along x and half that along z): autograd 9.1521 against 9.1521 from central differences (ε = 10⁻⁵), relative error 2.9 × 10⁻¹⁰ with the private exact distance and 2.8 × 10⁻¹⁰ with the shared voxeliser. Gradients
w.r.t. the inflow and the initial state are non-zero and finite, and LES steps are
differentiable including w.r.t. the cell-centre solid fraction (tests). The fused kernels have
no autograd; `FlowSolver3D` uses them only under `torch.no_grad`.

The distance is differentiable *through* the surface as well: a grid point exactly on a face
(or within the inside test's 0.002-cell ray jitter of it) takes the face's signed plane
distance, whose sign comes from the face's outward side, instead of `±√(d²)` — which has no
derivative at d = 0 and whose sign the jittered parity can get wrong there — and a point
exactly as far from two faces (a medial plane through a row of centres, a box edge) takes the
mean of their derivatives, the one central differences measure for a two-way tie. Before,
an axis-aligned box with faces on grid planes gave a discontinuous objective and gradient
errors of 25–30 % (median over modes) in a direct check of the masks; now 10⁻⁴ or better
(tests/test_geometry3d_voxel.py, and docs/theory_sensitivity.md §9.4 for the adjoint).

## 12. Limitations — what to trust

| regime | what to trust | evidence |
|---|---|---|
| Re ≤ ~300, ≥ 24 cells across the body | forces within ~1–2 % of the grid limit plus the domain's confinement, wake length, separation angle; cp; cf a lower bound | sphere §10 |
| Re ≤ ~300, 16 cells across | Cd within ~3 % of the 24-cell value at Re 100 (converged at Re ≤ 50); surface integral 7–13 % low | sphere rows |
| 300 < Re ≲ 10⁴ | laminar/transitional 3D physics; boundary layers need cells ∝ √Re across the body, so typical grids under-resolve them | — |
| Re ≳ 10⁵ (vehicles, riders, drones) | wall-unresolved LES: flow topology, separation regions, *differences* between designs; absolute drag uncertain (no wall model, no transition model) | moving-road run at Re 10⁶ is stable |

* **Resolution.** Bodies need ≥ 24 cells across for force accuracy of a few per cent; thin
  features need ≥ ~3 cells across to be solid. Open sheets and sub-cell features are thickened
  to one cell by the voxeliser, which a stiff penalisation turns into a leaky wall — model
  spoilers and wings as closed solids a few cells thick.
* **Confinement.** At low Re a 2.85 %-blockage box raises the drag by ~3–5 % (§10); the planner's
  domain sizes decide this error. Slip side walls and a slip top are not a free stream.
* **Wall position.** The diffuse wall sits within ~0.1 h of the surface on average (the 2D
  calibration, which the sphere results confirm), with sub-cell scatter on grid-aligned flat
  walls (±0.2 h, theory_ns2d.md §6.3).
* **Surface cp and cf.** Extrapolated from probes 1.5 and 2.5 cells off the wall: cp usable,
  cf a lower bound wherever the boundary layer spans fewer than ~4 cells; probes that enter
  another body or leave the domain (wheel–road contact) are undefined. Reported forces always
  come from the penalisation impulse.
* **Uniform grid** (the transforms need it): no refinement near the body; memory ~200 bytes per
  cell in float32 (12 M cells in 2.4 GB), throughput ~45 steps/s at 4.2 M cells on the
  development GPU.
* **Fused kernels** need CUDA and NVRTC (part of PyTorch's CUDA build); without them the reference
  core runs ~3.5× slower. A Reynolds-number change re-captures the CUDA graphs.
* **Compressibility**: none (Mach → 0).
* **Placed physics** (heat transfer, buoyancy, fans, jets, emitters, painting): see the
  limitations at the end of §18.

## 13. References

* Clift, Grace & Weber (1978), *Bubbles, Drops and Particles*, Academic Press.
* Dhole, Chhabra & Eswaran (2006), A numerical study on the forced convection heat transfer from
  an isothermal and isoflux sphere in the steady symmetric flow regime, Int. J. Heat Mass Transf.
  49, 984 (cited in §15.7; its tables were not accessible).
* Hunt, Wray & Moin (1988), Eddies, streams and convergence zones in turbulent flows, CTR-S88.
* Incropera, DeWitt, Bergman & Lavine, *Fundamentals of Heat and Mass Transfer*, Wiley:
  laminar flat plate at uniform heat flux, Nu_x = 0.453 Re_x^½ Pr^⅓ (eq. 7.45 in the 6th ed.; §15.7).
* Johnson & Patel (1999), Flow past a sphere up to a Reynolds number of 300, JFM 378.
* Maskell (1963), A theory of the blockage effects on bluff bodies and stalled wings in a closed
  wind tunnel, ARC R&M 3400.
* Nicoud & Ducros (1999), Subgrid-scale stress modelling based on the square of the velocity
  gradient tensor, Flow Turb. Combust. 62.
* Taneda (1956), Experimental investigation of the wake behind a sphere at low Reynolds numbers,
  J. Phys. Soc. Japan 11.
* van Leer (1974), Towards the ultimate conservative difference scheme II. Monotonicity and
  conservation combined in a second-order scheme, J. Comput. Phys. 14 (the limiter of §14).
* van Oosterom & Strackee (1983), The solid angle of a plane triangle, IEEE Trans. Biomed. Eng. 30.
* Vreman (2004), An eddy-viscosity subgrid-scale model for turbulent shear flow, Phys. Fluids 16.
* and the 2D references in [theory_ns2d.md](theory_ns2d.md) §14 (Shu & Osher, Harlow & Welch,
  Morinishi et al., Angot et al., Makhoul, Smagorinsky/Lilly).

## 14. Passive smoke: flow visualisation in the volume

Users asked to see the flow *around* the body in 3D -- smoke and streamlines in space, not only
colour on the surface (DESIGN.md 7.13). Streamlines need nothing from the solver beyond the
velocity it already stores; smoke does: the solver transports a passive scalar `c` (0 = clean air,
1 = undiluted smoke) released by emitters upstream of the body, the 2D `dye` carried to 3D
(`solvers/ns3d/dye.py`; `FlowSolver3D.field("dye")`).

**Equation and placement.** `dc/dt + div(u c) = 0`, no physical diffusion (smoke particles
follow the flow; the numerics add the only spreading). `c` is cell-centred on the MAC grid, so
the face velocities the solver already holds are exactly the advecting velocities of a
finite-volume flux: no interpolation. With the end-of-step velocity, which is discretely
divergence-free after the projection, a uniform `c` stays uniform to round-off and the total
smoke is conserved exactly away from emitters, solids and the outlet.

**Scheme.** MUSCL reconstruction with the van Leer limiter (the harmonic mean of the one-sided
differences, zero at extrema) and one forward Euler step per solver step; the face value is
taken from the upwind side. A forward Euler step with a limited slope is total-variation
diminishing when the sum of the face Courant numbers is at most `1 / (1 + psi_max / 2)`, which is
1/2 for van Leer (`psi_max = 2`). The solver's time step is `cfl h / (max|u| + max|v| + max|w|)`
with `cfl = 0.5` (§5), exactly that bound, so the scheme creates no new extrema; the update is
then clipped to [0, 1] (an unsplit TVD bound is sharp; round-off excursions are removed, not
allowed to grow) and values below 10⁻⁸ are set to exactly zero -- invisible, and on x86 CPUs
the limiter's decaying tails would otherwise run into subnormal floats (~100× slower). First
order in time is a deliberate choice: smoke is a picture, not a measurement, and one update per
step keeps the cost to three flux passes and one update pass.

**Boundaries and solids.** Clean air enters at the inlet (zero ghost cells); the outlet is
zero-gradient; walls, the road and the symmetry plane mirror the field (their zero normal
velocity makes their flux zero anyway; the mirror only keeps the limiter's slopes symmetric, so
a half-domain run gives the exact mirror half of the full run); periodic axes wrap. Cells whose
solid fraction exceeds one half are set to zero after every step, so the field is zero inside
bodies. The Brinkman velocity there is ~0, so almost no smoke flows in: this removes what
numerical spreading puts into the diffuse interface rather than acting as a sink for smoke that
hugs the surface.

**Emitters** (`DyeRake`: "rake" a row of nozzles across the span, "grid" a span × height grid,
"plane" an emitting band). After each step the smoke inside a nozzle is relaxed to 1,
`c += s (1 - c)`, with a nozzle weight `s` that is 1 inside the radius and falls linearly to 0
over one cell (anti-aliased: a nozzle between cell centres still emits, and moving it by a
fraction of a cell moves its stream). A nozzle is two cells deep along x. Optional on/off pulses
cut the streams into time lines; the pulse switch is evaluated on the device from the solver
time and the pulse parameters live in a tensor, so neither ever leaves a CUDA graph. The default
radius is 1.25 cells (a stream ~2.5 cells across, the thinnest the limiter carries over a body
length at Draft resolution) and nozzles are about six radii apart (at least 4.8 once the count is rounded), so the streams stay
distinct. The default rake (the GUI's, `viz3d.flow.resolve_rake`, and the solver's own
`default_dye_rake`) stands 0.4 of the body's smaller dimension upstream of the nose, spans the
body plus 20 % on each side and sits at 40 % of its height, where the streams split at the nose
-- over the top, round the sides, under the body -- and roll up in the wake.

**Fast path.** On CUDA the update runs as fused jiterator kernels (`fused.dye_flux`, one per
direction, reading the four-cell stencil as strided views of the padded field; `fused.dye_update`)
inside the same CUDA graphs as the flow step; the smoke buffer is part of the graphs' static
state. Before using them the solver compiles them and compares one step with the reference
scheme (`check_dye_kernels`); a failure falls back to the plain-torch reference with
`dye_error` set. The smoke is solver state, not part of the differentiable `State3D`: it never
feeds back into the flow, so shape sensitivities never need it.

**Verification** (`tests/test_ns3d_dye.py`):

| check | result |
|---|---|
| total smoke over 60 steps, periodic flow, no emitters (float64) | conserved to < 10⁻¹³ (relative) |
| rough field in [0.1, 0.8], 80 steps at CFL 0.5 | no value leaves [0.1, 0.8] (± 10⁻¹²): no new extrema |
| uniform field, 10 steps | stays uniform to < 10⁻¹³ |
| half domain with symmetry plane vs full domain (40 steps, sphere) | identical to < 10⁻⁹ |
| fused CUDA kernels vs reference, float64, 4 boundary-condition cases | < 10⁻¹² |
| GPU solver (fused, graphs) vs CPU solver, float64, 40 steps | < 10⁻⁹; graphs equal eager bit for bit |
| solid cells after any step | exactly 0 |
| moving the emitters within their window | captured CUDA graphs kept |

**Cost.** At 256 × 128 × 128 (4.2 M cells) on the development GPU one smoke update takes 1.6 ms
against 25.9 ms for a whole step including it (6 %; timed eagerly, so including ~17 kernel
launches that a graph replays more cheaply; the GPU was shared). Whole-step throughput with and
without smoke: 38.7 vs 39.3 steps/s. Memory: one more cell-centred field (4 bytes per cell) plus
transient flux arrays that reuse the step's freed temporaries; the measured peak rose from 0.52
to 0.56 GB at 4.2 M cells, inside the margin of `estimate_memory_bytes`.

**Limitations.** Numerical spreading widens a stream by roughly a cell per body length at Draft
grids, so far-wake smoke is a haze rather than filaments; the concentration is a picture of
where fluid released at the rake went, not a measured quantity. Smoke from a rake at a fixed
position shows streaklines, which equal streamlines only in steady flow -- in a shedding wake the
smoke shows the vortex street, the instantaneous streamlines do not.

### 14.1 From the solver to the 3D view

* **Live** (`simulation/live3d.py`): with `ViewSettings.view3d` every frame carries the most
  recent down-sampled volume (`LiveFrame.volume`: float16 smoke, and u, v, w when 3D streamlines
  are on, block-averaged on the GPU to at most 128³ cells and 20 MB) and streamlines traced on
  the solver thread from the rake (`LiveFrame.streamlines3d`), so the GUI thread only uploads
  them. Volumes are taken at most every 0.1 s and at most 10 % of the wall time, streamlines
  about once a second and at most 10 %; consecutive frames share a volume (`volume_serial`).
* **Results** (`simulation/runner3d.py`): the result keeps an instantaneous snapshot at the end
  of the run on the stored volume's block grid -- the smoke and the velocity of that moment
  (`CFD3DResult.extra_volumes`, label "instant"; an average of smoke is a blur, not a picture) --
  next to the mean velocity, so a result shows the smoke, the mean-flow streamlines and the
  instantaneous ones (which, in a shedding wake, look like the smoke and not like the mean).
* **Rendering** (`viz3d.Scene.add_smoke`): GPU ray casting (`vtkSmartVolumeMapper`) of the smoke
  through a fixed transfer function -- transparent below c = 0.02, then a convex opacity rise per
  voxel (the faint haze stays clear and the cores carry the picture), softly shaded by the
  scene's lights, near-white on the dark theme and dark grey on the light one; fixed sample
  distance and no jittering, so a developing run does not flicker. Streamlines are traced with
  `vtkStreamTracer` (classical RK4, one-cell steps: the velocity is trilinear between samples one
  cell apart, so smaller steps resolve nothing the data holds; adaptive RK45 took twice as long
  for the same lines, end points within a cell) through the velocity with solid cells zeroed; a
  symmetry-plane volume is traced in its own half (no streamline crosses a plane with zero normal
  velocity) and the lines of mirrored seeds are reflected. Tubes are 0.18 % of the lines' extent
  in radius, at most a fifth of the seed spacing, so neighbouring lines never merge.
* **Vortex cores** (`viz3d.Scene.add_isosurface`, field Q): the penalised wall of §6 shears the
  flow over its last three cells or so, which gives a thin layer of large Q along every surface.
  It is not a vortex and would wrap the body in a skin, so the iso-surface is cut 3.5 cells from
  the body (`WALL_CLEARANCE_CELLS`). The cut follows a smooth distance to the body triangles (a
  k-d tree of points half a cell apart; `vtkClipPolyData` interpolates it along the edges), never
  a mask of whole voxels: zeroing Q in a dilated band of cells leaves a saw-tooth rim of voxel
  steps where the skin meets the real vortices. On the Draft sphere, Q above the default level
  fills 40 % of the cells 1.5–2 cells from the wall, 17 % at 3–4 and 7 % at 6–10 cells (the
  separated shear layer); a 3.0-cell cut still leaves flaps of the skin. The trim costs 22 ms for
  the sphere and 86 ms for the 13.7 k-triangle car.

Measured on the development machine (RTX 4060 Ti, 1600 × 1000 px, Draft runs of a sphere and of a
car-like body, half domain):

| step | time |
|---|---|
| smoke update + render, 128³ voxels | 33–40 ms (medians of two sessions) |
| smoke update + render, live car volume (84 × 76 × 304, mirrored to 3.9 M voxels) | 43–47 ms |
| trace 300 seeds through the car's 1.94 M-cell mean volume (plane rake, 41 k points) | 57–59 ms |
| `Scene.add_streamlines`, same 300 seeds, traced and drawn as tubes | 94 ms |
| in-place update with 300 lines traced on the solver thread, + render | 49 ms |
| live: volume extraction on the solver thread (car / sphere) | 7.0 / 4.0 ms each, 3.4 / 2.1 % of the run |
| live: 30 streamlines traced on the solver thread (car / sphere) | 79 / 48 ms each, ≤ 7 % of the run |
| live frame payload (car, dye + u, v, w + mask, float16) | 18.5 MB |

---

## 15. Heat transfer and buoyancy (DESIGN.md 10.2)

Everything in §15–18 is off by default. A configuration without heat transfer, fans, ports,
diagnostics or the PID time step builds no physics object (`SolverConfig3D.has_placed_physics`)
and runs the base step: the default path is bit-identical to the solver without placed physics (laminar and WALE,
eager and CUDA graphs, 57 steps of a sphere with smoke and running means: largest difference of
every field, the history and the means exactly 0) and launches the same CUDA kernels (2176 per 5
steps, the same multiset of kernel names under `torch.profiler`), so it costs exactly what it
cost before. Emitters (§16) run in the smoke step, which is part of the default path.

### 15.1 Equations and units

The temperature is carried as θ = (T − T_amb) / ΔT_ref, cell-centred like the pressure:

    ∂θ/∂t + ∇·(u θ) = ∇·((1/Pe + ν_t/Pr_t) ∇θ) + q,      Pe = Re Pr,  Pr_t = 0.85
    ∂u/∂t + …       = … + Ri θ e_z                         (gravity along −z, Boussinesq)

with Ri = g β ΔT_ref L / U². Heat quantities cross the solver's interface in **conduction units**
k ΔT_ref L (a heat rate Q means Q k ΔT_ref L watts), so a body's mean Nusselt number on the
reference length is Nu = Q / (A θ_b) with A its area in L². Natural convection in a closed box
uses the free-fall scaling U = √(gβΔT L): Re = √(Ra/Pr), Pe = √(Ra Pr), Ri = 1
(`thermal3d_cases.cavity_config`). The planner chooses ΔT_ref and the scaling and converts the
SI model types (DESIGN.md 10.2).

### 15.2 Discretisation

* **Coupling.** θ is advanced inside the same SSP-RK3 stages as the momentum, with the same
  (a, b) pairs and the stage's discretely divergence-free face velocities, so transport and
  buoyancy are coupled at the scheme's order and a uniform θ stays uniform to round-off.
* **Fluxes.** Advection is the smoke's MUSCL / van Leer face value (§14: bounded and
  second order where smooth) times the face velocity; diffusion is the 7-point flux form
  κ (θ_c − θ_b)/h. The update is the divergence of face fluxes, so the scheme is conservative:
  the heat that enters through walls, bodies, sources and jets is accounted for exactly (§15.5).
  With an LES model the face diffusivity is κ + (ν_t,L + ν_t,R)/(2 Pr_t), molecular on
  non-periodic boundary faces (the wall rules use the molecular value).
* **Time step: a combined bound, not the minimum of two.** A forward-Euler stage of the MUSCL /
  van Leer advection is TVD for a Courant sum C ≤ ½ (§14) and the 7-point diffusion is positive
  for 6D ≤ 1 (D = κ_eff dt/h²); together the stage is a convex combination of neighbouring values
  — bounded, no new extrema — only when C/½ + 6D ≤ 1, and SSP-RK3 keeps the property because
  its stages are convex combinations of such steps. The heat equation therefore limits the step to
  dt ≤ 1 / (Σmax|u_i| / (½ h) + κ_eff / (h²/6)) on top of the momentum's own limits (§5). The
  first implementation took min(CFL limit, diffusion limit) instead, which lets C = 0.49 and
  6D = 1 occur together: a heated sphere at 24 cells per diameter (Re 50, Pr 0.71, where the two
  limits coincide) grew an oscillation ×1.13 per step from step ~120 until θ overflowed, while the
  drag was unaffected (Ri = 0). The regression test is the worst case — uniform flow with
  κ = hU/3, so both limits coincide exactly, and a checkerboard θ: the old rule amplified it 10⁷
  in 25 steps; the combined rule keeps θ inside its initial range to 10⁻¹² (a discrete maximum
  principle). The price is a smaller step where advection and diffusion are both near their limits
  (Pe ~ 1 per cell): at most half the minimum of the two — 55 % more steps for the heated sphere
  at Re 50 and 16 cells per diameter (4250 against 2750) — and nothing where one effect
  dominates. The 2D solver arrived at the same bound independently (theory_ns2d.md; a heated
  cylinder reached 10³⁰ there).
* **From rest.** In still air the CFL rule sees no velocity, so two more limits apply (also as in
  2D): the momentum sources' speed scale (jet speeds, a regulated fan's target, √(T/A) or √Δp of a
  thrust or pressure-jump fan, times 1 + |swirl|) enters the CFL bound, and a parcel at rest,
  which accelerates at Ri |θ|, may move at most `cfl` cells in one step: dt ≤ cfl √(h / (Ri
  max|θ|)). Neither binds in any validation case of §15.6 or §17 (checked); they matter for fans,
  jets and hot bodies started in still air at high Re (test: first step within both).
* **Domain sides** enter through two ghost layers: "temperature" 2θ_w − θ (mirror about the
  wall value), "adiabatic" and the symmetry plane the even mirror, "heat_flux" the mirror plus
  G h (first layer) and 3 G h (second), so the face gradient is exactly −∂θ/∂n = G, a tunnel
  inlet carries θ_inlet and the outlet is zero-gradient; periodic axes wrap. A thermal
  condition set on a side that is an inlet, outlet or symmetry plane is ignored with a note.
* **Fused path.** On CUDA the face fluxes and the stage update run as jiterator kernels
  (`fused.th_flux_s` with κ/h as a launch value, `th_flux_t` with an LES diffusivity tensor,
  `th_upd`), inside the step's CUDA graphs. Before use they are compiled and compared with the
  reference arithmetic (`check_theta_kernels`); a failure falls back to plain torch for θ only,
  with `theta_error` set.

### 15.3 Bodies, and why buoyancy is added after the penalisation

* **"temperature"** bodies are penalised implicitly towards their temperature with the
  momentum's mask and calibrated wall offset: θ ← (θ + k χ θ_b)/(1 + k χ), k = b dt/η_T,
  η_T = min(0.05 h, h² Pe/60) — the momentum rule (theory_ns2d.md §6.3) with Pe for Re. The
  heat the penalisation adds, weighted per stage like the momentum increments (1/6, 2/3, 1), is
  the body's heat-transfer rate, exactly as its momentum impulse is its force; with several
  bodies each one's share is its fraction of the overlap.
* **"adiabatic"** bodies block conduction: the diffusive flux through a face is multiplied by
  (1 − χ_face). Inside an insulating slab that spans the box, the flux through its central face
  is 0 to 10⁻¹⁵ (test). The interior of an adiabatic body is then isolated; its temperature is
  whatever it was.
* **"heat_flux"** bodies release G = q″L/(kΔT) through a skin: the derivative of the unshifted
  smooth mask (a cosine delta one mask width across, centred on the surface), normalised over a
  grid holding the whole body so that the released heat is exactly G·A (cut by the domain: the
  in-domain part). Their **wall temperature** is measured in that skin — the release-weighted
  mean of θ over the cells the heat goes into (`ThermalGeometry.wall_theta`), recorded every step
  as `theta_wall_<i>` — and their Nusselt number is Q/(A θ_w) from it (§15.7 says why the earlier
  probe extrapolation was wrong).
* **Buoyancy.** Ri θ is added to each stage value *after* the penalisation and *everywhere*,
  also inside solids; the fictitious fluid inside a solid then feels buoyancy too, and the
  momentum balance of the body's volume gives the force on the body as the penalisation impulse
  minus Ri ∫θ dV over the body (with a volume-accurate indicator, below). This is a deliberate
  change from the first implementation, which switched buoyancy off inside the diffuse solid and
  applied it before the penalisation. Measured on a fixed insulating sphere in a linearly
  stratified, non-conducting fluid (the exact answer: the fluid at rest, F_z = −Ri θ_c V),
  that formulation had no discrete rest state — spurious currents of 1–2 % of the buoyancy
  velocity — and gave 62–79 % of the Archimedes force at 6.4–9.6 cells per diameter, because
  (1 − χ) Ri θ is not a gradient across the interface and the penalisation acts on the predictor
  before the projection removes the gradient part. Added after the penalisation (like the
  pressure, which balances most of it), a linear stratification is an exact discrete
  equilibrium. The force correction uses the smoothed indicator **without** the calibrated wall
  offset (`ThermalGeometry.vol_w`): the momentum mask sits ~0.25 h inside the surface and
  under-counts the volume by ~20 % at 6 cells per diameter, while the unshifted one integrates
  to the true volume at second order. The fluid away from solids sees the same arithmetic as
  before: the cavity results of §15.6, re-run at two grids per Rayleigh number after this change
  and the time-step change of §15.2, agree with the earlier runs to all printed digits.

**Heat sources and the step from rest (review round 5 before the release).** The step rule bounds the heat
equation by its combined advection–diffusion limit and the buoyant parcel by
`cfl √(h / (Ri |θ|max))`; in fluid at rest with θ = 0 neither applies, so a box warmed only by
heaters took its first step at the diffusion limit. The planned Draft room
(`fan_room_case(dimension=3)` without its fan, one 300 W box heater, 52 × 40 × 32, Re 2.4 × 10⁴,
Ri 0.2) took a 68.7 L/U first step, reached θ 6.2 × 10⁵ and |u| 1.6 × 10⁶, and was NaN on step 2.
`Physics3D.dt_rule` now also takes, with `S = κ max|q|` the heating rate of the strongest cell
(heaters and heat-flux skins summed where they overlap, `ThermalOps.q_max`, refreshed in place
on a live Reynolds-number change), `Δt ≤ 1/S` and `Δt ≤ (6 cfl h / (|Ri| S))^(1/3)` — a parcel
heated from rest at rate S moves Ri S t³/6, at most cfl cells — the bounds the 2D solver got in
review round 3.4. The same room now takes 1.03 L/U (= 1/S) first, θ max peaks at 2.31 on step 8
in the heater's cells and falls as the plume forms, and a 20 L/U run through the public API
completes (2.8 s on the GPU, shared) with the heater's measured heat 300 W.

Archimedes check (`thermal3d_cases.run_archimedes`: sphere d = 0.4 at z = 0.62 in a closed unit
box, θ = z − ½ between isothermal top and bottom walls, Ri = 5, Pr = 10⁶ so the stratification
is an equilibrium, insulating sphere, float64, 100 steps):

| n³ | cells per diameter | F_z / (−Ri θ_c V) − 1 | max speed |
|---|---|---|---|
| 16 | 6.4 | +5.65 % | 4.0e-08 |
| 24 | 9.6 | +2.39 % | 1.4e-08 |
| 32 | 12.8 | +1.42 % | 6.7e-09 |

Observed order of the force error: 2.13, 1.80 (second order, the volume integral of the smoothed indicator).

The residual speed is the conduction of Pr = 10⁶ (κ = 2·10⁻⁸) acting on the insulated body.

### 15.4 Heat sources, heat rates and the energy balance

Heat sources ("box", "sphere") are anti-aliased footprints normalised over the whole shape, so
the released power is exact (a source cut by the domain releases its in-domain share, with a
note when it misses the domain). `FlowSolver3D.heat_rates()` reports each body's rate and
Nusselt number, `heat_transfer(mean=…)` the 2D solver's structure (bodies with θ_wall —
the held value, the measured skin temperature of a heat-flux body (§15.7), or for an adiabatic
body the area-weighted surface temperature extrapolated from the flow — and Nu =
Q/(A(θ_wall − θ_ref)), walls, Pe), `physics_history()` the per-step `heat_rate_<i>`,
`theta_wall_<i>` and `nu` = ΣQ / Σ A θ_w over the heated bodies (held and heat-flux), `wall_heat_rates()` each side's heat from the scheme's own
ghost-cell fluxes (at a tunnel's inlet and outlet including the advected heat), and
`heat_balance()` every term the analysis layer reads (`analysis.physics.HEAT_IN_TERMS`): bodies,
walls, sources, jets, inlet, outlet, vents, the storage rate from the last diagnostics row and the
residual. `surface()` adds `theta_wall` and `heat_flux` per vertex (local Nu × θ_wall, from the
quadratic through the wall value and two probes, as cf).

### 15.5 Verification: exact discrete identities

`tests/test_ns3d_physics.py`, float64 on the CPU unless noted:

| check | result |
|---|---|
| conduction between two isothermal walls, from the linear profile, 20 steps | profile to < 10⁻¹³; wall heat = k ΔT A / L (Nu = 1) to 10⁻¹² |
| the same from a cold start | error decreasing monotonically |
| two heaters (+3, −1) in a closed adiabatic box | stored heat grows at exactly 2.0 (10⁻¹⁰); diagnostics storage = 2.0; balance residual < 10⁻⁹ |
| heat-flux wall (G = 0.7) and heat-flux body (G = 2) | storage = G_wall A_wall + G_body A_body to 10⁻¹⁰; body rate = G A to 10⁻¹² |
| held-temperature body in a closed box | reported rate (penalisation) = storage rate to 10⁻⁹ every 10 steps |
| insulating slab spanning the box | conductive flux through its centre face < 10⁻¹⁵ |
| heat-flux body's wall temperature (§15.7) in a uniform field θ = 0.37 | read back to 10⁻¹²; history `theta_wall_0`, `heat_transfer`, `heat_rates` and the mixed-body `nu` all equal Q/(A θ_w) from it to 10⁻¹²; the skin follows a live `set_bodies` move |
| planning helpers (§15.8) | `thermal_dt_estimate` equals `ThermalOps.dt_limit` to 10⁻¹² for any speed sum |
| linear stratification (Ri = 5), no body, 100 steps | max speed < 10⁻¹², θ unchanged to 10⁻¹² |
| insulated sphere in the stratification, 16³ | at rest (< 10⁻⁶), Archimedes force within 7 % (§15.3 table: 5.7 %) |
| uniform inflow at θ_inlet = 1 | θ = 1 everywhere to 10⁻⁹; symmetry plane overrides a wall value with a note |
| fused kernels vs reference core, every feature on (heated body, Ri 0.3, heater, swirling PI fan, jet with heat and smoke, vent, 3 smoke channels, WALE), float64, 12 steps | ≤ 10⁻¹² (fields, heat rate, thrust); pressure ≤ 10⁻¹⁰ |
| CPU vs CUDA reference, same case | ≤ 10⁻¹² |
| CUDA graphs vs eager, same case + particles, float32, 34 steps | bit for bit |
| half domain (symmetry plane) vs full domain: heated sphere, smoke rake and pulsed disc straddling the plane, Ri 0.2 | smoke, θ, u and heat rate equal to < 10⁻¹⁰ |

### 15.6 Validation

**Differentially heated cubic cavity** (air, Pr = 0.71, hot and cold opposite walls at θ = ±½,
the other four adiabatic, all no-slip, Ra = gβΔT L³/(νκ); `thermal3d_cases.run_cavity`, float32 on
the GPU, started from the conduction profile and run until the hot-wall Nu changes by less than
2·10⁻⁵ over 20 free-fall times). Nu is the scheme's own wall flux; the hot and cold walls agree
to < 10⁻⁷ (relative) at steady state and the total flux through the mid-plane x = ½ to < 3·10⁻⁵
(the remaining drift); the RMS continuity residual is ≤ 2.2·10⁻⁶ (float32).

| Ra | n³ | mean Nu | vs Tric | Nu, plane y = ½ | u_max (z) | w_max (x) | steps | wall s |
|---|---|---|---|---|---|---|---|---|
| 10⁴ | 16 | 2.1470 | +4.52 % | 2.3503 | 0.1989 (0.826) | 0.2238 (0.115) | 2 000 | 2.6 |
| 10⁴ | 24 | 2.0965 | +2.06 % | 2.2959 | 0.1979 (0.826) | 0.2215 (0.116) | 2 000 | 1.7 |
| 10⁴ | 32 | 2.0785 | +1.18 % | 2.2766 | 0.1981 (0.826) | 0.2211 (0.116) | 4 000 | 3.2 |
| 10⁴ | 48 | 2.0656 | +0.55 % | 2.2627 | 0.1981 (0.826) | 0.2210 (0.116) | 8 000 | 7.1 |
| 10⁴ | 64 | 2.0610 | +0.33 % | 2.2578 | 0.1981 (0.826) | 0.2210 (0.116) | 12 000 | 14.5 |
| 10⁴ | 96 | 2.0577 | +0.17 % | 2.2544 | 0.1981 (0.826) | 0.2210 (0.116) | 27 000 | 83.8 |
| 10⁴ | extrapolated (24/48/96, order 1.99) | **2.0551** | **+0.04 %** | 2.2516 | | | | |
| 10⁵ | 24 | 4.6218 | +6.56 % | 4.9133 | 0.1466 (0.854) | 0.2491 (0.065) | 2 000 | 2.5 |
| 10⁵ | 32 | 4.4993 | +3.74 % | 4.7829 | 0.1444 (0.854) | 0.2516 (0.062) | 3 000 | 2.5 |
| 10⁵ | 48 | 4.4095 | +1.67 % | 4.6882 | 0.1426 (0.854) | 0.2455 (0.063) | 5 000 | 4.8 |
| 10⁵ | 64 | 4.3777 | +0.94 % | 4.6551 | 0.1421 (0.854) | 0.2456 (0.063) | 6 000 | 7.8 |
| 10⁵ | 96 | 4.3550 | +0.41 % | 4.6314 | 0.1417 (0.854) | 0.2456 (0.064) | 14 000 | 45.0 |
| 10⁵ | 128 | 4.3471 | +0.23 % | 4.6231 | 0.1415 (0.854) | 0.2456 (0.064) | 24 000 | 227.6 |
| 10⁵ | extrapolated (32/64/128, order 1.99) | **4.3367** | **−0.01 %** | 4.6125 | 0.1414 | | | |

Grid convergence index (Roache, safety factor 1.25) on the finest grid: 0.16 % (Ra 10⁴) and
0.30 % (Ra 10⁵) for the mean Nu. References (`thermal3d_cases` docstring for where each value was
read): mean Nu 2.0542 / 4.3371 (Tric, Labrosse & Betrouni 2000, Chebyshev pseudo-spectral),
2.062 / 4.366 (Wakashima & Saitoh 2004), 2.100 / 4.361 (Fusegi et al. 1991, 62³ finite
differences); symmetry-plane Nu 2.25 / 4.61 (Tric et al.), 2.302 / 4.646 (Fusegi et al.). The
extrapolated mean and symmetry-plane Nusselt numbers agree with the spectral solution to 0.04 /
0.01 % and 0.07 / 0.05 % (Ra 10⁴ / 10⁵); Fusegi's coarser values sit 2.2 % / 0.6 % above it,
ours converge to it from above at second order.

The velocity maxima on the centre lines of the symmetry plane agree with Wakashima & Saitoh at Ra
10⁴ (u_max 0.1981 at z = 0.826 vs 0.198 at 0.825; w_max 0.2210 at x = 0.116 vs 0.222 at 0.117) and
w_max at Ra 10⁵ (0.2456 at 0.064 vs 0.246 at 0.068 and Fusegi's 0.247 at 0.065), but u_max at Ra
10⁵ converges to 0.1414 at z = 0.854, **3.8 % below** the 0.147 at z ≈ 0.85 that both Wakashima &
Saitoh and Fusegi et al. report (the position matches). We could not access Tric et al.'s velocity
tables to settle it. Independent 3D solutions do scatter in this quantity: the cascaded
lattice-Boltzmann solution of Hajabdollahi & Premnath (2017, Table 1, 91³) gives 0.1441, 1.8 %
below the 0.1468 of Fusegi et al. in the same table and 1.9 % above ours (its symmetry-plane Nu,
4.46, is itself 3 % below Tric's 4.61, where ours agrees to 0.05 %). As an independent check of the velocities the same solver was run in its 2D
limit (`run_cavity_2d`: four cells deep between slip, adiabatic y walls, so the flow is exactly
two-dimensional) against de Vahl Davis (1983, Table V; κ/L units converted to free fall):

| Ra | n² | Nu (vs 2.243 / 4.519) | u_max (z) | ref | w_max (x) | ref |
|---|---|---|---|---|---|---|
| 10⁴ | 64 | 2.2506 (+0.34 %) | 0.1921 (0.823) | 0.1920 (0.823) | 0.2329 (0.119) | 0.2328 (0.119) |
| 10⁴ | 128 | 2.2462 (+0.14 %) | 0.1921 (0.823) | | 0.2329 (0.119) | |
| 10⁵ | 64 | 4.5610 (+0.93 %) | 0.1311 (0.855) | 0.1303 (0.855) | 0.2577 (0.066) | 0.2574 (0.066) |
| 10⁵ | 128 | 4.5314 (+0.27 %) | 0.1306 (0.855) | | 0.2577 (0.066) | |

Both maxima and their positions agree with the 2D benchmark to 0.2 % at 128², so the solver's
velocities are not the suspect; the Ra 10⁵ 3D u_max discrepancy remains an open question about
the 3D references' resolution or definition, recorded rather than tuned away.

**Forced convection from an isothermal sphere** (`run_heated_sphere`: the drag-validation sphere
of §10 — half domain, 2.85 % blockage, level-4 icosphere — held at θ = 1 in a free stream at θ = 0,
Ri = 0, Pr = 0.71; Nu from the penalisation's heat averaged over the last quarter of t = 40; the
energy balance is the heat carried out of the outlet minus the inlet's, over the body's):

| Re | cells/D | grid | Nu (penalisation) | vs Clift | vs Yuge | Nu (surface) | energy balance | Cd | steps |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 12 | 180×32×64 | 5.5137 | +0.66 % | +0.50 % | 5.2187 | 0.99996 | 1.6763 | 3000 |
| 50 | 16 | 240×42×84 | 5.5093 | +0.58 % | +0.42 % | 5.4196 | 0.99995 | 1.6542 | 4250 |
| 50 | 24 | 360×63×126 | 5.4974 | +0.37 % | +0.21 % | 5.4251 | 0.99990 | 1.6477 | 7750 |
| 100 | 12 | 180×32×64 | 7.3968 | +6.86 % | +6.74 % | 5.9165 | 0.99993 | 1.2061 | 2500 |
| 100 | 16 | 240×42×84 | 7.1462 | +3.24 % | +3.12 % | 6.4974 | 0.99995 | 1.1690 | 3750 |
| 100 | 24 | 360×63×126 | 7.0299 | +1.56 % | +1.44 % | 6.9434 | 0.99991 | 1.1454 | 6000 |

* Re 50: Nu 5.5137, 5.5093, 5.4974 at 12, 16, 24 cells/D; not in the asymptotic range, no extrapolation
* Re 100: Nu 7.3968, 7.1462, 7.0299 at 12, 16, 24 cells/D; observed order 3.31, extrapolated 6.9888 (+0.91 % from the Clift–Yuge centre 6.926)

Correlations (`thermal3d_cases` docstring): Ranz & Marshall 5.785 / 7.353, Whitaker 5.177 /
6.616, Yuge (air experiments) 5.486 / 6.930, Clift, Grace & Weber 5.479 / 6.922 at Re 50 / 100.
Clift's (checked against DNS for 1 < Re < 400, 0.25 < Pr < 100, Chouippe et al. 2019) and Yuge's
agree within 0.2 %; the published correlations span ±6 % around them, which bounds what a
comparison with correlations can prove. "Nu (surface)" integrates the local flux extrapolated from
probes 1.5 and 2.5 cells off the wall (as cf in §6.3): a cross-check that converges to the
penalisation value from below; the reported Nu is the penalisation's, which is exactly the heat
the body puts into the flow (energy balance).

### 15.7 Heat-flux bodies: the wall temperature and the Nusselt number

A body with a set heat flux knows its heat rate exactly (Q = G A, §15.3); its Nusselt number
Nu = Q/(A θ_w) is only as good as the wall temperature θ_w. Until this round θ_w came from
`surface()`: θ at probes 1.5 h and 2.5 h off the wall, extrapolated linearly to it. That needs
the thermal boundary layer to span both probes. At Draft resolution a body metres long at a few
m/s has a boundary layer of about one cell, both probes sit in nearly ambient fluid, and the
extrapolation returns almost the ambient temperature. The review's case — a 0.5 m box at
200 W/m², 1 m/s, Re 3.4·10⁴, 30 cells per length — reported Q = 156 W (exactly flux × area) but
Nu 1,821, i.e. a 2 K wall excess, while the colour map showed 13 K warm fluid along the wall. The
heat rate was right; the temperature it was divided by was not.

**The measurement now.** The skin of a heat-flux body is where the model puts the wall's heat: a
cosine delta one mask width either side of the surface. `ThermalGeometry.wall_theta` takes the
mean of θ over those cells, weighted like the release (a gather of ~10³ cells per body, recorded
every step as `theta_wall_<i>`; the per-step `nu` then covers heat-flux bodies too). At a
resolved boundary layer it agrees with θ sampled on the surface and with the probe extrapolation
(table: "surface", "probe" columns); unlike the probes it stays near the wall temperature when
the layer is one or two cells thick. `surface()` reports θ sampled on the surface itself for
heat-flux bodies (the per-vertex map), and `heat_transfer()`, `heat_rates()` and
`physics_history()` all use the skin temperature.

**Validation: laminar flat plate at uniform heat flux** (`thermal3d_cases.run_flux_plate`,
`bench_ns3d.py thermal --parts flux`). A plate of chord L, 10 % thick with semi-elliptic nose and
tail (2:1 aspect), spanning a periodic slab four cells wide (quasi-2D; the end caps lie outside
the domain), slip walls 1.5 L above and below (3.3 % blockage), 1 L of inlet and 2.5 L of wake,
Pr 0.71, Ri 0, run to t = 20 L/U and averaged over the last quarter. Reference: Nu_x = 0.453
Re_x^½ Pr^⅓ for a laminar plate at uniform flux (Incropera & DeWitt, eq. 7.45), whose mean wall
excess gives Nu_L = 0.680 Re_L^½ Pr^⅓ (12.12 at Re 400, 19.17 at Re 1000; the isothermal plate's
0.664 gives 11.85 / 18.73). Nu = G_in/θ_w with G_in the flux the fluid actually received per unit
in-domain area: the skin is normalised over the whole mesh, end caps included, so for this
periodic cut the in-domain release differs from the set G by −6.6 to +0.2 % with the grid
("released" column; for a body wholly inside the domain the total is exact). "local" is the mean
ratio of θ on the top surface to the similarity profile G x^½/(0.453 Re^½ Pr^⅓) over x = 0.2–0.8 L.

| Re_L | cells/L | cells across δ | Nu (skin) | vs 0.680 | Nu (surface) | Nu (probe) | released | local | energy balance | steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 32 | 8.0 | 13.733 | +13.3 % | 13.377 | 15.150 | 0.9790 | 0.912 | 0.9999 | 2400 |
| 400 | 48 | 12.0 | 13.229 | +9.1 % | 13.047 | 13.572 | 0.9679 | 0.936 | 0.9999 | 4000 |
| 400 | 64 | 16.0 | 13.132 | +8.3 % | 13.004 | 12.977 | 0.9929 | 0.946 | 0.9999 | 5800 |
| 400 | 80 | 20.0 | 13.336 | +10.0 % | 13.195 | 13.026 | 1.0022 | 0.929 | 0.9999 | 7800 |
| 400 | 96 | 24.0 | 13.813 | +13.9 % | 13.635 | 13.483 | 0.9830 | 0.887 | 0.9999 | 10000 |
| 400 | 128 | 32.0 | 13.385 | +10.4 % | 13.231 | 13.181 | 0.9335 | 0.910 | 0.9998 | 15000 |
| 1000 | 32 | 5.1 | 19.816 | +3.4 % | 19.115 | 25.663 | 0.9790 | 1.015 | 0.9916 | 2200 |
| 1000 | 48 | 7.6 | 19.225 | +0.3 % | 18.902 | 21.087 | 0.9679 | 1.029 | 0.9953 | 3600 |
| 1000 | 64 | 10.1 | 19.307 | +0.7 % | 19.097 | 19.583 | 0.9929 | 1.025 | 0.9954 | 5000 |
| 1000 | 80 | 12.6 | 19.698 | +2.8 % | 19.441 | 19.412 | 1.0022 | 1.002 | 0.9961 | 6400 |
| 1000 | 96 | 15.2 | 21.454 | +11.9 % | 21.120 | 21.136 | 0.9830 | 0.894 | 0.9980 | 7800 |
| 1000 | 128 | 20.2 | 20.259 | +5.7 % | 20.008 | 20.058 | 0.9335 | 0.946 | 0.9971 | 11000 |

* At **Draft density (32 cells per chord)** the skin value is within +13.3 % (Re 400) and +3.4 %
  (Re 1000) of the correlation, the bound the review asked for being ±30 %.
* The values do **not converge monotonically**: from 48 to 128 cells per chord they scatter by
  ±3 % (Re 400) and ±6 % (Re 1000) around +10 % / +4 %, with 96 cells high at both Reynolds
  numbers. The surface and probe readings move with the skin value, so this is the immersed
  plate's solution, not the measurement; we attribute it to where the plate's surfaces fall within a
  cell (the sub-cell placement of the diffuse wall); that is a hypothesis, not tested. An extrapolated value
  is therefore not quoted.
* **Plate thickness is not the cause of the offset**: Nu at 64 / 128 cells per chord is 13.26 / 13.32 for a
  6.25 % plate, 13.13 / 13.38 at 10 % and 13.93 / 13.47 at 15 % (Re 400), a weak effect inside the grid scatter.
  The offset shrinks with Re (+8 to +14 % at Re 400, mostly below +6 % at Re 1000), as the leading-edge
  corrections to the boundary-layer solution do (they scale like Re^-1/2); we have not computed them, so
  that attribution is a consistency argument, not a proof.
* **Under-resolved boundary layers** (32 cells per chord): at Re 3,000 (2.9 cells across the boundary layer) the skin gives Nu 30.1 (-9.2 %) while the pre-fix probe extrapolation reads 50.3 (+51 %); at Re 10,000 (1.6 cells) 55.3 (-8.7 %) against 137.3 (+127 %). The probe error is the review's box in miniature. (These two runs are not steady -- the 10 % thick trailing edge sheds, wall temperature drift 3.3 / 5.1 % between the halves of the window -- so they illustrate the bias, they do not validate.)

**Uniform-flux sphere** (`run_flux_sphere`: the validation sphere of §15.6 releasing G = 1, Re 100):
Nu (skin) 7.736 at 16 cells/D and 7.472 at 24 (surface 7.661 / 7.440, probe 8.876 / 7.847; energy
balance 0.9987 / 0.9982), against 7.146 / 7.030 for the isothermal sphere on the same grids. An
isoflux sphere's mean-temperature Nusselt number is close to and slightly above the isothermal
one in this regime, which is what the run shows (+8.3 % / +6.3 %); we found no tabulated
isoflux value we could access to compare with directly (Dhole, Chhabra & Eswaran 2006, Int. J.
Heat Mass Transfer 49, 984, give correlations for both; full text not accessible here), so the
sphere is a consistency check and the plate is the reference.

**The review's box, re-run** (0.5 m × 0.3 m × 0.3 m, 200 W/m², 1 m/s, Draft, buoyancy on, the same
plan; scratch project, 2 min 00 s on the GPU): Q 156.0 W as before; Nu 182.7 ± 12 (time series, now in
the Summary) and 181.6 per body from the mean fields, wall 41.4 °C (21.4 K above ambient), against 1,821
and an implied 2 K before. That sits inside the 100–300 the review estimated for forced convection over a
box at Re 3.4·10⁴. The run still reports itself not converged (the Nusselt number drifts by −24 between the
halves of the window while the box's interior warms) and warns that the grid gives one cell across the
thermal boundary layer: at this Reynolds number the heat transfer is indicative, as the plan says.

### 15.8 When buoyancy dominates: the time step of a Ri ≫ 1 run

A heated body in a slow stream is a plume, not a wake. The review's case — a 0.1 m sphere at 60 °C
in 20 °C air at Re 100 (U = 0.0146 m/s) — has Ri = gβΔT L/U² = 627 and Gr = 6.3·10⁶: the buoyant
velocity U_b = √(gβΔT L) = U √Ri is 25 times the stream. The solver's step follows the flow it
actually has: the heat equation's combined bound (§15.2) is dt ≤ 1/(S/(h/2) + 6κ/h²) with S the sum
of the three velocity maxima, and once the plume has formed S is set by U_b, not by U. The
planner had estimated the step from the free-stream speed factor and the free-fall bound
dt ≤ C √(h/(Ri |θ|)) of a parcel starting from rest, which is ~10× longer than the developed
step: the run was planned at 1 min 24 s and advanced 0.024 L/U per second (≈ 28 min for 40 L/U).

Measured (`thermal3d_cases.run_buoyant_sphere_steps`, `bench_ns3d.py thermal --parts plume`: the
validation sphere held at θ = 1, gravity along −z, 40 buoyant times L/U_b, float32 on the shared
GPU; "speed sum" is (max|u| + max|v| + max|w|)/U_b over the second half of the run):

| Re | Ri | cells/D | speed sum / U_b | mean dt (L/U) | planned dt | steps / planned | free-fall bound |
|---|---|---|---|---|---|---|---|
| 100 | 1 | 8 | 2.93 | 0.0195 | 0.0144 | 0.74 | 0.177 |
| 100 | 10 | 8 | 2.47 | 0.00770 | 0.00481 | 0.63 | 0.0559 |
| 100 | 100 | 8 | 3.65 | 0.00181 | 0.00155 | 0.86 | 0.0177 |
| 100 | 627 | 8 | 3.95 | 0.000675 | 0.000622 | 0.92 | 0.00706 |
| 100 | 627 | 16 | 4.57 | 0.000294 | 0.000310 | 1.06 | 0.00499 |
| 20 | 627 | 8 | 3.86 | 0.000682 | 0.000614 | 0.90 | 0.00706 |

"Planned dt" is the rule the planner now has from the solver package,
`ns3d.physics.thermal_dt_estimate(h, planning_speed_sum(Ri, S_forced), Pe)` with
`planning_speed_sum = max(S_forced, PLUME_SPEED_FACTOR √Ri)`, `PLUME_SPEED_FACTOR = 4.0` (U_b
units) and S_forced = 2.1 (the forced sphere's factor, `plan3d.SPEED_FACTOR3`): its step counts
are within 0.63–1.06 of the measured ones from Ri 1 to 627 (conservative where buoyancy and the
stream are comparable), while the free-fall bound overestimates the developed step 9–17×. The
test `test_buoyancy_dominated_sphere_steps_as_planned` holds the Ri 627 case within a factor 2.
Applied to the review's Draft grid (20 cells/D, Pe 71) the rule gives dt = 2.5·10⁻⁴ L/U, the
step the run took (0.024 L/U per second at ≈ 100 steps/s). What the planner must do with it
(owner: the simulation layer): when Ri ≳ 10, take U_b = max(U, √(gβΔT L)) as the velocity scale for
the duration and the averaging window (40 L/U_b is 1.6 L/U here) and the run-time estimate, and
say that natural convection governs (Gr, Ra), not Re.

## 16. Emitters: smoke channels and tracer particles (DESIGN.md 10.1)

**Smoke channels.** Up to four passive scalars — fields "dye", "dye1", "dye2", "dye3" — each
transported exactly like the base solver's smoke (§14: bounded, zero in solids, conserved away from
sources). An `Emitter3D` of species "dye" (solver units; the planner converts the model's metres,
seconds, colours to channels and the wind / body frame through `Emitter3D.scaled`) relaxes its
channel towards `rate` inside its footprint after every step, c += s·on·(rate − c), with `s` the
anti-aliased footprint weight and `on` the release switch: enabled, `start` ≤ t < `stop`, and the
first `pulse_duty` of every `pulse_period`. The switch is evaluated on the device from the solver
time and a per-emitter parameter tensor, so pulses, start and stop never leave a CUDA graph and
editing them never re-captures. Footprints: "point" (a nozzle of `radius`, default 1.25 cells,
two cells deep along `normal`), "line" (a capsule), "rake" (`count` nozzles), "rect" (an
axis-aligned box; a zero extent is a two-cell slab), "disc" (radius `size[0]`, facing `normal`),
"plane" (a rectangle facing `normal`, or the whole cross-section). `emitters=None` keeps the base
automatic rake on channel 0 (the default path), `emitters=[]` releases nothing; a jet's
`dye_channel` adds a channel too. Cost: one more cell-centred field and one fused advection
(§14, ~6 % of a step) per extra channel.

**Tracer particles** live in fixed device buffers (position, alive, birth, nozzle), one ring of
`particle_capacity` / n (default 2¹⁶) slots per particle emitter. Every step an emitter adds
`rate·dt·on` to an accumulator and releases its integer part — at most k_max per step, a static
size (k_max = ⌈4 rate dt_estimate⌉, at most a quarter of the ring), so the release is
graph-capturable; the candidate slots are always k_max long and those beyond the count keep their
contents. "point" and "rake" particles leave from their nozzle centres (a rake's nozzles take
turns, so each carries the same share; within `radius` when one is given), area kinds from
uniformly random points of their footprint. The releases of a step are spread evenly over it in
time: the j-th of n leaves at t − f_j dt with f_j = (j + ½)/n and moves only for f_j dt (its
full-step displacement is scaled), so a nozzle draws an evenly spaced streak line at any release
rate and `age = t − birth` is the time each has moved. (The first version stamped the step's end
time on particles that were then moved through the whole step, so every age was one dt short, and
scattered a nozzle's releases over its two-cell depth, which made streak lines zig-zag by ±1 cell
for 0.08 L of advance per step; both found by the tests and fixed.) Particles are advected with the
midpoint rule through the trilinearly interpolated staggered velocity of the step's final,
divergence-free state, and retired when they leave the domain, enter a solid (solid fraction > ½,
bodies and paint) or outlive `lifetime`; the ring overwrites its oldest particles (recycling); a
symmetry plane reflects them (the velocity is sampled at the mirror image, v reversed); periodic
axes wrap. `FlowSolver3D.particles()` returns positions, age, birth, emitter and nozzle;
`particle_lines()` one polyline per nozzle of every point / rake emitter, ordered from the nozzle
outwards and cut where a pulsed emitter paused (its dashes line up into timelines across a rake).

*CPU cost.* Graph capture needs static shapes, so a step advects every slot of the ring, alive or
not: free on a GPU, but on the CPU a 2¹⁶ ring holding 494 particles cost 412 ms per step against
27.8 ms for the flow step of the 32 × 12 × 12 test tunnel (float64, medians of 5 interleaved rounds
on a CPU shared with other jobs, `torch` default 16 threads; rounds scattered ±50 %). Eager CPU
steps therefore advect only the live slots (`ParticleSystem3D.step(compact=True)`; finding them is
a host sync, which costs nothing there and is never captured): 40.9 ms per step, ×1.47 the flow
step instead of ×14.8. The arithmetic per particle is unchanged, so the live particles are the
full ring's bit for bit (test); CUDA, eager or graphs, keeps the static path.

`tests/test_ns3d_emitters.py`:

| check | result |
|---|---|
| every footprint kind | weights in [0, 1], peak > ½, centred within one cell of where it was put; full-plane = whole cross-section |
| release switch | off before `start`, on/off within each pulse, off after `stop`; disabled = rate 0 |
| channels | channel 0 and 2 carry smoke, 1 and 3 stay exactly 0; values in [0, 1] |
| closed box at rest, emitter stops at t = 0.5 | saturates at the rate (0.6, 10⁻⁶); total smoke then constant to 10⁻¹³ |
| particle release, 73 steps at rate 300 | count = rate·t within one particle; all inside the disc |
| uniform flow U = 1 | positions = release point + age to within the nozzle depth (RK2 exact); none past the outlet |
| sphere in the flow, lifetime 1.5 | no particle in the solid; no age above the lifetime |
| ring of 64 at 2000 /s | 64 alive, the newest four steps' particles |
| CPU live-slot advection vs the full ring (disc with a lifetime + rake, sphere, 60 steps) | identical positions, births, emitters and nozzles |
| symmetry plane | no particle below y0 |
| rake of 3 nozzles | 3 streak lines, monotone downstream, starting at the nozzles; pulsed: more dashes; area kinds: no lines |
| CUDA graphs with a pulsed smoke rake and a particle rake | count = rate·t within 2, captured graphs used |

## 17. Fans, jets and vents (DESIGN.md 10.3)

**Actuator discs.** A fan is a uniform force density over a thin disc (thickness `thickness`,
default max(D/8, 2 h)), anti-aliased and normalised separately on the u-, v- and w-faces so the
force adds up to exactly T n̂ (the normalisation runs over the whole disc before the domain clips
it, so a disc centred on a symmetry plane puts exactly half of itself into a half-domain run, and
the report doubles it back). Modes: "thrust" (`value` = T/(ρU²L²)), "pressure_jump" (T = Δp A) and
"velocity": a PI controller in the step (no host sync) drives the measured disc-averaged axial
velocity to `value`. The controller starts from the momentum-theory thrust 2A u_t (u_t − U n_x),
with gains K_P = ½ g, K_I = g/τ, g = 2A max(|2u_t − U n_x|, |U n_x|, 0.1) (the derivative of the
momentum-theory thrust with respect to the disc velocity) and τ = 4 D / max(|u_t|, |U n_x|, 0.1)
(four disc transit times), and the thrust is clamped to ±(50 A (|u_t| + 1)² + |T_0|). `swirl` adds
a solid-body tangential force whose torque matches the angular-momentum flux of a swirl of `swirl`
× the disc velocity at the rim; its net force is removed per component (the samples of n̂ × r on a
staggered grid only cancel when the centre is on the grid; before this, an off-grid fan pushed
sideways: 1.9·10⁻³ of side momentum after 40 steps, now 10⁻¹⁷). Swirl about an axis cut by the
symmetry plane is not mirror-symmetric and is ignored there with a note. `fans_report(mean=…)`:
thrust, pressure jump, disc and induced velocity, volume flow, ideal power T·u_d, target and the
in-domain share; per-step series `fan<i>_thrust`, `fan<i>_velocity`, `fan<i>_flow` in
`physics_history()`.

**Jets and vents.** A jet is a mass source m = Q φ over a disc footprint (Q = speed·πD²/4) carrying
the momentum m U_j n̂, its temperature m θ_j and, with `dye_channel`, smoke; a vent is the matching
sink m = −Q φ, which removes the local velocity, temperature and smoke. The projection solves
∇²ψ = ∇·û − m, so the continuity residual is ∇·u − m (`field("divergence")`); a tunnel's outlet
carries the inflow plus Σm h³. In a closed domain the sources must balance: an unbalanced set gets
a note and the all-Neumann Poisson problem removes the excess uniformly (its compatibility
condition — the zero mode is dropped), i.e. the residual becomes a uniform Q_net/V. A jet is a
source rather than a penalised velocity region (DESIGN.md 10.3 suggested the latter) because the
mass and momentum fluxes are then exact by construction, and a vent can remove the local fluid
with whatever it carries.

The fluid the uniform removal takes out carries its temperature and smoke with it (review round 5 before the release):
with Q_net > 0 in a closed or fully periodic box, `PortSet.scalar_sources` also subtracts
`b Δt (Q_net/V) c` on every cell (`PortSet.leak`), and `heat_balance` books `(Q_net/V) ∫θ dV` as heat
out of the ports. Before, the conservative update kept it — a source `+c Q_net/V` everywhere — so the
mean temperature of a 32³ closed box fed by a θ = 1 jet (Q 0.0314) grew linearly at Q/V and would
have passed the jet's temperature at t ≈ 32, while heat out of the ports read 0. Now the mean follows
1 − exp(−Q t / V): 0.4505 at t = 19.06 against 0.4504, max θ 0.936, residual 3 × 10⁻⁴ of the heat in.
A net sink needs no term (the created fluid arrives at θ = 0 and dilutes).

`tests/test_ns3d_physics.py`:

| check | result |
|---|---|
| fan in a periodic box (no walls), along x and along (0.3, −0.5, 0.8) | total momentum = T t n̂ to 10⁻¹¹ |
| swirling fan, off-grid centre | axial momentum = T t (10⁻¹⁰), side momentum < 10⁻¹² ; v < 0 above the axis (right-hand rule) |
| velocity-regulated fan, target 1.4 in U = 1 | disc velocity within 1 %, induced velocity within 3 % after 600 steps |
| fan footprint on a symmetry plane | in-domain share ½ to 10⁻¹²; reported thrust = full fan |
| jet footprint | Σm h³ = Q and momentum flux = Q U_j n̂ to 10⁻¹² |
| jet in a tunnel | outflow − inflow = Q to 10⁻¹⁰; residual < 10⁻¹⁰; jet heat in = Q θ_j Pe |
| balanced jet + vent in a closed room | residual < 10⁻¹⁰, no note; a lone jet gets the balance note |
| lone θ = 1 jet in a closed adiabatic box (`test_physics_round5.py`) | θ ≤ 1 + 10⁻⁶; heat out of the ports = (Q/V) Pe ∫θ to 10⁻⁹ |

**Validation: actuator disc vs momentum theory** (`run_actuator_disc`: a uniformly loaded disc
of diameter 1 and thickness D/8 normal to U = 1, Re = 400 on D, T = C_T ½ A, half domain with the
symmetry plane through the axis, a slip-walled square tunnel 12 D long (4 D upstream) and 6 D wide,
blockage B = 2.18 %, laminar, averages over the last third of t = 30). Two branches: a disc that
pushes the flow downstream (a propeller or fan, direction +x) and one that holds it back (a
turbine, direction −x). References: Froude's unconfined theory (propeller C_T = 4a(1 + a), disc
1 + a, far wake 1 + 2a; turbine C_T = 4a(1 − a), 1 − a, 1 − 2a) and its confined form for the
tunnel, the streamtube model of Garrett & Cummins (2007: Bernoulli in the wake core and the
bypass, continuity, the tunnel's momentum balance; `confined_disc_theory`, which agrees with the
validation runner's independent implementation to all printed digits and tends to Froude's for
B → 0). The confined values are the like-for-like reference; "u at 4 D" is the axis velocity
4 diameters behind the disc, against the confined far-wake core velocity.

| branch | C_T | cells/D | C_T delivered | disc velocity | confined theory | error | Froude | u at 4 D | confined far wake | error |
|---|---|---|---|---|---|---|---|---|---|---|
| propeller | 0.4 | 12 | 0.4000 | 1.0860 | 1.0899 | -0.36 % | 1.0916 | 1.1793 | 1.1801 | -0.07 % |
| propeller | 0.4 | 16 | 0.4000 | 1.0868 | 1.0899 | -0.29 % | 1.0916 | 1.1793 | 1.1801 | -0.07 % |
| propeller | 0.8 | 12 | 0.8000 | 1.1614 | 1.1680 | -0.57 % | 1.1708 | 1.3356 | 1.3368 | -0.09 % |
| propeller | 0.8 | 16 | 0.8000 | 1.1627 | 1.1680 | -0.45 % | 1.1708 | 1.3355 | 1.3368 | -0.10 % |
| turbine | 0.4 | 12 | 0.4000 | 0.8974 | 0.8905 | +0.77 % | 0.8873 | 0.7833 | 0.7818 | +0.19 % |
| turbine | 0.4 | 16 | 0.4000 | 0.8965 | 0.8905 | +0.67 % | 0.8873 | 0.7831 | 0.7818 | +0.18 % |
| turbine | 0.8 | 12 | 0.8000 | 0.7625 | 0.7381 | +3.30 % | 0.7236 | 0.4876 | 0.4854 | +0.46 % |
| turbine | 0.8 | 16 | 0.8000 | 0.7604 | 0.7381 | +3.02 % | 0.7236 | 0.4873 | 0.4854 | +0.39 % |

The delivered thrust is the set thrust (C_T to 10⁻⁶, float32). For the **propeller** the disc
velocity is 0.3–0.6 % below the confined theory and the axis velocity 4 D downstream within 0.1 %
of the far-wake value; refining 12 → 16 cells/D moves the disc velocity by ≤ 0.11 %. The
**turbine**'s wake agrees as well (+0.2 / +0.4 %), its disc velocity is +0.7 % above theory at
C_T 0.4 but **+3.0 % at C_T 0.8**. Probes at 16 cells/D, C_T 0.8 (`actuator_disc_config(..., re=2000)` and a
disc of thickness D/4): raising Re from 400 to 2000 lowers the excess to +2.1 % (the inviscid theory has no viscous diffusion of
the wake's shear layer, so about a third of it is viscous); doubling the disc thickness to D/4
changes it by +0.3 %; the footprint-weighted disc velocity and the plain area average over r < R
in the disc plane agree to 0.03 %. The disc-plane profile is not flat (turbine: u = 0.758 on the
axis, 0.751 at r = 0.4 D, 0.742 at r ≈ 0.47 D; propeller: 1.173 on the axis, 1.155 at r ≈ 0.47 D):
the 1D theory assumes a uniform induction, while non-linear actuator-disc CFD finds the induction
rising towards the rim (Madsen 2023), and uniformly loaded Euler / RANS discs are reported not to
reproduce momentum theory exactly (Zengler, Troldborg & Gaunaa 2024, who attribute it to the
discretisation, after Mikkelsen 2004 and Hodgson et al. 2021). We found no tabulated value for
the size of that deviation at C_T 0.8, so the remaining ~2 % of the heavily loaded turbine is
recorded as measured, not explained; the slow test holds the propeller at C_T 0.8 to 1 % and the
turbine at C_T 0.4 to 1.5 % of the confined theory, and both wakes to 1 %.

**Validation: laminar round jet** (`run_round_jet`: a jet port of diameter d = 1 and speed 1,
Re_j = 100, at x = 2 d blowing along +x into a closed no-slip box 32 d long and 2w square,
balanced by a slow vent of diameter 2w − 4 and thickness 2 d near the far wall; from rest,
averaged over t ∈ [800, 880]; the axis lies on a row of cell centres). Reference: Schlichting's
(1933) similarity solution of the laminar round jet, the slender limit of the exact Landau–Squire
jet (for this jet's K/ν² = 7854 the exact solution's centre-line speed is 1.1 % higher at the same
K — Batchelor 1967 §4.6, whose momentum-flux formula we checked by integrating the solution's
convective, pressure and viscous fluxes through a sphere): u = u_c (1 + ξ²/4)⁻², u_c = 3K/(8πνx), Q = 8πνx
(independent of K), r_½ = 2√(√2 − 1) νx/√(3K/16π), K = ∫u² dA conserved. Measured over
10 ≤ x/d ≤ 22: the slopes of 1/u_c, r_½ and of the volume flux inside r = 2r_½ (62.4 % of
Schlichting's Q) against the similarity laws for the measured K (every slope is independent of
the virtual origin), the 1/u_c slope also for the injected flux K₀ = Q U_j = π/4 (exact by
construction of the port), the rms deviation of u/u_c from the similarity profile inside 2r_½,
and the momentum budget of the jet core (the forward flow around the axis; below).

| cells/d | box w | grid | K / K₀ | K drift | 1/u_c slope (K / K₀) | r_½ slope | inner Q slope | profile rms | budget residual | steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 12 d | 128×97×97 | 0.9473 | -4.60 % | -5.96 % / -0.72 % | -7.88 % | -10.81 % | 0.0106 | -0.12 % | 9000 |
| 6 | 12 d | 192×145×145 | 0.9513 | -4.56 % | -5.50 % / -0.66 % | -7.03 % | -10.69 % | 0.0077 | -0.12 % | 14500 |
| 4 | 24 d | 128×193×193 | 0.9730 | +0.90 % | -3.87 % / -1.21 % | -5.16 % | -4.10 % | 0.0087 | -1.53 % | 9000 |

* **Momentum is conserved by the scheme.** Integrating the steady x-momentum equation over the
  core, whose edge has u = 0, gives d/dx[K + K_p − K_v] = S − A dp_e/dx (K_p = ∫(p − p_e) dA,
  K_v = ∫ν ∂u/∂x dA, S the viscous shear of the surroundings on the core's edge, A its area, p_e
  the pressure just outside). Its residual over the span is −0.12 % of K at both grids: the
  4.6 % decline of K over the span is the shear of the box's return flow (−3.65 %) and its
  pressure gradient (−0.77 %), not the scheme. (Averaged over t ∈ [160, 240] instead, the
  residual was −3.0 %: the jet settles within ~100 time units, but the box-scale return flow it
  drives takes several hundred to spin up — the box holds ~18 000 d³ and the jet entrains ~5 d³
  per unit time — and until then the outer flow is unsteady; −0.36 % over [400, 480].)
* **Profile and decay.** Inside 2r_½ the profile is Schlichting's to 1.1 % (4 cells/d) and 0.8 %
  (6 cells/d) of u_c rms. With the injected K₀ the centre-line decay slope is within 0.7 %; with
  the jet's measured K (5 % below K₀) the decay is 5.5–6 % slower, r_½ grows 7–8 % slower and the
  inner volume flux 11 % slower than the similarity laws.
* **Those deviations are confinement, not resolution.** Refining from 4 to 6 cells/d changes
  them by ≤ 0.9 % (points), while a box of four times the cross-section (w = 24) moves all of them
  towards the free jet: 1/u_c −6.0 → −3.9 %, r_½ −7.9 → −5.2 %, inner flux −10.8 → −4.1 %, K below
  K₀ 5.3 → 2.7 %. (That box has not quite spun up by t = 880: budget residual −1.5 %.) The return
  flow around a jet in a closed box opposes it (u ≈ −0.003 to −0.010 at 4–8 d from the axis) and
  cancels its slowly decaying outer tail — a tenth of Schlichting's Q lies beyond 5r_½, which is
  why the whole forward flux is not compared. The 2D solver's plane jet (theory_ns2d.md) shows the
  same confinement.

## 18. Closed domains, live editing, diagnostics and the time-step controller (DESIGN.md 10.4)

**Closed domains** (`domain = "closed"`): walls on all six sides (`bc_x` at x0 / x1, the y and z
settings elsewhere), u = 0 on the x faces, no inflow, the flow starts from rest, and the pressure
reference is the domain mean. Natural convection, fans and jets drive the flow (§15.6, §17).

**Live edits** never rebuild the grid or the Poisson solver and keep the flow:

| call | what changes | graphs |
|---|---|---|
| `set_emitters(list / None)` | smoke and particle emitters | kept when every smoke source keeps its cells (rate, pulses, start/stop are device values); particles kept when unchanged |
| `set_sources(heat_sources=, fans=, ports=)` | heaters, fans, ports (each None = unchanged) | re-captured once; a fan that only changed its setting keeps its controller state |
| `set_thermal(walls=, theta_inlet=, richardson=, prandtl=)` | wall and inlet temperatures in place; Ri / Pr rebuild the operators | kept for wall values |
| `set_bodies`, `set_inflow(re=)` | as without placed physics; thermal masks follow the bodies | as without placed physics |
| `paint_solid(center, radius, erase=, shape="sphere"/"box", size=)`, `clear_paint()` | the paint layer | the first stroke adds the layer (one re-capture), later strokes edit it in place |
| `set_temperature(θ)` | the temperature field | kept |

On CUDA a smoke-rate change, a wall-temperature change and repeated paint strokes leave the
captured graphs and the state buffers in place (test).

**Painted obstacles** (explore mode's mouse-drawn solids) are a solid fraction on the whole grid
with the bodies' calibrated interface, penalised to rest after the bodies, blocking smoke and
particles; they carry no force report, conduct heat like fluid at rest and never enter the
differentiable core (`step3d_full` knows no paint; the body geometry is untouched). Erasing cuts a
hole whose wall sits on the eraser's outline, χ ← min(χ, 1 − χ_eraser), as in the 2D solver; a
stroke's smooth edge reaches ~0.7 cells beyond its radius and an eraser clears completely only
~1.3 cells inside its own, so erasing a dab with the same radius leaves a sub-cell film (χ ≤ 0.15
where the two ramps cross; 0.11 measured); an eraser two cells larger removes it. Like bodies,
painted parts thinner than ~2.6 cells (twice the interface half-width plus the offset) never
become fully solid. Inside a painted ball of 2.7 cells radius the residual speed is ~2 % of U
(penalisation leakage, as in bodies).

**Diagnostics** (`diagnostics=True`; the model's `record_diagnostics` defaults to on): one
device-side row per 10-step block (the block's last step, inside the CUDA graph, no host sync),
copied to the host with the force history: t, dt, the CFL number of the step taken, Σ max|u_i|,
the RMS and max continuity residual (∇·u − m), the relative Poisson residual of the last
projection, and per field u, v, w, p, θ the change over the step (discrete L2, RMS, max, relative
L2) and the heat-storage rate. `diagnostics()` adds `step`, `steps_per_s` (host-measured per
`advance` call) and `p_residual` (the 2D name); `diagnostics_csv(path)` exports them.

**Solver trace** (`trace()`, `trace_step(commit=False)`): one step on the reference core with every
Runge-Kutta stage captured as cell-centred fields — the stage value after advection, diffusion,
fans and ports, after the penalisation and buoyancy, the projection's right-hand side, ψ, the
correction −∇ψ, the projected velocity, θ — and norms (largest right-hand side, largest divergence
after the projection, relative Poisson residual |Lψ − rhs|/|rhs|, largest correction). The run is
untouched: the state is copied and every physics accumulator restored; `commit=True` then advances
one normal step (it agrees with the traced one to the fused/reference round-off, not bit for bit).
Measured on a heated sphere with a regulated fan (float64): projected divergence < 10⁻¹¹, Poisson
residual < 10⁻¹⁰ in every stage.

**PID time step** (`adaptive_dt = "pid"`): the CFL rule jumps to the stability limit every step,
so dt follows every flicker of max|u|. The controller acts on e = log(dt_target/dt_prev) with
dt_target = `pid_target` (0.85) × the limit: dt = dt_prev exp(K_I e + K_P (e − e₁) + K_D (e − 2e₁ +
e₂)), (K_I, K_P, K_D) = (0.3, 0.4, 0.05) (Söderlind-style digital control), the factor clamped
to [0.5, 1.2] per step and dt never above the limit, so the CFL number never exceeds `cfl`. All
state is on the device. Measured over 200 steps of an impulsively started sphere: step-to-step
dt ratios within [0.5, 1.2], the CFL number never above 0.5 and settled at 0.85 × 0.5 within 10 %
(the CFL rule sits at exactly 0.5).

**Performance.** *The default path costs what it cost before placed physics existed.* Without placed physics the step is
the base step: on the final code, 57 steps of a sphere with smoke and running means agree with
the pre-physics snapshot bit for bit (every field, the force history and the means: largest difference
exactly 0, laminar and WALE, CUDA graphs and eager) and launch the same CUDA kernels (2176 per 5
steps, the same multiset of names under `torch.profiler`); the host side adds a few Python
branches per step, which graphs never execute. Measured on a launch-bound 24 × 16 × 16 case,
interleaved over 7 rounds, the working tree runs at 1.002× (CUDA graphs; 6 of 7 rounds within
0.1 %), 1.006× (eager) and 0.984× (CPU, rounds scattered ±7 %) of the pre-physics snapshot's
throughput (the same `FlowSolver3D.advance` timed alternately on both trees) — no measurable cost against
the 2 % budget. On the idle GPU of the table below the default 4.2 M-cell case runs 54.7 steps/s with 17.5 ms
of kernels per step (§9's 44.9–49.2 steps/s and 23 ms were measured on a shared GPU).

*What each feature costs* (`bench_ns3d.py physics`: the 256 × 128 × 128 = 4.2 M-cell sphere,
fused kernels, float32). The cost is the GPU kernel time per step of this process (the sum of
each launch's least-disturbed duration over six 10-step blocks, eager), relative to the default
path; "spread" is the relative scatter of three such measurements, which exposes contention.
Launch counts and memory (peak above the pre-build allocation, CUDA-graph path) are
deterministic.

| configuration | launches / step | kernel ms / step | relative cost | spread | peak memory | extra bytes / cell |
|---|---|---|---|---|---|---|
| default (no placed physics) | 441 | 17.46 | ×1.00 | 0 % | 0.59 GB | +0 |
| diagnostics (one row / 10 steps) | 465 | 18.07 | ×1.03 | 0 % | 0.58 GB | -2 |
| PID time step | 477 | 17.77 | ×1.02 | 0 % | 0.58 GB | -2 |
| heat transfer, heated body, Ri 0.3 | 616 | 22.78 | ×1.30 | 0 % | 0.63 GB | +10 |
| heat transfer + WALE (Re 1e5) | 731 | 37.40 | ×2.14 | 0 % | 0.79 GB | +47 |
| fan (PI) + jet + vent | 542 | 17.96 | ×1.03 | 0 % | 0.58 GB | -2 |
| 4 smoke channels | 533 | 20.72 | ×1.19 | 0 % | 0.68 GB | +22 |
| 3 particle rakes (3 x 65536) | 1214 | 19.39 | ×1.11 | 0 % | 0.60 GB | +1 |
| everything | 1619 | 28.70 | ×1.64 | 0 % | 0.75 GB | +38 |

Measured 2026-10-19 00:11 (NVIDIA GeForce RTX 4060 Ti); largest spread 0 %.

Heat transfer costs 30 % of a step (θ's advection–diffusion in each of the three stages, its
penalisation, the buoyancy and the heat-rate accumulators) and ~10 bytes per cell at the peak;
the WALE row also carries WALE's own cost (§9). Diagnostics, the PID time step and a regulated fan
with a jet and a vent each cost 2–4 %. Each further smoke channel costs ~6 % and ~7 bytes per
cell (`BYTES_PER_CELL_DYE_CHANNEL_F32` = 8), 196 608 tracer particles 11 % (24 bytes each).
`estimate_memory_bytes` keeps a conservative 48 bytes per cell for heat transfer (the first
measurement, before the bench subtracted the previous solver's graph pool).

**Limitations — what to trust.**

* Heat transfer is validated for natural convection in an enclosure (Ra 10⁴–10⁵, 0.04 % from the
  spectral benchmark with a grid study), forced convection from a sphere (Re 50–100, inside the
  correlations' band; Re 50: +0.3 % from the Clift–Yuge centre at 24 cells/D; Re 100: +1.5 % from the Clift–Yuge centre at 24 cells/D) and the hydrostatic force on bodies. Mixed convection
  around bodies (Ri ~ 1 with a body) is computed by the same terms but has no 3D benchmark here.
* The 3D u_max at Ra 10⁵ is 3.8 % below the two 3D references that report it and 1.9 % below a
  third (lattice Boltzmann) solution (§15.6, open; the 2D-limit velocities agree to 0.2 %).
* Heated bodies need the momentum's resolution rules (≥ 16–24 cells across); the thermal boundary
  layer is thinner than the momentum one for Pr > 1, so high-Pr fluids (water, Pr ≈ 7) need
  ~√Pr times finer grids for the same accuracy — not validated here.
* θ is not clipped: with sources, diffusion and jets it is a physical field, bounded only by the
  limiter's TVD property for advection.
* Fans are uniformly loaded actuator discs (no blade-element loading, no tip loss); the
  momentum-theory comparison is laminar and confined (§17), and a heavily loaded disc that holds
  the flow back (C_T 0.8) shows a 3 % higher disc velocity than 1D theory (2 % of it unexplained).
* Jets are volumetric sources of finite thickness (≥ 2 cells): a jet thinner than ~4 cells across
  is smeared. The laminar round jet is validated (§17); turbulent jets (whose spreading the LES
  models would have to produce) are not.
* Particles are massless tracers (no inertia, no diffusion); a fixed ring per emitter caps the
  count (oldest recycled).
* Not in this release (DESIGN.md 10.4): Lattice-Boltzmann solver, neural-operator surrogates.

**References (§15–18).**

* Al-Rashed, Kalidasan, Kolsi, Borjini & Kanna (2017), Three-dimensional natural convection of
  CNT-water nanofluid confined in an inclined enclosure with Ahmed body, J. Therm. Sci. Tech. 12,
  Table 3 (the Wakashima & Saitoh and Fusegi et al. values quoted).
* Chouippe, Krayer, Uhlmann, Dušek, Kiselev & Leisner (2019), Heat and water vapor transfer in
  the wake of a falling ice sphere, arXiv:1902.08255 (status of the Clift correlation).
* Clift, Grace & Weber (1978), *Bubbles, Drops and Particles*, Academic Press.
* de Vahl Davis (1983), Natural convection of air in a square cavity: a bench mark numerical
  solution, Int. J. Numer. Meth. Fluids 3, 249–264.
* Fusegi, Hyun, Kuwahara & Farouk (1991), A numerical study of three-dimensional natural
  convection in a differentially heated cubical enclosure, Int. J. Heat Mass Transf. 34, 1543.
* Hajabdollahi & Premnath (2017), Central moments-based cascaded LBM for thermal convective flows
  in three dimensions, arXiv:1710.04751, Table 1.
* Li, Yang & Zhang (2015), Lattice Boltzmann method simulation of 3-D natural convection with
  double MRT model, arXiv:1511.04633, Tables 1–2.
* Oztop, Almeshaal, Kolsi, Rashidi & Ali (2019), Natural convection and irreversibility evaluation
  in a cubic cavity with partial opening, Entropy 21, 116, Table 1.
* Ranz & Marshall (1952), Evaporation from drops, Chem. Eng. Prog. 48, 141–146.
* Roache (1994), Perspective: a method for uniform reporting of grid refinement studies, J. Fluids
  Eng. 116 (GCI).
* Söderlind (2003), Digital filters in adaptive time-stepping, ACM TOMS 29.
* Tric, Labrosse & Betrouni (2000), A first incursion into the 3D structure of natural convection
  of air in a differentially heated cubic cavity, from accurate numerical solutions, Int. J. Heat
  Mass Transf. 43, 4043–4056.
* Wakashima & Saitoh (2004), Benchmark solutions for natural convection in a cubic cavity using
  the high-order time–space method, Int. J. Heat Mass Transf. 47, 853–864.
* Wang, Sadat & Prax (2018), A new meshless approach for three dimensional fluid flow and related
  heat transfer problems, arXiv:1809.02502, Table 1 (the Tric et al. and Fusegi et al. values).
* Whitaker (1972), Forced convection heat transfer correlations for flow in pipes, past flat
  plates, single cylinders, single spheres, and for flow in packed beds and tube bundles, AIChE J.
  18, 361–371.
* Will, Kruyt & Venner (2017), An experimental study of forced convective heat transfer from
  smooth, solid spheres, Int. J. Heat Mass Transf. 109, 1059–1067 (open access; the Ranz–Marshall,
  Whitaker and Yuge correlations as compiled there).
* Yuge (1960), Experiments on heat transfer from spheres including combined natural and forced
  convection, J. Heat Transfer 82, 214–220.
* Froude / Rankine actuator-disc momentum theory, e.g. Glauert (1935), Airplane propellers, in
  *Aerodynamic Theory* IV (Durand, ed.).
* Garrett & Cummins (2007), The efficiency of a turbine in a tidal channel, J. Fluid Mech. 588,
  243–251 (the confined actuator disc).
* Madsen (2023), An analytical linear two-dimensional actuator disc model and comparisons with
  computational fluid dynamics (CFD) simulations, Wind Energ. Sci. 8, 1853–1872.
* Zengler, Troldborg & Gaunaa (2024), Modelling the influence of streamwise flow field
  acceleration on the aerodynamic performance of an actuator disc, arXiv:2410.19527.
* Schlichting (1933), Laminare Strahlausbreitung, Z. Angew. Math. Mech. 13, 260–263; Schlichting
  & Gersten (2017), *Boundary-Layer Theory*, 9th ed., Springer (the round free jet).
* Squire (1951), The round laminar jet, Q. J. Mech. Appl. Math. 4, 321–329; Landau (1944) (the
  exact solution); Batchelor (1967), *An Introduction to Fluid Dynamics*, §4.6.
