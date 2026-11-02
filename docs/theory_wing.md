# Finite-wing analysis: theory and validity

Aerolume's **Wing** analysis predicts the lift, drag, pitching moment, lateral loads and span
loading of a wing from its planform and the 2D polars of its airfoils, in the spirit of XFLR5:
seconds per polar instead of the minutes-to-hours of 3D CFD. Two methods are available:

| | Vortex lattice (VLM, default) | Nonlinear lifting line (LLT) |
|---|---|---|
| Model | ring vortices on the cambered mean surface | horseshoes on the quarter-chord line |
| Planforms | any: low AR, sweep, dihedral, kinks, sideslip | straight or mildly swept (≤ ~20°), AR ≳ 4 |
| Lift | linear (potential flow), no stall | nonlinear: section lift curves from the viscous polars up to cl_max |
| Drag | Trefftz-plane induced + strip profile drag | Trefftz-plane induced + strip profile drag |
| Use for | design-point loads, any shape, moments | lift curve towards stall, stall onset |

Validation against theory, lifting-surface results and wind-tunnel data is in
[validation_wing.md](validation_wing.md). Code: `src/aerolume/solvers/wing/`
(`geometry.py`, `vortex.py`, `vlm.py`, `llt.py`, `polars.py`, `solver.py`).

## 1. Geometry

A wing is a `WingSpec`: a list of `WingSection` stations (root first) joined by straight lines.
Leading edge `(x_le, y, z_le)`, chord and twist vary **linearly in y** between two stations, so
the leading and trailing edges are straight and the planform is exactly what the user typed.
`symmetric=True` mirrors the stations about y = 0 (the sections describe the right half,
y ≥ 0, DESIGN.md 7.2); a first station at y > 0 leaves a gap at the centre (two separate lifting
surfaces).

* **Twist** rotates a section nose-up about the spanwise axis through its leading edge.
  **Dihedral** is the slope of `z_le` in y; camber offsets are applied along the local surface
  normal (tilted by the dihedral), so a dihedral wing's sections stay perpendicular to its surface.
* **Airfoils** are NACA 4/5-digit codes (generated) or `design:<id>` (a 2D airfoil design of the
  project, resolved by the caller). Only the **mean line** enters the vortex methods (thin-wing
  theory). It is taken as the average of the upper and lower surfaces at equal x/c in the
  *chord-line frame of the polar solver* (`prepare_coords`), so camber, section incidence and polar
  angles share one reference line. Between two stations with different airfoils the mean line and,
  in viscous runs, the section polars are blended linearly with the span position.
* **Spanwise panels** (`n_span` per half, `spacing` "cosine" or "uniform"): the distribution is
  defined over the whole half span and split at every section station, so kinks fall on panel
  edges and each segment gets panels in proportion to its share (at least one). "cosine" clusters
  panels at the tip only (`y = (b/2) sin θ`, which makes the full wing the classical full-cosine
  distribution `y = (b/2) cos θ`) or, when the root is a free edge too, at both ends.
* **Control points** sit at the midpoint of the *mapped coordinate θ*, not of y (Multhopp's
  "semicircle" rule). Lift falls like √(distance) at a tip; collocating at the θ-midpoint makes
  the discrete solution of an elliptic wing exactly elliptic (e = 0.997–0.999 with any panel count,
  against e = 1.025 at 24 panels per half and first-order convergence with y-midpoints).
* **Chordwise panels** (`n_chord`) are uniform, with the vortex at 1/4 and the control point at
  3/4 of each panel (exact 2D lift and moment for flat and parabolic mean lines). The control-point
  normal uses the *local* mean-line slope at the 3/4 point. The panel's corner points would give
  the secant slope, which is the slope at the panel *midpoint*: an O(Δx) error that put NACA 2412's
  zero-lift angle at −1.87° instead of −2.12° with 8 chordwise panels (now −2.117°).

Planform metrics (`planform_metrics(wing)`) are exact for the piecewise-linear planform:
`area` is the planform area projected on the x–y plane, `span` the tip-to-tip distance in y,
`aspect_ratio = span²/area`, `mac = ∫c² dy / ∫c dy` with its leading-edge x and spanwise station,
`taper`, the root-to-tip quarter-chord `sweep_c4_deg` and `mean_chord = area/span`.

## 2. Frames, references and signs

The wing is solved in its own frame (x aft, y spanwise, z up). Following DESIGN.md 7.2, the angle
of attack α rotates the wing about +y through the moment reference by +α (nose up) and the
sideslip β (FlowConditions' `yaw_deg` convention) rotates it about +z by −β first. Equivalently the
free stream seen by the wing is

    v̂∞ = (cos α cos β, cos α sin β, sin α)

(`freestream_direction`); positive β gives the flow a +y component, so the −y half is windward.
Forces and moments are rotated back to **wind axes**: CD along the free stream, CY along +y, CL
along +z; C_roll, Cm and Cn are moments about +x, +y and +z through `moment_ref` (right-hand rule;
Cm > 0 is nose up). Coefficients use `q·ref_area`; Cm also `ref_chord`, C_roll and Cn
`ref_span`. Defaults: planform area, MAC and span, and the moment reference at the quarter chord of
the MAC (x = x_le,MAC + MAC/4, y = z = 0). A wing analysed from a project uses the design's
`ref_area`, `length_ref`, `span_ref` and `moment_ref3`.

Strip outputs (`WingPoint` arrays, both halves, ordered in +y): `cl_local = 2Γ/(V∞ c)` (the
section lift coefficient based on the free stream; for the lattice Γ is the strip's total bound
circulation), `c_cl = c·cl/c̄` with c̄ = area/span (its span average equals CL), `alpha_i` the
induced angle [deg] and `cd_local` the strip profile drag.

## 3. Vortex-lattice method

Each panel of the mean surface carries a vortex ring: its leading segment on the panel's
quarter-chord line, its trailing segment on the next panel's quarter-chord line (a quarter panel
behind the trailing edge for the last row) (Katz & Plotkin 2001, §12.3). The last row's side
segments continue as **semi-infinite trailing legs along the free stream** (rebuilt at every α and
β, so the wake leaves the trailing edge with the flow); their strength is the trailing-edge ring's,
which is the Kutta condition. Shared segments are evaluated once with their net circulation.

The Biot–Savart kernels (`vortex.py`) are the singularity-free forms used by AVL and Katz &
Plotkin: a finite segment a→b induces `(r₁×r₂)(|r₁|+|r₂|)/(4π|r₁||r₂|(|r₁||r₂|+r₁·r₂))` and a
semi-infinite one `(d×r₁)/(4π|r₁|(|r₁|−d·r₁))`; points closer than 10⁻⁶ of the segment length to
its line (its own midpoint included) get zero. Flow tangency at every control point gives a dense
linear system `A Γ = −V∞·n`; the bound-vortex part of `A` is built once per wing, only the wake legs
change with the flow direction. 8 × 48 panels solve in ~25 ms per angle on this machine.

**Forces.** Kutta–Joukowski `F = ρ Γ (V × dl)` on every bound segment (spanwise and chordwise),
with V the free stream plus the velocity induced by the whole system at the segment midpoint
("near field"): lift, side force and all moments come from these. **Induced drag** comes from the
**Trefftz plane**: the trailing legs, projected on the plane normal to the free stream, are 2D point
vortices; `D_i = −(ρ/2) Σ Γ_s (v·n)_s Δs_s` over the trace segments, with the normal-wash evaluated
at the same semicircle fractions as the lattice. The near-field drag of a lattice needs the
leading-edge suction a lattice cannot resolve and converges poorly; the far-field value is accurate
on coarse lattices (an elliptic wing gives e = 0.997–0.999). The strip induced angle is half the
Trefftz-plane downwash angle (the lifting-line relation).

**Compressibility (Prandtl–Glauert / Göthert).** Influence coefficients are evaluated on the
geometry stretched by 1/β in x (β = √(1−M²)), velocities are mapped back as (u/β, v, w), and the
tangency condition uses the physical normals. Γ, hence the Kutta–Joukowski lift and the Trefftz
drag, carry over unchanged (tested: CL(M) equals CL/β of the incompressible wing stretched by 1/β).
Valid for subcritical flow; above M 0.6 the result carries a warning, M ≥ 0.9 is refused.

## 4. Nonlinear lifting line (Phillips & Snyder 2000)

Horseshoe vortices with bound segments on the quarter-chord line between the spanwise nodes and
trailing legs along the free stream. At each control point the 3D vortex lifting law is equated to
the section lift:

    R_i = 2 |V_i × dl_i| Γ_i − |V_i|² cl_i(α_i) dA_i = 0
    V_i = v̂∞ + Σ_j v_ij Γ_j,    α_i = atan2(V_i·u_n, V_i·u_a)

with `dA_i` the strip area and `u_a`, `u_n` the section's chordwise and normal directions (twist
and dihedral included). The full local velocity is kept, so sweep, dihedral and sideslip enter
directly, and `cl_i(α)` may be nonlinear. Newton's method with the analytic Jacobian and a
backtracking line search solves the system from the linearised (Prandtl) solution or from the
neighbouring angle's solution (sweeps run outward from 0°); a damped fixed-point iteration is the
fallback. A point that does not converge is marked `converged=False`.

Section lift curves: inviscid runs use thin-airfoil theory `cl = 2π(α − α_L0)/β` with α_L0 and
cm_c/4 integrated exactly from the piecewise-linear mean line; viscous runs use the section polars
(§5) through a monotone cubic (Fritsch–Carlson) on the *attached branch*, from the angle of cl_min
to the angle of cl_max. **Beyond the branch the lift is clipped** (held at cl_max with zero slope)
and the result carries a warning: an integral boundary-layer polar does not predict post-stall
lift, so those points are not a stall prediction. Moments add the section cm (c/4) to the moment of
the vortex forces. Compressibility enters through the section lift curves only.

## 5. Viscous strip coupling

`SectionPolars` caches 2D viscous polars (`solvers.airfoil.polar`, the XFOIL-class solver with
its `ncrit`, `mach` and trip settings) per **(airfoil, Reynolds number)** and serves interpolated
`PolarTable`s:

* **Reynolds number** per strip `ρ V∞ c / μ` with the strip's mean chord. Polars are computed at
  "nodes": the strips' extreme Reynolds numbers and enough geometric steps between them that two
  nodes are at most a factor 2 apart; existing nodes are reused whenever they already bracket a
  strip that closely, so re-analysing a wing (another method, other angles) costs nothing. Strips
  below 30 % of the largest strip Re (tips) use the polar at that floor (a note says so).
* **Interpolation**: linear in α between the computed angles (a 1° grid aligned to multiples of
  the step, so nodes extend cleanly), linear in log Re between bracketing nodes for cl and cm and
  linear in log Re of log cd for the drag (exact for a power law cd ∝ Re⁻ⁿ).
* **Range**: before any polar runs, an inviscid pass over all requested angles estimates every
  strip's section angle (thin-airfoil inversion of its lift) and the polars are computed only over
  that range plus margins (2° below; 2° + 15 % of the angle from zero lift above). When a strip
  later needs more lift than a polar's computed range reaches while that polar is still rising
  ("range" rather than "stall" limited), the polar is extended by 3° steps (up to 25°), seeded
  from its converged edge so the sweep starts from a known boundary-layer state.
* **Stall detection**: a table's cl_max is a real stall when the lift falls by more than 0.02
  after it or the solver failed above it.
* **Persistence**: `state()` returns the computed points as JSON-safe data and `load_state()` merges
  them back (refused when Ncrit, Mach, trips, panels or the angle step differ). Airfoils are keyed
  by name, so a caller that persists the cache must drop it when a `design:<id>` outline changes.

**Lattice:** the strip lift `cl_local` is inviscid; its profile drag is the section polar's at the
angle where the polar gives that lift (the inverse of the attached-branch cl(α); XFLR5 does the
same). Strips asked for more than the section cl_max take the drag at cl_max and the result
warns that the lattice's linear lift is optimistic there. **Lifting line:** profile drag and cm at
each strip's solved effective angle. In both, the profile-drag force `q cd A` acts along the free
stream at the strip's quarter-chord point (so it contributes to Cm and Cn), and
`CDv = Σ cd_j A_j / S_ref`, `CD = CDi + CDv`.

**Stall onset (critical section).** In viscous runs, the margin of the most loaded strip (lattice:
cl_local − cl_max of its section; lifting line: α_eff − α(cl_max)) is followed up the sweep; where
it crosses zero, α and CL are interpolated and reported as `WingResult.stall_alpha` / `stall_cl`
(NaN when the sweep does not reach it, and for inviscid runs), in the notes ("Stall onset … CL ≈
…") and in `details["stall"]`. This is the classical critical-section estimate of CL_max; on a washed-out
wing the stall spreads gradually and the measured CL_max is usually 0–0.1 higher (validation).

**Tips.** Strips within 5 % of the semispan from a tip are left out of the stall-onset estimate and
of the clipping warnings (their loads still count; a note lists angles where only they exceeded
cl_max). Strip theory fails where the tip vortex forms, and a rounded tip modelled by sections whose
chord falls to zero is the extreme case for a lifting line: the strong trailing leg just inboard of
the last, tiny strips puts them in upwash, and their local cl reaches 1.2 where the inboard wing
carries 0.45 (NACA TN 1270 planform, α = 6°; the lattice, which resolves the chord, gives 0.39).
Without the exclusion those strips would "stall" first at every angle.

**Span efficiency and Oswald factor (`analysis.wing_metrics`).** Two different numbers, both
reported. The *span efficiency* e = CL² / (π AR CDi) is per point, from the Trefftz-plane induced
drag only (≈ 0.99 for an untwisted taper-0.6 wing). The *Oswald factor* is the aircraft-design
one: fit CD = CD0 + b·CL + k·CL² and take e = 1 / (π AR k), so it also contains the growth of
profile drag with lift and is lower (0.7–0.85 for most wings). It depends on the CL range fitted,
so the fit is restricted to the linear, attached polar: from the CL of minimum drag up to 0.8 ×
CL_max (the critical-section `stall_cl` when known), never at or past `stall_alpha`, where the
lattice keeps strips at cl_max while their lift keeps rising. The range is reported with the
figure (`oswald_cl_lo` / `oswald_cl_hi`), and with fewer than 4 points or R² < 0.98 the factor is
not reported at all. Fitted over a whole −4° to 14° sweep the tapered wing of section 7 gave
0.54, which reads as a very poor wing; over CL 0.10–0.92 it is 0.76. For an inviscid run the fit
sees induced drag only and returns the span efficiency (rectangular AR 8: 0.968 vs 0.968).

## 6. Assumptions and validity limits

* **Thin wing, potential flow.** Thickness affects the solution only through the section polars
  (lifting line) — the lattice's lift is that of the mean surface. No fuselage, nacelles, gaps,
  flaps or control-surface deflections; wing–body interference is not modelled.
* **Attached flow.** No stall prediction beyond what the section polars contain: the lattice is
  linear; the lifting line clips each section at its cl_max (warned) and reports the
  critical-section stall onset. Use CFD for post-stall behaviour.
* **Strip theory for viscous effects.** Each strip behaves like a 2D section in the free stream
  direction: no spanwise boundary-layer flow (relevant on swept wings), no tip-vortex suction on
  the outer few percent of the span, no influence of the wake on transition. Section data are as
  good as the 2D solver (docs/validation_airfoil.md): cd within ~6–15 % of wind-tunnel data,
  cl_max late and high.
* **Flat, prescribed wake.** The wake follows the free stream from the trailing edge; no roll-up,
  no ground effect.
* **Lifting line**: no chordwise loading — over-predicts the lift slope of low-aspect-ratio wings
  (3–12 % for elliptic wings of AR 12–4, against the lattice) and is unreliable beyond ~20° of
  sweep (warnings below AR 4 or above 20°).
* **Compressibility**: Prandtl–Glauert, subcritical only (M < 0.6 trusted).
* **Reynolds number**: the section solver's limits apply (indicative below Re ≈ 5×10⁴; the tip
  floor above keeps very small tip chords from requesting such polars).

## 7. Cost

On the development machine (single process, `OPENBLAS_NUM_THREADS=2`): the inviscid lattice
(8 × 48 panels) takes ~25 ms per angle and the lifting line ~1 ms per angle. A viscous polar is
dominated by the 2D section polars (≈0.5 s per section angle at Re 2–4×10⁵, more near stall):
a 17-point polar of a two-section wing needs ~40 section points, ~20–25 s cold and < 1 s once
cached (validation_wing.md, Performance). Section points near and beyond stall cost several times
more (Newton retries), so sweeps far past cl_max are slow: the 19-angle (−4° to 14°) polar of a
tapered NACA 2412 wing (AR 8.3, strip Re 3.1–5.1×10⁵) computed 52 section points in 26–44 s cold,
depending on the load on the machine. **A cold viscous wing polar takes under a minute, not
seconds**; what makes later runs fast is the cache:

* **Reynolds buckets.** Section polars are computed at Reynolds-number nodes on a fixed
  logarithmic grid (48 per decade, 4.9 % apart; every Re is within 2.4 % of a node, inside the
  5 % at which a strip uses one node alone). New nodes extend the existing ones greedily, so a
  study that changes the speed adds only the nodes its new strip Reynolds numbers need: the
  wing above at 30 m/s after 25 m/s took 9.5 s (22 new points), where the first placement
  (nodes at both ends of every uncovered range) added two nodes and took 36 s, longer than the
  cold run.
* **Kept between sessions.** `simulation.wing` writes each cache to the user's configuration
  folder (`cache/section_polars`, a few KB) after a run that computed points and reads it back
  the first time a later session needs it, so a script, the CLI or the assistant (each a new
  process) re-running a wing analysed before takes ~0.4 s instead of repeating the cold cost.
  The file key holds the solver settings, Ncrit, Mach, a version and the airfoil fingerprint
  (project airfoils by a hash of their coordinates); `AEROLUME_NO_POLAR_CACHE=1` turns it off and
  `clear_polar_cache()` deletes it.
* **Estimates from the cache.** `section_points_needed(...)` plans the first pass of a sweep
  without computing anything and asks the cache what is missing;
  `simulation.wing.estimate_wing_seconds(design, sim)` turns that into a wall time
  (0.6 s per point, ×1.2 for the polars a sweep extends towards cl_max). Measured on the wing
  above: cold 34 s estimated / 44 s run, another speed 17 s / 9.5 s, cached 0.9 s / 0.4 s.

## 8. API

```python
from aerolume.solvers.wing import analyze_wing, wing_polar, SectionPolars, planform_metrics

sp = SectionPolars(ncrit=9, mach=0.0, coords_provider=lookup)   # one cache per flow condition
res = wing_polar(wing, alphas, speed=20.0, fluid=Fluid(), method="vlm", viscous=True,
                 beta_deg=0.0, moment_ref=None, polar_provider=sp, progress=cb, cancel=event,
                 details=extra)                                  # -> WingResult
pt = analyze_wing(wing, alpha_deg=4.0, speed=20.0, polar_provider=sp)   # -> WingPoint
```

`progress(fraction, message)` is reported during the section-polar warm-up (90 % of a cold
viscous run) and per angle; `cancel` returns the points finished so far with a warning; a point
that cannot be computed is `converged=False` with NaN coefficients. Bad inputs (fewer than two
sections, non-increasing y, unknown airfoil, Mach ≥ 0.9, |α| ≥ 45°) raise `ValueError` before any
work. `details` (optional dict) receives `stall`, `strip_y`, `strip_re`, `polar_nodes` and
`reference`.

## References

* W. F. Phillips, D. O. Snyder, "Modern adaptation of Prandtl's classic lifting-line theory",
  *Journal of Aircraft* 37(4), 662–670, 2000.
* J. Katz, A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001
  (ch. 10, 12).
* M. Drela, H. Youngren, *AVL* documentation (vortex kernels, Trefftz-plane drag), MIT.
* H. Multhopp, "Methods for calculating the lift distribution of wings (subsonic lifting-surface
  theory)", ARC R&M 2884, 1950 (semicircle collocation; circular-wing benchmark).
* J. C. Sivells, R. H. Neely, "Method for calculating wing characteristics by lifting-line theory
  using nonlinear section lift data", NACA TN 1269, 1947 (critical-section stall estimate).
* A. Deperrois, *XFLR5 v6 Guidelines*, 2013 (strip coupling of VLM and section polars).
