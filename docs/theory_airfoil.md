# Airfoil analysis (polars): theory and validity

Aerolume's **Polar** analysis computes lift, drag and moment of a single airfoil section over a
range of angles of attack in a few seconds. It is an *XFOIL-class viscous–inviscid interaction*
(VII) method, built on **mfoil** by K. J. Fidkowski (MIT licence, vendored in
`src/aerolume/solvers/airfoil/_mfoil.py`; provenance and local patches in
`third_party/mfoil/README.md`). This page explains what the method solves, what the knobs mean,
and where its results stop being trustworthy. Validation against wind-tunnel data is in
[validation_airfoil.md](validation_airfoil.md).

For anything that is not a single, attached-or-mildly-separated airfoil (multi-element sections,
vehicles, deep stall, bluff bodies, low Reynolds numbers below ~5×10⁴) use the CFD analysis
(docs/theory_ns2d.md).

## 1. What is solved

**Outer (inviscid) flow.** A linear-vorticity panel method with one panel per pair of surface
points (199 panels by default, re-panelled from the user's outline with curvature-based
clustering). The stream function is constant on the surface, and the Kutta condition makes the
vorticity at the two trailing-edge nodes sum to zero. A blunt trailing edge carries an extra
source/vortex panel across the base. The viscous effect on the outer flow enters as
*transpiration sources* `σ = d(ue δ*)/ds` on the surface and along a wake streamline that leaves the
trailing edge, so the edge velocity is

    ue = ue_inviscid + D · (ue δ*)        (D = sensitivity matrix of ue to the sources)

**Boundary layer.** Two-equation integral boundary layer (von Kármán momentum and kinetic-energy
shape-parameter equations) on the upper surface, lower surface and wake, with Drela's laminar and
turbulent closures (as in XFOIL). The third equation is

* laminar: the **eᴺ envelope amplification equation** `dn/dξ = f(Hk, Reθ)/θ`;
* turbulent: a **lagged shear-stress (Cτ) equation**, which lets the turbulent stresses lag the
  local equilibrium value, important near separation.

**Coupling.** The unknowns at every node are `[θ, δ*, n or √Cτ, ue]`. All boundary-layer
equations plus the edge-velocity relation above are solved *simultaneously* by Newton's method,
with the transition location and the stagnation point linearised into the system. Solving
everything at once is what lets the method handle laminar separation bubbles and trailing-edge
separation without the instabilities of "direct" (prescribed-pressure) coupling.

**Forces.** Lift and moment come from integrating the surface pressure (`cm` about the quarter
chord, positive nose-up). Drag `cd` comes from the wake momentum thickness far downstream
(Squire–Young), which is more accurate than integrating surface pressure; `cdf` integrates the skin
friction and `cdp = cd − cdf` is the pressure (form) drag.

**Compressibility.** For Mach > 0 the inviscid pressures get the Kármán–Tsien correction and the
boundary-layer closures use the local edge Mach number. This is valid while the flow stays
subsonic everywhere; the polar warns when the minimum Cp drops below the critical (sonic) value.

## 2. Inputs and what they mean

| Input | Meaning | Guidance |
|---|---|---|
| Re | chord Reynolds number `ρ U c / μ` | the Setup page computes it from speed, chord and fluid |
| α | angle of attack from the **chord line** (LE → TE) | see section 3 for how the chord line is found |
| Ncrit | log of the disturbance amplification at which laminar flow transitions (eᴺ method) | 9 = clean flight / good low-turbulence tunnel (default); 4–6 = noisy tunnel, sailplanes in turbulence; 11–12 = very quiet flight. Rule of thumb (Mack): `Ncrit = −8.43 − 2.4 ln(Tu)`, Tu the free-stream turbulence fraction |
| xtr upper / lower | forced transition position (x/c) — a trip strip, grit, or a "fully turbulent" assumption | 1.0 = free transition (default). A trip upstream of the free location wins; one downstream has no effect. Trips closer than 0.06 chord (arc length) to the stagnation point act at 0.06 chord |
| Mach | free-stream Mach number | keep below ~0.5 and the flow subsonic |
| panels | number of surface panels | 199 is converged for engineering use; more costs time quadratically in places |

## 3. Geometry handling (`prepare_coords`)

Any outline (Design, Body or point array, any scale, position, orientation or point order) is
converted to Selig order (TE → upper → LE → lower → TE) at unit chord:

* The **trailing edge** is the end of the longest dimension where the outline turns through most
  of 180° within ~1 % of the length: a single sharp point, or the two corners of a blunt base
  (gap up to 10 % of chord; above that the shape is a bluff body and is rejected).
* The **leading edge** is the contour point farthest from the TE (XFOIL's definition), refined on a
  local spline. Exception: an outline already in the design frame (LE at the origin, TE on +x —
  every Aerolume template and NACA generator) keeps that chord line, so polar angles match the
  CFD analysis of the same Design. On a cambered nose the two definitions differ slightly (NACA
  4412: 0.18° and 3×10⁻⁴ c).
* Shapes facing downstream are mirrored; multi-body designs, vehicles, rounded or very thick
  tails (no Kutta condition), self-intersecting outlines, > 50 % thickness or > 25 % camber are
  rejected with a message starting *"Polar analysis needs a single airfoil-like body"*.

**Trailing-edge gap.** The viscous solver opens exactly sharp or near-sharp trailing edges
(< 0.05 % c) to 0.05 % c, blending the change linearly along the chord (XFOIL's `TGAP`). With an
exactly sharp TE the coupled iteration frequently stalls or converges to a spurious branch; with a
0.05 % gap it converges and the results match the converged sharp-TE ones to ~0.05 % in Cl
(measured on NACA 2412, Re 10⁶, 10°). The inviscid panel solution instead closes tiny gaps to an
exact sharp TE, because the base source panel is ill-conditioned when it is much shorter than its
neighbours.

## 4. Sweeps, convergence and robustness

* **Warm-started sweeps.** A polar starts with a cold solve at the angle nearest 0°, then walks
  upwards and downwards from there; every point starts from its converged neighbour, keeping the
  viscous edge-velocity correction `ue − ue_inviscid` (XFOIL-style), which roughly halves the
  Newton iterations per point.
* **Retries.** A point that fails is retried with the step split into 2 and 4 sub-steps
  (intermediate angles), then from a freshly marched boundary layer, then by ramping from 0° in
  2° steps. If all fail it is returned with `converged = False` and NaN coefficients, listed in the
  result's warnings, and excluded from `PolarResult.table()`. After three failures in a row (deep
  stall) each further angle gets a single cold attempt.
* **Convergence.** A point is converged when the 2-norm of all boundary-layer residuals is below
  10⁻⁸. In a few free-transition cases the transition location flips between two neighbouring
  panel intervals and the residual settles at ~10⁻³–10⁻² for ever (the amplification-rate onset
  is so steep that the transition point is not a smooth function of the state). Such a point is
  accepted when six successive iterates all have ‖R‖ < 0.02 and differ by less than 0.002 in Cl
  and 1 % in Cd; the polar's notes list these angles.
* **Cancel / progress.** `polar(..., progress=f, cancel=event)` reports `f(fraction, message)` after
  each angle and checks the event between Newton iterations; a cancelled sweep returns the points
  finished so far with a warning.
* **Threads.** Each call owns its solver state; there is no shared mutable state, so concurrent
  calls from worker threads are safe and give bit-identical results to serial calls.

Typical cost on the development machine (numpy/scipy, 2 BLAS threads): 0.3–1 s per converged
point; a 19-point NACA 2412 polar at Re 10⁶ took 12.5 s (validation page). Points that need
retries (stall, transition hopping) cost several times more. The hot loops run through
`_fast.py`, vectorised re-implementations of mfoil's kernels that are tested against the
originals to round-off; the pure-Python originals take ~6 s per point.

## 5. Validity limits — read before trusting a number

| Regime | Behaviour |
|---|---|
| Attached flow, Re 2×10⁵–10⁷, M < 0.3 | Cl within ~0.03–0.05, Cd within ~10–20 % of good wind-tunnel data (see validation). This is the method's home ground. |
| Laminar separation bubbles (Re ≲ 5×10⁵) | modelled (the coupled method captures bubbles), but sensitive to Ncrit and panel density; treat Cd as ±20–30 %. |
| Re < 5×10⁴ | warned; bubble bursting and unsteady separation dominate. Use CFD or experiment. |
| Approaching stall (last few degrees before Cl,max) | Cl,max and the stall angle are **over-predicted**: measured +0.05 to +0.1 on NACA 0012 (Re 3–6×10⁶) and +0.15 to +0.25 on NACA 2412/4412 (Re 6×10⁶), with stall 2–4° late. Integral boundary layers cannot represent massive trailing-edge separation and the wake model assumes a thin wake. |
| Cambered sections, linear range | lift slightly high: NACA 4412 +0.07 in Cl (≈0.7° in zero-lift angle) against Abbott & von Doenhoff; NACA 2412 within 0.01–0.02. A kink in Cl and Cm appears where the lower-surface transition point runs off the trailing edge (edge of the drag bucket; see below). |
| Post-stall | **not predicted.** The method has no physics for large separated regions; points either fail to converge (marked) or return optimistic values. Never use post-stall polars from this solver. |
| Compressibility | Kármán–Tsien correction only; no shocks. Above the critical Mach (warned) results are invalid. |
| Surface roughness, 3D effects, tunnel walls | not modelled (roughness only via forced transition). Sections are 2D (infinite span). |
| Very thin (< 0.5 % t/c), very thick (> 30 % t/c) sections | warned; leading-edge separation or early separation makes results uncertain. |

**Ncrit and transition.** Free transition is the largest single source of drag uncertainty: the
eᴺ method predicts *where natural transition would occur in a flow with a given disturbance
level*, and real surfaces (paint, insects, rivets, rain) usually trip earlier. When in doubt, run
the polar twice — free (Ncrit 9) and tripped near the leading edge (xtr ≈ 0.05) — and treat the
pair as the bounds of the real drag.

**The lift-curve kink at the edge of the drag bucket.** As the angle of attack rises, the
pressure-side (lower) transition point moves aft and finally off the trailing edge. Over the last
1–3° before it leaves, the laminar boundary layer ahead of it separates a few per cent of chord
before the trailing edge (H rises from ~3.5 to 6–7 on NACA 2412 at 3.5°, Re 10⁶). Its displacement
thickness acts as extra aft camber: the lift slope rises by up to 50 % and Cm turns nose-down by
~0.01, and both recover once that surface is laminar to the trailing edge. NACA 2412, Re 10⁶,
Ncrit 9: dCl/dα = 0.10–0.11 /deg up to 1.5°, 0.137 and 0.153 /deg over 2–4°, then 0.08–0.09 /deg;
Cm −0.045 at 2°, −0.060 at 4°, −0.056 at 5°. This is the model's prediction, not a numerical
artefact: the pristine mfoil run cold at every angle, and XFOIL 6.99 run on the same coordinates,
give the same curve (Cl within 0.006, Cm within 0.001 over 0–6°; validation page). How
strongly a real wing section shows it depends on the free-stream disturbance level and surface
finish, which Ncrit only approximates; a lower Ncrit or a trip moves or removes it. The polar's
notes name the angles where it occurs.

## 6. References

* K. J. Fidkowski, "A coupled inviscid–viscous airfoil analysis solver, revisited", *AIAA Journal*,
  2022 — the mfoil method (<https://websites.umich.edu/~kfid/codes.html>).
* M. Drela, M. B. Giles, "Viscous-inviscid analysis of transonic and low Reynolds number airfoils",
  *AIAA Journal* 25(10), 1987; M. Drela, "XFOIL: an analysis and design system for low Reynolds
  number airfoils", 1989 — the closures, eᴺ envelope method and coupling strategy.
* L. M. Mack, "Transition and laminar instability", JPL Publication 77-15, 1977 — the Ncrit–Tu
  relation.
