# Changelog

All notable changes to Aerolume are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Planned for 0.2
The three largest physics limits of 0.1.0 (see *Known limitations* below and the README's
*Limits*) are the work of the next version. A design for them, with prototype measurements on
the current solvers, has been written; none of it is in 0.1.0.
- **Local grid refinement and a sharper immersed wall**: stretched or nested grids, so thin
  boundary layers and sharp edges converge on fewer cells.
- **A wall model with a transition model**, so absolute drag at high Reynolds numbers can be
  validated (flat-plate friction, the cylinder at Re 3900, the Ahmed body).
- **Finite-width 2D vehicles**: run a side profile as a 3D body of a chosen width, checked
  against published finite-width drag data.

## [0.1.0] - 2026-11-01

First public release: an open-source desktop studio and Python library for aerodynamics and
heat transfer in 2D and 3D, with a Windows installer. Every validation number below is taken
from `docs/validation.md`, where each check is listed with its reference, its tolerance band
and the run that measured it. Two checks fail and are listed as failing: the square cylinder's
drag at 32 cells/D (expected for that grid; it passes at 48 cells/D) and the plane jet's
momentum-flux drift.

### 2D Navier–Stokes solver (`aerolume.solvers.ns2d`)
- Incompressible 2D Navier–Stokes on a uniform staggered grid in PyTorch, GPU (CUDA) or CPU,
  float32 or float64: third-order upwind or central advection, SSP-RK3 time stepping, exact
  projection with a direct DCT/FFT Poisson solver, convective outlet, slip, no-slip or moving
  (rolling-road) walls, parabolic inflow for channel benchmarks.
- Immersed bodies by Brinkman volume penalisation with smoothed masks from a differentiable
  signed distance; rotating bodies (rolling wheels); several bodies at once.
- LES closures: Smagorinsky, WALE and Vreman.
- Forces from the penalisation impulse; surface c<sub>p</sub> and c<sub>f</sub>; mean and RMS
  fields; CUDA-graph capture of 10-step blocks (8× faster than eager at 1024×512 under WDDM).
- Differentiable functional core (`step`, `build_geometry`, `rollout`) with checkpointing.
- Validation: cylinder Re 40 and 100 grid study, Taylor–Green second-order convergence,
  divergence at round-off, Schäfer–Turek 2D-1 / 2D-2 (inside the benchmark bands or within
  0.5 % at 80 cells/D) and the square cylinder at Re 100.

### 3D Navier–Stokes solver (`aerolume.solvers.ns3d`)
- Incompressible 3D Navier–Stokes on GPU or CPU with immersed STL/OBJ/PLY geometry, a
  symmetry plane, moving road and rolling wheels, and WALE LES. VTK 3D views show smoke,
  streamlines and slices.
- Validation: sphere Re 100 at 24 cells/D, C<sub>d</sub> 1.148 (band 1.05–1.15, 2.85 %
  blockage) and wake length L/D 0.894 (band 0.80–0.95). Sphere Re 300, St 0.135 (−1.5 %
  against Johnson & Patel). Half domain equals full domain to 2.6 × 10⁻⁷ in C<sub>d</sub>.
  3D Taylor–Green observed order 1.97 / 1.99 / 2.00.

### Placed physics: smoke, heat, fans, jets and rooms (DESIGN section 10)
- Smoke emitters: up to four dye channels and GPU tracer particles (streaklines and
  timelines), steady or pulsed. They are passive. Boundedness and mirror symmetry are checked
  (half against full domain to 4.4 × 10⁻¹⁵).
- Heat transfer: an energy equation with Boussinesq buoyancy; heated, cooled, heat-flux and
  adiabatic bodies and walls; volumetric heat sources; Nusselt number and heat rate per body.
  - De Vahl Davis cavity Ra 10³–10⁶: mean Nu 1.118, 2.251, 4.531 and 8.841 against 1.117,
    2.238, 4.509 and 8.817 (+0.1 to +0.6 %).
  - Cubic cavity Ra 10⁴ (48³): Nu 2.066 (published 2.054–2.100).
  - Rayleigh–Bénard onset: Ra<sub>c</sub> 1701 (exact 1707.76, −0.4 %).
  - Heated cylinder Re 20 and 40: Nu 2.503 and 3.336, which is 2.2 % and 2.5 % above the
    published ranges at 5 % blockage. Heated sphere Re 100: Nu 7.03 (correlations 6.62–7.35).
  - Heated sphere in a closed box, natural convection at Ra 10⁴: Nu 6.385 at 27 cells/D against
    6.545 from Churchill's (1983) correlation (−2.4 %), on a monotone 12 / 18 / 27 cells/D grid
    study (GCI 2.9 %).
  - Grid convergence (Ra 10³ cavity): observed order 2.03. The 128² value lies at 0.69 of the
    64² GCI half-width.
- Fans (2D actuator strip, 3D actuator disc, set by thrust or target velocity): disc velocity
  within −0.34 % to +0.45 % of confined momentum theory; the set thrust is delivered to
  3 × 10⁻⁸.
- Ports: jets and vents. Plane jet Re 25: centreline decay −4.7 % and half-width −3.9 %
  against Bickley. The momentum-flux check fails; see Known limitations.
- Closed rooms (`domain_mode = "closed"`) with walls that can be heated or cooled.
- What the heat numbers rely on, each checked by a test (`tests/test_physics_round5.py`):
  buoyancy is added after the body penalisation in 2D and 3D, so a uniformly warm fluid around
  a body stays at rest and the body feels exactly the Archimedes force (2D cylinder: C<sub>l</sub>
  −1.5718 against −1.5708, |u| 4 × 10⁻⁷); heaters and heat-flux skins bound the time step from
  rest (a 3D heater-only room diverged on its second step without it); the uniform leak that
  balances a closed box with unbalanced jets and vents carries its heat and smoke out (θ stays
  within its bounds and the mean follows 1 − e<sup>−Qt/V</sup>: 0.4505 against 0.4504 in 3D); in
  a capped-Reynolds run the heat rate is the capped flow's Nusselt number times the real
  conductivity; box heaters placed on a body turn with it at an angle of attack or yaw.
- Live editing while a run is going: draw and erase obstacles (Undo restores a checkpoint of
  the drawing: 1.4 ms at 20 strokes on the room grid), and move emitters, fans and
  heaters. AeroJAX-style diagnostics: divergence, field changes, CFL number, residual, a
  per-stage solver trace, an optional PID-controlled time step, and CSV export.

### Airfoils and finite wings
- XFOIL-class viscous–inviscid airfoil polars built on vendored mfoil (MIT, K. Fidkowski) with
  robustness patches, warm-started sweeps, e<sup>N</sup> or forced transition and Kármán–Tsien
  compressibility; agrees with XFOIL 6.99 to 0.012 in c<sub>l</sub> on identical geometry.
  Validated against Ladson, Gregory & O'Reilly and Abbott & von Doenhoff
  (docs/validation_airfoil.md).
- Finite wings by a vortex lattice or a nonlinear lifting line on the viscous section polars:
  lift, induced and profile drag, span loading and a stall-onset estimate. The 45° swept wing of
  Weber & Brebner is within −3.2 to +3.9 % in C<sub>L</sub>; C<sub>L,max</sub> is not
  validated (docs/validation_wing.md).

### Simulation layer, sensitivity and optimisation
- Automatic planning from the design and three quality presets: domain sized for blockage,
  resolution in cells per chord or body height, thin-feature checks, FFT-friendly grids within
  a per-device cell budget, real or capped Reynolds number, laminar or LES, a run-time estimate
  that learns per kind of case, and a written rationale. For heat transfer the planner chooses
  the temperature and speed scales and states Ra, Gr, Pr, Ri and Pe with the rule behind each.
- Production runs with live frames, MSER transient detection, an early stop once the mean
  coefficients (or the Nusselt number) are known to the preset's tolerance, honest "not
  converged" warnings, cancel with a kept partial result and readable GPU errors.
- Error bars come from batch means; too few independent samples are flagged, replaced by a
  conservative estimate and never used for a significance test. Grid studies give a Roache
  grid-convergence index.
- Explore mode: a real-time session whose angle, speed, road and view change while it runs.
- Surface sensitivity maps dJ/dn for C<sub>d</sub>, C<sub>l</sub>, L/D or C<sub>m</sub> by
  reverse-mode differentiation, checked against central finite differences (cylinder: 4 × 10⁻⁶
  to 2.7 × 10⁻⁵ relative), in 2D and on imported 3D meshes. CST airfoil optimisation with the
  polar solver; gradient-based CFD shape changes (experimental).

### Desktop application
- PySide6 workflow Geometry → Setup → Simulate → Results → Compare → Report with a project
  tree, property grids and dockable panels with saved workspaces (Default, Analysis, Live,
  Scripting); a Start page with study types (airfoil, 2D vehicle, 3D body from a mesh, wing,
  heated room, electronics enclosure) and seven example projects with finished results.
- Geometry: NACA 4/5-digit sections, airfoil .dat (Selig, Lednicer), SVG, DXF, CSV points,
  image silhouettes, 15 parametric templates, STL/OBJ/PLY meshes with orientation, units and
  repair checks; imported files are recorded with their size and SHA-256.
- Results quote every value rounded to its standard error (Particle Data Group rule), "steady"
  for a settled series and "≈0" inside the error bar; the line a finished run writes to
  Messages, the Assistant, the MCP server and the scripting API's printed results use the same
  rule (`aerolume.core.format`). The Results header names the share of the heat input still
  being stored in the fluid beside a *Converged* verdict. Exports to CSV, NPZ, VTK and PNG; PDF and HTML reports with the methodology and the
  validity of every number.
- Interface (DESIGN section 13): semantic colour tokens with measured WCAG contrast in the
  light and dark themes, Atkinson Hyperlegible Next and Mono fonts (OFL). Keyboard: Tab goes
  step bar, page, then panels; F6 / Shift+F6 cycle the regions; every control is at least
  24 px; Ctrl+Tab leaves the Script editor.
- The look follows the website (docs/ui_guidelines.md, v4): its monochrome palette (ink on
  paper, and near-black in the dark theme), 6 px controls and 8 px frames, a filled-ink primary
  button and outlined secondary ones, ink links, tabs, checked icons and focus ring, and its
  flow motif, still, beside the product name on the Start page. The app icon is the website's
  mark. Plots, colour maps, status colours and the blue marks in the data views are unchanged;
  every text pairing still meets WCAG AA in both themes (73 measured rules over 68 colour pairings).
- Units: "2,5" reads as 2.5 and an ambiguous "1,500" is refused; hand-set grids that cannot fit
  the device's memory are refused. A damaged design or simulation file no longer stops a
  project from opening, and a result that cannot be saved is retried, never lost.

### Windows installer and packaging
- Windows installer (Inno Setup, per user, `.aerolume` file association) and a portable
  folder. Both are built by `packaging/build.py` from a snapshot of the sources and smoke-tested
  headless before the installer is compiled. The installer is checked file by file (size and
  SHA-256) against the tested folder. See docs/install.md.
- Single instance, Windows taskbar identity, crash reports with home paths redacted,
  `Aerolume.exe --selftest`, and `build.py --status`, which says whether a build matches the
  sources. A folder that fails its smoke test is reported as failed and no installer is left
  for it.
- The `licenses` folder of the build holds the licence files of every bundled Python package.
  Nested ones keep their folders, so PyTorch's third-party licences are all there. It also
  holds the LGPL-3.0 and GPL-3.0 texts for Qt, CPython's licences, notices for the
  proprietary NVIDIA CUDA/cuDNN and Intel OpenMP libraries that PyTorch brings, the licence
  texts of the 20 libraries VTK builds into its own DLLs (zlib, libpng, libjpeg-turbo, libtiff,
  FreeType, expat, lz4, pugixml, JsonCpp, ...) and Qt's attribution of the third-party code it
  contains (HarfBuzz, FreeType, libpng, md4c, PCRE2, ...) and of Mesa's llvmpipe in
  `opengl32sw.dll`. See THIRD_PARTY_NOTICES.md.
- `aerolume-cli` reports the heat-balance residual as a share of the heat passing through the
  domain, as the Results page does: a wall-heated cavity's round-off residual read "118 % of the
  largest heat flow" against its walls' net term of ~0.
<!-- build-summary:begin -->
- Release build, measured by `packaging/build.py` (docs/install.md has the table of every run): the app folder is 3.54 GB in 4,241 files and `Aerolume-0.1.0-cuda-setup.exe` is 1.94 GB. In the full build `d7b30ff02f9f`, the last recorded run, the bundled self-test passed all 14 checks in 20.1 s on the first launch and 6.4 s on the second. The full build `d7b30ff02f9f` took 8.4 min. Times depend on the machine's load.
<!-- build-summary:end -->

### Accounts (optional; `aerolume.accounts`, docs/accounts.md)
- A Supabase backend over its REST APIs, with no new dependency. Sign in by email and
  password, email link or 6-digit code, or GitHub / Google through the system browser (PKCE).
  Profiles, cloud copies of projects with conflict detection by content hash, and expiring
  read-only share links.
- Off by default. Nothing is sent anywhere until you connect a Supabase project you own.
  Browser sign-in waits on a one-time loopback path, so another web page cannot end a pending
  sign-in through the fixed redirect port.
  Migrations with row-level security are in `supabase/`. The client is tested against an
  in-process mock server (`python -m aerolume.accounts.mock`), not against a live project.
- Projects larger than Supabase's 50 MB file limit upload in parts (45 MB each) and are
  reassembled and checksum-verified on download; the Cloud projects window shows the packed
  size, parts and largest runs before uploading.

### Scripting, Assistant and MCP
- `import aerolume as al`: a scripting API (`aerolume.api`, docs/scripting.md) that also runs
  in the in-app Script console, with referenced cases (`al.cavity_case`,
  `al.heated_cylinder_case`, `al.fan_room_case`) that check a run against its benchmark.
- Experiment Assistant using Claude (anthropic SDK, your own key) or a local Ollama model.
  Suggestions are cards that change nothing until applied; numbers and actions in its answers
  are checked against the tool calls it made (docs/ai_assistant.md).
- MCP server (`aerolume-mcp`, docs/mcp.md) so Claude Desktop, Claude Code and other MCP
  clients can drive studies on your computer. `confirm: true` counts only for a call the server
  has already listed to the client, and `--read-only` opens existing projects only and writes
  nothing into them.
- Approvals you can rely on: a script's approval card wraps every line and shows long runs of
  spaces and invisible characters (bidirectional overrides, zero-width marks) as visible
  markers with a warning; a script that changes the approval rules has them put back and is
  reported as an error; digits inside names someone typed (a design called "NACA 2412 (Cd
  0.0061)") are never evidence in the claims check.

### Command line (`aerolume-cli`)
- `doctor`, `bench`, `polar`, `quick`, `quick3d`, `cavity`, `wing`, `run`, `script` and
  `validate`; Ctrl+C stops a run and keeps the partial result.

### Documentation and validation
- User guide, simulation guide, installation guide, scripting, Assistant, MCP and accounts
  guides; theory documents for the 2D and 3D solvers, airfoils, wings and sensitivity maps;
  design guidelines; a consolidated validation report whose results section is written by
  `python validation/run_all.py --all --write-docs`, with a digest of the code it measured.
- Project website generated from the repository (`website/build.py`).

### Known limitations
- **Square cylinder drag at 32 cells/D fails its band.** C<sub>d</sub> is 1.582, +5.9 % against
  the ±5 % band around Sohankar et al. (1.403–1.569). The one-cell diffuse interface at sharp
  corners makes drag converge from above; at 48 cells/D it is 1.549 (+3.7 %) and passes. Use at
  least 48 cells across sharp-edged bodies when absolute drag matters (docs/validation.md,
  sections 3 and 9).
- **Jet momentum conservation is not validated.** A jet blowing into a closed box loses 6.2 %
  of its core momentum flux between 8 and 24 port widths, against a 5 % band. Three identical
  runs gave the same 6.158 %. The closed box's return flow is the likely cause, but the test
  that would separate confinement from discretisation has not been run
  (docs/validation.md, section 10).
- **High Reynolds numbers.** There is no wall model and no transition model in the
  Navier–Stokes solvers, so absolute drag and lift of vehicles are not validated; compare
  designs run the same way. 2D vehicles are infinitely wide side profiles, and grids are
  uniform (no local refinement). These three are planned for 0.2 (see *Unreleased*).
- Heat transfer is validated for laminar natural and forced convection at Pr 0.71 only. Not
  validated: turbulent convection, other Prandtl numbers, mixed convection around a heated
  body (a run says so in its warnings), and heat-flux bodies or heat sources against data.
  Radiation and conjugate heat transfer are not modelled.
- In a 2D closed room the fan's speeds, disc velocity and volume flow are qualitative and
  grid-dependent (theory_ns2d.md section 18.4); the thrust is exact. Use 3D for rooms.
- The Lattice-Boltzmann solver and neural-operator surrogates are not in this release.
- Only Windows has an installer. The CUDA build is large (see docs/install.md); on other
  systems, install from source. Linux and macOS have not been tested.
