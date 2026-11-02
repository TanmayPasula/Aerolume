"""Automatic simulation set-up: from a Design + Simulation to a ready solver config.

This is the AirShaper-style automation that lets a non-expert press *Run*: the
planner chooses the domain, the grid, the Reynolds-number treatment, the run
length and the averaging window, estimates the run time, and writes down *why*
in plain language (``SimPlan.notes``) together with honest caveats
(``SimPlan.warnings``). The Setup page shows those lists and ``summary()``
verbatim, so every sentence here is user-facing.

Decisions, in order (each one is explained in a note at run time):

1. **Wind frame.** Flow comes from the left along +x; the angle of attack (or
   vehicle pitch) is applied by rotating the geometry by -alpha about
   ``Design.moment_ref`` (DESIGN.md 3). With a road, wheels stay on the road and
   only the other bodies pitch.
2. **Domain** from the geometry class: airfoils get 2-3 chords upstream, 5-8
   downstream and a height that keeps blockage under ~5 %; vehicles sit on the
   road (a moving belt or a fixed floor) with 1.25-2 lengths upstream, 3-5
   downstream and as much headroom as it takes to keep the blockage (height of
   the vehicle above the road / height of the domain) at 8 % for Draft and 5 %
   for Standard and Fine -- in 2D the air cannot pass around the sides, so a
   low ceiling speeds the flow over the roof up and inflates drag and lift.
   Wings on a road (front wings) are laid out like airfoils. Other bodies are
   treated like bluff bodies, scaled by their frontal size, with 8-12 sizes
   upstream and 16-24 downstream (an inlet 4 sizes upstream measurably raised
   the cylinder's drag).
3. **Resolution**: a preset per quality and device -- cells per chord for
   airfoils, per vehicle *height* for vehicles (the wake and the separated
   shear layers scale with it), per frontal size for bluff bodies -- raised so
   that thin features (airfoil thickness, ground clearance) get 4-8 cells, then
   fitted into a cell budget per device (GPUs ~0.25-2 M cells, CPUs ~0.1-0.3 M)
   by first shrinking the domain toward its minimum (never past the blockage
   limit) and only then coarsening the grid. The grid is uniform, so the tall
   vehicle domains cost cells far from the car; the vehicle presets are chosen
   so that a Draft car still needs only about a minute of solver time on a
   mid-range GPU (the sedan: ~29 000 steps at ~430 steps/s on an idle RTX
   4060 Ti). Grid sizes are rounded to FFT-friendly numbers (the pressure
   solve is a DCT) and then fitted back into the budget.
4. **Reynolds number**: "real" runs at the real Re; when the grid cannot
   resolve the laminar boundary layer the flow model is a WALE large-eddy
   simulation with upwind-biased advection, otherwise laminar (central
   differences when the cell Reynolds number is small enough). "capped" runs at
   the largest Re the grid resolves (or the user's cap).
5. **Duration**: enough convective times for the start-up transient to leave
   and several shedding cycles to be averaged; the runner stops earlier once
   the mean drag is known to the quality's tolerance (Draft ±5 %, Standard
   ±3 %, Fine ±2 %, one standard error) and the mean lift to twice that --
   see :mod:`.runner`.
6. **Run-time estimate**: steps (duration over the CFL time step, with the
   speed factor (max|u|+max|v|)/U of the geometry class) over the solver
   throughput, divided by the share of the loop spent stepping (the app's live
   view costs ~20 %), plus build and output time. Throughput, speed factor and
   loop share start from values measured on the reference machine and are
   corrected by measurements of this machine *near the planned grid size*:
   every run and ``aerolume-cli bench`` record them, and with
   :func:`use_calibration_file` (the app and the CLI use the user config
   folder) they survive restarts.

Everything here is cheap (no solver construction, no GPU work): the Setup page
calls :func:`plan_simulation` on the GUI thread whenever a setting changes.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..core import device as devmod
from ..core import units
from ..geometry import polygon as P
from ..model import Body, Design, Fluid, Simulation
from ..solvers.ns2d.config import ImmersedBody, SolverConfig
from .defaults import AIRFOIL_CAP_RE, default_cfd_settings
from .defaults import geometry_class as _geometry_class
from .physics import (
    PhysicsScales,
    Placed,
    PlacedPhysicsPlan,
    cavity_nusselt,
    closed_box,
    draft_buoyant_duration,
    has_placed_physics,
    merge_warnings,
    peak_source,
    peak_speed,
    physics_cost,
    physics_dt,
    physics_rows,
    physics_scales,
    placed_physics_2d,
    plan_monitors,
    room_2d_fan_warning,
    thermal_checks,
    unbounded_heating_note,
)

__all__ = [
    "CALIBRATION_FILE", "SimPlan", "calibration_kind", "default_cfd_settings", "estimate_steps_per_second",
    "fft_friendly", "fmt_duration", "fmt_estimate", "focus_window", "loop_efficiency", "plan_live", "plan_simulation", "record_throughput",
    "reset_calibration", "rotate_points", "use_calibration_file", "wind_frame_bodies",
]

# ---------------------------------------------------------------------------
# tunables (each one justified where it is used)
# ---------------------------------------------------------------------------

#: Largest blockage (frontal height / domain height; with a road: height above
#: the road / domain height) per geometry class and quality (Draft, Standard,
#: Fine). 2D blockage is severe -- nothing can flow around the sides -- and the
#: wake-blockage estimate eps ~ (Cd/4)(h/H) (Allen & Vincenti 1944) puts the
#: dynamic-pressure error at ~2 % for a car at 8 % and ~1 % at 5 %. Airfoils
#: and bluff bodies are cheap enough to always get 5 % (the validated value).
BLOCKAGE_MAX = {
    "airfoil": (0.05, 0.05, 0.05),
    "vehicle": (0.08, 0.05, 0.05),
    "generic": (0.05, 0.05, 0.05),
}
#: Explore mode trades a little blockage for a smaller, faster grid.
BLOCKAGE_LIVE = 0.10
#: Backwards-compatible name: the free-flight blockage of airfoils and bluff bodies.
BLOCKAGE_TARGET = 0.05
#: Cells a laminar boundary layer must span to count as resolved.
BL_CELLS_MIN = 4.0
#: Central differences only below this cell Reynolds number U h / nu (classic
#: cell-Peclet bound for wiggle-free central advection-diffusion).
CENTRAL_CELL_RE = 2.0
#: Solver CFL number (the value the solver was validated with).
CFL = 0.5
#: The sub-grid model used for under-resolved runs (validated at Re 1e6 with a
#: moving road, docs/theory_ns2d.md section 11).
LES_MODEL = "wale"
#: Fewer cells than this across a gap or a solid feature means it is closed or lost
#: (the diffuse wall is about two cells wide).
CLOSED_GAP_CELLS = 2.5
#: Solver memory per grid cell [bytes] by precision: the GPU allocator's peak for a
#: cylinder run with WALE (float32: 302 B at 1024 x 512, 265 B at 2048 x 1024, CUDA
#: graph pools included; float64: 397 B at 1024 x 512) and the CPU process's private
#: bytes (float64: 455-498 B), rounded up. Used only to refuse grids that cannot fit.
BYTES_PER_CELL = {"float32": 320, "float64": 560}
#: A plan whose solver would need more than this share of the device's memory (GPU
#: memory, or the computer's RAM for a CPU run) is refused with a ValueError.
MEMORY_FRACTION = 0.8
#: Live (explore) sessions cover pitch angles within +-this many degrees without
#: re-planning; the solver's live body window uses the same range. On a road the
#: range is much smaller: a car pitched more than a few degrees hits the road.
LIVE_PITCH_DEG = 25.0
LIVE_PITCH_GROUND_DEG = 5.0
#: Speed range of the explore slider relative to the planned speed (Simulate page).
LIVE_SPEED_RANGE = (0.25, 2.0)
#: Solver threads on CPU. On the development machine (hybrid 8P+8E cores) 4
#: intra-op threads ran a 512 x 256 grid 1.4-2x faster than the default 16: each
#: elementwise op on such a field takes ~10 us, comparable to OpenMP fork/join.
#: (Used by live.cpu_threads; defined here so the plan can name it.)
CPU_SOLVER_THREADS = 4

QUALITY_ORDER = ("draft", "standard", "fine")

# Resolution presets per geometry class, device class and quality:
# * airfoil: cells per chord;
# * vehicle: cells per vehicle *height* (road to roof with a road, else the
#   frontal height). The separated shear layers and the wake scale with it, and
#   with a uniform grid it is what the tall, blockage-limited domain can afford;
#   for a 4.7 m x 1.45 m sedan, 36 / 48 / 64 per height are 117 / 156 / 207 per
#   length (the ground-clearance rule below usually raises them further);
# * generic: cells per *frontal size* s (see _sizes); the cylinder validation
#   converges at 32 cells per diameter (Cd within 1 %, St within 0.1 %).
CELLS_PER_SIZE = {
    "airfoil": {"gpu": (64, 128, 256), "cpu": (40, 64, 96)},
    "vehicle": {"gpu": (36, 48, 64), "cpu": (24, 32, 40)},
    "generic": {"gpu": (24, 40, 64), "cpu": (16, 24, 32)},
}
# Cell budgets [cells] per device class and quality: what fits in memory and
# finishes in minutes (GPU: ~350 steps/s at 0.3 M cells; CPU: ~25 steps/s at 0.15 M).
# Budgets are deliberately *not* tied to measured speed, so the same design gets
# the same grid on every run (comparisons between designs need that).
# Draft is what a first-time user runs, so it has to come back in well under a
# minute on a mid-range GPU: a run that sits at a low percentage for minutes
# reads as "the program does nothing". 250 k cells keep every airfoil and bluff-
# body template's Draft run at or under ~35 s (estimate) on an RTX 4060 Ti; the
# Re-100 cylinder measured 8 s and NACA 2412 at Re 1e6 5-7 s.
CELL_BUDGET = {
    "gpu": (250_000, 1_000_000, 2_000_000),
    "cpu": (100_000, 200_000, 300_000),
}
#: Vehicles keep a larger GPU Draft budget (see :func:`_cell_budget`): their domain
#: height is fixed by the blockage limit, so a smaller budget can only shorten the
#: road ahead and the wake behind the car -- and that moved the Draft sedan from
#: Cd 0.6 / Cl 2.5 (324 k cells, 1.25 L of road ahead, 3 L of wake) to Cd 0.95 /
#: Cl 1.4 (249 k cells, road and wake 0.6 L shorter) in two measured runs of 20
#: and 30 L/U: a systematic shift, not noise. The sedan's Draft run takes ~75 s
#: on the RTX 4060 Ti at this size.
VEHICLE_DRAFT_GPU_BUDGET = 400_000
LIVE_BUDGET = {"gpu": 300_000, "cpu": 45_000}
# Thin features (airfoil thickness, ground clearance) get at least this many cells:
# 4 keeps a car's underbody channel open at Draft (the diffuse wall takes ~1 cell
# on each side), 6-8 resolve its flow at Standard / Fine.
THIN_CELLS = (4.0, 6.0, 8.0)
LIVE_THIN_CELLS = 3.0
#: Cells across a car's ground clearance below which the underbody flow (and the
#: suction under the car that sets its lift) counts as under-resolved. Draft's 4
#: keep the channel open but leave ~2 cells of flow between the two diffuse walls;
#: 8 (the Fine target) resolve it. Raising Draft to 8 would cost ~8x the run time
#: (4x the cells, 2x the steps), so Draft says so instead (see _underbody_warning).
UNDERBODY_CELLS = 8.0
# AIRFOIL_CAP_RE (when a new airfoil simulation starts capped) lives in .defaults,
# with default_cfd_settings, so the GUI can use them without importing torch.
#: A thin feature, or the grid as a whole, is reported as coarser than intended
#: only below this fraction of its target: fitting the FFT-rounded grid into the
#: budget may cost a few per cent of resolution, which is not worth a warning.
THIN_SLACK = 0.9

# Domain factors (preferred per quality, minimum) per geometry class:
#   airfoil: multiples of the chord; height also >= frontal height / blockage
#   vehicle: up/down in vehicle lengths; the height comes from BLOCKAGE_MAX only
#            (it is never shrunk below that)
#   generic: multiples of the frontal size s
DOMAIN = {
    "airfoil": {"up": (4.0, 4.0, 5.0), "down": (5.0, 6.0, 8.0), "height": (6.0, 6.0, 6.0),
                "min": {"up": 2.0, "down": 5.0, "height": 3.0}},
    "vehicle": {"up": (1.25, 1.5, 2.0), "down": (3.0, 4.0, 5.0), "height": (0.0, 0.0, 0.0),
                "min": {"up": 1.0, "down": 2.5, "height": 0.0}},
    "generic": {"up": (8.0, 10.0, 12.0), "down": (16.0, 20.0, 24.0), "height": (10.0, 12.0, 16.0),
                "min": {"up": 4.0, "down": 8.0, "height": 8.0}},
}
# Why airfoils get 4 chords upstream and 6 of height: the frictionless walls and the
# uniform inlet bias a lifting section's forces. NACA 2412 at 4 deg, Re 6,400, 64 cells
# per chord, 20 L/U (2026-10, RTX 4060 Ti; Cl known to about +-0.002 per run), as
# (upstream, downstream, height) in chords -> Cd, Cl, Cm:
#   (2, 5, 4) the old Draft domain   0.0990 0.1481 -0.0244
#   (4, 5, 6) Draft now               0.0971 0.1462 -0.0241
#   (10, 5, 5)                        0.0973 0.1447 -0.0235
#   (4, 5, 16)                        0.0962 0.1414 -0.0229
#   (6, 8, 8)                         0.0956 0.1402 -0.0228
#   (10, 15, 16) reference            0.0953 0.1395 -0.0228
# The height matters most, then the inlet. (6, 8, 8) needs 491 k cells at Draft (twice
# the budget) and would leave Standard, which has to shrink into its own budget, with a
# smaller domain than Draft -- and a quality step must refine the grid of the same case.
# So Draft and Standard share (4, 5-6, 6) (246 k and ~1 M cells) and the remaining bias,
# about +2 % in Cd, +5 % in Cl and +6 % in Cm, is stated in every airfoil plan
# (AIRFOIL_DOMAIN_NOTE); Fine still shrinks toward the minimum to fit its budget.
# Why bluff bodies get long domains: the cylinder at Re 100, Draft (24 cells/D, 5 %
# blockage) measured Cd 1.534 / St 0.177 with the inlet 4 D upstream and the outlet
# 8 D downstream, 1.469 / 0.172 at 6 / 12 D and 1.427 / 0.169 at 10 / 20 D (the
# validation value, theory_ns2d.md section 11). Their grids are cheap, so the
# presets afford 8-12 D upstream; the budget shrinks the domain first on a CPU.
# Run length (t_end, average_from) in convective times. Airfoils and vehicles in
# L/U; generic bluff bodies in s/U because their shedding period scales with the
# frontal size (cylinder: shedding locks in by ~40 D/U, period ~6 D/U). A car's
# wake sheds at St ~0.2 on its height, a period of ~1.5 L/U for a sedan, so the
# Draft window (t = 6 ... 15) already spans about six cycles. The Draft cylinder
# at Re 100 was measured both ways: t_end 40 / averaging from 10-16 left only 13
# effectively independent samples after the slow start-up of shedding near onset
# ("not converged"), while 60 / 30 stopped early, converged, at t = 54 after 8.4 s
# of wall time -- so Draft keeps 60 s/U for bluff bodies; it is cheap.
# Airfoils run long enough for the capped flow every new airfoil simulation starts
# with (Re 2,500, simulation.defaults): its trailing-edge shedding settles slowly, and
# NACA 0012, 2412 and 4412 at 4 deg were all "not converged" (7-8 effective samples)
# at the old Draft 20 L/U, while with room to run they stopped early, converged, at
# t = 30.5-30.7 after 10-17 s of wall time (Draft, RTX 4060 Ti). Runs that settle
# sooner still stop early, a little later than before because the early stop waits for
# MIN_AVG_FRACTION of the planned averaging window (runner): NACA 0012 at Re 1,000 at
# t = 18.4 in 6.5 s (was 11.6 in 4.6 s).
DURATION = {
    "airfoil": ((40.0, 8.0), (50.0, 12.0), (60.0, 16.0)),
    "vehicle": ((30.0, 6.0), (36.0, 8.0), (40.0, 10.0)),
    "generic": ((60.0, 30.0), (80.0, 40.0), (120.0, 50.0)),
}
#: Longest Draft run in convective times (L/U). Bluff-body durations scale with
#: s / L, so a body much taller than its reference length would otherwise get a
#: Draft run of 100+ L/U; the runner still stops earlier once the mean drag is
#: known to +-5 %, and Standard / Fine keep the long windows for converged means.
DRAFT_T_END_MAX = 60.0
# Why vehicles run that long: a 2D car wake at Re ~1e7 is chaotic, not periodic.
# Measured on the Draft sedan (moving road, t = 0 ... 30): Cd fluctuates by
# +-40 % of its mean with an integral time scale of ~0.26 L/U, and big vortex
# events modulate Cl for the first ~15 L/U. The standard error of the mean is
# ~ rms * sqrt(2 tau / T), so +-5 % needs ~15 L/U of averaging and +-1 % would
# need ~400 -- hence the per-quality targets below, and the honest "not
# converged" verdict when a run ends short of them.
#: Convergence target per quality: the run stops early once the standard error of
#: the mean Cd is below this fraction of the mean, that of the mean Cl below
#: CL_RTOL_FACTOR times it, and nothing drifts; CFDResult.converged means exactly
#: that. Lift gets the looser target because in separated 2D flows it fluctuates
#: far more than drag (Standard sedan: Cl rms 54 % of the mean, Cd 31 %), and the
#: early stop is about the drag.
CONVERGENCE_RTOL = (0.05, 0.03, 0.02)
CL_RTOL_FACTOR = 2.0
# (max|u| + max|v|) / U that sets the CFL step: measured ~1.8-2.0 for laminar bluff
# bodies, higher for lifting airfoils (suction peak), 3.6-3.9 for the Draft sedan
# (LES, Re 1e7) and up to 5-6.5 for finer 2D LES grids, where compact 2D vortices
# reach 3-4.7 U. Refined by measurements (record_throughput).
SPEED_FACTOR = {"laminar": 2.2, "les": 4.5}
#: ... and per geometry class, from full runs on the reference machine (2026-10,
#: mean over the run): airfoils 2.3 (NACA 2412, Re 1e6, 4°) to 2.9 (8°) with LES
#: and 1.8 (NACA 0012, Re 1000, 5°) to 2.7 (20°) laminar; the cylinder 2.0 at
#: Re 100 and 4.0 with LES at Re 7e4 at both Draft (23 cells/D) and Standard
#: (40 cells/D: 3.5 while the wake forms, 4.0-4.3 once it sheds); the Draft sedan
#: 3.9. The airfoil values lean high so that a stalled section is not under-
#: estimated. SPEED_FACTOR remains the fallback for classes without a value.
SPEED_FACTOR_KIND = {
    ("airfoil", "les"): 2.8, ("airfoil", "laminar"): 2.3,
    ("generic", "les"): 4.1, ("generic", "laminar"): 2.0,
    ("vehicle", "les"): 4.0,
    # closed boxes (CLOSED_SPEED_FACTOR explains the values)
    ("closed-natural", "laminar"): 0.5, ("closed-natural", "les"): 0.6,
    # a box warmed by a heater runs on the heater's plume: the Draft heater-only 4 x 3 m room (300 W/m,
    # sealed, LES, 96 x 72) measured 2.35 of its free-fall velocity, two runs agreeing to 0.3 %
    # (fix round 3.4, 2026-10); laminar not measured, taken the same
    ("closed-natural+heater", "laminar"): 2.4, ("closed-natural+heater", "les"): 2.4,
    ("closed-forced", "laminar"): 1.5, ("closed-forced", "les"): 1.5,
    # relative to the peak speed (physics.peak_speed, twice the fan's disc velocity): the Draft
    # electronics enclosure (2 m/s fan, 1.5 m/s jets, laminar) measured (max|u| + max|v|) of 4.4 -
    # 6.5 times the fan speed over its run, 3.1 of the peak; the Draft 2D fan room (1 N/m thrust,
    # LES) 10.2 times the fan's momentum-theory disc velocity, 5.1 of the peak (2026-10)
    ("closed-fan", "laminar"): 3.1, ("closed-fan", "les"): 5.1,
    # per case signature (calibration_kind), measured on the reference machine (fix round 3.4, 2026-10):
    # the plain Draft fan room (1 N/m, LES) 5.09 of its peak, confirming the class value; the same room with
    # a supply jet, a vent and a 300 W/m heater only 3.87 (the jet and the vent bleed the recirculation);
    # the Draft electronics enclosure (board, chip, fan, intake, exhaust; laminar) 2.68
    ("closed-fan+jet+heat", "les"): 3.9, ("closed-fan+jet+heat+body", "laminar"): 2.7,
}
#: The CFL speed factor of a tunnel run whose placed physics outruns the free stream
#: (``speed_ratio`` > PEAK_RATIO_MIN), relative to that peak speed, per kind of source. A plume:
#: measured on the Draft heated cylinder (D 20 mm, Re 40, body 60 °C in 20 °C air, buoyancy on:
#: Ri 30, the buoyant velocity 5.4 U), (max|u| + max|v|) rose from 2.0 to 3.0 of it over 10 L/U,
#: harmonic mean 2.6 (2026-10); the plume, not the wake, sets the step. A jet or a fan's
#: slipstream: not measured yet -- the jet's own speed plus the cross-flow it entrains -- so an
#: estimate the first run of the class replaces.
PEAK_SPEED_FACTOR = {"plume": {"laminar": 2.7, "les": 3.0}, "jet": {"laminar": 1.5, "les": 2.0}}
#: Peak speed over U from which a tunnel run is planned (and calibrated) on its peak speed.
PEAK_RATIO_MIN = 1.05
#: Peak eddy viscosity of the WALE runs in units of U h (theory_ns2d.md section 11:
#: 3-4e-4 at h = 1/256), for the viscous time-step limit of the estimate.
NUT_PEAK = 0.1
#: The domain's own bias for airfoils (see the DOMAIN table's comment), stated with
#: every airfoil plan because the result's error bars do not include it.
AIRFOIL_DOMAIN_NOTE = (
    "The finite domain (frictionless walls, a uniform inlet) adds a bias the error bars do not include: "
    "measured on NACA 2412 at 4° against a domain 10 chords ahead and 16 chords tall, about +2 % in Cd, "
    "+5 % in Cl and +6 % in Cm at Draft. It is the same for designs run at the same quality, so "
    "differences between them are not affected.")
#: How to read vehicle coefficients (shown with every vehicle plan).
VEHICLE_2D_NOTE = (
    "Reading the numbers: this is a 2D side profile, an infinitely wide car. Air cannot escape around the "
    "sides, so the flow over the roof is faster, the wake sheds as a 2D vortex street and the underbody is "
    "a channel; Cd and Cl (per frontal height) therefore come out well above 3D wind-tunnel values. The "
    "sedan template on a moving road gives Cd ≈ 0.6 and Cl ≈ 2.5-2.8 at Draft and 0.72 and 2.2 at Standard, "
    "where a real saloon measures about 0.3 and 0.1-0.4. The quality preset itself moves the numbers by "
    "20-25 % (resolution, underbody cells, blockage), so rank and compare designs simulated at the same "
    "quality; do not quote them as absolute values.")


# ---------------------------------------------------------------------------
# throughput model (steps per second) -- calibrated, then corrected by measurement
# ---------------------------------------------------------------------------

# Reference throughput of FlowSolver.advance on the development machine
# (RTX 4060 Ti float32 with CUDA graphs; i7-13700F with 4 threads), cylinder
# benchmark, cells -> steps/s. Throughput is not a power law (launch overhead
# floors small grids, cache misses hurt big ones), so it is tabulated and
# interpolated in log-log. The GPU rows were re-measured on an idle GPU on
# 2026-10-31 (validation/records/planner-timing-2026-10-31.json holds the numbers and
# the procedure): 1.4-1.6 times the earlier rows, which had been taken while other work
# shared the GPU and made every first 2D estimate about 1.6 times too long (a review
# run of the cylinder took 0.61 of its estimate, the cavity 0.42). A shared GPU runs
# slower; each run's measured throughput corrects the estimate for this machine.
_REF_RATES: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("gpu", "upwind3"): [(32768, 1704), (73728, 1459), (131072, 1226), (294912, 767), (524288, 492),
                         (819200, 297), (1179648, 185), (2097152, 75)],
    ("gpu", "les"): [(32768, 1150), (73728, 977), (131072, 801), (294912, 495), (524288, 316),
                     (819200, 200), (1179648, 115), (2097152, 43)],
    ("gpu", "central"): [(32768, 2008), (73728, 1731), (131072, 1488), (294912, 975), (524288, 623),
                         (819200, 382), (1179648, 235), (2097152, 102)],
    # CPU, 4 threads, float64 (float32 measured ~1.25-2x faster, see _DTYPE_FACTOR)
    ("cpu", "upwind3"): [(32768, 185), (73728, 59), (131072, 34.5), (294912, 17.7)],
    ("cpu", "les"): [(32768, 101), (73728, 31.5), (131072, 21.7), (294912, 8.6)],
    ("cpu", "central"): [(32768, 205), (73728, 66), (131072, 39), (294912, 20)],
}
# float32 on CPU vs the float64 table; float64 on consumer GPUs (1/64-rate FP64,
# but the solver is mostly bandwidth-bound) vs the float32 table.
_DTYPE_FACTOR = {("cpu", "float32"): 1.6, ("gpu", "float64"): 0.35}

#: Share of a run's stepping-loop wall time spent advancing the solver; the rest
#: is the per-chunk force read-back, convergence checks and -- with the live view
#: the app always shows -- rendering and delivering frames (the runner caps the
#: render itself at RENDER_DUTY = 20 %). Measured on the reference machine with a
#: trivial frame consumer: 0.86 for the 1.0 M-cell Standard cylinder and 0.90 for
#: the Draft sedan; the app's own frame handling costs more (a Standard cylinder
#: run in the GUI implied ~0.72), so the live default sits between. Headless runs
#: (CLI, scripts) lose ~1-2 %. Refined by measurement like the throughput.
LOOP_EFFICIENCY = {"live": 0.8, "headless": 0.98}
#: Seconds the polar cross-check of a single airfoil adds after the run (runner).
POLAR_CHECK_S = 2.0
#: Real Reynolds number from which a single airfoil's CFD result is cross-checked
#: against the viscous polar (XFOIL-class methods are trustworthy from ~5e4 up;
#: below that the CFD resolves the boundary layer itself and is the better answer).
POLAR_CHECK_RE = 5.0e4

# Measured corrections. Each store maps a key to observations (ln size, ln value),
# newest last; _local() says how they are combined. "Size" is the cell count for
# the throughput and the loop efficiency, and cells per reference length for the
# speed factor (finer grids resolve more compact, faster vortices).
#: (device class, flavour, dtype) -> measured / model throughput ratio.
_RATE_CACHE: dict[tuple[str, ...], list[tuple[float, float]]] = {}
#: (kind, model) -> measured speed factor (max|u|+max|v|)/U.
_SPEED_CACHE: dict[tuple[str, ...], list[tuple[float, float]]] = {}
#: (device class, "live" | "headless") -> measured loop efficiency.
_LOOP_CACHE: dict[tuple[str, ...], list[tuple[float, float]]] = {}
#: The same three stores for 3D runs (:mod:`.plan3d`), keys prefixed "3d". They are
#: kept apart so that a 3D measurement never moves a 2D estimate (the 2D family
#: filters would otherwise pool them), and persisted in the same calibration file.
_RATE_CACHE3D: dict[tuple[str, ...], list[tuple[float, float]]] = {}
_SPEED_CACHE3D: dict[tuple[str, ...], list[tuple[float, float]]] = {}
_LOOP_CACHE3D: dict[tuple[str, ...], list[tuple[float, float]]] = {}
_CACHE_LOCK = threading.Lock()
#: Observations kept per key (older ones carry negligible weight anyway).
_KEEP = 16
#: Kernel width in ln(size): a measurement on a grid twice as large (or with
#: twice the cells per length) counts 61 % of one at the same size.
_KERNEL_WIDTH = math.log(2.0)
#: Weight of the machine-wide median against the local measurements: small, so
#: that a run near the planned size decides, but nonzero, so that a single run
#: at another size still tells the planner how fast this machine is overall.
_PRIOR_WEIGHT = 0.15
#: File name of the persisted calibration in the user config folder (the app
#: and the CLI share it).
CALIBRATION_FILE = "run_time_calibration.json"
#: Where measurements persist between sessions (None: this process only); see
#: use_calibration_file().
_CALIBRATION_FILE: str | None = None


def _device_class(device: str) -> str:
    return "cpu" if device.startswith("cpu") else "gpu"


def _flavour(advection: str, turbulence: str) -> str:
    if turbulence not in ("none", "laminar", ""):
        return "les"
    return "central" if advection == "central" else "upwind3"


def _model_rate(dev_class: str, flavour: str, dtype: str, cells: float) -> float:
    """Reference steps/s for a grid of ``cells`` (log-log interpolation of _REF_RATES)."""
    table = _REF_RATES[(dev_class, flavour)]
    xs = np.log([c for c, _ in table])
    ys = np.log([r for _, r in table])
    x = math.log(max(cells, 1.0))
    if x <= xs[0]:
        # small grids are launch/latency bound: throughput saturates
        y = ys[0] + 0.3 * (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    elif x >= xs[-1]:
        y = ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    else:
        y = float(np.interp(x, xs, ys))
    return math.exp(y) * _DTYPE_FACTOR.get((dev_class, dtype), 1.0)


def _observe(store: dict, key: tuple, size: float, value: float) -> None:
    """Append one measurement (the caller holds _CACHE_LOCK)."""
    obs = store.setdefault(key, [])
    obs.append((math.log(size), math.log(value)))
    del obs[:-_KEEP]


def _local(store: dict, key: tuple, size: float, family: Callable[[tuple], bool]) -> float | None:
    """The measured value for ``key`` at ``size``, or None without measurements.

    A single machine-wide correction (what this module kept before) cannot be
    right at every size: launch overhead dominates small grids and memory
    bandwidth large ones, so a ratio measured on a 0.13 M-cell airfoil said
    little about a 0.3 M-cell car. Here the observations of ``key`` are averaged
    in log space with Gaussian weights in ln(size) of width _KERNEL_WIDTH, each
    newer observation halving the weight of the older ones (so one odd run on a
    busy GPU does not swing everything), and shrunk with weight _PRIOR_WEIGHT
    toward the median of every observation of the keys ``family`` accepts.
    Without observations of ``key`` itself the result is that median, so a
    sibling flavour's measurement is still used. The caller holds _CACHE_LOCK.
    """
    pool = [v for k, obs in store.items() if family(k) for _, v in obs]
    if not pool:
        return None
    prior = float(np.median(pool))
    x = math.log(max(size, 1e-30))
    num, den = _PRIOR_WEIGHT * prior, _PRIOR_WEIGHT
    for age, (xs, v) in enumerate(reversed(store.get(key, []))):
        w = 0.5 ** age * math.exp(-0.5 * ((x - xs) / _KERNEL_WIDTH) ** 2)
        num += w * v
        den += w
    return math.exp(num / den)


def estimate_steps_per_second(device: str, cells: float, *, advection: str = "upwind3",
                              turbulence: str = "none", dtype: str = "float32") -> float:
    """Expected solver throughput [steps/s] on ``device`` for a grid of ``cells``.

    The reference model is corrected by the measured/model ratios of this
    device class and dtype, weighted toward grids of a similar size (see
    :func:`_local`). They come from :func:`record_throughput`, which every run
    and ``aerolume-cli bench`` call, so estimates converge to this machine's
    speed at the sizes it actually runs.
    """
    dc, fl = _device_class(device), _flavour(advection, turbulence)
    base = _model_rate(dc, fl, dtype, cells)
    with _CACHE_LOCK:
        corr = _local(_RATE_CACHE, (dc, fl, dtype), cells, lambda k: k[0] == dc and k[2] == dtype)
    return base * (corr if corr is not None else 1.0)


def loop_efficiency(device: str, cells: float, *, live_view: bool = True) -> float:
    """Share of the stepping loop's wall time that advances the solver (see
    LOOP_EFFICIENCY), measured on this machine near this grid size when available."""
    dc, mode = _device_class(device), ("live" if live_view else "headless")
    with _CACHE_LOCK:
        v = _local(_LOOP_CACHE, (dc, mode), cells, lambda k: k == (dc, mode))
    return float(np.clip(v if v is not None else LOOP_EFFICIENCY[mode], 0.2, 1.0))


def record_throughput(device: str, cells: float, steps_per_s: float | None, *, advection: str = "upwind3",
                      turbulence: str = "none", dtype: str = "float32", kind: str | None = None,
                      speed_factor: float | None = None, cells_per_ref: float | None = None,
                      loop_efficiency: float | None = None, live_view: bool = False) -> None:
    """Feed a run's measurements back into the estimates.

    ``steps_per_s`` is the solver's own throughput (stepping only);
    ``speed_factor`` the run's mean CFL speed factor (max|u|+max|v|)/U for
    geometry class ``kind`` on a grid with ``cells_per_ref`` cells per
    reference length; ``loop_efficiency`` the stepping share of the loop's
    wall time, with (``live_view``) or without frames. Missing, non-finite or
    non-positive values are ignored. With a calibration file in use
    (:func:`use_calibration_file`) the measurements are saved to it.
    """
    if not (math.isfinite(cells) and cells > 0):
        return
    dc, fl = _device_class(device), _flavour(advection, turbulence)
    model = "les" if fl == "les" else "laminar"

    def ok(v: float | None) -> bool:
        return v is not None and math.isfinite(v) and v > 0

    changed = False
    with _CACHE_LOCK:
        if ok(steps_per_s):
            _observe(_RATE_CACHE, (dc, fl, dtype), cells, float(steps_per_s) / _model_rate(dc, fl, dtype, cells))
            changed = True
        if kind and ok(speed_factor):
            cpl = float(cells_per_ref) if ok(cells_per_ref) else 64.0
            current = _speed_factor_locked(kind, model, cpl)
            _observe(_SPEED_CACHE, (kind, model), cpl, clamp_update(float(speed_factor), current))
            changed = True
        if ok(loop_efficiency):
            _observe(_LOOP_CACHE, (dc, "live" if live_view else "headless"), cells, min(float(loop_efficiency), 1.0))
            changed = True
    if changed:
        _save_calibration()


#: Largest factor one run may move a speed factor by (:func:`clamp_update`).
SPEED_UPDATE_MAX = 1.5
#: Weight ratio of each older speed-factor observation against the next newer one in
#: :func:`speed_median` (milder than :func:`_local`'s halving: a median needs several voices).
_SPEED_AGE_DECAY = 0.8


def clamp_update(measured: float, current: float | None) -> float:
    """A run's measured speed factor, limited to within :data:`SPEED_UPDATE_MAX` of the value the
    planner currently uses for its key: one unusual run (a busy GPU, a room that spun up further
    than usual) moves the next estimate by at most that factor."""
    if current is None or not (math.isfinite(current) and current > 0):
        return measured
    return float(min(max(measured, current / SPEED_UPDATE_MAX), current * SPEED_UPDATE_MAX))


def speed_median(store: dict, key: tuple, size: float, prior: float | None = None) -> float | None:
    """The speed factor measured for ``key`` near ``size`` (cells per reference length): the
    weighted median of the key's observations, each weighted by the resolution kernel of
    :func:`_local` and by :data:`_SPEED_AGE_DECAY` per newer observation, with the class
    default ``prior`` as one more voice of weight :data:`_PRIOR_WEIGHT` (so a measurement made
    at a far resolution does not stand alone); None without observations.

    A median, not a mean: the Draft fan room was planned 3.3 times too long after other runs
    had fed a mean (critic round 3.4), and a median ignores a lone outlier. Keys are case
    signatures (:func:`calibration_kind`), so only like cases share one. The caller holds
    _CACHE_LOCK."""
    obs = store.get(key) or []
    if not obs:
        return None
    x = math.log(max(size, 1e-30))
    pairs = [(v, _SPEED_AGE_DECAY ** age * math.exp(-0.5 * ((x - xs) / _KERNEL_WIDTH) ** 2))
             for age, (xs, v) in enumerate(reversed(obs))]
    if prior is not None and math.isfinite(prior) and prior > 0:
        pairs.append((math.log(prior), _PRIOR_WEIGHT))
    pairs.sort()
    total = sum(w for _, w in pairs)
    if not total > 0:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= 0.5 * total:
            return math.exp(v)
    return math.exp(pairs[-1][0])


def base_kind(kind: str) -> str:
    """The class of a calibration key without its case signature: "closed-fan+jet+heat@64" ->
    "closed-fan" (the key of the measured default in SPEED_FACTOR_KIND)."""
    return str(kind).split("@", 1)[0].split("+", 1)[0]


def _speed_factor_locked(kind: str, model: str, cells_per_ref: float | None = None) -> float:
    """:func:`_speed_factor` for a caller that holds _CACHE_LOCK."""
    default = _default_speed_factor(kind, model)
    v = speed_median(_SPEED_CACHE, (kind, model), cells_per_ref or 64.0, prior=default)
    return v if v is not None else default


def signature_defaults(kind: str) -> list[str]:
    """The keys a calibration key's default is looked up under, most specific first:
    "closed-fan+jet+heat@64" -> ["closed-fan+jet+heat", "closed-fan+jet", "closed-fan"] (the grid
    class is dropped: the defaults were measured at Draft)."""
    parts = str(kind).split("@", 1)[0].split("+")
    return ["+".join(parts[:n]) for n in range(len(parts), 0, -1)]


def _default_speed_factor(kind: str, model: str) -> float:
    """The class default of a calibration key (before this machine has measured it): the most
    specific signature SPEED_FACTOR_KIND holds (:func:`signature_defaults`), else the class's."""
    for key in signature_defaults(kind)[:-1]:
        if (key, model) in SPEED_FACTOR_KIND:
            return SPEED_FACTOR_KIND[(key, model)]
    base = base_kind(kind)
    # a class with placed physics ("airfoil-physics") starts from its plain class's value, one
    # planned on its peak speed ("generic-plume", "airfoil-jet") from that source's
    for src in ("plume", "jet"):
        if base.endswith("-" + src):
            return SPEED_FACTOR_KIND.get((base, model), PEAK_SPEED_FACTOR[src][model])
    return SPEED_FACTOR_KIND.get((base, model), SPEED_FACTOR_KIND.get((base.removesuffix("-physics"), model),
                                                                      SPEED_FACTOR[model]))


def _speed_factor(kind: str, model: str, cells_per_ref: float | None = None) -> float:
    """The CFL speed factor to plan with: measured for this calibration key near this
    resolution when available (:func:`speed_median`), else the class default (SPEED_FACTOR_KIND)."""
    with _CACHE_LOCK:
        return _speed_factor_locked(kind, model, cells_per_ref)


def _grid_class(cells_across: float | None) -> str:
    """"@64": the cells across a closed box's smaller side, rounded to a power of two."""
    if cells_across is None or not (math.isfinite(cells_across) and cells_across > 0):
        return ""
    return f"@{2 ** round(math.log2(cells_across))}"


def calibration_kind(base: str, placed: Placed | None, *, closed: bool, natural: bool, speed_ratio: float,
                     source: str = "reference", bodies: bool = False, cells_across: float | None = None) -> str:
    """The key a plan's CFL speed factor is planned with and recorded under (2D and 3D).

    Closed boxes: "closed-natural" (buoyancy is the reference velocity), "closed-fan" (a fan
    drives the box: its slipstream sets the peak speed) or "closed-forced" (jets, a moving
    floor), followed by the case signature: "+jet" with jets, "+heat" with buoyant heat
    transfer besides a fan or jets, "+heater" for a naturally convecting box warmed by heaters
    (a plume, not a cavity's wall layers), "+body" with bodies inside, and the grid class "@n" (cells
    across the smaller side, :func:`_grid_class`). How fast a box recirculates depends on all of
    them: a plain fan room spins up to a disc velocity of 1.17 m/s where the same fan with a
    supply jet, a vent and a heater reached 0.16 m/s, and one shared "closed-fan" factor moved
    the second's estimate from 52 s to 164 s for a 49 s run (critic round 3.4).

    Tunnels: the geometry class ``base``, "-physics" with placed physics, "-plume" or "-jet" when
    a plume or a jet / fan outruns the free stream (``speed_ratio`` > :data:`PEAK_RATIO_MIN`,
    ``source`` from :func:`.physics.peak_source`), whose factor is then relative to that peak
    speed. Keys whose factor is relative to the peak are new names, so measurements made
    relative to U under the old ones never mix with them.
    """
    if closed:
        if natural:
            kind = "closed-natural"
        else:
            kind = "closed-fan" if placed is not None and placed.fans else "closed-forced"
        if placed is not None:
            if any(getattr(p, "kind", "") == "jet" for p in placed.ports_si):
                kind += "+jet"
            th = placed.thermal
            if not natural and th is not None and float(getattr(th, "richardson", 0.0) or 0.0):
                kind += "+heat"
            elif natural and placed.heat_sources:
                # a heater's plume, not a heated wall's boundary layer, drives this box: it runs several
                # times faster relative to the free-fall velocity than the cavities the default is from
                kind += "+heater"
        if bodies:
            kind += "+body"
        return kind + _grid_class(cells_across)
    if speed_ratio > PEAK_RATIO_MIN:
        return base + ("-plume" if source == "plume" else "-jet")
    return base + ("-physics" if placed is not None and placed.any else "")


def plume_note(gap_m: float, u_peak: float, time_unit: float) -> str:
    """The note on a buoyancy-dominated tunnel run's ceiling: the tunnel is sized for blockage, so
    a rising plume reaches its frictionless ceiling and spreads under it. On the Draft hot sphere
    (Ri 627) the fastest air of the whole run was there, 1.8 u_b rising and 1.5 u_b sideways."""
    if not (gap_m > 0 and u_peak > 0):
        return ""
    t = gap_m / u_peak
    return (f"The plume rises {units.sig(gap_m, 2)} m to the tunnel's frictionless ceiling in about {fmt_time(t)} "
            f"({units.sig(t / time_unit, 2)} L/U) and spreads under it: the domain is sized for blockage, not for a "
            "rising plume, so the flow above the body is confined (the fastest air of the hot-sphere run was "
            "there).")


def peak_note(ratio: float, u_peak: float, basis: str) -> str:
    """The "The step is sized for the ..." sentence of a plan whose peak speed is above U ("" else)."""
    if not ratio > PEAK_RATIO_MIN:
        return ""
    return (f" The step is sized for the {basis}, {units.sig(u_peak, 3)} m/s ({ratio:.2g} times the reference "
            "velocity), the fastest air the run is expected to reach.")


# --- persistence ----------------------------------------------------------------

def _stores() -> dict[str, dict]:
    # looked up at call time: tests swap the module-level stores
    return {"rate": _RATE_CACHE, "speed": _SPEED_CACHE, "loop": _LOOP_CACHE,
            "rate3d": _RATE_CACHE3D, "speed3d": _SPEED_CACHE3D, "loop3d": _LOOP_CACHE3D}


def use_calibration_file(path: str | os.PathLike | None) -> None:
    """Keep the measured corrections in ``path`` (JSON): load it now, save after each run.

    Without a file every session starts from the reference model, so the first
    estimates of each session were the least trustworthy. The app and the CLI
    point this at the user config folder. ``None`` stops saving (what was
    loaded stays). A missing or corrupt file is ignored: the estimate is
    advisory and must never stop a run.
    """
    global _CALIBRATION_FILE
    _CALIBRATION_FILE = None if path is None else os.fspath(path)
    if _CALIBRATION_FILE is None:
        return
    try:
        with open(_CALIBRATION_FILE, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return
    if not isinstance(doc, dict) or doc.get("version") != 1:
        return
    with _CACHE_LOCK:
        for name, store in _stores().items():
            section = doc.get(name)
            if not isinstance(section, dict):
                continue
            for k, obs in section.items():
                try:
                    pairs = [(float(x), float(v)) for x, v in obs]
                except (TypeError, ValueError):
                    continue
                pairs = [p for p in pairs if math.isfinite(p[0]) and math.isfinite(p[1])]
                if pairs:
                    key = tuple(str(k).split("|"))
                    # the file's history first, then anything measured in this session
                    store[key] = (pairs + store.get(key, []))[-_KEEP:]


def _save_calibration() -> None:
    path = _CALIBRATION_FILE
    if path is None:
        return
    with _CACHE_LOCK:
        doc = {"version": 1, **{name: {"|".join(k): [list(p) for p in obs] for k, obs in store.items()}
                                for name, store in _stores().items()}}
    from ..project.fileio import write_json
    # a read-only or vanished config folder must not fail a finished run
    with contextlib.suppress(OSError, TypeError, ValueError):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_json(path, doc)


def reset_calibration() -> None:
    """Forget every measurement held by this process (the file is not touched)."""
    with _CACHE_LOCK:
        for store in _stores().values():
            store.clear()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def fft_friendly(n: int) -> int:
    """Smallest even 5-smooth integer (factors 2, 3, 5 only) >= n.

    The pressure solve is a DCT computed with real FFTs along each axis; FFT
    libraries are fastest on sizes made of small primes, and a large prime
    factor can make a transform several times slower.
    """
    n = max(math.ceil(n), 8)
    m = n + (n % 2)
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 2


def rotate_points(p: np.ndarray, deg: float, about: tuple[float, float]) -> np.ndarray:
    """Rotate (N, 2) points counter-clockwise by ``deg`` about ``about``."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    q = np.asarray(p, dtype=np.float64) - np.asarray(about, dtype=np.float64)
    return np.c_[c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]] + np.asarray(about, dtype=np.float64)


def _wheel_geometry(b: Body) -> tuple[tuple[float, float], float]:
    """Centre and mean radius of a wheel body (centre from the Body, else the centroid)."""
    pts = np.asarray(b.points, dtype=np.float64)
    c = tuple(float(v) for v in (b.center if b.center is not None else P.centroid(pts)))
    r = float(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]).mean())
    return c, r  # type: ignore[return-value]


def wind_frame_bodies(design: Design, alpha_deg: float, *, ground: bool = False) -> list[Body]:
    """The design's bodies in the wind frame (metres): rotated by -alpha about
    ``moment_ref`` (positive alpha = nose up, DESIGN.md 3).

    With a road (``ground=True``) the wheels stay where they are -- a pitched car
    still stands on its wheels -- and only the other bodies rotate. Wheel
    ``omega`` is copied unchanged; the planner sets it.
    """
    out = []
    about = tuple(design.moment_ref)
    for b in design.bodies:
        pts = np.asarray(b.points, dtype=np.float64)
        center = b.center
        if not (ground and b.role == "wheel") and alpha_deg:
            pts = rotate_points(pts, -alpha_deg, about)
            if center is not None:
                center = tuple(float(v) for v in rotate_points(np.array([center]), -alpha_deg, about)[0])
        # the thermal condition travels with the body (DESIGN.md 10.2): a heated cylinder stays heated
        out.append(Body(name=b.name, points=pts, role=b.role, omega=b.omega, center=center,
                        thermal=getattr(b, "thermal", "adiabatic"), thermal_value=getattr(b, "thermal_value", 0.0)))
    return out


def focus_window(polys: list[np.ndarray], extent: tuple[float, float, float, float], *,
                 ground: bool) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) around ``polys`` and their near wake, clipped to ``extent``
    (see :attr:`SimPlan.focus_extent`)."""
    ex0, ex1, ey0, ey1 = (float(e) for e in extent)
    if not polys:
        return ex0, ex1, ey0, ey1
    bx0, bx1, by0, by1 = P.bbox(polys)
    lb = max(bx1 - bx0, 1e-9)
    hb = max((by1 - (0.0 if ground else by0)), 1e-9)
    x0, x1 = max(ex0, bx0 - 0.5 * lb), min(ex1, bx1 + 2.0 * lb)
    height = min(max((x1 - x0) / 2.4, 2.5 * hb), ey1 - ey0)
    if ground:
        y0 = ey0
    else:
        y0 = min(max(0.5 * (by0 + by1) - 0.5 * height, ey0), ey1 - height)
    return x0, x1, y0, y0 + height


def fmt_duration(seconds: float) -> str:
    """Human run-time: '< 1 s', '45 s', '2 min 30 s', '1 h 05 min'."""
    if seconds is None or not math.isfinite(seconds):
        return "—"
    s = max(0.0, float(seconds))
    if s < 1:
        return "< 1 s"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        m, r = divmod(round(s), 60)
        return f"{m} min {r:02d} s" if m < 10 else f"{m} min"
    h, r = divmod(round(s), 3600)
    return f"{h} h {r // 60:02d} min"


def fmt_estimate(seconds: float | None) -> str:
    """A run-time estimate for a button or a list: '~40 s', '~5 min', '~1 h 10 min'.

    Rounded harder than :func:`fmt_duration` (5 s below a minute, whole minutes
    below an hour) because it is a forecast: '~5 min' on the Run button makes a
    long run a visible choice without pretending to know it to the second.
    Empty for an unknown estimate, so a caller can write ``f"Run  {label}"``.
    """
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return ""
    s = float(seconds)
    if s < 10:
        return "< 10 s"
    if s < 57.5:
        return f"~{5 * round(s / 5):.0f} s"
    if s < 3570:
        return f"~{max(1, round(s / 60))} min"
    h, m = divmod(round(s / 60), 60)
    return f"~{h} h" + (f" {m:02d} min" if m else "")


def fmt_time(seconds: float) -> str:
    """A physical time with an SI prefix: '1.23 s', '45.6 ms', '103 µs'."""
    if not math.isfinite(seconds):
        return "—"
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")):
        if abs(seconds) >= scale or unit == "ns":
            return f"{units.sig(seconds / scale, 3)} {unit}"
    return f"{seconds:g} s"


def _unique_names(bodies: list[Body]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, b in enumerate(bodies):
        name = (b.name or f"Body {i + 1}").strip() or f"Body {i + 1}"
        k = seen.get(name, 0)
        seen[name] = k + 1
        out.append(name if k == 0 else f"{name} ({k + 1})")
    return out


def _max_thickness(poly: np.ndarray, n: int = 256) -> float:
    """Largest vertical extent of a polygon over its x range (airfoil thickness in
    the design frame, where the chord lies along x)."""
    p = np.asarray(poly, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
    xs = np.linspace(x0, x1, n + 2)[1:-1]
    ax, ay, bx, by = p[:, 0][None], p[:, 1][None], q[:, 0][None], q[:, 1][None]
    X = xs[:, None]
    cross = (ax - X) * (bx - X) <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(bx != ax, (X - ax) / (bx - ax), 0.0)
    y = np.where(cross, ay + t * (by - ay), np.nan)
    with np.errstate(all="ignore"):
        ext = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
    ext = ext[np.isfinite(ext)]
    return float(ext.max()) if ext.size else 0.0


def _mean_thickness(poly: np.ndarray) -> float:
    """2 x area / perimeter: the thickness of a thin plate-like shape (a rectangle
    L x t gives ~t), the radius of a round one -- never limiting for compact bodies."""
    per = P.perimeter(poly)
    return 2.0 * abs(P.signed_area(poly)) / per if per > 0 else 0.0


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

@dataclass
class SimPlan(PlacedPhysicsPlan):
    """Everything the runner needs, plus the rationale shown in the Setup page.

    The first block of fields is the DESIGN.md 5.4 contract; the trailing ones
    (with defaults) are extras the runner, the live session and the CLI use.
    Times (``t_end``, ``t_avg_start``, ``dt``) are in convective units
    ``length_ref / speed``; ``extent`` and ``bodies_wind`` are in metres.
    """
    solver_config: SolverConfig
    design: Design
    sim: Simulation
    length_ref: float
    speed: float
    fluid: Fluid
    re_real: float
    re_sim: float
    turbulence: str                  # "laminar" | "les-<model>"
    t_end: float
    t_avg_start: float
    dt: float
    steps_estimate: int
    est_seconds: float
    cells: int
    cells_per_ref: float
    bodies_wind: list[Body]
    extent: tuple[float, float, float, float]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # --- extras ---------------------------------------------------------------
    device: str = "cpu"                     # resolved torch device string
    device_name: str = ""                   # human-readable device
    precision: str = "float32"
    advection: str = "upwind3"
    quality: str = "standard"
    ground: bool = False
    ground_moving: bool = False
    strouhal_length: float = 1.0            # St reference length [m] (chord / frontal height)
    blockage: float = float("nan")          # frontal height / domain height
    ref_label: str = "reference length"     # "chord", "vehicle length", ...
    live: bool = False
    capped: bool = False
    geometry_class: str = "generic"         # layout rules used: "airfoil" | "vehicle" | "generic"
    conv_rtol: float = 0.01                 # early-stop target: stderr(mean Cd) / |mean Cd|
    blockage_limit: float = 0.05            # the blockage the domain was sized for
    #: ``est_seconds`` is for a run with the live view (the app); this one for a
    #: headless run (CLI, scripts), which skips rendering frames.
    est_seconds_headless: float = float("nan")
    #: Whether the grid resolves the laminar boundary layer at the simulated Re.
    #: When it does not, an airfoil's forces -- and its lift oscillation -- belong
    #: to the under-resolved flow (the runner then reports no Strouhal number).
    bl_resolved: bool = True
    # --- placed physics (DESIGN.md 10; simulation/physics.py) ---------------------------
    #: "tunnel" (inflow from the left) or "closed" (walls on every side: rooms, cavities)
    domain_mode: str = "tunnel"
    #: scales and dimensionless groups (U, ΔT, Re, Pr, Gr, Ra, Ri, Pe), None without placed physics
    physics: PhysicsScales | None = None
    #: the solver specs of the emitters, heat transfer, heaters, fans and jets, with what the
    #: outputs need to convert them back to SI; None without placed physics
    placed: Placed | None = None
    #: convergence monitors judged besides the forces: "nu" (Nusselt number), "fan<i>_velocity"
    #: / "fan<i>_thrust", "ke" (mean kinetic energy, closed boxes)
    monitors: tuple[str, ...] = ()
    #: key of the measured CFL speed factor (plan.record_throughput): the geometry class, with a
    #: suffix for runs whose time step the placed physics sets ("-physics", "closed-natural" ...)
    calibration_kind: str = ""
    #: time per step relative to the plain solver (physics.physics_cost): the estimate divides the
    #: throughput by it, and the runner multiplies the measured one back before recording it
    step_cost: float = 1.0
    #: peak speed over U (physics.peak_speed, >= 1): the planned time step and the estimate use
    #: speed factor x this ratio, and the runner records the measured factor divided by it
    speed_ratio: float = 1.0
    #: what sets the peak speed ("slipstream of fan “Fan” (2 × its disc velocity)", ...)
    peak_basis: str = ""
    #: the planner's own estimate; ``est_seconds`` is replaced by the runner's measured
    #: re-estimate once a run has timed its first blocks (nan until the plan is built)
    est_seconds_planned: float = float("nan")
    #: seconds the estimate expects after the stepping loop (the sensitivity map), which the
    #: runner's re-estimate keeps
    est_after_seconds: float = 0.0
    #: (cell size [m], cells across the laminar boundary layer, cells wanted) of the thermal checks
    #: (physics.thermal_checks), so a run can re-derive them on the temperature rise it measured
    #: (physics_run.apply_measured_thermal); None without heat transfer planned
    thermal_grid: tuple[float, float, float] | None = None

    @property
    def time_unit(self) -> float:
        """Seconds per convective time unit (length_ref / speed)."""
        return self.length_ref / self.speed

    @property
    def grid(self) -> tuple[int, int]:
        return (self.solver_config.nx, self.solver_config.ny)

    @property
    def focus_extent(self) -> tuple[float, float, float, float]:
        """A default view window (metres, wind frame): the bodies and their near wake.

        The domain itself is sized for blockage, not for looking at: a car's is
        12-20 times the car's height tall, so showing all of it makes the car a
        sliver at the bottom. The window spans half a body length upstream to
        two downstream, at a 2.4 : 1 aspect ratio (but at least 2.5 body heights
        tall), sits on the road when there is one, and stays inside the domain.
        """
        return focus_window([np.asarray(b.points) for b in self.bodies_wind], self.extent, ground=self.ground)

    @property
    def estimate_label(self) -> str:
        """The run-time estimate for the Run button ('~40 s', '~5 min'; '' for explore)."""
        return "" if self.live else fmt_estimate(self.est_seconds)

    def summary(self) -> dict[str, str]:
        """Label -> formatted value for the Setup page (shown verbatim)."""
        cfg = self.solver_config
        if self.turbulence.startswith("les"):
            model = self.turbulence.split("-", 1)[1].upper() if "-" in self.turbulence else "sub-grid"
            flow = f"LES ({model} sub-grid model), {self.advection} advection"
        else:
            flow = f"Laminar (all scales resolved), {self.advection} advection"
        xmin, xmax, ymin, ymax = self.extent
        sim_re = units.fmt_reynolds(self.re_sim)
        if self.capped:
            sim_re += " (capped)"
        rows = {
            "Real Reynolds number": units.fmt_reynolds(self.re_real),
            "Simulated Reynolds number": sim_re,
            "Flow model": flow,
            "Grid": f"{cfg.nx:,} × {cfg.ny:,}",
            "Cells": f"{self.cells:,}",
            f"Cells per {self.ref_label}": f"{self.cells_per_ref:.0f}",
            "Domain": f"{units.sig(xmax - xmin, 3)} m × {units.sig(ymax - ymin, 3)} m",
            "Blockage": f"{100 * self.blockage:.1f} %",
            "Road": ("moving belt" if self.ground_moving else "fixed floor") if self.ground else "none (free flow)",
            "Time step": f"{units.sig(self.dt, 3)} L/U ({fmt_time(self.dt * self.time_unit)})",
            "Duration": f"{self.t_end:.4g} L/U ({fmt_time(self.t_end * self.time_unit)} of flow)",
            "Averaging from": f"t = {self.t_avg_start:.4g} L/U",
            "Stops early when": (f"Cd known to ±{100 * self.conv_rtol:g} %, "
                                 f"Cl to ±{100 * CL_RTOL_FACTOR * self.conv_rtol:g} %"),
            "Steps (estimate)": f"≈ {self.steps_estimate:,}",
            "Estimated run time": fmt_duration(self.est_seconds_planned if math.isfinite(self.est_seconds_planned)
                                               else self.est_seconds),
            "Device": f"{self.device_name} · {self.precision}",
        }
        if self.closed:
            # no free stream: the Reynolds number is the reference velocity's, there is no blockage
            rows.pop("Real Reynolds number", None)
            rows["Simulated Reynolds number"] = f"{units.fmt_reynolds(self.re_sim)} (U L / ν with U the reference velocity)"
            rows.pop("Blockage", None)
            rows["Road"] = ("moving floor" if self.ground_moving else "fixed walls")
        if self.monitors and not self.live:
            rows["Stops early when"] = stop_rule_text(self)
        rows.update(physics_rows(self.physics, self.placed, closed=self.closed))
        if self.live:
            for k in ("Duration", "Averaging from", "Stops early when", "Steps (estimate)", "Estimated run time"):
                rows.pop(k, None)
        return rows


#: The Nusselt number of a heat-transfer run converges to a tighter tolerance than the drag: it is
#: an average over a whole surface (smooth in time), and the cavity and cylinder references are
#: quoted to 0.1-1 %, so the early stop waits until its mean is known to this share of conv_rtol.
NU_RTOL_FACTOR = 0.5


def stop_rule_text(plan) -> str:
    """The "Stops early when" row of a plan with convergence monitors besides (or instead of) the forces."""
    r = float(plan.conv_rtol)
    parts = []
    closed = getattr(plan, "domain_mode", "tunnel") == "closed"
    if _plan_bodies(plan) and not closed:
        # a closed box is judged on its monitors only (runner._combined_verdict)
        parts.append(f"Cd known to ±{100 * r:g} %, Cl to ±{100 * CL_RTOL_FACTOR * r:g} %")
    if "nu" in plan.monitors:
        parts.append(f"Nu to ±{100 * NU_RTOL_FACTOR * r:g} %")
    if "t_rise_max" in plan.monitors or "t_rise_mean" in plan.monitors:
        parts.append(f"the peak and mean fluid temperature rises to ±{100 * r:g} %")
    if any(m.startswith("fan") for m in plan.monitors):
        parts.append(f"the fans' flow to ±{100 * r:g} %")
    if "ke" in plan.monitors:
        parts.append(f"the kinetic energy to ±{100 * r:g} %")
    return ", ".join(parts) if parts else f"Cd known to ±{100 * r:g} %"


def _plan_bodies(plan) -> bool:
    cfg = plan.solver_config
    return bool(getattr(cfg, "bodies", None))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _resolve_device(requested: str | None, warnings: list[str]) -> str:
    try:
        return devmod.resolve(requested or "auto")
    except RuntimeError:
        best = devmod.resolve("auto")
        warnings.append(f"The requested device '{requested}' is not available here; using "
                        f"{devmod.info(best).name} instead.")
        return best


def _resolve_precision(precision: str | None, device: str, live: bool, warnings: list[str]) -> str:
    """"float32" | "float64" for the solver. "auto" is float32 on GPUs (the
    solver is bandwidth-bound and consumer GPUs run float64 at 1/64 rate) and
    float64 on CPUs; explore mode is always float32 unless set explicitly."""
    if live and precision in (None, "", "auto"):
        return "float32"            # explore mode: speed over the last digit
    try:
        dt = devmod.resolve_dtype(precision, device)
    except (RuntimeError, KeyError):
        warnings.append(f"Precision '{precision}' is not supported on this device; using float32.")
        return "float32"
    return "float64" if "64" in str(dt) else "float32"


def _device_name(dev: str, dinfo) -> str:
    if _device_class(dev) == "gpu":
        return "GPU · " + dinfo.name
    # DeviceInfo reports torch's default thread count, but the runner pins the
    # solver to CPU_THREADS (see live.cpu_threads); say what actually runs
    import platform
    cpu = platform.processor() or platform.machine() or "CPU"
    n = max(1, min(CPU_SOLVER_THREADS, int(getattr(dinfo, "threads", CPU_SOLVER_THREADS) or 1)))
    return f"CPU · {cpu} · {n} solver thread{'s' if n != 1 else ''}"


def _sizes(bodies: list[Body], ground: bool) -> dict:
    """Characteristic sizes of the wind-frame geometry (metres)."""
    polys = [np.asarray(b.points, dtype=np.float64) for b in bodies]
    xmin, xmax, ymin, ymax = P.bbox(polys)
    frontal = ymax - ymin
    lx = xmax - xmin
    not_wheels = [p for p, b in zip(polys, bodies) if b.role != "wheel"] or polys
    clearance = min(float(p[:, 1].min()) for p in not_wheels) if ground else float("nan")
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "frontal": frontal, "lx": lx,
            "s": max(frontal, 0.25 * lx), "clearance": clearance,
            "frontal_above": (ymax - max(clearance, 0.0)) if ground else frontal}


def _thin_features(design: Design, bodies_wind: list[Body], sz: dict, ground: bool) -> list[tuple[str, float]]:
    """(description, thickness [m]) of features that need a minimum cell count."""
    out = []
    for b in design.bodies:
        pts = np.asarray(b.points, dtype=np.float64)
        if len(pts) < 3 or b.role == "wheel":
            continue
        if design.kind == "airfoil" or b.role == "element":
            t = _max_thickness(pts)
            what = f"the {100 * t / max(P.bbox(pts)[1] - P.bbox(pts)[0], 1e-30):.0f} % thickness of '{b.name}'"
        else:
            t = _mean_thickness(pts)
            what = f"the thickness of '{b.name}'"
        if t > 0:
            out.append((what, t))
    if ground and math.isfinite(sz["clearance"]) and sz["clearance"] > 0:
        out.append(("the ground clearance", sz["clearance"]))
    return out


def _gap_report(bodies: list[Body], h_m: float) -> list[str]:
    """Warnings for gaps between bodies that the grid closes or barely resolves."""
    msgs = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a, b = bodies[i], bodies[j]
            pa, pb = np.asarray(a.points), np.asarray(b.points)
            if len(pa) < 3 or len(pb) < 3 or P.polygons_overlap(pa, pb):
                continue
            gap = P.polygon_distance(pa, pb)
            cells = gap / h_m
            if cells < CLOSED_GAP_CELLS:
                msgs.append(f"The {units.sig(gap * 1000, 3)} mm gap between '{a.name}' and '{b.name}' is only "
                            f"{cells:.1f} cells wide: the solver treats it as closed (no flow through it). "
                            "Use a finer grid or a wider gap if that flow matters.")
            elif cells < 6:
                msgs.append(f"The gap between '{a.name}' and '{b.name}' spans {cells:.1f} cells: flow passes "
                            "but its losses are under-resolved.")
    return msgs


def _underbody_warning(geo: str, ground: bool, clearance: float, h_m: float, quality: str) -> str | None:
    """The 'underbody gap under-resolved' warning for a car on a road, or None.

    In 2D the underbody is a channel as wide as the car; its flow and the suction
    it produces are a large part of the lift. With fewer than UNDERBODY_CELLS
    cells across the clearance the channel stays open but is mostly diffuse wall,
    so lift is the least trustworthy number of the run.
    """
    if geo != "vehicle" or not ground or not (math.isfinite(clearance) and clearance > 0):
        return None
    cells = clearance / h_m
    if cells >= THIN_SLACK * UNDERBODY_CELLS:
        return None
    # one-line summaries show the text up to the first ':' / '. ' outside parentheses
    return (f"Underbody gap under-resolved (the {units.sig(clearance * 1000, 3)} mm ground clearance spans "
            f"{cells:.1f} grid cells; about {UNDERBODY_CELLS:.0f} resolve the flow under the car). Lift is the "
            "number this affects most: compare it only between designs run at the same quality"
            + (", and use Fine quality before quoting it." if quality != "fine" else "."))


def _in_refs(v: float, L: float) -> str:
    """A length in reference lengths, 3 significant figures."""
    return units.sig(v / L, 3)


def _cell_budget(dclass: str, geo: str, qi: int, *, live: bool) -> int:
    """Cell budget for a run (see CELL_BUDGET and VEHICLE_DRAFT_GPU_BUDGET)."""
    if live:
        return LIVE_BUDGET[dclass]
    if dclass == "gpu" and geo == "vehicle" and qi == 0:
        return VEHICLE_DRAFT_GPU_BUDGET
    return CELL_BUDGET[dclass][qi]


def _check_memory(cells: int, precision: str, dclass: str, dinfo, *, user_cpl: bool, geo: str) -> None:
    """Refuse a grid the device cannot hold (ValueError, shown by Setup and the CLI).

    The presets stay inside their cell budgets, but a hand-set resolution skipped
    every size check: 4096 cells per chord planned 806 M cells (an estimated
    52,000 h) and Run stayed enabled; on a CPU the allocation then either failed
    with the allocator's raw message or paged the machine to a standstill.
    """
    per_cell = BYTES_PER_CELL.get(precision, max(BYTES_PER_CELL.values()))
    need = cells * per_cell
    if dclass == "gpu":
        have_gb, what = float(getattr(dinfo, "memory_gb", 0.0) or 0.0), "GPU memory"
    else:
        have_gb, what = devmod.system_memory_gb(), "memory (RAM)"
    have = have_gb * 1024**3
    if have <= 0 or need <= MEMORY_FRACTION * have:
        return
    ref = {"airfoil": "chord", "vehicle": "vehicle length"}.get(geo, "reference length")
    fix = (f"Lower the cells per {ref} in Setup (Mesh & time)" if user_cpl else
           "Choose a lower quality")
    raise ValueError(f"This grid of {cells:,} cells needs about {need / 1024**3:.1f} GB of {what}, more than "
                     f"{MEMORY_FRACTION:.0%} of the {have_gb:.1f} GB this device has. {fix}, or run on a device "
                     "with more memory.")


def _durations(kind: str, qi: int, s_over_l: float) -> tuple[float, float]:
    t_end, t_avg = DURATION[kind][qi]
    if kind == "generic":
        t_end, t_avg = t_end * s_over_l, t_avg * s_over_l
    if qi == 0 and t_end > DRAFT_T_END_MAX:
        # a bluff body larger than its reference length would otherwise get a
        # Draft run of 100+ L/U; keep the averaging window's share of the run
        t_end, t_avg = DRAFT_T_END_MAX, t_avg * DRAFT_T_END_MAX / t_end
    return t_end, t_avg


def _solver_physics_2d(placed: Placed | None, cfd) -> dict:
    """The SolverConfig keywords of the placed physics and of the AeroJAX diagnostics / PID step."""
    kw = {"diagnostics": bool(getattr(cfd, "record_diagnostics", False)),
          "adaptive_dt": "pid" if getattr(cfd, "adaptive_dt", "cfl") == "pid" else "cfl"}
    if placed is not None:
        kw.update(emitters=list(placed.emitters), thermal=placed.thermal, heat_sources=list(placed.heat_sources),
                  fans=list(placed.fans), ports=list(placed.ports), dye_inlet=placed.dye_inlet)
    return kw


# --- closed domains (DESIGN.md 10: rooms, cavities, enclosures) ---------------------------------

#: Cells across the smaller side of a closed box, per quality (Draft, Standard, Fine) and device
#: class. The de Vahl Davis cavity reads Nu within 0.3 % at 64 cells (Ra 10^4), 0.7 % at 128
#: (Ra 10^5) and 1.0 % at 128 (Ra 10^6) of the benchmark (theory_ns2d.md 21.1); the thermal
#: boundary-layer rule (physics.thermal_checks) raises the count for higher Rayleigh numbers.
CLOSED_CELLS = {"gpu": (64, 96, 128), "cpu": (48, 64, 80)}
#: Explore sessions of a closed box: cells across its smaller side.
LIVE_CLOSED_CELLS = {"gpu": 96, "cpu": 48}
#: Duration of natural convection in free-fall times H / U: (a + b Pe_H), capped. The validation
#: cavities (64-256 cells) reached a steady Nu (change < 1e-6 per free-fall time) by 12-26
#: free-fall times at Ra 10^3, 28-62 at 10^4, 54-80 at 10^5 and 125-144 at 10^6 (Pe_H 27, 84, 266,
#: 843): about 30 + 0.15 Pe_H. The plan allows 1.5 times that; the run stops once Nu has settled.
NATURAL_T = (45.0, 0.225, 1500.0)
#: Draft natural-convection boxes from this Rayleigh number on (unsteady, then turbulent boundary
#: layers: rooms) run NATURAL_DRAFT_FF free-fall times instead of NATURAL_T's steady-state rule, which
#: planned the Draft heater-only 4 × 3 m room at 1500 of them, 9000 L/U and 49 minutes (critic 3.4).
NATURAL_DRAFT_RA = 1.0e8
NATURAL_DRAFT_FF = 40.0
#: Forced flows in a closed box (fans, jets, a moving floor): box turnovers max(W, H) / U per quality.
FORCED_TURNOVERS = (30.0, 40.0, 50.0)
#: Conduction-dominated boxes (Ra < 10^3, or no buoyancy and nothing blowing): diffusion times
#: H^2 / kappa; the slowest conduction mode decays as exp(-pi^2 t kappa / H^2), to 4e-7 by 1.5.
CONDUCTION_T = 1.5
#: (max|u| + max|v|) / U of closed boxes before this machine has measured any: the cavity's
#: maxima are 0.19 + 0.23 (Ra 10^4) and 0.08 + 0.26 (Ra 10^6) of the free-fall velocity
#: (de Vahl Davis), so 0.5; forced boxes are driven at their reference speed: 1.5.
CLOSED_SPEED_FACTOR = {"natural": 0.5, "forced": 1.5}


def _grid_for_box(width: float, height: float, h_target: float) -> tuple[int, int, float]:
    """(nx, ny, h): a uniform grid that fits the box exactly in height and to the nearest cell in
    width, with FFT-friendly sizes where the box shape allows (the pressure solve is a DCT)."""
    ny0 = max(8, math.ceil(height / h_target - 1e-9))
    best = None
    for ny in sorted({fft_friendly(ny0), fft_friendly(ny0 + 2), ny0 + (ny0 % 2)}):
        h = height / ny
        nx_exact = width / h
        for nx in sorted({fft_friendly(math.floor(nx_exact)), fft_friendly(math.ceil(nx_exact)),
                          max(8, round(nx_exact))}):
            err = abs(nx * h - width) / width
            # an exact fit of the walls is worth up to 30 % more cells (a 4 x 3 m room: 96 x 72, not 85 x 64)
            score = (err > 0.01, nx * ny * (1.0 if err < 1e-9 else 1.3), err)
            if best is None or score < best[0]:
                best = (score, nx, ny, h)
    _, nx, ny, h = best
    return nx, ny, h


def _plan_closed(design: Design, sim: Simulation, device: str | None, *, live: bool) -> SimPlan:
    """Plan a closed 2D domain: a room, a cavity or an enclosure with walls on every side.

    There is no free stream, so the planner takes the box (``cfd.domain_box``, or the bodies and
    the placed items with one size of room around them), chooses the reference velocity and the
    temperature scale (:func:`.physics.physics_scales`), resolves the box, its bodies and its
    thermal boundary layers, and sets the run length from the physics that drives the flow
    (free-fall times, box turnovers or diffusion times). Gravity acts along −y.
    """
    cond, cfd = sim.conditions, sim.cfd
    L = float(design.length_ref)
    if not (L > 0 and math.isfinite(L)):
        raise ValueError("The design's reference length must be positive.")
    if not (cond.fluid.density > 0 and cond.fluid.viscosity > 0):
        raise ValueError("The fluid needs a positive density and viscosity.")
    notes: list[str] = []
    warnings: list[str] = []
    quality = cfd.quality if cfd.quality in QUALITY_ORDER else "standard"
    qi = QUALITY_ORDER.index(quality)
    dev = _resolve_device(device if device not in (None, "", "auto") else cfd.device, warnings)
    dclass = _device_class(dev)
    precision = _resolve_precision(cfd.precision, dev, live, warnings)
    dinfo = devmod.info(dev)
    device_name = _device_name(dev, dinfo)

    # -- geometry and box
    ground = bool(sim.ground.enabled)
    moving = ground and bool(sim.ground.moving)
    alpha = float(cond.alpha_deg)
    bodies_wind = wind_frame_bodies(design, alpha, ground=False) if design.bodies else []
    for b, n in zip(bodies_wind, _unique_names(bodies_wind)):
        b.name = n
    polys = [np.asarray(b.points, dtype=np.float64) for b in bodies_wind]
    bb = P.bbox(polys) if polys else None
    box, explicit = closed_box(design, sim, 2, bb, alpha)
    bx0, bx1, by0, by1 = box
    W, H = bx1 - bx0, by1 - by0
    if bb is not None and (bb[0] < bx0 or bb[1] > bx1 or bb[2] < by0 or bb[3] > by1):
        warnings.append("The design reaches through the walls of the closed box: the walls cut it (only the part "
                        "inside is simulated). Enlarge the box (Domain ▸ Extent) if the whole body matters.")
    scales = physics_scales(design, sim, 2, box=box)
    U = scales.speed

    # -- resolution: the box, the thermal boundary layers, the bodies
    n_box = LIVE_CLOSED_CELLS[dclass] if live else CLOSED_CELLS[dclass][qi]
    h_t = min(W, H) / n_box
    why = f"{n_box} cells across the box's smaller side ({quality.capitalize()} quality on the {dclass.upper()})"
    n_thin = THIN_CELLS[qi] if not live else LIVE_THIN_CELLS
    if scales.thermal and scales.natural and scales.ra >= 1e3 and scales.surfaces > 0 and not live:
        # (only walls and bodies that exchange heat have a boundary layer; a room warmed by a heater
        # has none, and this rule once spent the whole cell budget on it: 576 × 432 for a 4 × 3 m room)
        nu_est = cavity_nusselt(scales.ra, scales.pr)
        h_bl = scales.length_ra / (2.0 * nu_est) / n_thin
        if h_bl < h_t:
            h_t, why = h_bl, (f"{n_thin:.0f} cells across the thermal boundary layer of the heated walls "
                              f"(about H / (2 Nu), Nu ≈ {nu_est:.2g} at Ra {units.fmt_reynolds(scales.ra)})")
    if polys:
        sz = _sizes(bodies_wind, False)
        per = CELLS_PER_SIZE["generic"][dclass][0 if live else qi]
        if sz["s"] / per < h_t:
            h_t, why = sz["s"] / per, f"{per} cells per body size ({units.sig(sz['s'], 3)} m)"
        for what, t in _thin_features(design, bodies_wind, sz, False):
            if t / n_thin < h_t:
                h_t, why = t / n_thin, f"{n_thin:.0f} cells across {what}"
    user_cpl = bool(cfd.cells_per_ref)
    if user_cpl:
        h_t, why = L / float(cfd.cells_per_ref), f"{float(cfd.cells_per_ref):g} cells per reference length (set by hand)"
    budget = _cell_budget(dclass, "generic", qi, live=live)
    if dclass == "gpu" and dinfo.memory_gb and dinfo.memory_gb < 4.0:
        budget = int(budget * dinfo.memory_gb / 4.0)
    if not user_cpl and W * H / h_t ** 2 > budget:
        h_new = math.sqrt(W * H / budget)
        if not live:
            warnings.append(f"The {dclass.upper()} cell budget ({budget:,}) limits the grid to cells of "
                            f"{units.sig(1000 * h_new, 3)} mm instead of {units.sig(1000 * h_t, 3)} mm (for {why}): "
                            "results are coarser than the quality preset intends.")
        h_t = h_new
    nx, ny, h_m = _grid_for_box(W, H, h_t)
    # Grow the *target* cell, not the fitted one: the fit rounds the cell count up to an FFT-friendly
    # size, so 3 % on the fitted cell can land on the same grid again and loop forever (a 4 x 3 m room
    # heated at Ra 2e11 did: 384 x 288 each time). A growing target lowers the count every few
    # steps and reaches the 8-cell floor at the latest, so this ends.
    h_try = h_t
    while not user_cpl and nx * ny > budget and ny > 8:
        h_try *= 1.03
        nx, ny, h_m = _grid_for_box(W, H, h_try)
    h = h_m / L
    cells = nx * ny
    extent = (bx0, bx0 + nx * h_m, by0, by0 + ny * h_m)
    _check_memory(cells, precision, dclass, dinfo, user_cpl=user_cpl, geo="generic")
    if abs(nx * h_m - W) > 1e-6 * W:
        notes.append(f"The box is {units.sig(W, 4)} m wide; the uniform grid fits it to the nearest cell "
                     f"({units.sig(nx * h_m, 4)} m).")

    # -- Reynolds number and flow model
    re_sim = U * L / scales.nu
    n_bl = 5.0 / (h * math.sqrt(re_sim))
    resolved = n_bl >= BL_CELLS_MIN
    if cfd.turbulence == "les":
        model = "les"
    elif cfd.turbulence == "laminar" or (scales.natural and scales.ra < 1e9):
        model = "laminar"
    else:
        model = "laminar" if resolved else "les"
    cell_re = re_sim * h
    advection = "central" if (model == "laminar" and cell_re <= CENTRAL_CELL_RE and not live) else "upwind3"
    turbulence = "laminar" if model == "laminar" else f"les-{LES_MODEL}"
    if cfd.reynolds_mode == "capped":
        notes.append("A closed box has no free stream to cap: its Reynolds number follows from the reference "
                     "velocity, so the Reynolds cap is not used.")

    # -- duration (convective times L / U)
    natural = scales.natural and scales.thermal
    # (time unit of the duration label, its name): the label is built from the duration used
    if natural and scales.ra >= 1e3 and scales.buoyancy:
        a, b, cap = NATURAL_T
        unit_lu, unit_name = scales.length_ra / L, "free-fall times H/U"
        n_ff = min(a + b * scales.pe * scales.length_ra / L, cap)
        if qi == 0 and not live and scales.ra >= NATURAL_DRAFT_RA and n_ff > NATURAL_DRAFT_FF and not cfd.duration:
            notes.append(f"Draft runs this room for {NATURAL_DRAFT_FF:g} free-fall times instead of the {n_ff:.0f} a "
                         f"steady laminar box would need: at Ra {units.fmt_reynolds(scales.ra)} the flow is unsteady, "
                         "so it never reaches the steady state the longer rule waits for, and the large-scale "
                         "circulation is established within a few tens of free-fall times. Enough to see the flow "
                         "and the temperatures; Standard quality, or a Duration set in Setup, averages longer.")
            n_ff = NATURAL_DRAFT_FF
        elif n_ff >= cap:
            warnings.append(f"At Ra {units.fmt_reynolds(scales.ra)} the box needs long to settle; the run is capped at "
                            f"{cap:g} free-fall times. Set Duration to run longer if Nu is still drifting at the end.")
        t_end = n_ff * unit_lu
    elif natural:
        # the speed scale is κ/H here, so one diffusion time H²/κ is H/L reference times
        unit_lu, unit_name = scales.length_ra / L, "diffusion times H²/κ"
        t_end = CONDUCTION_T * unit_lu
    else:
        unit_lu, unit_name = max(W, H) / L, "box turnovers max(W, H)/U"
        t_end = FORCED_TURNOVERS[qi] * unit_lu
    t_avg = 0.5 * t_end
    if cfd.duration:
        t_end = float(cfd.duration)
        if not cfd.average_from:
            t_avg = 0.5 * t_end
    dur_txt = (f"{units.sig(t_end / unit_lu, 3)} {unit_name}" + (", set in Setup" if cfd.duration else "")
               if unit_lu > 0 else "set in Setup")
    if cfd.average_from is not None and cfd.average_from > 0:
        t_avg = float(cfd.average_from)
    if t_avg >= 0.9 * t_end:
        warnings.append(f"Averaging was set to start at t = {t_avg:g}, too close to the end (t = {t_end:g}); the last "
                        "half of the run is averaged instead.")
        t_avg = 0.5 * t_end
    if live:
        t_end, t_avg = math.inf, 0.0

    # -- placed physics, bodies, solver config
    placed = placed_physics_2d(design, sim, scales, alpha_deg=alpha, extent=extent, h_m=h_m, bodies_wind=bodies_wind,
                               closed=True)
    u_peak, peak_basis = peak_speed(scales, sim, 2)
    ratio = max(1.0, u_peak / U)
    cal_kind = calibration_kind("generic", placed, closed=True, natural=natural, speed_ratio=ratio,
                                bodies=bool(polys), cells_across=min(nx, ny))
    kfac = _speed_factor(cal_kind, model, float(L / h_m)) * ratio
    nu_max = 1.0 / re_sim + (NUT_PEAK * h if model == "les" else 0.0)
    dt = min(CFL * h / kfac, 0.25 * h * h / nu_max)
    dt = min(dt, physics_dt(h, kfac, scales, placed, sim, closed=True, cfl=CFL, dim=2,
                            nut=NUT_PEAK * h if model == "les" else 0.0))
    steps = math.ceil(t_end / dt) if math.isfinite(t_end) else 0
    cost = physics_cost(placed, bool(cfd.record_diagnostics), cfd.adaptive_dt == "pid", 2)
    sps = estimate_steps_per_second(dev, cells, advection=advection, turbulence=turbulence, dtype=precision) / cost
    overhead = (3.0 if dclass == "gpu" else 1.5) + cells * 2e-6
    est = steps / (sps * loop_efficiency(dev, cells, live_view=True)) + overhead if steps else float("nan")
    est_headless = steps / (sps * loop_efficiency(dev, cells, live_view=False)) + overhead if steps else float("nan")
    imm = []
    for b, (th_mode, th_val) in zip(bodies_wind, placed.body_thermal):
        center = b.center if b.center is not None else tuple(P.centroid(b.points))
        b.center = (float(center[0]), float(center[1]))
        imm.append(ImmersedBody(points=np.asarray(b.points, dtype=np.float64) / L, name=b.name, role=b.role,
                                omega=float(b.omega or 0.0) * L / U, center=(center[0] / L, center[1] / L),
                                thermal=th_mode, thermal_value=th_val))
    mr = design.moment_ref
    phys_kw = _solver_physics_2d(placed, cfd)
    cfg = SolverConfig(
        nx=nx, ny=ny, h=h, x0=extent[0] / L, y0=extent[2] / L, re=re_sim, bodies=imm, domain="closed",
        bc_left="noslip", bc_right="noslip", bc_top="noslip", bc_bottom="moving" if moving else "noslip",
        ground_speed=float(cond.speed) / U if moving else 1.0, advection=advection,
        turbulence="none" if model == "laminar" else LES_MODEL, cfl=CFL,
        force_ref=max(float(design.force_ref), 1e-12) / L, moment_ref=(mr[0] / L, mr[1] / L), dtype=precision,
        device=dev, use_cuda_graphs=True, dye=bool(placed.dye_channels), perturb=True, **phys_kw)

    # -- explanations
    notes.insert(0, f"Closed box: {units.sig(W, 3)} m × {units.sig(H, 3)} m with no-slip walls on every side "
                    + ("(from Domain ▸ Extent)" if explicit else
                       "(around the design and the placed items, one size of room on every side; set Domain ▸ "
                       "Extent to fix the walls)")
                    + (", the floor moving at the flow speed (a driven cavity)" if moving else "")
                    + "; gravity along −y. There is no free stream: the reference velocity U = "
                    f"{units.sig(U, 3)} m/s is the {scales.speed_basis}, and coefficients use q = ½ρU².")
    notes.append(f"Grid: {nx:,} × {ny:,} = {cells:,} cells of {units.sig(h_m * 1000, 3)} mm, set by {why}.")
    if model == "les":
        notes.append(f"Flow model: WALE large-eddy model at Re = {units.fmt_reynolds(re_sim)} (U L/ν).")
    else:
        notes.append(f"Flow model: laminar at Re = {units.fmt_reynolds(re_sim)} (U L/ν); advection uses "
                     + ("central differences." if advection == "central" else "third-order upwind differences."))
    if not live:
        notes.append(f"Duration: {t_end:.4g} L/U ({dur_txt}, {fmt_time(t_end * L / U)} of real flow); averages from "
                     f"t = {t_avg:.4g}, and the run stops early once the monitored quantities have settled.")
        notes.append(f"Time step ≈ {units.sig(dt, 2)} L/U: about {steps:,} steps, estimated {fmt_duration(est)} on "
                     f"{device_name}" + (f" (the placed physics costs about {100 * (cost - 1):.0f} % per step)."
                                         if cost > 1.02 else ".") + peak_note(ratio, u_peak, peak_basis))
    notes.extend(placed.notes)
    warnings.extend(w for w in placed.warnings if w not in warnings)
    t_notes, t_warn = thermal_checks(scales, sim, h_m=h_m, n_bl=n_bl, bl_target=BL_CELLS_MIN, closed=True,
                                     live=live, dim=2)
    notes.extend(t_notes)
    warnings.extend(t_warn)
    unbounded = unbounded_heating_note(scales, placed, sim, box, 2)
    if unbounded and not live:
        notes.append(unbounded)
    if not scales.thermal and not placed.fans and not placed.ports and not moving:
        raise ValueError("Nothing drives the flow in this closed box: add a fan or a jet, heat a wall or a body with "
                         "heat transfer on, or move the floor.")
    if placed.fans:
        # 2D rooms with fans have no grid-converged speeds (theory_ns2d.md 18.4): a warning, so the
        # Results banner and the report carry it with every number of the run
        warnings.append(room_2d_fan_warning(None if live else t_end))
    if placed.fans and not live:
        notes.append("A fan in a closed box drives the air until the walls' friction balances its thrust, so the "
                     "box recirculates faster than the fan's own disc velocity (the reference used here). The time "
                     "step is planned on the peak such boxes measured (max|u| + max|v| about 3 times twice the disc "
                     "velocity at Draft); later runs of the same kind of box (a fan alone, or with jets or heaters, "
                     "with or without bodies, on a similar grid) refine it, by at most 1.5 times per run.")
    if polys:
        warnings.extend(_gap_report(bodies_wind, h_m))
    warnings = merge_warnings(warnings)
    return SimPlan(
        solver_config=cfg, design=design, sim=sim, length_ref=L, speed=U, fluid=cond.fluid, re_real=re_sim,
        re_sim=re_sim, turbulence=turbulence, t_end=t_end, t_avg_start=t_avg, dt=dt, steps_estimate=steps,
        est_seconds=est, cells=cells, cells_per_ref=float(L / h_m), bodies_wind=bodies_wind, extent=extent,
        notes=notes, warnings=warnings, device=dev, device_name=device_name, precision=precision,
        advection=advection, quality=quality, ground=ground, ground_moving=moving, strouhal_length=L,
        blockage=float("nan"), ref_label="reference length", live=live, capped=False,
        geometry_class=_geometry_class(design), conv_rtol=CONVERGENCE_RTOL[qi], blockage_limit=float("nan"),
        est_seconds_headless=est_headless, bl_resolved=bool(resolved), domain_mode="closed", physics=scales,
        placed=placed, monitors=plan_monitors(scales, placed, closed=True), calibration_kind=cal_kind,
        step_cost=cost, speed_ratio=ratio, peak_basis=peak_basis, est_seconds_planned=est,
        thermal_grid=(float(h_m), float(n_bl), float(BL_CELLS_MIN)) if scales.thermal else None)


def _plan(design: Design, sim: Simulation, device: str | None, *, live: bool) -> SimPlan:
    """Shared implementation of :func:`plan_simulation` and :func:`plan_live`."""
    if getattr(sim.cfd, "domain_mode", "tunnel") == "closed":
        return _plan_closed(design, sim, device, live=live)
    if not design.bodies:
        raise ValueError("The design has no bodies to simulate.")
    cond, cfd = sim.conditions, sim.cfd
    U, L = float(cond.speed), float(design.length_ref)
    if not (U > 0 and math.isfinite(U)):
        raise ValueError("The flow speed must be positive.")
    if not (L > 0 and math.isfinite(L)):
        raise ValueError("The design's reference length must be positive.")
    if not (cond.fluid.density > 0 and cond.fluid.viscosity > 0):
        raise ValueError("The fluid needs a positive density and viscosity.")
    notes: list[str] = []
    warnings: list[str] = []
    quality = cfd.quality if cfd.quality in QUALITY_ORDER else "standard"
    qi = QUALITY_ORDER.index(quality)

    dev = _resolve_device(device if device not in (None, "", "auto") else cfd.device, warnings)
    dclass = _device_class(dev)
    precision = _resolve_precision(cfd.precision, dev, live, warnings)
    dinfo = devmod.info(dev)
    device_name = _device_name(dev, dinfo)

    # -- 1. wind frame -----------------------------------------------------------
    ground = bool(sim.ground.enabled)
    moving = ground and bool(sim.ground.moving)
    alpha = float(cond.alpha_deg)
    bodies_wind = wind_frame_bodies(design, alpha, ground=ground)
    names = _unique_names(bodies_wind)
    for b, n in zip(bodies_wind, names):
        b.name = n
    if ground:
        low = min(float(np.asarray(b.points)[:, 1].min()) for b in bodies_wind)
        if low < -1e-9 * L:
            raise ValueError(f"At {alpha:g}° the geometry reaches {units.sig(-low * 1000, 3)} mm below the road. "
                             "Reduce the pitch angle or raise the vehicle.")
    elif design.kind == "vehicle" or (design.source or {}).get("ground"):
        notes.append("The road is switched off: the vehicle is simulated in free air, as if in flight. "
                     "Turn the road on in Setup for ground effect.")
    # live sessions must contain every pitch the slider can reach
    geom_for_domain = bodies_wind
    if live:
        pitch = LIVE_PITCH_GROUND_DEG if ground else LIVE_PITCH_DEG
        geom_for_domain = [b for a in (-pitch, 0.0, pitch)
                           for b in wind_frame_bodies(design, a, ground=ground)]
    sz = _sizes(geom_for_domain, ground)
    sz0 = _sizes(bodies_wind, ground)

    # -- 2. domain (metres) ---------------------------------------------------------
    geo = _geometry_class(design)
    dom = DOMAIN[geo]
    if geo == "airfoil":
        sx, sy = L, L
    elif geo == "vehicle":
        # the vehicle "height": road to roof with a road (what sets the shear layers
        # and the wake on a road), the frontal height in free air
        sx, sy = L, max(sz0["ymax"] if ground else sz0["frontal"], 1e-6 * L)
    else:
        sx = sy = sz["s"]
    pref = {"up": dom["up"][qi] * sx, "down": dom["down"][qi] * sx}
    mins = {"up": dom["min"]["up"] * sx, "down": dom["min"]["down"] * sx}
    # Blockage is judged on the geometry at the planned angle (a live session's
    # envelope over all pitch angles only has to fit inside, with a margin). With
    # a road the vehicle and its mirror image in the road fill 2 h of a 2 H
    # channel, so the ratio is (height above the road) / (domain height).
    blk_limit = BLOCKAGE_LIVE if live else BLOCKAGE_MAX[geo][qi]
    blocking = sz0["frontal_above"] if ground else sz0["frontal"]
    if geo == "vehicle":
        # the height is set by the blockage alone and never traded for cells:
        # everything the rest of the planner does keeps it
        pref["height"] = mins["height"] = blocking / blk_limit
    else:
        pref["height"] = max(dom["height"][qi] * sy, blocking / blk_limit)
        mins["height"] = max(dom["min"]["height"] * sy, blocking / (1.6 * blk_limit))
    fit = (sz["ymax"] + (0.5 * sy if live else 0.0)) if ground else (sz["frontal"] + (sy if live else 0.0))
    pref["height"] = max(pref["height"], fit)
    mins["height"] = max(mins["height"], fit)
    user_dom = {}
    if cfd.domain_upstream:
        user_dom["up"] = cfd.domain_upstream * L
    if cfd.domain_downstream:
        user_dom["down"] = cfd.domain_downstream * L
    if cfd.domain_height:
        user_dom["height"] = max(cfd.domain_height * L, 1.05 * (sz["ymax"] - (0.0 if ground else sz["ymin"])))
    pref.update(user_dom)
    mins.update(user_dom)
    mins = {k: min(mins[k], v) for k, v in pref.items()}
    span_x = sz["xmax"] - sz["xmin"]

    def area(ext: dict) -> float:
        return (ext["up"] + span_x + ext["down"]) * ext["height"]

    # -- 3. resolution --------------------------------------------------------------
    s_over_l = sz["s"] / L

    def preset_cpl(q: int) -> float:
        """The quality preset as cells per length_ref."""
        per_size = CELLS_PER_SIZE[geo][dclass][q]
        if geo == "generic":
            return per_size / s_over_l
        if geo == "vehicle":
            return per_size * L / sy
        return float(per_size)

    if cfd.cells_per_ref:
        cpl_target = float(cfd.cells_per_ref)
        user_cpl = True
    else:
        cpl_target = preset_cpl(qi)
        user_cpl = False
    if live:
        cpl_target = min(cpl_target, preset_cpl(0))
    thin = _thin_features(design, bodies_wind, sz0, ground)
    n_thin = THIN_CELLS[qi] if not live else LIVE_THIN_CELLS
    cpl_need = max((n_thin * L / t for _, t in thin), default=0.0)
    cpl = cpl_target if user_cpl else max(cpl_target, cpl_need)
    budget = _cell_budget(dclass, geo, qi, live=live)
    if dclass == "gpu" and dinfo.memory_gb and dinfo.memory_gb < 4.0:
        budget = int(budget * dinfo.memory_gb / 4.0)          # small GPUs: stay well inside memory
    ext = dict(pref)
    shrunk = budget_limited = False

    def cells_at(c: float, e: dict) -> float:
        return area(e) * (c / L) ** 2

    if not user_cpl and cells_at(cpl, ext) > budget:
        # shrink the domain toward its minimum first (blockage/inlet effects are a
        # few per cent; an under-resolved boundary layer is worse), then coarsen
        lo, hi = 0.0, 1.0
        if cells_at(cpl, {k: mins[k] for k in ext}) <= budget:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                e = {k: pref[k] - mid * (pref[k] - mins[k]) for k in ext}
                lo, hi = (lo, mid) if cells_at(cpl, e) <= budget else (mid, hi)
            lam = hi
        else:
            lam = 1.0
        ext = {k: pref[k] - lam * (pref[k] - mins[k]) for k in ext}
        shrunk = lam > 1e-3
        if cells_at(cpl, ext) > budget:
            cpl = math.sqrt(budget / area(ext)) * L
            budget_limited = True
    # whole cells per reference length; round *up* when a thin-feature target set
    # the value, so "4 cells across the clearance" is not 3.99 after rounding
    cpl = max(round(cpl) if (budget_limited or user_cpl) else math.ceil(cpl - 1e-6), 8)
    # grid, rounded up to FFT-friendly sizes; the extra room goes downstream and up
    width_m = ext["up"] + span_x + ext["down"]

    def grid_at(c: int) -> tuple[int, int]:
        return fft_friendly(math.ceil(width_m * c / L)), fft_friendly(math.ceil(ext["height"] * c / L))

    nx, ny = grid_at(cpl)
    if not user_cpl:
        # rounding both sizes up can add ~15 % cells: give back a little resolution
        # rather than exceed the budget (memory and run time are what it protects)
        while nx * ny > budget and cpl > 8:
            cpl -= 1
            nx, ny = grid_at(cpl)
    h_m = L / cpl
    h = 1.0 / cpl                                # solver units (length / L)
    x0_m = sz["xmin"] - ext["up"]
    if ground:
        y0_m = 0.0
    else:
        yc = 0.5 * (sz["ymin"] + sz["ymax"])
        y0_m = yc - 0.5 * ny * h_m
    extent = (x0_m, x0_m + nx * h_m, y0_m, y0_m + ny * h_m)
    cells = nx * ny
    _check_memory(cells, precision, dclass, dinfo, user_cpl=user_cpl, geo=geo)
    dom_h = extent[3] - extent[2]
    blockage = sz0["frontal_above"] / dom_h if ground else sz0["frontal"] / dom_h
    # placed physics (DESIGN.md 10): scales on the free stream, solver specs, checks
    scales: PhysicsScales | None = None
    placed: Placed | None = None
    if has_placed_physics(sim):
        scales = physics_scales(design, sim, 2)
        placed = placed_physics_2d(design, sim, scales, alpha_deg=alpha, extent=extent, h_m=h_m,
                                   bodies_wind=bodies_wind, closed=False)

    # -- 4. Reynolds number and flow model -----------------------------------------------
    re_real = cond.reynolds(L)
    re_resolvable = (5.0 / (BL_CELLS_MIN * h)) ** 2          # Blasius delta(L) = 5 L / sqrt(Re)
    capped = cap_from_grid = False
    if cfd.reynolds_mode == "capped":
        # The cap is a property of the simulation (simulation.defaults fixes it when the
        # simulation is created), so a change of quality or device refines the grid of the
        # same flow. Only a simulation without a stored cap (projects from before it was
        # stored, a capped vehicle) falls back to what this grid resolves - and is told
        # that its Reynolds number now depends on the grid.
        if cfd.reynolds_cap:
            cap = float(cfd.reynolds_cap)
        else:
            cap, cap_from_grid = re_resolvable, True
        re_sim = min(re_real, cap)
        capped = re_sim < 0.999 * re_real
    else:
        re_sim = re_real
    re_check = re_sim * (LIVE_SPEED_RANGE[1] if live and not capped else 1.0)
    n_bl = 5.0 / (h * math.sqrt(re_check))
    cell_re = re_check * h
    resolved = n_bl >= BL_CELLS_MIN
    if cfd.turbulence == "les":
        model = "les"
    elif cfd.turbulence == "laminar":
        model = "laminar"
    else:
        model = "laminar" if resolved else "les"
    advection = "central" if (model == "laminar" and cell_re <= CENTRAL_CELL_RE and not live) else "upwind3"
    turbulence = "laminar" if model == "laminar" else f"les-{LES_MODEL}"

    # -- 5. duration ---------------------------------------------------------------
    t_end, t_avg = _durations(geo, qi, s_over_l)
    if cfd.duration:
        t_end = float(cfd.duration)
        if not cfd.average_from:
            t_avg = min(t_avg, 0.5 * t_end)
    if cfd.average_from is not None and cfd.average_from > 0:
        t_avg = float(cfd.average_from)
    if t_avg >= 0.9 * t_end:
        warnings.append(f"Averaging was set to start at t = {t_avg:g}, too close to the end "
                        f"(t = {t_end:g}); the last half of the run is averaged instead.")
        t_avg = 0.5 * t_end
    if live:
        t_end, t_avg = math.inf, 0.0

    # -- time step and run-time estimate -------------------------------------------------
    u_peak, peak_basis = peak_speed(scales, sim, 2) if scales is not None else (U, "free-stream speed")
    if qi == 0 and not live and not cfd.duration:
        # a Draft run whose plume outruns its free stream: the step follows the plume, so the free
        # stream's duration would cost u_peak/U times the steps (physics.draft_buoyant_duration)
        t_cap, cap_note = draft_buoyant_duration(scales, t_end, u_peak)
        if cap_note:
            t_end = t_cap
            if not (cfd.average_from and 0 < float(cfd.average_from) < 0.9 * t_end):
                t_avg = 0.5 * t_end
            notes.append(cap_note)
    ratio = max(1.0, u_peak / U)
    cal_kind = calibration_kind(geo, placed, closed=False, natural=False, speed_ratio=ratio,
                                source=peak_source(peak_basis))
    if ratio <= PEAK_RATIO_MIN:
        ratio = 1.0                       # a jet barely faster than the stream: the class factor covers it
    kfac = _speed_factor(cal_kind, model, float(cpl)) * ratio
    # the solver's rule (theory_ns2d.md section 5): advective CFL limit with the
    # measured (max|u| + max|v|) / U, viscous limit with the peak eddy viscosity
    nu_max = 1.0 / re_sim + (NUT_PEAK * h if model == "les" else 0.0)
    dt = min(CFL * h / kfac, 0.25 * h * h / nu_max)
    if placed is not None:
        dt = min(dt, physics_dt(h, kfac, scales, placed, sim, closed=False, cfl=CFL, dim=2,
                                nut=NUT_PEAK * h if model == "les" else 0.0))
    steps = math.ceil(t_end / dt) if math.isfinite(t_end) else 0
    cost = physics_cost(placed, bool(cfd.record_diagnostics), cfd.adaptive_dt == "pid", 2)
    sps = estimate_steps_per_second(dev, cells, advection=advection, turbulence=turbulence, dtype=precision) / cost
    overhead = (3.0 if dclass == "gpu" else 1.5) + cells * 2e-6     # build, graph capture, outputs
    if (design.kind == "airfoil" and len(design.bodies) == 1 and not ground
            and math.isfinite(re_real) and re_real >= POLAR_CHECK_RE):
        overhead += POLAR_CHECK_S                                   # the runner's polar cross-check
    # the app always runs with the live view (frames cost wall time); scripts and the CLI do not
    est = steps / (sps * loop_efficiency(dev, cells, live_view=True)) + overhead if steps else float("nan")
    est_headless = (steps / (sps * loop_efficiency(dev, cells, live_view=False)) + overhead
                    if steps else float("nan"))

    # -- solver config ---------------------------------------------------------------
    omega_note = None
    imm = []
    rotate_wheels = moving and bool(sim.ground.rotate_wheels)
    wheels_touching = []
    body_thermal = placed.body_thermal if placed is not None else [("adiabatic", 0.0)] * len(bodies_wind)
    for b, (th_mode, th_val) in zip(bodies_wind, body_thermal):
        pts_nd = np.asarray(b.points, dtype=np.float64) / L
        omega_si = float(b.omega or 0.0)        # a spinning body in free flow (Magnus) keeps its own
        center = tuple(b.center) if b.center is not None else tuple(P.centroid(b.points))
        if b.role == "wheel":
            (cx, cy), r = _wheel_geometry(b)
            center = (cx, cy)
            if ground:
                if float(np.asarray(b.points)[:, 1].min()) < 1.5 * h_m:
                    wheels_touching.append(b.name)
                omega_si = 0.0
                if rotate_wheels and r > 0:
                    # rolling with the belt: the contact point moves at +U with the road,
                    # so the wheel turns counter-clockwise, omega = +U / r (solver convention)
                    omega_si = U / r
                    omega_note = (f"Wheels roll with the belt at {units.sig(omega_si, 3)} rad/s "
                                  f"(counter-clockwise, omega = U / r).")
        b.center = (float(center[0]), float(center[1]))
        b.omega = omega_si
        imm.append(ImmersedBody(points=pts_nd, name=b.name, role=b.role, omega=omega_si * L / U,
                                center=(center[0] / L, center[1] / L), thermal=th_mode, thermal_value=th_val))
    mr = design.moment_ref
    cfg = SolverConfig(
        nx=nx, ny=ny, h=h, x0=x0_m / L, y0=y0_m / L, re=re_sim, bodies=imm,
        inlet="uniform", bc_top="slip",
        bc_bottom=("moving" if moving else "noslip") if ground else "slip",
        ground_speed=1.0, advection=advection,
        turbulence="none" if model == "laminar" else LES_MODEL,
        cfl=CFL, force_ref=design.force_ref / L, moment_ref=(mr[0] / L, mr[1] / L),
        dtype=precision, device=dev, use_cuda_graphs=True, dye=True, perturb=True,
        **_solver_physics_2d(placed, cfd))

    # -- 6. explanations ----------------------------------------------------------------
    ref_label = {"airfoil": "chord", "vehicle": "vehicle length"}.get(geo, "reference length")
    if geo == "airfoil":
        st_len = L
    elif geo == "vehicle":
        st_len = float(design.force_ref)
    else:
        st_len = sz0["frontal"]
    if ground:
        notes.append(f"Domain: {_in_refs(ext['up'], L)} {ref_label}s of road ahead of the design, "
                     f"{_in_refs(extent[1] - sz['xmax'], L)} behind it, and {units.sig(dom_h, 3)} m of headroom "
                     f"({units.sig(dom_h / max(sz['ymax'], 1e-30), 2)}× its height, {100 * blockage:.1f} % blockage) "
                     "under a frictionless ceiling. The "
                     + ("belt moves with the free stream, as on a rolling-road wind tunnel." if moving else
                        "floor is fixed, like a wind tunnel without a belt: a boundary layer grows along it."))
    else:
        notes.append(f"Domain: {_in_refs(ext['up'], L)} {ref_label}s upstream, {_in_refs(extent[1] - sz['xmax'], L)} "
                     f"downstream and {_in_refs(dom_h, L)} {ref_label}s tall ({100 * blockage:.1f} % blockage) with "
                     "frictionless walls top and bottom.")
    if not live and blockage > 1.05 * blk_limit and (geo == "vehicle" or blockage > 0.08):
        warnings.append(f"The domain is only {units.sig(1.0 / blockage, 2)}× as tall as the "
                        f"{'vehicle' if geo == 'vehicle' else 'body'} ({100 * blockage:.0f} % blockage; "
                        f"{quality.capitalize()} aims for {100 * blk_limit:.0f} %). In 2D the air cannot pass around "
                        "the sides, so the flow over it speeds up and drag and lift read high. Differences between "
                        "designs simulated the same way remain meaningful; a taller domain reduces the effect.")
    if shrunk:
        notes.append("The domain was made more compact than usual to keep the cell count within the "
                     f"{dclass.upper()} budget of {budget:,} cells; this costs a little accuracy (blockage, "
                     "inlet proximity) but keeps the boundary layers resolved.")
    preset_txt = (f"{cpl} cells per {ref_label} (set by hand)" if user_cpl else
                  f"{cpl} cells per {ref_label} ({quality.capitalize()} quality on the {dclass.upper()})")
    notes.append(f"Grid: {nx:,} × {ny:,} = {cells:,} cells of {units.sig(h_m * 1000, 3)} mm; {preset_txt}. "
                 f"Sizes are rounded to FFT-friendly numbers for the pressure solver.")
    underbody = None if live else _underbody_warning(geo, ground, sz0["clearance"], h_m, quality)
    for what, t in thin:
        nc = t / h_m
        if underbody and what == "the ground clearance":
            continue                               # said more usefully by the underbody warning
        if nc < THIN_SLACK * n_thin and not live:
            warnings.append(f"{what[0].upper() + what[1:]} spans only {nc:.1f} grid cells (target {n_thin:.0f}); "
                            "the flow around it is under-resolved. "
                            + ("Choose a finer quality or run on a GPU." if not user_cpl
                               else "Increase the cells per reference length."))
        elif not live and (cpl > cpl_target + 0.5) and abs(cpl - n_thin * L / t) < 1.0:
            notes.append(f"The resolution was raised so that {what} spans {nc:.0f} cells.")
    if cpl < THIN_SLACK * cpl_target and not live:
        warnings.append(f"The {dclass.upper()} cell budget ({budget:,}) limits the grid to {cpl} cells per "
                        f"{ref_label} instead of {cpl_target:.0f}. Results are coarser than the "
                        f"{quality.capitalize()} preset intends.")
    if user_cpl and cells > 1.5 * CELL_BUDGET[dclass][-1]:
        warnings.append(f"{cells:,} cells is far above what this device handles comfortably "
                        f"({CELL_BUDGET[dclass][-1]:,}); expect a long run and high memory use.")
    warnings.extend(_gap_report(bodies_wind, h_m))
    if wheels_touching and ground:
        warnings.append("The wheels touch the road. In 2D a wheel is an infinitely wide cylinder, so this seals "
                        "the underbody: no air flows under the car and the lift is not meaningful. Remove the "
                        "wheels or lift them slightly (wheel gap) if the underbody flow matters.")
    if omega_note:
        notes.append(omega_note)

    # Reynolds number explanation
    re_txt = units.fmt_reynolds
    bl_txt = (f"a laminar boundary layer is about {units.sig(5.0 / math.sqrt(re_check), 2)} "
              f"{ref_label}s thick, i.e. {n_bl:.1f} cells")
    if capped:
        notes.append(f"Reynolds number capped at {re_txt(re_sim)} (real: {re_txt(re_real)}). The flow is "
                     f"simulated as if the fluid were {units.sig(re_real / re_sim, 3)}× more viscous; "
                     "coefficients come from the capped flow and forces are scaled to the real speed and "
                     "density.")
        if scales is not None and scales.thermal:
            # the heat rates follow the forces' convention (simulation.physics.PhysicsScales.conduction_unit)
            notes.append(f"Heat transfer in the capped flow: the Nusselt number comes from it (Péclet number "
                         f"{re_txt(re_sim * scales.pr)} = Re × Pr, not the real {re_txt(re_real * scales.pr)}), and "
                         "heat rates are that Nusselt number times the real conductivity, Q = Nu k ΔT P / L. Use the "
                         "real Reynolds number (Setup ▸ Solver) for a heat transfer that depends on it.")
        if cap_from_grid and not live:
            warnings.append(f"No Reynolds cap is set, so the cap is the highest Reynolds number this grid resolves "
                            f"({re_txt(re_sim)}): it changes with the quality and the device, and runs at another "
                            "quality then simulate a different flow. Set the Reynolds cap in Setup (Solver) to "
                            "keep the flow fixed.")
        elif not resolved and not live and model == "les":
            # (forced laminar flow on such a grid has its own warning below)
            warnings.append(f"The Reynolds cap ({re_txt(re_sim)}) is above what this grid resolves "
                            f"({re_txt(re_resolvable)} at {cpl} cells per {ref_label}): {bl_txt}, so a "
                            "large-eddy model represents the unresolved scales. Choose a finer quality or a "
                            "lower cap for a resolved flow.")
    if model == "les":
        if not resolved:
            notes.append(f"Flow model: at Re = {re_txt(re_check)} {bl_txt} -- too thin for this grid -- so a "
                         "WALE large-eddy model represents the unresolved scales and third-order upwind "
                         "advection keeps the run stable.")
        else:
            notes.append(f"Flow model: WALE large-eddy model at Re = {re_txt(re_sim)}, as requested in the "
                         "settings (the grid would resolve this flow without one).")
    else:
        why = "central differences (cell Reynolds number ≤ 2)" if advection == "central" else \
            "third-order upwind differences"
        notes.append(f"Flow model: laminar at Re = {re_txt(re_sim)}; {bl_txt}, so no turbulence model is "
                     f"needed. Advection uses {why}.")
        if not resolved:
            warnings.append(f"Laminar flow was requested, but {bl_txt}: the boundary layers are not resolved and "
                            "the run may be noisy or unstable. 'Auto' would use a large-eddy model.")
    single_airfoil = design.kind == "airfoil" and len(design.bodies) == 1
    if re_sim > 1e4 and not resolved:
        if geo == "airfoil":
            # measured: NACA 2412, Re 1e6, 4 deg, Draft -> Cl 0.16 / Cd 0.096 against
            # the viscous polar's 0.72 / 0.007 (see AIRFOIL_CAP_RE)
            msg = (f"Lift and drag are not reliable at this Reynolds number: at Re = {re_txt(re_sim)} the "
                   "boundary layers are much thinner than a grid cell and this 2D solver has no wall or "
                   "transition model, so an airfoil's Cl comes out several times too low and its Cd several "
                   "times too high. Use the flow pattern (separation, wake) and the differences between "
                   "designs run the same way. ")
            msg += ("For Cl and Cd at this Reynolds number use the Polar analysis (XFOIL-class); the result "
                    "lists the polar value next to the CFD one." if single_airfoil else
                    "Set the Reynolds number to Capped for a flow the grid resolves.")
        else:
            msg = (f"At Re = {re_txt(re_sim)} the boundary layers are much thinner than a grid cell and this "
                   "2D solver has no wall, transition or 3D-turbulence model: use the flow pattern and the "
                   "differences between designs run the same way; absolute Cd and Cl are indicative only.")
        warnings.append(msg)
    elif capped and geo == "airfoil" and re_real > AIRFOIL_CAP_RE:
        warnings.append(f"Cl and Cd describe the capped flow at Re = {re_txt(re_sim)}, not the real "
                        f"Re = {re_txt(re_real)}: a low-Reynolds airfoil has less lift and much more drag. "
                        + ("The result lists the viscous polar at the real Reynolds number next to them."
                           if single_airfoil else "Use them to compare designs run the same way."))
    elif re_sim > 1e4:
        notes.append(f"At Re = {re_txt(re_sim)} real flows have 3D turbulence a 2D simulation cannot "
                     "represent; compare designs run the same way rather than quoting absolute values.")
    elif re_sim > 200 and geo != "airfoil":
        notes.append(f"Above Re ≈ 200 real wakes behind bluff bodies become three-dimensional; this 2D "
                     f"result at {re_txt(re_sim)} is the idealised two-dimensional flow.")
    if geo == "vehicle" and not live:
        notes.append(VEHICLE_2D_NOTE)
    if geo == "airfoil" and not ground and not live and not user_dom:
        notes.append(AIRFOIL_DOMAIN_NOTE)
    if underbody:
        warnings.insert(0, underbody)          # the number it explains (Cl) is the one people quote

    conv_rtol = CONVERGENCE_RTOL[qi]
    if not live:
        tu = L / U
        notes.append(f"Duration: {t_end:g} convective times L/U ({fmt_time(t_end * tu)} of real flow); forces "
                     f"are averaged from t = {t_avg:g} (or later, where the start-up transient is found to end), "
                     f"and the run stops early once the mean drag is known to ±{100 * conv_rtol:g} % and "
                     f"the mean lift to ±{100 * CL_RTOL_FACTOR * conv_rtol:g} % (one standard error) and "
                     "neither drifts.")
        notes.append(f"Time step ≈ {units.sig(dt, 2)} L/U from the stability limit (CFL {CFL:g}): about "
                     f"{steps:,} steps, estimated {fmt_duration(est)} on {device_name}"
                     + (" (calibrated estimate; refined after each run)." if not _RATE_CACHE else ".")
                     + peak_note(ratio, u_peak, peak_basis))
        if est > 3600:
            warnings.append(f"Estimated run time {fmt_duration(est)}. Draft quality or a GPU would be much faster.")
    # geometry checks the grid makes relevant (thin trailing edges, gaps under 2 cells, ...)
    try:
        from ..geometry import check_design
        base = set(check_design(design))
        for m in check_design(design, cpl):
            if m in base or m in warnings or m.startswith("The gap between"):     # gaps: _gap_report
                continue
            # every sharp trailing edge is thinner than two cells somewhere: information, not alarm
            (notes if m.startswith("Trailing edge") else warnings).append(m)
    except Exception as exc:  # noqa: BLE001 - geometry diagnostics are advisory only
        notes.append(f"(Geometry diagnostics unavailable: {type(exc).__name__}.)")
    monitors: tuple[str, ...] = ()
    if placed is not None:
        notes.extend(placed.notes)
        warnings.extend(w for w in placed.warnings if w not in warnings)
        t_notes, t_warn = thermal_checks(scales, sim, h_m=h_m, n_bl=n_bl, bl_target=BL_CELLS_MIN, closed=False,
                                         live=live, dim=2)
        notes.extend(t_notes)
        warnings.extend(t_warn)
        monitors = plan_monitors(scales, placed, closed=False)
        if cost > 1.02 and not live:
            notes.append(f"The placed physics costs about {100 * (cost - 1):.0f} % more time per step (measured "
                         "per feature, theory_ns2d.md 20.3); the estimate includes it.")

    plan = SimPlan(
        solver_config=cfg, design=design, sim=sim, length_ref=L, speed=U, fluid=cond.fluid,
        re_real=re_real, re_sim=re_sim, turbulence=turbulence, t_end=t_end, t_avg_start=t_avg,
        dt=dt, steps_estimate=steps, est_seconds=est, cells=cells, cells_per_ref=float(cpl),
        bodies_wind=bodies_wind, extent=extent, notes=notes, warnings=warnings,
        device=dev, device_name=device_name, precision=precision, advection=advection,
        quality=quality, ground=ground, ground_moving=moving, strouhal_length=st_len,
        blockage=blockage, ref_label=ref_label, live=live, capped=capped, geometry_class=geo,
        conv_rtol=conv_rtol, blockage_limit=blk_limit, est_seconds_headless=est_headless,
        bl_resolved=bool(resolved), domain_mode="tunnel", physics=scales, placed=placed, monitors=monitors,
        calibration_kind=cal_kind, step_cost=cost, speed_ratio=ratio, peak_basis=peak_basis,
        thermal_grid=(float(h_m), float(n_bl), float(BL_CELLS_MIN)) if scales is not None and scales.thermal else None)
    plan.warnings = merge_warnings(plan.warnings)
    if not live and plan.calibration_kind.endswith("-plume") and bodies_wind:
        top = max(float(np.max(np.asarray(b.points)[:, 1])) for b in bodies_wind)
        pn = plume_note(extent[3] - top, u_peak, L / U)
        if pn:
            plan.notes.append(pn)
    if not live and getattr(sim.cfd, "compute_sensitivity", False):
        _add_sensitivity_estimate(plan)
    plan.est_seconds_planned = plan.est_seconds
    return plan


def _add_sensitivity_estimate(plan: SimPlan) -> None:
    """Fold the sensitivity map's reverse sweep into the run-time estimate.

    The map runs after the flow solve and can take as long as the solve itself
    (measured on the Draft airfoil: ~27 s on the GPU, ~60 s on a CPU), so an
    estimate without it would be wrong by up to a factor of two. The model lives
    with the sensitivity code (:func:`aerolume.optimize.sensitivity.estimate_seconds`);
    a failure there must never stop planning, so it only drops the extra time.
    """
    try:
        from ..optimize.sensitivity import estimate_seconds as sensitivity_seconds
        extra = float(sensitivity_seconds(plan))
    except Exception as exc:  # noqa: BLE001 - advisory: the run itself does not depend on it
        plan.notes.append(f"(Sensitivity-map time not included in the estimate: {type(exc).__name__}.)")
        return
    if not math.isfinite(extra) or extra <= 0:
        return
    plan.est_seconds += extra
    plan.est_after_seconds += extra
    if math.isfinite(plan.est_seconds_headless):
        plan.est_seconds_headless += extra
    plan.notes.append(f"Sensitivity map: about {fmt_duration(extra)} more after the flow solve (included in the "
                      "estimate; a shedding flow takes 2-4 times longer, it averages whole periods from "
                      "several phases).")


def _check_cfd_analysis(sim: Simulation) -> None:
    """A wing analysis is not a CFD run: say where it goes instead of planning nonsense."""
    if getattr(sim, "analysis", "cfd") == "wing":
        raise ValueError("This simulation is a wing analysis (vortex lattice / lifting line), not a CFD run; "
                         "run it with simulation.run_wing().")


def plan_simulation(design: Design, sim: Simulation, *, device: str | None = None,
                    airfoil_coords: Callable[[str], np.ndarray] | None = None) -> SimPlan:
    """Plan a production CFD run of ``sim`` on ``design`` (DESIGN.md 5.4).

    ``device`` overrides ``sim.cfd.device`` ("auto", "cuda", "cpu", ...); an
    unavailable device falls back to the best available one with a warning.
    Raises ``ValueError`` with a user-readable message for impossible set-ups
    (no bodies, zero speed, a pitch that pushes the car into the road). Fast and
    side-effect free, so it can run on the GUI thread on every settings change.

    A 3D design (``design.dimension == 3``) gets a
    :class:`~aerolume.simulation.plan3d.SimPlan3D` (DESIGN.md 7.7);
    ``airfoil_coords(name)`` resolves "design:<id>" airfoils of a wing planform
    that has to be lofted (3D only).
    """
    if getattr(design, "dimension", 2) == 3:
        _check_cfd_analysis(sim)
        from .plan3d import plan_simulation3d
        return plan_simulation3d(design, sim, device=device, airfoil_coords=airfoil_coords)  # type: ignore[return-value]
    return _plan(design, sim, device, live=False)


def plan_live(design: Design, sim: Simulation, *, device: str | None = None,
              airfoil_coords: Callable[[str], np.ndarray] | None = None) -> SimPlan:
    """Plan for an explore (real-time) session: a small grid that runs at a few
    hundred steps per second, a domain that fits every pitch angle the slider
    can reach (+-25°, +-5° on a road) at up to 10 % blockage, float32 unless set
    otherwise, upwind advection, and a flow model chosen for the highest speed
    the explore slider can reach. 3D designs get :func:`.plan3d.plan_live3d`."""
    if getattr(design, "dimension", 2) == 3:
        from .plan3d import plan_live3d
        return plan_live3d(design, sim, device=device, airfoil_coords=airfoil_coords)  # type: ignore[return-value]
    return _plan(design, sim, device, live=True)
