"""Simulation layer: automatic set-up, production runs and real-time exploration.

Public API (DESIGN.md sections 5.4 and 7.7)::

    plan_simulation(design, sim, *, device=None, airfoil_coords=None) -> SimPlan | SimPlan3D
                                                                # plan.py (plan3d.py for 3D)
    default_cfd_settings(design, conditions) -> CFDSettings     # what a new simulation starts from
    default_analysis(design), default_simulation(design, ...)   # defaults.py ("wing" for wings)
    simulation_name(design, sim, *, by_reynolds=None) -> str    # "CFD 3D · 30 m/s · α 0°", "... · Re 100 · ..."
    fmt_estimate(seconds) -> str                                # '~5 min', for the Run button
    use_calibration_file(path)                                  # persist run-time measurements
    run_cfd(plan, *, progress, frame, cancel, view) -> CFDResult | CFD3DResult  # runner.py / runner3d.py
    run_polar(design, sim, *, progress, cancel) -> PolarResult
    run_simulation(design, sim, *, project=None, device=None, progress, frame, cancel, view, planned)
                                                                # any analysis, 2D or 3D: the one dispatcher
    run_wing(design, sim, *, progress, cancel, project=None, airfoil_coords=None) -> WingResult
                                                                # wing.py ("design:<id>" airfoils)
    ViewSettings, LiveFrame, LiveSession (3D: LiveSession3D)    # live.py / live3d.py

The submodules import torch (through the solver), so names are resolved lazily:
``import aerolume.simulation`` itself stays cheap.

``SUPPORTED_PHYSICS`` (DESIGN.md 10) is the one place that says which placed physics this
installation's planners and solvers really run, per dimension. The scripting API
(:func:`aerolume.api.supported_physics`), the AI assistant's run tools and the Setup page's
planner-gap check all read it, and refuse or warn about anything missing from it rather
than let a run silently drop a fan or a heater. It is a plain constant on purpose: reading
it must not import torch.
"""
from __future__ import annotations

import importlib
from typing import Any

#: The placed-physics features (``aerolume.api.PHYSICS_FEATURES`` names) each dimension runs,
#: checked feature by feature against the planners and runners in
#: tests/test_simulation_supported_physics.py:
#:
#: * ``emitters`` -- smoke channels and tracer particles (plan.py / plan3d.py ``Placed.emitters``,
#:   ns2d / ns3d ``SolverConfig.emitters``);
#: * ``thermal`` -- the energy equation, Boussinesq buoyancy, heated bodies and walls;
#: * ``heat_sources`` -- volumetric heaters (need ``thermal``);
#: * ``fans`` -- actuator discs / strips with thrust, pressure or velocity control;
#: * ``ports`` -- jets and vents;
#: * ``closed_domain`` -- ``CFDSettings.domain_mode == "closed"`` (plan._plan_closed, plan3d's
#:   closed planner);
#: * ``adaptive_dt`` -- the PID time-step control (``SolverConfig.adaptive_dt == "pid"``).
SUPPORTED_PHYSICS: dict[int, frozenset[str]] = {
    2: frozenset({"emitters", "thermal", "heat_sources", "fans", "ports", "closed_domain", "adaptive_dt"}),
    3: frozenset({"emitters", "thermal", "heat_sources", "fans", "ports", "closed_domain", "adaptive_dt"}),
}

_LAZY = {
    "SimPlan": "plan", "plan_simulation": "plan", "plan_live": "plan",
    "estimate_steps_per_second": "plan", "record_throughput": "plan",
    "default_cfd_settings": "plan", "fmt_estimate": "plan", "use_calibration_file": "plan",
    "run_cfd": "runner", "run_polar": "runner", "run_simulation": "runner",
    "airfoil_resolver": "wing",
    "ViewSettings": "live", "LiveFrame": "live", "LiveSession": "live", "FrameRenderer": "live",
    "default_analysis": "defaults", "default_simulation": "defaults", "default_wing_settings": "defaults",
    "simulation_name": "defaults",
    "SimPlan3D": "plan3d", "plan_simulation3d": "plan3d", "plan_live3d": "plan3d",
    "run_cfd3d": "runner3d", "run_wing": "wing",
    "LiveSession3D": "live3d", "FrameRenderer3D": "live3d",
}

__all__ = ["SUPPORTED_PHYSICS", "FrameRenderer", "FrameRenderer3D", "LiveFrame", "LiveSession", "LiveSession3D",
           "SimPlan", "SimPlan3D",
           "ViewSettings", "airfoil_resolver", "default_analysis", "default_cfd_settings", "default_simulation",
           "default_wing_settings", "estimate_steps_per_second", "fmt_estimate", "plan_live", "plan_live3d",
           "plan_simulation", "plan_simulation3d", "record_throughput", "run_cfd", "run_cfd3d", "run_polar",
           "run_simulation", "run_wing", "simulation_name", "use_calibration_file"]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
