"""Finite-wing analysis (XFLR5-like): vortex lattice and nonlinear lifting line on 2D section polars.

Public API (DESIGN.md section 7.6)::

    analyze_wing(wing, *, alpha_deg, speed, fluid, method="vlm"|"llt", viscous=True, ncrit=9, mach=0,
                 beta_deg=0, moment_ref=None, polar_provider=None, coords_provider=None,
                 ref_area=None, ref_chord=None, ref_span=None) -> WingPoint
    wing_polar(wing, alphas, *, (same keywords), sim_id="", progress=None, cancel=None,
               details=None) -> WingResult
    section_points_needed(wing, alphas, *, speed, fluid, ncrit, mach, beta_deg, polar_provider,
                          coords_provider) -> int      section points a viscous run would still compute
    SectionPolars(ncrit=9, mach=0, ..., coords_provider=None)     cached 2D viscous section polars
        .ensure(demands), .table(airfoil, re), .state() / .load_state(d) (persist the cache)
    planform_metrics(wing) -> dict(area, span, aspect_ratio, mac, mac_le_x, mac_y, taper, sweep_c4_deg, mean_chord)
    freestream_direction(alpha_deg, beta_deg) -> unit free stream in the wing frame

Only numpy / scipy and :mod:`aerolume.solvers.airfoil` are used (no torch, no Qt), so
the GUI can call it from a worker thread. Theory, assumptions and validity limits:
docs/theory_wing.md; validation: docs/validation_wing.md.
"""
from __future__ import annotations

from .geometry import WingGeometryError, planform_metrics
from .polars import PolarTable, SectionPolarError, SectionPolars, airfoil_key
from .solver import analyze_wing, freestream_direction, section_points_needed, wing_polar

__all__ = [
    "PolarTable",
    "SectionPolarError",
    "SectionPolars",
    "WingGeometryError",
    "airfoil_key",
    "analyze_wing",
    "freestream_direction",
    "planform_metrics",
    "section_points_needed",
    "wing_polar",
]
