"""Viscous–inviscid airfoil analysis (XFOIL-class polars) built on vendored mfoil.

Public API (DESIGN.md section 5.3)::

    prepare_coords(poly_or_design) -> (coords, chord, info)
    analyze(coords, re, alpha_deg, *, mach=0, ncrit=9, xtr_upper=1, xtr_lower=1,
            viscous=True, panels=199) -> PolarPoint
    polar(coords, re, alphas, *, ..., progress=None, cancel=None) -> PolarResult
    inviscid(coords, alpha_deg, *, mach=0, panels=199) -> PolarPoint

The solver is mfoil v2023-06-28 by K. J. Fidkowski (MIT licence), vendored in
``_mfoil.py`` with the minimal patches listed at the top of that file; the
vectorised kernels in ``_fast.py`` reproduce its hot loops to round-off.
Theory and validity limits: docs/theory_airfoil.md; validation against wind
tunnel data: docs/validation_airfoil.md.
"""
from __future__ import annotations

from .geometry import AirfoilGeometryError, prepare_coords
from .solver import Cancelled, alpha_range, analyze, inviscid, polar

__all__ = [
           "AirfoilGeometryError",
           "Cancelled",
           "alpha_range",
           "analyze",
           "inviscid",
           "polar",
           "prepare_coords",
]
