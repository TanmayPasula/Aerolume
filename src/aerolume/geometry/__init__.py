"""Geometry: shape creation, import, polygon operations, SDF and parametrisation.

Public API (DESIGN.md section 5.1)::

    naca4, naca5, naca                         NACA generators (Selig order, CCW)
    read_airfoil_dat, format_airfoil_dat       Selig / Lednicer .dat files
    read_svg, read_dxf, read_image_silhouette, read_points_csv, import_file
    design_from_airfoil, design_from_polygons, check_design
    polygon ops: orient_ccw, signed_area, centroid, bbox, perimeter, resample, smooth,
                 remove_duplicates, close_gaps, is_self_intersecting,
                 repair_self_intersections, point_in_polygon, frontal_height, transform
    templates: list_templates, make, TemplateInfo, ParamSpec
    params: CSTParams, cst_fit, cst_eval, normal_basis, apply_normal_displacement
    sdf: signed_distance, volume_fraction          (torch; imported on first use)

Readers with heavy dependencies (ezdxf, Pillow/contourpy) and the torch-based SDF
are loaded lazily on first access, so importing this package -- which the GUI does
at start-up -- costs only numpy.
"""
from __future__ import annotations

import importlib
from typing import Any

from . import polygon, templates
from .airfoils import format_airfoil_dat, naca, naca4, naca5, read_airfoil_dat, write_airfoil_dat
from .csvio import read_points_csv
from .design import check_design, design_from_airfoil, design_from_polygons, normalize_airfoil
from .importer import SUPPORTED_EXTENSIONS, ImportResult, import_file
from .params import CSTParams, apply_normal_displacement, cst_eval, cst_fit, normal_basis, vertex_normals
from .polygon import (
    bbox,
    centroid,
    close_gaps,
    frontal_height,
    is_self_intersecting,
    orient_ccw,
    perimeter,
    point_in_polygon,
    remove_duplicates,
    repair_self_intersections,
    resample,
    signed_area,
    simplify_rdp,
    smooth,
    transform,
)
from .svg import read_svg
from .templates import ParamSpec, TemplateInfo, list_templates, make

_LAZY = {
    "read_dxf": ("dxf", "read_dxf"),
    "read_image_silhouette": ("image", "read_image_silhouette"),
    "signed_distance": ("sdf", "signed_distance"),
    "volume_fraction": ("sdf", "volume_fraction"),
    "sdf": ("sdf", None),
    "dxf": ("dxf", None),
    "image": ("image", None),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        return mod if attr is None else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SUPPORTED_EXTENSIONS", "CSTParams", "ImportResult", "ParamSpec", "TemplateInfo",
    "apply_normal_displacement", "bbox", "centroid", "check_design", "close_gaps", "cst_eval", "cst_fit",
    "design_from_airfoil", "design_from_polygons", "format_airfoil_dat", "frontal_height", "import_file",
    "is_self_intersecting", "list_templates", "make", "naca", "naca4", "naca5", "normal_basis",
    "normalize_airfoil", "orient_ccw", "perimeter", "point_in_polygon", "polygon", "read_airfoil_dat",
    "read_dxf", "read_image_silhouette", "read_points_csv", "read_svg", "remove_duplicates",
    "repair_self_intersections", "resample", "signed_area", "signed_distance", "simplify_rdp", "smooth",
    "templates", "transform", "vertex_normals", "volume_fraction", "write_airfoil_dat",
]
