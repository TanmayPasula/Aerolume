"""3D geometry: mesh files, mesh operations, references, primitives, wings, voxelisation.

Public API (DESIGN.md section 7.4)::

    read_mesh, read_stl, read_obj, read_ply      STL (ASCII/binary), OBJ (groups), PLY (ASCII/binary)
    write_stl, write_obj, write_ply, guess_units, unit_scale, SUPPORTED_EXTENSIONS_3D
    mesh ops: transform, rotation_matrix, mirror, translate, merge, weld, remove_degenerate,
              compact, face_normals, vertex_normals, normals, face_areas, area, volume,
              is_watertight, boundary_edges, edge_counts, inconsistent_edges, components,
              mesh_info, bbox, ORIENT_PRESETS, orient_preset, auto_orient, orient_outward,
              place_on_road, centre_y, place_nose, symmetry_error, frontal_area, projection_mask,
              sample_surface, wheel_params, decimate
    design_from_meshes, check_design3d, find_wheels3d, part_gap
    primitives: sphere, box, cylinder, sedan
    wing: loft, planform, design_from_wing, tapered_wing, section_outline, naca_code
    voxel (torch): Grid, grid_around, voxelize, inside_mask, sdf_band, sdf_bands, solid_fraction

Everything except :mod:`.voxel` needs only numpy (and scipy inside a few
functions), so the GUI can import this package at start-up; the torch-based
voxeliser is loaded on first access, like :mod:`aerolume.geometry`'s SDF.
"""
from __future__ import annotations

import importlib
from typing import Any

from .design3d import check_design3d, design_from_meshes, find_wheels3d, part_gap
from .mesh_ops import (
    ORIENT_PRESETS,
    area,
    auto_orient,
    bbox,
    boundary_edges,
    centre_y,
    compact,
    components,
    decimate,
    edge_counts,
    face_areas,
    face_normals,
    frontal_area,
    inconsistent_edges,
    is_watertight,
    merge,
    mesh_info,
    mirror,
    normals,
    orient_outward,
    orient_preset,
    place_nose,
    place_on_road,
    projection_mask,
    remove_degenerate,
    rotation_matrix,
    sample_surface,
    symmetry_error,
    transform,
    translate,
    vertex_normals,
    volume,
    weld,
    wheel_params,
)
from .meshio import (
    SUPPORTED_EXTENSIONS_3D,
    UNIT_SCALES,
    guess_units,
    read_mesh,
    read_obj,
    read_ply,
    read_stl,
    unit_scale,
    write_obj,
    write_ply,
    write_stl,
)
from .primitives import box, cylinder, sedan, sphere
from .wing import design_from_wing, loft, naca_code, planform, section_outline, tapered_wing

_LAZY = {
    "Grid": ("voxel", "Grid"),
    "grid_around": ("voxel", "grid_around"),
    "voxelize": ("voxel", "voxelize"),
    "inside_mask": ("voxel", "inside_mask"),
    "sdf_band": ("voxel", "sdf_band"),
    "sdf_bands": ("voxel", "sdf_bands"),
    "solid_fraction": ("voxel", "solid_fraction"),
    "voxel": ("voxel", None),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        return mod if attr is None else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ORIENT_PRESETS",
    "SUPPORTED_EXTENSIONS_3D",
    "UNIT_SCALES",
    "Grid",
    "area",
    "auto_orient",
    "bbox",
    "boundary_edges",
    "box",
    "centre_y",
    "check_design3d",
    "compact",
    "components",
    "cylinder",
    "decimate",
    "design_from_meshes",
    "design_from_wing",
    "edge_counts",
    "face_areas",
    "face_normals",
    "find_wheels3d",
    "frontal_area",
    "grid_around",
    "guess_units",
    "inconsistent_edges",
    "inside_mask",
    "is_watertight",
    "loft",
    "merge",
    "mesh_info",
    "mirror",
    "naca_code",
    "normals",
    "orient_outward",
    "orient_preset",
    "part_gap",
    "place_nose",
    "place_on_road",
    "planform",
    "projection_mask",
    "read_mesh",
    "read_obj",
    "read_ply",
    "read_stl",
    "remove_degenerate",
    "rotation_matrix",
    "sample_surface",
    "sdf_band",
    "sdf_bands",
    "section_outline",
    "sedan",
    "solid_fraction",
    "sphere",
    "symmetry_error",
    "tapered_wing",
    "transform",
    "translate",
    "unit_scale",
    "vertex_normals",
    "volume",
    "voxel",
    "voxelize",
    "weld",
    "wheel_params",
    "write_obj",
    "write_ply",
    "write_stl",
]
