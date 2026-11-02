"""Data exports (DESIGN.md 5.6b, 7.10): CSV tables, field files (NPZ, VTK) and geometry
(SVG, DXF, Selig .dat; STL, OBJ for 3D), 3D surfaces (VTK PolyData) and volumes (VTK
ImageData), and the research exports (solver diagnostics, fan tables, provenance
records, grid-convergence studies). Every function writes atomically and returns the
written Path."""
from .fields import cell_centres, fields_npz, fields_vtk
from .geometry import design_dat, design_dxf, design_svg, selig_order
from .mesh3d import mesh_obj, mesh_stl, surface_vtp
from .research import diagnostic_keys, diagnostics_csv, fans_csv, grid_study_csv, provenance_json
from .tables import (
    history_csv,
    polar_csv,
    read_csv,
    result_csv,
    slice_csv,
    surface_csv,
    wing_csv,
    wing_span_csv,
)
from .volume import cell_centres3, mirror_volume, volume_npz, volume_vti

__all__ = [
    "cell_centres", "cell_centres3", "design_dat", "design_dxf", "design_svg", "diagnostic_keys", "diagnostics_csv",
    "fans_csv", "fields_npz", "fields_vtk", "grid_study_csv", "history_csv", "mesh_obj", "mesh_stl", "mirror_volume",
    "polar_csv", "provenance_json", "read_csv", "result_csv", "selig_order", "slice_csv", "surface_csv", "surface_vtp",
    "volume_npz", "volume_vti", "wing_csv", "wing_span_csv",
]
