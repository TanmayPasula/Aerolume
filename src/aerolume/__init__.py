"""Aerolume: open-source studio for 2D and 3D aerodynamics.

``import aerolume as al`` gives the scripting API (DESIGN.md section 8;
:mod:`aerolume.api`, tutorial in docs/scripting.md)::

    import aerolume as al
    p = al.new_project("Study")
    r = p.run_cfd(p.add_naca("2412"), speed=30, alpha=4)
    print(r.cd, r.cl, r.converged)
    r = p.run_case(al.cavity_case(ra=1e4))          # natural convection, with its reference Nu

The API names are resolved on first use, so importing the package stays fast
and loads neither torch, VTK nor Qt: the GUI, the command line and the project
store import ``aerolume`` for its version string alone.
"""
from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

#: Names re-exported from :mod:`aerolume.api` (loaded on first access).
_API = ("SETTINGS", "AerolumeError", "Cancelled", "Case", "Comparison", "Design", "Emitter", "Fan", "GridStudy",
        "HeatSource", "Host", "InputError", "NotFoundError", "Plan", "Port", "Project", "ProjectFileError", "Result",
        "RunError", "Setting", "Sweep", "ThermalSettings", "WallThermal", "cavity_case", "context", "fan_room_case",
        "file_source", "grid_convergence", "heated_cylinder_case", "new_project", "open_project", "physics_features", "plain",
        "plan", "reference_case", "set_verbose", "show", "supported_physics", "version")

__all__ = ["SETTINGS", "AerolumeError", "Cancelled", "Case", "Comparison", "Design", "Emitter", "Fan", "GridStudy",
           "HeatSource", "Host", "InputError", "NotFoundError", "Plan", "Port", "Project", "ProjectFileError",
           "Result", "RunError", "Setting", "Sweep", "ThermalSettings", "WallThermal", "__version__", "cavity_case",
           "context", "fan_room_case", "file_source", "grid_convergence", "heated_cylinder_case", "new_project", "open_project",
           "physics_features", "plain", "plan", "reference_case", "set_verbose", "show", "supported_physics",
           "version"]


def __getattr__(name: str) -> Any:
    if name in _API:
        from . import api
        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'aerolume' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_API))
