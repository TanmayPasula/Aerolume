"""Shape sensitivity maps and shape optimisation.

Public API::

    compute_sensitivity(design, sim, *, objective, plan, state, window, n_per_body,
                        device, progress, cancel) -> SensitivityMaps   # list[np.ndarray] + metadata
    attach_sensitivity(result: CFDResult, maps) -> CFDResult            # fills SurfaceData.sensitivity
    compute_sensitivity3d(design, sim, *, objective, plan, solver, window, device, progress,
                          cancel, max_cells) -> SensitivityMaps3D       # per mesh vertex, 3D (DESIGN 7.8)
    attach_sensitivity3d(result: CFD3DResult, maps) -> CFD3DResult      # fills SurfaceData3D.sensitivity
    optimize_airfoil(coords, re, *, alpha | target_cl, objective, min_thickness, keep_area,
                     n_iter, progress, cancel) -> OptimizationResult    # CST + viscous panel solver
    optimize_shape_cfd(design, sim, *, objective, n_modes, n_iter, ...) -> OptimizationResult
                                                                        # experimental, autograd CFD
    OptimizationResult, SensitivityMaps, WindowedObjective, Cancelled
    SensitivityMaps3D, WindowedObjective3D, SensitivityBudgetError (3D sweep over budget)

``optimize_airfoil`` needs only numpy / scipy and the airfoil solver; the CFD
parts import torch. Names are therefore resolved lazily, so importing this
package (which the GUI does at start-up) is cheap.
"""
from __future__ import annotations

import importlib
from typing import Any

from .common import Cancelled, OptimizationResult

_LAZY = {
    "compute_sensitivity": "sensitivity", "attach_sensitivity": "sensitivity",
    "SensitivityMaps": "sensitivity", "WindowedObjective": "sensitivity",
    "finite_difference_check": "sensitivity", "normal_modes": "sensitivity", "OBJECTIVES": "sensitivity",
    "optimize_airfoil": "airfoil", "airfoil_metrics": "airfoil",
    "optimize_shape_cfd": "shape",
    "compute_sensitivity3d": "sensitivity3d", "attach_sensitivity3d": "sensitivity3d",
    "SensitivityMaps3D": "sensitivity3d", "WindowedObjective3D": "sensitivity3d",
    "SensitivityBudgetError": "sensitivity3d", "finite_difference_check3d": "sensitivity3d",
}

__all__ = ["OBJECTIVES", "Cancelled", "OptimizationResult", "SensitivityBudgetError", "SensitivityMaps",
           "SensitivityMaps3D", "WindowedObjective", "WindowedObjective3D", "airfoil_metrics",
           "attach_sensitivity", "attach_sensitivity3d", "compute_sensitivity", "compute_sensitivity3d",
           "finite_difference_check", "finite_difference_check3d", "normal_modes", "optimize_airfoil",
           "optimize_shape_cfd"]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
