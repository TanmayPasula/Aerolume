"""2D incompressible Navier-Stokes solver (PyTorch, GPU or CPU, differentiable).

Public API (DESIGN.md section 5.2, placed physics section 10):

* :class:`SolverConfig`, :class:`ImmersedBody` - non-dimensional problem setup
* placed physics in solver units: :class:`EmitterSpec`, :class:`ThermalConfig`,
  :class:`WallBC`, :class:`HeatSourceSpec`, :class:`FanSpec`, :class:`PortSpec`
* :class:`FlowSolver` - stateful, fast (CUDA graphs), live-editable solver for the app
* functional core for autograd: :func:`make_operators`, :func:`init_state`,
  :func:`build_geometry`, :func:`step` (+ :func:`rollout` with checkpointing)

See ``docs/theory_ns2d.md`` for the numerics and their verification.
"""
from __future__ import annotations

from .config import (
    EmitterSpec,
    FanSpec,
    HeatSourceSpec,
    ImmersedBody,
    PortSpec,
    SolverConfig,
    ThermalConfig,
    WallBC,
)
from .core import Ops, State, init_state, make_operators, step, step_full
from .diagnostics import DIAG_COLUMNS
from .geometry import Geometry, ThermalGeometry, build_geometry, signed_distance
from .rollout import rollout
from .solver import FIELDS, PHYSICS_FIELDS, FlowSolver

__all__ = [
    "DIAG_COLUMNS",
    "FIELDS",
    "PHYSICS_FIELDS",
    "EmitterSpec",
    "FanSpec",
    "FlowSolver",
    "Geometry",
    "HeatSourceSpec",
    "ImmersedBody",
    "Ops",
    "PortSpec",
    "SolverConfig",
    "State",
    "ThermalConfig",
    "ThermalGeometry",
    "WallBC",
    "build_geometry",
    "init_state",
    "make_operators",
    "rollout",
    "signed_distance",
    "step",
    "step_full",
]
