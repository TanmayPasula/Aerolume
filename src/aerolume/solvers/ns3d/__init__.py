"""3D incompressible Navier-Stokes solver (PyTorch, GPU or CPU, differentiable).

Public API (DESIGN.md sections 7.5 and 10):

* :class:`SolverConfig3D`, :class:`ImmersedMesh`, :class:`DyeRake` - non-dimensional problem setup
  (DyeRake: the smoke emitters, DESIGN.md 7.13)
* placed physics in solver units (DESIGN.md 10, all off by default): :class:`Emitter3D` (smoke
  channels and tracer particles), :class:`ThermalConfig3D` (heat transfer, Boussinesq buoyancy,
  wall conditions), :class:`HeatSource3D`, :class:`Fan3D` (actuator discs), :class:`Port3D` (jets
  and vents)
* :class:`FlowSolver3D` - stateful, fast (fused CUDA kernels + CUDA graphs),
  live-editable solver for the application
* functional core for autograd: :func:`make_operators3d`, :func:`init_state3d`,
  :func:`build_geometry3d`, :func:`step3d` / :func:`step3d_full` and
  :func:`rollout3d` (checkpointed); :func:`step3d_phys` is the step with placed physics

See ``docs/theory_ns3d.md`` for the numerics and their verification (sections 15-18 for the
placed physics).
"""
from __future__ import annotations

from .config import (
    DyeRake,
    Emitter3D,
    Fan3D,
    HeatSource3D,
    ImmersedMesh,
    Port3D,
    SolverConfig3D,
    ThermalConfig3D,
)
from .core import (
    COEFF_NAMES,
    Ops3D,
    State3D,
    init_state3d,
    make_operators3d,
    step3d,
    step3d_full,
    step3d_phys,
)
from .geometry import Geometry3D, build_geometry3d, mesh_signed_distance
from .rollout import rollout3d
from .solver import DIAG_KEYS, FIELDS3D, FlowSolver3D, estimate_memory_bytes

__all__ = [
    "COEFF_NAMES",
    "DIAG_KEYS",
    "FIELDS3D",
    "DyeRake",
    "Emitter3D",
    "Fan3D",
    "FlowSolver3D",
    "Geometry3D",
    "HeatSource3D",
    "ImmersedMesh",
    "Ops3D",
    "Port3D",
    "SolverConfig3D",
    "State3D",
    "ThermalConfig3D",
    "build_geometry3d",
    "estimate_memory_bytes",
    "init_state3d",
    "make_operators3d",
    "mesh_signed_distance",
    "rollout3d",
    "step3d",
    "step3d_full",
    "step3d_phys",
]
