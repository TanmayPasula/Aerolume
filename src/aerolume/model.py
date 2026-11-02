"""Shared data model.

Every module in Aerolume reads and writes these types, so they are the contract
between the geometry tools, the solvers, the project store, the report writer and
the desktop application. All quantities are SI. 2D results give forces per unit
span (N/m), 3D results (:class:`CFD3DResult`, :class:`WingResult`) whole-body
forces (N). DESIGN.md section 3 has the 2D frame and sign conventions, section 7.2
the 3D ones.

Each dataclass round-trips through ``to_dict`` / ``from_dict`` as plain JSON-safe
values. Small arrays (polygons, polar tables) serialise as nested lists; large
arrays (fields, long histories, mesh vertices) are written to ``.npz`` by
:mod:`aerolume.project` and are deliberately left out of ``to_dict`` there.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np

#: Newest project schema this version reads and writes. Schema 2 marks a project that
#: holds 3D designs (DESIGN.md 7.10). Aerolume 0.1 reads schema 1 and warns about a
#: higher one but still opens the project, so the store writes 2 only when it must:
#: a project with 2D designs only stays schema 1 and opens in 0.1 exactly as before.
SCHEMA_VERSION = 2
#: The schema of a project without 3D designs (what Aerolume 0.1 wrote).
SCHEMA_VERSION_2D = 1

KINDS = ("airfoil", "vehicle", "generic", "wing")
KINDS_2D = ("airfoil", "vehicle", "generic")     # "wing" is a 3D planform (Design.wing)
ROLES = ("body", "wheel", "element")
ANALYSES = ("cfd", "polar", "wing")
DIMENSIONS = (2, 3)
WING_METHODS = ("vlm", "llt")
QUALITIES = ("draft", "standard", "fine", "custom")
TURBULENCE = ("auto", "laminar", "les")
REYNOLDS_MODES = ("real", "capped")
STATUSES = ("new", "queued", "running", "done", "failed", "cancelled")


def new_id() -> str:
    """A short unique id for designs and simulations."""
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    """Local time stamp in ISO-8601, second resolution."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _arr(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _jsonable(v: Any) -> Any:
    """``v`` as plain JSON types. numpy scalars become Python numbers and ``np.bool_``
    a ``bool``: any numpy comparison (``converged = err < tol``) yields ``np.bool_``,
    which is neither ``np.integer`` nor ``np.floating`` and which ``json`` refuses."""
    if is_dataclass(v):
        return v.to_dict()  # type: ignore[attr-defined]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


class _Serial:
    """Mixin: shallow dataclass <-> dict conversion with nested-type hooks."""

    _nested: ClassVar[dict[str, Any]] = {}
    _nested_list: ClassVar[dict[str, Any]] = {}
    _arrays: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> dict:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: dict):
        if not isinstance(d, dict):
            # a damaged file (a list or a string where an object belongs, at any level)
            # must fail with a reason the project store can report, not an AttributeError
            raise TypeError(f"{cls.__name__}: expected a JSON object, got {type(d).__name__}")
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        kw = {}
        for k, v in d.items():
            if k not in names:
                continue
            if k in cls._nested and v is not None:
                v = cls._nested[k].from_dict(v)
            elif k in cls._nested_list and v is not None:
                v = [cls._nested_list[k].from_dict(x) for x in v]
            elif k in cls._arrays and v is not None:
                v = _arr(v)
            elif isinstance(v, list) and k in ("center", "moment_ref"):
                v = tuple(v)
            kw[k] = v
        return cls(**kw)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def isa_sea_level_air() -> tuple[float, float]:
    """(density kg/m3, dynamic viscosity Pa s) of ISA sea-level air.

    The single source of the "Air, ISA sea level" numbers: :class:`Fluid`'s
    defaults (so ``default_simulation`` and every GUI-made simulation) and
    :func:`aerolume.core.fluid.air_isa` (the script API, the CLI, the assistant)
    both come from :func:`aerolume.core.fluid.isa` and Sutherland's law here.
    The defaults used to be the rounded constants 1.225 / 1.7894e-5 while the
    API stored 1.2250000180 / 1.7892976e-5, so Compare flagged a GUI run and a
    script run of the same air as "different fluids".
    """
    from .core.fluid import isa, sutherland_viscosity  # lazy: core.fluid imports this module
    t, _, rho = isa(0.0)
    return float(rho), float(sutherland_viscosity(t))


@dataclass
class Fluid(_Serial):
    """A Newtonian fluid. Build common ones with :mod:`aerolume.core.fluid`.

    The defaults are ISA sea-level air, identical to ``core.fluid.air_isa(0.0)``
    (see :func:`isa_sea_level_air`): about 1.225 kg/m3 and 1.7893e-5 Pa s.
    """
    name: str = "Air, ISA sea level"
    density: float = field(default_factory=lambda: isa_sea_level_air()[0])      # kg/m3
    viscosity: float = field(default_factory=lambda: isa_sea_level_air()[1])    # Pa s (dynamic)

    @property
    def kinematic_viscosity(self) -> float:
        return self.viscosity / self.density


@dataclass
class Body(_Serial):
    """One closed solid outline.

    ``points`` is an (N, 2) array in metres, counter-clockwise, implicitly closed.
    A ``wheel`` body may spin: ``omega`` [rad/s, positive counter-clockwise] about
    ``center``. In the vehicle frame the air comes from the left and a moving road
    runs at +U in x, so a wheel of radius r that rolls on it without slipping has its
    contact point moving at +U: it spins counter-clockwise, omega = +U / r (the top of
    the wheel moves against the flow). The planner, ``ImmersedBody`` and
    docs/theory_ns2d.md section 11 use the same sign.
    """
    name: str
    points: np.ndarray
    role: str = "body"
    omega: float = 0.0
    center: tuple[float, float] | None = None
    # heat transfer (used only when Simulation.thermal.enabled): "adiabatic",
    # "temperature" (thermal_value in °C, e.g. a heated cylinder) or "heat_flux" (W/m²)
    thermal: str = "adiabatic"
    thermal_value: float = 0.0

    _arrays = ("points",)

    def __post_init__(self) -> None:
        self.points = _arr(self.points).reshape(-1, 2)
        if self.role not in ROLES:
            raise ValueError(f"unknown body role {self.role!r}")
        if self.thermal not in THERMAL_MODES:
            raise ValueError(f"unknown thermal mode {self.thermal!r}")


@dataclass
class Mesh3D(_Serial):
    """One triangulated 3D surface (DESIGN.md section 7.2 for the frame).

    ``vertices`` (N, 3) float64 metres, ``faces`` (M, 3) int64 indices. The mesh
    does NOT have to be watertight or consistently oriented: the voxeliser decides
    inside/outside robustly (like AirShaper, small holes and gaps are tolerated).
    A ``wheel`` spins at ``omega`` [rad/s] about ``axis`` through ``center`` by the
    right-hand rule: its surface moves at ``omega * axis x (x - center)``. In the 3D
    frame (x downstream, z up) the road moves at +U in x, and at the contact point
    below an axle along +y that velocity is ``-omega * r`` in x, so a wheel rolling
    on the road without slipping has **omega = -U / r about +y** (equivalently
    +U / r about -y; the top of the wheel moves upstream). The planner sets this
    from the speed when ``GroundConfig.rotate_wheels`` is on; the 2D :class:`Body`
    gets +U / r because its counter-clockwise rotation is seen from the -y side.

    The vertex and face arrays are large, so ``to_dict`` leaves them out (it records
    their sizes); the project store keeps them in ``designs/<id>.npz`` and puts them
    back with :meth:`Design.attach_mesh_arrays`. Use :meth:`Design.mesh_arrays` to get
    them for saving.
    """
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    role: str = "body"
    omega: float = 0.0
    center: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    thermal: str = "adiabatic"          # as Body.thermal
    thermal_value: float = 0.0

    _large: ClassVar[tuple[str, ...]] = ("vertices", "faces")

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.role not in ROLES:
            raise ValueError(f"unknown body role {self.role!r}")
        if self.thermal not in THERMAL_MODES:
            raise ValueError(f"unknown thermal mode {self.thermal!r}")
        for k in ("center", "axis"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, tuple(float(c) for c in v))

    def to_dict(self) -> dict:
        d = {f.name: _jsonable(getattr(self, f.name)) for f in fields(self) if f.name not in self._large}
        d["n_vertices"], d["n_faces"] = len(self.vertices), len(self.faces)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Mesh3D:
        if not isinstance(d, dict):
            raise TypeError(f"Mesh3D: expected a JSON object, got {type(d).__name__}")
        d = dict(d)
        d.setdefault("vertices", np.zeros((0, 3)))
        d.setdefault("faces", np.zeros((0, 3), dtype=np.int64))
        return super().from_dict(d)

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """(xmin, xmax, ymin, ymax, zmin, zmax), metres."""
        v = self.vertices
        return (float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()),
                float(v[:, 1].max()), float(v[:, 2].min()), float(v[:, 2].max()))


@dataclass
class WingSection(_Serial):
    """One spanwise station of a wing planform (wing frame, metres, see 7.2).

    ``y`` is the spanwise position (>= 0 on the right half of a symmetric wing),
    ``x_le`` / ``z_le`` place the leading edge (sweep and dihedral come from how
    they change with ``y``), ``twist_deg`` rotates the section nose-up about its
    leading edge. ``airfoil`` is a NACA code ("2412") or "design:<id>" naming a 2D
    airfoil design in the same project.
    """
    y: float
    chord: float
    x_le: float = 0.0
    z_le: float = 0.0
    twist_deg: float = 0.0
    airfoil: str = "0012"


@dataclass
class WingSpec(_Serial):
    """A lifting surface built from :class:`WingSection` stations (root first).

    ``symmetric`` mirrors the sections about y = 0 (a full wing from a half
    planform). ``n_chord`` / ``n_span`` set the vortex-lattice / lifting-line
    discretisation per half wing; ``spacing`` is "cosine" or "uniform".
    """
    sections: list[WingSection]
    symmetric: bool = True
    n_chord: int = 8
    n_span: int = 24
    spacing: str = "cosine"

    _nested_list: ClassVar[dict[str, Any]] = {"sections": WingSection}


@dataclass
class Design(_Serial):
    """A geometry variant: one or more bodies plus its reference quantities.

    2D (``dimension == 2``): ``bodies`` are closed outlines; ``length_ref`` sets the
    Reynolds number (chord, vehicle length); ``force_ref`` normalises force
    coefficients (chord for airfoils, frontal height for vehicles); ``moment_ref``
    is the 2D moment reference point [m].

    3D (``dimension == 3``, DESIGN.md section 7): ``meshes`` hold the surfaces
    (``kind == "wing"`` designs carry a ``wing`` planform instead and are lofted on
    demand); ``ref_area`` [m²] normalises force coefficients (frontal area for
    vehicles and bluff bodies, planform area for wings), ``length_ref`` sets Re and
    the pitching-moment arm (vehicle length or mean aerodynamic chord),
    ``span_ref`` [m] is the roll/yaw moment arm (span or track width) and
    ``moment_ref3`` the 3D moment reference point. ``symmetric`` says the geometry
    is mirror-symmetric about y = 0, which lets the planner simulate half a domain.
    ``force_ref`` of a 3D design is informational: the frontal height for imported
    vehicles and bluff bodies, the mean aerodynamic chord for wings.

    ``source`` records how the design was made. Imported meshes
    (:func:`aerolume.geometry3d.design_from_meshes`) add ``scale_to_m`` (metres per
    file unit applied), ``frontal_area`` [m²], ``symmetry_error`` (RMS mirror error
    across y = 0, metres), ``compact`` (the reference length is the cross-stream
    size, not the length), ``on_road`` and ``import_notes`` (messages for the user);
    a wing (``design_from_wing``) has ``{"type": "wing"}``.
    """
    name: str
    kind: str
    bodies: list[Body]
    length_ref: float
    force_ref: float
    moment_ref: tuple[float, float] = (0.0, 0.0)
    id: str = field(default_factory=new_id)
    source: dict = field(default_factory=dict)      # how it was made: {"type": "naca", "code": "2412"}
    created: str = field(default_factory=now_iso)
    notes: str = ""
    # --- 3D (all optional, defaults keep every 2D project unchanged) ---
    dimension: int = 2
    meshes: list[Mesh3D] = field(default_factory=list)
    wing: WingSpec | None = None
    ref_area: float = 0.0                # 0 = not set (3D: the geometry tools fill it)
    span_ref: float = 0.0                # 0 = not set
    moment_ref3: tuple[float, float, float] = (0.0, 0.0, 0.0)
    symmetric: bool = False

    _nested: ClassVar[dict[str, Any]] = {"wing": WingSpec}
    _nested_list: ClassVar[dict[str, Any]] = {"bodies": Body, "meshes": Mesh3D}

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown design kind {self.kind!r}")
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"dimension must be 2 or 3, got {self.dimension!r}")
        if self.kind == "wing" and self.dimension != 3:
            raise ValueError("a 'wing' design is three-dimensional (dimension=3)")
        self.moment_ref = tuple(float(c) for c in self.moment_ref)  # type: ignore[assignment]
        self.moment_ref3 = tuple(float(c) for c in self.moment_ref3)  # type: ignore[assignment]

    @property
    def is_3d(self) -> bool:
        return self.dimension == 3

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) over all 2D bodies, metres."""
        pts = np.vstack([b.points for b in self.bodies])
        return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())

    def bounds3(self) -> tuple[float, float, float, float, float, float]:
        """(xmin, xmax, ymin, ymax, zmin, zmax) over all meshes, metres."""
        b = np.array([m.bounds() for m in self.meshes])
        return (float(b[:, 0].min()), float(b[:, 1].max()), float(b[:, 2].min()),
                float(b[:, 3].max()), float(b[:, 4].min()), float(b[:, 5].max()))

    def mesh_arrays(self) -> dict[str, np.ndarray]:
        """Mesh vertex/face arrays keyed ``mesh{i}_vertices`` / ``mesh{i}_faces`` (for .npz)."""
        out: dict[str, np.ndarray] = {}
        for i, m in enumerate(self.meshes):
            out[f"mesh{i}_vertices"] = m.vertices
            out[f"mesh{i}_faces"] = m.faces
        return out

    def attach_mesh_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        """Put arrays saved by :meth:`mesh_arrays` back onto ``meshes`` (after from_dict)."""
        for i, m in enumerate(self.meshes):
            if f"mesh{i}_vertices" in arrays:
                m.vertices = np.asarray(arrays[f"mesh{i}_vertices"], dtype=np.float64).reshape(-1, 3)
                m.faces = np.asarray(arrays[f"mesh{i}_faces"], dtype=np.int64).reshape(-1, 3)


@dataclass
class FlowConditions(_Serial):
    """Free-stream state. ``alpha_deg`` is angle of attack (airfoils, wings) or pitch
    (vehicles), positive nose-up. ``yaw_deg`` (3D only) is the sideslip angle,
    applied by rotating the geometry by -yaw about +z (DESIGN.md 7.2). Positive yaw
    turns the nose (at -x) toward +y, so in the vehicle's own frame the free stream
    is ``U (cos yaw, sin yaw, 0)``: the wind comes from the **-y side, which is the
    left of a driver or pilot facing forward** (-x); +y is their right.
    """
    fluid: Fluid = field(default_factory=Fluid)
    speed: float = 30.0             # m/s
    alpha_deg: float = 0.0
    yaw_deg: float = 0.0

    _nested: ClassVar[dict[str, Any]] = {"fluid": Fluid}

    def dynamic_pressure(self) -> float:
        return 0.5 * self.fluid.density * self.speed ** 2

    def reynolds(self, length: float) -> float:
        return self.fluid.density * self.speed * length / self.fluid.viscosity


@dataclass
class GroundConfig(_Serial):
    """Road plane (vehicles): y = 0 in 2D, z = 0 in 3D. ``moving`` = belt moving at the
    free-stream speed; ``rotate_wheels`` spins wheel bodies / meshes to roll with it."""
    enabled: bool = False
    moving: bool = True
    rotate_wheels: bool = True


@dataclass
class CFDSettings(_Serial):
    """Navier–Stokes run settings. ``quality`` presets fill the rest unless 'custom'.

    Domain extents are in multiples of ``Design.length_ref`` measured from the body's
    bounding box. ``duration`` and ``average_from`` are in convective times
    (length_ref / speed); ``None`` lets the planner choose.
    """
    quality: str = "standard"
    cells_per_ref: int | None = None
    domain_upstream: float | None = None
    domain_downstream: float | None = None
    domain_height: float | None = None      # total height (no ground) or height above ground
    duration: float | None = None
    average_from: float | None = None
    turbulence: str = "auto"
    reynolds_mode: str = "real"
    reynolds_cap: float | None = None       # used when reynolds_mode == "capped"
    precision: str = "auto"                 # "auto" | "float32" | "float64"
    device: str = "auto"                    # "auto" | "cuda" | "cpu" | "mps"
    compute_sensitivity: bool = False
    sensitivity_objective: str = "cd"       # "cd" | "cl" | "ld"
    # 3D: "auto" simulates half the domain with a symmetry plane at y = 0 when the
    # design is symmetric and yaw is zero; "full" always simulates both halves
    symmetry: str = "auto"                  # "auto" | "full"
    domain_width: float | None = None       # 3D: total width (full) in length_ref units
    # "tunnel": inflow from the left, outflow on the right (default). "closed": every side
    # is a wall (a cavity, a room, an enclosure) — no free stream; natural convection,
    # fans and jets drive the flow. ``domain_box`` (metres, design frame) fixes the extent:
    # (xmin, xmax, ymin, ymax) in 2D, + (zmin, zmax) in 3D; None lets the planner size it.
    domain_mode: str = "tunnel"
    domain_box: tuple[float, ...] | None = None
    # AeroJAX-style diagnostics recorded during the run (cost: a few % of a step)
    record_diagnostics: bool = True         # divergence, change norms, residual history
    adaptive_dt: str = "cfl"                # "cfl" (current) | "pid" (PID-controlled CFL target)

    def __post_init__(self) -> None:
        if self.quality not in QUALITIES:
            raise ValueError(f"unknown quality {self.quality!r}")
        if self.turbulence not in TURBULENCE:
            raise ValueError(f"unknown turbulence option {self.turbulence!r}")
        if self.domain_mode not in DOMAIN_MODES:
            raise ValueError(f"unknown domain mode {self.domain_mode!r}")
        if self.domain_box is not None:
            self.domain_box = tuple(float(c) for c in self.domain_box)


@dataclass
class AirfoilSettings(_Serial):
    """Viscous–inviscid polar sweep settings (mfoil / XFOIL-class)."""
    alpha_start: float = -4.0
    alpha_end: float = 14.0
    alpha_step: float = 1.0
    ncrit: float = 9.0
    xtr_upper: float = 1.0          # forced transition x/c (1 = free)
    xtr_lower: float = 1.0
    mach: float = 0.0
    panels: int = 199

    def alphas(self) -> np.ndarray:
        n = round((self.alpha_end - self.alpha_start) / self.alpha_step) + 1
        return self.alpha_start + self.alpha_step * np.arange(max(n, 1))


@dataclass
class WingSettings(_Serial):
    """Finite-wing analysis (vortex lattice or nonlinear lifting line) settings.

    ``viscous`` adds profile drag (and, for "llt", the nonlinear section lift)
    from 2D viscous polars of each section's airfoil at its local Reynolds number.
    """
    alpha_start: float = -4.0
    alpha_end: float = 12.0
    alpha_step: float = 1.0
    method: str = "vlm"             # "vlm" | "llt"
    viscous: bool = True
    ncrit: float = 9.0
    mach: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in WING_METHODS:
            raise ValueError(f"unknown wing method {self.method!r}")

    def alphas(self) -> np.ndarray:
        n = round((self.alpha_end - self.alpha_start) / self.alpha_step) + 1
        return self.alpha_start + self.alpha_step * np.arange(max(n, 1))


# ---------------------------------------------------------------------------
# placed physics: emitters, heat, fans, ports (DESIGN.md section 10)
# ---------------------------------------------------------------------------

EMITTER_SPECIES = ("dye", "particles")
EMITTER_KINDS = ("point", "line", "rake", "rect", "disc", "plane")
PLACEMENT_FRAMES = ("wind", "body")      # fixed in the tunnel, or moving with the geometry
THERMAL_MODES = ("adiabatic", "temperature", "heat_flux")
FAN_MODES = ("thrust", "pressure_jump", "velocity")
PORT_KINDS = ("jet", "vent")
DOMAIN_MODES = ("tunnel", "closed")
WALL_SIDES = ("left", "right", "bottom", "top", "front", "back")   # front/back: 3D y-min / y-max


def _vec(v: Any) -> tuple[float, ...] | None:
    return None if v is None else tuple(float(c) for c in v)


@dataclass
class Emitter(_Serial):
    """A source of flow-visualisation tracer — smoke (``species="dye"``, a passive
    scalar) or tracer particles (streaklines) — placed anywhere in the domain.

    Geometry by ``kind``: "point" at ``position``; "line"/"rake" from ``position`` to
    ``end`` ("rake" = ``count`` discrete nozzles, "line" = a continuous slot); "rect" an
    axis-aligned box of ``size`` centred at ``position``; "disc" (3D) / "plane" (3D) of
    radius / extents ``size`` facing ``normal``. Positions are metres in the design
    frame; ``frame="wind"`` keeps the emitter fixed in the tunnel when the geometry is
    pitched or yawed, ``"body"`` moves it with the geometry. 2D emitters use (x, y).
    ``rate`` is the released concentration (dye, 0..1) or particles per second;
    ``pulse_period`` > 0 turns it on for ``pulse_duty`` of each period (timelines).
    ``start``/``stop`` are seconds of simulated flow time.
    """
    name: str = "Emitter"
    species: str = "dye"
    kind: str = "point"
    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    end: tuple[float, ...] | None = None
    size: tuple[float, ...] = ()
    normal: tuple[float, ...] | None = None
    count: int = 8
    radius: float = 0.0              # nozzle radius [m]; 0 = about two grid cells
    rate: float = 1.0
    color: str = "#f5f5f5"
    pulse_period: float = 0.0
    pulse_duty: float = 0.5
    start: float = 0.0
    stop: float | None = None
    frame: str = "wind"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.species not in EMITTER_SPECIES:
            raise ValueError(f"unknown emitter species {self.species!r}")
        if self.kind not in EMITTER_KINDS:
            raise ValueError(f"unknown emitter kind {self.kind!r}")
        if self.frame not in PLACEMENT_FRAMES:
            raise ValueError(f"unknown placement frame {self.frame!r}")
        self.position = _vec(self.position)  # type: ignore[assignment]
        self.end = _vec(self.end)
        self.normal = _vec(self.normal)
        self.size = tuple(float(c) for c in self.size)


@dataclass
class WallThermal(_Serial):
    """Thermal boundary condition on one side of the computational domain.

    ``mode`` "adiabatic" (no heat flux), "temperature" (``value`` in °C) or
    "heat_flux" (``value`` in W/m², positive into the fluid)."""
    side: str
    mode: str = "adiabatic"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.side not in WALL_SIDES:
            raise ValueError(f"unknown wall side {self.side!r}")
        if self.mode not in THERMAL_MODES:
            raise ValueError(f"unknown thermal mode {self.mode!r}")


@dataclass
class ThermalSettings(_Serial):
    """Heat transfer: temperature transport with optional Boussinesq buoyancy.

    Off by default, so every existing simulation is unchanged. ``t_ambient`` is the
    free-stream / initial temperature [°C]; gravity acts along -y (2D) or -z (3D).
    ``beta`` (1/K) defaults to the ideal-gas 1/T_ambient for gases; ``prandtl``,
    ``conductivity`` [W/(m K)] and ``cp`` [J/(kg K)] default from the fluid (air,
    water presets) when None. ``walls`` set domain-side conditions (adiabatic when
    not listed). Body conditions live on :class:`Body` / :class:`Mesh3D`.
    """
    enabled: bool = False
    t_ambient: float = 20.0
    buoyancy: bool = True
    gravity: float = 9.80665
    beta: float | None = None
    prandtl: float | None = None
    conductivity: float | None = None
    cp: float | None = None
    walls: list[WallThermal] = field(default_factory=list)

    _nested_list: ClassVar[dict[str, Any]] = {"walls": WallThermal}


@dataclass
class HeatSource(_Serial):
    """A volumetric heater (electronics, a radiator, a burner) in the flow.

    ``kind`` "rect" (2D) / "box" (3D) of ``size`` or "disc" (2D) / "sphere" (3D) of
    radius ``size[0]``, centred at ``position`` (metres, design frame). ``power`` is the
    heat released in W (3D) or W per metre of span (2D); negative cools.
    """
    name: str = "Heater"
    kind: str = "rect"
    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    size: tuple[float, ...] = (0.1, 0.1, 0.1)
    power: float = 100.0
    frame: str = "body"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("rect", "box", "disc", "sphere"):
            raise ValueError(f"unknown heat source kind {self.kind!r}")
        if self.frame not in PLACEMENT_FRAMES:
            raise ValueError(f"unknown placement frame {self.frame!r}")
        self.position = _vec(self.position)  # type: ignore[assignment]
        self.size = tuple(float(c) for c in self.size)


@dataclass
class Fan(_Serial):
    """An actuator-disc fan or propeller: a thin momentum source.

    Centred at ``position``, blowing along ``direction`` (unit vector, design frame),
    of ``diameter`` (3D disc) or width (2D actuator line) and ``thickness``
    (default ``diameter / 8``). ``mode``: "thrust" (``value`` in N, or N/m in 2D),
    "pressure_jump" (Pa across the disc) or "velocity" (target mean axial speed
    through the disc, m/s; the source is regulated to reach it). ``swirl`` (3D) is the
    tangential-to-axial velocity ratio imparted, positive by the right-hand rule
    about ``direction``.
    """
    name: str = "Fan"
    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0)
    diameter: float = 0.3
    thickness: float | None = None
    mode: str = "thrust"
    value: float = 10.0
    swirl: float = 0.0
    frame: str = "body"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in FAN_MODES:
            raise ValueError(f"unknown fan mode {self.mode!r}")
        if self.frame not in PLACEMENT_FRAMES:
            raise ValueError(f"unknown placement frame {self.frame!r}")
        self.position = _vec(self.position)  # type: ignore[assignment]
        self.direction = _vec(self.direction)  # type: ignore[assignment]


@dataclass
class Port(_Serial):
    """A jet (fluid blown in) or vent (fluid sucked out) inside the domain.

    A small region at ``position`` of width/diameter ``size`` whose flow is driven to
    ``speed`` m/s along ``direction`` (jet: out of the port; vent: into it). A jet
    may carry its own ``temperature`` (°C, needs heat transfer on) and release smoke
    (``dye``). Mass balance is kept by the outlet (tunnel) or by pairing jets with
    vents (closed domains; the planner warns when they do not balance).
    """
    name: str = "Jet"
    kind: str = "jet"
    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0)
    size: float = 0.05
    speed: float = 5.0
    temperature: float | None = None
    dye: bool = False
    frame: str = "body"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.kind not in PORT_KINDS:
            raise ValueError(f"unknown port kind {self.kind!r}")
        if self.frame not in PLACEMENT_FRAMES:
            raise ValueError(f"unknown placement frame {self.frame!r}")
        self.position = _vec(self.position)  # type: ignore[assignment]
        self.direction = _vec(self.direction)  # type: ignore[assignment]


@dataclass
class Simulation(_Serial):
    """One analysis of one design.

    ``analysis``: "cfd" (Navier–Stokes, 2D or 3D after the design), "polar"
    (2D airfoil viscous–inviscid sweep) or "wing" (3D finite-wing sweep).
    CFD runs may add placed physics (DESIGN.md section 10): ``emitters`` (smoke /
    particles), ``thermal`` (heat transfer and buoyancy), ``heat_sources``, ``fans``
    and ``ports`` (jets and vents). All default to empty / off.
    """
    name: str
    design_id: str
    analysis: str = "cfd"
    conditions: FlowConditions = field(default_factory=FlowConditions)
    ground: GroundConfig = field(default_factory=GroundConfig)
    cfd: CFDSettings = field(default_factory=CFDSettings)
    airfoil: AirfoilSettings = field(default_factory=AirfoilSettings)
    id: str = field(default_factory=new_id)
    status: str = "new"
    created: str = field(default_factory=now_iso)
    finished: str | None = None
    notes: str = ""
    error: str | None = None
    wing: WingSettings = field(default_factory=WingSettings)
    emitters: list[Emitter] = field(default_factory=list)
    thermal: ThermalSettings = field(default_factory=ThermalSettings)
    heat_sources: list[HeatSource] = field(default_factory=list)
    fans: list[Fan] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)

    _nested: ClassVar[dict[str, Any]] = {"conditions": FlowConditions, "ground": GroundConfig,
               "cfd": CFDSettings, "airfoil": AirfoilSettings, "wing": WingSettings,
               "thermal": ThermalSettings}
    _nested_list: ClassVar[dict[str, Any]] = {"emitters": Emitter, "heat_sources": HeatSource,
                                              "fans": Fan, "ports": Port}

    def __post_init__(self) -> None:
        if self.analysis not in ANALYSES:
            raise ValueError(f"unknown analysis {self.analysis!r}")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class Stat(_Serial):
    """Statistics of a (time-averaged) scalar. ``stderr`` is the standard error of the
    mean from batch means; ``nan`` when it could not be estimated.

    ``stderr_reliable`` is False when the averaging window held too few correlation
    lengths for batch means to work (``analysis.stats.batch_means``): ``stderr`` is
    then a conservative estimate, not a measured error bar, and comparisons must
    not call a difference "significant" on it. Results saved before the flag
    existed load as True."""
    mean: float = float("nan")
    rms: float = float("nan")         # rms of fluctuation about the mean
    min: float = float("nan")
    max: float = float("nan")
    stderr: float = float("nan")
    stderr_reliable: bool = True


@dataclass
class SurfaceData(_Serial):
    """Time-averaged surface distributions on one body (wind frame, metres)."""
    body: str
    s: np.ndarray                     # arc length from the first point [m]
    x: np.ndarray
    y: np.ndarray
    nx: np.ndarray                    # outward unit normal
    ny: np.ndarray
    cp: np.ndarray
    cf: np.ndarray                    # signed skin friction, + along increasing s
    sensitivity: np.ndarray | None = None   # dJ/dn per metre of outward displacement

    _arrays = ("s", "x", "y", "nx", "ny", "cp", "cf", "sensitivity")


@dataclass
class FieldData:
    """Cell-centred fields on the solver grid (wind frame, metres).

    ``fields`` maps a name ("u", "v", "p", "cp", "vorticity", "speed", ...) to an
    (ny, nx) array in solver order: row 0 is the bottom row (y-min), so cell (j, i)
    has its centre at (xmin + (i + 0.5) h, ymin + (j + 0.5) h). ``extent`` =
    (xmin, xmax, ymin, ymax) is the outer edge of the cell grid, as for an image.
    Velocities in m/s, pressure in Pa (gauge), vorticity in 1/s, cp dimensionless.
    Note the opposite row order of ``LiveFrame.rgba`` (row 0 = top), which is an
    image ready to paint.
    """
    extent: tuple[float, float, float, float]
    fields: dict[str, np.ndarray]
    mask: np.ndarray | None = None    # solid fraction 0..1, (ny, nx)
    label: str = "mean"               # "mean" | "instant"


@dataclass
class CFDResult(_Serial):
    """Outcome of a Navier–Stokes run."""
    sim_id: str
    cd: Stat = field(default_factory=Stat)
    cl: Stat = field(default_factory=Stat)
    cm: Stat = field(default_factory=Stat)
    drag: float = float("nan")        # N/m, mean
    lift: float = float("nan")        # N/m, mean (negative = downforce)
    moment: float = float("nan")      # N m / m
    drag_area: float = float("nan")   # Cd * force_ref [m^2/m]
    power: float = float("nan")       # W/m to overcome mean drag
    strouhal: float | None = None
    #: length St is based on [m] (chord, vehicle force reference or the frontal height of
    #: a pitched bluff body: SimPlan.strouhal_length); nan in results saved before it existed
    strouhal_length: float = float("nan")
    re_real: float = float("nan")
    re_sim: float = float("nan")
    grid: tuple[int, int] = (0, 0)
    cells_per_ref: float = float("nan")
    t_end: float = float("nan")       # convective times simulated
    t_avg_start: float = float("nan")
    converged: bool = False
    wall_seconds: float = float("nan")
    device: str = ""
    precision: str = ""
    turbulence: str = ""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_body: dict[str, dict[str, float]] = field(default_factory=dict)  # name -> {"cd","cl","cm"}
    # heat transfer and placed physics (DESIGN.md section 10); nan / empty when unused
    nusselt: Stat = field(default_factory=Stat)       # mean Nusselt number over heated bodies
    heat_rate: float = float("nan")                   # W (3D) or W/m (2D) from bodies into the fluid
    fans_report: list[dict[str, float]] = field(default_factory=list)   # per fan: thrust, flow, power
    diagnostics: dict[str, float] = field(default_factory=dict)        # final divergence, change norms
    bodies_wind: list[Body] = field(default_factory=list)                # solved geometry (rotated)
    # large data — persisted to arrays.npz by the project store, not to_dict:
    history: dict[str, np.ndarray] = field(default_factory=dict)        # t, cd, cl, cm
    surface: list[SurfaceData] = field(default_factory=list)
    mean_fields: FieldData | None = None
    instant_fields: FieldData | None = None

    _nested: ClassVar[dict[str, Any]] = {"cd": Stat, "cl": Stat, "cm": Stat, "nusselt": Stat}
    _nested_list: ClassVar[dict[str, Any]] = {"bodies_wind": Body}
    _large = ("history", "surface", "mean_fields", "instant_fields")

    def to_dict(self) -> dict:
        d = {f.name: _jsonable(getattr(self, f.name)) for f in fields(self) if f.name not in self._large}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CFDResult:
        d = {k: v for k, v in d.items() if k not in cls._large}
        if "grid" in d and d["grid"] is not None:
            d["grid"] = tuple(d["grid"])
        return super().from_dict(d)

    @property
    def lift_to_drag(self) -> float:
        return self.cl.mean / self.cd.mean if self.cd.mean else float("nan")


@dataclass
class PolarPoint(_Serial):
    """One converged (or not) viscous–inviscid operating point."""
    alpha: float
    cl: float = float("nan")
    cd: float = float("nan")
    cdp: float = float("nan")         # pressure drag
    cdf: float = float("nan")         # friction drag
    cm: float = float("nan")          # about the quarter chord
    xtr_upper: float = float("nan")   # transition x/c
    xtr_lower: float = float("nan")
    converged: bool = False
    # distributions (optional)
    x: np.ndarray | None = None       # surface x/c, upper TE -> LE -> lower TE
    y: np.ndarray | None = None
    cp: np.ndarray | None = None
    cp_inviscid: np.ndarray | None = None
    cf: np.ndarray | None = None

    _arrays = ("x", "y", "cp", "cp_inviscid", "cf")


@dataclass
class PolarResult(_Serial):
    """A sweep of :class:`PolarPoint` at fixed Re / Mach / Ncrit."""
    sim_id: str
    re: float
    mach: float = 0.0
    ncrit: float = 9.0
    points: list[PolarPoint] = field(default_factory=list)
    wall_seconds: float = float("nan")
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    _nested_list: ClassVar[dict[str, Any]] = {"points": PolarPoint}

    def table(self) -> dict[str, np.ndarray]:
        """Converged points as arrays: alpha, cl, cd, cdp, cdf, cm, xtr_upper, xtr_lower."""
        pts = [p for p in self.points if p.converged]
        keys = ("alpha", "cl", "cd", "cdp", "cdf", "cm", "xtr_upper", "xtr_lower")
        return {k: np.array([getattr(p, k) for p in pts], dtype=float) for k in keys}

    def best_ld(self) -> PolarPoint | None:
        pts = [p for p in self.points if p.converged and p.cd > 0]
        return max(pts, key=lambda p: p.cl / p.cd) if pts else None


# ---------------------------------------------------------------------------
# 3D results (DESIGN.md section 7)
# ---------------------------------------------------------------------------

@dataclass
class SurfaceData3D:
    """Time-averaged surface quantities on one 3D mesh (wind frame, metres).

    Per-vertex arrays on the mesh as solved (rotated by alpha / yaw, and mirrored
    to the full body when a symmetry plane was used): ``cp``; ``cf`` the skin
    friction coefficient magnitude; ``tau`` (N, 3) the wall shear direction scaled
    by cf (for surface streak lines); ``sensitivity`` the shape-sensitivity density
    of the objective J (CD, CL, L/D or Cm), positive where pushing the surface
    outward increases J (DESIGN.md 5.4b), or None. Its unit is 1/m: it is the
    non-dimensional density g times ``L_ref²``, so that an outward displacement
    ``δn`` [m] of the surface changes J by ``ΔJ ≈ Σ sensitivity · δn · dA / L_ref²``
    (dA the vertex's share of the surface area in m²; docs/theory_sensitivity.md
    section 9.2). The 2D :class:`SurfaceData` uses ``L_ref · g`` (also 1/m), which
    integrates over arc length instead.
    """
    body: str
    vertices: np.ndarray              # (N, 3)
    faces: np.ndarray                 # (M, 3)
    cp: np.ndarray                    # (N,)
    cf: np.ndarray | None = None      # (N,)
    tau: np.ndarray | None = None     # (N, 3)
    sensitivity: np.ndarray | None = None   # (N,)


@dataclass
class SliceData:
    """One planar cut through the 3D solution (wind frame).

    ``axis`` is the plane normal ("x", "y" or "z") at ``position`` [m]. Fields are
    2D arrays (rows, cols): for "y" slices rows run along z (row 0 = z-min) and
    columns along x; for "z" slices rows along y, columns along x; for "x" slices
    rows along z, columns along y. ``extent`` = (col_min, col_max, row_min,
    row_max) in metres, outer cell edges (the 2D FieldData convention).
    """
    axis: str
    position: float
    extent: tuple[float, float, float, float]
    fields: dict[str, np.ndarray]
    mask: np.ndarray | None = None
    label: str = "mean"


@dataclass
class VolumeData:
    """A (possibly down-sampled) 3D field block for the 3D viewer and exports.

    Arrays are (nz, ny, nx) cell-centred; ``origin`` is the min corner of the
    block's outer cell edges [m] and ``spacing`` the cell size (dx, dy, dz) [m].
    Stored as float16 by the project store. Velocities m/s, pressure Pa, cp,
    vorticity 1/s, "q" the Q-criterion [1/s²], "dye" the smoke concentration
    0..1 (DESIGN.md 7.13). ``label`` says what the block is: "mean" (time
    averages) or "instant" (one snapshot, e.g. the smoke of the last step).
    """
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    fields: dict[str, np.ndarray]
    mask: np.ndarray | None = None
    label: str = "mean"
    mirrored: bool = False            # True when only the y >= 0 half was simulated


@dataclass
class CFD3DResult(_Serial):
    """Outcome of a 3D Navier–Stokes run (wind axes, DESIGN.md 7.2).

    Coefficients use q * ref_area; moments additionally length_ref (pitch) or
    span_ref (roll, yaw), about ``Design.moment_ref3``.

    ``volume`` is the time-averaged block (mean u, v, w, cp, q, mask), from which
    the viewer traces streamlines and draws iso-surfaces. ``extra_volumes`` holds
    further blocks a run chose to keep, each named by its ``label`` -- typically
    an "instant" snapshot with the smoke ("dye") and the velocity of the last
    step, so the Results page can show 3D smoke and instantaneous streamlines
    without re-running (DESIGN.md 7.13). Smoke is kept as a snapshot rather than
    averaged into ``volume``: a time-averaged smoke cloud is a blur that shows
    none of the filaments people look at smoke for. :meth:`find_volume` picks the
    block that holds a field.
    """
    sim_id: str
    cd: Stat = field(default_factory=Stat)        # drag (+x force)
    cs: Stat = field(default_factory=Stat)        # side force (+y)
    cl: Stat = field(default_factory=Stat)        # lift (+z; negative = downforce)
    c_roll: Stat = field(default_factory=Stat)    # rolling moment about +x
    cm: Stat = field(default_factory=Stat)        # pitching moment, + nose-up
    cn: Stat = field(default_factory=Stat)        # yawing moment about +z
    drag: float = float("nan")                    # N
    side: float = float("nan")                    # N
    lift: float = float("nan")                    # N
    moments: tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))  # N m roll, pitch, yaw
    drag_area: float = float("nan")               # Cd * ref_area [m²]
    power: float = float("nan")                   # W to overcome mean drag
    strouhal: float | None = None
    strouhal_length: float = float("nan")
    re_real: float = float("nan")
    re_sim: float = float("nan")
    grid: tuple[int, int, int] = (0, 0, 0)        # (nx, ny, nz) as solved
    cells: int = 0
    cells_per_ref: float = float("nan")
    symmetry_plane: bool = False
    t_end: float = float("nan")
    t_avg_start: float = float("nan")
    converged: bool = False
    wall_seconds: float = float("nan")
    device: str = ""
    precision: str = ""
    turbulence: str = ""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_body: dict[str, dict[str, float]] = field(default_factory=dict)  # name -> {"cd","cs","cl",...}
    # heat transfer and placed physics (DESIGN.md section 10); nan / empty when unused
    nusselt: Stat = field(default_factory=Stat)       # mean Nusselt number over heated bodies
    heat_rate: float = float("nan")                   # W (3D) or W/m (2D) from bodies into the fluid
    fans_report: list[dict[str, float]] = field(default_factory=list)   # per fan: thrust, flow, power
    diagnostics: dict[str, float] = field(default_factory=dict)        # final divergence, change norms
    # large data — persisted to arrays.npz by the project store, not to_dict:
    history: dict[str, np.ndarray] = field(default_factory=dict)        # t, cd, cs, cl, c_roll, cm, cn
    surface: list[SurfaceData3D] = field(default_factory=list)
    slices: list[SliceData] = field(default_factory=list)
    volume: VolumeData | None = None
    extra_volumes: list[VolumeData] = field(default_factory=list)   # e.g. label "instant" with "dye"

    _nested: ClassVar[dict[str, Any]] = {"cd": Stat, "cs": Stat, "cl": Stat, "c_roll": Stat,
                                         "cm": Stat, "cn": Stat, "nusselt": Stat}
    _large = ("history", "surface", "slices", "volume", "extra_volumes")

    def to_dict(self) -> dict:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self) if f.name not in self._large}

    @classmethod
    def from_dict(cls, d: dict) -> CFD3DResult:
        d = {k: v for k, v in d.items() if k not in cls._large}
        for k in ("grid", "moments"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return super().from_dict(d)

    @property
    def lift_to_drag(self) -> float:
        return self.cl.mean / self.cd.mean if self.cd.mean else float("nan")

    def volumes(self) -> list[VolumeData]:
        """Every stored volume block: ``volume`` (when present) first, then ``extra_volumes``."""
        return ([self.volume] if self.volume is not None else []) + list(self.extra_volumes)

    def find_volume(self, field_name: str, label: str | None = None) -> VolumeData | None:
        """The first block (in :meth:`volumes` order) holding ``field_name`` -- and with
        that ``label`` when one is given -- or None. ``find_volume("dye")`` is the smoke
        snapshot; ``find_volume("u", "mean")`` the mean flow for streamlines."""
        for vol in self.volumes():
            if field_name in vol.fields and (label is None or vol.label == label):
                return vol
        return None


@dataclass
class WingPoint(_Serial):
    """One finite-wing operating point (wind axes; coefficients on the wing's ref_area).

    Span arrays run over the strip centres of the whole wing (both halves when
    symmetric), y in metres: ``cl_local`` section lift coefficient, ``c_cl`` local
    chord times cl over the mean chord (span loading), ``alpha_i`` induced angle
    [deg], ``cd_local`` section profile drag (viscous runs).
    """
    alpha: float
    cl: float = float("nan")
    cd: float = float("nan")          # cdi + cdv
    cdi: float = float("nan")         # induced drag
    cdv: float = float("nan")         # viscous (profile) drag from section polars
    cm: float = float("nan")          # pitching moment about moment_ref3, + nose-up
    cy: float = float("nan")          # side force
    c_roll: float = float("nan")
    cn: float = float("nan")
    e: float = float("nan")           # span efficiency CL^2 / (pi AR CDi)
    converged: bool = False
    y: np.ndarray | None = None
    cl_local: np.ndarray | None = None
    c_cl: np.ndarray | None = None
    alpha_i: np.ndarray | None = None
    cd_local: np.ndarray | None = None

    _arrays = ("y", "cl_local", "c_cl", "alpha_i", "cd_local")


@dataclass
class WingResult(_Serial):
    """A finite-wing sweep: geometry metrics plus :class:`WingPoint` per alpha.

    ``stall_alpha`` [deg] / ``stall_cl`` are the critical-section estimate of stall
    onset (the angle and wing CL at which the first strip away from the tips reaches
    its section cl_max, interpolated between computed angles). It is the wing's
    CL_max estimate, not a stall prediction; NaN when the sweep did not reach it,
    for inviscid sweeps (no section cl_max) and in results saved before the field
    existed. The same estimate is quoted in ``notes``.
    """
    sim_id: str
    method: str = "vlm"
    viscous: bool = True
    area: float = float("nan")        # planform area [m²]
    span: float = float("nan")        # [m]
    aspect_ratio: float = float("nan")
    mac: float = float("nan")         # mean aerodynamic chord [m]
    re_mac: float = float("nan")
    points: list[WingPoint] = field(default_factory=list)
    wall_seconds: float = float("nan")
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stall_alpha: float = float("nan")  # deg, critical-section stall onset (NaN = none)
    stall_cl: float = float("nan")     # wing CL at stall_alpha

    _nested_list: ClassVar[dict[str, Any]] = {"points": WingPoint}

    def table(self) -> dict[str, np.ndarray]:
        """Converged points as arrays: alpha, cl, cd, cdi, cdv, cm, e."""
        pts = [p for p in self.points if p.converged]
        keys = ("alpha", "cl", "cd", "cdi", "cdv", "cm", "e")
        return {k: np.array([getattr(p, k) for p in pts], dtype=float) for k in keys}
