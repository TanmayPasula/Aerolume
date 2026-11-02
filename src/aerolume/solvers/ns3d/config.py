"""Configuration types of the 3D Navier-Stokes solver (DESIGN.md section 7.5).

Everything here is **non-dimensional**, exactly as in 2D: lengths in units of
the reference length L (sphere diameter, vehicle length, mean aerodynamic
chord), velocities in units of the free-stream speed U, time in L/U and
pressure in rho U^2. The 3D planner (``aerolume.simulation.plan3d``) owns the
conversion from the SI model types (``Mesh3D`` in metres) to these.

Frame (DESIGN.md 7.2): right-handed, x downstream, y to the left (span), z up.
The flow enters at x = x0 along +x. Forces come out in wind axes (Fx drag, Fy
side force, Fz lift) and moments about ``moment_ref`` (Mx roll, My pitch,
positive nose-up, Mz yaw).

The trailing fields after ``perturb`` are advanced numerics knobs with defaults
validated in ``docs/theory_ns3d.md``; normal callers never set them.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

INLETS = ("uniform",)
DOMAINS = ("tunnel", "closed")
BC_X = ("slip", "noslip")                 # the x = x0 / x1 walls of a closed domain
BC_Y = ("slip", "noslip", "symmetry_ymin", "periodic")
BC_ZMIN = ("slip", "noslip", "moving")
BC_ZMAX = ("slip", "noslip")
ADVECTION = ("upwind3", "central")
TURBULENCE = ("none", "smagorinsky", "wale", "vreman")
DTYPES = ("float32", "float64")
ROLES = ("body", "wheel", "element")
DYE_KINDS = ("rake", "grid", "plane")
# placed physics (DESIGN.md section 10), the solver-unit mirror of aerolume.model
THERMAL_MODES = ("adiabatic", "temperature", "heat_flux")
#: domain sides in the order of every per-side array: x0, x1, y0, y1, z0, z1
WALL_SIDES = ("left", "right", "front", "back", "bottom", "top")
EMITTER_SPECIES = ("dye", "particles")
EMITTER_KINDS = ("point", "line", "rake", "rect", "disc", "plane")
FAN_MODES = ("thrust", "pressure_jump", "velocity")
PORT_KINDS = ("jet", "vent")
HEAT_SOURCE_KINDS = ("box", "sphere")
ADAPTIVE_DT = ("cfl", "pid")
#: smoke channels: "dye" (channel 0) and "dye1" .. "dye3"
MAX_DYE_CHANNELS = 4


def _is_tensor(x: Any) -> bool:
    """True for a torch tensor, without importing torch here."""
    return hasattr(x, "detach") and hasattr(x, "requires_grad")


def _vec3(value: Any, name: str, *, unit: bool = False) -> tuple[float, float, float]:
    """A finite 3-vector (normalised with ``unit``), or ValueError naming the field."""
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be three finite numbers")
    if unit:
        n = float(np.linalg.norm(v))
        if not n > 0:
            raise ValueError(f"{name} must be a non-zero vector")
        v = v / n
    return (float(v[0]), float(v[1]), float(v[2]))


def _finite(value: Any, name: str, *, positive: bool = False, nonneg: bool = False) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if positive and not x > 0:
        raise ValueError(f"{name} must be positive")
    if nonneg and x < 0:
        raise ValueError(f"{name} must be >= 0")
    return x


@dataclass
class ImmersedMesh:
    """One triangulated surface in solver units.

    ``vertices`` (N, 3) and ``faces`` (M, 3) describe a triangle soup that does
    not have to be closed or consistently oriented: the mask builder decides
    inside / outside robustly (the shared voxeliser of ``aerolume.geometry3d``,
    or the solver's generalised-winding-number fallback). ``vertices`` may be a
    torch tensor that requires grad; the masks, the flow and the forces are then
    differentiable with respect to the vertex positions (shape sensitivities).

    The body may spin rigidly about ``axis`` through ``center`` with angular
    velocity ``omega`` (right-hand rule, units of U/L): the solid velocity is
    ``omega * axis x (x - center)``. That is how wheels are modelled; the surface
    itself never moves, so the solid mask stays static.

    Sign for rolling wheels (DESIGN 7.2 frame: x downstream, y left, z up): with
    ``axis = (0, 1, 0)`` the point below the axle moves at ``-omega r`` in x, so a
    wheel of radius r that rolls without slip on a road moving at
    ``+ground_speed`` has ``omega = -ground_speed / r`` (equivalently
    ``+ground_speed / r`` about ``axis = (0, -1, 0)``). The 2D convention
    (counter-clockwise, +U/r) is the same rotation seen from the -y side.

    Heat transfer (used only when the configuration has ``thermal``):
    ``thermal = "adiabatic"`` (no heat crosses the surface), ``"temperature"``
    (the body is held at ``thermal_value`` = theta, the non-dimensional
    temperature (T - T_amb) / dT_ref) or ``"heat_flux"`` (``thermal_value`` =
    G = q'' L / (k dT_ref), the dimensionless heat flux into the fluid).
    """
    vertices: Any
    faces: Any
    name: str = ""
    role: str = "body"
    omega: float = 0.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    thermal: str = "adiabatic"
    thermal_value: float = 0.0

    def __post_init__(self) -> None:
        if self.thermal not in THERMAL_MODES:
            raise ValueError(f"unknown thermal mode {self.thermal!r}; use one of {', '.join(THERMAL_MODES)}")
        self.thermal_value = _finite(self.thermal_value, "thermal_value")
        if not _is_tensor(self.vertices):
            self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        elif self.vertices.ndim != 2 or self.vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        faces = np.asarray(self.faces.detach().cpu().numpy() if _is_tensor(self.faces) else self.faces)
        self.faces = faces.astype(np.int64).reshape(-1, 3)
        n = int(self.vertices.shape[0])
        if len(self.faces) == 0:
            raise ValueError(f"immersed mesh {self.name!r} has no faces")
        if self.faces.min() < 0 or self.faces.max() >= n:
            raise ValueError(f"immersed mesh {self.name!r}: face indices out of range 0..{n - 1}")
        if self.role not in ROLES:
            raise ValueError(f"unknown body role {self.role!r}")
        self.omega = float(self.omega)
        self.center = tuple(float(c) for c in self.center)        # type: ignore[assignment]
        ax = np.asarray(self.axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(ax))
        if not norm > 0:
            raise ValueError("the rotation axis must be a non-zero vector")
        self.axis = tuple(float(c) for c in ax / norm)              # type: ignore[assignment]

    def vertices_numpy(self) -> np.ndarray:
        """The vertex positions as a float64 numpy array (detached)."""
        if _is_tensor(self.vertices):
            return self.vertices.detach().cpu().double().numpy()
        return self.vertices

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """(xmin, xmax, ymin, ymax, zmin, zmax) in solver units."""
        v = self.vertices_numpy()
        lo, hi = v.min(axis=0), v.max(axis=0)
        return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


@dataclass(eq=False)
class DyeRake:
    """Smoke emitters upstream of the body (DESIGN.md 7.13), in the units of the
    configuration that holds them.

    In a :class:`SolverConfig3D` lengths are solver units (metres / length_ref,
    the same origin) and times convective times (L/U); the runner and the live
    session build it from the GUI's metres and seconds with :meth:`scaled`.

    * ``kind``: "rake" (a row of nozzles across the span), "grid" (a span x height
      grid of nozzles) or "plane" (an emitting band in a plane x = const);
    * ``positions``: (N, 3) nozzle centres (x, y, z) for "rake" and "grid"; for
      "plane" two opposite corners ``[(x, y0, z0), (x, y1, z1)]`` of the band;
    * ``radius``: nozzle radius (0 = ``dye.NOZZLE_RADIUS_CELLS`` cells);
    * ``pulse_on`` / ``pulse_off`` (both > 0): release for ``pulse_on``, pause for
      ``pulse_off``, repeating -- the streams are cut into time lines; 0 means
      continuous release.
    """
    kind: str = "rake"
    positions: Any = field(default_factory=lambda: np.zeros((0, 3)))
    radius: float = 0.0
    pulse_on: float = 0.0
    pulse_off: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in DYE_KINDS:
            raise ValueError(f"unknown dye rake kind {self.kind!r}; use one of {', '.join(DYE_KINDS)}")
        pos = np.asarray(self.positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(pos)):
            raise ValueError("dye rake positions must be finite")
        if self.kind == "plane" and len(pos) != 2:
            raise ValueError("a 'plane' dye rake needs two corners [(x, y0, z0), (x, y1, z1)]")
        self.positions = pos
        self.radius = float(self.radius)
        self.pulse_on = float(self.pulse_on)
        self.pulse_off = float(self.pulse_off)
        if not (self.radius >= 0 and math.isfinite(self.radius)):
            raise ValueError("dye nozzle radius must be >= 0")
        if not (self.pulse_on >= 0 and self.pulse_off >= 0):
            raise ValueError("dye pulse times must be >= 0")

    def __eq__(self, other: object) -> bool:
        # the generated __eq__ would compare numpy arrays element-wise (ambiguous truth value)
        if not isinstance(other, DyeRake):
            return NotImplemented
        return (self.kind == other.kind and self.radius == other.radius and self.pulse_on == other.pulse_on
                and self.pulse_off == other.pulse_off and np.array_equal(self.positions, other.positions))

    __hash__ = None             # mutable value object

    @property
    def pulsed(self) -> bool:
        return self.pulse_on > 0 and self.pulse_off > 0

    def scaled(self, length: float, time: float | None = None, *, shift=(0.0, 0.0, 0.0)) -> DyeRake:
        """A copy with lengths ``(x - shift) * length`` and times ``* time``
        (metres and seconds -> solver units: ``length = 1 / L``, ``time = U / L``)."""
        s = float(length)
        t = float(time) if time is not None else 1.0
        pos = (np.asarray(self.positions, dtype=np.float64) - np.asarray(shift, dtype=np.float64)) * s
        return DyeRake(kind=self.kind, positions=pos, radius=self.radius * s, pulse_on=self.pulse_on * t,
                       pulse_off=self.pulse_off * t)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "positions": np.asarray(self.positions).tolist(), "radius": self.radius,
                "pulse_on": self.pulse_on, "pulse_off": self.pulse_off}

    @classmethod
    def coerce(cls, value) -> DyeRake | None:
        """None, a DyeRake or its dict form -> DyeRake | None."""
        if value is None or isinstance(value, DyeRake):
            return value
        if isinstance(value, dict):
            keys = ("kind", "positions", "radius", "pulse_on", "pulse_off")
            return cls(**{k: value[k] for k in keys if k in value})
        raise TypeError(f"dye_rake must be a DyeRake, a dict or None, not {type(value).__name__}")


# ---------------------------------------------------------------------------
# placed physics (DESIGN.md section 10), solver units
# ---------------------------------------------------------------------------

def _coerce(cls, value):
    """An instance of ``cls`` from itself or its keyword dict."""
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"expected a {cls.__name__} or a dict, not {type(value).__name__}")


@dataclass(eq=False)
class Emitter3D:
    """A flow-visualisation source in solver units (the model's ``Emitter``,
    converted by the planner: metres / L, seconds * U / L, wind frame).

    ``species``: "dye" (a passive smoke channel, ``channel`` 0..3 = fields
    "dye", "dye1", "dye2", "dye3") or "particles" (GPU tracer particles).
    Footprint by ``kind``: "point" a ball of ``radius`` at ``position``;
    "line" a slot (capsule) from ``position`` to ``end``; "rake" ``count``
    nozzles along that segment; "rect" an axis-aligned box of ``size`` (sx, sy,
    sz; a zero extent is a two-cell-thick slab) centred at ``position``;
    "disc" a disc of radius ``size[0]`` facing ``normal``; "plane" a rectangle of
    extents ``size`` = (a, b) facing ``normal`` (no size: the whole domain cross
    section). ``radius`` 0 means ~1.25 cells (the default nozzle).

    ``rate``: the concentration the smoke is relaxed to inside the footprint
    (0..1), or particles released per unit time (L/U). ``pulse_period`` > 0
    releases only during the first ``pulse_duty`` of each period (timelines);
    the release runs from ``start`` to ``stop`` (None = forever), in solver
    time. ``lifetime`` (particles): age after which a particle is recycled.
    """
    species: str = "dye"
    kind: str = "point"
    position: Any = (0.0, 0.0, 0.0)
    end: Any = None
    size: tuple = ()
    normal: Any = None
    count: int = 8
    radius: float = 0.0
    rate: float = 1.0
    channel: int = 0
    pulse_period: float = 0.0
    pulse_duty: float = 0.5
    start: float = 0.0
    stop: float | None = None
    lifetime: float | None = None
    enabled: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.species not in EMITTER_SPECIES:
            raise ValueError(f"unknown emitter species {self.species!r}")
        if self.kind not in EMITTER_KINDS:
            raise ValueError(f"unknown emitter kind {self.kind!r}; use one of {', '.join(EMITTER_KINDS)}")
        self.position = _vec3(self.position, "emitter position")
        if self.kind in ("line", "rake"):
            if self.end is None:
                raise ValueError(f"a {self.kind!r} emitter needs an end point")
            self.end = _vec3(self.end, "emitter end")
        elif self.end is not None:
            self.end = _vec3(self.end, "emitter end")
        self.normal = _vec3(self.normal if self.normal is not None else (1.0, 0.0, 0.0), "emitter normal", unit=True)
        self.size = tuple(_finite(s, "emitter size", nonneg=True) for s in self.size)
        if self.kind == "disc" and (not self.size or not self.size[0] > 0):
            raise ValueError("a 'disc' emitter needs a positive radius size=(r,)")
        if self.kind == "rect" and len(self.size) != 3:
            raise ValueError("a 'rect' emitter needs size=(sx, sy, sz)")
        if self.kind == "plane" and len(self.size) not in (0, 2):
            raise ValueError("a 'plane' emitter takes size=(a, b) or no size (the whole cross-section)")
        self.count = max(1, int(self.count))
        self.radius = _finite(self.radius, "emitter radius", nonneg=True)
        self.rate = _finite(self.rate, "emitter rate", nonneg=True)
        self.channel = int(self.channel)
        if not 0 <= self.channel < MAX_DYE_CHANNELS:
            raise ValueError(f"dye channel must be 0..{MAX_DYE_CHANNELS - 1}")
        self.pulse_period = _finite(self.pulse_period, "pulse_period", nonneg=True)
        self.pulse_duty = min(max(_finite(self.pulse_duty, "pulse_duty"), 0.0), 1.0)
        self.start = _finite(self.start, "start")
        self.stop = None if self.stop is None else _finite(self.stop, "stop")
        self.lifetime = None if self.lifetime is None else _finite(self.lifetime, "lifetime", positive=True)
        self.enabled = bool(self.enabled)

    __hash__ = None             # mutable value object

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Emitter3D):
            return NotImplemented
        return dataclasses.astuple(self) == dataclasses.astuple(other)

    def scaled(self, length: float, time: float = 1.0, *, shift=(0.0, 0.0, 0.0), rotate=None) -> Emitter3D:
        """A copy in solver units: positions ``R (x - shift) * length`` (``rotate`` an
        optional 3 x 3 rotation, e.g. a body-frame emitter following the attitude),
        sizes ``* length``, times ``* time`` and particle rates ``/ time``."""
        s, t = float(length), float(time)
        sh = np.asarray(shift, dtype=np.float64)
        R = np.eye(3) if rotate is None else np.asarray(rotate, dtype=np.float64)

        def pos(p):
            return None if p is None else tuple(float(c) for c in R @ ((np.asarray(p) - sh) * s))
        return dataclasses.replace(
            self, position=pos(self.position), end=pos(self.end), normal=tuple(R @ np.asarray(self.normal)),
            size=tuple(c * s for c in self.size), radius=self.radius * s,
            rate=self.rate / t if self.species == "particles" else self.rate,
            pulse_period=self.pulse_period * t, start=self.start * t,
            stop=None if self.stop is None else self.stop * t,
            lifetime=None if self.lifetime is None else self.lifetime * t)


@dataclass
class ThermalConfig3D:
    """Heat transfer in solver units (DESIGN.md 10.2): theta = (T - T_amb) / dT_ref.

    ``prandtl``: the thermal diffusivity is 1 / Pe with Pe = Re Pr (it follows
    ``set_inflow(re=...)``); ``richardson`` = g beta dT_ref L / U^2, the
    Boussinesq buoyancy ``richardson * theta`` along +z (gravity along -z; 0 =
    forced convection, theta is then passive); ``prandtl_turbulent`` the
    turbulent Prandtl number used with an LES model.

    ``walls`` sets domain sides -- "left" (x0), "right" (x1), "front" (y0),
    "back" (y1), "bottom" (z0), "top" (z1) -- as a dict ``{side: (mode,
    value)}`` or a list of objects / dicts with ``side``, ``mode``, ``value``
    (the 2D ``WallBC`` and the model's ``WallThermal`` both fit): "adiabatic",
    "temperature" (``value`` = theta on the wall) or "heat_flux" (``value`` = G
    = q'' L / (k dT_ref), the dimensionless gradient into the fluid). Unlisted
    sides are adiabatic; a tunnel's inlet carries ``theta_inlet`` and its
    outlet is zero-gradient whatever ``walls`` says, and a symmetry plane is
    adiabatic. ``theta_initial`` is the starting field; ``eta`` the penalisation
    permeability of heated bodies (None: the momentum rule with Pe for Re).

    Heat quantities are in conduction units: a heat rate Q means Q k dT_ref L
    watts, so a body's Nusselt number is Q / (A theta_body) with A its area in
    units of L^2.
    """
    prandtl: float = 0.71
    richardson: float = 0.0
    prandtl_turbulent: float = 0.85
    walls: Any = field(default_factory=dict)
    theta_inlet: float = 0.0
    theta_initial: float = 0.0
    eta: float | None = None

    def __post_init__(self) -> None:
        self.prandtl = _finite(self.prandtl, "Prandtl number", positive=True)
        self.richardson = _finite(self.richardson, "Richardson number")
        self.prandtl_turbulent = _finite(self.prandtl_turbulent, "turbulent Prandtl number", positive=True)
        self.theta_inlet = _finite(self.theta_inlet, "inlet temperature")
        self.theta_initial = _finite(self.theta_initial, "initial temperature")
        if self.eta is not None:
            self.eta = _finite(self.eta, "thermal eta", positive=True)
        if isinstance(self.walls, dict):
            items = list(self.walls.items())
        else:
            items = []
            for wspec in self.walls:
                if isinstance(wspec, dict):
                    items.append((wspec["side"], (wspec.get("mode", "adiabatic"), wspec.get("value", 0.0))))
                else:
                    items.append((wspec.side, (wspec.mode, getattr(wspec, "value", 0.0))))
        walls = {}
        for side, spec in items:
            if side not in WALL_SIDES:
                raise ValueError(f"unknown wall side {side!r}; use one of {', '.join(WALL_SIDES)}")
            if side in walls:
                raise ValueError(f"the {side} wall is listed twice")
            mode, value = (spec, 0.0) if isinstance(spec, str) else tuple(spec)
            if mode not in THERMAL_MODES:
                raise ValueError(f"unknown thermal mode {mode!r} for the {side} wall")
            walls[side] = (mode, _finite(value, f"{side} wall value"))
        self.walls = walls

    def wall(self, side: str) -> tuple[str, float]:
        """(mode, value) of a side (adiabatic when not listed)."""
        return self.walls.get(side, ("adiabatic", 0.0))


@dataclass
class HeatSource3D:
    """A volumetric heater in solver units: a "box" of extents ``size`` (sx, sy,
    sz) or a "sphere" of radius ``size[0]`` centred at ``center``, releasing
    ``power`` in conduction units (P / (k dT_ref L); negative cools), spread
    uniformly with a one-cell anti-aliased edge; the discrete total is exact.
    ``axes`` (None = the grid axes) is a 3x3 rotation whose columns are the box's own
    x, y, z axes in the solver frame: a body-frame heater turns with the body at an
    angle of attack or yaw instead of staying aligned with the grid."""
    kind: str = "box"
    center: Any = (0.0, 0.0, 0.0)
    size: tuple = (0.1, 0.1, 0.1)
    power: float = 1.0
    enabled: bool = True
    name: str = ""
    axes: Any = None

    def __post_init__(self) -> None:
        if self.kind not in HEAT_SOURCE_KINDS:
            raise ValueError(f"unknown heat source kind {self.kind!r}; use 'box' or 'sphere'")
        if self.axes is not None:
            r = np.asarray(self.axes, dtype=np.float64).reshape(3, 3)
            if not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
                raise ValueError("heat source axes must be a rotation (orthonormal columns)")
            self.axes = tuple(tuple(float(v) for v in row) for row in r)
        self.center = _vec3(self.center, "heat source centre")
        self.size = tuple(_finite(s, "heat source size", positive=True) for s in self.size)
        if (self.kind == "box" and len(self.size) != 3) or (self.kind == "sphere" and len(self.size) < 1):
            raise ValueError("a box heat source needs size=(sx, sy, sz), a sphere size=(r,)")
        self.power = _finite(self.power, "heat source power")


@dataclass
class Fan3D:
    """An actuator-disc fan in solver units (DESIGN.md 10.3): a thin disc of
    ``diameter`` and ``thickness`` (None: diameter / 8, at least two cells)
    centred at ``center`` and blowing along ``direction``.

    ``mode`` "thrust": ``value`` = T / (rho U^2 L^2); "pressure_jump": ``value``
    = dp / (rho U^2) (thrust = dp A); "velocity": ``value`` = the target
    disc-averaged axial speed (units U), reached by an in-graph PI controller on
    the thrust. ``swirl`` imparts a solid-body tangential velocity of ``swirl``
    times the axial disc velocity at the rim (right-hand rule about
    ``direction``).
    """
    center: Any = (0.0, 0.0, 0.0)
    direction: Any = (1.0, 0.0, 0.0)
    diameter: float = 1.0
    thickness: float | None = None
    mode: str = "thrust"
    value: float = 0.1
    swirl: float = 0.0
    enabled: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.mode not in FAN_MODES:
            raise ValueError(f"unknown fan mode {self.mode!r}; use one of {', '.join(FAN_MODES)}")
        self.center = _vec3(self.center, "fan centre")
        self.direction = _vec3(self.direction, "fan direction", unit=True)
        self.diameter = _finite(self.diameter, "fan diameter", positive=True)
        if self.thickness is not None:
            self.thickness = _finite(self.thickness, "fan thickness", positive=True)
        self.value = _finite(self.value, "fan value")
        self.swirl = _finite(self.swirl, "fan swirl")

    @property
    def area(self) -> float:
        return math.pi * 0.25 * self.diameter ** 2


@dataclass
class Port3D:
    """A jet (fluid injected) or vent (fluid removed) in solver units.

    A disc of ``diameter`` and ``thickness`` (None: max(diameter / 4, two cells))
    at ``center``; the volume flow is ``speed * pi diameter^2 / 4``. A jet adds
    that much fluid moving at ``speed`` along ``direction`` (a mass source with
    its momentum, so the jet's mass and momentum fluxes are exact), with
    temperature ``theta`` (None = ambient, 0) and, with ``dye_channel``, smoke
    on that channel; a vent removes the same flow with the local velocity,
    temperature and smoke. ``direction`` is unused by a vent.
    """
    kind: str = "jet"
    center: Any = (0.0, 0.0, 0.0)
    direction: Any = (1.0, 0.0, 0.0)
    diameter: float = 0.1
    speed: float = 1.0
    thickness: float | None = None
    theta: float | None = None
    dye_channel: int | None = None
    enabled: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PORT_KINDS:
            raise ValueError(f"unknown port kind {self.kind!r}; use 'jet' or 'vent'")
        self.center = _vec3(self.center, "port centre")
        self.direction = _vec3(self.direction, "port direction", unit=True)
        self.diameter = _finite(self.diameter, "port diameter", positive=True)
        self.speed = _finite(self.speed, "port speed", nonneg=True)
        if self.thickness is not None:
            self.thickness = _finite(self.thickness, "port thickness", positive=True)
        if self.theta is not None:
            self.theta = _finite(self.theta, "port temperature")
        if self.dye_channel is not None and not 0 <= int(self.dye_channel) < MAX_DYE_CHANNELS:
            raise ValueError(f"port dye channel must be 0..{MAX_DYE_CHANNELS - 1}")

    @property
    def flow(self) -> float:
        """Volume flow (units U L^2): positive for a jet, negative for a vent."""
        q = self.speed * math.pi * 0.25 * self.diameter ** 2
        return q if self.kind == "jet" else -q


@dataclass
class SolverConfig3D:
    """Grid, physics and numerics of one 3D flow solve.

    The domain is ``[x0, x0 + nx h] x [y0, y0 + ny h] x [z0, z0 + nz h]`` on a
    uniform staggered (MAC) grid. Flow enters at x = x0 along +x (unless
    ``periodic``) and leaves through a convective outlet.

    * ``bc_y``: ``"slip"`` walls on both y sides; ``"symmetry_ymin"`` makes the
      y = y0 plane a symmetry plane (a half-domain run of a mirror-symmetric
      body; the far side stays slip, and forces and moments are reported for the
      full body, DESIGN.md 7.2); ``"periodic"`` in y.
    * ``bc_zmin``: ``"slip"``, ``"noslip"`` or ``"moving"`` (a road moving at
      ``ground_speed`` in +x); ``bc_zmax``: ``"slip"`` or ``"noslip"``.
    * ``periodic``: verification mode, periodic in x, y and z (no inlet).

    Coefficients: ``C = F / (0.5 ref_area)``, pitch ``Cm = My / (0.5 ref_area
    length_ref)``, roll and yaw ``M / (0.5 ref_area span_ref)``, moments about
    ``moment_ref``.

    Advanced knobs: ``eta`` (Brinkman permeability, None = the calibrated rule),
    ``mask_width`` (half-width of the diffuse interface in cells),
    ``les_constant`` (None = model default), ``fused_kernels`` (use the fused
    CUDA kernels of :mod:`.fused` on the fast path when available),
    ``window_pitch_deg`` / ``window_yaw_deg`` (size the mask window so the bodies
    can be pitched / yawed live about ``moment_ref`` by up to these angles
    without growing it, which keeps captured CUDA graphs valid).

    Smoke (DESIGN.md 7.13): with ``dye`` the solver transports a passive scalar
    released by ``dye_rake`` (:class:`DyeRake`, solver units; None = a rake
    upstream of the bodies chosen by ``dye.default_dye_rake``), readable as
    ``FlowSolver3D.field("dye")``. It costs a few per cent of a step and never
    affects the flow.

    Placed physics (DESIGN.md section 10; everything off by default, and the
    default path is then the v0.2 solver unchanged):

    * ``domain = "closed"``: every side is a wall (``bc_x`` at x0 / x1, ``bc_y``,
      ``bc_zmin``, ``bc_zmax``), no inflow -- cavities, rooms, enclosures where
      buoyancy, fans and jets drive the flow;
    * ``thermal`` (:class:`ThermalConfig3D`): temperature transport, Boussinesq
      buoyancy, heated bodies (``ImmersedMesh.thermal``), wall conditions;
      ``heat_sources`` (:class:`HeatSource3D`);
    * ``fans`` (:class:`Fan3D`, actuator discs) and ``ports``
      (:class:`Port3D`, jets and vents);
    * ``emitters`` (:class:`Emitter3D`): smoke channels and tracer particles
      (None keeps the v0.2 behaviour: ``dye_rake`` on channel 0); at most
      ``particle_capacity`` particles (0: 2^16 per particle emitter);
    * ``diagnostics``: a device-side history of divergence, change norms, CFL,
      the Poisson residual (``FlowSolver3D.diagnostics()``); ``adaptive_dt =
      "pid"``: a PID controller holds the CFL number near ``pid_target`` times
      ``cfl`` instead of jumping to the limit every step.
    """
    nx: int
    ny: int
    nz: int
    h: float
    x0: float
    y0: float
    z0: float
    re: float
    bodies: list[ImmersedMesh] = field(default_factory=list)
    inlet: str = "uniform"
    bc_y: str = "slip"
    bc_zmin: str = "slip"
    bc_zmax: str = "slip"
    ground_speed: float = 1.0
    periodic: bool = False
    advection: str = "upwind3"
    turbulence: str = "none"
    cfl: float = 0.5
    dt: float | None = None
    ref_area: float = 1.0
    length_ref: float = 1.0
    span_ref: float = 1.0
    moment_ref: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dtype: str = "float32"
    device: str = "auto"
    use_cuda_graphs: bool = True
    perturb: bool = True
    # --- advanced numerics (defaults validated in docs/theory_ns3d.md) ---------
    eta: float | None = None
    mask_width: float = 1.0
    les_constant: float | None = None
    fused_kernels: bool = True
    window_pitch_deg: float = 0.0
    window_yaw_deg: float = 0.0
    # --- passive smoke (DESIGN.md 7.13) ---------------------------------------
    dye: bool = True
    dye_rake: DyeRake | None = None
    # --- placed physics (DESIGN.md section 10) ---------------------------------
    domain: str = "tunnel"
    bc_x: str = "noslip"
    thermal: ThermalConfig3D | None = None
    heat_sources: list = field(default_factory=list)
    fans: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    emitters: list | None = None
    particle_capacity: int = 0
    diagnostics: bool = False
    adaptive_dt: str = "cfl"
    pid_target: float = 0.85

    def __post_init__(self) -> None:
        self.nx, self.ny, self.nz = int(self.nx), int(self.ny), int(self.nz)
        if min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("the grid needs at least 4 x 4 x 4 cells")
        if not (self.h > 0 and math.isfinite(self.h)):
            raise ValueError("cell size h must be positive")
        if not (self.re > 0 and math.isfinite(self.re)):
            raise ValueError("Reynolds number must be positive and finite")
        for name, value, allowed in (("inlet", self.inlet, INLETS), ("bc_y", self.bc_y, BC_Y),
                                     ("bc_zmin", self.bc_zmin, BC_ZMIN), ("bc_zmax", self.bc_zmax, BC_ZMAX),
                                     ("advection", self.advection, ADVECTION),
                                     ("turbulence", self.turbulence, TURBULENCE),
                                     ("dtype", self.dtype, DTYPES)):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; use one of {', '.join(allowed)}")
        if not 0 < self.cfl <= 1.5:
            raise ValueError("cfl must be in (0, 1.5]")
        if self.dt is not None and not self.dt > 0:
            raise ValueError("a fixed dt must be positive")
        if self.mask_width <= 0:
            raise ValueError("mask_width must be positive")
        if self.eta is not None and not self.eta > 0:
            raise ValueError("eta must be positive")
        for name in ("ref_area", "length_ref", "span_ref"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        self.bodies = [b if isinstance(b, ImmersedMesh) else ImmersedMesh(*b) for b in self.bodies]
        self.moment_ref = tuple(float(c) for c in self.moment_ref)        # type: ignore[assignment]
        self.window_pitch_deg = abs(float(self.window_pitch_deg))
        self.window_yaw_deg = abs(float(self.window_yaw_deg))
        self.dye = bool(self.dye)
        self.dye_rake = DyeRake.coerce(self.dye_rake)
        for name, value, allowed in (("domain", self.domain, DOMAINS), ("bc_x", self.bc_x, BC_X),
                                     ("adaptive_dt", self.adaptive_dt, ADAPTIVE_DT)):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; use one of {', '.join(allowed)}")
        if self.domain == "closed" and self.periodic:
            raise ValueError("a closed domain cannot be periodic")
        if self.thermal is not None:
            self.thermal = _coerce(ThermalConfig3D, self.thermal)
        self.heat_sources = [_coerce(HeatSource3D, s) for s in self.heat_sources]
        self.fans = [_coerce(Fan3D, f) for f in self.fans]
        self.ports = [_coerce(Port3D, p) for p in self.ports]
        if self.emitters is not None:
            self.emitters = [_coerce(Emitter3D, e) for e in self.emitters]
        self.particle_capacity = max(0, int(self.particle_capacity))
        self.diagnostics = bool(self.diagnostics)
        if not 0.1 <= self.pid_target <= 1.0:
            raise ValueError("pid_target must be in [0.1, 1]")
        if self.heat_sources and self.thermal is None:
            raise ValueError("heat sources need heat transfer (thermal=ThermalConfig3D(...))")

    @property
    def closed(self) -> bool:
        """True for a closed domain (walls on every side, no inflow)."""
        return self.domain == "closed"

    @property
    def has_placed_physics(self) -> bool:
        """True when any DESIGN.md section 10 feature is configured (the solver then
        runs the extended step; otherwise the v0.2 step, unchanged)."""
        return bool(self.thermal is not None or self.fans or self.ports or self.diagnostics
                    or self.adaptive_dt != "cfl")

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """(xmin, xmax, ymin, ymax, zmin, zmax) of the domain in solver units."""
        return (self.x0, self.x0 + self.nx * self.h, self.y0, self.y0 + self.ny * self.h,
                self.z0, self.z0 + self.nz * self.h)

    @property
    def nu(self) -> float:
        """Non-dimensional kinematic viscosity, 1 / Re."""
        return 1.0 / self.re

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def symmetry(self) -> bool:
        """True when the y = y0 plane is a symmetry plane (half-domain run)."""
        return self.bc_y == "symmetry_ymin" and not self.periodic

    @property
    def periodic_axes(self) -> tuple[bool, bool, bool]:
        """Periodicity of the (x, y, z) axes."""
        if self.periodic:
            return (True, True, True)
        return (False, self.bc_y == "periodic", False)

    @property
    def side_walls(self) -> dict[str, str]:
        """The velocity condition on each side ("inflow", "outflow", "slip", "noslip",
        "moving", "symmetry", "periodic"), keyed like :data:`WALL_SIDES`."""
        if self.periodic:
            return dict.fromkeys(WALL_SIDES, "periodic")
        x = (self.bc_x, self.bc_x) if self.closed else ("inflow", "outflow")
        y = {"periodic": ("periodic", "periodic"), "symmetry_ymin": ("symmetry", "slip")}.get(
            self.bc_y, (self.bc_y, self.bc_y))
        return dict(zip(WALL_SIDES, (*x, *y, self.bc_zmin, self.bc_zmax)))
