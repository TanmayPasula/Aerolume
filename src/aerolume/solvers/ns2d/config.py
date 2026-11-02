"""Configuration types of the 2D Navier-Stokes solver.

Everything here is **non-dimensional**: lengths are in units of the reference
length L (chord, cylinder diameter, vehicle length), velocities in units of the
free-stream speed U, time in L/U and pressure in rho U^2. The simulation planner
(:mod:`aerolume.simulation.plan`) owns the conversion from the SI model types;
the solver never sees metres.

The two dataclasses are the public contract of DESIGN.md section 5.2. A few
trailing fields (``eta``, ``mask_width``, ``les_constant``) are advanced numerics
knobs with defaults chosen by the validation in ``docs/theory_ns2d.md``; normal
callers never set them.

Placed physics (DESIGN.md section 10, theory doc sections 15-20) is described by
the specs below - :class:`EmitterSpec`, :class:`ThermalConfig` (+ :class:`WallBC`),
:class:`HeatSourceSpec`, :class:`FanSpec`, :class:`PortSpec` - also in solver
units. Every one of them defaults to "absent", so a config without them runs
exactly the validated v0.1 solver.

Non-dimensional thermal quantities (the planner converts from SI):

* temperature ``theta = (T - T_ambient) / dT_ref``;
* diffusivity ``1 / Pe`` with ``Pe = Re Pr`` (so ``ThermalConfig`` stores Pr and the
  Peclet number follows live Reynolds-number changes);
* buoyancy ``f = -Ri theta g_hat`` with ``Ri = g beta dT_ref L / U^2``;
* a heat flux enters as the dimensionless wall-normal gradient
  ``G = q'' L / (k dT_ref)`` (the flux into the fluid is ``G / Pe`` in solver units);
* a heat rate (heater power, body heat transfer) is ``Q / (rho c_p U dT_ref L)``
  per unit span.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INLETS = ("uniform", "parabolic")
BC_TOP = ("slip", "noslip")
BC_BOTTOM = ("slip", "noslip", "moving")
BC_SIDE = ("noslip", "slip")
ADVECTION = ("upwind3", "central")
TURBULENCE = ("none", "smagorinsky", "wale", "vreman")
DTYPES = ("float32", "float64")
DOMAINS = ("tunnel", "closed")
ADAPTIVE_DT = ("cfl", "pid")
#: wall-shear models of the domain walls (theory doc 18.4); opt-in, see there for why
WALL_MODELS = ("none", "werner_wengle")
THERMAL_MODES = ("adiabatic", "temperature", "heat_flux")
WALL_SIDES = ("left", "right", "bottom", "top")
EMITTER_KINDS = ("point", "line", "rake", "rect", "disc")
EMITTER_SPECIES = ("dye", "particles")
FAN_MODES = ("thrust", "pressure_jump", "velocity")
PORT_KINDS = ("jet", "vent")
HEAT_SOURCE_KINDS = ("rect", "disc")
#: dye channels: "dye" (channel 0) and "dye1".."dye3"
MAX_DYE_CHANNELS = 4


def _pair(v, name: str) -> tuple[float, float]:
    """A finite 2-vector from anything array-like (3D inputs keep x and y)."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size < 2 or not np.all(np.isfinite(a[:2])):
        raise ValueError(f"{name} must be a finite (x, y) pair")
    return (float(a[0]), float(a[1]))


def _unit(v, name: str) -> tuple[float, float]:
    x, y = _pair(v, name)
    n = math.hypot(x, y)
    if n < 1e-12:
        raise ValueError(f"{name} must be a non-zero direction")
    return (x / n, y / n)


@dataclass
class EmitterSpec:
    """A flow-visualisation source in solver units (DESIGN.md 10.1).

    ``kind``: "point" (a nozzle at ``position``), "line" (a continuous slot from
    ``position`` to ``end``), "rake" (``count`` nozzles evenly spaced from
    ``position`` to ``end``), "rect" (an axis-aligned box of ``size`` = (w, h)
    centred at ``position``) or "disc" (a nozzle of radius ``size[0]``).
    ``radius`` is the nozzle radius (0 = 1.25 cells, the thinnest stream the
    limiter carries without smearing it into a haze).

    ``species="dye"`` relaxes smoke channel ``channel`` (0..3) to the
    concentration ``rate`` (0..1) inside the footprint; ``species="particles"``
    releases tracer particles from the nozzles, ``rate`` per unit time (L/U) for
    the whole emitter. ``pulse_period`` > 0 switches the emitter on for
    ``pulse_duty`` of each period (dye: puffs; particles: one batch per period,
    i.e. timelines). ``start`` / ``stop`` are in convective time units.
    """
    position: tuple[float, float]
    kind: str = "point"
    end: tuple[float, float] | None = None
    size: tuple[float, ...] = ()
    count: int = 8
    radius: float = 0.0
    species: str = "dye"
    channel: int = 0
    rate: float = 1.0
    pulse_period: float = 0.0
    pulse_duty: float = 0.5
    start: float = 0.0
    stop: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EMITTER_KINDS:
            raise ValueError(f"unknown emitter kind {self.kind!r}; use one of {', '.join(EMITTER_KINDS)}")
        if self.species not in EMITTER_SPECIES:
            raise ValueError(f"unknown emitter species {self.species!r}")
        self.position = _pair(self.position, "emitter position")
        if self.kind in ("line", "rake"):
            if self.end is None:
                raise ValueError(f"a {self.kind} emitter needs an end point")
            self.end = _pair(self.end, "emitter end")
        elif self.end is not None:
            self.end = _pair(self.end, "emitter end")
        self.size = tuple(float(s) for s in np.asarray(self.size, dtype=np.float64).reshape(-1))
        if self.kind == "rect" and (len(self.size) < 2 or min(self.size[:2]) <= 0):
            raise ValueError("a rect emitter needs a positive size (w, h)")
        if self.kind == "disc" and (len(self.size) < 1 or self.size[0] <= 0):
            raise ValueError("a disc emitter needs a positive radius in size[0]")
        self.count = int(self.count)
        if self.count < 1:
            raise ValueError("an emitter needs at least one nozzle")
        if not 0 <= int(self.channel) < MAX_DYE_CHANNELS:
            raise ValueError(f"dye channel must be 0..{MAX_DYE_CHANNELS - 1}")
        self.channel = int(self.channel)
        if self.radius < 0 or not math.isfinite(self.radius):
            raise ValueError("emitter radius must be >= 0")
        if not (self.rate >= 0 and math.isfinite(self.rate)):
            raise ValueError("emitter rate must be >= 0")
        if self.pulse_period < 0 or not 0 < self.pulse_duty <= 1:
            raise ValueError("pulse_period must be >= 0 and pulse_duty in (0, 1]")
        if self.stop is not None and self.stop <= self.start:
            raise ValueError("emitter stop must be after start")

    def nozzles(self) -> np.ndarray:
        """Nozzle centres (n, 2): the point, the rake nozzles, or samples of a line / box."""
        p = np.asarray(self.position)
        if self.kind in ("point", "disc"):
            return p[None, :]
        if self.kind in ("line", "rake"):
            e = np.asarray(self.end)
            if self.count == 1:
                return 0.5 * (p + e)[None, :]
            return p + np.linspace(0.0, 1.0, self.count)[:, None] * (e - p)
        # rect: a near-square grid of about ``count`` nozzles over the box
        w, h = self.size[0], self.size[1]
        nx = max(1, round(math.sqrt(self.count * w / h)))
        ny = max(1, math.ceil(self.count / nx))
        xs = p[0] + (np.arange(nx) + 0.5 - 0.5 * nx) * (w / nx)
        ys = p[1] + (np.arange(ny) + 0.5 - 0.5 * ny) * (h / ny)
        gx, gy = np.meshgrid(xs, ys)
        return np.c_[gx.ravel(), gy.ravel()]


@dataclass
class WallBC:
    """Thermal condition on one domain side: ``mode`` "adiabatic", "temperature"
    (``value`` = theta on the wall) or "heat_flux" (``value`` = G, the
    dimensionless gradient q'' L / (k dT_ref) into the fluid)."""
    side: str
    mode: str = "adiabatic"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.side not in WALL_SIDES:
            raise ValueError(f"unknown wall side {self.side!r}; use one of {', '.join(WALL_SIDES)}")
        if self.mode not in THERMAL_MODES:
            raise ValueError(f"unknown thermal mode {self.mode!r}")
        self.value = float(self.value)


@dataclass
class ThermalConfig:
    """Heat transfer: temperature transport with optional Boussinesq buoyancy.

    ``prandtl`` sets the diffusivity 1/Pe = 1/(Re Pr); ``richardson`` the buoyancy
    coefficient (0 = forced convection, temperature is then passive);
    ``gravity`` the unit direction of g (default -y). ``walls`` override the
    domain sides (adiabatic unless listed; in tunnel mode the inlet is held at
    ``theta_inlet`` and the outlet has zero gradient, so "left"/"right" entries
    only apply to closed domains). ``eta`` overrides the thermal Brinkman
    permeability (default: the momentum rule with Pe for Re, theory doc 16.2).
    """
    prandtl: float = 0.71
    richardson: float = 0.0
    gravity: tuple[float, float] = (0.0, -1.0)
    prandtl_turbulent: float = 0.85
    walls: list[WallBC] = field(default_factory=list)
    theta_inlet: float = 0.0
    theta_initial: float = 0.0
    eta: float | None = None

    def __post_init__(self) -> None:
        if not (self.prandtl > 0 and math.isfinite(self.prandtl)):
            raise ValueError("Prandtl number must be positive")
        if not math.isfinite(self.richardson):
            raise ValueError("Richardson number must be finite")
        if not self.prandtl_turbulent > 0:
            raise ValueError("turbulent Prandtl number must be positive")
        self.gravity = _unit(self.gravity, "gravity")
        self.walls = [w if isinstance(w, WallBC) else WallBC(**w) for w in self.walls]
        sides = [w.side for w in self.walls]
        if len(set(sides)) != len(sides):
            raise ValueError("each wall side may be listed once")
        if self.eta is not None and not self.eta > 0:
            raise ValueError("thermal eta must be positive")

    def wall(self, side: str) -> WallBC:
        """The condition on ``side`` (adiabatic when not listed)."""
        for w in self.walls:
            if w.side == side:
                return w
        return WallBC(side)


@dataclass
class HeatSourceSpec:
    """A volumetric heater: "rect" of ``size`` (w, h) or "disc" of radius ``size[0]``
    centred at ``position``. ``power`` is the non-dimensional heat rate released
    per unit span (negative cools); it is spread over a smooth footprint whose
    discrete integral is exactly ``power``. ``direction`` is the rect's first axis
    (its width ``size[0]`` lies along it): a heater placed in the body frame turns
    with the body at an angle of attack instead of staying aligned with x."""
    position: tuple[float, float]
    kind: str = "rect"
    size: tuple[float, ...] = (0.1, 0.1)
    power: float = 1.0
    name: str = ""
    direction: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if self.kind not in HEAT_SOURCE_KINDS:
            raise ValueError(f"unknown heat source kind {self.kind!r}")
        self.position = _pair(self.position, "heat source position")
        self.direction = _unit(self.direction, "heat source direction")
        self.size = tuple(float(s) for s in np.asarray(self.size, dtype=np.float64).reshape(-1))
        need = 2 if self.kind == "rect" else 1
        if len(self.size) < need or min(self.size[:need]) <= 0:
            raise ValueError("heat source size must be positive")
        if not math.isfinite(self.power):
            raise ValueError("heat source power must be finite")


@dataclass
class FanSpec:
    """An actuator-disc fan (a 2D actuator strip): a momentum source along
    ``direction`` over a strip of ``width`` (across) and ``thickness`` (along;
    None = width / 8, at least two cells) centred at ``position``.

    ``mode``: "thrust" (``value`` = thrust per unit span, rho U^2 L), "pressure_jump"
    (``value`` = dP / (rho U^2), thrust = dP * width) or "velocity" (``value`` = the
    target disc-averaged speed along ``direction``, units of U; a PI controller in
    the solver regulates the pressure jump, theory doc 18.2).
    """
    position: tuple[float, float]
    direction: tuple[float, float] = (1.0, 0.0)
    width: float = 1.0
    thickness: float | None = None
    mode: str = "thrust"
    value: float = 0.5
    name: str = ""

    def __post_init__(self) -> None:
        if self.mode not in FAN_MODES:
            raise ValueError(f"unknown fan mode {self.mode!r}; use one of {', '.join(FAN_MODES)}")
        self.position = _pair(self.position, "fan position")
        self.direction = _unit(self.direction, "fan direction")
        if not self.width > 0:
            raise ValueError("fan width must be positive")
        if self.thickness is not None and not self.thickness > 0:
            raise ValueError("fan thickness must be positive")
        if not math.isfinite(self.value):
            raise ValueError("fan value must be finite")


@dataclass
class PortSpec:
    """A jet (fluid created and blown out) or vent (fluid sucked in and removed).

    ``position`` is the centre of the open face, ``size`` its width, and the
    fluid crosses the face at ``speed`` along ``direction`` (for a jet out of
    the port, for a vent into it). The port body - a region ``depth`` deep
    (None = 3 cells) behind the face for a jet, beyond it for a vent - holds a
    mass source / sink of volume flow ``speed * size`` and is penalised to the
    matching velocity field (theory doc 19). A jet carries ``theta`` (None =
    ambient) and, with ``dye_channel``, releases smoke on that channel.
    """
    position: tuple[float, float]
    direction: tuple[float, float] = (1.0, 0.0)
    size: float = 0.2
    speed: float = 1.0
    kind: str = "jet"
    depth: float | None = None
    theta: float | None = None
    dye_channel: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PORT_KINDS:
            raise ValueError(f"unknown port kind {self.kind!r}")
        self.position = _pair(self.position, "port position")
        self.direction = _unit(self.direction, "port direction")
        if not self.size > 0 or not (self.speed >= 0 and math.isfinite(self.speed)):
            raise ValueError("port size must be positive and speed >= 0")
        if self.depth is not None and not self.depth > 0:
            raise ValueError("port depth must be positive")
        if self.dye_channel is not None and not 0 <= int(self.dye_channel) < MAX_DYE_CHANNELS:
            raise ValueError(f"port dye channel must be 0..{MAX_DYE_CHANNELS - 1}")


@dataclass
class ImmersedBody:
    """One solid outline in solver units.

    ``points`` is an (N, 2) polygon, counter-clockwise and implicitly closed
    (last point differs from the first). The body may spin rigidly about
    ``center`` with angular velocity ``omega`` (positive counter-clockwise,
    in units of U/L): that is how wheels on a moving road are modelled. The
    outline itself never moves, so the solid mask stays static.

    Sign for rolling wheels: the flow comes from the left and the road
    (``bc_bottom="moving"``) moves at ``+ground_speed`` in x, so a wheel of
    radius r that rolls without slip has ``omega = +ground_speed / r``
    (counter-clockwise: the contact point moves with the road, the top of the
    wheel against the flow).

    With heat transfer on (``SolverConfig.thermal``), ``thermal`` is the body's
    condition: "adiabatic" (insulated, the default), "temperature" (held at
    theta = ``thermal_value``) or "heat_flux" (releases the flux
    G = ``thermal_value`` = q'' L / (k dT_ref) through its surface).
    """
    points: np.ndarray
    name: str = ""
    role: str = "body"
    omega: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    thermal: str = "adiabatic"
    thermal_value: float = 0.0

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]                      # tolerate explicitly closed input
        if len(pts) < 3:
            raise ValueError("an immersed body needs at least 3 distinct points")
        self.points = pts
        self.center = (float(self.center[0]), float(self.center[1]))
        self.omega = float(self.omega)
        if self.thermal not in THERMAL_MODES:
            raise ValueError(f"unknown thermal mode {self.thermal!r}")
        self.thermal_value = float(self.thermal_value)

    @property
    def perimeter(self) -> float:
        """Length of the closed outline (solver units)."""
        pts = self.points
        if hasattr(pts, "detach"):              # a vertex tensor (sensitivity sweeps): read, never differentiate
            pts = pts.detach().double().cpu().numpy()
        q = np.vstack([pts, pts[:1]])
        return float(np.hypot(*np.diff(q, axis=0).T).sum())


@dataclass
class SolverConfig:
    """Grid, physics and numerics of one flow solve.

    The domain is ``[x0, x0 + nx h] x [y0, y0 + ny h]`` on a uniform staggered
    (MAC) grid. Flow enters at the left boundary along +x (unless ``periodic``),
    leaves through a convective outlet on the right, and the top / bottom are
    walls of the chosen type. ``bottom = "moving"`` is the rolling road of
    vehicle studies: no-slip with tangential speed ``ground_speed``.

    ``force_ref`` and ``moment_ref`` only affect how forces are normalised into
    coefficients: ``C = F / (0.5 force_ref)`` and ``Cm = M / (0.5 force_ref)``
    (the reference length is 1 in solver units), with Cm positive nose-up.

    ``domain="closed"`` replaces the inlet and outlet by walls (``bc_left`` /
    ``bc_right``): a cavity, a room or an enclosure, driven by buoyancy, fans,
    jets or a moving bottom wall. Placed physics - ``emitters``, ``thermal``,
    ``heat_sources``, ``fans``, ``ports`` - and the AeroJAX-style
    ``diagnostics`` and ``adaptive_dt="pid"`` are all off by default
    (theory doc sections 15-20). ``dye_inlet`` keeps the inlet smoke comb on
    channel 0 when ``dye`` is on (tunnel mode); planners switch it off when the
    user places emitters.

    ``wall_model="werner_wengle"`` adds the turbulent wall shear of the
    Werner-Wengle law on every no-slip domain wall where the first face lies
    outside the viscous sublayer (theory doc 18.4). It is opt-in: in a 2D closed
    room it makes the wall friction grid-independent but *raises* the room's
    speed, because 2D has no interior turbulent dissipation to share the fan's
    power with (measured in 18.4), so no planner switches it on by default.
    """
    nx: int
    ny: int
    h: float
    x0: float
    y0: float
    re: float
    bodies: list[ImmersedBody] = field(default_factory=list)
    inlet: str = "uniform"
    bc_top: str = "slip"
    bc_bottom: str = "slip"
    ground_speed: float = 1.0
    periodic: bool = False
    advection: str = "upwind3"
    turbulence: str = "none"
    cfl: float = 0.5
    dt: float | None = None
    force_ref: float = 1.0
    moment_ref: tuple[float, float] = (0.25, 0.0)
    dtype: str = "float32"
    device: str = "auto"
    use_cuda_graphs: bool = True
    dye: bool = True
    perturb: bool = True
    # --- advanced numerics (defaults validated in docs/theory_ns2d.md) ---------
    eta: float | None = None        # Brinkman permeability; None -> tied to the stage step
    mask_width: float = 1.0         # half-width of the smoothed solid interface, in cells
    les_constant: float | None = None   # model constant; None -> model default
    # --- domain and placed physics (DESIGN.md 10; all off by default) --------
    domain: str = "tunnel"          # "tunnel" | "closed" (walls on every side)
    bc_left: str = "noslip"         # closed domains: "noslip" | "slip"
    bc_right: str = "noslip"
    dye_inlet: bool = True          # inlet smoke comb on channel 0 (tunnel, dye on)
    emitters: list[EmitterSpec] = field(default_factory=list)
    thermal: ThermalConfig | None = None
    heat_sources: list[HeatSourceSpec] = field(default_factory=list)
    fans: list[FanSpec] = field(default_factory=list)
    ports: list[PortSpec] = field(default_factory=list)
    particle_capacity: int = 32768  # tracer-particle pool when particle emitters exist
    diagnostics: bool = False       # device-side diagnostics history (theory doc 20)
    adaptive_dt: str = "cfl"        # "cfl" | "pid" (smooth PID-controlled step)
    wall_model: str = "none"        # "none" | "werner_wengle" (domain-wall shear, theory doc 18.4)

    def __post_init__(self) -> None:
        self.nx, self.ny = int(self.nx), int(self.ny)
        if self.nx < 4 or self.ny < 4:
            raise ValueError("the grid needs at least 4 x 4 cells")
        if not (self.h > 0 and math.isfinite(self.h)):
            raise ValueError("cell size h must be positive")
        if not (self.re > 0 and math.isfinite(self.re)):
            raise ValueError("Reynolds number must be positive and finite")
        for name, value, allowed in (("inlet", self.inlet, INLETS), ("bc_top", self.bc_top, BC_TOP),
                                     ("bc_bottom", self.bc_bottom, BC_BOTTOM),
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
        self.bodies = [b if isinstance(b, ImmersedBody) else ImmersedBody(np.asarray(b))
                       for b in self.bodies]
        self.moment_ref = (float(self.moment_ref[0]), float(self.moment_ref[1]))
        for name, value, allowed in (("domain", self.domain, DOMAINS), ("bc_left", self.bc_left, BC_SIDE),
                                     ("bc_right", self.bc_right, BC_SIDE),
                                     ("adaptive_dt", self.adaptive_dt, ADAPTIVE_DT),
                                     ("wall_model", self.wall_model, WALL_MODELS)):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; use one of {', '.join(allowed)}")
        if self.periodic and self.domain == "closed":
            raise ValueError("a periodic domain cannot also be closed")
        self.emitters = [e if isinstance(e, EmitterSpec) else EmitterSpec(**e) for e in self.emitters]
        self.heat_sources = [q if isinstance(q, HeatSourceSpec) else HeatSourceSpec(**q) for q in self.heat_sources]
        self.fans = [f if isinstance(f, FanSpec) else FanSpec(**f) for f in self.fans]
        self.ports = [p if isinstance(p, PortSpec) else PortSpec(**p) for p in self.ports]
        if self.thermal is not None and not isinstance(self.thermal, ThermalConfig):
            self.thermal = ThermalConfig(**self.thermal)
        if self.particle_capacity < 0:
            raise ValueError("particle_capacity must be >= 0")

    @property
    def closed(self) -> bool:
        """True for a closed domain (walls on all four sides)."""
        return self.domain == "closed"

    @property
    def uses_wall_model(self) -> bool:
        """True when the Werner-Wengle wall shear acts on the domain walls (``wall_model``)."""
        return self.wall_model == "werner_wengle"

    @property
    def dye_channels(self) -> int:
        """Number of smoke channels the solver carries (0 = no dye at all)."""
        chans = [e.channel for e in self.emitters if e.species == "dye"]
        chans += [int(p.dye_channel) for p in self.ports if p.dye_channel is not None and p.kind == "jet"]
        if chans:
            return max(chans) + 1
        return 1 if self.dye else 0

    @property
    def has_physics(self) -> bool:
        """True when any placed physics (section 10) is configured."""
        return bool(self.emitters or self.thermal is not None or self.heat_sources or self.fans or self.ports)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the domain in solver units."""
        return (self.x0, self.x0 + self.nx * self.h, self.y0, self.y0 + self.ny * self.h)

    @property
    def nu(self) -> float:
        """Non-dimensional kinematic viscosity, 1 / Re."""
        return 1.0 / self.re
