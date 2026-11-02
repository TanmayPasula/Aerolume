"""Placed physics of the 2D solver: heat transfer, buoyancy, fans, ports, smoke emitters, paint.

DESIGN.md section 10 asks for AeroJAX-style placed physics on top of the
validated flow solver; ``docs/theory_ns2d.md`` sections 15-20 derive and verify
each piece. This module turns the solver-unit specs of :mod:`.config` into
device tensors (:class:`Physics`, stored as ``ops.phys``) and provides the
pieces of the time step that use them. The rules that keep the validated solver
intact:

* **Off means absent.** ``ops.phys`` is None unless a config asks for placed
  physics, and every branch in the step tests a static Python flag, so the
  default path runs exactly the kernels it ran before (CUDA graphs included).
* **Static structure, live values.** Which features exist (thermal on, number of
  fans, smoke channels, emitter timing groups, whether ports or paint exist) is
  static and baked into a captured CUDA graph. Every *value* - footprints,
  positions, thrusts, wall temperatures, Ri, Pe - lives in a device tensor that
  :class:`~aerolume.solvers.ns2d.FlowSolver` rewrites in place for live edits,
  so moving a heater or dragging an emitter never re-captures a graph. Only a
  change of structure does (lazily, on the next ``advance``).
* **Exact totals.** Every distributed source (heater power, fan thrust, port
  volume flow, a heat-flux body's heat) is normalised by the discrete integral
  of its footprint, so the released total is exactly the requested one on any
  grid, and smooth footprints (the C1 step of the body masks, one cell wide)
  move continuously when an object moves by a fraction of a cell.
* **No host synchronisation** inside a step (controllers, pulses and emitter
  clocks are device arithmetic), so everything is CUDA-graph capturable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .config import EmitterSpec, FanSpec, HeatSourceSpec, PortSpec, SolverConfig, ThermalConfig
from .geometry import smooth_step
from .scalar import DYE_FLOOR, diffusive_fluxes, face_average_x, face_average_y, flux_divergence, tvd_fluxes

#: default nozzle radius of a smoke emitter, in cells (as in 3D, ns3d/dye.py)
NOZZLE_RADIUS_CELLS = 1.25
#: default port depth (the region behind the open face that holds the source), in cells
PORT_DEPTH_CELLS = 3.0
#: a fan strip is at least this many cells thick (a thinner strip aliases the force)
FAN_MIN_THICKNESS_CELLS = 2.0
#: velocity-mode fan PI controller: closed-loop time = FAN_LOOP_TIME * width / |target|
FAN_LOOP_TIME = 2.0
#: velocity-mode fans: the pressure jump is bounded by this multiple of max(target^2, 1)
FAN_DP_LIMIT = 50.0
#: stop time used for "never stops"
NEVER = 1e30


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------

@dataclass
class DyeGroup:
    """Smoke emitters of one channel that share their timing: one weight / target map.

    ``weight`` (ny, nx) in [0, 1] is the union of the footprints, ``target`` the
    released concentration there, ``timing`` (4,) = [start, stop, period, duty].
    """
    channel: int
    weight: torch.Tensor
    target: torch.Tensor
    timing: torch.Tensor


@dataclass
class Physics:
    """Device-side description of the placed physics (see the module docstring).

    Static fields (Python values) define the structure; tensor fields hold the
    values. :meth:`signature` summarises the structure: two Physics with equal
    signatures can be copied into each other with :meth:`copy_from`.
    """
    # temperature
    thermal: bool = False
    buoyant: bool = False                      # Ri != 0 (static: switching it on re-captures graphs)
    prandtl: float = 0.71
    prt: float = 0.85
    gravity: tuple[float, float] = (0.0, -1.0)
    wall_modes: tuple[str, str, str, str] = ("adiabatic",) * 4
    kappa: torch.Tensor | None = None          # 0-d, 1 / (Re Pr)
    ri: torch.Tensor | None = None             # 0-d Richardson number
    eta_t: torch.Tensor | None = None          # 0-d thermal permeability
    eta_t_fixed: float | None = None
    theta_in: torch.Tensor | None = None       # 0-d inlet temperature
    theta_init: float = 0.0
    wall_values: torch.Tensor | None = None    # (4,) left, right, bottom, top
    diff_coef: float = 4.0                     # largest diffusion coefficient / (kappa / h^2), static
    heat: torch.Tensor | None = None           # (ny, nx) volumetric heat-rate density
    heat_max: torch.Tensor | None = None       # 0-d max |heat| (theta per unit time), for the step bound
    # ports
    ports: bool = False
    pchi_u: torch.Tensor | None = None
    pa_u: torch.Tensor | None = None
    pchi_v: torch.Tensor | None = None
    pa_v: torch.Tensor | None = None
    src: torch.Tensor | None = None            # (ny, nx) mass source used by the projection
    src_neg: torch.Tensor | None = None        # (ny, nx) vent sinks + uniform leak (scalars leave with the fluid)
    q_net: torch.Tensor | None = None          # 0-d net volume flow of all ports
    pt_chi: torch.Tensor | None = None         # jets: temperature penalisation
    pt_a: torch.Tensor | None = None
    port_dye: dict[int, torch.Tensor] = field(default_factory=dict)
    # paint (mouse-drawn obstacles)
    paint: bool = False
    paint_c: torch.Tensor | None = None
    paint_u: torch.Tensor | None = None
    paint_v: torch.Tensor | None = None
    paint_wu: torch.Tensor | None = None       # paint's share of the combined penalisation (force weights)
    paint_wv: torch.Tensor | None = None
    # combined full-field velocity penalisation (ports + paint)
    xchi_u: torch.Tensor | None = None
    xa_u: torch.Tensor | None = None
    xchi_v: torch.Tensor | None = None
    xa_v: torch.Tensor | None = None
    # fans
    nfan: int = 0
    fan_modes: tuple[str, ...] = ()
    fan_gu: torch.Tensor | None = None         # (nf, ny, nx+1) force per unit thrust
    fan_gv: torch.Tensor | None = None         # (nf, ny+1, nx)
    fan_phi: torch.Tensor | None = None        # (nf, ny, nx) normalised averaging weight
    fan_dir: torch.Tensor | None = None        # (nf, 2)
    fan_value: torch.Tensor | None = None      # (nf,) thrust / pressure jump / target speed
    fan_width: torch.Tensor | None = None      # (nf,)
    fan_kp: torch.Tensor | None = None
    fan_ki: torch.Tensor | None = None
    fan_dpmax: torch.Tensor | None = None
    fan_isvel: torch.Tensor | None = None      # (nf,) 1.0 for velocity-regulated fans
    fan_isthrust: torch.Tensor | None = None   # (nf,) 1.0 for fixed-thrust fans
    # smoke
    n_dye: int = 0
    comb: bool = False
    dye_inflow: torch.Tensor | None = None     # (n_dye, ny, 1) inlet values
    groups: list[DyeGroup] = field(default_factory=list)
    # controllers and time step
    pid: bool = False
    ctrl_size: int = 0
    u_floor: torch.Tensor | None = None        # 0-d speed scale of the sources (CFL)
    specs: dict = field(default_factory=dict)  # the specs this was built from (reports)

    FAN_CTRL_OFFSET = 2                        # ctrl[0:2] = PID memory, then (I, dp) per fan

    def signature(self) -> tuple:
        return (self.thermal, self.buoyant, self.wall_modes, self.gravity, self.heat is not None, self.ports,
                tuple(sorted(self.port_dye)), self.pt_chi is not None, self.paint, self.nfan, self.fan_modes,
                self.n_dye, self.comb, tuple(g.channel for g in self.groups), self.pid, self.ctrl_size)

    def tensors(self) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                out[k] = v
        for c, t in self.port_dye.items():
            out[f"port_dye{c}"] = t
        for i, g in enumerate(self.groups):
            out[f"g{i}_w"], out[f"g{i}_r"], out[f"g{i}_t"] = g.weight, g.target, g.timing
        return out

    def copy_from(self, other: Physics) -> None:
        """Copy every tensor of ``other`` (same signature) into this one, in place."""
        mine, theirs = self.tensors(), other.tensors()
        for k, t in mine.items():
            if k in theirs and theirs[k].shape == t.shape:
                t.copy_(theirs[k])
        self.prandtl, self.prt, self.specs = other.prandtl, other.prt, other.specs
        self.eta_t_fixed, self.theta_init = other.eta_t_fixed, other.theta_init


# ---------------------------------------------------------------------------
# grids and footprints
# ---------------------------------------------------------------------------

def _mesh(ops, where: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) of the cell centres ("c"), x-faces ("u") or y-faces ("v")."""
    d, t, h = ops.device, ops.dtype, ops.h
    nx, ny = ops.nx, ops.ny
    if where == "c":
        xs = ops.x0 + (torch.arange(nx, device=d, dtype=t) + 0.5) * h
        ys = ops.y0 + (torch.arange(ny, device=d, dtype=t) + 0.5) * h
    elif where == "u":
        xs = ops.x0 + torch.arange(nx + 1, device=d, dtype=t) * h
        ys = ops.y0 + (torch.arange(ny, device=d, dtype=t) + 0.5) * h
    else:
        xs = ops.x0 + (torch.arange(nx, device=d, dtype=t) + 0.5) * h
        ys = ops.y0 + torch.arange(ny + 1, device=d, dtype=t) * h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def _zero_boundary(f: torch.Tensor, where: str, ops) -> torch.Tensor:
    """Boundary faces carry boundary conditions, never sources or penalisation."""
    if ops.periodic or where == "c":
        return f
    f = f.clone()
    if where == "u":
        f[:, 0] = 0.0
        f[:, -1] = 0.0
    else:
        f[0] = 0.0
        f[-1] = 0.0
    return f


def box_profile(x, y, center, direction, half_along: float, half_across: float, w: float) -> torch.Tensor:
    """Smooth indicator of an oriented box: product of two C1 steps (1 inside, 0 outside)."""
    nx, ny = direction
    dx, dy = x - center[0], y - center[1]
    xi = dx * nx + dy * ny
    zeta = -dx * ny + dy * nx
    return smooth_step(xi.abs() - half_along, w) * smooth_step(zeta.abs() - half_across, w)


def disc_profile(x, y, center, radius: float, w: float) -> torch.Tensor:
    return smooth_step(torch.hypot(x - center[0], y - center[1]) - radius, w)


def capsule_profile(x, y, a, b, radius: float, w: float) -> torch.Tensor:
    """Smooth indicator of the points within ``radius`` of the segment a-b."""
    ax, ay = a
    ex, ey = b[0] - ax, b[1] - ay
    L2 = max(ex * ex + ey * ey, 1e-30)
    t = (((x - ax) * ex + (y - ay) * ey) / L2).clamp(0.0, 1.0)
    return smooth_step(torch.hypot(x - ax - t * ex, y - ay - t * ey) - radius, w)


def emitter_footprint(spec: EmitterSpec, x, y, h: float) -> torch.Tensor:
    """Smoke-release weight (0..1) of one emitter at points (x, y)."""
    r = spec.radius if spec.radius > 0 else NOZZLE_RADIUS_CELLS * h
    w = 0.5 * h                                   # anti-aliased edge: half a cell each way
    if spec.kind == "point":
        return disc_profile(x, y, spec.position, r, w)
    if spec.kind == "disc":
        return disc_profile(x, y, spec.position, spec.size[0], w)
    if spec.kind == "line":
        return capsule_profile(x, y, spec.position, spec.end, r, w)
    if spec.kind == "rake":
        out = None
        for p in spec.nozzles():
            f = disc_profile(x, y, (float(p[0]), float(p[1])), r, w)
            out = f if out is None else torch.maximum(out, f)
        return out
    return box_profile(x, y, spec.position, (1.0, 0.0), 0.5 * spec.size[0], 0.5 * spec.size[1], w)


def _heat_footprint(spec: HeatSourceSpec, x, y, h: float) -> torch.Tensor:
    if spec.kind == "disc":
        return disc_profile(x, y, spec.position, spec.size[0], h)
    return box_profile(x, y, spec.position, spec.direction, 0.5 * spec.size[0], 0.5 * spec.size[1], h)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def needs_physics(cfg: SolverConfig) -> bool:
    """True when the config needs an ``ops.phys`` (placed physics or the PID step)."""
    return cfg.has_physics or cfg.adaptive_dt == "pid"


def thermal_wall_modes(cfg: SolverConfig) -> tuple[tuple[str, str, str, str], list[float]]:
    """Boundary modes and values (left, right, bottom, top) of the temperature field."""
    th = cfg.thermal
    if cfg.periodic:
        return ("periodic",) * 4, [0.0] * 4
    modes, vals = [], []
    for side in ("left", "right", "bottom", "top"):
        if not cfg.closed and side == "left":
            modes.append("inlet")
            vals.append(th.theta_inlet)
        elif not cfg.closed and side == "right":
            modes.append("outlet")
            vals.append(0.0)
        else:
            w = th.wall(side)
            modes.append(w.mode)
            vals.append(w.value)
    return tuple(modes), vals


def thermal_eta(h: float, pe: float) -> float:
    """Default thermal Brinkman permeability: the momentum rule with Pe for Re
    (theory doc 16.2: the penalised conduction problem is the penalised Stokes
    problem the wall offset was calibrated on)."""
    from .core import ETA_PER_H, STIFFNESS_MIN
    return min(ETA_PER_H * h, h * h * pe / STIFFNESS_MIN)


def build_physics(cfg: SolverConfig, ops) -> Physics | None:
    """The :class:`Physics` of a config on the grid of ``ops`` (None when nothing is placed)."""
    if not needs_physics(cfg):
        return None
    dev, dt = ops.device, ops.dtype

    def scalar(v):
        return torch.tensor(float(v), device=dev, dtype=dt)

    ph = Physics()
    ph.specs = {"emitters": list(cfg.emitters), "heat_sources": list(cfg.heat_sources),
                "fans": list(cfg.fans), "ports": list(cfg.ports)}
    u_floor = 0.0
    th: ThermalConfig | None = cfg.thermal
    if th is not None:
        ph.thermal = True
        ph.buoyant = th.richardson != 0.0
        ph.prandtl, ph.prt, ph.gravity = float(th.prandtl), float(th.prandtl_turbulent), th.gravity
        pe = cfg.re * th.prandtl
        ph.kappa = scalar(1.0 / pe)
        ph.ri = scalar(th.richardson)
        ph.eta_t_fixed = th.eta
        ph.eta_t = scalar(th.eta if th.eta is not None else thermal_eta(cfg.h, pe))
        ph.theta_in = scalar(th.theta_inlet)
        ph.theta_init = float(th.theta_initial)
        modes, vals = thermal_wall_modes(cfg)
        ph.wall_modes = modes
        ph.diff_coef = diffusion_coefficient(modes)
        ph.wall_values = torch.tensor(vals, device=dev, dtype=dt)
        if cfg.heat_sources:
            ph.heat = heat_source_field(cfg.heat_sources, ops)
            ph.heat_max = ph.heat.abs().amax()
    _build_ports(cfg, ops, ph)
    if ph.ports:
        u_floor = max(u_floor, max(p.speed for p in cfg.ports))
    _build_fans(cfg, ops, ph)
    for f in cfg.fans:
        if f.mode == "velocity":
            u_floor = max(u_floor, abs(f.value))
        else:
            dp = f.value / f.width if f.mode == "thrust" else f.value
            u_floor = max(u_floor, math.sqrt(abs(dp)))
    _build_dye(cfg, ops, ph)
    ph.pid = cfg.adaptive_dt == "pid"
    if ph.pid or ph.nfan:
        ph.ctrl_size = Physics.FAN_CTRL_OFFSET + 2 * ph.nfan
    ph.u_floor = scalar(u_floor)
    combine_penalisation(ph)
    return ph


def empty_physics(ops) -> Physics:
    """A Physics with nothing placed (the start for paint on a plain run)."""
    ph = Physics()
    ph.u_floor = torch.zeros((), device=ops.device, dtype=ops.dtype)
    return ph


def heat_source_field(specs: list[HeatSourceSpec], ops) -> torch.Tensor:
    """Volumetric heat-rate density (ny, nx); each source integrates exactly to its power."""
    x, y = _mesh(ops, "c")
    q = torch.zeros_like(x)
    for s in specs:
        phi = _heat_footprint(s, x, y, ops.h)
        tot = float(phi.sum()) * ops.h * ops.h
        if tot <= 0:
            raise ValueError(f"heat source {s.name or s.position} lies outside the domain")
        q = q + phi * (s.power / tot)
    return q


def _port_fields(p: PortSpec, ops):
    """(chi, target) on u-faces and v-faces, mass source and theta/dye weight at centres."""
    h = ops.h
    depth = p.depth if p.depth is not None else PORT_DEPTH_CELLS * h
    depth = max(depth, 2.0 * h)
    nx, ny = p.direction
    jet = p.kind == "jet"
    # the port body lies behind the open face (jet) or beyond it (vent), along the direction
    xi_c = -0.5 * depth if jet else 0.5 * depth
    cx, cy = p.position[0] + xi_c * nx, p.position[1] + xi_c * ny
    w = 0.5 * h
    out = {}
    for where in ("u", "v", "c"):
        x, y = _mesh(ops, where)
        phi = _zero_boundary(box_profile(x, y, (cx, cy), (nx, ny), 0.5 * depth, 0.5 * p.size, w), where, ops)
        xi = (x - p.position[0]) * nx + (y - p.position[1]) * ny
        # velocity ramps from 0 at the closed back of the port to `speed` at the open face
        ramp = ((xi + depth) / depth).clamp(0.0, 1.0) if jet else (1.0 - xi / depth).clamp(0.0, 1.0)
        out[where] = (phi, ramp)
    phi_c = out["c"][0]
    tot = float(phi_c.sum()) * h * h
    if tot <= 0:
        raise ValueError(f"port {p.name or p.position} lies outside the domain")
    q = p.speed * p.size * (1.0 if jet else -1.0)            # volume flow per unit span
    src = phi_c * (q / tot)
    phi_u, ramp_u = out["u"]
    phi_v, ramp_v = out["v"]
    return {"chi_u": phi_u, "a_u": phi_u * (p.speed * nx) * ramp_u, "chi_v": phi_v,
            "a_v": phi_v * (p.speed * ny) * ramp_v, "src": src, "phi_c": phi_c, "q": q}


def _build_ports(cfg: SolverConfig, ops, ph: Physics) -> None:
    if not cfg.ports:
        return
    ph.ports = True
    f = [_port_fields(p, ops) for p in cfg.ports]
    ph.pchi_u = torch.stack([d["chi_u"] for d in f]).sum(0).clamp(max=1.0)
    ph.pchi_v = torch.stack([d["chi_v"] for d in f]).sum(0).clamp(max=1.0)
    ph.pa_u = torch.stack([d["a_u"] for d in f]).sum(0)
    ph.pa_v = torch.stack([d["a_v"] for d in f]).sum(0)
    src = torch.stack([d["src"] for d in f]).sum(0)
    q_net = float(src.sum()) * ops.h * ops.h
    ph.q_net = torch.tensor(q_net, device=ops.device, dtype=ops.dtype)
    ph.src_neg = src.clamp(max=0.0)
    if cfg.closed or cfg.periodic:
        # the all-Neumann (or periodic) pressure problem needs zero net source: an
        # unbalanced set of ports is compensated by a uniform sink over the domain
        area = ops.nx * ops.ny * ops.h * ops.h
        if q_net > 1e-12:                         # (a balanced set nets to round-off)
            # ... and the fluid that sink removes takes its heat and smoke along (theta_rhs,
            # update_dye and the reported vent heat read src_neg). Without it the conservative
            # update kept them -- a source +theta q/A everywhere -- and a closed box fed by a
            # theta = 1 jet with no vent climbed to theta 2.77 while its heat balance "closed"
            # (critic round 5). A net sink (q_net < 0) needs nothing: the fluid created evenly
            # arrives at 0 and only dilutes, and the vents keep their own exact sinks.
            ph.src_neg = ph.src_neg - q_net / area
        src = src - q_net / area
    ph.src = src
    if cfg.thermal is not None:
        jets = [(p, d) for p, d in zip(cfg.ports, f) if p.kind == "jet"]
        if jets:
            ph.pt_chi = torch.stack([d["phi_c"] for _, d in jets]).sum(0).clamp(max=1.0)
            amb = cfg.thermal.theta_inlet
            ph.pt_a = torch.stack([d["phi_c"] * (p.theta if p.theta is not None else amb) for p, d in jets]).sum(0)
    dye: dict[int, torch.Tensor] = {}
    for p, d in zip(cfg.ports, f):
        if p.kind == "jet" and p.dye_channel is not None:
            c = int(p.dye_channel)
            dye[c] = d["phi_c"] if c not in dye else torch.maximum(dye[c], d["phi_c"])
    ph.port_dye = dye


def _fan_fields(fan: FanSpec, ops):
    h = ops.h
    th = fan.thickness if fan.thickness is not None else fan.width / 8.0
    th = max(th, FAN_MIN_THICKNESS_CELLS * h)
    nx, ny = fan.direction
    out = {}
    for where in ("u", "v", "c"):
        x, y = _mesh(ops, where)
        out[where] = _zero_boundary(box_profile(x, y, fan.position, fan.direction, 0.5 * th, 0.5 * fan.width, h),
                                    where, ops)
    su, sv, sc = (float(out[k].sum()) for k in ("u", "v", "c"))
    if min(su, sv, sc) <= 0:
        raise ValueError(f"fan {fan.name or fan.position} lies outside the domain")
    a = h * h
    return (out["u"] * (nx / (su * a)), out["v"] * (ny / (sv * a)), out["c"] / sc)


def fan_gains(fan: FanSpec, cfg: SolverConfig) -> tuple[float, float, float]:
    """PI gains (kp, ki) and pressure-jump bound of a velocity-regulated fan (theory doc 18.2).

    The plant is the disc-averaged speed u_d as a function of the pressure jump dp.
    Its quasi-steady slope follows from actuator-disc momentum theory: with the
    free-stream component U along the fan axis, dp = T / D = 2 u_d (u_d - U), so
    d(dp)/d(u_d) = 2 |2 u_d - U| (in a closed room U = 0). With ``kp`` equal to that
    slope (loop gain 1) and ``ki = 2 omega kp`` the error of the integral loop decays
    at the rate omega = |target| / (FAN_LOOP_TIME D), i.e. within a few convective
    times of the disc. (Gains sized for the fan's own fluid mass alone made the loop
    four times too slow: measured, 2 % short of the target after 25 time units.)
    """
    speed = max(abs(fan.value), 0.05) if fan.mode == "velocity" else 1.0
    u_inf = 0.0 if (cfg.closed or cfg.periodic) else float(fan.direction[0])     # inflow is +x at speed 1
    target = fan.value if fan.mode == "velocity" else 1.0
    slope = 2.0 * max(abs(2.0 * target - u_inf), speed, 0.1)
    om = speed / (FAN_LOOP_TIME * fan.width)
    return slope, 2.0 * om * slope, FAN_DP_LIMIT * max(speed * speed, 1.0)


def _build_fans(cfg: SolverConfig, ops, ph: Physics) -> None:
    if not cfg.fans:
        return
    dev, dt = ops.device, ops.dtype
    ph.nfan = len(cfg.fans)
    ph.fan_modes = tuple(f.mode for f in cfg.fans)
    fields = [_fan_fields(f, ops) for f in cfg.fans]
    ph.fan_gu = torch.stack([g[0] for g in fields])
    ph.fan_gv = torch.stack([g[1] for g in fields])
    ph.fan_phi = torch.stack([g[2] for g in fields])
    ph.fan_dir = torch.tensor([f.direction for f in cfg.fans], device=dev, dtype=dt)
    ph.fan_value = torch.tensor([f.value for f in cfg.fans], device=dev, dtype=dt)
    ph.fan_width = torch.tensor([f.width for f in cfg.fans], device=dev, dtype=dt)
    kp, ki, dpmax = [], [], []
    for f in cfg.fans:
        kp_f, ki_f, dpmax_f = fan_gains(f, cfg)
        kp.append(kp_f)
        ki.append(ki_f)
        dpmax.append(dpmax_f)
    ph.fan_kp = torch.tensor(kp, device=dev, dtype=dt)
    ph.fan_ki = torch.tensor(ki, device=dev, dtype=dt)
    ph.fan_dpmax = torch.tensor(dpmax, device=dev, dtype=dt)
    ph.fan_isvel = torch.tensor([1.0 if f.mode == "velocity" else 0.0 for f in cfg.fans], device=dev, dtype=dt)
    ph.fan_isthrust = torch.tensor([1.0 if f.mode == "thrust" else 0.0 for f in cfg.fans], device=dev, dtype=dt)


def dye_comb_column(ny: int):
    from .core import dye_comb
    return dye_comb(ny)


def _build_dye(cfg: SolverConfig, ops, ph: Physics) -> None:
    n = cfg.dye_channels
    if n == 0:
        return
    dye_emitters = [e for e in cfg.emitters if e.species == "dye"]
    if not (dye_emitters or ph.port_dye or n > 1):
        # only the inlet comb on channel 0: the legacy path (State.dye, core.advect_dye) handles it
        return
    ph.n_dye = n
    ph.comb = bool(cfg.dye and cfg.dye_inlet and not cfg.closed and not cfg.periodic)
    inflow = torch.zeros(n, ops.ny, 1, device=ops.device, dtype=ops.dtype)
    if ph.comb:
        inflow[0, :, 0] = torch.as_tensor(dye_comb_column(ops.ny), device=ops.device, dtype=ops.dtype)
    ph.dye_inflow = inflow
    ph.groups = build_dye_groups(dye_emitters, ops)


def build_dye_groups(emitters: list[EmitterSpec], ops) -> list[DyeGroup]:
    """Group smoke emitters by (channel, timing) and paint their footprints."""
    x, y = _mesh(ops, "c")
    groups: dict[tuple, list[EmitterSpec]] = {}
    for e in emitters:
        key = (e.channel, e.start, e.stop if e.stop is not None else NEVER, e.pulse_period, e.pulse_duty)
        groups.setdefault(key, []).append(e)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        wsum = torch.zeros_like(x)
        rsum = torch.zeros_like(x)
        wmax = torch.zeros_like(x)
        for e in members:
            f = emitter_footprint(e, x, y, ops.h)
            wsum = wsum + f
            rsum = rsum + f * e.rate
            wmax = torch.maximum(wmax, f)
        target = torch.where(wsum > 0, rsum / wsum.clamp_min(1e-30), torch.zeros_like(wsum))
        timing = torch.tensor(key[1:], device=ops.device, dtype=ops.dtype)
        out.append(DyeGroup(int(key[0]), wmax, target, timing))
    return out


def ensure_paint(ph: Physics, ops) -> None:
    """Allocate the paint layer (mouse-drawn obstacles) on first use."""
    if ph.paint:
        return
    ph.paint = True
    ph.paint_c = torch.zeros(ops.ny, ops.nx, device=ops.device, dtype=ops.dtype)
    ph.paint_u = torch.zeros(ops.ny, ops.nx + 1, device=ops.device, dtype=ops.dtype)
    ph.paint_v = torch.zeros(ops.ny + 1, ops.nx, device=ops.device, dtype=ops.dtype)


def paint_faces(ph: Physics, ops) -> None:
    """Face fractions of the paint layer averaged from its centres (in place): only used
    to restore an old state that saved the centres alone; brushes write every grid."""
    c = ph.paint_c
    ph.paint_u.copy_(_zero_boundary(face_average_x(c, ops), "u", ops))
    ph.paint_v.copy_(_zero_boundary(face_average_y(c, ops), "v", ops))


def paint_brush(ph: Physics, ops, sdf, mask_width: float, erase: bool) -> None:
    """Add (or erase) the solid ``sdf(x, y) < 0`` to the paint layer, in place.

    The solid fraction is evaluated *at each grid's own points* (centres, u-faces,
    v-faces) with the bodies' calibrated interface (C1 step of half-width
    ``mask_width`` cells centred MASK_SHIFT cells inside the outline), so a painted
    obstacle has its effective no-slip wall on the outline exactly like an
    :class:`~.config.ImmersedBody` of the same shape (theory doc 6.3, 19.3).
    """
    from .core import MASK_SHIFT
    h = ops.h
    w = mask_width * h
    for where, dst in (("c", ph.paint_c), ("u", ph.paint_u), ("v", ph.paint_v)):
        x, y = _mesh(ops, where)
        chi = _zero_boundary(smooth_step(sdf(x, y) + MASK_SHIFT * h, w), where, ops)
        dst.copy_(torch.minimum(dst, 1.0 - chi) if erase else torch.maximum(dst, chi))


def combine_penalisation(ph: Physics) -> None:
    """(Re)build the combined full-field velocity penalisation of ports and paint.

    Painted solid wins where both overlap (target velocity zero). In place when
    the combined tensors already exist (live edits keep CUDA graphs valid).
    """
    if not (ph.ports or ph.paint):
        return
    if ph.ports and ph.paint:
        cu = (ph.pchi_u + ph.paint_u).clamp(max=1.0)
        au = ph.pa_u * (1.0 - ph.paint_u)
        cv = (ph.pchi_v + ph.paint_v).clamp(max=1.0)
        av = ph.pa_v * (1.0 - ph.paint_v)
    elif ph.ports:
        cu, au, cv, av = ph.pchi_u, ph.pa_u, ph.pchi_v, ph.pa_v
    else:
        cu, au = ph.paint_u, torch.zeros_like(ph.paint_u)
        cv, av = ph.paint_v, torch.zeros_like(ph.paint_v)
    if ph.xchi_u is None or ph.xchi_u.shape != cu.shape:
        ph.xchi_u, ph.xa_u, ph.xchi_v, ph.xa_v = cu.clone(), au.clone(), cv.clone(), av.clone()
    else:
        ph.xchi_u.copy_(cu)
        ph.xa_u.copy_(au)
        ph.xchi_v.copy_(cv)
        ph.xa_v.copy_(av)
    if ph.paint:
        # force weights: the paint's share of the penalisation at each face (1 on its own
        # support), as the bodies' per-body shares - not the solid fraction itself, which
        # would discount the interface where most of the momentum is removed
        if ph.ports:
            wu = ph.paint_u / (ph.paint_u + ph.pchi_u).clamp_min(1e-30)
            wv = ph.paint_v / (ph.paint_v + ph.pchi_v).clamp_min(1e-30)
        else:
            wu, wv = (ph.paint_u > 0).to(ph.paint_u.dtype), (ph.paint_v > 0).to(ph.paint_v.dtype)
        if ph.paint_wu is None or ph.paint_wu.shape != wu.shape:
            ph.paint_wu, ph.paint_wv = wu.clone(), wv.clone()
        else:
            ph.paint_wu.copy_(wu)
            ph.paint_wv.copy_(wv)


# ---------------------------------------------------------------------------
# pieces of the time step
# ---------------------------------------------------------------------------

def buoyancy(theta: torch.Tensor, ph: Physics, ops) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Boussinesq body force ``-Ri theta g_hat`` on the u- and v-faces (zero on boundary faces)."""
    gx, gy = ph.gravity
    fu = fv = None
    if gy != 0.0:
        tv = face_average_y(theta, ops)
        if not ops.periodic:
            tv = torch.cat([torch.zeros_like(tv[:1]), tv[1:-1], torch.zeros_like(tv[:1])], dim=0)
        fv = tv * (ph.ri * (-gy))
    if gx != 0.0:
        tu = face_average_x(theta, ops)
        if not ops.periodic:
            tu = torch.cat([torch.zeros_like(tu[:, :1]), tu[:, 1:-1], torch.zeros_like(tu[:, :1])], dim=1)
        fu = tu * (ph.ri * (-gx))
    return fu, fv


def fan_disc_velocity(u: torch.Tensor, v: torch.Tensor, ph: Physics) -> torch.Tensor:
    """Footprint-averaged velocity along each fan's direction, (nf,)."""
    uc = 0.5 * (u[:, 1:] + u[:, :-1])
    vc = 0.5 * (v[1:] + v[:-1])
    ua = torch.einsum("kij,ij->k", ph.fan_phi, uc)
    va = torch.einsum("kij,ij->k", ph.fan_phi, vc)
    return ua * ph.fan_dir[:, 0] + va * ph.fan_dir[:, 1]


def fan_control(u, v, ctrl: torch.Tensor, dt, ph: Physics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thrust of every fan for this step, the new controller state and the disc velocities.

    "thrust" and "pressure_jump" fans are fixed; "velocity" fans run a PI
    controller on the footprint-averaged speed (anti-windup by clamping the
    integral so the pressure jump stays within its bound). Device arithmetic only.
    """
    ud = fan_disc_velocity(u, v, ph)
    o = Physics.FAN_CTRL_OFFSET
    integ = ctrl[o::2]
    err = ph.fan_value - ud
    ilim = ph.fan_dpmax / ph.fan_ki
    integ_new = torch.maximum(torch.minimum(integ + err * dt * ph.fan_isvel, ilim), -ilim)
    dp_pi = torch.maximum(torch.minimum(ph.fan_kp * err + ph.fan_ki * integ_new, ph.fan_dpmax), -ph.fan_dpmax)
    thrust_fixed = torch.where(ph.fan_isthrust > 0, ph.fan_value, ph.fan_value * ph.fan_width)
    thrust = torch.where(ph.fan_isvel > 0, dp_pi * ph.fan_width, thrust_fixed)
    dp = thrust / ph.fan_width
    new = torch.stack([integ_new, dp], dim=1).reshape(-1)
    ctrl = torch.cat([ctrl[:o], new])
    return thrust, ctrl, ud


def fan_forces(thrust: torch.Tensor, ph: Physics) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.einsum("k,kij->ij", thrust, ph.fan_gu), torch.einsum("k,kij->ij", thrust, ph.fan_gv)


def penalise_extra(uh: torch.Tensor, vh: torch.Tensor, ph: Physics, k) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Implicit penalisation of the full-field regions (ports, paint); returns the
    paint impulse (2,) of this stage when paint exists (for its drag)."""
    un = (uh + k * ph.xa_u) / (1.0 + k * ph.xchi_u)
    vn = (vh + k * ph.xa_v) / (1.0 + k * ph.xchi_v)
    imp = None
    if ph.paint:
        imp = torch.stack([((un - uh) * ph.paint_wu).sum(), ((vn - vh) * ph.paint_wv).sum()])
    return un, vn, imp


def _window_apply(field: torch.Tensor, win: tuple[int, int, int, int], fn, extra_cols: int = 0, extra_rows: int = 0
                  ) -> torch.Tensor:
    """field with ``fn`` applied to a window block (functional: clone + slice write)."""
    j0, j1, i0, i1 = win
    part = fn(field[..., j0:j1 + extra_rows, i0:i1 + extra_cols])
    out = field.clone()
    out[..., j0:j1 + extra_rows, i0:i1 + extra_cols] = part
    return out


def theta_rhs(th: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nut: torch.Tensor | None, geom, ph: Physics, ops
              ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Tendency of theta: bounded advection, diffusion (1/Pe + nu_t/Pr_t, insulated
    solids, wall conditions on the boundary faces), heaters, heat-flux skins, vents.

    Returns the tendency and the diffusive face fluxes (for wall heat rates).
    """
    fx, fy = tvd_fluxes(th, u, v, ops, inflow=ph.theta_in)
    lt = flux_divergence(fx, fy, ops.h)
    tg = geom.thermal if geom is not None else None
    insulated = tg is not None and not geom.empty
    if nut is not None:
        nc = nut[1:-1, 1:-1] * (1.0 / ph.prt)
        kx = ph.kappa + face_average_x(nc, ops)
        ky = ph.kappa + face_average_y(nc, ops)
    elif ph.paint or insulated:
        kx = ph.kappa.expand(ops.ny, ops.nx + 1)
        ky = ph.kappa.expand(ops.ny + 1, ops.nx)
    else:
        kx = ky = ph.kappa                        # uniform: a 0-d tensor is enough
    if ph.paint:
        kx = kx * (1.0 - ph.paint_u)
        ky = ky * (1.0 - ph.paint_v)
    if insulated:
        kx = _window_apply(kx.contiguous(), geom.window, lambda b: b * tg.ins_u, extra_cols=1)
        ky = _window_apply(ky.contiguous(), geom.window, lambda b: b * tg.ins_v, extra_rows=1)
    Fx, Fy = diffusive_fluxes(th, kx, ky, ops, ph.wall_modes, ph.wall_values)
    lt = lt + flux_divergence(Fx, Fy, ops.h)
    if ph.heat is not None:
        lt = lt + ph.heat
    if tg is not None and not geom.empty and tg.flux_ids:
        lt = _window_apply(lt, geom.window, lambda b: b + ph.kappa * tg.q_skin)
    if ph.ports:
        lt = lt + ph.src_neg * th                 # fluid leaving through a vent takes its heat along
    return lt, (Fx, Fy)


def penalise_theta(th: torch.Tensor, geom, ph: Physics, bdt
                   ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Implicit penalisation of theta toward the temperature bodies and jets.

    Returns the new field, the per-temperature-body heat added this stage (sum
    over the body's share of the increments, in cell units), which - like the
    momentum impulse - is the heat the bodies put into the fluid, and the heat
    the jets added (their enthalpy input: created fluid enters at the ambient
    temperature of the conservative flux form and is penalised to the jet's).
    """
    k = bdt / ph.eta_t
    heat = jet_heat = None
    tg = geom.thermal if geom is not None else None
    if tg is not None and not geom.empty and tg.temp_ids:
        j0, j1, i0, i1 = geom.window
        tw = th[j0:j1, i0:i1]
        tn = (tw + k * tg.a_t) / (1.0 + k * tg.chi_t)
        heat = torch.einsum("bij,ij->b", tg.w_t, tn - tw)
        th = th.clone()
        th[j0:j1, i0:i1] = tn
    if ph.pt_chi is not None:
        tn = (th + k * ph.pt_a) / (1.0 + k * ph.pt_chi)
        jet_heat = (tn - th).sum().reshape(1)
        th = tn
    return th, heat, jet_heat


def wall_heat_rates(th: torch.Tensor, ph: Physics, ops) -> torch.Tensor:
    """Heat rate into the fluid through (left, right, bottom, top) per unit span.

    The boundary-face fluxes of :func:`~.scalar.diffusive_fluxes` with the
    molecular diffusivity (the scheme's own fluxes, so steady states balance to
    round-off): a "temperature" or "inlet" side conducts ``2 kappa (theta_w -
    theta_1)`` per cell, a "heat_flux" side releases ``kappa G`` per unit length.
    """
    k, h = ph.kappa, ops.h
    first = (th[:, 0], th[:, -1], th[0, :], th[-1, :])
    rates = []
    for i, mode in enumerate(ph.wall_modes):
        if mode in ("temperature", "inlet"):
            rates.append((2.0 * k) * (ph.wall_values[i] - first[i]).sum())
        elif mode == "heat_flux":
            rates.append(k * ph.wall_values[i] * (first[i].numel() * h))
        else:
            rates.append(torch.zeros((), device=th.device, dtype=th.dtype))
    return torch.stack(rates)


def flow_heat_rates(th: torch.Tensor, u: torch.Tensor, ph: Physics, ops) -> torch.Tensor:
    """Advective heat (inlet in, outlet out) per unit span, (2,): zero in closed domains.

    The face values are the ones the bounded scheme uses there: the inlet
    temperature at the inlet (the ghost state; the limiter's slope vanishes) and the
    last cell's value at the outlet (zero-gradient ghosts)."""
    if ops.periodic or ops.closed:
        return torch.zeros(2, device=th.device, dtype=th.dtype)
    h = ops.h
    inflow = (u[:, 0] * ph.theta_in).sum() * h
    outflow = (u[:, -1] * th[:, -1]).sum() * h
    return torch.stack([inflow, outflow])


def emitter_on(t: torch.Tensor, timing: torch.Tensor) -> torch.Tensor:
    """1.0 while an emitter group is releasing at time t (device arithmetic)."""
    start, stop, period, duty = timing[0], timing[1], timing[2], timing[3]
    active = (t >= start) & (t < stop)
    safe = torch.where(period > 0, period, torch.ones_like(period))
    phase = torch.remainder(t - start, safe) / safe
    on = active & ((period <= 0) | (phase < duty))
    return on.to(t.dtype)


def update_dye(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt, t, geom, ph: Physics, ops) -> torch.Tensor:
    """One forward-Euler step of all smoke channels (c: (n, ny, nx)) with the end-of-step velocity.

    TVD advection (inlet comb on channel 0 when enabled, clean air elsewhere),
    vents remove smoke with the fluid, emitters relax their footprint toward
    their concentration while on, smoking jets toward 1, and the concentration
    is zero inside solids (bodies with chi > 1/2 and painted obstacles).
    """
    fx, fy = tvd_fluxes(c, u, v, ops, inflow=ph.dye_inflow)
    c = c + dt * flux_divergence(fx, fy, ops.h)
    if ph.ports:
        c = c + (dt * ph.src_neg) * c
    if ph.groups or ph.port_dye:
        chans = list(c.unbind(0))
        for g in ph.groups:
            w = g.weight * emitter_on(t, g.timing)
            chans[g.channel] = torch.addcmul(chans[g.channel], w, g.target - chans[g.channel])
        for ch, w in ph.port_dye.items():
            chans[ch] = torch.addcmul(chans[ch], w, 1.0 - chans[ch])
        c = torch.stack(chans)
    if geom is not None and not geom.empty:
        keep = (geom.chi_c <= 0.5).to(c.dtype)
        c = _window_apply(c, geom.window, lambda b: b * keep)
    if ph.paint:
        c = c * (ph.paint_c <= 0.5).to(c.dtype)
    return torch.nn.functional.threshold(c, DYE_FLOOR, 0.0).clamp(max=1.0)


def diffusion_coefficient(wall_modes: tuple[str, ...]) -> float:
    """Largest diagonal coefficient of the discrete temperature diffusion, in units of kappa / h^2.

    Interior cells have four faces of conductance 1; a Dirichlet side ("temperature"
    or "inlet": the half-cell difference) has conductance 2 and an adiabatic, heat-flux
    or outlet side 0. So a cell next to one Dirichlet side sums to 5 and a corner
    between two Dirichlet sides to 6 (theory doc 16.1)."""
    if "periodic" in wall_modes:
        return 4.0
    dirichlet = [m in ("temperature", "inlet") for m in wall_modes]     # left, right, bottom, top
    best = 4.0
    if any(dirichlet):
        best = 5.0
    for a, b in ((0, 2), (0, 3), (1, 2), (1, 3)):                        # the four corners
        best = max(best, 2.0 + 2.0 * dirichlet[a] + 2.0 * dirichlet[b])
    return best


def physics_dt_limit(dt: torch.Tensor, u: torch.Tensor, v: torch.Tensor, theta: torch.Tensor | None, nut_max,
                     ph: Physics, ops) -> torch.Tensor:
    """Extra stability limits: source speeds (jets, fans) in the CFL bound, the
    bounded (positivity-preserving) temperature step and the buoyant free-fall limit.

    The temperature limit is the forward-Euler positivity condition of the whole
    explicit update: the MUSCL / van Leer flux contributes at most ``2 c`` (face
    Courant numbers summed over both directions, limiter slope ratio <= 2) and the
    diffusion ``c_d kappa dt / h^2`` (:func:`diffusion_coefficient`), so
    ``dt (2 (|u|max + |v|max) / h + c_d kappa / h^2) <= 1`` makes every Runge-Kutta
    stage a convex combination of neighbouring values; SSP-RK3 is a convex
    combination of such stages, so theta obeys a discrete maximum principle. The
    momentum limits alone (advective CFL and 0.25 h^2 / kappa, each separately)
    are not enough: where both are close to their bounds the sum of the two
    coefficients exceeds 1 and the temperature grows without bound (measured:
    heated cylinder, Re 40, 32 cells/D, theta reached 1e30 in 400 steps).
    """
    if ph.ports or ph.nfan:
        dt = torch.minimum(dt, ops.cfl * ops.h / ph.u_floor.clamp_min(1e-6))
    if ph.thermal and theta is not None:
        kap = ph.kappa if nut_max is None else ph.kappa + nut_max * (1.0 / ph.prt)
        speed = torch.maximum(u.abs().amax() + v.abs().amax(), ph.u_floor)
        rate = speed * (2.0 / ops.h) + kap * (ph.diff_coef / (ops.h * ops.h))
        dt = torch.minimum(dt, 1.0 / rate)
        # a fluid parcel at rest accelerates by Ri |theta| per unit time: limit the
        # step so it moves at most cfl cells in one step from rest
        acc = ph.ri.abs() * theta.abs().amax()
        dt = torch.minimum(dt, ops.cfl * torch.sqrt(ops.h / acc.clamp_min(1e-12)))
        if ph.heat_max is not None:
            # Heaters in a box at rest: neither bound above applies on the first step (u = 0, theta = 0),
            # so only the viscous limit was left, and the sealed Draft heater room took a 56 L/U first
            # step, heated its heater cells to theta 1e5 and diverged (critic round 3.4). A parcel heated
            # from rest at S = max|q| has theta = S t, accelerates by Ri S t and moves Ri S t^3 / 6: keep
            # that within cfl cells, and keep each step's theta increment S dt within the scale (<= 1).
            s_max = ph.heat_max.clamp_min(1e-12)
            dt = torch.minimum(dt, 1.0 / s_max)
            dt = torch.minimum(dt, (6.0 * ops.cfl * ops.h / (ph.ri.abs() * s_max).clamp_min(1e-12)) ** (1.0 / 3.0))
    return dt


__all__ = [
    "DyeGroup",
    "Physics",
    "box_profile",
    "build_dye_groups",
    "build_physics",
    "buoyancy",
    "capsule_profile",
    "combine_penalisation",
    "diffusion_coefficient",
    "disc_profile",
    "emitter_footprint",
    "emitter_on",
    "empty_physics",
    "ensure_paint",
    "fan_control",
    "fan_disc_velocity",
    "fan_forces",
    "fan_gains",
    "flow_heat_rates",
    "heat_source_field",
    "needs_physics",
    "paint_brush",
    "paint_faces",
    "penalise_extra",
    "penalise_theta",
    "physics_dt_limit",
    "thermal_eta",
    "thermal_wall_modes",
    "theta_rhs",
    "update_dye",
    "wall_heat_rates",
]
