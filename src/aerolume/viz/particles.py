"""Massless tracer particles for the live view ("smoke wand" dots).

Tracers make unsteady flow legible in a way colour maps cannot: you see the
wake roll up and the flow speed up over the suction side. They are advected
on the same device as the solver fields with ``torch.nn.functional.grid_sample``
(bilinear, cell-centred), so a few thousand of them cost two tiny kernels per
frame and the velocity field never leaves the GPU.

Life cycle: particles are released from an inlet *rake* (a segment, by
default the whole inlet edge), carried by the flow with a second-order
(midpoint) Runge-Kutta step, and recycled back to the rake when they leave
the domain, enter a solid, stop being finite, or outlive their (randomised)
lifetime. Recycling uses ``torch.where`` over all particles rather than
boolean indexing, so a step never synchronises with the host. Each particle
carries an age, from which :meth:`Particles.alphas` derives a fade-in /
fade-out opacity so recycled dots do not pop in and out.

Units: positions and ``extent`` share a length unit (metres in Aerolume) and
``u * dt`` must be in that unit. With non-dimensional solver velocities and a
non-dimensional time step, pass ``dt = dt_nd * length_ref``.

:class:`EmitterParticles` is the placed-physics version (DESIGN.md 10.1): tracer
released from the footprints of the simulation's emitters (points, lines, rakes,
boxes, discs, planes) at a rate per second, with pulses (timelines), in 2D or 3D,
plus short position trails for drawing streak lines.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _as_tensor(x: Any, device: torch.device) -> torch.Tensor:
    t = x.detach() if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    return t.to(device=device, dtype=torch.float32)


class Particles:
    """A fixed-size population of tracers seeded on an inlet rake.

    ``extent`` = (xmin, xmax, ymin, ymax) of the field cells. ``rake`` is a
    segment ``((x0, y0), (x1, y1))`` where particles are (re)released; the
    default is the inlet edge x = xmin over the full height. ``lifetime`` (in
    the time unit of ``dt``) recycles particles after roughly that age --
    individual lifetimes are spread +-25 % so they do not expire together;
    ``None`` keeps them until they leave. ``device`` defaults to the device of
    the first field passed to :meth:`advect` (the population follows the
    solver if it moves). ``seed`` makes the random release reproducible.
    """

    def __init__(self, extent: tuple[float, float, float, float], n: int = 3000, *,
                 rake: tuple[tuple[float, float], tuple[float, float]] | None = None,
                 lifetime: float | None = None, device: str | torch.device | None = None,
                 seed: int = 0) -> None:
        if n <= 0:
            raise ValueError("need at least one particle")
        self.n = int(n)
        self.lifetime = None if lifetime is None else float(lifetime)
        self._seed = int(seed)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._device_fixed = device is not None
        self.extent = tuple(float(e) for e in extent)
        self.rake = rake
        self.seed(fill=True)

    # -- set-up ------------------------------------------------------------
    def _prepare(self) -> None:
        """Cache per-extent/device constants so a step launches as few kernels as possible.

        On Windows (WDDM) every kernel launch costs tens of microseconds of
        host time, far more than the arithmetic on a few thousand particles,
        so the step is written around fused ops (``addcmul``) on these.
        """
        xmin, xmax, ymin, ymax = self.extent
        dev = self.device
        # grid_sample coordinates (align_corners=False): -1/+1 = outer cell edges, row 0 = ymin.
        self._scl = torch.tensor([2.0 / (xmax - xmin), 2.0 / (ymax - ymin)], device=dev)
        self._off = torch.tensor([-1.0 - 2.0 * xmin / (xmax - xmin), -1.0 - 2.0 * ymin / (ymax - ymin)],
                                 device=dev)
        if self.rake is None:
            eps = 1e-6 * (xmax - xmin)       # a hair inside, so a release is never "outside"
            a, b = (xmin + eps, ymin), (xmin + eps, ymax)
        else:
            a, b = self.rake
        self._rake_a = torch.tensor(a, dtype=torch.float32, device=dev)
        self._rake_ab = torch.tensor(b, dtype=torch.float32, device=dev) - self._rake_a

    def set_extent(self, extent: tuple[float, float, float, float]) -> None:
        """Change the domain (e.g. a new plan) and re-seed everything."""
        self.extent = tuple(float(e) for e in extent)
        self.seed(fill=True)

    def seed(self, fill: bool = True) -> None:
        """(Re)initialise all particles.

        ``fill=True`` scatters them uniformly over the domain with random ages,
        so the view is populated immediately instead of waiting one flow-through
        time for the first release to cross the screen. ``fill=False`` puts them
        all on the rake with age 0.
        """
        self._prepare()
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(self._seed)
        xmin, xmax, ymin, ymax = self.extent
        r = torch.rand(self.n, 2, generator=self._gen, device=self.device)
        if fill:
            lo = torch.tensor([xmin, ymin], device=self.device)
            size = torch.tensor([xmax - xmin, ymax - ymin], device=self.device)
            self.pos = lo + r * size
        else:
            self.pos = torch.addcmul(self._rake_a, r[:, :1], self._rake_ab)
        spread = 0.75 + 0.5 * torch.rand(self.n, generator=self._gen, device=self.device)
        self.life = (spread * self.lifetime) if self.lifetime else torch.full(
            (self.n,), float("inf"), device=self.device)
        if fill and self.lifetime:
            self.age = torch.rand(self.n, generator=self._gen, device=self.device) * self.life
        else:
            self.age = torch.zeros(self.n, device=self.device)

    def to(self, device: str | torch.device) -> Particles:
        """Move the population to another device (keeps positions and ages)."""
        dev = torch.device(device)
        if dev != self.device:
            self.pos, self.age, self.life = self.pos.to(dev), self.age.to(dev), self.life.to(dev)
            self.device = dev
            self._prepare()
            self._gen = torch.Generator(device=dev)
            self._gen.manual_seed(self._seed + 1)
        return self

    # -- stepping ----------------------------------------------------------
    @staticmethod
    def _sample(fields: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of a (1, C, ny, nx) stack at grid coordinates g (N, 2) -> (N, C)."""
        out = F.grid_sample(fields, g.view(1, 1, -1, 2), mode="bilinear", padding_mode="border",
                            align_corners=False)
        return out[0, :, 0, :].T

    def advect(self, u: Any, v: Any, dt: float, *, mask: Any = None) -> None:
        """Move every particle one RK2 step through ``(u, v)`` and recycle the lost ones.

        ``u``, ``v`` (and ``mask``, the solid fraction) are (ny, nx) in solver
        orientation (row 0 = ymin). Particles that end outside the domain,
        where ``mask > 0.5``, on a non-finite velocity, or past their lifetime
        are re-released on the rake with a random downstream offset of up to
        one step, which breaks the "pulse train" look of synchronous releases.
        """
        if not self._device_fixed and isinstance(u, torch.Tensor) and u.device != self.device:
            self.to(u.device)
        dev = self.device
        chans = [_as_tensor(u, dev), _as_tensor(v, dev)]
        if mask is not None:
            chans.append(_as_tensor(mask, dev))
        fields = torch.stack(chans)[None]                 # (1, 2|3, ny, nx)
        dt = float(dt)
        n = self.n

        # Midpoint RK2, done in grid coordinates (velocity scaled by _scl).
        g0 = torch.addcmul(self._off, self.pos, self._scl)
        k1 = self._sample(fields, g0)[:, :2]
        k2 = self._sample(fields, torch.addcmul(g0, k1, self._scl * (0.5 * dt)))[:, :2]
        pos = torch.add(self.pos, k2, alpha=dt)
        age = self.age + dt

        # A fresh release for every particle (used only where one dies), sampled
        # together with the new positions: one grid_sample for mask + jitter.
        t = torch.rand(n, 2, generator=self._gen, device=dev)
        fresh = torch.addcmul(self._rake_a, t[:, :1], self._rake_ab)
        g1 = torch.addcmul(self._off, pos, self._scl)
        gf = torch.addcmul(self._off, fresh, self._scl)
        both = self._sample(fields, torch.cat([g1, gf]))
        # NaN compares False, so non-finite positions also count as "outside".
        dead = ~(g1.abs() <= 1.0).all(dim=1) | (age > self.life)
        if mask is not None:
            dead |= both[:n, 2] > 0.5
        fresh = torch.addcmul(fresh, t[:, 1:] * dt, both[n:, :2].nan_to_num())
        self.pos = torch.where(dead[:, None], fresh, pos)
        self.age = torch.where(dead, 0.0, age)

    # -- read-out ----------------------------------------------------------
    def positions(self) -> np.ndarray:
        """(N, 2) float64 positions in the length unit of ``extent``."""
        return self.pos.detach().cpu().double().numpy()

    def ages(self) -> np.ndarray:
        """(N,) ages in the time unit of ``dt``; reset to 0 on release."""
        return self.age.detach().cpu().double().numpy()

    def alphas(self, fade_in: float | None = None, fade_out: float | None = None) -> np.ndarray:
        """(N,) opacity in 0..1 for drawing: ramps up after release, down before expiry.

        Defaults: 15 % of ``lifetime`` for both ramps; without a lifetime there
        is no fade-out and the fade-in is off unless ``fade_in`` is given.
        """
        if self.lifetime:
            fade_in = 0.15 * self.lifetime if fade_in is None else fade_in
            fade_out = 0.15 * self.lifetime if fade_out is None else fade_out
        a = torch.ones_like(self.age)
        if fade_in:
            a = a * (self.age / fade_in).clamp(0.0, 1.0)
        if fade_out and self.lifetime:
            a = a * ((self.life - self.age) / fade_out).clamp(0.0, 1.0)
        return a.detach().cpu().double().numpy()


# ---------------------------------------------------------------------------
# particles released from placed emitters (DESIGN.md 10.1)
# ---------------------------------------------------------------------------

#: Release footprints: 0 = an affine box/segment (s = u), 1 = discrete rake nozzles, 2 = a disc.
_MODE_LINEAR, _MODE_RAKE, _MODE_DISC = 0, 1, 2


def _emitter_value(e: Any, key: str, default: Any = None) -> Any:
    """``e.key`` for a model Emitter, ``e[key]`` for a dict (scripts and tests pass dicts)."""
    if isinstance(e, dict):
        return e.get(key, default)
    return getattr(e, key, default)


def _vec3(v: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if v is None:
        return np.array(default, dtype=float)
    a = np.zeros(3)
    vals = [float(c) for c in v][:3]
    a[:len(vals)] = vals
    return a


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to ``normal`` (any orthonormal pair)."""
    n = normal / (np.linalg.norm(normal) or 1.0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1) or 1.0
    return e1, np.cross(n, e1)


def emitter_footprint(e: Any, *, dim: int, cell: float) -> dict:
    """Where an emitter releases its tracer: an origin, up to three edge vectors, a mode and a jitter radius.

    The release point of a particle is ``origin + s1 A + s2 B + s3 C`` plus a
    random offset within ``radius``, with ``s`` uniform in [0, 1] (mode
    linear), ``s1`` quantised to ``count`` nozzles (rake) or (s1, s2) a point
    of the unit disc (disc). This covers every :data:`aerolume.model.EMITTER_KINDS`:
    a point (a ball of the nozzle radius, about one cell when 0), a line
    (continuous slot), a rake (``count`` discrete nozzles from ``position`` to
    ``end``), an axis-aligned box ``size`` centred at ``position``, and in 3D a
    disc of radius ``size[0]`` or a plane of ``size[:2]`` facing ``normal``. In
    2D a disc is a filled circle and a plane a slot of length ``size[0]``
    across ``normal``.
    """
    kind = str(_emitter_value(e, "kind", "point"))
    p = _vec3(_emitter_value(e, "position"))
    end = _emitter_value(e, "end")
    size = [float(s) for s in (_emitter_value(e, "size", ()) or ())]
    normal = _vec3(_emitter_value(e, "normal"), (1.0, 0.0, 0.0))
    if dim == 2:
        p[2], normal[2] = 0.0, normal[2] * 0.0
    r = float(_emitter_value(e, "radius", 0.0) or 0.0)
    r = r if r > 0 else 0.75 * cell
    zero = np.zeros(3)
    A, B, C, mode, count = zero.copy(), zero.copy(), zero.copy(), _MODE_LINEAR, 1
    origin = p.copy()
    if kind in ("line", "rake") and end is not None:
        A = _vec3(end) - p
        if dim == 2:
            A[2] = 0.0
        if kind == "rake":
            count = max(1, int(_emitter_value(e, "count", 8) or 1))
            mode = _MODE_RAKE if count > 1 else _MODE_LINEAR
            if count == 1:                       # one nozzle: the middle of the segment
                origin, A = p + 0.5 * A, zero.copy()
        else:
            r = float(_emitter_value(e, "radius", 0.0) or 0.0) or 0.25 * cell
    elif kind == "rect":
        s = np.zeros(3)
        s[:min(len(size), dim)] = size[:dim]
        origin = p - 0.5 * s
        A, B, C = np.array([s[0], 0, 0]), np.array([0, s[1], 0]), np.array([0, 0, s[2]])
        r = 0.0
    elif kind == "disc":
        rad = size[0] if size else 4.0 * cell
        if dim == 2:
            A, B, mode = np.array([rad, 0, 0]), np.array([0, rad, 0]), _MODE_DISC
        else:
            e1, e2 = _plane_basis(normal)
            A, B, mode = rad * e1, rad * e2, _MODE_DISC
        r = 0.0
    elif kind == "plane":
        if dim == 2:
            n2 = normal[:2] / (np.linalg.norm(normal[:2]) or 1.0)
            t2 = np.array([-n2[1], n2[0], 0.0])
            length = size[0] if size else 10.0 * cell
            origin, A = p - 0.5 * length * t2, length * t2
        else:
            e1, e2 = _plane_basis(normal)
            a = size[0] if size else 10.0 * cell
            b = size[1] if len(size) > 1 else a
            origin, A, B = p - 0.5 * (a * e1 + b * e2), a * e1, b * e2
        r = 0.0
    return {"origin": origin, "A": A, "B": B, "C": C, "mode": mode, "count": count, "radius": r}


def emitter_on(e: Any, t_s: float) -> bool:
    """Whether an emitter releases at simulated time ``t_s`` [s]: enabled, inside start/stop, and
    in the 'on' part of its pulse (``pulse_period`` > 0 turns it on for ``pulse_duty`` of each period)."""
    if not _emitter_value(e, "enabled", True):
        return False
    start = float(_emitter_value(e, "start", 0.0) or 0.0)
    stop = _emitter_value(e, "stop")
    if t_s < start or (stop is not None and t_s >= float(stop)):
        return False
    period = float(_emitter_value(e, "pulse_period", 0.0) or 0.0)
    if period > 0:
        duty = min(max(float(_emitter_value(e, "pulse_duty", 0.5) or 0.0), 0.0), 1.0)
        return ((t_s - start) % period) < duty * period
    return True


class EmitterParticles:
    """Tracer particles released from placed emitters (DESIGN.md 10.1), in 2D or 3D.

    A fixed pool of ``n_max`` slots on the solver's device. Each emitter (a
    :class:`aerolume.model.Emitter` with ``species="particles"``, or a dict with
    the same keys) releases ``rate`` particles per second from its footprint
    (:func:`emitter_footprint`) while it is on (:func:`emitter_on`: start /
    stop and pulses -- a pulsed line gives timelines); every particle is
    carried with a midpoint (RK2) step through the bilinear (2D) or trilinear
    (3D) velocity and dies when it leaves the domain, enters a solid, meets a
    non-finite velocity or outlives ``lifetime`` (seconds, +-25 % so they do
    not expire together; None = never). A continuously fed point emitter thus
    draws its own **streakline**.

    Units are explicit so the same class serves SI fields and the solver's
    non-dimensional ones: the displacement over a step is
    ``u * dt * velocity_scale`` metres and the step lasts ``dt * time_unit_s``
    seconds (SI fields and dt in seconds: both 1; non-dimensional velocity and
    dt in L/U: ``velocity_scale = L``, ``time_unit_s = L / U``).

    Nothing in :meth:`step` waits for the GPU: release counts are accumulated
    on the host, free slots are found with a cumulative sum and every update is
    a ``torch.where`` over the pool. Reading positions (:meth:`positions`)
    copies to the host, once per drawn frame. ``trail`` > 0 keeps the last
    ``trail`` positions of every particle for streak lines (:meth:`trails`).
    """

    def __init__(self, extent: tuple[float, ...], emitters: Any = (), *, n_max: int = 20000,
                 lifetime: float | None = None, trail: int = 0, velocity_scale: float = 1.0,
                 time_unit_s: float = 1.0, cell: float | None = None, device: str | torch.device | None = None,
                 seed: int = 0, species: tuple[str, ...] = ("particles",)) -> None:
        ext = tuple(float(e) for e in extent)
        if len(ext) not in (4, 6):
            raise ValueError("extent must be (x0, x1, y0, y1) in 2D or (x0, x1, y0, y1, z0, z1) in 3D")
        if n_max <= 0:
            raise ValueError("need at least one particle slot")
        self.dim = 2 if len(ext) == 4 else 3
        self.extent = ext
        self.n_max = int(n_max)
        self.lifetime = None if lifetime is None else float(lifetime)
        self.trail = max(0, int(trail))
        self.velocity_scale = float(velocity_scale)
        self.time_unit_s = float(time_unit_s)
        spans = [ext[2 * k + 1] - ext[2 * k] for k in range(self.dim)]
        self.cell = float(cell) if cell else min(spans) / 200.0
        self.species = tuple(species)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._device_fixed = device is not None
        self._seed = int(seed)
        self.t = 0.0                                   # simulated seconds since the start
        self.emitters: list = []
        self._acc: list[float] = []
        self._reset_pool()
        self.set_emitters(emitters)

    # -- set-up ------------------------------------------------------------
    def _reset_pool(self) -> None:
        dev, n, d = self.device, self.n_max, self.dim
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(self._seed)
        self.pos = torch.zeros(n, d, device=dev)
        self.alive = torch.zeros(n, dtype=torch.bool, device=dev)
        self.age = torch.zeros(n, device=dev)
        self.life = torch.full((n,), float("inf"), device=dev)
        self.source = torch.zeros(n, dtype=torch.long, device=dev)
        self._hist = torch.full((max(self.trail, 1), n, d), float("nan"), device=dev) if self.trail else None
        self._head = 0
        lo = torch.tensor([self.extent[2 * k] for k in range(d)], device=dev)
        hi = torch.tensor([self.extent[2 * k + 1] for k in range(d)], device=dev)
        self._lo, self._hi = lo, hi
        self._scl = 2.0 / (hi - lo)
        self._off = -1.0 - lo * self._scl

    def set_emitters(self, emitters: Any) -> None:
        """Use these emitters from now on (a live move): particles already released keep flying.

        Emitters of other species (``species="dye"``: smoke) and disabled ones
        release nothing. Positions are metres in the frame the velocity is
        given in (the caller turns a body-frame emitter into the wind frame).
        """
        self.emitters = [e for e in (emitters or []) if str(_emitter_value(e, "species", "particles")) in self.species]
        self._acc = [0.0] * len(self.emitters)
        feet = [emitter_footprint(e, dim=self.dim, cell=self.cell) for e in self.emitters]
        dev, d = self.device, self.dim
        if feet:
            def stack(key: str) -> torch.Tensor:
                return torch.tensor(np.array([f[key][:d] for f in feet]), dtype=torch.float32, device=dev)

            self._origin, self._A, self._B, self._C = stack("origin"), stack("A"), stack("B"), stack("C")
            self._mode = torch.tensor([f["mode"] for f in feet], dtype=torch.long, device=dev)
            self._count = torch.tensor([f["count"] for f in feet], dtype=torch.float32, device=dev)
            self._radius = torch.tensor([f["radius"] for f in feet], dtype=torch.float32, device=dev)
        self._feet = feet

    def to(self, device: str | torch.device) -> EmitterParticles:
        """Move the pool to another device (keeps the particles)."""
        dev = torch.device(device)
        if dev == self.device:
            return self
        for k in ("pos", "alive", "age", "life", "source", "_lo", "_hi", "_scl", "_off"):
            setattr(self, k, getattr(self, k).to(dev))
        if self._hist is not None:
            self._hist = self._hist.to(dev)
        self.device = dev
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(self._seed + 1)
        self.set_emitters(self.emitters)
        return self

    def clear(self) -> None:
        """Remove every particle (a reset of the flow) and restart the emitters' clocks."""
        self.t = 0.0
        self._reset_pool()
        self.set_emitters(self.emitters)

    # -- release -----------------------------------------------------------
    def _release_counts(self, dt_s: float) -> list[int]:
        out = []
        for i, e in enumerate(self.emitters):
            rate = max(0.0, float(_emitter_value(e, "rate", 0.0) or 0.0))
            if emitter_on(e, self.t) and rate > 0:
                self._acc[i] += rate * dt_s
            k = int(self._acc[i])
            self._acc[i] -= k
            out.append(k)
        return out

    def _release_points(self, src: torch.Tensor) -> torch.Tensor:
        """Release positions for slots fed by emitter indices ``src`` (one per slot)."""
        n, d, dev = src.shape[0], self.dim, self.device
        u = torch.rand(n, 3, generator=self._gen, device=dev)
        mode = self._mode[src]
        s1 = u[:, 0]
        cnt = self._count[src]
        rake = torch.floor(s1 * cnt).clamp(max=cnt - 1) / (cnt - 1).clamp(min=1.0)
        s1 = torch.where(mode == _MODE_RAKE, rake, s1)
        rad = torch.sqrt(u[:, 0])
        ang = 2.0 * math.pi * u[:, 1]
        disc = mode == _MODE_DISC
        a = torch.where(disc, rad * torch.cos(ang), s1)
        b = torch.where(disc, rad * torch.sin(ang), u[:, 1])
        p = self._origin[src] + a[:, None] * self._A[src] + b[:, None] * self._B[src] + u[:, 2:3] * self._C[src]
        # nozzle jitter: uniform in a ball (disc in 2D) of the emitter's radius
        g = torch.randn(n, d, generator=self._gen, device=dev)
        g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-12)
        rr = self._radius[src] * torch.rand(n, generator=self._gen, device=dev) ** (1.0 / d)
        return p + g * rr[:, None]

    def _release(self, counts: list[int]) -> None:
        total = sum(counts)
        if total <= 0 or not self.emitters:
            return
        dev = self.device
        dead = ~self.alive
        rank = torch.cumsum(dead.to(torch.int32), 0) - 1
        take = dead & (rank < total)
        cum = torch.tensor(np.cumsum(counts), dtype=torch.long)
        if dev.type == "cuda":
            # a pageable host -> device copy blocks until the GPU is idle; a pinned one is queued
            cum = cum.pin_memory().to(dev, non_blocking=True)
        else:
            cum = cum.to(dev)
        src = torch.searchsorted(cum, rank.clamp(min=0).to(torch.long), right=True).clamp(max=len(counts) - 1)
        fresh = self._release_points(src)
        self.pos = torch.where(take[:, None], fresh, self.pos)
        self.source = torch.where(take, src, self.source)
        self.age = torch.where(take, torch.zeros_like(self.age), self.age)
        if self.lifetime:
            spread = 0.75 + 0.5 * torch.rand(self.n_max, generator=self._gen, device=dev)
            self.life = torch.where(take, spread * self.lifetime, self.life)
        if self._hist is not None:       # a new particle has no trail yet (where, not a boolean index: no sync)
            self._hist = torch.where(take[None, :, None], torch.full_like(self._hist[:1], float("nan")), self._hist)
        self.alive = self.alive | take

    # -- stepping ----------------------------------------------------------
    def _sample(self, fields: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of a (1, C, [nz,] ny, nx) stack at positions x (N, dim) -> (N, C)."""
        g = torch.addcmul(self._off, x, self._scl)
        if self.dim == 2:
            out = F.grid_sample(fields, g.view(1, 1, -1, 2), mode="bilinear", padding_mode="border",
                                align_corners=False)
            return out[0, :, 0, :].T
        out = F.grid_sample(fields, g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border",
                            align_corners=False)
        return out[0, :, 0, 0, :].T

    def step(self, u: Any, v: Any, w: Any = None, *, dt: float, mask: Any = None, substeps: int = 1) -> None:
        """Release, advect and retire particles over ``dt`` (in the caller's time unit, see the class).

        ``u``, ``v`` (and ``w`` in 3D, and ``mask`` = solid fraction) are
        cell-centred arrays in solver orientation: (ny, nx) or (nz, ny, nx),
        row 0 = the minimum coordinate. ``substeps`` splits ``dt`` so that no
        particle jumps more than about 1.5 cells per sub-step near a body.
        """
        ref = u if isinstance(u, torch.Tensor) else None
        if not self._device_fixed and ref is not None and ref.device != self.device:
            self.to(ref.device)
        dev = self.device
        chans = [_as_tensor(u, dev), _as_tensor(v, dev)]
        if self.dim == 3:
            if w is None:
                raise ValueError("3D particles need the w velocity")
            chans.append(_as_tensor(w, dev))
        has_mask = mask is not None
        if has_mask:
            chans.append(_as_tensor(mask, dev))
        fields = torch.stack(chans)[None]
        n_sub = max(1, int(substeps))
        h = float(dt) / n_sub
        for _ in range(n_sub):
            self._release(self._release_counts(h * self.time_unit_s))
            k = self.velocity_scale * h
            k1 = self._sample(fields, self.pos)[:, :self.dim]
            k2 = self._sample(fields, self.pos + (0.5 * k) * k1)[:, :self.dim]
            pos = self.pos + k * k2
            age = self.age + h * self.time_unit_s
            inside = ((pos >= self._lo) & (pos <= self._hi)).all(dim=1)       # NaN compares False
            alive = self.alive & inside & (age <= self.life)
            if has_mask:
                alive &= self._sample(fields, pos)[:, -1] <= 0.5
            self.pos = torch.where(alive[:, None], pos, self.pos)
            self.age = torch.where(alive, age, self.age)
            self.alive = alive
            self.t += h * self.time_unit_s
        if self._hist is not None:
            self._hist[self._head] = torch.where(self.alive[:, None], self.pos,
                                                 torch.full_like(self.pos, float("nan")))
            self._head = (self._head + 1) % self.trail

    @staticmethod
    def suggest_substeps(dt: float, speed_bound: float, cell: float, velocity_scale: float = 1.0,
                         max_substeps: int = 24) -> int:
        """Sub-steps that keep each move under ~1.5 cells for velocities up to ``speed_bound``."""
        if not (cell > 0 and dt > 0 and speed_bound > 0):
            return 1
        return int(min(max_substeps, max(1, math.ceil(speed_bound * velocity_scale * dt / (1.5 * cell)))))

    # -- read-out ----------------------------------------------------------
    def count(self) -> int:
        """Particles in flight (a host sync)."""
        return int(self.alive.sum())

    def positions(self) -> np.ndarray:
        """(N, dim) float64 positions [m] of the particles in flight."""
        return self.pos[self.alive].detach().cpu().double().numpy()

    def emitter_ids(self) -> np.ndarray:
        """(N,) index into :attr:`emitters` of each particle in flight (for colouring by emitter)."""
        return self.source[self.alive].detach().cpu().numpy()

    def ages(self) -> np.ndarray:
        """(N,) ages [s] of the particles in flight."""
        return self.age[self.alive].detach().cpu().double().numpy()

    def trails(self) -> np.ndarray:
        """(N, K, dim) recent positions of each particle in flight, oldest first; NaN before its release.

        Recorded once per :meth:`step` call; empty (N, 0, dim) without ``trail``.
        """
        n_alive = self.alive
        if self._hist is None:
            return np.zeros((int(n_alive.sum()), 0, self.dim))
        order = [(self._head + i) % self.trail for i in range(self.trail)]
        h = self._hist[order][:, n_alive]                    # (K, N, dim)
        return h.permute(1, 0, 2).detach().cpu().double().numpy()
