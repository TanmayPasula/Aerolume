"""Flow-visualisation emitters of the 3D solver (DESIGN.md 10.1): smoke channels and
GPU tracer particles.

Smoke
-----
Up to four independent passive scalars ("dye", "dye1", "dye2", "dye3"), each
transported exactly like the v0.2 smoke (:mod:`.dye`: MUSCL / van Leer,
bounded, zero in solids). An :class:`~.config.Emitter3D` of species "dye"
relaxes its channel towards ``rate`` inside its footprint after every step,
``c += s on (rate - c)``, where ``s`` is the anti-aliased footprint weight and
``on`` the release switch -- enabled, between ``start`` and ``stop``, and in
the "on" part of the pulse. The switch is evaluated on the device from the
solver time and per-emitter parameters held in a tensor, so pulses, start and
stop never leave a CUDA graph and editing them never re-captures. The v0.2
:class:`~.config.DyeRake` is the special case rate = 1, start = 0 on
channel 0, built from the same weights, so the default path is unchanged.

Particles
---------
Tracer particles live in fixed device buffers (positions, alive flag, birth
time), one ring per emitter. Every step an emitter adds ``rate dt`` to an
accumulator and releases its integer part (up to ``k_max`` per step, a static
size, so the release is graph-capturable: the candidate slots are always
``k_max`` long and those beyond the count keep their old contents): "point" and
"rake" emitters at their nozzle centres (taking turns along a rake; within
``radius`` when one is given), area kinds at uniformly random points of their
footprint. The releases of a step are spread evenly over it in time (each new
particle moves only for its share of the step), so a nozzle draws an evenly
spaced streak line at any rate. The ring overwrites its oldest particles, which
is the recycling. Particles are advected with the midpoint rule (RK2) through
the trilinearly interpolated staggered velocity of the step's final
(divergence-free) state, and are retired when they leave the domain, enter a
solid (solid fraction > 1/2) or outlive ``lifetime``; a symmetry plane
reflects them and periodic axes wrap. A continuous emitter draws streaklines; a
pulsed one cuts them into dashes whose ends line up into timelines.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from .config import Emitter3D
from .dye import NOZZLE_DEPTH_CELLS, NOZZLE_RADIUS_CELLS, source_weights
from .fields import trilinear
from .physics import _axes_of, grid_coordinates, grid_window, ramp, window_empty

#: emitters whose pulse is "always on" use this period / duration (exceeds any run)
_FOREVER = 1e30
#: particle slots per particle emitter when ``particles_max`` is 0
DEFAULT_PARTICLES_PER_EMITTER = 1 << 16


class EmitterSource(NamedTuple):
    """One smoke emitter on the grid: cell window, weights on it, channel, and the device
    parameters ``[rate, start, stop, period, t_on]`` of the release switch."""
    window: tuple[int, int, int, int, int, int]
    weight: torch.Tensor
    params: torch.Tensor
    channel: int

    @property
    def empty(self) -> bool:
        return window_empty(self.window)


def emitter_params(e: Emitter3D | None, device, dtype) -> torch.Tensor:
    """``[rate, start, stop, period, t_on]``; None = the v0.2 rake (always on, rate 1)."""
    if e is None:
        return torch.tensor([1.0, 0.0, _FOREVER, _FOREVER, _FOREVER], device=device, dtype=dtype)
    stop = _FOREVER if e.stop is None else e.stop
    if e.pulse_period > 0:
        period, t_on = e.pulse_period, e.pulse_duty * e.pulse_period
    else:
        period, t_on = _FOREVER, _FOREVER
    rate = e.rate if e.enabled else 0.0
    return torch.tensor([rate, e.start, stop, period, t_on], device=device, dtype=dtype)


def release_switch(t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """1 while the emitter releases at time ``t`` (device scalar), else 0."""
    since = t - params[1]
    on = (since >= 0) & (t < params[2]) & (torch.remainder(since, params[3]) < params[4])
    return on.to(params.dtype)


def footprint_function(e: Emitter3D, h: float, extent):
    """(fn(X, Y, Z) -> weight, lo, hi) of an emitter's footprint in solver units."""
    p = np.asarray(e.position, dtype=np.float64)
    n = np.asarray(e.normal, dtype=np.float64)
    r = e.radius if e.radius > 0 else NOZZLE_RADIUS_CELLS * h
    depth = NOZZLE_DEPTH_CELLS * h
    e1, e2 = _axes_of(n)

    def tube(c):
        def fn(X, Y, Z):
            dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
            s = dx * n[0] + dy * n[1] + dz * n[2]
            rr = np.sqrt(np.maximum(dx * dx + dy * dy + dz * dz - s * s, 0.0))
            return ramp(rr, r, h) * ramp(np.abs(s), depth, h)
        return fn
    ext = np.array([r + depth + h] * 3)
    if e.kind == "point":
        return tube(p), p - ext, p + ext
    if e.kind in ("line", "rake"):
        q = np.asarray(e.end, dtype=np.float64)
        lo, hi = np.minimum(p, q) - ext, np.maximum(p, q) + ext
        if e.kind == "rake":
            cs = [p + (q - p) * (k / max(e.count - 1, 1)) for k in range(e.count)] if e.count > 1 else [0.5 * (p + q)]
            fns = [tube(c) for c in cs]

            def fn(X, Y, Z):
                out = fns[0](X, Y, Z)
                for f in fns[1:]:
                    out = np.maximum(out, f(X, Y, Z))
                return out
            return fn, lo, hi
        seg = q - p
        ll = float(seg @ seg)

        def fn(X, Y, Z):
            t = ((X - p[0]) * seg[0] + (Y - p[1]) * seg[1] + (Z - p[2]) * seg[2]) / max(ll, 1e-30)
            t = np.clip(t, 0.0, 1.0)
            d = np.sqrt((X - p[0] - t * seg[0]) ** 2 + (Y - p[1] - t * seg[1]) ** 2 + (Z - p[2] - t * seg[2]) ** 2)
            return ramp(d, r, h)
        return fn, lo, hi
    if e.kind == "rect":
        hs = np.array([s if s > 0 else 2 * depth for s in e.size]) * 0.5

        def fn(X, Y, Z):
            return ramp(np.abs(X - p[0]), hs[0], h) * ramp(np.abs(Y - p[1]), hs[1], h) * ramp(np.abs(Z - p[2]), hs[2], h)
        return fn, p - hs - h, p + hs + h
    if e.kind == "disc":
        R = e.size[0]

        def fn(X, Y, Z):
            dx, dy, dz = X - p[0], Y - p[1], Z - p[2]
            s = dx * n[0] + dy * n[1] + dz * n[2]
            rr = np.sqrt(np.maximum(dx * dx + dy * dy + dz * dz - s * s, 0.0))
            return ramp(rr, R, h) * ramp(np.abs(s), depth, h)
        ext = np.array([R + depth + h] * 3)
        return fn, p - ext, p + ext
    # plane: a rectangle (a, b) in the plane through p facing n; no size = the whole cross-section
    if e.size:
        a, b = 0.5 * e.size[0], 0.5 * e.size[1]
    else:
        big = 4.0 * max(extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
        a = b = big

    def fn(X, Y, Z):
        dx, dy, dz = X - p[0], Y - p[1], Z - p[2]
        s = dx * n[0] + dy * n[1] + dz * n[2]
        q1 = dx * e1[0] + dy * e1[1] + dz * e1[2]
        q2 = dx * e2[0] + dy * e2[1] + dz * e2[2]
        return ramp(np.abs(q1), a, h) * ramp(np.abs(q2), b, h) * ramp(np.abs(s), depth, h)
    corners = np.array([p + sa * a * e1 + sb * b * e2 for sa in (-1, 1) for sb in (-1, 1)])
    lo = np.maximum(corners.min(axis=0) - depth - h, np.array([extent[0], extent[2], extent[4]]) - h)
    hi = np.minimum(corners.max(axis=0) + depth + h, np.array([extent[1], extent[3], extent[5]]) + h)
    return fn, lo, hi


def build_smoke_sources(emitters: Sequence[Emitter3D] | None, rake, ops, extent) -> list[EmitterSource]:
    """Smoke sources on the grid: the "dye" emitters, or (``emitters`` None) the v0.2 rake
    on channel 0 with its own weights. Emitters that miss the domain are dropped."""
    dev, dt = ops.device, ops.dtype
    out = []
    if emitters is None:
        if rake is None:
            return out
        win, w = source_weights(rake, nx=ops.nx, ny=ops.ny, nz=ops.nz, h=ops.h, x0=ops.x0, y0=ops.y0, z0=ops.z0)
        out.append(EmitterSource(win, torch.as_tensor(w, device=dev, dtype=dt), legacy_params(rake, dev, dt), 0))
        return out
    for e in emitters:
        if e.species != "dye":
            continue
        fn, lo, hi = footprint_function(e, ops.h, extent)
        win = grid_window(lo, hi, ops, margin=1.0)
        if window_empty(win):
            continue
        Z, Y, X = grid_coordinates(win, ops, "c")
        w = np.broadcast_to(fn(X, Y, Z), np.broadcast_shapes(Z.shape, Y.shape, X.shape))
        if not w.any():
            continue
        out.append(EmitterSource(win, torch.as_tensor(np.array(w, dtype=np.float64), device=dev, dtype=dt),
                                 emitter_params(e, dev, dt), e.channel))
    return out


def legacy_params(rake, device, dtype) -> torch.Tensor:
    """The release parameters of a v0.2 :class:`DyeRake` (rate 1 from t = 0, its pulses)."""
    if rake is not None and rake.pulsed:
        return torch.tensor([1.0, 0.0, _FOREVER, rake.pulse_on + rake.pulse_off, rake.pulse_on], device=device,
                            dtype=dtype)
    return emitter_params(None, device, dtype)


def apply_smoke_sources(channels: Sequence[torch.Tensor], t, sources: Sequence[EmitterSource]) -> None:
    """In place: relax each emitter's channel towards its rate inside the footprint."""
    for src in sources:
        if src.empty:
            continue
        k0, k1, j0, j1, i0, i1 = src.window
        win = channels[src.channel][k0:k1, j0:j1, i0:i1]
        p = src.params
        on = release_switch(t, p)
        win.add_(src.weight * on * (p[0] - win))


# ---------------------------------------------------------------------------
# tracer particles
# ---------------------------------------------------------------------------

class _Sampler:
    """Uniform random points of one emitter's footprint (device tensors, static sizes)."""

    def __init__(self, e: Emitter3D, h: float, extent, device, dtype):
        self.kind = e.kind
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), device=device, dtype=dtype)
        self.p = t(e.position)
        self.n = t(e.normal)
        e1, e2 = _axes_of(np.asarray(e.normal, dtype=np.float64))
        self.e1, self.e2 = t(e1), t(e2)
        self.r = e.radius if e.radius > 0 else NOZZLE_RADIUS_CELLS * h
        # point and rake particles leave from the nozzle centre (within an explicit radius only), so
        # each nozzle draws one clean streak line; area kinds fill their footprint
        self.r_nozzle = e.radius
        self.depth = NOZZLE_DEPTH_CELLS * h
        self.q = t(e.end) if e.end is not None else self.p
        self.count = e.count
        if e.kind == "rect":
            self.size = t([s if s > 0 else 2 * self.depth for s in e.size])
        elif e.kind == "disc":
            self.R = e.size[0]
        elif e.kind == "plane":
            if e.size:
                self.ab = (e.size[0], e.size[1])
            else:
                L = max(extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
                self.ab = (4.0 * L, 4.0 * L)

    def _disc(self, k: int, radius: float, dev, dt):
        rr = radius * torch.sqrt(torch.rand(k, device=dev, dtype=dt))
        ph = (2.0 * math.pi) * torch.rand(k, device=dev, dtype=dt)
        return (rr * torch.cos(ph))[:, None] * self.e1 + (rr * torch.sin(ph))[:, None] * self.e2

    def __call__(self, k: int, nozzle: torch.Tensor | None = None) -> torch.Tensor:
        """``k`` random points of the footprint; a rake places point ``i`` in nozzle
        ``nozzle[i]`` (long tensor, 0..count-1; None = random nozzles)."""
        dev, dt = self.p.device, self.p.dtype
        if self.kind == "point":
            return self.p + self._disc(k, self.r_nozzle, dev, dt)
        if self.kind == "rake":
            idx = (torch.randint(0, self.count, (k,), device=dev) if nozzle is None else nozzle).to(dt)
            frac = idx / max(self.count - 1, 1) if self.count > 1 else torch.full_like(idx, 0.5)
            return self.p + frac[:, None] * (self.q - self.p) + self._disc(k, self.r_nozzle, dev, dt)
        along = (2.0 * torch.rand(k, device=dev, dtype=dt) - 1.0)[:, None] * self.depth * self.n
        if self.kind == "line":
            frac = torch.rand(k, device=dev, dtype=dt)[:, None]
            g = torch.randn(k, 3, device=dev, dtype=dt)
            g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            rad = self.r * torch.rand(k, device=dev, dtype=dt).pow(1.0 / 3.0)[:, None]
            return self.p + frac * (self.q - self.p) + rad * g
        if self.kind == "rect":
            return self.p + (torch.rand(k, 3, device=dev, dtype=dt) - 0.5) * self.size
        if self.kind == "disc":
            return self.p + self._disc(k, self.R, dev, dt) + along
        a, b = self.ab
        return (self.p + ((torch.rand(k, device=dev, dtype=dt) - 0.5) * a)[:, None] * self.e1
                + ((torch.rand(k, device=dev, dtype=dt) - 0.5) * b)[:, None] * self.e2 + along)


class ParticleSystem3D:
    """GPU tracer particles released by the "particles" emitters (see the module docstring).

    Buffers (device): ``pos`` (N, 3), ``alive`` (N,) 0/1, ``birth`` (N,),
    ``nozzle`` (N,) the nozzle a particle left from (a rake's nozzles take turns,
    so every nozzle carries the same share; 0 for every other kind), per emitter
    a ring cursor and a release accumulator. ``dt_estimate`` sizes the largest
    release per step (``k_max = ceil(4 rate dt_estimate)``, at least 1); a step
    longer than four times the estimate releases at most ``k_max``.
    """

    def __init__(self, emitters: Sequence[Emitter3D], ops, extent, *, capacity: int = 0, dt_estimate: float = 0.01):
        dev, dt = ops.device, ops.dtype
        self.ops = ops
        self.extent = tuple(float(c) for c in extent)
        self.emitters = [e for e in emitters if e.species == "particles"]
        ne = len(self.emitters)
        per = (capacity // max(ne, 1)) if capacity > 0 else DEFAULT_PARTICLES_PER_EMITTER
        per = max(per, 16)
        self.per = per
        self.capacity = per * ne
        self.pos = torch.zeros(self.capacity, 3, device=dev, dtype=dt)
        self.alive = torch.zeros(self.capacity, device=dev, dtype=dt)
        self.birth = torch.zeros(self.capacity, device=dev, dtype=dt)
        self.nozzle = torch.zeros(self.capacity, device=dev, dtype=torch.long)
        self.cursor = torch.zeros(max(ne, 1), device=dev, dtype=torch.long)
        self.acc = torch.zeros(max(ne, 1), device=dev, dtype=dt)
        self.samplers = [_Sampler(e, ops.h, self.extent, dev, dt) for e in self.emitters]
        self.params = [emitter_params(e, dev, dt) for e in self.emitters]
        self.kmax = [max(1, min(per // 4, math.ceil(4.0 * e.rate * dt_estimate))) for e in self.emitters]
        self.lifetime = torch.tensor([e.lifetime if e.lifetime is not None else _FOREVER for e in self.emitters] or
                                     [_FOREVER], device=dev, dtype=dt)
        self.owner = torch.arange(self.capacity, device=dev) // per if ne else torch.zeros(0, device=dev,
                                                                                           dtype=torch.long)
        self._ar = {k: torch.arange(k, device=dev) for k in set(self.kmax)}
        x0, x1, y0, y1, z0, z1 = self.extent
        self._lo = torch.tensor([x0, y0, z0], device=dev, dtype=dt)
        self._span = torch.tensor([x1 - x0, y1 - y0, z1 - z0], device=dev, dtype=dt)

    def __len__(self) -> int:
        return len(self.emitters)

    def buffers(self) -> list[torch.Tensor]:
        """The tensors a step mutates in place (positions, alive, birth, nozzle first: the
        solver copies those four across a live emitter edit that keeps the capacity)."""
        return [self.pos, self.alive, self.birth, self.nozzle, self.cursor, self.acc]

    def reset(self) -> None:
        for b in self.buffers():
            b.zero_()

    # ---------------------------------------------------------------- stepping
    def _release(self, t, dt) -> list[tuple]:
        """Release the step's new particles, spread evenly over the step in time: the j-th
        of n leaves at ``t - f_j dt`` with ``f_j = (j + 1/2) / n`` and must move only for
        ``f_j dt`` (:meth:`step` scales its displacement), so a nozzle's particles form an
        evenly spaced streak line at any release rate and ``age = t_now - birth`` is the time
        each has been moving. Returns per emitter (slots, mask, start points, f) for
        :meth:`step` (static shapes: graph-capturable)."""
        out = []
        for i, (smp, prm, k) in enumerate(zip(self.samplers, self.params, self.kmax)):
            on = release_switch(t, prm)
            a = (self.acc[i] + prm[0] * dt * on).clamp(max=float(k))
            n = torch.floor(a)
            self.acc[i] = a - n
            ar = self._ar[k]
            slots = i * self.per + torch.remainder(self.cursor[i] + ar, self.per)
            mask = ar < n
            frac = (ar.to(n.dtype) + 0.5) / n.clamp(min=1.0)
            noz = torch.remainder(self.cursor[i] + ar, smp.count) if smp.kind == "rake" else torch.zeros_like(ar)
            new = smp(k, noz)
            self.pos[slots] = torch.where(mask[:, None], new, self.pos[slots])
            self.alive[slots] = torch.where(mask, torch.ones_like(n), self.alive[slots])
            self.birth[slots] = torch.where(mask, t - frac * dt, self.birth[slots])
            self.nozzle[slots] = torch.where(mask, noz, self.nozzle[slots])
            self.cursor[i] = torch.remainder(self.cursor[i] + n.long(), self.per)
            out.append((slots, mask, new, frac))
        return out

    def _velocity(self, p, u, v, w) -> torch.Tensor:
        ops = self.ops
        q = p
        flip = None
        if ops.symmetry:
            low = q[:, 1] < ops.y0
            q = torch.where(low[:, None], torch.stack([q[:, 0], 2.0 * ops.y0 - q[:, 1], q[:, 2]], dim=1), q)
            flip = torch.where(low, -1.0, 1.0).to(p.dtype)
        vx = trilinear(u, q, ops, (0.0, 0.5, 0.5))
        vy = trilinear(v, q, ops, (0.5, 0.0, 0.5))
        vz = trilinear(w, q, ops, (0.5, 0.5, 0.0))
        if flip is not None:
            vy = vy * flip
        return torch.stack([vx, vy, vz], dim=1)

    def step(self, u, v, w, t, dt, solid=None, *, compact: bool = False) -> None:
        """Release, advect (midpoint rule) and retire particles for one solver step that ended
        at time ``t`` with step ``dt``; ``solid(pos) -> bool`` marks positions in solids.

        ``compact`` advects only the live slots instead of the whole ring. Why: graph capture
        needs static shapes, so the default advects every slot, which costs nothing extra on a
        GPU but dominates a CPU run -- a 2^16 ring cost 14x the flow step of a 4,600-cell
        tunnel holding a few hundred particles (theory_ns3d.md 16). Finding the live slots
        is a host sync, free on the CPU and not capturable, so the solver asks for it only
        there (eager CPU steps). Per particle the arithmetic is the same, so live particles
        move bit for bit as in the full path; dead slots keep stale positions, which nothing
        reads (a release overwrites them)."""
        if not self.capacity:
            return
        released = self._release(t, dt)
        if compact:
            idx = torch.nonzero(self.alive > 0).squeeze(1)       # includes this step's releases
            p = self.pos.clone()
            q = p[idx]
            k1 = self._velocity(q, u, v, w)
            k2 = self._velocity(q + (0.5 * dt) * k1, u, v, w)
            p[idx] = q + dt * k2
        else:
            p = self.pos
            k1 = self._velocity(p, u, v, w)
            k2 = self._velocity(p + (0.5 * dt) * k1, u, v, w)
            p = p + dt * k2
        for slots, mask, start, frac in released:          # new particles move for f dt only
            part = start + frac[:, None] * (p[slots] - start)
            p[slots] = torch.where(mask[:, None], part, p[slots])
        ops = self.ops
        y0 = self.extent[2]
        lo, span = self._lo, self._span
        per = ops.periodic_axes
        keep = self.alive > 0
        for a in range(3):
            if per[a]:
                p[:, a] = lo[a] + torch.remainder(p[:, a] - lo[a], span[a])
            else:
                if a == 1 and ops.symmetry:
                    p[:, 1] = torch.where(p[:, 1] < y0, 2.0 * y0 - p[:, 1], p[:, 1])
                keep = keep & (p[:, a] >= lo[a]) & (p[:, a] <= lo[a] + span[a])
        if solid is not None:
            if compact:
                hit = torch.zeros_like(keep)
                hit[idx] = solid(p[idx])
                keep = keep & ~hit
            else:
                keep = keep & ~solid(p)
        if len(self.emitters):
            keep = keep & ((t - self.birth) <= self.lifetime[self.owner])
        self.pos.copy_(p)
        self.alive.copy_(keep.to(self.alive.dtype))

    # ------------------------------------------------------------------ output
    def snapshot(self) -> dict[str, np.ndarray]:
        """The live particles (one host copy): ``positions`` (n, 3) float32, ``birth``
        (n,) release time, ``emitter`` (n,) int16 index among the particle emitters,
        ``nozzle`` (n,) int16 (solver units; the age is ``t - birth``)."""
        a = self.alive.detach().cpu().numpy() > 0
        pos = self.pos.detach().cpu().numpy()[a].astype(np.float32)
        birth = self.birth.detach().cpu().double().numpy()[a]
        noz = self.nozzle.detach().cpu().numpy()[a].astype(np.int16)
        owner = self.owner.detach().cpu().numpy()[a].astype(np.int16) if len(self.emitters) else np.zeros(0, np.int16)
        return {"positions": pos, "birth": birth, "emitter": owner, "nozzle": noz}

    def lines(self) -> list[np.ndarray]:
        """Streak lines for drawing, each an (n, 3) float32 polyline ordered from the
        nozzle outwards (youngest first), one per nozzle of every "point" and "rake"
        emitter. A pulsed emitter's line is cut wherever the release paused, so its
        dashes are the pulses; across a rake their ends line up into timelines.
        Area emitters (line, rect, disc, plane) release at random points of their
        footprint: they have no lines, draw their particles as points."""
        snap = self.snapshot()
        out: list[np.ndarray] = []
        for i, e in enumerate(self.emitters):
            if e.kind not in ("point", "rake"):
                continue
            m = snap["emitter"] == i
            if not m.any():
                continue
            pos, birth, noz = snap["positions"][m], snap["birth"][m], snap["nozzle"][m]
            gap = 0.5 * (1.0 - e.pulse_duty) * e.pulse_period if e.pulse_period > 0 else math.inf
            for n in np.unique(noz):
                mm = noz == n
                order = np.argsort(-birth[mm], kind="stable")
                p, b = pos[mm][order], birth[mm][order]
                cuts = np.flatnonzero(np.abs(np.diff(b)) > gap) + 1 if math.isfinite(gap) and gap > 0 else []
                for seg in np.split(p, cuts):
                    if len(seg) >= 2:
                        out.append(seg)
        return out

    def count(self) -> int:
        return int((self.alive > 0).sum())


__all__ = [
    "DEFAULT_PARTICLES_PER_EMITTER",
    "EmitterSource",
    "ParticleSystem3D",
    "apply_smoke_sources",
    "build_smoke_sources",
    "emitter_params",
    "footprint_function",
    "legacy_params",
    "release_switch",
]
