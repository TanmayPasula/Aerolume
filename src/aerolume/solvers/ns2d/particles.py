"""GPU tracer particles released from emitters: streaklines and timelines.

DESIGN.md 10.1: particle emitters release massless tracers from their nozzles;
a continuous release draws *streaklines* (the locus of everything that passed
a nozzle - what a smoke wire photograph shows), a pulsed release draws
*timelines* (one batch per period from every nozzle of a line or rake, the
batch then deforms with the flow).

* **Pool and recycling.** Every nozzle owns a fixed ring of ``L`` slots in one
  pool of ``capacity`` particles. A release writes the nozzle's next slot, so the
  oldest particle of that nozzle is recycled; a particle that leaves the domain
  or enters a solid is marked dead and its slot waits for its turn. The pool
  never grows, so the memory is fixed and the whole update - advection, kill,
  release - is device arithmetic with static shapes: it runs inside the
  solver's CUDA graphs (no host synchronisation, no allocation).
* **Advection.** Midpoint (RK2) in time with the end-of-step velocity, sampled
  bilinearly from the staggered faces where u and v live (no averaging pass).
  Second order in space and time for a frozen field; the field changes by
  O(dt) per step, so the path error over a step is O(dt^2), consistent.
* **Release clock.** Continuous emitters accumulate ``rate / nozzles * dt`` on the
  device and release one particle from every nozzle each time the accumulator
  passes an integer - up to :data:`MAX_RELEASE` per nozzle and step, each placed
  along the local velocity by the time since its threshold crossing, so the
  streakline stays evenly spaced and the released number is ``rate * t`` to
  within one particle (a faster rate is capped there; the planner warns). Pulsed
  emitters release once at the start of every period's on-window.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import EmitterSpec

#: most particles a continuous emitter nozzle releases in one time step
MAX_RELEASE = 4


def sample_velocity(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, ops
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (u, v) at points from the staggered faces (points clamped to the grid)."""
    return _bilinear(u, x, y, ops, 0.0, 0.5), _bilinear(v, x, y, ops, 0.5, 0.0)


def _bilinear(f: torch.Tensor, x: torch.Tensor, y: torch.Tensor, ops, xoff: float, yoff: float) -> torch.Tensor:
    ny, nx = f.shape
    fx = ((x - ops.x0) / ops.h - xoff).clamp(0.0, nx - 1.0)
    fy = ((y - ops.y0) / ops.h - yoff).clamp(0.0, ny - 1.0)
    i0 = fx.floor().clamp(max=nx - 2).long()
    j0 = fy.floor().clamp(max=ny - 2).long()
    tx, ty = fx - i0, fy - j0
    flat = f.reshape(-1)
    k = j0 * nx + i0
    f00, f01 = flat[k], flat[k + 1]
    f10, f11 = flat[k + nx], flat[k + nx + 1]
    return (1 - ty) * ((1 - tx) * f00 + tx * f01) + ty * ((1 - tx) * f10 + tx * f11)


class TracerParticles:
    """A fixed pool of tracer particles fed by particle emitters (see the module docstring)."""

    def __init__(self, emitters: list[EmitterSpec], ops, capacity: int = 32768):
        self.ops = ops
        dev, dt = ops.device, ops.dtype
        self.specs = [e for e in emitters if e.species == "particles"]
        noz, owner = [], []
        for k, e in enumerate(self.specs):
            for p in e.nozzles():
                noz.append(p)
                owner.append(k)
        self.n_nozzles = len(noz)
        if self.n_nozzles == 0:
            raise ValueError("no particle emitters")
        self.ring = max(4, int(capacity) // self.n_nozzles)
        P = self.ring * self.n_nozzles
        self.capacity = P
        self.pos = torch.zeros(P, 2, device=dev, dtype=dt)
        self.birth = torch.zeros(P, device=dev, dtype=dt)
        self.alive = torch.zeros(P, device=dev, dtype=dt)
        self.noz_pos = torch.as_tensor(np.asarray(noz, dtype=np.float64), device=dev, dtype=dt).reshape(-1, 2)
        self.noz_owner = torch.as_tensor(owner, device=dev, dtype=torch.long)
        self.base = torch.arange(self.n_nozzles, device=dev, dtype=torch.long) * self.ring
        self.head = torch.zeros(self.n_nozzles, device=dev, dtype=torch.long)
        self.slot_nozzle = torch.arange(P, device=dev, dtype=torch.long) // self.ring
        counts = np.bincount(np.asarray(owner), minlength=len(self.specs)).astype(np.float64)
        self.rate = torch.as_tensor([e.rate / c for e, c in zip(self.specs, counts)], device=dev, dtype=dt)
        self.timing = torch.as_tensor([[e.start, e.stop if e.stop is not None else 1e30, e.pulse_period,
                                        e.pulse_duty] for e in self.specs], device=dev, dtype=dt)
        self.pulsed = (self.timing[:, 2] > 0).to(dt)
        self.acc = torch.zeros(len(self.specs), device=dev, dtype=dt)
        self.last = torch.full((len(self.specs),), -1.0, device=dev, dtype=dt)
        self._layout = tuple(len(e.nozzles()) for e in self.specs)

    # ------------------------------------------------------------------ state
    def buffers(self) -> list[torch.Tensor]:
        """Mutable device buffers (snapshotted around CUDA-graph warm-up)."""
        return [self.pos, self.birth, self.alive, self.head, self.acc, self.last]

    def clear(self) -> None:
        for b in (self.pos, self.birth, self.alive, self.head, self.acc):
            b.zero_()
        self.last.fill_(-1.0)

    def update_emitters(self, emitters: list[EmitterSpec]) -> bool:
        """Move / retime the emitters in place; False when the nozzle layout changed
        (the caller then builds a new pool)."""
        specs = [e for e in emitters if e.species == "particles"]
        if tuple(len(e.nozzles()) for e in specs) != self._layout:
            return False
        noz = np.vstack([e.nozzles() for e in specs])
        dev, dt = self.pos.device, self.pos.dtype
        self.noz_pos.copy_(torch.as_tensor(noz, device=dev, dtype=dt))
        counts = np.asarray(self._layout, dtype=np.float64)
        self.rate.copy_(torch.as_tensor([e.rate / c for e, c in zip(specs, counts)], device=dev, dtype=dt))
        self.timing.copy_(torch.as_tensor([[e.start, e.stop if e.stop is not None else 1e30, e.pulse_period,
                                            e.pulse_duty] for e in specs], device=dev, dtype=dt))
        self.pulsed.copy_((self.timing[:, 2] > 0).to(dt))
        self.specs = specs
        return True

    # ------------------------------------------------------------------- step
    def step(self, u: torch.Tensor, v: torch.Tensor, dt: torch.Tensor, t: torch.Tensor,
             solid: torch.Tensor | None) -> None:
        """Advect, kill and release (in place; graph-capturable). ``t`` is the time
        at the end of the step, ``solid`` an optional (ny, nx) solid fraction."""
        ops = self.ops
        x, y = self.pos[:, 0], self.pos[:, 1]
        u1, v1 = sample_velocity(u, v, x, y, ops)
        xm, ym = x + (0.5 * dt) * u1, y + (0.5 * dt) * v1
        u2, v2 = sample_velocity(u, v, xm, ym, ops)
        xn, yn = x + dt * u2, y + dt * v2
        x1, y1 = ops.x0 + ops.nx * ops.h, ops.y0 + ops.ny * ops.h
        inside = (xn > ops.x0) & (xn < x1) & (yn > ops.y0) & (yn < y1)
        if solid is not None:
            i = ((xn - ops.x0) / ops.h).clamp(0, ops.nx - 1).long()
            j = ((yn - ops.y0) / ops.h).clamp(0, ops.ny - 1).long()
            inside = inside & (solid.reshape(-1)[j * ops.nx + i] <= 0.5)
        alive = self.alive * inside.to(self.alive.dtype)
        keep = alive > 0
        self.pos.copy_(torch.where(keep[:, None], torch.stack([xn, yn], 1), self.pos))
        self.alive.copy_(alive)
        # release clock (device arithmetic)
        start, stop, period, duty = self.timing.unbind(1)
        active = (t >= start) & (t < stop)
        safe = torch.where(period > 0, period, torch.ones_like(period))
        k = torch.floor((t - start) / safe)
        phase = (t - start) / safe - k
        on_cont = active.to(dt.dtype) * (1.0 - self.pulsed)
        acc = self.acc + dt * self.rate * on_cont
        n_cont = torch.floor(acc).clamp(max=float(MAX_RELEASE))
        # keep only the fractional part: a backlog beyond MAX_RELEASE per step is dropped
        self.acc.copy_(acc - torch.floor(acc))
        rel_p = (self.pulsed > 0) & active & (k > self.last) & (phase < duty)
        self.last.copy_(torch.where(rel_p, k, self.last))
        # released particles start along the local velocity by the time since their
        # threshold crossing, so a fast continuous release draws an evenly spaced streakline
        un, vn = sample_velocity(u, v, self.noz_pos[:, 0], self.noz_pos[:, 1], ops)
        vel_n = torch.stack([un, vn], 1)
        safe_rate = self.rate.clamp_min(1e-30)
        for r in range(MAX_RELEASE):
            if r == 0:
                rel = torch.where(self.pulsed > 0, rel_p, n_cont > 0)
            else:
                rel = (self.pulsed <= 0) & (n_cont > r)
            age = torch.where(self.pulsed > 0, torch.zeros_like(acc), ((acc - (r + 1.0)) / safe_rate).clamp(min=0.0))
            age = torch.minimum(age, dt.expand_as(age))
            rel_n, age_n = rel[self.noz_owner], age[self.noz_owner]
            slot = self.base + self.head
            start_pos = self.noz_pos + vel_n * age_n[:, None]
            self.pos.index_copy_(0, slot, torch.where(rel_n[:, None], start_pos, self.pos[slot]))
            self.birth.index_copy_(0, slot, torch.where(rel_n, (t - age_n).to(self.birth.dtype), self.birth[slot]))
            self.alive.index_copy_(0, slot, torch.where(rel_n, torch.ones_like(self.alive[slot]), self.alive[slot]))
            self.head.copy_(torch.remainder(self.head + rel_n.long(), self.ring))

    # ---------------------------------------------------------------- outputs
    def snapshot(self, t: float) -> dict[str, np.ndarray]:
        """Living particles on the host: x, y, age, nozzle, emitter (one sync)."""
        alive = self.alive.detach().cpu().numpy() > 0
        pos = self.pos.detach().cpu().double().numpy()[alive]
        birth = self.birth.detach().cpu().double().numpy()[alive]
        noz = self.slot_nozzle.cpu().numpy()[alive]
        owner = self.noz_owner.cpu().numpy()[noz]
        return {"x": pos[:, 0], "y": pos[:, 1], "age": t - birth, "nozzle": noz, "emitter": owner,
                "birth": birth}

    def lines(self, t: float) -> list[np.ndarray]:
        """Polylines for drawing: per nozzle the streakline (continuous emitters,
        ordered from the nozzle outwards) and per release the timeline (pulsed
        emitters, ordered along the nozzles). Each line is an (n, 2) array."""
        s = self.snapshot(t)
        out: list[np.ndarray] = []
        pulsed = self.pulsed.cpu().numpy() > 0
        for n in np.unique(s["nozzle"]):
            m = s["nozzle"] == n
            if pulsed[int(s["emitter"][m][0])]:
                continue
            order = np.argsort(s["age"][m])
            if order.size >= 2:
                out.append(np.c_[s["x"][m][order], s["y"][m][order]])
        for e in np.flatnonzero(pulsed):
            m = s["emitter"] == e
            for b in np.unique(np.round(s["birth"][m], 9)):
                mm = m & (np.abs(s["birth"] - b) < 1e-7 + 1e-9 * abs(b))
                order = np.argsort(s["nozzle"][mm])
                if order.size >= 2:
                    out.append(np.c_[s["x"][mm][order], s["y"][mm][order]])
        return out


def particle_emitters(emitters: list[EmitterSpec]) -> bool:
    return any(e.species == "particles" for e in emitters)


__all__ = ["MAX_RELEASE", "TracerParticles", "particle_emitters", "sample_velocity"]

