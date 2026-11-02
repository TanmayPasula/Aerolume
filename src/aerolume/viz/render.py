"""Field -> RGBA image rendering and robust, flicker-free colour ranges.

The live view redraws a 1024x512 field 20-60 times a second while the solver
keeps the GPU busy, so the expensive part -- normalise, look up the colour,
blend the solid bodies -- runs as a handful of torch ops on whatever device
the field already lives on. Only the finished (H, W, 4) uint8 image crosses to
the host (2 MB at 1024x512), never the float field.

Orientation: solver fields are indexed ``[j, i]`` with row 0 at the *bottom*
(y = ymin, DESIGN.md 5.2). Images follow the screen convention, row 0 at the
*top* (DESIGN.md 5.4 ``LiveFrame.rgba``), so :func:`render_field` flips rows.

Colour ranges: a raw min/max is dominated by a few cells next to a sharp
corner or inside the immersed boundary, which washes the whole picture into
one colour. :func:`auto_range` therefore uses percentiles of a fixed lattice
subsample of the fluid cells, and blends towards the new range exponentially
so the colours do not pump from frame to frame.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .colormaps import DYE_BACKGROUND, DYE_COLORS, LUT_SIZE, get_lut, hex_rgb

#: Neutral fill for solids: mid grey reads on both themes and on every map.
SOLID_RGBA: tuple[int, int, int, int] = (150, 155, 163, 255)
#: Thin darker outline drawn where the solid fraction crosses 1/2.
EDGE_RGBA: tuple[int, int, int, int] = (52, 56, 64, 255)

_LUT_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _as_tensor(x: Any) -> torch.Tensor:
    """A float tensor view of ``x`` on its own device (numpy -> CPU, no copy when possible)."""
    if isinstance(x, torch.Tensor):
        t = x.detach()
    else:
        t = torch.from_numpy(np.ascontiguousarray(x))
    if not t.is_floating_point():
        t = t.float()
    elif t.dtype not in (torch.float32, torch.float64):
        t = t.float()                       # half / bfloat16: do the maths in f32
    return t


def _lut(name: str | None, device: torch.device) -> torch.Tensor:
    """(256, 4) uint8 LUT (alpha 255) cached per map and device."""
    key = (str(name), str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        rgb = get_lut(name)
        rgba = np.concatenate([rgb, np.full((LUT_SIZE, 1), 255, np.uint8)], axis=1)
        lut = torch.from_numpy(rgba).to(device)
        _LUT_CACHE[key] = lut
    return lut


def _soft_mask(m: torch.Tensor) -> torch.Tensor:
    """Solid fraction with a one-cell anti-aliasing ramp.

    The solver's mask is already fractional across the boundary cells, and is
    used as is. A binary mask (every value 0 or 1, e.g. rasterised from a
    polygon) would give a staircase outline, so it gets a 3x3 box filter, which
    turns each step into a one-cell ramp centred on the true edge.
    """
    partial = ((m > 0.02) & (m < 0.98)).any()
    if bool(partial):
        return m
    blurred = F.avg_pool2d(F.pad(m[None, None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return blurred[0, 0]


def render_field(field: Any, cmap: str | None = "viridis", vmin: float | None = None,
                 vmax: float | None = None, *, mask: Any = None,
                 solid_rgba: tuple[int, int, int, int] = SOLID_RGBA,
                 edge_rgba: tuple[int, int, int, int] | None = EDGE_RGBA,
                 log: bool = False, shade: Any = None, shade_strength: float = 0.5,
                 nan_rgba: tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
    """Colour-map a 2D field into an (H, W, 4) uint8 image with row 0 = top.

    ``field`` is a (ny, nx) torch tensor or numpy array in solver orientation
    (row 0 = ymin); all work happens on its device and only the uint8 image is
    copied back. Values are mapped linearly (or by log10 with ``log=True``)
    from ``[vmin, vmax]`` onto the LUT and clipped; ``None`` limits come from
    :func:`auto_range`. ``mask`` is the solid fraction (0 fluid .. 1 solid):
    solids are painted ``solid_rgba`` blended by that fraction, which gives an
    anti-aliased silhouette, plus an optional outline (``edge_rgba``) peaking
    where the fraction is 1/2. ``shade`` is an optional (ny, nx) texture in
    0..1 (e.g. :func:`aerolume.viz.lic.lic`) that modulates brightness by
    +-``shade_strength`` around 0.5 before the solids are composited.
    Non-finite values are drawn ``nan_rgba`` (transparent by default).
    """
    f = _as_tensor(field)
    if f.ndim != 2:
        raise ValueError(f"render_field expects a 2D field, got shape {tuple(f.shape)}")
    dev = f.device
    if vmin is None or vmax is None:
        lo, hi = auto_range(f, mask=mask, log=log)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    vmin, vmax = float(vmin), float(vmax)

    if log:
        if vmax <= 0.0:
            vmax = 1.0
        if vmin <= 0.0:
            vmin = vmax * 1e-4                  # four decades below the top
        f = torch.log10(f.clamp_min(vmin * 1e-3))
        vmin, vmax = math.log10(vmin), math.log10(vmax)
    span = vmax - vmin
    if not (math.isfinite(span) and span > 0.0):
        # Degenerate range: show the field in the middle colour instead of
        # dividing by zero (a uniform free stream at t = 0 hits this).
        scale, offset = 0.0, LUT_SIZE / 2
    else:
        scale, offset = (LUT_SIZE - 1) / span, 0.5

    finite = torch.isfinite(f)
    # Non-finite cells get their own colour below; give them a harmless index
    # here (inf * 0 would be nan, and nan -> long is undefined).
    big = 1e30 if f.dtype == torch.float32 else 1e300
    x = torch.nan_to_num(f, nan=0.0, posinf=big, neginf=-big)
    idx = ((x - vmin) * scale + offset).clamp_(0.0, LUT_SIZE - 1).long()
    # Gather straight into uint8: the plain case never touches floats per pixel.
    rgba = _lut(cmap, dev)[idx]                          # (ny, nx, 4) uint8
    rgba = torch.where(finite[..., None], rgba,
                       torch.tensor(nan_rgba, dtype=torch.uint8, device=dev))

    if shade is not None:
        s = _as_tensor(shade).to(device=dev, dtype=torch.float32)
        if s.shape != f.shape:
            s = F.interpolate(s[None, None], size=tuple(f.shape), mode="bilinear",
                              align_corners=False)[0, 0]
        gain = 1.0 + float(shade_strength) * (2.0 * s.clamp(0.0, 1.0) - 1.0)
        rgb = (rgba[..., :3].float() * gain[..., None] + 0.5).clamp_(0.0, 255.0)
        rgba = torch.cat([rgb.to(torch.uint8), rgba[..., 3:]], dim=-1)

    if mask is not None:
        m = _as_tensor(mask).to(device=dev, dtype=torch.float32)
        if m.shape != f.shape:
            raise ValueError(f"mask shape {tuple(m.shape)} != field shape {tuple(f.shape)}")
        m = m.clamp(0.0, 1.0)
        # Composite only inside the bodies' bounding box (+1 cell for the
        # anti-aliasing ramp): bodies cover a few per cent of the domain, and on
        # the CPU the float blend is the dominant cost.
        rows = torch.nonzero(m.amax(dim=1) > 0.0).flatten()
        cols = torch.nonzero(m.amax(dim=0) > 0.0).flatten()
        if rows.numel():
            r0, r1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, m.shape[0])
            c0, c1 = max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, m.shape[1])
            mb = _soft_mask(m[r0:r1, c0:c1])
            blk = rgba[r0:r1, c0:c1].float()
            solid = torch.tensor(solid_rgba, dtype=torch.float32, device=dev)
            w = (mb * (solid[3] / 255.0))[..., None]
            blk = blk * (1.0 - w) + solid * w
            blk[..., 3:] = torch.maximum(blk[..., 3:], (mb * 255.0)[..., None])
            if edge_rgba is not None:
                edge = torch.tensor(edge_rgba, dtype=torch.float32, device=dev)
                e = ((4.0 * mb * (1.0 - mb)).clamp(0.0, 1.0) * (edge[3] / 255.0))[..., None]
                blk = blk * (1.0 - e) + edge * e
            rgba[r0:r1, c0:c1] = (blk + 0.5).clamp_(0.0, 255.0).to(torch.uint8)

    return rgba.flip(0).contiguous().cpu().numpy()


def render_dye(channels: Any, colors: Any = None, *, mask: Any = None,
               background: tuple[int, int, int] = DYE_BACKGROUND, gain: float = 1.0,
               solid_rgba: tuple[int, int, int, int] = SOLID_RGBA,
               edge_rgba: tuple[int, int, int, int] | None = EDGE_RGBA) -> np.ndarray:
    """Composite up to four smoke channels into one (H, W, 4) uint8 picture, row 0 = top.

    A smoke tunnel shows each emitter's smoke in its own colour against a dark
    background (DESIGN.md 10.1). Each channel (``dye``, ``dye1`` ...: (ny, nx),
    0..1, solver orientation) is laid *over* the picture in its colour with
    opacity ``clip(gain * c, 0, 1)``, in channel order: where two smokes mix
    the later one dominates in proportion to its concentration, which keeps
    both colours legible (an additive mix of orange and blue would read as an
    unrelated white). ``channels`` may be a sequence of arrays / tensors or a
    (C, ny, nx) stack; ``colors`` one '#rrggbb' (or rgb tuple) per channel,
    :data:`~aerolume.viz.colormaps.DYE_COLORS` by default. Solids are painted
    like :func:`render_field` paints them. Runs on the channels' device; only
    the uint8 image is copied back.
    """
    if isinstance(channels, (list, tuple)):
        chans = [_as_tensor(c) for c in channels]
        if not chans:
            raise ValueError("render_dye needs at least one channel")
        dev = chans[0].device
        stack = torch.stack([c.to(device=dev, dtype=torch.float32) for c in chans])
    else:
        stack = _as_tensor(channels).to(torch.float32)
        if stack.ndim == 2:
            stack = stack[None]
        dev = stack.device
    if stack.ndim != 3:
        raise ValueError(f"render_dye expects (C, ny, nx) channels, got shape {tuple(stack.shape)}")
    n = stack.shape[0]
    cols = list(colors) if colors is not None else list(DYE_COLORS)
    if len(cols) < n:
        cols += list(DYE_COLORS[len(cols):]) + [DYE_COLORS[-1]] * n
    rgb = torch.tensor([hex_rgb(c) for c in cols[:n]], dtype=torch.float32, device=dev)   # (C, 3)
    out = torch.empty(stack.shape[1:] + (3,), dtype=torch.float32, device=dev)
    out[:] = torch.tensor(background, dtype=torch.float32, device=dev)
    a = (torch.nan_to_num(stack, nan=0.0) * float(gain)).clamp_(0.0, 1.0)
    for k in range(n):
        out = torch.lerp(out, rgb[k].expand_as(out), a[k][..., None])
    rgba = torch.cat([(out + 0.5).clamp_(0.0, 255.0).to(torch.uint8),
                      torch.full(out.shape[:2] + (1,), 255, dtype=torch.uint8, device=dev)], dim=-1)
    if mask is not None:
        m = _as_tensor(mask).to(device=dev, dtype=torch.float32).clamp(0.0, 1.0)
        if m.shape != rgba.shape[:2]:
            raise ValueError(f"mask shape {tuple(m.shape)} != channel shape {tuple(rgba.shape[:2])}")
        mb = _soft_mask(m)
        solid = torch.tensor(solid_rgba, dtype=torch.float32, device=dev)
        w = (mb * (solid[3] / 255.0))[..., None]
        blk = rgba.float() * (1.0 - w) + solid * w
        if edge_rgba is not None:
            edge = torch.tensor(edge_rgba, dtype=torch.float32, device=dev)
            e = ((4.0 * mb * (1.0 - mb)).clamp(0.0, 1.0) * (edge[3] / 255.0))[..., None]
            blk = blk * (1.0 - e) + edge * e
        rgba = (blk + 0.5).clamp_(0.0, 255.0).to(torch.uint8)
    return rgba.flip(0).contiguous().cpu().numpy()


def _lattice(t: torch.Tensor, max_samples: int) -> tuple[slice, ...]:
    """Strides that pick at most ~max_samples cells on a regular lattice.

    A 2D lattice (rather than a flat stride) covers every region of the field
    evenly and cannot alias with the row length into sampling one column.
    """
    if t.ndim != 2 or t.numel() <= max_samples:
        return tuple(slice(None) for _ in range(t.ndim))
    ratio = math.sqrt(t.numel() / max_samples)
    sy, sx = max(1, math.ceil(ratio)), max(1, math.ceil(ratio))
    return (slice(sy // 2, None, sy), slice(sx // 2, None, sx))


def auto_range(field: Any, *, diverging: bool = False, mask: Any = None,
               prev: tuple[float, float] | None = None, smoothing: float = 0.8,
               lo: float = 1.0, hi: float = 99.0, log: bool = False,
               max_samples: int = 65536, snap_ratio: float = 8.0) -> tuple[float, float]:
    """Robust (vmin, vmax) for displaying a field, smoothed against ``prev``.

    The target range is the ``lo``-th / ``hi``-th percentile of the fluid
    cells (mask < 1/2, finite values) on a lattice subsample of at most
    ``max_samples`` cells, so a handful of extreme cells cannot set the
    colours and the cost is fixed whatever the grid size. ``diverging=True``
    makes it symmetric about zero using the ``hi`` percentile of ``|f|``, so
    white/black always means zero on a diverging map. ``log=True`` uses only
    positive values (for log display).

    With ``prev`` the result moves a fraction ``1 - smoothing`` of the way
    from ``prev`` to the target, i.e. an exponential moving average over
    roughly ``1 / (1 - smoothing)`` frames; ``smoothing=0`` disables it. When
    the target is not comparable with ``prev`` -- no overlap, or a width
    different by more than ``snap_ratio`` (the user switched field, the run
    was reset) -- it snaps straight to the target instead of crawling there.
    Degenerate (constant) fields get a small non-zero width so callers can
    always divide by ``vmax - vmin``.
    """
    f = _as_tensor(field)
    sl = _lattice(f, max_samples)
    vals = f[sl].reshape(-1)
    keep = torch.isfinite(vals)
    if mask is not None:
        m = _as_tensor(mask).to(f.device)
        if m.shape == f.shape:
            keep &= m[sl].reshape(-1) < 0.5
    if log:
        keep &= vals > 0
    vals = vals[keep]
    if vals.numel() == 0:
        return (float(prev[0]), float(prev[1])) if prev is not None else ((1e-3, 1.0) if log else (0.0, 1.0))

    vals = vals.float()
    if diverging:
        a = float(torch.quantile(vals.abs(), hi / 100.0))
        t0, t1 = -a, a
    else:
        if log:
            vals = torch.log10(vals)
        q = torch.quantile(vals, torch.tensor([lo / 100.0, hi / 100.0], device=vals.device)).tolist()
        t0, t1 = (10.0 ** q[0], 10.0 ** q[1]) if log else (q[0], q[1])

    # Guarantee a usable width: constant fields (free stream at t = 0) otherwise
    # give vmin == vmax.
    width = t1 - t0
    floor = max(1e-3 * max(abs(t0), abs(t1)), 1e-12)
    if not width > floor:
        mid = 0.5 * (t0 + t1)
        t0, t1 = (mid - floor, mid + floor) if not diverging else (-floor, floor)
        if log:
            t0 = max(t0, mid * 0.999)

    if prev is None or smoothing <= 0.0:
        return float(t0), float(t1)
    p0, p1 = float(prev[0]), float(prev[1])
    pw, tw = p1 - p0, t1 - t0
    comparable = (math.isfinite(p0) and math.isfinite(p1) and pw > 0.0
                  and p0 < t1 and t0 < p1 and 1.0 / snap_ratio < tw / pw < snap_ratio)
    if not comparable:
        return float(t0), float(t1)
    a = min(max(float(smoothing), 0.0), 1.0)
    return float(a * p0 + (1 - a) * t0), float(a * p1 + (1 - a) * t1)
