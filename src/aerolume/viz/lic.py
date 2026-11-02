"""Line integral convolution (LIC): a dense "oil flow" texture of a velocity field.

LIC (Cabral & Leedom 1993) smears a fixed white-noise image along the
streamlines: every output pixel averages the noise over a short streamline
segment through it. Pixels on the same streamline average almost the same
noise, so the image shows fine streaks aligned with the flow everywhere --
more detail than discrete streamlines, no seeding decisions, and stable from
frame to frame because the noise is fixed (cached per size, seed and device).

The implementation advances *all* pixels at once: each integration step is a
``grid_sample`` of a precomputed displacement field (unit direction scaled to
``step`` pixels) followed by a ``grid_sample`` of the noise. Forward and
backward halves are stacked into one batch so every step is two kernels at
twice the image size: 1024x512 with the default 15-pixel half-length takes
5-9 ms on an RTX 4060 Ti, but about 1 s on a CPU, so on CPU-only machines use
it on a reduced ``shape`` or not at all. Directions are normalised before sampling, which makes the texture
independent of speed -- colour already shows speed.

Orientation follows the solver (row 0 = ymin); pass the result to
``render_field(..., shade=tex)`` to modulate a colour map on the device, or
use :func:`overlay` on an already rendered image.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_NOISE: dict[tuple[int, int, int, str], torch.Tensor] = {}


def _as_tensor(x: Any) -> torch.Tensor:
    t = x.detach() if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    return t.float()


def noise(shape: tuple[int, int], seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    """Cached white noise in 0..1 as (1, 1, H, W) on ``device``.

    Generated on the CPU with a seeded generator so the pattern is identical on
    every device (screenshots from a CPU and a GPU machine match).
    """
    dev = torch.device(device)
    key = (int(shape[0]), int(shape[1]), int(seed), str(dev))
    t = _NOISE.get(key)
    if t is None:
        g = torch.Generator().manual_seed(int(seed))
        t = torch.rand(1, 1, int(shape[0]), int(shape[1]), generator=g).to(dev)
        if len(_NOISE) > 16:
            _NOISE.clear()
        _NOISE[key] = t
    return t


def lic(u: Any, v: Any, *, extent: tuple[float, float, float, float] | None = None,
        shape: tuple[int, int] | None = None, length: float = 15.0, step: float = 1.0,
        mask: Any = None, seed: int = 0, contrast: float = 0.2) -> torch.Tensor:
    """LIC texture of ``(u, v)`` as an (H, W) float32 tensor in 0..1, on the field's device.

    ``u``, ``v`` are (ny, nx) in solver orientation; ``shape`` = (H, W) of the
    output (default: the field's own size; the velocity is sampled
    bilinearly, so a larger shape gives a finer texture). ``length`` is the
    half-length of the convolution in output pixels and ``step`` the
    integration step in pixels. ``extent`` is only needed when the cells are
    not square (it fixes the pixel aspect ratio). The raw LIC is re-centred to
    mean 0.5 with standard deviation ``contrast`` over the fluid (LIC of white
    noise loses contrast like 1/sqrt(length), so without this longer kernels
    would fade to flat grey); ``contrast=0`` returns the raw average. Solids
    (``mask > 0.5``) come out as neutral 0.5.
    """
    U = _as_tensor(u)
    V = _as_tensor(v).to(U.device)
    if U.ndim != 2 or U.shape != V.shape:
        raise ValueError(f"u and v must be equal 2D arrays, got {tuple(U.shape)} and {tuple(V.shape)}")
    dev = U.device
    ny, nx = U.shape
    H, W = (int(shape[0]), int(shape[1])) if shape is not None else (ny, nx)

    # Velocity in output-pixel units, so "unit direction" means one pixel.
    if extent is not None:
        xmin, xmax, ymin, ymax = (float(e) for e in extent)
        pw, ph = (xmax - xmin) / W, (ymax - ymin) / H
    else:
        pw, ph = nx / W, ny / H                       # square cells assumed
    solid = None
    if mask is not None:
        solid = _as_tensor(mask).to(dev) > 0.5
        U = torch.where(solid, 0.0, U)
        V = torch.where(solid, 0.0, V)
    du, dv = torch.nan_to_num(U / pw), torch.nan_to_num(V / ph)
    mag = torch.sqrt(du * du + dv * dv)
    inv = torch.where(mag > 0, 1.0 / mag.clamp_min(1e-30), torch.zeros_like(mag))   # stagnant: stay put
    # Displacement per step in grid_sample's normalised coordinates
    # (the image spans 2 units across W and H pixels).
    disp = torch.stack([du * inv * (2.0 * step / W), dv * inv * (2.0 * step / H)])[None]  # (1,2,ny,nx)

    # Pixel centres, forward half on top of backward half: (1, 2H, W, 2).
    gy, gx = torch.meshgrid(torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=dev),
                            torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=dev), indexing="ij")
    base = torch.stack([gx, gy], dim=-1)
    pos = torch.cat([base, base], dim=0)[None]
    sign = torch.cat([torch.ones(H, 1, 1, device=dev), -torch.ones(H, 1, 1, device=dev)])[None]

    nz = noise((H, W), seed, dev)
    n_steps = max(1, math.ceil(length / step))
    acc = nz[0, 0].clone()                             # k = 0 sample, weight 1
    wsum = 1.0
    for k in range(1, n_steps + 1):
        d = F.grid_sample(disp, pos, mode="bilinear", padding_mode="border", align_corners=False)
        pos = pos + sign * d[0].permute(1, 2, 0)[None]
        s = F.grid_sample(nz, pos, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
        w = 0.5 * (1.0 + math.cos(math.pi * k / (n_steps + 1)))     # Hann taper
        acc = acc + w * (s[:H] + s[H:])
        wsum += 2.0 * w
    tex = acc / wsum

    solid_out = None
    if solid is not None:
        solid_out = solid if (H, W) == (ny, nx) else F.interpolate(
            solid[None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0.5
    if contrast > 0:
        fluid = ~solid_out if solid_out is not None else torch.ones_like(tex, dtype=torch.bool)
        cnt = fluid.sum().clamp_min(1)
        mean = (tex * fluid).sum() / cnt
        std = torch.sqrt((((tex - mean) * fluid) ** 2).sum() / cnt).clamp_min(1e-6)
        tex = (0.5 + (tex - mean) * (float(contrast) / std)).clamp(0.0, 1.0)
    if solid_out is not None:
        tex = torch.where(solid_out, 0.5, tex)
    return tex


#: Package-level name (``aerolume.viz.lic_texture``); ``aerolume.viz.lic`` is this module.
lic_texture = lic


def overlay(rgba: np.ndarray, tex: Any, strength: float = 0.5) -> np.ndarray:
    """Modulate an already rendered (H, W, 4) uint8 image (row 0 = top) by a LIC texture.

    ``tex`` is in solver orientation (row 0 = ymin) and must have the image's
    size; brightness is scaled by ``1 + strength * (2 tex - 1)``. Prefer
    ``render_field(..., shade=tex)``, which does the same on the device.
    """
    t = tex.detach().float().cpu().numpy() if isinstance(tex, torch.Tensor) else np.asarray(tex, np.float32)
    t = np.clip(t[::-1], 0.0, 1.0)
    if t.shape != rgba.shape[:2]:
        raise ValueError(f"texture shape {t.shape} != image shape {rgba.shape[:2]}")
    gain = 1.0 + float(strength) * (2.0 * t - 1.0)
    out = rgba.copy()
    out[..., :3] = np.clip(rgba[..., :3].astype(np.float32) * gain[..., None] + 0.5, 0, 255).astype(np.uint8)
    return out
