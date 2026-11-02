"""Line integral convolution (aerolume.viz.lic)."""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from aerolume.viz import lic_texture, render_field
from aerolume.viz.lic import overlay

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def streakiness(tex: np.ndarray) -> tuple[float, float]:
    """Mean |difference| between horizontal and between vertical neighbours."""
    return float(np.abs(np.diff(tex, axis=1)).mean()), float(np.abs(np.diff(tex, axis=0)).mean())


def test_cpu_lic_streaks_follow_the_flow():
    ny, nx = 64, 128
    tex = lic_texture(np.ones((ny, nx)), np.zeros((ny, nx)), length=10)
    assert tex.shape == (ny, nx) and tex.dtype == torch.float32 and tex.device.type == "cpu"
    t = tex.numpy()
    assert t.min() >= 0 and t.max() <= 1
    assert abs(t.mean() - 0.5) < 0.05 and 0.12 < t.std() < 0.25        # contrast-normalised
    along, across = streakiness(t)
    assert along < 0.4 * across                                       # horizontal streaks
    tv = lic_texture(np.zeros((ny, nx)), np.ones((ny, nx)), length=10).numpy()
    along_v, across_v = streakiness(tv)
    assert across_v < 0.4 * along_v                                   # vertical streaks


def test_lic_is_stable_between_calls_and_masks_solids():
    ny, nx = 48, 96
    u, v = np.ones((ny, nx)), np.full((ny, nx), 0.3)
    mask = np.zeros((ny, nx))
    mask[10:30, 20:40] = 1.0
    a = lic_texture(u, v, mask=mask)
    b = lic_texture(u, v, mask=mask)
    assert torch.equal(a, b)                                          # fixed noise: no flicker
    assert torch.all(a[mask > 0.5] == 0.5)
    c = lic_texture(u, v, shape=(96, 192))
    assert c.shape == (96, 192)


def test_overlay_and_render_shade_agree():
    ny, nx = 32, 64
    u, v = np.ones((ny, nx)), np.zeros((ny, nx))
    tex = lic_texture(u, v)
    f = np.linspace(0, 1, nx)[None].repeat(ny, 0)
    plain = render_field(f, "viridis", 0, 1)
    a = overlay(plain, tex, 0.4)
    b = render_field(f, "viridis", 0, 1, shade=tex, shade_strength=0.4)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    with pytest.raises(ValueError):
        overlay(plain, tex[:10], 0.4)


@pytest.mark.gpu
@cuda
def test_gpu_lic_matches_cpu_and_is_fast():
    ny, nx = 512, 1024
    g = torch.Generator().manual_seed(0)
    u = 1 + 0.3 * torch.randn(ny, nx, generator=g)
    v = 0.3 * torch.randn(ny, nx, generator=g)
    u = torch.nn.functional.avg_pool2d(u[None, None], 9, 1, 4)[0, 0]
    v = torch.nn.functional.avg_pool2d(v[None, None], 9, 1, 4)[0, 0]
    small_cpu = lic_texture(u[:64, :128], v[:64, :128])
    small_gpu = lic_texture(u[:64, :128].cuda(), v[:64, :128].cuda())
    assert small_gpu.device.type == "cuda"
    assert (small_cpu - small_gpu.cpu()).abs().mean() < 1e-3

    uc, vc = u.cuda(), v.cuda()
    for _ in range(3):                                                # warm-up (noise, kernels, GPU clocks)
        lic_texture(uc, vc)
    times = []
    for _ in range(7):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lic_texture(uc, vc)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    # best of 7: the cost of the kernel itself. In the full suite the GPU is still clocking up or
    # shared with the desktop, and the first timed calls read 80 ms against a 3 ms best (measured)
    assert min(times) < 0.030, times
