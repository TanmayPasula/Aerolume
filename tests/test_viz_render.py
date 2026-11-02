"""Field rendering and robust colour ranges (aerolume.viz.render)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aerolume.viz import COLORMAPS
from aerolume.viz.render import EDGE_RGBA, SOLID_RGBA, auto_range, render_field

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ramp(ny=8, nx=5):
    """Field equal to its row index: row 0 (ymin) = 0, top row = ny - 1."""
    return np.repeat(np.arange(ny, dtype=np.float64)[:, None], nx, axis=1)


def test_orientation_row0_is_top_and_colours_are_exact():
    f = ramp(256, 3)
    img = render_field(f, "viridis", 0.0, 255.0)
    assert img.shape == (256, 3, 4) and img.dtype == np.uint8
    lut = COLORMAPS["viridis"]
    assert np.array_equal(img[0, 0, :3], lut[255])          # top row = largest y = high value
    assert np.array_equal(img[-1, 0, :3], lut[0])
    assert np.array_equal(img[::-1, 1, :3], lut)            # every level maps to its own entry
    assert (img[..., 3] == 255).all()


def test_values_outside_range_are_clipped():
    f = np.array([[-10.0, 0.0, 0.5, 1.0, 10.0]])
    img = render_field(f, "magma", 0.0, 1.0)
    lut = COLORMAPS["magma"]
    assert np.array_equal(img[0, 0, :3], lut[0]) and np.array_equal(img[0, 4, :3], lut[255])
    assert np.array_equal(img[0, 2, :3], lut[128])


def test_tensor_and_numpy_agree_and_auto_range_default():
    f = np.random.default_rng(1).normal(size=(40, 60))
    a = render_field(f, "turbo", -1, 1)
    b = render_field(torch.from_numpy(f).float(), "turbo", -1, 1)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    c = render_field(f, "turbo")                            # limits from auto_range
    assert c.shape == (40, 60, 4)


def test_mask_paints_solids_with_antialiased_edge():
    ny, nx = 40, 40
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    sdf = np.hypot(xx - 20, yy - 20) - 10                   # disc of radius 10 cells
    mask = np.clip(0.5 - sdf, 0, 1)                         # fractional, like the solver's
    img = render_field(np.zeros((ny, nx)), "viridis", -1, 1, mask=mask)
    flipped_mask = mask[::-1]
    assert np.array_equal(img[20, 20], np.array(SOLID_RGBA, np.uint8))          # deep inside
    assert np.array_equal(img[0, 0, :3], COLORMAPS["viridis"][128])               # far fluid
    edge = (flipped_mask > 0.3) & (flipped_mask < 0.7)
    assert edge.any()
    # Edge pixels are pulled towards the dark outline colour.
    lum = img[..., :3].astype(float).mean(-1)
    assert lum[edge].mean() < min(np.mean(SOLID_RGBA[:3]), COLORMAPS["viridis"][128].mean()) - 10
    no_edge = render_field(np.zeros((ny, nx)), "viridis", -1, 1, mask=mask, edge_rgba=None)
    assert no_edge[..., :3].astype(float).mean(-1)[edge].mean() > lum[edge].mean()
    assert EDGE_RGBA[3] == 255


def test_binary_mask_gets_a_soft_edge():
    mask = np.zeros((30, 30))
    mask[10:20, 10:20] = 1.0
    img = render_field(np.zeros((30, 30)), "grayscale", -1, 1, mask=mask, edge_rgba=None)
    row = img[15, :, 0].astype(int)                          # through the middle of the square
    fluid, solid = int(COLORMAPS["grayscale"][128, 0]), SOLID_RGBA[0]
    inter = [v for v in row if min(fluid, solid) < v < max(fluid, solid)]
    assert len(inter) >= 2                                   # a ramp, not a hard step


def test_nan_is_transparent_and_degenerate_range_is_mid_colour():
    f = np.ones((4, 4))
    f[1, 2] = np.nan
    img = render_field(f, "viridis", 1.0, 1.0)
    assert tuple(img[2, 2]) == (0, 0, 0, 0)                  # row 1 from the bottom of 4 -> row 2
    assert np.array_equal(img[0, 0, :3], COLORMAPS["viridis"][128])
    inf = render_field(np.array([[np.inf, 0.0]]), "viridis", 0.0, 1.0)
    assert inf[0, 0, 3] == 0 and inf[0, 1, 3] == 255


def test_log_scale():
    f = np.array([[1e-3, 1e-2, 1e-1, 1.0]])
    img = render_field(f, "grayscale", 1e-3, 1.0, log=True)
    assert list(img[0, :, 0]) == [0, 85, 170, 255]


def test_shade_modulates_brightness():
    f = np.zeros((10, 10))
    flat = render_field(f, "grayscale", -1, 1, shade=np.full((10, 10), 0.5))
    plain = render_field(f, "grayscale", -1, 1)
    assert np.array_equal(flat, plain)
    bright = render_field(f, "grayscale", -1, 1, shade=np.ones((10, 10)), shade_strength=0.5)
    assert bright[0, 0, 0] == round(128 * 1.5)


def test_auto_range_ignores_outliers_and_solids():
    rng = np.random.default_rng(0)
    f = rng.normal(1.0, 0.1, size=(300, 400))
    f.flat[rng.choice(f.size, 100, replace=False)] = 1e6     # spikes at sharp corners
    lo, hi = auto_range(f)
    assert 0.6 < lo < 0.9 and 1.1 < hi < 1.4
    mask = np.zeros_like(f)
    mask[100:200, 100:200] = 1.0
    g = f.copy()
    g[mask > 0.5] = -500.0                                    # junk inside the body
    lo2, _ = auto_range(g, mask=mask)
    assert lo2 > 0.5
    assert auto_range(torch.from_numpy(f)) == pytest.approx((lo, hi), rel=1e-5)


def test_auto_range_diverging_is_symmetric():
    f = np.random.default_rng(2).normal(0.3, 1.0, size=(200, 200))
    lo, hi = auto_range(f, diverging=True)
    assert lo == pytest.approx(-hi) and hi > 2.0


def test_auto_range_smoothing_and_snap():
    f = np.linspace(0, 2, 10_000).reshape(100, 100)          # target ~ (0.02, 1.98)
    lo, hi = auto_range(f, prev=(0.0, 1.0), smoothing=0.8)
    t_lo, t_hi = auto_range(f)
    assert hi == pytest.approx(0.8 * 1.0 + 0.2 * t_hi)
    assert lo == pytest.approx(0.2 * t_lo)
    assert auto_range(f, prev=(0.0, 1.0), smoothing=0.0) == pytest.approx((t_lo, t_hi))
    # Incomparable previous range (another field): snap straight to the target.
    assert auto_range(f, prev=(1000.0, 5000.0)) == pytest.approx((t_lo, t_hi))
    assert auto_range(f, prev=(0.0, 1e-3)) == pytest.approx((t_lo, t_hi))


def test_auto_range_smoothing_removes_flicker():
    rng = np.random.default_rng(3)
    base = np.linspace(0, 1, 4096).reshape(64, 64)
    raw, smooth, prev = [], [], None
    for _ in range(60):
        f = base * (1.0 + 0.1 * rng.standard_normal())       # frame-to-frame amplitude jitter
        raw.append(auto_range(f)[1])
        prev = auto_range(f, prev=prev, smoothing=0.8)
        smooth.append(prev[1])
    assert np.std(np.diff(smooth[10:])) < 0.35 * np.std(np.diff(raw[10:]))


def test_auto_range_degenerate_inputs():
    lo, hi = auto_range(np.full((10, 10), 3.0))
    assert lo < 3.0 < hi and hi - lo < 0.1
    assert auto_range(np.full((5, 5), np.nan)) == (0.0, 1.0)
    assert auto_range(np.full((5, 5), np.nan), prev=(2.0, 4.0)) == (2.0, 4.0)
    lo, hi = auto_range(np.zeros((10, 10)), diverging=True)
    assert lo < 0 < hi
    lo, hi = auto_range(np.logspace(-3, 0, 100).reshape(10, 10), log=True)
    assert 0 < lo < hi


@pytest.mark.gpu
@cuda
def test_gpu_render_matches_cpu():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(64, 96)).astype(np.float32)
    mask = np.zeros_like(f)
    mask[20:40, 30:50] = 1.0
    cpu = render_field(f, "icefire", -2, 2, mask=mask)
    gpu = render_field(torch.from_numpy(f).cuda(), "icefire", -2, 2, mask=torch.from_numpy(mask).cuda())
    assert np.abs(cpu.astype(int) - gpu.astype(int)).max() <= 1
    lo, hi = auto_range(torch.from_numpy(f).cuda(), mask=torch.from_numpy(mask).cuda())
    assert (lo, hi) == pytest.approx(auto_range(f, mask=mask), rel=1e-4)
