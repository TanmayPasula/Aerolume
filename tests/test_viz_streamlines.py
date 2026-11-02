"""Evenly spaced streamlines (aerolume.viz.streamlines)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aerolume.viz import streamlines


def grid(ext, nx, ny):
    x = np.linspace(ext[0], ext[1], nx + 1)
    y = np.linspace(ext[2], ext[3], ny + 1)
    return np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return float(d.min())


def test_uniform_flow_gives_parallel_evenly_spaced_lines():
    ext = (0.0, 4.0, -1.0, 1.0)
    X, _ = grid(ext, 200, 100)
    u, v = np.ones_like(X), np.zeros_like(X)
    lines = streamlines(u, v, ext, density=1.0)
    d_sep = 2.0 / 30.0
    assert 28 <= len(lines) <= 31                            # height / d_sep = 30
    ys = []
    for ln in lines:
        assert np.ptp(ln[:, 1]) < 1e-9                        # straight and horizontal
        assert ln[0, 0] == pytest.approx(0.0, abs=1e-6)       # inlet to outlet, downstream order
        assert ln[-1, 0] == pytest.approx(4.0, abs=1e-6)
        assert np.all(np.diff(ln[:, 0]) > 0)
        ys.append(ln[0, 1])
    gaps = np.diff(np.sort(ys))
    assert gaps.min() > 0.9 * d_sep and gaps.max() < 1.15 * d_sep


def test_density_scales_the_line_count():
    ext = (0.0, 4.0, -1.0, 1.0)
    X, _ = grid(ext, 200, 100)
    u, v = np.ones_like(X), np.zeros_like(X)
    n1 = len(streamlines(u, v, ext, density=1.0))
    n2 = len(streamlines(u, v, ext, density=2.0))
    assert 1.8 < n2 / n1 < 2.2
    assert len(streamlines(u, v, ext, spacing=0.5)) == 3       # y = -0.5, 0, 0.5 (+-1 is the wall)


def test_solid_body_rotation_gives_concentric_closed_circles():
    ext = (-1.0, 1.0, -1.0, 1.0)
    X, Y = grid(ext, 128, 128)
    u, v = -Y, X                                              # counter-clockwise rotation
    lines = streamlines(u, v, ext, density=1.0)
    d_sep = 2.0 / 30.0
    radii = []
    for ln in lines:
        r = np.hypot(ln[:, 0], ln[:, 1])
        if r.max() > 0.95:                                    # clipped by the square's corners
            continue
        assert np.ptp(r) < 0.02 * max(r.mean(), 0.1)          # stays on its circle
        length = np.hypot(*np.diff(ln, axis=0).T).sum()
        assert length < 2 * np.pi * r.mean() * 1.05           # one loop, no re-tracing
        assert length > 2 * np.pi * r.mean() * 0.80           # ... and nearly closed
        radii.append(r.mean())
    radii = np.sort(radii)
    assert len(radii) >= 10
    gaps = np.diff(radii)
    assert gaps.min() > 0.7 * d_sep and np.median(gaps) < 1.3 * d_sep


def test_lines_keep_their_distance():
    ext = (-1.0, 1.0, -1.0, 1.0)
    X, Y = grid(ext, 96, 96)
    lines = streamlines(-Y + 0.3, X, ext, density=0.7)
    d_test = 0.5 * 2.0 / (30 * 0.7)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # The integration stops within one raster pixel (d_sep/8) and one step of d_test.
            assert min_distance(lines[i], lines[j]) > 0.5 * d_test


def test_streamlines_avoid_solids():
    ext = (-2.0, 6.0, -2.0, 2.0)
    X, Y = grid(ext, 256, 128)
    a = 0.5
    R2 = X**2 + Y**2
    u = 1 - a * a * (X**2 - Y**2) / R2**2                     # potential flow past a cylinder
    v = -2 * a * a * X * Y / R2**2
    mask = (R2 < a * a).astype(float)
    u[mask > 0] = 0.0
    v[mask > 0] = 0.0
    lines = streamlines(torch.from_numpy(u), torch.from_numpy(v), ext, mask=mask)
    assert len(lines) > 20
    pts = np.vstack(lines)
    assert (np.hypot(pts[:, 0], pts[:, 1]) > 0.9 * a).all()
    # Every line runs downstream (potential flow has no recirculation).
    assert all(ln[-1, 0] > ln[0, 0] for ln in lines)


def test_degenerate_inputs():
    ext = (0.0, 1.0, 0.0, 1.0)
    z = np.zeros((32, 32))
    assert streamlines(z, z, ext) == []
    with pytest.raises(ValueError):
        streamlines(z, np.zeros((16, 16)), ext)
    with pytest.raises(ValueError):
        streamlines(z, z, (0.0, 0.0, 0.0, 1.0))


def test_user_seeds_are_used_first():
    ext = (0.0, 4.0, -1.0, 1.0)
    X, _ = grid(ext, 100, 50)
    lines = streamlines(np.ones_like(X), np.zeros_like(X), ext, seeds=[(2.0, 0.123)])
    assert lines[0][0, 1] == pytest.approx(0.123)
