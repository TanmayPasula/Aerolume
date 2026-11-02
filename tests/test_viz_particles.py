"""Tracer particles (aerolume.viz.particles)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aerolume.viz import Particles

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
EXT = (0.0, 4.0, -1.0, 1.0)


def uniform(u0=2.0, shape=(50, 100)):
    return np.full(shape, u0), np.zeros(shape)


def test_uniform_flow_moves_every_particle_by_u_dt():
    p = Particles(EXT, 500, seed=1)
    before = p.positions()
    u, v = uniform(2.0)
    p.advect(u, v, 0.05)
    after = p.positions()
    moved = after[:, 0] > before[:, 0]                       # not recycled
    assert moved.sum() > 450
    assert np.allclose(after[moved, 0] - before[moved, 0], 0.1, atol=1e-5)
    assert np.allclose(after[moved, 1], before[moved, 1], atol=1e-6)


def test_particles_leaving_are_recycled_on_the_inlet_rake():
    p = Particles(EXT, 400, seed=2)
    u, v = uniform(2.0)
    for _ in range(30):                                       # 30 * 0.1 = 3 length units
        p.advect(u, v, 0.05)
    pos, age = p.positions(), p.ages()
    assert (pos[:, 0] >= 0.0).all() and (pos[:, 0] <= 4.0).all()
    assert (np.abs(pos[:, 1]) <= 1.0).all()
    # A particle of age a released at the inlet (jitter <= u dt) sits at x in [u a, u (a + dt)].
    released = age < 30 * 0.05 - 1e-6
    assert released.sum() > 50
    x_expect = 2.0 * age[released]
    assert np.all(pos[released, 0] >= x_expect - 1e-4)
    assert np.all(pos[released, 0] <= x_expect + 2.0 * 0.05 + 1e-4)


def test_seed_on_rake_and_custom_rake():
    p = Particles(EXT, 100, rake=((1.0, -0.5), (1.0, 0.5)))
    p.seed(fill=False)
    pos = p.positions()
    assert np.allclose(pos[:, 0], 1.0) and (np.abs(pos[:, 1]) <= 0.5).all()
    assert (p.ages() == 0).all()


def test_particles_never_rest_inside_solids():
    ny, nx = 100, 200
    mask = np.zeros((ny, nx))
    mask[30:70, 80:120] = 1.0                                 # a block in the middle
    u, v = uniform(1.0, (ny, nx))
    p = Particles(EXT, 2000, seed=3)
    for _ in range(40):
        p.advect(u, v, 0.05, mask=mask)
        pos = p.positions()
        i = np.clip(((pos[:, 0] - EXT[0]) / 4.0 * nx).astype(int), 0, nx - 1)
        j = np.clip(((pos[:, 1] - EXT[2]) / 2.0 * ny).astype(int), 0, ny - 1)
        inside = mask[j, i] > 0.5
        # Bilinear sampling of the mask: allow the half-cell rim, nothing deeper.
        deep = inside & (np.abs(pos[:, 0] - 2.0) < 0.38) & (np.abs(pos[:, 1]) < 0.38)
        assert not deep.any()


def test_rk2_keeps_particles_on_circles_in_solid_body_rotation():
    ext = (-1.0, 1.0, -1.0, 1.0)
    x = np.linspace(-1, 1, 129)
    xc = 0.5 * (x[1:] + x[:-1])
    X, Y = np.meshgrid(xc, xc)
    p = Particles(ext, 300, seed=4)
    r0 = np.hypot(*p.positions().T)
    for _ in range(200):                                      # ~1 revolution
        p.advect(-Y, X, 2 * np.pi / 200)
    r1 = np.hypot(*p.positions().T)
    keep = r0 < 0.9                                           # well inside (no border clamping)
    assert np.abs(r1[keep] - r0[keep]).max() < 2e-3


def test_lifetime_recycling_and_fading_alpha():
    p = Particles(EXT, 300, lifetime=0.5, seed=5)
    u, v = np.zeros((50, 100)), np.zeros((50, 100))           # no motion: only ageing recycles
    for _ in range(20):
        p.advect(u, v, 0.05)
    age = p.ages()
    assert age.max() <= 0.5 * 1.25 + 1e-6
    a = p.alphas()
    assert a.shape == (300,) and (a >= 0).all() and (a <= 1).all()
    assert ((a > 0) & (a < 1)).any()


def test_set_extent_reseeds_inside_the_new_domain():
    p = Particles(EXT, 200)
    p.set_extent((10.0, 11.0, 0.0, 1.0))
    pos = p.positions()
    assert (pos[:, 0] >= 10).all() and (pos[:, 0] <= 11).all()


@pytest.mark.gpu
@cuda
def test_particles_follow_cuda_fields():
    p = Particles(EXT, 1000, seed=6)
    u = torch.full((50, 100), 2.0, device="cuda")
    v = torch.zeros(50, 100, device="cuda")
    before = p.positions()
    p.advect(u, v, 0.05)
    assert p.device.type == "cuda"
    after = p.positions()
    moved = after[:, 0] > before[:, 0]
    assert np.allclose(after[moved, 0] - before[moved, 0], 0.1, atol=1e-5)
