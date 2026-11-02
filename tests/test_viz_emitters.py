"""Placed-physics visualisation: emitter particles, smoke channels, new colour maps (aerolume.viz)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerolume.model import Emitter
from aerolume.viz import colormaps as C
from aerolume.viz.particles import EmitterParticles, emitter_footprint, emitter_on
from aerolume.viz.render import render_dye

EXTENT2 = (0.0, 2.0, 0.0, 1.0)


def _uniform2(nx=64, ny=32, u=1.0, v=0.0):
    return np.full((ny, nx), u, np.float32), np.full((ny, nx), v, np.float32)


# ---------------------------------------------------------------- particles
def test_point_emitter_draws_an_exact_streakline_in_uniform_flow():
    """A point source in a uniform stream: every particle sits at x0 + U * age on the line y = y0."""
    e = Emitter(name="P", species="particles", kind="point", position=(0.2, 0.5), radius=1e-9, rate=2000.0)
    p = EmitterParticles(EXTENT2, [e], n_max=4000, cell=0.01)
    u, v = _uniform2()
    for _ in range(50):
        p.step(u, v, dt=0.01)
    pos, age = p.positions(), p.ages()
    assert len(pos) == pytest.approx(2000 * 0.5, rel=0.02)            # rate x time
    assert np.allclose(pos[:, 1], 0.5, atol=1e-6)                       # straight streakline
    assert np.allclose(pos[:, 0], 0.2 + age, atol=2e-6)                 # carried at U = 1
    assert p.emitter_ids().max() == 0


def test_particles_leave_the_domain_and_die_in_solids_and_of_old_age():
    e = Emitter(species="particles", kind="point", position=(0.1, 0.5), radius=1e-9, rate=1000.0)
    u, v = _uniform2(u=4.0)
    mask = np.zeros_like(u)
    mask[:, 32:40] = 1.0                                                 # a wall across x = 1.0 .. 1.25
    p = EmitterParticles(EXTENT2, [e], n_max=2000, cell=0.01)
    for _ in range(60):
        p.step(u, v, dt=0.01, mask=mask)
    assert p.positions()[:, 0].max() < 1.0 + 1e-6                       # none got through the wall
    q = EmitterParticles(EXTENT2, [e], n_max=2000, cell=0.01, lifetime=0.1)
    for _ in range(40):
        q.step(*_uniform2(u=1.0), dt=0.01)
    assert q.ages().max() <= 0.125 + 1e-6                               # lifetime +25 % at most
    r = EmitterParticles(EXTENT2, [e], n_max=2000, cell=0.01)
    for _ in range(80):
        r.step(*_uniform2(u=4.0), dt=0.01)
    assert r.positions()[:, 0].max() <= 2.0 and r.count() < 1000       # the outflow takes them


def test_rake_releases_from_its_nozzles_and_pulses_make_timelines():
    rake = Emitter(species="particles", kind="rake", position=(0.1, 0.2), end=(0.1, 0.8), count=4,
                   radius=1e-9, rate=4000.0)
    p = EmitterParticles(EXTENT2, [rake], n_max=4000, cell=0.01)
    p.step(*_uniform2(u=0.0), dt=0.05)
    ys = np.unique(np.round(p.positions()[:, 1], 6))
    assert np.allclose(ys, [0.2, 0.4, 0.6, 0.8])
    pulsed = Emitter(species="particles", kind="line", position=(0.1, 0.2), end=(0.1, 0.8), rate=5000.0,
                     pulse_period=0.2, pulse_duty=0.25)
    q = EmitterParticles(EXTENT2, [pulsed], n_max=8000, cell=0.002)
    for _ in range(40):
        q.step(*_uniform2(u=1.0), dt=0.01)
    t_release = q.t - (q.positions()[:, 0] - 0.1)                        # x = x0 + U * age, U = 1
    phase = np.mod(t_release + 0.002, 0.2)                               # + the nozzle jitter
    assert np.all(phase < 0.05 + 0.012)                                  # only released while on
    assert phase.max() > 0.03                                            # ... over the whole 'on' part
    assert emitter_on(pulsed, 0.01) and not emitter_on(pulsed, 0.1)


def test_footprints_cover_every_emitter_kind():
    for kind, extra in (("point", {}), ("line", {"end": (1.0, 0.5)}), ("rake", {"end": (1.0, 0.5)}),
                        ("rect", {"size": (0.2, 0.1)}), ("disc", {"size": (0.1,)}), ("plane", {"size": (0.3,)})):
        f = emitter_footprint(Emitter(kind=kind, position=(0.5, 0.5), **extra), dim=2, cell=0.01)
        assert f["origin"].shape == (3,) and np.all(np.isfinite(f["A"]))
    disc = Emitter(species="particles", kind="disc", position=(0.5, 0.5, 0.5), size=(0.1,), normal=(1, 0, 0),
                   rate=3000.0)
    ext3 = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    p = EmitterParticles(ext3, [disc], n_max=3000, cell=0.01)
    z = np.zeros((16, 16, 16), np.float32)
    p.step(z, z, z, dt=0.5)
    pos = p.positions()
    assert np.allclose(pos[:, 0], 0.5, atol=1e-6)                        # in the disc's plane x = 0.5
    r = np.hypot(pos[:, 1] - 0.5, pos[:, 2] - 0.5)
    assert r.max() <= 0.1 + 1e-6 and r.mean() == pytest.approx(2 / 3 * 0.1, rel=0.05)   # uniform over the disc


def test_3d_advection_follows_a_uniform_stream_and_trails_record_history():
    e = Emitter(species="particles", kind="point", position=(0.1, 0.5, 0.5), radius=1e-9, rate=500.0)
    ext3 = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    p = EmitterParticles(ext3, [e], n_max=1000, cell=0.01, trail=5)
    one = np.ones((8, 8, 8), np.float32)
    zero = np.zeros_like(one)
    for _ in range(20):
        p.step(one * 0.5, zero, zero + 0.25, dt=0.02)
    pos, age = p.positions(), p.ages()
    assert np.allclose(pos[:, 0], 0.1 + 0.5 * age, atol=1e-5)
    assert np.allclose(pos[:, 2], 0.5 + 0.25 * age, atol=1e-5)
    tr = p.trails()
    assert tr.shape == (len(pos), 5, 3)
    old = tr[np.all(np.isfinite(tr[:, :, 0]), axis=1)]
    assert len(old) and np.all(np.diff(old[:, :, 0], axis=1) > 0)        # oldest first, moving downstream


def test_live_move_keeps_particles_and_units_convert_nondimensional_fields():
    e = Emitter(species="particles", kind="point", position=(0.2, 0.5), radius=1e-9, rate=1000.0)
    # non-dimensional: u = 1 means U = 10 m/s, L = 0.5 m, dt in L/U
    p = EmitterParticles(EXTENT2, [e], n_max=2000, cell=0.01, velocity_scale=0.5, time_unit_s=0.05)
    u, v = _uniform2(u=1.0)
    for _ in range(10):
        p.step(u, v, dt=0.2)                              # 0.2 L/U = 0.01 s, 0.1 m per step
    n0 = p.count()
    assert n0 == pytest.approx(1000 * 0.1, abs=2)
    assert np.allclose(p.positions()[:, 0], 0.2 + 10.0 * p.ages(), atol=1e-5)    # 10 m/s
    p.set_emitters([Emitter(species="particles", kind="point", position=(0.2, 0.2), radius=1e-9, rate=1000.0)])
    p.step(u, v, dt=0.2)
    assert p.count() >= n0                                 # the old ones keep flying
    assert np.isclose(p.positions()[:, 1], 0.2).any() and np.isclose(p.positions()[:, 1], 0.5).any()
    smoke_only = EmitterParticles(EXTENT2, [Emitter(species="dye")], cell=0.01)
    smoke_only.step(u, v, dt=0.2)
    assert smoke_only.count() == 0                         # smoke emitters release no particles


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_step_needs_no_host_sync():
    e = Emitter(species="particles", kind="line", position=(0.1, 0.1), end=(0.1, 0.9), rate=50000.0)
    p = EmitterParticles(EXTENT2, [e], n_max=20000, cell=0.01, device="cuda")
    u = torch.ones(64, 128, device="cuda")
    v = torch.zeros_like(u)
    p.step(u, v, dt=0.01)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        p.step(u, v, dt=0.01)                                 # raises on any synchronising call
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert p.count() > 0


# ---------------------------------------------------------------- smoke channels
def test_dye_channels_follow_emitter_colours():
    em = [Emitter(name="A"), Emitter(name="B", color="#e69f00"), Emitter(name="C"),
          Emitter(name="D", species="particles", color="#56b4e9"), Emitter(name="E", color="#56b4e9"),
          Emitter(name="F", color="#cc79a7"), Emitter(name="G", color="#123456")]
    ch = C.dye_channels(em)
    assert [c["field"] for c in ch] == ["dye", "dye1", "dye2", "dye3"]
    assert ch[0]["names"] == ["A", "C"] and ch[1]["color"] == "#e69f00"
    assert "G" in ch[-1]["names"]                         # a fifth colour folds into the last channel
    assert C.hex_rgb("#fff") == (255, 255, 255) and C.hex_rgb((0.5, 0.0, 1.0)) == (128, 0, 255)


def test_render_dye_composites_channels_over_the_dark_background():
    ny, nx = 4, 6
    a = np.zeros((ny, nx), np.float32)
    b = np.zeros((ny, nx), np.float32)
    a[0, 0] = 1.0                    # bottom-left: channel 0 only
    b[0, 1] = 1.0                    # channel 1 only
    a[0, 2] = b[0, 2] = 0.5          # both, half concentration
    img = render_dye([a, b], ["#ffffff", "#e69f00"])
    assert img.shape == (ny, nx, 4) and img.dtype == np.uint8
    bottom = img[-1]                                     # row 0 of the field is the image's last row
    assert tuple(bottom[0, :3]) == (255, 255, 255)
    assert tuple(bottom[1, :3]) == C.hex_rgb("#e69f00")
    assert tuple(bottom[3, :3]) == C.DYE_BACKGROUND
    mix = bottom[2, :3].astype(int)
    assert mix[0] > mix[2]                               # the later (orange) smoke shows over the white
    mask = np.zeros_like(a)
    mask[1:3, 2:4] = 1.0
    img2 = render_dye(np.stack([a, b]), mask=mask)
    assert tuple(img2[1, 2, :3]) != tuple(img[1, 2, :3])  # solids painted grey


# ---------------------------------------------------------------- colour maps
@pytest.mark.parametrize("name", ["thermal", "fire"])
def test_cet_style_sequential_maps_have_linear_lightness(name):
    L = C.srgb_to_lab(C.COLORMAPS[name] / 255.0)[:, 0]
    assert np.all(np.diff(L) > 0)                         # strictly monotone: no false fronts
    ideal = np.linspace(L[0], L[-1], 256)
    assert np.abs(L - ideal).max() < 1.0                  # linear in L* to within uint8 rounding
    assert L[-1] - L[0] > 80


def test_bwr_is_a_symmetric_diverging_map():
    L = C.srgb_to_lab(C.COLORMAPS["bwr"] / 255.0)[:, 0]
    assert 125 <= int(np.argmax(L)) <= 130 and L.max() > 95
    assert np.abs(L - L[::-1]).max() < 1.0
    assert "bwr" in C.DIVERGING_CMAPS


def test_temperature_fields_have_their_own_map_and_units():
    assert C.default_cmap("temperature") == "thermal"
    assert C.FIELD_INFO["temperature"].unit == "°C"
    assert C.FIELD_INFO["heat_flux"].unit == "W/m²"
    assert C.theta_to_celsius(np.array([0.0, 1.0]), t_ambient=20.0, delta_t=40.0).tolist() == [20.0, 60.0]
    fam = {name for _, names, _ in C.CMAP_FAMILIES for name in names}
    assert fam == set(C.CMAP_NAMES)                       # every map is in exactly one family
