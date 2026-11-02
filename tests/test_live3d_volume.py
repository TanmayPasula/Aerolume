"""Live 3D frames with flow in the volume (DESIGN.md 7.13) and the smoke in results.

* the rake dict becomes solver nozzles in solver units (metres / L, seconds x U / L),
  only on the solved side of a symmetry plane;
* with ``view3d`` frames carry a float16 volume (smoke, velocity, mask) under the cell
  and byte caps, a growing ``volume_serial``, traced streamlines and the rake;
* posting a rake moves the solver's nozzles; a bad rake is refused in the caller's thread;
* 2D sessions accept (and ignore) the 3D view keys;
* a production run with the 3D view delivers volumes and keeps the smoke in its result.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerolume.core import fluid as F
from aerolume.geometry3d import design_from_meshes, sphere
from aerolume.model import CFDSettings, FlowConditions, Simulation
from aerolume.simulation import LiveSession, ViewSettings, run_cfd
from aerolume.simulation import live3d as L3
from aerolume.simulation import plan as planmod
from aerolume.simulation.plan import plan_simulation
from aerolume.viz3d.flow import StreamlineSet, smoke_volume

AIR = F.air_isa(0.0)


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def sphere_case(*, re: float = 50.0, **cfd):
    d = design_from_meshes([sphere(1.0, 24)], "Sphere", kind="generic")
    cfd.setdefault("cells_per_ref", 6)
    cfd.setdefault("duration", 1.0)
    cfd.setdefault("average_from", 0.5)
    speed = re * AIR.viscosity / AIR.density
    sim = Simulation("s", d.id, conditions=FlowConditions(AIR, speed, 0.0),
                     cfd=CFDSettings(quality=cfd.pop("quality", "draft"), device="cpu", **cfd))
    return d, sim, plan_simulation(d, sim)


def test_dye_rake_for_converts_units_and_keeps_the_solved_side():
    _, _, plan = sphere_case()
    assert plan.symmetry_plane
    L, U = plan.length_ref, plan.speed
    rake, sr = L3.dye_rake_for(plan, {"kind": "rake", "n": 5, "pulse": [0.2, 0.4]})
    assert len(sr.points) == 5
    keep = sr.points[sr.points[:, 1] >= -1e-12]
    assert len(rake.positions) == len(keep) == 3                     # y = -w, -w/2 dropped; the plane kept
    assert np.allclose(rake.positions, keep / L)
    assert rake.pulse_on == pytest.approx(0.2 * U / L) and rake.pulse_off == pytest.approx(0.4 * U / L)
    band, _ = L3.dye_rake_for(plan, {"kind": "plane"})
    assert band.kind == "plane" and band.positions[0, 1] >= 0.0
    default, _ = L3.dye_rake_for(plan, None)
    assert default.kind == "rake" and np.all(default.positions[:, 0] * L < plan.body_bounds[0])


def test_volume_factor_respects_the_cell_and_byte_caps(monkeypatch):
    _, _, plan = sphere_case()
    r = L3.FrameRenderer3D(plan)
    assert r.volume_factor(4) == 1
    monkeypatch.setattr(L3, "LIVE_VOLUME_MAX_CELLS", plan.cells // 5)
    f = r.volume_factor(1)
    nx, ny, nz = plan.grid3
    assert (nx // f) * (ny // f) * (nz // f) <= plan.cells // 5 and f >= 2
    monkeypatch.setattr(L3, "LIVE_VOLUME_MAX_CELLS", 10 ** 9)
    monkeypatch.setattr(L3, "LIVE_VOLUME_MAX_BYTES", plan.cells * 2 * 5 // 9)
    g = r.volume_factor(4)
    assert (nx // g) * (ny // g) * (nz // g) * 5 * 2 <= plan.cells * 2 * 5 // 9


def test_live_session_frames_carry_volume_streamlines_and_rake(monkeypatch):
    monkeypatch.setattr(L3, "VOLUME_MIN_PERIOD_S", 0.0)
    monkeypatch.setattr(L3, "STREAMLINE3D_PERIOD_S", 0.0)
    d, sim, _ = sphere_case()
    view = ViewSettings(view3d=True, smoke=True, streamlines3d=True, fps=1000.0)
    s = LiveSession(d, sim, device="cpu", view=view)
    try:
        frames = [s.tick(60.0) for _ in range(4)]
        fr = frames[-1]
        v = fr.volume
        assert v is not None and set(v.fields) == {"dye", "u", "v", "w"} and v.mask is not None
        assert all(a.dtype == np.float16 for a in v.fields.values()) and v.label == "instant"
        assert not v.mirrored                                       # explore sessions solve the full domain
        nbytes = sum(a.nbytes for a in v.fields.values()) + v.mask.nbytes
        assert nbytes <= L3.LIVE_VOLUME_MAX_BYTES
        assert frames[-1].volume_serial > frames[0].volume_serial >= 1
        assert float(v.fields["dye"].max()) > 0.3                   # smoke is flowing
        speed = np.sqrt(sum(v.fields[k].astype(np.float32) ** 2 for k in "uvw"))
        assert 0.5 * s.speed < float(np.median(speed[v.mask < 0.5])) < 1.5 * s.speed   # SI units
        assert isinstance(fr.streamlines3d, StreamlineSet) and fr.streamlines3d.n_lines > 0
        assert fr.seed_rake is not None and fr.seed_rake["kind"] == "rake"
        # move the rake: the solver's nozzles follow on the next tick
        before = s.solver.dye_rake.positions.copy()
        spec = dict(fr.seed_rake)
        spec["z"] = spec["z"] + 0.2
        s.post(seed_rake=spec)
        s.tick(30.0)
        after = s.solver.dye_rake.positions
        assert np.allclose(after[:, 2] - before[:, 2], 0.2 / s.plan.length_ref)
        with pytest.raises(ValueError):
            s.post(seed_rake={"kind": "fan"})
        with pytest.raises(ValueError):
            s.post(seed_rake="rake")
        # smoke only: no velocity in the volume, no streamlines
        s.post(streamlines3d=False)
        fr2 = s.tick(30.0)
        assert set(fr2.volume.fields) == {"dye"} and fr2.streamlines3d is None
        s.post(view3d=False)
        assert s.tick(30.0).volume is None
        s.post(field="dye")
        fr3 = s.tick(30.0)
        assert fr3.field == "dye" and fr3.cmap == "grayscale"       # the smoke on a slice
    finally:
        s.close()


def test_2d_sessions_accept_the_3d_view_keys():
    from aerolume.geometry import design_from_airfoil, naca
    d = design_from_airfoil(naca("0012", 160, 1.0), "NACA 0012", 1.0)
    sim = Simulation("live", d.id, conditions=FlowConditions(AIR, 300.0 * AIR.viscosity / AIR.density, 2.0),
                     cfd=CFDSettings(quality="draft"))
    s = LiveSession(d, sim, device="cpu")
    try:
        s.post(view3d=True, smoke=True, streamlines3d=True, seed_rake={"kind": "rake"})
        assert s.view.view3d and s.tick(20.0).volume is None
    finally:
        s.close()


def test_run_with_the_3d_view_delivers_volumes_and_keeps_the_smoke(monkeypatch):
    monkeypatch.setattr(L3, "VOLUME_MIN_PERIOD_S", 0.0)
    _, _, plan = sphere_case(duration=0.8, average_from=0.4)
    frames: list = []
    rake = {"kind": "rake", "n": 3, "z": 0.1}
    res = run_cfd(plan, frame=frames.append, view=ViewSettings(view3d=True, streamlines3d=True, fps=200.0,
                                                               seed_rake=rake))
    with_vol = [f for f in frames if f.volume is not None]
    assert with_vol and frames[-1].phase == "done" and frames[-1].volume is not None
    assert frames[-1].volume.mirrored and set(frames[-1].volume.fields) == {"dye", "u", "v", "w"}
    assert frames[-1].seed_rake["z"] == pytest.approx(0.1)
    smoke = smoke_volume(res)
    assert smoke is not None and float(smoke.fields["dye"].max()) > 0.2
    assert smoke.fields["dye"].shape == res.volume.fields["u"].shape
    # the snapshot of 7.13: smoke and the velocity of the same moment, next to the mean volume
    assert res.volume.label == "mean" and smoke in res.extra_volumes and smoke.label == "instant"
    assert set(smoke.fields) == {"dye", "u", "v", "w"} and smoke.mirrored == res.volume.mirrored
    assert res.find_volume("dye") is smoke and res.find_volume("u", "instant") is smoke
    assert res.find_volume("u", "mean") is res.volume and smoke.mask is res.volume.mask
    assert not np.array_equal(smoke.fields["u"], res.volume.fields["u"])     # instantaneous, not the mean
    assert 0.0 <= float(smoke.fields["dye"].min()) and float(smoke.fields["dye"].max()) <= 1.0
    # the smoke left from the user's rake: the stream at z = 0.1 m, not the default height
    nz, _, nx = smoke.fields["dye"].shape
    zc = smoke.origin[2] + (np.arange(nz) + 0.5) * smoke.spacing[2]
    xc = smoke.origin[0] + (np.arange(nx) + 0.5) * smoke.spacing[0]
    x_rake = frames[-1].seed_rake["x"]
    near = (xc > x_rake) & (xc < x_rake + 0.3)                       # just downstream of the nozzles
    profile = smoke.fields["dye"][:, :, near].max(axis=(1, 2))
    assert abs(zc[int(np.argmax(profile))] - 0.1) <= 1.5 * smoke.spacing[2]
