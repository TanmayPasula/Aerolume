"""Placed physics of the 3D solver: heat transfer, buoyancy, fans, jets and vents
(DESIGN.md 10.2 / 10.3, docs/theory_ns3d.md sections 15-17).

What is proven here (small float64 grids on the CPU unless marked):

* the default path is untouched: no physics object, no temperature field, and
  ``step3d_phys`` without physics is ``step3d_full``;
* conduction between two walls reaches the exact linear profile and Nu = 1;
* energy is conserved exactly by the discrete scheme: a volumetric heater, a
  heat-flux wall and a heat-flux body raise the stored heat at exactly their power,
  and a held-temperature body's reported heat rate (from the penalisation) is
  exactly the rate of change of the stored heat;
* an adiabatic body blocks conduction; buoyancy of a linear stratification is a
  discrete gradient, so a stratified fluid stays at rest to round-off -- also around
  a body, which then feels the Archimedes force (buoyancy is added after the
  penalisation, theory_ns3d.md 15.3);
* the heat equation's time step obeys the combined advection-diffusion bound, which
  keeps theta inside its initial range where separate limits blew up, and the first
  step from rest respects the sources' speed and the free-fall scale;
* a fan's force adds exactly T dt of momentum to a periodic box, the
  velocity-regulated fan reaches its target, and the footprints integrate exactly
  to the set thrust, flow and momentum flux (also when a symmetry plane cuts them);
* a jet's volume flow leaves through the tunnel outlet and the continuity residual
  ``div u - m`` stays at round-off with sources;
* on CUDA, the fused kernels reproduce the reference core with every feature on
  (float64, 1e-12) and CUDA graphs replay the eager run bit for bit.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns3d import (
    Emitter3D,
    Fan3D,
    FlowSolver3D,
    HeatSource3D,
    ImmersedMesh,
    Port3D,
    SolverConfig3D,
    ThermalConfig3D,
    core,
    estimate_memory_bytes,
)
from aerolume.solvers.ns3d import physics as P


def _load_cases():
    path = Path(__file__).resolve().parents[1] / "validation" / "ns3d_cases.py"
    mod = sys.modules.get("ns3d_cases")
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("ns3d_cases", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ns3d_cases"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("ns3d_cases", None)
        raise
    return mod


cases = _load_cases()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sphere(d=0.5, center=(0.0, 0.0, 0.0), level=2, **kw) -> ImmersedMesh:
    v, f = cases.icosphere(d, level, center)
    return ImmersedMesh(v, f, name=kw.pop("name", "sphere"), **kw)


def _box(n=12, **kw) -> SolverConfig3D:
    """A closed unit box at rest (slip walls unless overridden), float64 on the CPU."""
    base = {"nx": n, "ny": n, "nz": n, "h": 1.0 / n, "x0": 0.0, "y0": 0.0, "z0": 0.0, "re": 20.0,
            "domain": "closed", "device": "cpu", "dtype": "float64", "dye": False, "diagnostics": True}
    base.update(kw)
    return SolverConfig3D(**base)


def _stored_heat(s: FlowSolver3D) -> float:
    """Heat held in the domain in conduction units: Pe sum(theta) h^3."""
    return float(s.theta.double().sum()) * s.ops.h ** 3 * s.physics.thermal.pe


# ---------------------------------------------------------------------------
# configuration and the default path
# ---------------------------------------------------------------------------

def test_default_config_builds_no_physics():
    cfg = SolverConfig3D(nx=16, ny=8, nz=8, h=0.25, x0=-1, y0=-1, z0=-1, re=50.0, device="cpu", dtype="float64",
                         bodies=[_sphere()])
    assert not cfg.has_placed_physics
    s = FlowSolver3D(cfg)
    assert s.physics is None and s.theta is None and not s.thermal_enabled
    assert s.heat_rates() == [] and s.fan_report() == [] and s.heat_balance() == {}
    assert s.heat_transfer()["bodies"] == []
    assert s.diagnostics()["t"].size == 0
    s.advance(3)
    assert s.physics is None


def test_step3d_phys_without_physics_is_the_v02_step():
    cfg = SolverConfig3D(nx=16, ny=8, nz=8, h=0.25, x0=-1, y0=-1, z0=-1, re=50.0, device="cpu", dtype="float64",
                         bodies=[_sphere()], dye=False)
    ops = core.make_operators3d(cfg)
    geom = core.build_geometry3d(cfg.bodies, ops)
    st = core.init_state3d(cfg, ops)
    a, ca = core.step3d_full(st, geom, ops)
    out = core.step3d_phys(st, geom, ops)
    assert out.theta is None
    for x, y in zip(a, out.state):
        assert torch.equal(x, y)
    assert torch.equal(ca, out.coeffs)


@pytest.mark.parametrize("bad", [
    {"thermal": {"walls": {"middle": "adiabatic"}}},
    {"thermal": {"walls": {"left": ("hot", 1.0)}}},
    {"heat_sources": [{"kind": "box"}]},                                 # no thermal
    {"domain": "closed", "periodic": True},
    {"adaptive_dt": "magic"},
    {"pid_target": 2.0},
])
def test_config_rejects_bad_physics(bad):
    with pytest.raises((ValueError, TypeError)):
        SolverConfig3D(nx=8, ny=8, nz=8, h=0.1, x0=0, y0=0, z0=0, re=10.0, **bad)


def test_physics_types_validate_their_fields():
    with pytest.raises(ValueError):
        Fan3D(mode="turbo")
    with pytest.raises(ValueError):
        Fan3D(direction=(0, 0, 0))
    with pytest.raises(ValueError):
        Port3D(kind="drain")
    with pytest.raises(ValueError):
        Emitter3D(kind="line")                                          # needs an end point
    with pytest.raises(ValueError):
        Emitter3D(channel=4)
    with pytest.raises(ValueError):
        ImmersedMesh(np.eye(3), [[0, 1, 2]], thermal="warm")
    f = Fan3D(direction=(2.0, 0.0, 0.0), diameter=2.0)
    assert f.direction == (1.0, 0.0, 0.0) and f.area == pytest.approx(math.pi)
    assert Port3D(kind="vent", diameter=0.2, speed=1.0).flow == pytest.approx(-math.pi * 0.01)
    th = ThermalConfig3D(walls=[{"side": "left", "mode": "temperature", "value": 1.0}])
    assert th.wall("left") == ("temperature", 1.0) and th.wall("top") == ("adiabatic", 0.0)


def test_memory_estimate_counts_the_physics():
    base = estimate_memory_bytes(64, 32, 32)
    assert estimate_memory_bytes(64, 32, 32, thermal=True) > base
    assert estimate_memory_bytes(64, 32, 32, dye_channels=4) > base
    assert estimate_memory_bytes(64, 32, 32, particles=10 ** 6) > base


# ---------------------------------------------------------------------------
# heat transfer: exact discrete identities
# ---------------------------------------------------------------------------

def _conduction_box(n, bodies=()):
    return _box(n, ny=4, nz=4, re=10.0, bodies=list(bodies), thermal=ThermalConfig3D(
        prandtl=1.0, walls={"left": ("temperature", 0.5), "right": ("temperature", -0.5)}))


def test_conduction_between_walls_is_exact():
    """The linear profile is the scheme's exact steady state (Nu = 1), and a cold start decays to it."""
    n = 12
    x = (np.arange(n) + 0.5) / n
    s = FlowSolver3D(_conduction_box(n))
    s.set_temperature(np.broadcast_to(0.5 - x, (4, 4, n)).copy())
    s.advance(20)
    th = s.field("temperature").double().numpy()
    assert np.abs(th - (0.5 - x)[None, None, :]).max() < 1e-13
    w = s.wall_heat_rates()
    area = (4 / n) ** 2
    assert w["left"] == pytest.approx(area, rel=1e-12)          # Nu = 1 on the wall: Q = k dT A / L
    assert w["right"] == pytest.approx(-area, rel=1e-12)
    assert abs(w["top"]) < 1e-15 and abs(w["front"]) < 1e-15
    assert float(s.field("speed").max()) < 1e-14               # Ri = 0: nothing moves
    cold = FlowSolver3D(_conduction_box(8))
    errs = []
    for _ in range(3):
        cold.advance(150)
        x8 = (np.arange(8) + 0.5) / 8
        errs.append(float(np.abs(cold.field("temperature").numpy() - (0.5 - x8)).max()))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.3 * errs[0]


def test_heat_source_raises_the_stored_heat_at_its_power():
    cfg = _box(thermal=ThermalConfig3D(prandtl=0.71),
               heat_sources=[HeatSource3D(kind="sphere", center=(0.4, 0.5, 0.55), size=(0.2,), power=3.0),
                             HeatSource3D(kind="box", center=(0.7, 0.3, 0.3), size=(0.2, 0.1, 0.3), power=-1.0)])
    s = FlowSolver3D(cfg)
    assert s.physics.thermal.sources.total == pytest.approx(2.0, rel=1e-12)
    q0 = _stored_heat(s)
    s.advance(20)
    assert (_stored_heat(s) - q0) / s.t == pytest.approx(2.0, rel=1e-10)
    assert s.latest_diagnostics()["heat_storage"] == pytest.approx(2.0, rel=1e-10)
    hb = s.heat_balance()
    assert hb["heat_in_sources"] == pytest.approx(2.0, rel=1e-12)
    assert abs(hb["heat_balance_residual"]) < 1e-9


def test_heat_flux_wall_and_body_release_exactly_their_flux():
    g_wall, g_body = 0.7, 2.0
    body = _sphere(0.4, (0.5, 0.5, 0.5), name="heater", thermal="heat_flux", thermal_value=g_body)
    cfg = _box(bodies=[body], thermal=ThermalConfig3D(prandtl=0.71, walls={"bottom": ("heat_flux", g_wall)}))
    s = FlowSolver3D(cfg)
    area = s.heat_rates()[0]["area"]
    q0 = _stored_heat(s)
    s.advance(20)
    expected = g_wall * 1.0 + g_body * area
    assert (_stored_heat(s) - q0) / s.t == pytest.approx(expected, rel=1e-10)
    assert s.heat_rates()[0]["heat_rate"] == pytest.approx(g_body * area, rel=1e-12)
    assert s.wall_heat_rates()["bottom"] == pytest.approx(g_wall, rel=1e-9)


def test_heat_flux_body_wall_temperature_is_measured_in_its_skin():
    """A heat-flux body's wall temperature is the release-weighted mean of theta over its skin
    (critic finding: the probe extrapolation read ~ambient at Draft and gave Nu 1,821): a
    uniform field reads exactly that value, the per-step history carries it as theta_wall_0, and
    the reported Nusselt numbers are Q / (A theta_w) from it everywhere."""
    g_body = 2.0
    body = _sphere(0.4, (0.5, 0.5, 0.5), name="heater", thermal="heat_flux", thermal_value=g_body)
    hot = _sphere(0.2, (0.2, 0.2, 0.2), name="held", thermal="temperature", thermal_value=0.5)
    s = FlowSolver3D(_box(bodies=[body, hot], thermal=ThermalConfig3D(prandtl=0.71)))
    g = s._phys.thermal.geom
    assert g.wall_idx is not None and float(g.wall_w.sum()) == pytest.approx(1.0, rel=1e-12)
    s.set_temperature(0.37)
    tw = g.wall_theta(s.theta).cpu().numpy()
    assert tw[0] == pytest.approx(0.37, rel=1e-12) and tw[1] == pytest.approx(0.5, rel=1e-15)
    s.advance(15)
    ht = s.heat_transfer()
    b0 = ht["bodies"][0]
    expected = float(g.wall_theta(s.theta)[0])
    assert b0["theta_wall"] == pytest.approx(expected, rel=1e-12)
    assert b0["nusselt"] == pytest.approx(b0["heat_rate"] / (b0["area"] * expected), rel=1e-12)
    assert b0["heat_rate"] == pytest.approx(g_body * b0["area"], rel=1e-12)
    ph = s.physics_history()
    assert len(ph["theta_wall_0"]) == 15 and ph["theta_wall_0"][-1] == pytest.approx(expected, rel=1e-12)
    assert np.all(ph["theta_wall_1"] == 0.5)
    q = s.heat_rates()
    den = b0["area"] * ph["theta_wall_0"][-1] + q[1]["area"] * 0.5
    assert ph["nu"][-1] == pytest.approx((q[0]["heat_rate"] + q[1]["heat_rate"]) / den, rel=1e-9)
    assert q[0]["theta"] == pytest.approx(expected, rel=1e-12)
    # surface(): the flux body's per-vertex wall temperature is theta on the surface itself
    surf = s.surface()[0]
    assert np.all(np.isfinite(surf["theta_wall"])) and np.all(surf["heat_flux"] == g_body)


def test_heat_flux_wall_temperature_survives_live_body_moves():
    """set_bodies with a moved heat-flux body rebuilds the skin weights (and the history keeps
    its layout), so theta_wall follows the body."""
    body = _sphere(0.4, (0.5, 0.5, 0.5), name="heater", thermal="heat_flux", thermal_value=1.0)
    s = FlowSolver3D(_box(bodies=[body], thermal=ThermalConfig3D(prandtl=0.71)))
    s.advance(5)
    s.set_bodies([_sphere(0.4, (0.45, 0.5, 0.5), name="heater", thermal="heat_flux", thermal_value=1.0)])
    s.advance(5)
    g = s._phys.thermal.geom
    assert s.physics_history()["theta_wall_0"][-1] == pytest.approx(float(g.wall_theta(s.theta)[0]), rel=1e-12)


def test_planning_speed_sum_and_thermal_dt_match_the_solver_rule():
    """The planner helpers reproduce ThermalOps.dt_limit exactly for a given speed sum, and the
    buoyant speed scale takes over from the free-stream factor only when sqrt(Ri) makes it
    larger."""
    assert P.planning_speed_sum(0.0, 2.1) == 2.1
    assert P.planning_speed_sum(0.1, 2.1) == 2.1
    assert P.planning_speed_sum(627.0, 2.1) == pytest.approx(P.PLUME_SPEED_FACTOR * math.sqrt(627.0))
    body = _sphere(0.4, (0.5, 0.5, 0.5), thermal="temperature", thermal_value=1.0)
    s = FlowSolver3D(_box(bodies=[body], re=80.0, thermal=ThermalConfig3D(prandtl=0.71)))
    th = s._phys.thermal
    for S in (0.5, 3.0, 40.0):
        ref = float(th.dt_limit(None, s.ops, torch.tensor(S, dtype=torch.float64)))
        assert P.thermal_dt_estimate(s.ops.h, S, th.pe) == pytest.approx(ref, rel=1e-12)


def test_temperature_body_heat_rate_is_the_stored_heat_rate():
    body = _sphere(0.4, (0.5, 0.5, 0.5), name="hot", thermal="temperature", thermal_value=1.0)
    cfg = _box(bodies=[body], thermal=ThermalConfig3D(prandtl=0.71))
    s = FlowSolver3D(cfg)
    for _ in range(3):
        s.advance(10)
        q = s.heat_rates()[0]["heat_rate"]
        assert q > 0
        assert s.latest_diagnostics()["heat_storage"] == pytest.approx(q, rel=1e-9)
    ht = s.heat_transfer()
    assert ht["bodies"][0]["nusselt"] == pytest.approx(q / ht["bodies"][0]["area"], rel=1e-12)
    # the body interior is held at its temperature (implicit penalisation)
    k = s.ops.nz // 2
    assert float(s.theta[k, k, k]) == pytest.approx(1.0, abs=1e-3)


def test_adiabatic_body_blocks_conduction():
    """No conduction crosses the faces inside an insulating slab that spans the box."""
    n = 12
    x = (np.arange(n) + 0.5) / n
    slab = ImmersedMesh(*cases.box_mesh(0.4, 2.0, 2.0, center=(0.5, 0.5, 0.5)), name="slab", thermal="adiabatic")
    s = FlowSolver3D(_conduction_box(n, [slab]))
    s.set_temperature(np.broadcast_to(0.5 - x, (4, 4, n)).copy())
    st = s.state
    fx, _, _ = s.physics.thermal.fluxes(s.theta, st.u, st.v, st.w, s.ops)
    assert float(fx[..., n // 2].abs().max()) < 1e-15            # the face at the slab's centre
    q0 = s.wall_heat_rates()["left"]
    s.advance(300)
    assert s.wall_heat_rates()["left"] < 0.5 * q0                # the walls stop feeding a blocked path
    assert s.heat_rates()[0]["heat_rate"] == 0.0 and math.isnan(s.heat_transfer()["bodies"][0]["nusselt"])


def test_stratified_fluid_stays_at_rest():
    """Ri theta e_z of a linear profile is a discrete gradient: the projection removes it."""
    n = 10
    cfg = _box(n, bc_zmin="noslip", bc_zmax="noslip", thermal=ThermalConfig3D(
        prandtl=0.71, richardson=5.0, walls={"bottom": ("temperature", -0.5), "top": ("temperature", 0.5)}))
    s = FlowSolver3D(cfg)
    z = (np.arange(n) + 0.5) / n - 0.5
    s.set_temperature(np.broadcast_to(z[:, None, None], (n, n, n)).copy())
    s.advance(100)
    assert float(s.field("speed").max()) < 1e-12
    assert np.abs(s.field("temperature").numpy()[:, 0, 0] - z).max() < 1e-12


def _archimedes(n: int, steps: int = 60) -> tuple[float, float, float]:
    """A fixed insulating sphere in a linearly stratified, non-conducting fluid at rest:
    (measured Fz, Archimedes -Ri theta_c V, largest speed)."""
    d, zc, ri = 0.4, 0.62, 5.0
    v, f = cases.icosphere(d, 3, (0.5, 0.5, zc))
    cfg = _box(n, bodies=[ImmersedMesh(v, f, name="s", thermal="adiabatic")], re=50.0, bc_zmin="noslip",
               bc_zmax="noslip", thermal=ThermalConfig3D(prandtl=1e6, richardson=ri, walls={
                   "bottom": ("temperature", -0.5), "top": ("temperature", 0.5)}))
    s = FlowSolver3D(cfg)
    z = (np.arange(n) + 0.5) / n - 0.5
    s.set_temperature(np.broadcast_to(z[:, None, None], (n, n, n)).copy())
    s.advance(steps)
    fz = float(s.coefficients()[2]) * 0.5 * cfg.ref_area
    return fz, -ri * (zc - 0.5) * cases.mesh_volume(v, f), float(s.field("speed").max())


def test_body_in_a_stratified_fluid_stays_at_rest_and_feels_archimedes():
    """Buoyancy is added after the penalisation and everywhere, so a stratified fluid around a
    body is an exact rest state; the body force is the Archimedes force of the displaced fluid
    (volume-accurate indicator: 5.7 % at 6.4 cells per diameter, 2.4 % at 9.6, second order)."""
    fz, ref, speed = _archimedes(16)
    assert speed < 1e-6                     # only the residual conduction of Pr = 1e6 moves anything
    assert fz == pytest.approx(ref, rel=0.07)


def test_buoyancy_drives_a_plume_above_a_heated_body():
    body = _sphere(0.3, (0.5, 0.5, 0.35), name="hot", thermal="temperature", thermal_value=1.0)
    cfg = _box(16, bodies=[body], re=100.0, thermal=ThermalConfig3D(prandtl=0.71, richardson=1.0))
    s = FlowSolver3D(cfg)
    s.advance(150)
    w = s.field("w").numpy()
    assert w[int(0.6 * 16), 8, 8] > 0.01                                # the warm plume rises
    assert w[int(0.6 * 16), 8, 8] == pytest.approx(w[int(0.6 * 16), 7, 7], rel=0.05)   # and stays centred


def test_theta_step_obeys_the_combined_bound_and_stays_bounded():
    """Uniform flow with kappa = h U / 3 makes the advective (CFL 1/2) and diffusive (h^2 / 6 kappa)
    limits coincide; a checkerboard theta is the worst mode. Taking the two limits separately
    blew it up x1e7 in 25 steps; the combined rule C / (1/2) + 6 D <= 1 makes every stage a
    convex combination, so theta stays inside its initial range (a discrete maximum principle)."""
    h, pr = 1 / 8, 0.71
    cfg = SolverConfig3D(nx=32, ny=8, nz=8, h=h, x0=0, y0=0, z0=0, re=(3.0 / h) / pr, bc_y="periodic",
                         device="cpu", dtype="float64", perturb=False, dye=False,
                         thermal=ThermalConfig3D(prandtl=pr), diagnostics=True)
    s = FlowSolver3D(cfg)
    k, j, i = torch.meshgrid(torch.arange(8), torch.arange(8), torch.arange(32), indexing="ij")
    s.set_temperature(0.5 + 0.5 * (-1.0) ** (i + j + k).double())
    s.advance(60)
    assert float(s.theta.min()) >= -1e-12 and float(s.theta.max()) <= 1.0 + 1e-12
    kappa = s.physics.thermal.kappa
    c, d6 = s.dt * 1.0 / h, 6.0 * kappa * s.dt / h ** 2
    assert c / 0.5 + d6 == pytest.approx(1.0, rel=1e-9)            # exactly at the bound here
    assert s.latest_diagnostics()["cfl"] == pytest.approx(0.25, rel=1e-9)


def test_first_step_from_rest_respects_the_source_and_free_fall_scales():
    """In still air the CFL rule sees no velocity; a fan, a jet or a hot body would then take
    its first step at the viscous limit (huge at high Re). The source speed scale and the
    free-fall limit (a parcel accelerating at Ri |theta| moves <= cfl cells) bound it."""
    fan = Fan3D(center=(0.5, 0.5, 0.5), direction=(1, 0, 0), diameter=0.4, mode="velocity", value=2.0)
    s = FlowSolver3D(_box(16, re=1e5, fans=[fan]))
    assert s.physics.u_floor == pytest.approx(2.0)
    s.advance(1)
    assert s.state.dt.item() <= 0.5 * (1 / 16) / 2.0 * (1 + 1e-12)
    hot = _sphere(0.3, (0.5, 0.5, 0.4), name="hot", thermal="temperature", thermal_value=3.0)
    b = FlowSolver3D(_box(16, re=1e5, bodies=[hot], thermal=ThermalConfig3D(prandtl=0.71, richardson=2.0)))
    b.set_temperature(3.0)                                 # a hot room: |theta| max = 3 from the start
    b.advance(1)
    assert b.state.dt.item() <= 0.5 * math.sqrt((1 / 16) / (2.0 * 3.0)) * (1 + 1e-12)


def test_turbulent_prandtl_adds_diffusivity():
    cfg = SolverConfig3D(nx=24, ny=12, nz=12, h=1 / 6, x0=-1.5, y0=-1, z0=-1, re=2000.0, device="cpu",
                         dtype="float64", turbulence="smagorinsky", dye=False, perturb=True,
                         bodies=[_sphere(0.6, name="hot", thermal="temperature", thermal_value=1.0)],
                         thermal=ThermalConfig3D(prandtl=0.71))
    s = FlowSolver3D(cfg)
    s.advance(5)
    th = s.physics.thermal
    nut = torch.full((12 + 2, 12 + 2, 24 + 2), 0.01, dtype=torch.float64)
    kx, _, _ = th.face_kappa(nut, s.ops)
    expect = th.kappa / s.ops.h + 0.01 / (th.prt * s.ops.h)
    assert float(kx[..., 5].mean()) == pytest.approx(expect, rel=1e-12)
    assert float(kx[..., 0].mean()) == pytest.approx(th.kappa / s.ops.h, rel=1e-12)   # walls: molecular only
    assert math.isfinite(s.heat_rates()[0]["heat_rate"])


def test_tunnel_inlet_temperature_and_symmetry_plane_are_handled():
    cfg = SolverConfig3D(nx=16, ny=4, nz=8, h=0.25, x0=0, y0=0, z0=-1, re=20.0, device="cpu", dtype="float64",
                         bc_y="symmetry_ymin", dye=False, perturb=False,
                         thermal=ThermalConfig3D(prandtl=1.0, theta_inlet=1.0, walls={"front": ("temperature", 3.0)}))
    s = FlowSolver3D(cfg)
    assert s.physics.thermal.side_kinds == ("inflow", "outflow", "adiabatic", "adiabatic", "adiabatic", "adiabatic")
    assert any("front" in n for n in s.notes)                     # the symmetry plane overrides the wall value
    s.advance(400)
    assert float(s.theta.min()) == pytest.approx(1.0, abs=1e-9)   # uniform flow carries the inlet value
    assert float(s.theta.max()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# fans
# ---------------------------------------------------------------------------

def _periodic_fan_cfg(**fan):
    return SolverConfig3D(nx=16, ny=16, nz=16, h=0.25, x0=-2, y0=-2, z0=-2, re=100.0, periodic=True, device="cpu",
                          dtype="float64", dye=False, fans=[Fan3D(center=(0.1, 0.0, -0.1), **fan)])


@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (0.3, -0.5, 0.8)])
def test_fan_adds_exactly_its_thrust_to_a_periodic_box(direction):
    s = FlowSolver3D(_periodic_fan_cfg(direction=direction, diameter=1.5, mode="thrust", value=0.2, swirl=0.0))
    s.advance(30)
    h3 = s.ops.h ** 3
    st = s.state
    mom = np.array([float(st.u[..., :-1].sum()), float(st.v[:, :-1].sum()), float(st.w[:-1].sum())]) * h3
    n = np.asarray(direction) / np.linalg.norm(direction)
    np.testing.assert_allclose(mom, 0.2 * s.t * n, rtol=1e-11, atol=1e-13)
    rep = s.fans_report()[0]
    assert rep["thrust"] == pytest.approx(0.2) and rep["pressure_jump"] == pytest.approx(0.2 / (math.pi * 0.5625))
    assert rep["flow_rate"] == rep["flow"]


def test_fan_swirl_adds_no_net_force_and_spins_the_flow():
    s = FlowSolver3D(_periodic_fan_cfg(direction=(1.0, 0.0, 0.0), diameter=1.5, mode="thrust", value=0.2,
                                       swirl=0.5))
    s.advance(40)
    h3 = s.ops.h ** 3
    st = s.state
    assert float(st.u[..., :-1].sum()) * h3 == pytest.approx(0.2 * s.t, rel=1e-10)
    assert abs(float(st.v[:, :-1].sum()) * h3) < 1e-12 and abs(float(st.w[:-1].sum()) * h3) < 1e-12
    # right-hand rule about +x: above the axis (z > 0) the swirl velocity n x r points along -y
    v = s.field("v").numpy()
    zc = (np.arange(16) + 0.5) * 0.25 - 2
    above = v[(zc > 0.2) & (zc < 0.7)][:, 6:10, 7:10].mean()
    below = v[(zc < -0.4) & (zc > -0.9)][:, 6:10, 7:10].mean()
    assert above < -0.01 and below > 0.01


def test_velocity_regulated_fan_reaches_its_target():
    cfg = SolverConfig3D(nx=36, ny=12, nz=12, h=1 / 6, x0=-2, y0=-1, z0=-1, re=200.0, device="cpu", dtype="float64",
                         dye=False, perturb=False,
                         fans=[Fan3D(center=(0, 0, 0), direction=(1, 0, 0), diameter=0.8, mode="velocity", value=1.4)])
    s = FlowSolver3D(cfg)
    s.advance(600)
    rep = s.fan_report()[0]
    assert rep["disc_velocity"] == pytest.approx(1.4, rel=0.01)
    assert rep["induced_velocity"] == pytest.approx(0.4, rel=0.03)
    assert rep["thrust"] > 0
    ph = s.physics_history()
    assert len(ph["fan0_thrust"]) == s.steps


def test_fan_footprint_integrates_to_the_thrust_also_on_a_symmetry_plane():
    for half in (False, True):
        kw = {"ny": 6, "y0": 0.0, "bc_y": "symmetry_ymin"} if half else {"ny": 12, "y0": -1.0}
        cfg = SolverConfig3D(nx=24, nz=12, h=1 / 6, x0=-2, z0=-1, re=50.0, device="cpu", dtype="float64",
                             dye=False, fans=[Fan3D(center=(0, 0, 0), direction=(1, 0, 0), diameter=1.0, value=0.3)],
                             **kw)
        ops = core.make_operators3d(cfg)
        fs = P.FanSet(cfg.fans, ops, cfg)
        it = fs.items[0]
        total = float(it["fu"].sum()) * ops.h ** 3
        assert total == pytest.approx(0.5 if half else 1.0, rel=1e-12)       # times the thrust
        assert it["frac"] == pytest.approx(0.5 if half else 1.0, rel=1e-12)
        s = FlowSolver3D(cfg)
        s.advance(2)
        assert s.fan_report()[0]["thrust"] == pytest.approx(0.3, rel=1e-12)   # mirrored to the full fan


def test_fan_outside_the_domain_is_dropped_with_a_note():
    cfg = SolverConfig3D(nx=8, ny=8, nz=8, h=0.25, x0=0, y0=0, z0=0, re=10.0, device="cpu", dtype="float64",
                         dye=False, fans=[Fan3D(center=(10.0, 0, 0), name="far")])
    s = FlowSolver3D(cfg)
    assert s.physics.fans is None and any("far" in n for n in s.notes)


# ---------------------------------------------------------------------------
# jets and vents
# ---------------------------------------------------------------------------

def test_port_footprints_carry_exact_flow_and_momentum():
    cfg = SolverConfig3D(nx=16, ny=16, nz=16, h=1 / 8, x0=0, y0=0, z0=0, re=50.0, device="cpu", dtype="float64",
                         dye=False, ports=[Port3D(kind="jet", center=(0.6, 1.0, 1.0), direction=(1, 1, 0),
                                                  diameter=0.5, speed=2.0)])
    ops = core.make_operators3d(cfg)
    ps = P.PortSet(cfg.ports, ops, cfg)
    it = ps.items[0]
    h3 = ops.h ** 3
    q = 2.0 * math.pi * 0.0625
    assert float(it["m"].sum()) * h3 == pytest.approx(q, rel=1e-12)
    n = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    flux = np.array([float(it[k].sum()) * h3 for k in ("fu", "fv", "fw")])
    np.testing.assert_allclose(flux, q * 2.0 * n, rtol=1e-12, atol=1e-15)


def test_jet_flow_leaves_through_the_outlet_and_continuity_holds():
    port = Port3D(kind="jet", center=(1.0, 0.0, 0.0), direction=(0, 0, 1), diameter=0.5, speed=1.5, theta=1.0,
                  dye_channel=1)
    cfg = SolverConfig3D(nx=32, ny=12, nz=12, h=1 / 6, x0=0, y0=-1, z0=-1, re=100.0, device="cpu", dtype="float64",
                         perturb=False, ports=[port], thermal=ThermalConfig3D(prandtl=0.71),
                         emitters=[Emitter3D(kind="point", position=(0.5, 0.0, 0.5))], diagnostics=True)
    s = FlowSolver3D(cfg)
    s.advance(60)
    st = s.state
    h2 = s.ops.h ** 2
    inflow = float(st.u[..., 0].sum()) * h2
    outflow = float(st.u[..., -1].sum()) * h2
    assert outflow - inflow == pytest.approx(port.flow, rel=1e-10)
    assert float(s.field("divergence").abs().max()) < 1e-10
    assert s.latest_diagnostics()["div_rms"] < 1e-11
    assert float(s.field("dye1").max()) > 0.1 and s.dye_channels == 2           # the jet's smoke channel
    hb = s.heat_balance()
    assert hb["heat_in_ports"] == pytest.approx(port.flow * 1.0 * s.physics.thermal.pe, rel=1e-10)


def test_balanced_jet_and_vent_in_a_closed_room():
    ports = [Port3D(kind="jet", center=(0.2, 0.5, 0.5), direction=(1, 0, 0), diameter=0.3, speed=1.0),
             Port3D(kind="vent", center=(0.8, 0.5, 0.5), diameter=0.3, speed=1.0)]
    s = FlowSolver3D(_box(16, re=100.0, ports=ports))
    assert not any("balance" in n for n in s.notes)
    s.advance(80)
    assert float(s.field("divergence").abs().max()) < 1e-10
    assert float(s.field("u").max()) > 0.2                                      # the room is ventilated
    unbalanced = FlowSolver3D(_box(8, ports=ports[:1]))
    assert any("balance" in n for n in unbalanced.notes)


# ---------------------------------------------------------------------------
# CUDA: fused kernels and graphs with every feature on
# ---------------------------------------------------------------------------

def _everything(device, dtype, **kw) -> SolverConfig3D:
    v, f = cases.icosphere(0.6, 2)
    base = {
        "nx": 40, "ny": 16, "nz": 16, "h": 1 / 8, "x0": -1.5, "y0": -1.0, "z0": -1.0, "re": 80.0, "device": device,
        "dtype": dtype, "bodies": [ImmersedMesh(v, f, name="hot", thermal="temperature", thermal_value=1.0)],
        "ref_area": math.pi * 0.09, "perturb": True, "diagnostics": True, "turbulence": "wale",
        "thermal": ThermalConfig3D(prandtl=0.71, richardson=0.3, walls={"bottom": ("temperature", 0.2)}),
        "heat_sources": [HeatSource3D(kind="sphere", center=(1.5, 0.3, 0.2), size=(0.25,), power=2.0)],
        "fans": [Fan3D(center=(-0.8, 0.0, 0.4), direction=(1, 0, 0.2), diameter=0.5, mode="velocity", value=1.3,
                       swirl=0.4)],
        "ports": [Port3D(kind="jet", center=(0.5, -0.5, -0.5), direction=(0, 0, 1), diameter=0.3, speed=0.8,
                         theta=0.5, dye_channel=1),
                  Port3D(kind="vent", center=(2.5, 0.5, 0.5), diameter=0.3, speed=0.4)],
        "emitters": [Emitter3D(kind="rake", position=(-1.2, -0.6, 0), end=(-1.2, 0.6, 0), count=4),
                     Emitter3D(kind="disc", position=(-1.2, 0.0, -0.5), size=(0.2,), channel=2, pulse_period=0.4)],
    }
    base.update(kw)
    return SolverConfig3D(**base)


def _max_rel(a: FlowSolver3D, b: FlowSolver3D) -> dict[str, float]:
    out = {}
    for name in ("u", "v", "w", "p", "temperature", "dye", "dye1", "dye2"):
        x, y = a.field(name).double().cpu(), b.field(name).double().cpu()
        out[name] = float((x - y).abs().max() / max(float(y.abs().max()), 1e-30))
    out["heat"] = abs(a.heat_rates()[0]["heat_rate"] - b.heat_rates()[0]["heat_rate"])
    out["thrust"] = abs(a.fan_report()[0]["thrust"] - b.fan_report()[0]["thrust"])
    return out


@needs_cuda
def test_fused_kernels_match_the_reference_with_every_feature():
    ref = FlowSolver3D(_everything("cuda", "float64", fused_kernels=False, use_cuda_graphs=False))
    fus = FlowSolver3D(_everything("cuda", "float64", use_cuda_graphs=False))
    assert fus.uses_fused_kernels and fus.theta_error is None
    ref.advance(12)
    fus.advance(12)
    errs = _max_rel(fus, ref)
    assert max(v for k, v in errs.items() if k not in ("p",)) < 1e-12, errs
    assert errs["p"] < 1e-10, errs


@needs_cuda
def test_cuda_graphs_replay_the_eager_run_bit_for_bit():
    parts = Emitter3D(species="particles", kind="plane", position=(-1.2, 0, 0), size=(1.0, 1.0), rate=500.0)
    cfg = _everything("cuda", "float32")
    cfg.emitters = list(cfg.emitters) + [parts]
    torch.manual_seed(3)
    eager = FlowSolver3D(_everything("cuda", "float32", use_cuda_graphs=False, emitters=cfg.emitters))
    eager.start_averaging()
    eager.advance(34)
    torch.manual_seed(3)
    graph = FlowSolver3D(cfg)
    graph.start_averaging()
    graph.advance(34)
    assert graph.uses_cuda_graphs and graph.graph_error is None
    errs = _max_rel(graph, eager)
    assert max(errs.values()) == 0.0, errs
    assert len(graph.diagnostics()["t"]) == len(eager.diagnostics()["t"]) == 3
    assert graph.particle_count() > 0


@needs_cuda
def test_cpu_and_cuda_reference_agree():
    a = FlowSolver3D(_everything("cpu", "float64"))
    b = FlowSolver3D(_everything("cuda", "float64", fused_kernels=False, use_cuda_graphs=False))
    a.advance(6)
    b.advance(6)
    errs = _max_rel(b, a)
    assert max(v for k, v in errs.items() if k != "p") < 1e-12, errs
