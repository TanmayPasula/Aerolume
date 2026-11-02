"""Synthetic 3D results for the report tests (no solver, no GPU).

Shaped like real output so layout and wording problems show: a sphere in
potential flow (Cp = 1 at the nose, -1.25 at the equator) with a skin-friction
map, three plane cuts and a volume with a Q-criterion blob in the wake; force
histories with a start-up transient and noise; a rounded "vehicle" on four
wheels run with a symmetry plane (side force, roll and yaw exactly zero); and
finite-wing sweeps whose lift curve, induced drag and span loading follow
lifting-line theory with a parabolic profile drag.
"""
from __future__ import annotations

import math

import numpy as np
import report_fixtures as rf
from test_viz3d_fixtures import sphere_flow_volume, sphere_surface, vehicle_surfaces

from aerolume.geometry3d import design_from_meshes, design_from_wing
from aerolume.model import (
    CFD3DResult,
    CFDSettings,
    FlowConditions,
    GroundConfig,
    Mesh3D,
    Simulation,
    SliceData,
    Stat,
    SurfaceData3D,
    VolumeData,
    WingPoint,
    WingResult,
    WingSection,
    WingSettings,
    WingSpec,
)

AIR = rf.AIR


def history3d(seed: int, cd: float, cl: float, *, t_end: float = 20.0, n: int = 1500,
              lateral: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, n)
    decay = np.exp(-t / 2.0)
    h = {"t": t}
    for k, mean, amp in (("cd", cd, 0.02), ("cs", 0.0, 0.01), ("cl", cl, 0.03), ("c_roll", 0.0, 0.002),
                         ("cm", -0.05, 0.01), ("cn", 0.0, 0.003)):
        if k in ("cs", "c_roll", "cn") and not lateral:
            h[k] = np.zeros(n)
            continue
        h[k] = mean + 0.6 * abs(mean or 0.1) * decay + amp * np.sin(2.1 * t) + 0.3 * amp * rng.standard_normal(n)
    return h


def _stat(mean: float, se: float, rms: float = 0.01) -> Stat:
    return Stat(mean=mean, rms=rms, min=mean - 3 * rms, max=mean + 3 * rms, stderr=se, stderr_reliable=True)


def _slices(vol: VolumeData) -> list[SliceData]:
    """y = 0, z = 0 and x = 1 m cuts of a full (not mirrored) volume, with the solver's field set."""
    f = vol.fields
    nz, ny, nx = f["u"].shape
    o, h = vol.origin, vol.spacing
    ext_x = (o[0], o[0] + nx * h[0])
    ext_y = (o[1], o[1] + ny * h[1])
    ext_z = (o[2], o[2] + nz * h[2])
    out = []
    for axis, idx, pos in (("y", ny // 2, o[1] + (ny // 2 + 0.5) * h[1]), ("z", nz // 2, o[2] + (nz // 2 + 0.5) * h[2]),
                           ("x", min(nx - 1, int((1.0 - o[0]) / h[0])), 1.0)):
        if axis == "y":
            cut = {k: a[:, idx, :] for k, a in f.items()}
            m = vol.mask[:, idx, :]
            ext = (*ext_x, *ext_z)
        elif axis == "z":
            cut = {k: a[idx, :, :] for k, a in f.items()}
            m = vol.mask[idx, :, :]
            ext = (*ext_x, *ext_y)
        else:
            cut = {k: a[:, :, idx] for k, a in f.items()}
            m = vol.mask[:, :, idx]
            ext = (*ext_y, *ext_z)
        cut = {k: np.asarray(a, np.float32) for k, a in cut.items()}
        cut.setdefault("p", cut["cp"] * 0.5 * AIR.density * 10.0 ** 2)
        cut.setdefault("vorticity", np.abs(np.gradient(cut["u"], axis=0)))
        out.append(SliceData(axis=axis, position=float(pos), extent=tuple(float(e) for e in ext), fields=cut,
                             mask=np.asarray(m, np.float32)))
    return out


def sphere_volume() -> VolumeData:
    vol = sphere_flow_volume(n=(64, 40, 40), d=1.0, speed=10.0)
    xx = vol.origin[0] + (np.arange(64) + 0.5) * vol.spacing[0]
    yy = vol.origin[1] + (np.arange(40) + 0.5) * vol.spacing[1]
    zz = vol.origin[2] + (np.arange(40) + 0.5) * vol.spacing[2]
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    ring = np.exp(-(((x - 1.0) / 0.25) ** 2) - ((np.hypot(y, z) - 0.35) / 0.12) ** 2)
    vol.fields["q"] = (400.0 * ring - 20.0 * np.exp(-((x - 1.0) / 0.3) ** 2 - (y ** 2 + z ** 2) / 0.05)).astype(np.float32)
    return vol


def sphere_design():
    from aerolume.geometry3d import sphere
    return design_from_meshes([sphere(1.0, 48)], "Sphere 1 m", kind="generic")


def sphere_result(sim: Simulation, *, seed: int = 1, cd: float = 0.47, sensitivity: bool = True) -> CFD3DResult:
    s = sphere_surface(1.0, 24, 48)
    v = np.asarray(s.vertices)
    theta = np.arccos(np.clip(-v[:, 0] / 0.5, -1, 1))            # 0 at the nose
    cf = (0.08 * np.sin(theta) * np.exp(-((theta - 1.6) / 0.6) ** 2)).astype(float)
    tdir = np.column_stack([np.sin(theta), -np.cos(theta) * v[:, 1], -np.cos(theta) * v[:, 2]])
    tdir /= np.maximum(np.linalg.norm(tdir, axis=1, keepdims=True), 1e-9)
    surf = SurfaceData3D(body="Sphere", vertices=v, faces=s.faces, cp=s.cp, cf=cf, tau=tdir * cf[:, None],
                         sensitivity=(np.cos(theta) * 1.5 + 0.2 * v[:, 2]) if sensitivity else None)
    vol = sphere_volume()
    q = 0.5 * AIR.density * sim.conditions.speed ** 2
    A, L = math.pi / 4, 1.0
    r = CFD3DResult(sim_id=sim.id, cd=_stat(cd, 0.004, 0.02), cs=_stat(0.001, 0.002), cl=_stat(0.012, 0.006, 0.03),
                    c_roll=_stat(0.0, 0.0005), cm=_stat(-0.05, 0.003), cn=_stat(0.0, 0.0008))
    r.drag, r.side, r.lift = cd * q * A, 0.001 * q * A, 0.012 * q * A
    r.moments = (0.0, -0.05 * q * A * L, 0.0)
    r.drag_area, r.power = cd * A, cd * q * A * sim.conditions.speed
    r.strouhal, r.strouhal_length = 0.19, 1.0
    r.re_real = r.re_sim = sim.conditions.reynolds(L)
    r.grid, r.cells, r.cells_per_ref, r.symmetry_plane = (240, 96, 96), 240 * 96 * 96, 24.0, False
    r.t_end, r.t_avg_start, r.converged, r.wall_seconds = 20.0, 6.0, True, 95.0
    r.device, r.precision, r.turbulence = "GPU · NVIDIA GeForce RTX 4060 Ti (8.0 GB)", "float32", "les-wale"
    r.notes = ["Grid: 240 × 96 × 96 = 2,211,840 cells; 24 cells per reference length.",
               ("Flow model: at Re = 6.85×10^5 a WALE large-eddy model represents the unresolved scales -- "
               "third-order upwind advection keeps the run stable.")]
    r.warnings = ["Only 24 cells per frontal size: forces are indicative."]
    r.per_body = {"Sphere": {"cd": cd, "cs": 0.001, "cl": 0.012, "c_roll": 0.0, "cm": -0.05, "cn": 0.0}}
    r.history = history3d(seed, cd, 0.012)
    r.surface = [surf]
    r.slices = _slices(vol)
    r.volume = vol
    return r


def vehicle_design():
    meshes = [Mesh3D(s.body, s.vertices, s.faces, role="wheel" if s.body.startswith("wheel") else "body")
              for s in vehicle_surfaces()]
    return design_from_meshes(meshes, "Rounded van", kind="vehicle", on_road=True, clearance=0.0)


def vehicle_result(sim: Simulation, design, *, cd: float = 0.62) -> CFD3DResult:
    surfs = vehicle_surfaces()
    q = 0.5 * AIR.density * sim.conditions.speed ** 2
    A, L = design.ref_area, design.length_ref
    r = CFD3DResult(sim_id=sim.id, cd=_stat(cd, 0.008, 0.03), cl=_stat(-0.08, 0.01, 0.05),
                    cm=_stat(0.04, 0.005), cs=Stat(0.0, 0.0, 0.0, 0.0, 0.0), c_roll=Stat(0.0, 0.0, 0.0, 0.0, 0.0),
                    cn=Stat(0.0, 0.0, 0.0, 0.0, 0.0))
    r.cd.stderr_reliable = False
    r.drag, r.side, r.lift = cd * q * A, 0.0, -0.08 * q * A
    r.moments = (0.0, 0.04 * q * A * L, 0.0)
    r.drag_area, r.power = cd * A, cd * q * A * sim.conditions.speed
    r.re_real = r.re_sim = sim.conditions.reynolds(L)
    r.grid, r.cells, r.cells_per_ref, r.symmetry_plane = (300, 76, 84), 300 * 76 * 84, 68.0, True
    r.t_end, r.t_avg_start, r.converged, r.wall_seconds = 20.0, 5.0, False, 206.0
    r.device, r.precision, r.turbulence = "GPU · NVIDIA GeForce RTX 4060 Ti (8.0 GB)", "float32", "les-wale"
    r.per_body = {s.body: {"cd": cd * (0.82 if s.body == "body" else 0.045), "cs": 0.0, "cl": -0.02,
                           "c_roll": 0.0, "cm": 0.01, "cn": 0.0} for s in surfs}
    r.history = history3d(7, cd, -0.08, lateral=False)
    r.surface = surfs
    return r


def wing_design(tapered: bool = True):
    if tapered:
        spec = WingSpec([WingSection(0.0, 1.0, airfoil="2412"),
                         WingSection(5.0, 0.55, x_le=0.15, twist_deg=-2.0, airfoil="2412")])
        return design_from_wing(spec, "Tapered wing")
    spec = WingSpec([WingSection(0.0, 0.8, airfoil="2412"), WingSection(4.0, 0.8, airfoil="2412")])
    return design_from_wing(spec, "Rectangular wing")


def wing_result(sim: Simulation, design, *, viscous: bool = True, e_span: float = 0.97) -> WingResult:
    """Lifting-line-shaped sweep: CL = a (α - α0), CDi = CL²/(π AR e), CDv parabolic, span loading near elliptic."""
    area, span = design.ref_area, design.span_ref
    ar = span * span / area
    a = 2 * math.pi * ar / (ar + 2)
    pts = []
    y = 0.5 * span * np.cos(np.linspace(math.pi, 0, 41)[:-1] + math.pi / 80)
    eta = 2 * y / span
    for al in sim.wing.alphas():
        cl = a * math.radians(al + 2.1) * (1 - 0.15 * max(0.0, al - 10) / 4)
        cdi = cl * cl / (math.pi * ar * e_span)
        cdv = (0.0062 + 0.004 * (cl - 0.2) ** 2) if viscous else 0.0
        c_cl = 4 / math.pi * cl * np.sqrt(np.clip(1 - eta ** 2, 0, None)) * (1 + 0.05 * (1 - eta ** 2))
        pts.append(WingPoint(alpha=float(al), cl=cl, cd=cdi + cdv, cdi=cdi, cdv=cdv, cm=-0.05 + 0.0005 * al, cy=0.0,
                             c_roll=0.0, cn=0.0, e=e_span, converged=True, y=y, cl_local=c_cl * 0.9,
                             c_cl=c_cl, alpha_i=np.full_like(y, math.degrees(cl / (math.pi * ar))),
                             cd_local=np.full_like(y, cdv) if viscous else None))
    return WingResult(sim_id=sim.id, method=sim.wing.method, viscous=viscous, area=area, span=span, aspect_ratio=ar,
                      mac=design.length_ref, re_mac=sim.conditions.reynolds(design.length_ref), points=pts,
                      wall_seconds=23.4, notes=["Vortex lattice: 8 chordwise x 48 spanwise ring vortices."])


def cfd3d_sim(design, name: str, sid: str, speed: float = 10.0, ground: bool = False) -> Simulation:
    return Simulation(name=name, design_id=design.id, analysis="cfd", id=sid, status="done",
                      conditions=FlowConditions(AIR, speed, 0.0),
                      ground=GroundConfig(enabled=ground, moving=True, rotate_wheels=True),
                      cfd=CFDSettings(quality="draft", compute_sensitivity=True),
                      created="2026-10-06T09:00:00", finished="2026-10-06T09:03:00")


def wing_sim(design, name: str, sid: str, method: str = "vlm", viscous: bool = True) -> Simulation:
    return Simulation(name=name, design_id=design.id, analysis="wing", id=sid, status="done",
                      conditions=FlowConditions(AIR, 30.0, 0.0),
                      wing=WingSettings(alpha_start=-4, alpha_end=14, alpha_step=2, method=method, viscous=viscous),
                      created="2026-10-06T09:10:00", finished="2026-10-06T09:10:30")


def make_project3d() -> rf.FakeProject:
    """Two sphere runs (3D comparison), a symmetric vehicle run and two wing sweeps (wing comparison)."""
    pr = rf.FakeProject("3D study")
    ds = sphere_design()
    s1 = cfd3d_sim(ds, "Sphere · 10 m/s", "sph-a")
    s2 = cfd3d_sim(ds, "Sphere · 12 m/s", "sph-b", speed=12.0)
    pr.add(ds, s1, sphere_result(s1, seed=1))
    pr.add(ds, s2, sphere_result(s2, seed=2, cd=0.45, sensitivity=False))
    dv = vehicle_design()
    sv = cfd3d_sim(dv, "Van · 30 m/s", "van", speed=30.0, ground=True)
    pr.add(dv, sv, vehicle_result(sv, dv))
    dw1, dw2 = wing_design(True), wing_design(False)
    w1 = wing_sim(dw1, "Tapered · VLM", "w-tap")
    w2 = wing_sim(dw2, "Rectangular · LLT", "w-rect", method="llt")
    pr.add(dw1, w1, wing_result(w1, dw1))
    pr.add(dw2, w2, wing_result(w2, dw2, e_span=0.93))
    return pr
