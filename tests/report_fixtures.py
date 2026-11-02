"""Realistic synthetic results for report and figure tests.

The report writer must be testable without running the solver, and a report built
from toy numbers (straight lines, random noise) hides layout problems: real Cp
curves have a suction spike that sets the axis range, real histories have an
impulsive-start spike, real polars stall. So these fixtures are physically shaped:

* airfoil Cp comes from the exact Karman-Trefftz-free Joukowski potential solution
  (with the Kutta condition) at a matching thickness/camber, interpolated onto the
  NACA surface, then given a viscous decambering and a trailing-edge recovery;
* force histories have an impulsive start, an exponential transient, vortex shedding
  at a realistic Strouhal number (drag at twice the lift frequency) and noise;
* mean fields are a doublet + bound vortex potential flow plus a spreading Gaussian
  wake, so contour plots and streamlines look like what the solver produces;
* polars follow thin-airfoil lift slope with a smooth stall, a parabolic drag bucket,
  moving transition points and two unconverged post-stall points.

Everything is deterministic (seeded) and small (fields ~ 320 x 160) so tests are fast.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.path import Path as MplPath

from aerolume.model import (
    AirfoilSettings,
    Body,
    CFDResult,
    CFDSettings,
    Design,
    FieldData,
    FlowConditions,
    Fluid,
    GroundConfig,
    PolarPoint,
    PolarResult,
    Simulation,
    Stat,
    SurfaceData,
)

AIR = Fluid("Air, ISA sea level", 1.225, 1.7894e-5)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def naca4(code: str, n: int = 81, chord: float = 1.0) -> np.ndarray:
    """NACA 4-digit section, closed TE, CCW from the TE over the upper surface (no repeated point)."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0, math.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    upper = np.column_stack([xu, yu])[::-1]          # TE -> LE
    lower = np.column_stack([xl, yl])[1:-1]          # LE -> TE, skip shared LE and TE
    return np.vstack([upper, lower]) * chord


def circle(cx: float, cy: float, r: float, n: int = 48) -> np.ndarray:
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def _chaikin(p: np.ndarray, iterations: int = 3) -> np.ndarray:
    for _ in range(iterations):
        q = np.roll(p, -1, axis=0)
        p = np.vstack([np.column_stack([0.75 * p[:, 0] + 0.25 * q[:, 0], 0.75 * p[:, 1] + 0.25 * q[:, 1]]),
                       np.column_stack([0.25 * p[:, 0] + 0.75 * q[:, 0], 0.25 * p[:, 1] + 0.75 * q[:, 1]])])
        # interleave so the polygon order is preserved
        n = len(p) // 2
        p = np.stack([p[:n], p[n:]], axis=1).reshape(-1, 2)
    return p


def sedan_profile() -> np.ndarray:
    """Side profile of a generic 4.5 m saloon, CCW (underbody first, nose at the left)."""
    key = np.array([
        (0.06, 0.32), (0.35, 0.18), (1.10, 0.15), (3.50, 0.15), (4.20, 0.22), (4.45, 0.40),
        (4.50, 0.80), (4.42, 0.98), (3.95, 1.03), (3.35, 1.38), (2.05, 1.42), (1.45, 1.05),
        (0.40, 0.88), (0.08, 0.70), (0.00, 0.52),
    ])
    return _chaikin(key, 3)


def airfoil_design(code: str = "2412", chord: float = 0.3, name: str | None = None) -> Design:
    pts = naca4(code, 81, chord)
    return Design(name=name or f"NACA {code}", kind="airfoil", bodies=[Body("airfoil", pts)],
                  length_ref=chord, force_ref=chord, moment_ref=(0.25 * chord, 0.0),
                  source={"type": "naca", "code": code}, id=f"d-naca{code}")


def vehicle_design() -> Design:
    r = 0.33
    bodies = [Body("body", sedan_profile()),
              Body("front wheel", circle(0.95, r, r), role="wheel", center=(0.95, r)),
              Body("rear wheel", circle(3.55, r, r), role="wheel", center=(3.55, r))]
    return Design(name="Saloon concept A", kind="vehicle", bodies=bodies, length_ref=4.5, force_ref=1.42,
                  moment_ref=(2.25, 0.0), source={"type": "template", "name": "saloon"}, id="d-saloon")


def rotate(points: np.ndarray, alpha_deg: float, about: tuple[float, float]) -> np.ndarray:
    a = -math.radians(alpha_deg)
    c, s = math.cos(a), math.sin(a)
    p = points - np.asarray(about)
    return np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]]) + np.asarray(about)


def outward_normals(p: np.ndarray) -> np.ndarray:
    e = np.roll(p, -1, axis=0) - p
    en = np.column_stack([e[:, 1], -e[:, 0]])
    en /= np.linalg.norm(en, axis=1, keepdims=True)
    vn = en + np.roll(en, 1, axis=0)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# airfoil surface pressure: Joukowski potential flow + viscous corrections
# ---------------------------------------------------------------------------

def joukowski_cp(alpha_deg: float, thickness: float, camber: float, n: int = 721) -> dict[str, np.ndarray]:
    """Exact potential-flow Cp of a Joukowski airfoil with the Kutta condition.

    Returns chordwise x/c (0 = LE) and Cp for the upper and lower surface, sorted by x/c.
    ``thickness`` ~ t/c and ``camber`` ~ m/c via the classic small-parameter relations.
    """
    eps = thickness / 1.3            # t/c ~ 1.3 eps for a Joukowski section
    mu = complex(-eps, 2.0 * camber)
    rad = abs(1.0 - mu)
    beta = math.asin(mu.imag / rad)
    a = math.radians(alpha_deg)
    gamma = 4 * math.pi * rad * math.sin(a + beta)
    th = np.linspace(-beta + 1e-3, -beta + 2 * math.pi - 1e-3, n)
    zeta = mu + rad * np.exp(1j * th)
    w = np.exp(-1j * a) - rad**2 * np.exp(1j * a) / (zeta - mu) ** 2 + 1j * gamma / (2 * math.pi * (zeta - mu))
    z = zeta + 1.0 / zeta
    vel = w / (1.0 - 1.0 / zeta**2)
    cp = 1.0 - np.abs(vel) ** 2
    x = z.real
    x = (x - x.min()) / (x.max() - x.min())
    i_le = int(np.argmin(x))
    up = slice(0, i_le + 1)          # theta from TE over the top to the LE
    lo = slice(i_le, n)
    ux, ucp = x[up][::-1], cp[up][::-1]
    lx, lcp = x[lo], cp[lo]
    return {"xu": ux, "cpu": ucp, "xl": lx, "cpl": lcp}


def airfoil_surface_cp(xc: np.ndarray, upper: np.ndarray, alpha_deg: float, code: str,
                       viscous: bool = True) -> np.ndarray:
    """Cp at chordwise stations ``xc`` (bool ``upper`` per station) for a NACA 4-digit code."""
    jc = joukowski_cp(alpha_deg * (0.88 if viscous else 1.0), int(code[2:]) / 100, int(code[0]) / 100 * 0.9)
    cp = np.where(upper, np.interp(xc, jc["xu"], jc["cpu"]), np.interp(xc, jc["xl"], jc["cpl"]))
    if viscous:
        # boundary-layer displacement: the TE pressure recovers to ~+0.15 rather than the cusp value
        w = np.clip((xc - 0.8) / 0.2, 0, 1) ** 2
        cp = (1 - w) * cp + w * 0.15
    return cp


def _skin_friction(xc: np.ndarray, re: float, x_tr: float) -> np.ndarray:
    rex = np.maximum(xc, 1e-3) * re
    lam = 0.664 / np.sqrt(rex)
    turb = 0.0592 * rex ** -0.2
    blend = 1 / (1 + np.exp(-(xc - x_tr) / 0.015))
    return (1 - blend) * lam + blend * turb


# ---------------------------------------------------------------------------
# synthetic flow fields
# ---------------------------------------------------------------------------

def synthetic_fields(bodies: list[Body], extent: tuple[float, float, float, float], speed: float, rho: float,
                     nx: int = 320, ny: int = 160, cl: float = 0.5, ground: bool = False) -> FieldData:
    """Doublet + vortex potential flow around the bodies' main outline plus a spreading wake."""
    x0, x1, y0, y1 = extent
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    main = bodies[0].points
    cy = float(main[:, 1].mean())
    length = np.ptp(main[:, 0])
    height = np.ptp(main[:, 1])
    a2 = (0.5 * height) * (0.5 * length) * 0.9           # doublet strength ~ blockage of the body
    gam = 0.5 * cl * speed * length                      # Kutta-Joukowski: L' = rho U Gamma
    # a line of regularised doublets + vortices along the chord spreads the disturbance over
    # the body length, as a real thickness/load distribution does (one point doublet gives a blob)
    nline = 16
    xsrc = main[:, 0].min() + (np.arange(nline) + 0.5) / nline * length
    core = (0.45 * height) ** 2 + (0.02 * length) ** 2
    u = np.full_like(X, speed)
    v = np.zeros_like(X)
    for xk in xsrc:
        for yk, sgn in ((cy, 1.0), (-cy, 1.0)) if ground else ((cy, 1.0),):
            dx, dy = X - xk, Y - yk
            r2 = dx**2 + dy**2 + core
            u += sgn * speed * (-a2 / nline * (dx**2 - dy**2) / r2**2)
            v += sgn * speed * (-a2 / nline * 2 * dx * dy / r2**2)
        dx, dy = X - xk, Y - cy
        r2 = dx**2 + dy**2 + core
        w = 2.0 * (1 - (xk - xsrc[0]) / length)          # front-loaded, like a real airfoil
        u += w * gam / nline / (2 * math.pi) * dy / r2
        v += -w * gam / nline / (2 * math.pi) * dx / r2
    xte = main[:, 0].max()
    xw = np.maximum(X - xte, 0.0)
    width = 0.35 * height + 0.12 * np.sqrt(xw * length) + 0.02 * length
    yw = cy - 0.15 * cl * xw * 0.2
    deficit = 0.55 * np.exp(-((Y - yw) / width) ** 2) * (0.5 * height + 0.05 * length) / width
    deficit *= np.where(X > xte - 0.1 * length, 1.0, np.exp(-((X - xte + 0.1 * length) / (0.1 * length)) ** 2))
    u -= speed * np.clip(deficit, 0, 0.95)
    speed_f = np.hypot(u, v)
    q = 0.5 * rho * speed**2
    p = q * (1 - (speed_f / speed) ** 2)
    p = np.where(deficit > 0.05, np.minimum(p, -0.25 * q * np.clip(deficit * 3, 0, 1)), p)
    mask = np.zeros((ny, nx))
    pts = np.column_stack([X.ravel(), Y.ravel()])
    for b in bodies:
        mask = np.maximum(mask, MplPath(b.points).contains_points(pts).reshape(ny, nx).astype(float))
    u = np.where(mask > 0.5, 0.0, u)
    v = np.where(mask > 0.5, 0.0, v)
    hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
    vort = np.gradient(v, hx, axis=1) - np.gradient(u, hy, axis=0)
    fields = {"u": u, "v": v, "p": p, "cp": p / q, "vorticity": vort, "speed": np.hypot(u, v)}
    return FieldData(extent=extent, fields=fields, mask=mask, label="mean")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def _history(rng: np.random.Generator, t_end: float, cd: float, cl: float, cm: float, st_num: float,
             amp_cl: float, amp_cd: float) -> dict[str, np.ndarray]:
    t = np.arange(0.0, t_end, 0.01)
    start = 1.0 / np.sqrt(t + 0.02)                       # impulsive start
    trans = np.exp(-t / 4.0)
    ramp = 1 - np.exp(-t / 6.0)                           # shedding grows from the perturbation
    phase = 2 * math.pi * st_num * t
    out = {
        "t": t,
        "cd": cd * (1 + 0.6 * trans) + 0.08 * start * trans + amp_cd * ramp * np.sin(2 * phase + 0.3)
        + rng.normal(0, amp_cd * 0.15, t.size),
        "cl": cl * (1 - 0.7 * trans) + amp_cl * ramp * np.sin(phase) + rng.normal(0, amp_cl * 0.1, t.size),
        "cm": cm * (1 - 0.5 * trans) + 0.25 * amp_cl * ramp * np.sin(phase + 1.0)
        + rng.normal(0, amp_cl * 0.02, t.size),
    }
    return out


def _stat(t: np.ndarray, x: np.ndarray, t0: float) -> Stat:
    w = x[t >= t0]
    nb = 16
    batches = np.array_split(w, nb)
    bm = np.array([b.mean() for b in batches])
    return Stat(mean=float(w.mean()), rms=float(w.std()), min=float(w.min()), max=float(w.max()),
                stderr=float(bm.std(ddof=1) / math.sqrt(nb)))


def _surface_for(body: Body, cp: np.ndarray, cf: np.ndarray, sens: np.ndarray | None = None) -> SurfaceData:
    p = body.points
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    n = outward_normals(p)
    return SurfaceData(body=body.name, s=s, x=p[:, 0].copy(), y=p[:, 1].copy(), nx=n[:, 0], ny=n[:, 1],
                       cp=cp, cf=cf, sensitivity=sens)


def airfoil_cfd(design: Design, sim: Simulation, *, seed: int = 0, with_sensitivity: bool = False,
                field_res: tuple[int, int] = (320, 160)) -> CFDResult:
    """A converged LES-like airfoil result at sim.conditions, Re capped, with fields."""
    rng = np.random.default_rng(seed)
    code = design.source.get("code", "2412")
    alpha = sim.conditions.alpha_deg
    chord = design.length_ref
    cl_mean = 2 * math.pi * 0.9 * math.radians(alpha + 2.1 * int(code[0]) / 2)
    cd_mean = 0.021 + 0.012 * cl_mean**2
    cm_mean = -0.025 * int(code[0]) - 0.002
    t_end, t0 = 40.0, 15.0
    hist = _history(rng, t_end, cd_mean, cl_mean, cm_mean, 0.62, 0.035, 0.004)
    t = hist["t"]
    body = design.bodies[0]
    wind = Body(body.name, rotate(body.points, alpha, design.moment_ref))
    # chordwise station and side of every surface point (design frame: LE at x=0)
    pts = body.points
    xc = pts[:, 0] / chord
    i_le = int(np.argmin(pts[:, 0]))
    upper = np.arange(len(pts)) <= i_le
    cp = airfoil_surface_cp(xc, upper, alpha, code) + rng.normal(0, 0.004, len(pts))
    re_sim = 2.0e4
    cf_mag = _skin_friction(xc, re_sim, 0.35 if alpha < 6 else 0.15)
    cf_mag = np.where(upper & (xc > 0.85) & (alpha > 3), -0.002 * (xc - 0.85) / 0.15, cf_mag)  # TE bubble
    # Cf sign along s: the loop runs TE -> LE on the upper surface, i.e. against the flow there
    cf = np.where(upper, -cf_mag, cf_mag)
    sens = None
    if with_sensitivity:
        # drag sensitivity: strongest at the LE suction peak and the upper-surface recovery
        sens = (3.0 * np.exp(-(xc / 0.04)) * np.where(upper, 1.0, 0.4)
                - 1.2 * np.exp(-((xc - 0.7) / 0.12) ** 2) * upper + 0.6 * np.exp(-((xc - 0.98) / 0.03) ** 2))
        sens = sens / chord
    q = sim.conditions.dynamic_pressure()
    speed = sim.conditions.speed
    ext = (-1.0 * chord, 3.0 * chord, -1.0 * chord, 1.0 * chord)
    fields = synthetic_fields([wind], ext, speed, sim.conditions.fluid.density, *field_res, cl=cl_mean)
    inst = synthetic_fields([wind], ext, speed, sim.conditions.fluid.density, *field_res, cl=cl_mean * 1.1)
    inst.label = "instant"
    xx = np.linspace(ext[0], ext[1], field_res[0])[None, :]
    yy = np.linspace(ext[2], ext[3], field_res[1])[:, None]
    wake = np.exp(-((yy - 0.0) / (0.12 * chord)) ** 2) * (xx > chord)
    inst.fields["vorticity"] = inst.fields["vorticity"] + 6 * speed / chord * wake * np.sin(
        2 * math.pi * (xx - chord) / (0.6 * chord))
    cds, cls, cms = (_stat(t, hist[k], t0) for k in ("cd", "cl", "cm"))
    return CFDResult(
        sim_id=sim.id, cd=cds, cl=cls, cm=cms,
        drag=cds.mean * q * chord, lift=cls.mean * q * chord, moment=cms.mean * q * chord**2,
        drag_area=cds.mean * chord, power=cds.mean * q * chord * speed, strouhal=0.62,
        re_real=sim.conditions.reynolds(chord), re_sim=re_sim, grid=(1280, 512), cells_per_ref=192.0,
        t_end=t_end, t_avg_start=t0, converged=True, wall_seconds=184.0 + 20 * seed,
        device="NVIDIA GeForce RTX 4060 Ti (8.0 GB)", precision="float32", turbulence="les-wale",
        notes=["Reynolds number capped at 2.0×10^4 for a resolved simulation (see Methodology)."],
        warnings=[("Re_sim (2.0×10^4) is 41× below the real Reynolds number; skin-friction drag "
                  "is over-predicted relative to full scale.")],
        per_body={body.name: {"cd": cds.mean, "cl": cls.mean, "cm": cms.mean}},
        bodies_wind=[wind], history=hist,
        surface=[_surface_for(wind, cp, cf, sens)], mean_fields=fields, instant_fields=inst,
    )


def vehicle_cfd(design: Design, sim: Simulation, *, seed: int = 3, field_res: tuple[int, int] = (360, 120),
                with_sensitivity: bool = True) -> CFDResult:
    """A bluff-body result: high drag, slight lift, base wake, rotating wheels, moving ground."""
    rng = np.random.default_rng(seed)
    t_end, t0 = 30.0, 10.0
    hist = _history(rng, t_end, 0.58, -0.08, 0.03, 0.21, 0.06, 0.012)
    t = hist["t"]
    q = sim.conditions.dynamic_pressure()
    speed = sim.conditions.speed
    L = design.length_ref
    h = design.force_ref
    surfaces = []
    for b in design.bodies:
        p = b.points
        n = outward_normals(p)
        if b.role == "wheel":
            cp = 0.3 * n[:, 0] * -1 - 0.4 * np.clip(n[:, 1], 0, 1)
            cf = 0.004 * np.ones(len(p))
        else:
            xn = (p[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0])
            front = np.clip(-n[:, 0], 0, 1)
            roof = np.clip(n[:, 1], 0, 1)
            cp = 0.95 * front**2 - 0.9 * roof * np.exp(-((xn - 0.35) / 0.18) ** 2) - 0.25 * (xn > 0.95)
            cp += -0.35 * np.clip(-n[:, 1], 0, 1) * np.exp(-((xn - 0.2) / 0.25) ** 2)
            cp += rng.normal(0, 0.01, len(p))
            cf = 0.003 * np.where(n[:, 0] > 0.7, -0.5, 1.0) * np.sign(-n[:, 1] + 1e-9)
        sens = None
        if with_sensitivity and b.role != "wheel":
            xn = (p[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0])
            sens = (1.5 * np.exp(-((xn - 0.97) / 0.04) ** 2) * (n[:, 1] > 0)
                    - 0.8 * np.exp(-((xn - 0.75) / 0.08) ** 2) * (n[:, 1] > 0.3)
                    + 0.6 * np.exp(-((xn - 0.02) / 0.03) ** 2)) / L
        surfaces.append(_surface_for(b, cp, cf, sens))
    ext = (-0.8 * L, 3.2 * L, 0.0, 1.33 * L)
    fields = synthetic_fields(design.bodies, ext, speed, sim.conditions.fluid.density, *field_res,
                              cl=-0.05, ground=True)
    cds, cls, cms = (_stat(t, hist[k], t0) for k in ("cd", "cl", "cm"))
    per_body = {"body": {"cd": cds.mean * 0.86, "cl": cls.mean * 0.8, "cm": cms.mean},
                "front wheel": {"cd": cds.mean * 0.06, "cl": cls.mean * 0.1, "cm": 0.0},
                "rear wheel": {"cd": cds.mean * 0.08, "cl": cls.mean * 0.1, "cm": 0.0}}
    return CFDResult(
        sim_id=sim.id, cd=cds, cl=cls, cm=cms,
        drag=cds.mean * q * h, lift=cls.mean * q * h, moment=cms.mean * q * h * L,
        drag_area=cds.mean * h, power=cds.mean * q * h * speed, strouhal=0.21,
        re_real=sim.conditions.reynolds(L), re_sim=5.0e4, grid=(1536, 512), cells_per_ref=96.0,
        t_end=t_end, t_avg_start=t0, converged=False, wall_seconds=402.5,
        device="NVIDIA GeForce RTX 4060 Ti (8.0 GB)", precision="float32", turbulence="les-vreman",
        warnings=[("Statistics not fully converged: the standard error of Cd is above 1 %. "
                  "Run longer for a ranking-grade number.")],
        per_body=per_body, bodies_wind=list(design.bodies), history=hist, surface=surfaces,
        mean_fields=fields,
    )


def airfoil_polar(design: Design, sim: Simulation, *, distributions: bool = True) -> PolarResult:
    code = design.source.get("code", "2412")
    m = int(code[0]) / 100
    alphas = sim.airfoil.alphas()
    pts = []
    coords = naca4(code, 81, 1.0)
    i_le = int(np.argmin(coords[:, 0]))
    upper = np.arange(len(coords)) <= i_le
    for a in alphas:
        a0 = -m * 100 * 1.05
        lin = 2 * math.pi * 0.94 * math.radians(a - a0)
        stall = 1.0 / (1.0 + np.exp((a - (12.5 + 40 * m)) / 1.1))
        cl = lin * (0.55 + 0.45 * stall) if a > 9 else lin
        cdp = 0.0018 + 0.0009 * cl**2 + 0.02 * (1 - stall)
        cdf = 0.0042 + 0.0006 * abs(cl - 0.4)
        conv = not (a >= 15.0)
        p = PolarPoint(alpha=float(a), cl=cl, cd=cdp + cdf, cdp=cdp, cdf=cdf,
                       cm=-0.025 * m * 100 - 0.004 * a * (1 - stall), xtr_upper=float(np.clip(0.62 - 0.045 * a, 0.02, 1)),
                       xtr_lower=float(np.clip(0.55 + 0.04 * a, 0.0, 1.0)), converged=bool(conv))
        if distributions:
            xc = coords[:, 0]
            p.x, p.y = xc.copy(), coords[:, 1].copy()
            p.cp_inviscid = airfoil_surface_cp(xc, upper, a, code, viscous=False)
            p.cp = airfoil_surface_cp(xc, upper, a, code)
            p.cf = _skin_friction(xc, sim.conditions.reynolds(design.length_ref), p.xtr_upper)
        pts.append(p)
    return PolarResult(sim_id=sim.id, re=sim.conditions.reynolds(design.length_ref), mach=0.0, ncrit=9.0,
                       points=pts, wall_seconds=11.4,
                       notes=["mfoil viscous-inviscid analysis, free transition (e^N, Ncrit = 9)."],
                       warnings=[] if all(p.converged for p in pts) else
                       ["2 post-stall points did not converge and are omitted from plots."])


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

class FakeProject:
    """Duck-typed stand-in for aerolume.project.Project (name, designs, simulations, load_result)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.designs: dict[str, Design] = {}
        self.simulations: dict[str, Simulation] = {}
        self.results: dict[str, object] = {}

    def add(self, design: Design, sim: Simulation, result: object | None) -> Simulation:
        self.designs[design.id] = design
        self.simulations[sim.id] = sim
        if result is not None:
            self.results[sim.id] = result
        return sim

    def load_result(self, sim_id: str):
        return self.results.get(sim_id)


def cfd_sim(design: Design, name: str, speed: float, alpha: float, sid: str, ground: bool = False) -> Simulation:
    return Simulation(name=name, design_id=design.id, analysis="cfd", id=sid, status="done",
                      conditions=FlowConditions(AIR, speed, alpha),
                      ground=GroundConfig(enabled=ground, moving=True, rotate_wheels=True),
                      cfd=CFDSettings(quality="standard", reynolds_mode="capped", reynolds_cap=2e4,
                                      compute_sensitivity=True),
                      created="2026-10-04T09:12:00", finished="2026-10-04T09:15:10")


def polar_sim(design: Design, name: str, sid: str, speed: float = 50.0) -> Simulation:
    return Simulation(name=name, design_id=design.id, analysis="polar", id=sid, status="done",
                      conditions=FlowConditions(AIR, speed, 0.0),
                      airfoil=AirfoilSettings(alpha_start=-4, alpha_end=16, alpha_step=1),
                      created="2026-10-04T10:02:00", finished="2026-10-04T10:02:12")


def make_project(field_res: tuple[int, int] = (320, 160)) -> FakeProject:
    """Two airfoils (CFD + polar each), one vehicle CFD run and one simulation not yet run."""
    pr = FakeProject("Wing section study")
    d1, d2, dv = airfoil_design("2412"), airfoil_design("4412"), vehicle_design()
    s1 = cfd_sim(d1, "NACA 2412 · cruise α 4°", 40.0, 4.0, "s-2412-cfd")
    s2 = cfd_sim(d2, "NACA 4412 · cruise α 4°", 40.0, 4.0, "s-4412-cfd")
    pr.add(d1, s1, airfoil_cfd(d1, s1, seed=1, with_sensitivity=True, field_res=field_res))
    pr.add(d2, s2, airfoil_cfd(d2, s2, seed=2, field_res=field_res))
    p1 = polar_sim(d1, "NACA 2412 polar", "s-2412-polar")
    p2 = polar_sim(d2, "NACA 4412 polar", "s-4412-polar")
    pr.add(d1, p1, airfoil_polar(d1, p1))
    pr.add(d2, p2, airfoil_polar(d2, p2))
    sv = cfd_sim(dv, "Saloon · 120 km/h", 120 / 3.6, 0.0, "s-saloon", ground=True)
    pr.add(dv, sv, vehicle_cfd(dv, sv, field_res=(field_res[0], max(field_res[1] * 3 // 4, 60))))
    sn = cfd_sim(d1, "NACA 2412 · α 8° (queued)", 40.0, 8.0, "s-2412-a8")
    sn.status = "new"
    pr.add(d1, sn, None)
    return pr
