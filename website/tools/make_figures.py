"""Render the website's flow-field images with Aerolume's own solver.

The site's opening image must be a real result, not an illustration, so this
script runs :class:`aerolume.solvers.ns2d.FlowSolver` on two short cases and
colour-maps the fields with :func:`aerolume.viz.render_field`:

* ``airfoil``: NACA 2412 at 8 deg, Re 5000, 256 cells per chord, smoke (dye)
  and vorticity at t = 12 convective times;
* ``vehicle``: the ``fastback`` template (wheels on, 30 mm gap to the road)
  over a moving road with rolling wheels, Re 2x10^5, WALE subgrid model,
  512 cells per length, vorticity at t = 4.

Every image is drawn for the site's plate, a dark ground (``PLATE``) that is the
same in the light and the dark theme: smoke is light on that ground and vorticity
uses ``icefire``, the diverging map whose zero is near-black, so a figure never
turns into a white slab on a dark page (or flashes when the field is switched).
The vehicle plate also gets the road drawn under it and its wheels drawn as tyres
with rims; in the solver's mask they are just more solid, and on a mid-grey body
they read as part of it. For phones each plate also has a ``-m`` crop around the
body, downsampled with a Lanczos filter: the browser's own downscaling of the full
image aliases the smoke streaks into moire at 400 px.

Each run takes on the order of a minute on a mid-range GPU (several on a CPU).
The raw fields are kept in an ``.npz`` next to the output so the images can be
re-rendered (``--render-only``) without solving again. The case parameters are
also written to ``assets/figures.json``; build.py reads it so the captions on
the site always state the conditions that produced the pictures.

Usage::

    python website/tools/make_figures.py [--case airfoil|vehicle|all]
        [--cache DIR] [--render-only] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "aerolume-website" / "fields"

# Smoke background: render_field maps vmin..vmax onto the greyscale LUT, so a
# negative vmin lifts zero concentration off pure black to a dark grey -- the
# "smoke on a dark field" convention of docs/ui_guidelines.md section 5 and of
# wind-tunnel smoke photographs.
SMOKE_VMIN = -0.16
SOLID = (128, 133, 140, 255)
EDGE = (30, 32, 36, 255)
#: website/assets/site.css --plate; the zero of ``icefire`` (#1d1d21) sits within 3 levels of it.
PLATE = (26, 28, 32)
#: Vorticity map on the dark plate: diverging with a near-black zero. ui_guidelines section 5
#: asks for a zero-centred diverging map; RdBu_r's white zero is wrong on a dark ground.
VORTICITY_CMAP = "icefire"
TYRE = (52, 55, 60)
TYRE_EDGE = (176, 181, 188)
RIM = (150, 155, 162)
ROAD_LINE = (176, 181, 188)
ROAD_HATCH = (78, 82, 89)
#: Phone crops in reference lengths (x0, x1); the full domain height is kept.
MOBILE_CROP = {"airfoil": (-0.40, 2.53), "vehicle": (-0.12, 1.80)}
MOBILE_WIDTH = 1100          # px: about 3x a 360 px phone column

CASES = {
    "airfoil": {
        "title": "NACA 2412, alpha 8 deg, Re 5000",
        "naca": "2412", "alpha_deg": 8.0, "re": 5000.0, "cells_per_ref": 256,
        "extent": (-1.25, 4.25, -1.1, 1.1), "t_end": 12.0, "turbulence": "none",
    },
    "vehicle": {
        "title": "Fastback over a moving road, Re 2x10^5",
        "template": "fastback", "template_params": {"wheels": True, "wheel_gap": 0.03},
        "re": 2.0e5, "cells_per_ref": 512, "extent": (-0.5, 2.5, 0.0, 0.75), "t_end": 4.0, "turbulence": "wale",
    },
}


def _rotate(pts: np.ndarray, deg: float, about: tuple[float, float]) -> np.ndarray:
    """Rotate counter-clockwise by ``deg`` about ``about`` (DESIGN.md section 3 uses -alpha)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(pts, float) - about
    return np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]] + np.asarray(about)


def _grid(extent: tuple[float, float, float, float], cells_per_ref: int) -> tuple[int, int, float]:
    h = 1.0 / cells_per_ref
    x0, x1, y0, y1 = extent
    return round((x1 - x0) / h), round((y1 - y0) / h), h


def airfoil_config(spec: dict, device: str):
    """NACA section rotated nose-up about the quarter chord, slip walls, uniform inflow."""
    from aerolume.geometry import naca4
    from aerolume.solvers.ns2d import ImmersedBody, SolverConfig

    pts = _rotate(naca4(spec["naca"], n=400), -spec["alpha_deg"], (0.25, 0.0))
    nx, ny, h = _grid(spec["extent"], spec["cells_per_ref"])
    return SolverConfig(nx=nx, ny=ny, h=h, x0=spec["extent"][0], y0=spec["extent"][2], re=spec["re"],
                        bodies=[ImmersedBody(pts, name="NACA " + spec["naca"])],
                        turbulence=spec["turbulence"], device=device, dye=True)


def vehicle_config(spec: dict, device: str):
    """Template vehicle scaled to unit length, road at y = 0 moving with the free stream.

    Wheels roll with the road. The solver's convention (ImmersedBody docstring)
    is omega = +ground_speed / r, counter-clockwise, so the contact point moves
    in +x with the belt.
    """
    from aerolume.geometry import templates
    from aerolume.solvers.ns2d import ImmersedBody, SolverConfig

    design = templates.make(spec["template"], **spec.get("template_params", {}))
    L = design.length_ref
    xmin = min(float(b.points[:, 0].min()) for b in design.bodies)
    bodies = []
    for b in design.bodies:
        pts = (np.asarray(b.points) - (xmin, 0.0)) / L
        if b.role == "wheel" and b.center is not None:
            cx, cy = (b.center[0] - xmin) / L, b.center[1] / L
            r = float(np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
            bodies.append(ImmersedBody(pts, name=b.name, role="wheel", omega=1.0 / r, center=(cx, cy)))
        else:
            bodies.append(ImmersedBody(pts, name=b.name, role=b.role))
    nx, ny, h = _grid(spec["extent"], spec["cells_per_ref"])
    spec["length_m"] = round(L, 3)
    return SolverConfig(nx=nx, ny=ny, h=h, x0=spec["extent"][0], y0=0.0, re=spec["re"], bodies=bodies,
                        bc_bottom="moving", ground_speed=1.0, bc_top="slip",
                        turbulence=spec["turbulence"], device=device, dye=True,
                        force_ref=design.force_ref / L, moment_ref=(0.5, 0.0))


def solve(case: str, device: str, cache: Path) -> Path:
    """Run one case to ``t_end`` and store the fields needed for rendering."""
    import torch

    import aerolume
    from aerolume.solvers.ns2d import FlowSolver

    spec = dict(CASES[case])
    cfg = airfoil_config(spec, device) if case == "airfoil" else vehicle_config(spec, device)
    solver = FlowSolver(cfg)
    t0 = time.perf_counter()
    while solver.t < spec["t_end"]:
        solver.advance(200)
    wall = time.perf_counter() - t0
    fields = {k: solver.field(k).detach().float().cpu().numpy() for k in ("dye", "vorticity", "mask")}
    hist = solver.history()
    n = len(hist["t"])
    tail = slice(n // 2, n)
    meta = {
        "case": case, **spec, "nx": cfg.nx, "ny": cfg.ny, "steps": int(solver.steps),
        "t": round(float(solver.t), 3), "wall_s": round(wall, 1),
        "device": str(solver.device), "dtype": cfg.dtype,
        "cd_mean_second_half": round(float(np.mean(hist["cd"][tail])), 4) if n else None,
        "cl_mean_second_half": round(float(np.mean(hist["cl"][tail])), 4) if n else None,
        "gpu": torch.cuda.get_device_name(0) if solver.device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "date": time.strftime("%Y-%m-%d"),
        # the release whose solver drew the figure: the site's title block states it next to the date
        "aerolume": aerolume.__version__,
    }
    out = cache / f"{case}_fields.npz"
    np.savez_compressed(out, meta=json.dumps(meta), **fields)
    del solver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{case}: {cfg.nx}x{cfg.ny}, {meta['steps']} steps to t={meta['t']} in {wall:.1f} s on {meta['device']}")
    return out


def _upsample(a: np.ndarray, factor: int):
    """Bicubic upsampling before colour-mapping, so the LUT is applied per output pixel."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(a)[None, None].float()
    return F.interpolate(t, scale_factor=factor, mode="bicubic", align_corners=False)[0, 0]


def _coverage(dist: np.ndarray) -> np.ndarray:
    """Anti-aliased coverage (0..1) of the region ``dist < 0``, ``dist`` in pixels."""
    return np.clip(0.5 - dist, 0.0, 1.0)[..., None]


def _blend(img: np.ndarray, cov: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    return img * (1.0 - cov) + np.asarray(rgb, float) * cov


def draw_wheels(rgb: np.ndarray, wheels: list[tuple[float, float, float]], extent, ppu: float) -> np.ndarray:
    """Paint each wheel (cx, cy, r in reference lengths) as a tyre with a rim and a hub.

    The distances are analytic, so the edges are anti-aliased without supersampling.
    Only a box around each wheel is touched.
    """
    img = rgb.astype(float)
    for cx, cy, r in wheels:
        px, py, pr = (cx - extent[0]) * ppu, (extent[3] - cy) * ppu, r * ppu
        r0, r1 = max(int(py - pr) - 3, 0), min(int(py + pr) + 4, img.shape[0])
        c0, c1 = max(int(px - pr) - 3, 0), min(int(px + pr) + 4, img.shape[1])
        yy, xx = np.mgrid[r0:r1, c0:c1].astype(float) + 0.5
        d = np.hypot(xx - px, yy - py)
        blk = img[r0:r1, c0:c1]
        blk = _blend(blk, _coverage(d - pr - 1.0), TYRE_EDGE)        # 1 px light line round the tyre
        blk = _blend(blk, _coverage(d - pr + 1.0), TYRE)
        blk = _blend(blk, _coverage(d - 0.56 * pr), RIM)
        blk = _blend(blk, _coverage(d - 0.14 * pr), TYRE)
        img[r0:r1, c0:c1] = blk
    return (img + 0.5).clip(0, 255).astype(np.uint8)


def add_road(rgb: np.ndarray, height: int) -> np.ndarray:
    """Append a drawn road under the image: a ground line over drawing-style hatching."""
    w = rgb.shape[1]
    band = np.empty((height, w, 3), np.uint8)
    band[:] = PLATE
    yy, xx = np.mgrid[0:height, 0:w]
    band[((xx + yy) % 16) < 2] = ROAD_HATCH
    band[:3] = ROAD_LINE
    return np.concatenate([rgb, band], axis=0)


def mobile_crop(rgb: np.ndarray, case: str, extent, ppu: float):
    """The phone version: a crop around the body, Lanczos-downsampled to MOBILE_WIDTH."""
    from PIL import Image

    x0, x1 = MOBILE_CROP[case]
    c0, c1 = round((x0 - extent[0]) * ppu), round((x1 - extent[0]) * ppu)
    img = Image.fromarray(np.ascontiguousarray(rgb[:, c0:c1]), "RGB")
    return img.resize((MOBILE_WIDTH, round(MOBILE_WIDTH * img.size[1] / img.size[0])), Image.LANCZOS)


def vehicle_wheels(spec: dict) -> list[tuple[float, float, float]]:
    """Wheel circles (cx, cy, r) of the vehicle case, in the solver's reference lengths.

    Same transform as :func:`vehicle_config`: shift to x = 0 at the nose, divide by L.
    """
    from aerolume.geometry import templates

    design = templates.make(spec["template"], **spec.get("template_params", {}))
    L = design.length_ref
    xmin = min(float(b.points[:, 0].min()) for b in design.bodies)
    out = []
    for b in design.bodies:
        if b.role == "wheel" and b.center is not None:
            pts = np.asarray(b.points)
            r = float(np.median(np.hypot(pts[:, 0] - b.center[0], pts[:, 1] - b.center[1])))
            out.append(((b.center[0] - xmin) / L, b.center[1] / L, r / L))
    return out


def render(case: str, cache: Path, factor: int = 2) -> dict:
    """Write ``assets/flow-<case>-<field>.webp``, the ``-m`` phone crops and the social card."""
    from PIL import Image

    from aerolume.viz import render_field

    data = np.load(cache / f"{case}_fields.npz")
    meta = json.loads(str(data["meta"]))
    mask = _upsample(data["mask"], factor).clamp(0.0, 1.0)
    extent = meta["extent"]
    ppu = meta["cells_per_ref"] * factor            # image pixels per reference length
    outputs = {}

    if case == "airfoil":
        # Smoke only for the airfoil: the inlet comb is ~40 streaks over the
        # domain height, which the vehicle's shallow domain resolves with too
        # few cells per streak -- they diffuse into a uniform grey before they
        # reach the car.
        dye = _upsample(data["dye"], factor).clamp(0.0, 1.0)
        outputs["smoke"] = render_field(dye, "grayscale", SMOKE_VMIN, 1.0, mask=mask,
                                        solid_rgba=SOLID, edge_rgba=EDGE)[..., :3]

    # Vorticity, centred on zero. The limit is the 99th percentile of |omega|
    # in the fluid so the shed cores saturate instead of a few wall cells
    # setting the scale.
    w = data["vorticity"]
    fluid = data["mask"] < 0.5
    lim = float(np.percentile(np.abs(w[fluid]), 99.0))
    outputs["vorticity"] = render_field(_upsample(w, factor), VORTICITY_CMAP, -lim, lim, mask=mask,
                                        solid_rgba=SOLID, edge_rgba=EDGE)[..., :3]
    meta["vorticity_limit"] = round(lim, 2)
    meta["vorticity_cmap"] = VORTICITY_CMAP

    if case == "vehicle":
        wheels = vehicle_wheels(meta)
        road = round(0.04 * ppu)
        outputs = {name: add_road(draw_wheels(rgb, wheels, extent, ppu), road) for name, rgb in outputs.items()}
        meta["road_px"] = road

    files, mobile = {}, {}
    for name, rgb in outputs.items():
        img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
        path = ASSETS / f"flow-{case}-{name}.webp"
        img.save(path, "WEBP", quality=90, method=6)
        files[name] = path.name
        small = mobile_crop(rgb, case, extent, ppu)
        mpath = ASSETS / f"flow-{case}-{name}-m.webp"
        small.save(mpath, "WEBP", quality=88, method=6)
        mobile[name] = {"file": mpath.name, "pixels": list(small.size)}
        print(f"  wrote {path.name} {img.size[0]}x{img.size[1]}, {mpath.name} {small.size[0]}x{small.size[1]}")
    if case == "airfoil":
        card = Image.fromarray(np.ascontiguousarray(outputs["smoke"]), "RGB")
        card = card.resize((1200, round(1200 * card.size[1] / card.size[0])), Image.LANCZOS)
        card.save(ASSETS / "social-card.jpg", "JPEG", quality=86, optimize=True)
    meta["files"] = files
    meta["mobile"] = mobile
    meta["pixels"] = [int(outputs["vorticity"].shape[1]), int(outputs["vorticity"].shape[0])]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", choices=("airfoil", "vehicle", "all"), default="all")
    # the cache holds solver fields, never committed or published: by default a folder under the
    # system temp directory, so no checkout depends on (or writes into) one person's folder layout
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help=f"folder for the computed data (default {DEFAULT_CACHE})")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    cases = list(CASES) if args.case == "all" else [args.case]
    index_path = ASSETS / "figures.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    for case in cases:              # strictly sequential: one solver on the GPU at a time
        if not args.render_only:
            solve(case, args.device, args.cache)
        index[case] = render(case, args.cache)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
