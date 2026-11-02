"""Flow visualisation in the volume: seed rakes, streamline tracing, smoke data (DESIGN.md 7.13).

Users asked for smoke and streamlines *in the space around the body*, not only
colour on its surface. This module holds the parts of that which need no
OpenGL, so the solver thread, the GUI thread and the report all share them:

* :class:`SeedRake` / :func:`resolve_rake` -- one definition of "the rake", the
  wind-tunnel device both smoke and streamlines start from. The GUI keeps it as
  a plain dict (``ViewSettings.seed_rake``, metres, wind frame; keys below);
  this resolves it against the body and the domain into nozzle / seed points.
  The same rake releases the smoke in the solver
  (:func:`aerolume.simulation.live3d.dye_rake_for`) and seeds the streamlines,
  so what the user sees in both always starts at the same place.
* :func:`trace_streamlines` -- ``vtkStreamTracer`` on a volume's velocity, with
  no render window: safe on a worker thread (one call per thread at a time), so
  live runs trace on the solver thread and the GUI only uploads lines.
  :class:`StreamlineSet` is the plain-numpy result (points, line offsets, speed).
* :func:`smoke_volume` -- where a result keeps its instantaneous smoke.

Rake dict keys (all optional; metres and seconds, wind frame):

=========== ==================================================================
``kind``    "rake" (a row across the span at one height), "vertical" (a row
            up the height at one span position), "grid" (span x height),
            "plane" (smoke: a continuous band; streamlines: a dense grid),
            "points" (explicit ``points``)
``x``       the rake's streamwise position (default: upstream of the nose)
``y``       span position (vertical) or (y0, y1) extent (rake, grid, plane)
``z``       height (rake) or (z0, z1) extent (vertical, grid, plane)
``n``       nozzle / seed count: an int, or (n_span, n_height) for grids
``radius``  nozzle radius [m] (smoke; default 1.25 solver cells)
``pulse``   (on, off) seconds of release and pause (smoke; default continuous)
``points``  [[x, y, z], ...] for kind "points"
=========== ==================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import volume_field

__all__ = [
    "RAKE_KINDS",
    "SeedRake",
    "StreamlineSet",
    "resolve_rake",
    "smoke_volume",
    "trace_streamlines",
    "volume_bounds",
]

RAKE_KINDS = ("rake", "vertical", "grid", "plane", "points")
#: default streamline seeds per kind (smoke nozzles follow the spacing rule of resolve_rake)
SEED_COUNTS = {"rake": 30, "vertical": 24, "grid": 120, "plane": 300}
#: default height of a horizontal rake, as a share of the body's height (see resolve_rake)
RAKE_HEIGHT = 0.40
#: velocity volumes larger than this are block-averaged before tracing: a streamline needs
#: the large scales, and 128^3 points trace 300 seeds in ~0.1 s on the development machine
TRACE_MAX_CELLS = 128 ** 3
#: streamline integration step [cells] (fixed-step RK4; see trace_streamlines)
TRACE_STEP_CELLS = 1.0


# ---------------------------------------------------------------------------
# the rake
# ---------------------------------------------------------------------------

@dataclass
class SeedRake:
    """A resolved rake (metres, wind frame): its kind, extent and points.

    ``points`` (N, 3) are the nozzle centres / seed points; ``y`` and ``z`` the
    span and height extent (equal ends for a line in that direction); ``n``
    (n_span, n_height); ``radius`` the smoke nozzle radius (None = the solver's
    default); ``pulse`` (on, off) seconds or None. :meth:`spec` gives the dict
    form back (what ``ViewSettings.seed_rake`` stores).
    """
    kind: str
    x: float
    y: tuple[float, float]
    z: tuple[float, float]
    n: tuple[int, int]
    points: np.ndarray
    radius: float | None = None
    pulse: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x, 0.5 * (self.y[0] + self.y[1]), 0.5 * (self.z[0] + self.z[1])])

    def spec(self) -> dict:
        d: dict[str, Any] = {"kind": self.kind, "x": float(self.x)}
        if self.kind == "points":
            d["points"] = np.asarray(self.points).tolist()
        else:
            d["y"] = float(self.y[0]) if self.kind == "vertical" else [float(self.y[0]), float(self.y[1])]
            d["z"] = float(self.z[0]) if self.kind == "rake" else [float(self.z[0]), float(self.z[1])]
            if self.extra.get("n_given"):           # a count only when the user chose one: smoke and
                d["n"] = int(self.n[0]) if self.kind == "rake" else (  # streamlines have their own defaults
                    int(self.n[1]) if self.kind == "vertical" else [int(self.n[0]), int(self.n[1])])
        if self.radius is not None:
            d["radius"] = float(self.radius)
        if self.pulse is not None:
            d["pulse"] = [float(self.pulse[0]), float(self.pulse[1])]
        return d

    def moved(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> SeedRake:
        """The same rake translated (what dragging its handle does)."""
        d = np.array([dx, dy, dz], dtype=np.float64)
        return SeedRake(self.kind, self.x + dx, (self.y[0] + dy, self.y[1] + dy), (self.z[0] + dz, self.z[1] + dz),
                        self.n, np.asarray(self.points, dtype=np.float64) + d, self.radius, self.pulse,
                        dict(self.extra))

    def outline(self) -> np.ndarray:
        """Segments (K, 2, 3) that draw the rake: its bar, or the frame of a grid / band."""
        x = self.x
        (y0, y1), (z0, z1) = self.y, self.z
        if self.kind in ("grid", "plane"):
            c = np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]])
            return np.stack([c, np.roll(c, -1, axis=0)], axis=1)
        if self.kind == "points":
            return np.zeros((0, 2, 3))
        return np.array([[[x, y0, z0], [x, y1, z1]]])


def _pair(v, default: tuple[float, float]) -> tuple[float, float]:
    if v is None:
        return default
    if np.ndim(v) == 0:
        f = float(v)
        return (f, f)
    a, b = (float(t) for t in list(v)[:2])
    return (min(a, b), max(a, b))


def resolve_rake(spec: dict | None, body_bounds, domain_bounds, *, purpose: str = "streamlines",
                 h: float | None = None) -> SeedRake:
    """Resolve a rake dict against the body box and the domain (both (x0, x1, y0, y1, z0, z1), m).

    Defaults (what a tunnel engineer does first): the rake stands upstream of the
    nose by 0.4 of the body's smaller dimension, length or cross-stream size (close
    enough that numerical spreading of the smoke over the approach stays small, far
    enough that the flow there is still nearly the free stream; a wing's chord, not
    its span, sets the distance), the same for smoke and streamlines so both start
    from the one rake the view draws; it spans the body plus
    20 % on each side; a "rake" sits at 40 % of the body's height -- low enough
    that the flow splits at the nose, over the top, round the sides and into the
    wake (a rake at roof height only shows the flow over the roof), and off the
    default mid-height result slice (lines in a drawn slice z-fight with it); a
    "vertical" rake on the centre plane. Smoke nozzles are about six nozzle
    radii apart so the streams stay distinct (``h``, the solver cell size,
    sets the default radius); streamlines use :data:`SEED_COUNTS`. Everything is
    clamped inside the domain.
    """
    spec = dict(spec or {})
    kind = str(spec.get("kind") or "rake").lower()
    if kind not in RAKE_KINDS:
        raise ValueError(f"unknown rake kind {kind!r}; use one of {', '.join(RAKE_KINDS)}")
    bx0, bx1, by0, by1, bz0, bz1 = (float(b) for b in body_bounds)
    vx0, vx1, vy0, vy1, vz0, vz1 = (float(b) for b in domain_bounds)
    lx, wy, hz = max(bx1 - bx0, 1e-9), max(by1 - by0, 1e-9), max(bz1 - bz0, 1e-9)
    size = max(wy, hz)
    eps = 1e-3 * max(vx1 - vx0, vy1 - vy0, vz1 - vz0)
    cell = float(h) if h else size / 20.0
    radius = spec.get("radius")
    radius = float(radius) if radius is not None else None
    r_eff = radius if radius else 1.25 * cell
    pulse = spec.get("pulse")
    pulse = (float(pulse[0]), float(pulse[1])) if pulse is not None and len(pulse) == 2 else None
    if kind == "points":
        pts = np.asarray(spec.get("points") or [], dtype=np.float64).reshape(-1, 3)
        lo = pts.min(axis=0) if len(pts) else np.array([bx0, by0, bz0])
        hi = pts.max(axis=0) if len(pts) else lo
        return SeedRake("points", float(lo[0]), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2])),
                        (len(pts), 1), pts, radius, pulse)
    up = 0.4 * min(size, lx)
    x = float(spec["x"]) if spec.get("x") is not None else bx0 - max(up, 2.0 * cell)
    x = min(max(x, vx0 + 2.0 * cell + eps), vx1 - eps)
    y_full = (by0 - 0.2 * wy, by1 + 0.2 * wy)
    z_full = (bz0 + 0.1 * hz, bz1 + 0.1 * hz)
    yc, zc = 0.5 * (by0 + by1), bz0 + RAKE_HEIGHT * hz
    m = r_eff + cell if purpose == "smoke" else eps
    if kind == "rake":
        ys = _pair(spec.get("y"), y_full)
        zs = _pair(spec.get("z"), (zc, zc))
    elif kind == "vertical":
        ys = _pair(spec.get("y"), (yc, yc))
        zs = _pair(spec.get("z"), z_full)
    else:
        ys = _pair(spec.get("y"), y_full)
        zs = _pair(spec.get("z"), z_full)
    ys = (min(max(ys[0], vy0 + m), vy1 - m), min(max(ys[1], vy0 + m), vy1 - m))
    zs = (min(max(zs[0], vz0 + m), vz1 - m), min(max(zs[1], vz0 + m), vz1 - m))
    gap = max(6.0 * r_eff, size / 7.0)
    n_spec = spec.get("n")

    def count(span: float, default: int) -> int:
        if purpose == "smoke":
            return max(2, min(17, round(span / gap) + 1))
        return default

    if kind == "rake":
        k = int(n_spec) if n_spec else count(ys[1] - ys[0], SEED_COUNTS["rake"])
        n = (max(1, k), 1)
    elif kind == "vertical":
        k = int(n_spec) if n_spec else count(zs[1] - zs[0], SEED_COUNTS["vertical"])
        n = (1, max(1, k))
    else:
        if n_spec is not None and np.ndim(n_spec) == 1:
            n = (max(1, int(n_spec[0])), max(1, int(n_spec[1])))
        else:
            total = int(n_spec) if n_spec else (None if purpose == "smoke" else SEED_COUNTS[kind])
            if total is None:
                n = (count(ys[1] - ys[0], 0), count(zs[1] - zs[0], 0))
                n = (min(n[0], 11), min(n[1], 9))
            else:
                aspect = (ys[1] - ys[0]) / max(zs[1] - zs[0], 1e-9)
                nz_ = max(2, round(math.sqrt(total / max(aspect, 1e-3))))
                n = (max(2, round(total / nz_)), nz_)
    yy = np.linspace(ys[0], ys[1], n[0]) if n[0] > 1 else np.array([0.5 * (ys[0] + ys[1])])
    zz = np.linspace(zs[0], zs[1], n[1]) if n[1] > 1 else np.array([0.5 * (zs[0] + zs[1])])
    Y, Z = np.meshgrid(yy, zz)
    pts = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], axis=1)
    return SeedRake(kind, x, ys, zs, n, pts, radius, pulse, {"n_given": n_spec is not None})


def volume_bounds(volume) -> tuple[float, float, float, float, float, float]:
    """(x0, x1, y0, y1, z0, z1) of a VolumeData as displayed (mirrored volumes whole).

    From the shape alone: building the mirrored float32 copy (:func:`.data.full_volume`)
    just to read its size cost ~50 ms on a 2 M-cell result, a third of a streamline update.
    """
    origin = tuple(float(c) for c in volume.origin)
    spacing = tuple(float(c) for c in volume.spacing)
    fields = getattr(volume, "fields", None) or {}
    mask = getattr(volume, "mask", None)
    shape = (np.shape(next(iter(fields.values()))) if fields else
             (np.shape(mask) if mask is not None else (0, 0, 0)))
    nz, ny, nx = shape
    y0, y1 = origin[1], origin[1] + ny * spacing[1]
    if getattr(volume, "mirrored", False):          # the reflected block, as full_volume builds it
        y0, y1 = -y1, -y1 + 2 * ny * spacing[1]
    return (origin[0], origin[0] + nx * spacing[0], y0, y1, origin[2], origin[2] + nz * spacing[2])


# ---------------------------------------------------------------------------
# streamlines
# ---------------------------------------------------------------------------

@dataclass
class StreamlineSet:
    """Traced streamlines as plain arrays (metres): ``points`` (N, 3) float32,
    ``offsets`` (L + 1,) int64 (line i is ``points[offsets[i]:offsets[i + 1]]``),
    ``values`` per-point arrays (always "speed" [m/s]) and the ``seeds`` used."""
    points: np.ndarray
    offsets: np.ndarray
    values: dict[str, np.ndarray]
    seeds: np.ndarray
    seconds: float = 0.0

    @property
    def n_lines(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __len__(self) -> int:
        return self.n_lines

    def lines(self) -> list[np.ndarray]:
        return [self.points[self.offsets[i]:self.offsets[i + 1]] for i in range(self.n_lines)]

    @classmethod
    def empty(cls, seeds=None) -> StreamlineSet:
        return cls(np.zeros((0, 3), np.float32), np.zeros(1, np.int64), {"speed": np.zeros(0, np.float32)},
                   np.zeros((0, 3)) if seeds is None else np.asarray(seeds, dtype=np.float64).reshape(-1, 3))

    def to_polydata(self):
        """A ``vtkPolyData`` of polylines with the point arrays (VTK imported lazily)."""
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

        pd = vtkPolyData()
        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(self.points, dtype=np.float32), deep=True))
        pd.SetPoints(pts)
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.ascontiguousarray(self.offsets, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(np.arange(len(self.points), dtype=np.int64), deep=True))
        pd.SetLines(cells)
        for k, a in self.values.items():
            va = numpy_to_vtk(np.ascontiguousarray(a, dtype=np.float32), deep=True)
            va.SetName(k)
            pd.GetPointData().AddArray(va)
        return pd


def _pool(a: np.ndarray, f: int) -> np.ndarray:
    """Block mean over f x f x f cells (the far ends cropped), float32: a sum of f^3 strided
    views, several times faster than a reshape-mean over interleaved axes."""
    nz, ny, nx = (s // f for s in a.shape)
    out = np.zeros((nz, ny, nx), np.float32)
    for i in range(f):
        for j in range(f):
            for k in range(f):
                out += a[i: nz * f: f, j: ny * f: f, k: nx * f: f]
    out *= 1.0 / f ** 3
    return out


def trace_streamlines(volume, seeds, *, direction: str = "forward", max_steps: int = 4000,
                      max_cells: int = TRACE_MAX_CELLS, extra: tuple[str, ...] = ()) -> StreamlineSet:
    """Streamlines of a VolumeData's (u, v, w) from ``seeds`` ((N, 3) metres, or a SeedRake).

    Velocity is set to zero in solid cells (mask > 1/2), so lines stop at walls
    instead of tunnelling through a Brinkman body. Volumes above ``max_cells``
    are block-averaged first. RK4 with steps of one cell, at most
    ``max_steps`` steps and 2.5 domain diagonals; lines end where the speed
    drops below 0.1 % of the free stream (stagnation, solids). Uses no render
    window, so it runs on any thread. ``extra`` fields of the volume are
    interpolated along the lines too.

    A symmetry-plane volume (``mirrored``) is traced in its own half: the plane
    has zero normal velocity, so no streamline crosses it, and the line from a
    seed at -y is the mirror image of the line from its image seed at +y. That
    is exact and avoids building (and tracing through) the mirrored copy.
    """
    import time

    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

    t0 = time.perf_counter()
    if isinstance(seeds, SeedRake):
        seeds = seeds.points
    pts = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    fields = volume.fields
    missing = [k for k in ("u", "v", "w") if k not in fields]
    if missing:
        raise KeyError(f"streamlines need u, v, w; the volume lacks {', '.join(missing)}")
    if not len(pts):
        return StreamlineSet.empty(pts)
    origin = tuple(float(c) for c in volume.origin)
    spacing = tuple(float(c) for c in volume.spacing)
    mirrored = bool(getattr(volume, "mirrored", False))
    flipped = np.zeros(len(pts), bool)
    if mirrored:                                   # image seeds in the solved half (y >= 0)
        flipped = pts[:, 1] < 0.0
        pts = pts.copy()
        pts[flipped, 1] *= -1.0
    vel = [fields[k] for k in ("u", "v", "w")]
    ex = {k: volume_field(fields, k) for k in extra if k in fields or k == "speed"}
    mask = volume.mask
    f = 1
    while vel[0].size / f ** 3 > max_cells:
        f += 1
    if f > 1:
        vel = [_pool(a, f) for a in vel]
        ex = {k: _pool(a, f) for k, a in ex.items()}
        mask = _pool(mask, f) if mask is not None else None
        spacing = tuple(s * f for s in spacing)
    nz, ny, nx = vel[0].shape
    g = 1 if mirrored else 0          # one mirrored layer below the plane: seeds on it lie inside the image
    v3 = np.empty((nz, ny + g, nx, 3), np.float32)
    for i, a in enumerate(vel):
        v3[:, g:, :, i] = a
    if mask is not None:
        v3[:, g:][np.asarray(mask) > 0.5] = 0.0
    if g:
        v3[:, 0] = v3[:, 1]
        v3[:, 0, :, 1] *= -1.0
        ex = {k: np.concatenate([np.asarray(a)[:, :1], np.asarray(a)], axis=1) for k, a in ex.items()}
        origin = (origin[0], origin[1] - spacing[1], origin[2])
        ny += 1
    img = vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing)
    img.SetOrigin(*(origin[i] + 0.5 * spacing[i] for i in range(3)))
    flat = v3.reshape(-1, 3)
    va = numpy_to_vtk(flat, deep=False)          # zero copy: ``flat`` outlives the tracer below
    va.SetName("velocity")
    img.GetPointData().AddArray(va)
    img.GetPointData().SetActiveVectors("velocity")
    keep = [flat]
    for k, a in ex.items():
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.float32).reshape(-1))
        keep.append(arr)
        xa = numpy_to_vtk(arr, deep=False)
        xa.SetName(k)
        img.GetPointData().AddArray(xa)
    vp = vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    verts = vtkCellArray()
    verts.SetData(numpy_to_vtkIdTypeArray(np.arange(len(pts) + 1, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(pts), dtype=np.int64), deep=True))
    seed_pd = vtkPolyData()
    seed_pd.SetPoints(vp)
    seed_pd.SetVerts(verts)
    sub = flat[:: max(1, len(flat) // 200_000)].astype(np.float64)
    u_ref = float(np.percentile(np.sqrt((sub ** 2).sum(axis=1)), 95)) or 1.0
    lo = np.array(origin)
    hi = lo + np.array(spacing) * np.array([nx, ny, nz])
    if mirrored:
        lo[1] = -hi[1]
    tr = vtkStreamTracer()
    tr.SetInputData(img)
    tr.SetSourceData(seed_pd)
    tr.SetInputArrayToProcess(0, 0, 0, 0, "velocity")
    # Classical RK4 with a fixed step of one cell: the velocity is trilinear between samples one
    # cell apart, so a smaller step resolves nothing the data holds. Measured on a 1.9 M-cell car
    # wake with 300 seeds it traces in 57 ms against 107 ms for adaptive RK45 (steps 0.05-1 cell,
    # most of its cost spent on error control), ends the same lines at the same places (end
    # points within one cell) and draws the same picture.
    tr.SetIntegratorTypeToRungeKutta4()
    tr.SetIntegrationStepUnit(2)                     # cell lengths
    tr.SetInitialIntegrationStep(TRACE_STEP_CELLS)
    tr.SetMinimumIntegrationStep(0.05)
    tr.SetMaximumIntegrationStep(TRACE_STEP_CELLS)
    tr.SetMaximumNumberOfSteps(int(max_steps))
    tr.SetMaximumPropagation(2.5 * float(np.linalg.norm(hi - lo)))
    tr.SetTerminalSpeed(1e-3 * u_ref)
    {"forward": tr.SetIntegrationDirectionToForward, "backward": tr.SetIntegrationDirectionToBackward,
     "both": tr.SetIntegrationDirectionToBoth}[direction]()
    tr.SetComputeVorticity(False)
    tr.Update()
    out = tr.GetOutput()
    del keep
    if out.GetNumberOfPoints() == 0:
        return StreamlineSet.empty(seeds)
    P = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float32)
    lines = out.GetLines()
    offs = vtk_to_numpy(lines.GetOffsetsArray()).astype(np.int64)
    conn = vtk_to_numpy(lines.GetConnectivityArray()).astype(np.int64)
    P = P[conn]                                       # points in line order (the tracer may share points)
    pdat = out.GetPointData()
    V = vtk_to_numpy(pdat.GetArray("velocity")).astype(np.float32)[conn]
    values = {"speed": np.sqrt((V.astype(np.float64) ** 2).sum(axis=1)).astype(np.float32)}
    for k in ex:
        a = pdat.GetArray(k)
        if a is not None:
            values[k] = vtk_to_numpy(a).astype(np.float32)[conn]
    if mirrored and flipped.any():
        ids = out.GetCellData().GetArray("SeedIds")
        if ids is not None:
            seed_of_line = vtk_to_numpy(ids).astype(np.int64)
            counts = np.diff(offs)
            per_point = np.repeat(flipped[seed_of_line], counts)
            P[per_point, 1] *= -1.0
            if "v" in values:
                values["v"][per_point] *= -1.0
    return StreamlineSet(P, offs, values, np.asarray(seeds, dtype=np.float64).reshape(-1, 3),
                         seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# smoke in results
# ---------------------------------------------------------------------------

def smoke_volume(result):
    """The instantaneous smoke of a 3D result as a VolumeData with a "dye" field, or None.

    The runner stores it as an extra volume labelled "instant" when the result
    type has ``extra_volumes``, otherwise as the "dye" field of the result's
    (mean) volume -- the documented home until the model grows the extra list.
    Either way the smoke is a snapshot at the end of the run (an average of smoke
    is a blur, not a picture).
    """
    for vol in getattr(result, "extra_volumes", None) or ():
        if "dye" in getattr(vol, "fields", {}):
            return vol
    vol = getattr(result, "volume", None)
    if vol is not None and "dye" in vol.fields:
        return vol
    return None
