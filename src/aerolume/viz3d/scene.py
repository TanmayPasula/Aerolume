"""Scene: the off-screen VTK renderer behind every 3D picture in Aerolume (DESIGN.md 7.9).

One class builds the scene (meshes, ground, slices, streamlines, iso-surfaces,
colour bar, axes), owns the camera and renders to a numpy image. The Qt widget
(:mod:`aerolume.app.widgets.view3d`) only paints that image, and reports call
the same :meth:`Scene.render`, so a 3D view looks identical on screen, on the
offscreen Qt platform (tests, screenshots) and in a PDF.

Design notes (the WHY):

* **Off-screen OpenGL, never a native VTK window.** A ``vtkRenderWindow`` with
  ``SetOffScreenRendering(1)`` renders into a framebuffer object of a hidden
  WGL context on Windows (EGL / GLX elsewhere, per VTK's build). Measured on
  the RTX 4060 Ti: ~3.5 ms per 1600x900 frame of a 200k-triangle mesh,
  read-back included, so the copy into Qt costs nothing noticeable and no
  native window has to be embedded into Qt (``QVTKRenderWindowInteractor``
  breaks on the offscreen platform and with high-DPI scaling). OSMesa is used
  only when asked for (``AEROLUME_VTK_BACKEND=osmesa``) *and* its DLL is
  present: VTK 9.7 aborts the whole process when it tries OSMesa without the
  library. When no OpenGL 3.2 context can be created (e.g. a remote desktop
  without a GPU driver), :attr:`Scene.available` is False, :attr:`Scene.error`
  says why, and :meth:`render` returns a background-coloured image instead
  of crashing the application.
* **Two renderer layers.** Layer 0 is the 3D view over the full image; when
  a colour bar is shown its projection centre moves left
  (``vtkCamera.SetWindowCenter``) so the model is framed in the area left of
  the bar and the legend never covers it (the 2D FieldView rule), while the
  floor still runs on under the bar instead of ending at a hard edge.
  Layer 1 holds the 2D overlays (colour bar, triad, caption) laid out in
  pixels, see :mod:`aerolume.viz3d.overlay`.
* **Engineering look.** VTK's light kit (key, fill, back and head lights,
  white, camera-relative: the model is always lit from the viewer's side),
  low and broad speculars, Phong interpolation and 8x multisampling. Colour
  maps are exactly the 2D LUTs; slices are drawn *unlit* so a colour on a
  slice means the same value as on the colour bar. Normals are recomputed
  with a 45 deg feature angle: curved surfaces shade smoothly while sharp
  edges of imported CAD stay crisp.
* **The camera lives in Python** (:mod:`aerolume.viz3d.camera`), copied into
  VTK before each render, so camera behaviour is testable without OpenGL. A
  preset with ``fit=True`` keeps re-fitting when the image's aspect changes,
  until the user orbits, pans or zooms (the FieldView auto-fit rule).
* **Change tracking.** :attr:`revision` increases with every change that
  alters the picture; the widget re-renders only when the revision or the
  size changed.
* **Flow in the volume** (DESIGN.md 7.13): smoke is GPU volume rendering of
  the solver's passive scalar through a *fixed* transfer function (white-grey
  wisps on dark, dark translucent grey on light), streamlines are tubes traced
  through the velocity volume from the seed rake (:mod:`.flow`), and both
  update in place (:meth:`Scene.update_smoke`, :meth:`Scene.update_streamlines`:
  same actors, so a live run never flashes). Translucent bodies composite
  correctly with the smoke (depth peeling also for volumes).

A Scene holds an OpenGL context: use it from one thread and call
:meth:`close` (or let it be garbage-collected) when done.
"""
from __future__ import annotations

import ctypes.util
import logging
import math
import os
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from . import camera as cam
from .colors import (
    DIVERGING_CMAPS,
    SceneTheme,
    colorize,
    display_range,
    field_info,
    lookup_table,
    scene_theme,
)
from .data import (
    full_volume,
    image_to_vtk,
    mesh_arrays,
    polydata_to_vtk,
    slice_corners,
    slice_from_volume,
    volume_field,
)
from .flow import SeedRake, StreamlineSet, resolve_rake, trace_streamlines, volume_bounds
from .overlay import ColorbarSpec, Overlay, colorbar_strip_width

log = logging.getLogger(__name__)

__all__ = ["PARTICLE_PX", "SCALAR_VOLUME_FLOOR", "SCALAR_VOLUME_OPACITY", "SMOKE_CORE_SHADE", "SMOKE_THRESHOLD",
           "TRAIL_OPACITY", "VORTEX_FIELDS", "WALL_CLEARANCE_CELLS", "Scene", "backend_info", "rake_seeds",
           "smoke_core_colour", "smoke_display_rgb", "smoke_gain"]

_VTK_READY = False
_VOLUME_READY = False
_BACKEND: dict[str, Any] | None = None
#: smoke below this concentration is fully transparent (the numerical haze around the streams)
SMOKE_THRESHOLD = 0.02
#: (concentration, share of the theme's per-voxel smoke opacity): a convex curve, tuned on Draft runs
SMOKE_OPACITY_CURVE = ((0.08, 0.01), (0.2, 0.06), (0.4, 0.22), (0.7, 0.6), (1.0, 1.0))
#: Share of the way from a *neutral* smoke colour (white, grey) towards black (light theme) or white
#: (dark theme) at full concentration: the cores get an edge, the wisps keep the emitter's colour.
SMOKE_CORE_SHADE = 0.45
#: HSV saturation from which a smoke colour counts as a colour (orange, blue) rather than a grey: such a
#: channel is drawn at full saturation and its cores keep the hue. Shading the car's orange and blue
#: channels towards white / black drew both as one grey-brown haze where they overlapped.
SMOKE_CHROMA_MIN = 0.25
#: How much darker a coloured channel's cores are on the light theme (its value, hue and full
#: saturation kept): enough for an edge against the floor, not enough to turn orange brown.
SMOKE_CHROMA_DARKEN = 0.2
#: Smoke within this many cells of a solid is cleared: linear interpolation of the voxels next to the
#: wall painted the dye of the boundary layer onto the body as a coat (the car's roof and flanks).
SMOKE_WALL_CLEAR_CELLS = 1
#: :func:`smoke_gain`: the concentration a channel's 90th percentile (of its smoky voxels) is lifted
#: to, the most it is lifted, and how sparse a channel must be (share of voxels holding smoke)
#: before its opacity is raised as well, up to SMOKE_SPARSE_BOOST times.
SMOKE_GAIN_TARGET = 0.45
SMOKE_GAIN_MAX = 4.0
SMOKE_SPARSE_SHARE = 0.02
SMOKE_SPARSE_BOOST = 2.5
#: Opacity curve of a sparse channel (a point emitter's thread): concave, so its diffuse 0.05-0.2 body
#: stays visible as a continuous line instead of a few dots where it is densest.
SMOKE_SPARSE_CURVE = ((0.05, 0.10), (0.2, 0.40), (0.5, 0.80), (1.0, 1.0))
#: vortex-identification fields: their iso-surfaces are trimmed near walls by default
VORTEX_FIELDS = frozenset({"q", "lambda2"})
#: Scalar volumes (a thermal plume): the clear bottom share of the range, and the per-voxel
#: opacity of the top of the range (unit distance one voxel, as for smoke, so the look does not
#: depend on the grid). 0.12 lets a plume ten voxels thick read as solid colour at its core
#: while its rim, where most of the gradient is, stays translucent over the body.
SCALAR_VOLUME_FLOOR = 0.12
SCALAR_VOLUME_OPACITY = 0.25
#: Exponent of the opacity rise above the floor: 1 is linear; above 1 the rim stays clearer. 1.2 keeps a
#: thin plume (5-10 voxels) readable at mid temperatures, where 1.5 left only a pink haze over the body.
SCALAR_VOLUME_GAMMA = 1.2
#: Tracer particle markers [px] and the opacity of their streak trails.
PARTICLE_PX = 3.0
TRAIL_OPACITY = 0.55
#: default clearance between a vortex iso-surface and the body, in cells of the volume. On the
#: Draft sphere, Q above the default level fills 40 % of the cells 1.5-2 cells from the wall,
#: 17 % at 3-4 and 7 % at 6-10 (the real shear layer): 3.5 removes the wall skin and its torn
#: rim, 3.0 still leaves flaps of it, 1.5 wraps the body.
WALL_CLEARANCE_CELLS = 3.5


def _init_vtk() -> None:
    """Import the VTK modules the scene needs and silence VTK's modal error window.

    On Windows VTK's default output window is a message box; a warning from a
    filter would then block the GUI thread. Messages go to stderr instead.
    """
    global _VTK_READY
    if _VTK_READY:
        return
    import vtkmodules.vtkRenderingFreeType
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  (object factory: OpenGL implementations)
    from vtkmodules.vtkCommonCore import vtkOutputWindow
    try:
        vtkOutputWindow.GetInstance().SetDisplayModeToAlwaysStdErr()
    except AttributeError:      # pragma: no cover - older VTK
        pass
    _VTK_READY = True


def _init_volume_rendering() -> None:
    """Register VTK's OpenGL volume mappers (the smart mapper's GPU ray caster)."""
    global _VOLUME_READY
    if not _VOLUME_READY:
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401  (object factory)
        _VOLUME_READY = True


def _osmesa_present() -> bool:
    return bool(ctypes.util.find_library("osmesa") or ctypes.util.find_library("OSMesa"))


def _make_render_window():
    """An off-screen render window of the configured backend, or (None, reason)."""
    _init_vtk()
    choice = os.environ.get("AEROLUME_VTK_BACKEND", "auto").strip().lower()
    if choice == "osmesa":
        if not _osmesa_present():
            return None, "AEROLUME_VTK_BACKEND=osmesa but osmesa.dll / libOSMesa was not found"
        from vtkmodules.vtkRenderingOpenGL2 import vtkOSOpenGLRenderWindow
        rw = vtkOSOpenGLRenderWindow()
    else:
        from vtkmodules.vtkRenderingCore import vtkRenderWindow
        rw = vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetShowWindow(False)
    return rw, ""


def backend_info() -> dict[str, Any]:
    """Probe (once per process) whether VTK can render off-screen here.

    Returns ``{"ok": bool, "window": class name, "renderer": GL renderer string,
    "version": GL version, "error": str}``. The probe creates and destroys one
    tiny context (~0.1 s).
    """
    global _BACKEND
    if _BACKEND is not None:
        return dict(_BACKEND)
    info: dict[str, Any] = {"ok": False, "window": "", "renderer": "", "version": "", "error": ""}
    try:
        rw, err = _make_render_window()
        if rw is None:
            info["error"] = err
        else:
            info["window"] = rw.GetClassName()
            ok = bool(rw.SupportsOpenGL())
            if ok:
                rw.SetSize(8, 8)
                rw.Render()                 # the capability report is empty before a first render
                caps = rw.ReportCapabilities() or ""
                for line in caps.splitlines():
                    if line.startswith("OpenGL renderer string:"):
                        info["renderer"] = line.split(":", 1)[1].strip()
                    elif line.startswith("OpenGL version string:"):
                        info["version"] = line.split(":", 1)[1].strip()
            else:
                info["error"] = "no OpenGL 3.2 context could be created for off-screen rendering"
            info["ok"] = ok
            rw.Finalize()
    except Exception as exc:  # noqa: BLE001 - any failure means "no 3D view", never a crash
        info["error"] = f"{type(exc).__name__}: {exc}"
    _BACKEND = info
    if not info["ok"]:
        log.warning("3D view unavailable: %s", info["error"])
    return dict(info)


def rake_seeds(focus_bounds, volume_bounds, kind: str = "rake", n: int | None = None) -> np.ndarray:
    """Seed points for streamlines upstream of the bodies (the fixed presets; the GUI's
    movable rake is a dict resolved by :func:`.flow.resolve_rake`).

    ``kind`` "rake": a horizontal line across the flow just above the bodies'
    mid-height, 15 % wider than them on each side; "plane": an (ny x nz) grid
    on the same upstream plane covering the frontal box; "vertical": a
    vertical line on the centre plane. The plane sits a quarter of a body
    length upstream of the bodies (inside the volume), where the flow is still
    undisturbed, so every line starts in the free stream.
    """
    fx0, fx1, fy0, fy1, fz0, fz1 = (float(b) for b in focus_bounds)
    vx0, vx1, vy0, vy1, vz0, vz1 = (float(b) for b in volume_bounds)
    lx, wy, hz = max(fx1 - fx0, 1e-9), max(fy1 - fy0, 1e-9), max(fz1 - fz0, 1e-9)
    eps = 1e-3 * max(vx1 - vx0, vy1 - vy0, vz1 - vz0)
    x = min(max(fx0 - 0.25 * lx, vx0 + eps), vx1 - eps)
    ya, yb = max(fy0 - 0.15 * wy, vy0 + eps), min(fy1 + 0.15 * wy, vy1 - eps)
    za, zb = max(fz0 - 0.1 * hz, vz0 + eps), min(fz1 + 0.1 * hz, vz1 - eps)
    if kind == "rake":
        k = int(n or 30)
        ys = np.linspace(ya, yb, k)
        # 55 % of the height rather than 50 %: the default horizontal result slice is at
        # mid-height, and lines lying exactly in a drawn slice z-fight with it
        return np.stack([np.full(k, x), ys, np.full(k, fz0 + 0.55 * hz)], axis=1)
    if kind == "vertical":
        k = int(n or 24)
        zs = np.linspace(za, zb, k)
        yc = min(max(0.5 * (fy0 + fy1), vy0 + eps), vy1 - eps)
        return np.stack([np.full(k, x), np.full(k, yc), zs], axis=1)
    if kind == "plane":
        total = int(n or 120)
        aspect = (yb - ya) / max(zb - za, 1e-9)
        nz = max(2, round(math.sqrt(total / max(aspect, 1e-3))))
        ny = max(2, round(total / nz))
        yy, zz = np.meshgrid(np.linspace(ya, yb, ny), np.linspace(za, zb, nz))
        return np.stack([np.full(yy.size, x), yy.ravel(), zz.ravel()], axis=1)
    raise ValueError(f"unknown seed kind {kind!r}; use 'rake', 'plane', 'vertical' or an (N, 3) array")


@dataclass
class _Layer:
    """One thing in the scene: its actors and, if colour-mapped, how."""
    kind: str
    actors: list
    bounds: tuple[float, float, float, float, float, float]
    field: str | None = None
    cmap: str | None = None
    vmin: float = 0.0
    vmax: float = 1.0
    values: np.ndarray | None = None          # for clipped-end detection on the colour bar
    luts: list = dc_field(default_factory=list)  # vtkLookupTables to update on range / map changes
    rgb_rebuild: Any = None                   # slices: re-colour the texture
    themed_color: bool = False                # surface colour follows the theme (body grey)
    visible: bool = True
    extra: dict = dc_field(default_factory=dict)


def _bounds_of(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if points.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


_DIRS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                  if (i, j, k) != (0, 0, 0)], dtype=float)


def _fit_sample(points: np.ndarray, n: int = 3000) -> np.ndarray:
    """A small point set that frames like the whole mesh: a strided sample plus extremes.

    The extreme points along the 26 axis / diagonal directions make sure the
    silhouette in any preset view is represented even if the stride skips it.
    """
    if len(points) <= n:
        return np.asarray(points, float)
    proj = points @ _DIRS.T
    ext = points[np.unique(proj.argmax(axis=0))]
    return np.concatenate([points[:: max(1, len(points) // n)], ext])


def _union(bs) -> tuple[float, float, float, float, float, float] | None:
    bs = [b for b in bs if b is not None]
    if not bs:
        return None
    a = np.array(bs, dtype=float)
    return (a[:, 0].min(), a[:, 1].max(), a[:, 2].min(), a[:, 3].max(), a[:, 4].min(), a[:, 5].max())


class Scene:
    """An off-screen VTK scene with a CAD-style camera (see the module docstring).

    Every ``add_*`` returns a layer handle (a short string) usable with
    :meth:`remove`, :meth:`set_visible`, :meth:`set_scalar_range`,
    :meth:`set_cmap` and :meth:`add_colorbar`.
    """

    def __init__(self, theme: str | dict | SceneTheme | None = "light", *, samples: int = 8,
                 gradient: bool = False, axes: bool = True) -> None:
        self.theme = scene_theme(theme, gradient=gradient)
        self.revision = 0
        self.last_render_ms = float("nan")
        self._layers: dict[str, _Layer] = {}
        self._count = 0
        self._edges = False
        self._axes = bool(axes)
        self._caption = ""
        self._colorbar: dict | None = None
        self._colorbar2: dict | None = None     # a second bar for a second colour map (add_colorbar)
        self._camera = cam.CameraState()
        self._auto_fit = True
        self._fit_dir: tuple[np.ndarray, np.ndarray] | None = cam.preset("iso")
        self._size = (0, 0)
        self._scale = 1.0
        self._closed = False
        self._window_cx = 0.0
        self._focus: tuple | None = None
        self._top_inset = 0.0
        self._show_colorbar = True          # set_annotations: a companion view may carry the legend
        self._show_caption = True
        self.error = ""
        info = backend_info()
        self.available = bool(info["ok"])
        if not self.available:
            self.error = info["error"] or "3D rendering is not available"
            self._rw = None
            return
        self._build_pipeline(samples)

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _build_pipeline(self, samples: int) -> None:
        from vtkmodules.vtkRenderingCore import vtkLightKit, vtkRenderer

        rw, err = _make_render_window()
        if rw is None:                      # pragma: no cover - backend_info said ok
            self.available, self.error, self._rw = False, err, None
            return
        self._rw = rw
        rw.SetMultiSamples(max(0, int(samples)))
        rw.SetNumberOfLayers(2)
        rw.SetAlphaBitPlanes(0)
        rw.SetDPI(72)                       # VTK font sizes are then exactly pixels
        self._ren = vtkRenderer()
        self._ren.SetLayer(0)
        self._ren.TwoSidedLightingOn()
        self._ren.SetUseDepthPeeling(1)     # correct translucent meshes / iso-surfaces
        self._ren.SetUseDepthPeelingForVolumes(1)    # ... and smoke seen through a translucent body
        self._ren.SetMaximumNumberOfPeels(6)
        self._ren.SetOcclusionRatio(0.0)
        kit = vtkLightKit()
        kit.SetKeyLightIntensity(0.78)
        kit.SetKeyToFillRatio(1.9)
        kit.SetKeyToHeadRatio(2.4)
        kit.SetKeyToBackRatio(3.5)
        for setter in ("SetKeyLightWarmth", "SetFillLightWarmth", "SetHeadLightWarmth", "SetBackLightWarmth"):
            getattr(kit, setter)(0.5)       # 0.5 = white: no colour cast on the colour maps
        kit.SetKeyLightElevation(48)
        kit.SetKeyLightAzimuth(12)
        kit.AddLightsToRenderer(self._ren)
        self._ovl = vtkRenderer()
        self._ovl.SetLayer(1)
        self._ovl.InteractiveOff()
        self._ovl.SetPreserveDepthBuffer(0)
        for r in (self._ren, self._ovl):
            rw.AddRenderer(r)
        self._overlay = Overlay(self._ovl)
        self._apply_background()

    def _apply_background(self) -> None:
        t = self.theme
        r = self._ren
        r.SetBackground(*t.background)
        if t.gradient:
            r.GradientBackgroundOn()
            r.SetBackground(*t.background2)      # bottom
            r.SetBackground2(*t.background)      # top
        else:
            r.GradientBackgroundOff()

    def _touch(self) -> None:
        self.revision += 1

    def _new_handle(self, kind: str) -> str:
        self._count += 1
        return f"{kind}{self._count}"

    # ------------------------------------------------------------------
    # layers: management
    # ------------------------------------------------------------------
    def layers(self) -> dict[str, str]:
        """{handle: kind} in insertion order."""
        return {h: lay.kind for h, lay in self._layers.items()}

    def remove(self, handle: str) -> None:
        """Remove one layer (unknown handles are ignored)."""
        lay = self._layers.pop(handle, None)
        if lay is None:
            return
        if self._rw is not None:
            for a in lay.actors:
                self._ren.RemoveViewProp(a)
        if self._colorbar is not None and self._colorbar.get("layer") == handle:
            self._colorbar["layer"] = None
        if self._colorbar2 is not None and self._colorbar2.get("layer") == handle:
            self._colorbar2 = None
        self._touch()

    def clear(self) -> None:
        """Remove every layer and the colour bars (camera, theme and axes stay)."""
        for h in list(self._layers):
            self.remove(h)
        self._colorbar = self._colorbar2 = None
        self._touch()

    def set_visible(self, handle: str, visible: bool) -> None:
        lay = self._layers.get(handle)
        if lay is None:
            return
        lay.visible = bool(visible)
        for a in lay.actors:
            a.SetVisibility(lay.visible)
        self._touch()

    def set_scalar_range(self, handle: str, vmin: float, vmax: float) -> None:
        """Change the colour range of a colour-mapped layer (re-centred if its map diverges? no: as given)."""
        lay = self._layers.get(handle)
        if lay is None or lay.cmap is None:
            return
        lay.vmin, lay.vmax = float(vmin), float(vmax)
        self._refresh_colors(lay)

    def set_cmap(self, handle: str, cmap: str) -> None:
        lay = self._layers.get(handle)
        if lay is None or lay.cmap is None:
            return
        lay.cmap = cmap
        self._refresh_colors(lay)

    def layer_points(self, handle: str) -> np.ndarray | None:
        """Points (N, 3) of a generated streamline / iso-surface layer (None for other layers).

        For streamlines these are the integrated line points (before tubes),
        so callers can export or check them; see also :meth:`layer_lines`.
        """
        lay = self._layers.get(handle)
        pd = lay.extra.get("polydata") if lay is not None else None
        if pd is None or pd.GetNumberOfPoints() == 0:
            return None if pd is None else np.zeros((0, 3))
        from vtkmodules.util.numpy_support import vtk_to_numpy
        return np.array(vtk_to_numpy(pd.GetPoints().GetData()), dtype=np.float64)

    def layer_lines(self, handle: str) -> list[np.ndarray]:
        """The polylines of a streamline layer, one (k, 3) array per line."""
        pts = self.layer_points(handle)
        lay = self._layers.get(handle)
        if pts is None or lay is None or lay.kind != "streamlines":
            return []
        from vtkmodules.util.numpy_support import vtk_to_numpy
        cells = lay.extra["polydata"].GetLines()
        offs = vtk_to_numpy(cells.GetOffsetsArray())
        conn = vtk_to_numpy(cells.GetConnectivityArray())
        return [pts[conn[offs[i]:offs[i + 1]]] for i in range(len(offs) - 1)]

    def scalar_info(self, handle: str) -> dict | None:
        """{field, cmap, vmin, vmax} of a colour-mapped layer, else None."""
        lay = self._layers.get(handle)
        if lay is None or lay.cmap is None:
            return None
        return {"field": lay.field, "cmap": lay.cmap, "vmin": lay.vmin, "vmax": lay.vmax}

    def _refresh_colors(self, lay: _Layer) -> None:
        if self._rw is not None:
            if lay.rgb_rebuild is not None:
                lay.rgb_rebuild()
            else:
                new = lookup_table(lay.cmap, lay.vmin, lay.vmax)
                for mapper in lay.luts:
                    mapper.SetLookupTable(new)
                    mapper.SetScalarRange(lay.vmin, lay.vmax)
        self._touch()

    def _resolve_range(self, values, rng, field_name, cmap) -> tuple[float, float]:
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
            if not hi > lo:
                hi = lo + max(abs(lo) * 1e-6, 1e-12)
            return lo, hi
        div = field_info(field_name).diverging if field_name else (cmap in DIVERGING_CMAPS)
        if cmap in DIVERGING_CMAPS:
            div = True
        return display_range(values, field=field_name, diverging=div)

    def _add_layer(self, kind: str, lay: _Layer) -> str:
        h = self._new_handle(kind)
        self._layers[h] = lay
        if self._rw is not None:
            for a in lay.actors:
                self._ren.AddViewProp(a)
        self._touch()
        return h

    # ------------------------------------------------------------------
    # meshes
    # ------------------------------------------------------------------
    def _surface_property(self, prop, color, opacity: float, edges: bool) -> None:
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.16)
        prop.SetDiffuse(0.84)
        prop.SetSpecular(0.10)
        prop.SetSpecularPower(18.0)
        prop.SetSpecularColor(1.0, 1.0, 1.0)
        if color is not None:
            prop.SetColor(*color)
        prop.SetOpacity(float(opacity))
        prop.SetEdgeColor(*self.theme.edge)
        prop.SetEdgeVisibility(bool(edges))
        prop.SetLineWidth(1.0)

    def add_mesh(self, mesh, scalars=None, *, cmap: str | None = None, range=None,
                 edges: bool | None = None, opacity: float = 1.0, color=None, field: str | None = None,
                 smooth: bool = True, feature_angle: float = 45.0) -> str:
        """Add a ``Mesh3D`` / ``SurfaceData3D`` (or (vertices, faces)), optionally colour-mapped.

        ``scalars``: None (uniform body grey or ``color``), a field name of a
        SurfaceData3D ("cp", "cf", "sensitivity", "tau") or a per-vertex array.
        ``range`` None picks :func:`display_range` (centred for Cp and
        sensitivity). ``edges`` None follows the scene's edge toggle.
        """
        ma = mesh_arrays(mesh)
        values, fname = None, field
        if isinstance(scalars, str):
            fname = fname or scalars
            if scalars not in ma.fields:
                raise KeyError(f"mesh {ma.name!r} has no per-vertex field {scalars!r}; "
                               f"available: {', '.join(ma.fields) or 'none'}")
            values = ma.fields[scalars]
        elif scalars is not None:
            values = np.asarray(scalars, dtype=np.float64).reshape(-1)
            if len(values) != len(ma.vertices):
                raise ValueError(f"scalars have {len(values)} values for {len(ma.vertices)} vertices")
        used = ma.vertices[np.unique(ma.faces)] if len(ma.faces) else ma.vertices[:0]
        lay = _Layer("mesh", [], _bounds_of(used))
        lay.extra.update(edges_explicit=edges, n_faces=len(ma.faces), name=ma.name, fit_points=_fit_sample(used),
                         surface=(ma.vertices, ma.faces))     # wall distance of vortex iso-surfaces
        if values is not None:
            cmap = cmap or (field_info(fname).cmap if fname else "viridis")
            lay.field, lay.cmap, lay.values = fname, cmap, values
            lay.vmin, lay.vmax = self._resolve_range(values, range, fname, cmap)
        if self._rw is not None and len(ma.faces):
            lay.actors.append(self._mesh_actor(ma, values, lay, color, opacity,
                                               self._edges if edges is None else edges, smooth, feature_angle))
        return self._add_layer("mesh", lay)

    def add_meshes(self, meshes, scalars=None, *, cmap: str | None = None, range=None,
                   **kw) -> list[str]:
        """Add several meshes with ONE shared colour range (e.g. all surfaces of a result)."""
        meshes = list(meshes)
        if scalars is not None and range is None and meshes:
            vals = []
            for m in meshes:
                ma = mesh_arrays(m)
                if isinstance(scalars, str) and scalars in ma.fields:
                    vals.append(ma.fields[scalars])
            if vals:
                fname = kw.get("field") or (scalars if isinstance(scalars, str) else None)
                c = cmap or (field_info(fname).cmap if fname else "viridis")
                range = self._resolve_range(np.concatenate(vals), None, fname, c)
        return [self.add_mesh(m, scalars, cmap=cmap, range=range, **kw) for m in meshes]

    def _mesh_actor(self, ma, values, lay: _Layer, color, opacity, edges, smooth, feature_angle):
        from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        pd = polydata_to_vtk(ma.vertices, ma.faces, {"scalars": values} if values is not None else None)
        mapper = vtkPolyDataMapper()
        if smooth:
            nrm = vtkPolyDataNormals()
            nrm.SetInputData(pd)
            nrm.SetFeatureAngle(float(feature_angle))
            nrm.SplittingOn()
            nrm.ConsistencyOn()           # imported meshes come with mixed winding
            nrm.AutoOrientNormalsOff()    # needs a closed surface; two-sided lighting covers the rest
            nrm.ComputeCellNormalsOff()
            nrm.Update()
            mapper.SetInputData(nrm.GetOutput())
        else:
            mapper.SetInputData(pd)
        if values is not None:
            mapper.SetLookupTable(lookup_table(lay.cmap, lay.vmin, lay.vmax))
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("scalars")
            mapper.SetScalarRange(lay.vmin, lay.vmax)
            mapper.InterpolateScalarsBeforeMappingOn()   # exact LUT colour per pixel, no Gouraud bleeding
            mapper.ScalarVisibilityOn()
            lay.luts.append(mapper)
        else:
            mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        lay.themed_color = color is None and values is None
        rgb = self.theme.body if color is None else _rgb(color)
        self._surface_property(actor.GetProperty(), rgb, opacity, edges)
        return actor

    def set_edges(self, on: bool) -> None:
        """Show / hide triangle edges on every mesh that did not fix its own setting."""
        self._edges = bool(on)
        for lay in self._layers.values():
            if lay.kind == "mesh" and lay.extra.get("edges_explicit") is None:
                for a in lay.actors:
                    a.GetProperty().SetEdgeVisibility(self._edges)
        self._touch()

    @property
    def edges(self) -> bool:
        return self._edges

    # ------------------------------------------------------------------
    # ground
    # ------------------------------------------------------------------
    def add_ground(self, z: float = 0.0, extent=None, *, spacing: float | None = None) -> str:
        """A ground plane at height ``z`` with a subtle grid (major lines every 5 cells).

        ``extent`` = (x0, x1, y0, y1); default: a square around the bodies, 3x
        their largest plan dimension, so it reads as a floor without
        dominating the picture. Grid spacing defaults to a round 1/2/5 step
        giving ~24 cells across.
        """
        if extent is None:
            fb = self.focus_bounds()
            cx, cy = 0.5 * (fb[0] + fb[1]), 0.5 * (fb[2] + fb[3])
            half = 1.6 * max(fb[1] - fb[0], fb[3] - fb[2], 1e-3)
            extent = (cx - half, cx + half, cy - half, cy + half)
        x0, x1, y0, y1 = (float(e) for e in extent)
        if spacing is None:
            spacing = _nice_step(max(x1 - x0, y1 - y0) / 24.0)
        lay = _Layer("ground", [], (x0, x1, y0, y1, z, z))
        lay.extra.update(z=float(z), extent=(x0, x1, y0, y1), spacing=float(spacing))
        if self._rw is not None:
            lay.actors = self._ground_actors(lay)
        return self._add_layer("ground", lay)

    def _ground_actors(self, lay: _Layer) -> list:
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        z = lay.extra["z"]
        x0, x1, y0, y1 = lay.extra["extent"]
        s = lay.extra["spacing"]
        t = self.theme
        quad = polydata_to_vtk(np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]),
                               np.array([[0, 1, 2], [0, 2, 3]]))
        m = vtkPolyDataMapper()
        m.SetInputData(quad)
        m.ScalarVisibilityOff()
        # push the floor back in depth so the grid lines (and a car's tyre contact) win the depth test
        m.SetRelativeCoincidentTopologyPolygonOffsetParameters(2.0, 4.0)
        plane = vtkActor()
        plane.SetMapper(m)
        p = plane.GetProperty()
        p.SetColor(*t.ground)
        p.SetAmbient(1.0)
        p.SetDiffuse(0.0)
        p.SetSpecular(0.0)
        p.LightingOff()
        acts = [plane]
        i0, i1 = math.ceil(x0 / s - 1e-9), math.floor(x1 / s + 1e-9)
        j0, j1 = math.ceil(y0 / s - 1e-9), math.floor(y1 / s + 1e-9)
        for major in (False, True):
            segs = []
            for i in range(i0, i1 + 1):
                if (i % 5 == 0) == major:
                    segs.append(((i * s, y0, z), (i * s, y1, z)))
            for j in range(j0, j1 + 1):
                if (j % 5 == 0) == major:
                    segs.append(((x0, j * s, z), (x1, j * s, z)))
            if not segs:
                continue
            a = _lines_actor(np.array(segs, dtype=float), t.grid_major if major else t.grid, 1.0)
            acts.append(a)
        # thin outline so the floor has an edge rather than fading into the background
        border = np.array([[(x0, y0, z), (x1, y0, z)], [(x1, y0, z), (x1, y1, z)],
                           [(x1, y1, z), (x0, y1, z)], [(x0, y1, z), (x0, y0, z)]], dtype=float)
        acts.append(_lines_actor(border, t.grid_major, 1.0))
        return acts

    def add_box_outline(self, box, *, width: float = 1.5) -> str:
        """The twelve edges of a closed domain's box (x0, x1, y0, y1, z0, z1) in the theme's strong border.

        A room's radiator, fan and column floated in empty space without their
        walls (Setup drew the box, the run view did not): the outline gives the
        scale and the boundaries the smoke and the heat stop at. Not counted in
        the fit (the page frames the room itself with :meth:`set_focus`).
        """
        b = tuple(float(v) for v in box)
        if len(b) != 6 or not all(math.isfinite(v) for v in b):
            raise ValueError("box needs six finite numbers (x0, x1, y0, y1, z0, z1)")
        lay = _Layer("outline", [], b)
        lay.extra.update(box=b, width=float(width))
        if self._rw is not None:
            lay.actors = self._outline_actors(lay)
        return self._add_layer("outline", lay)

    def _outline_actors(self, lay: _Layer) -> list:
        x0, x1, y0, y1, z0, z1 = lay.extra["box"]
        c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        segs = np.array([[c[i], c[j]] for i, j in edges], dtype=float)
        return [_lines_actor(segs, self.theme.border, lay.extra["width"])]

    # ------------------------------------------------------------------
    # slices
    # ------------------------------------------------------------------
    def add_slice(self, data, axis: str | None = None, position: float | None = None, field: str = "speed", *,
                  cmap: str | None = None, range=None, opacity: float = 1.0,
                  interpolate: bool = True, outline: bool = True) -> str:
        """A coloured cut plane from a ``SliceData`` or cut from a ``VolumeData``.

        For a VolumeData, ``axis`` ("x", "y", "z"; default "y") and
        ``position`` (metres, default the middle) choose the plane. A
        SliceData is drawn where it was cut (its own axis / position). Solid
        cells (mask > 0.5) are neutral grey, like the 2D views. Slices are
        unlit, so colours match the colour bar exactly.
        """
        from ..model import SliceData, VolumeData

        if isinstance(data, VolumeData):
            sd = slice_from_volume(data, axis or "y", position, field)
        elif isinstance(data, SliceData) or hasattr(data, "extent"):
            sd = data
            if axis is not None and axis != sd.axis:
                raise ValueError(f"slice is a {sd.axis!r} cut, not {axis!r}")
        else:
            raise TypeError(f"add_slice needs SliceData or VolumeData, not {type(data).__name__}")
        values = volume_field(sd.fields, field) if field in sd.fields or field == "speed" else None
        if values is None:
            raise KeyError(f"slice has no field {field!r}; available: {', '.join(sd.fields)}")
        values = np.asarray(values, dtype=np.float64)
        mask = None if sd.mask is None else np.asarray(sd.mask)
        cmap = cmap or field_info(field).cmap
        valid = values if mask is None else values[~(mask > 0.5)]
        corners = slice_corners(sd.axis, float(sd.position), sd.extent)
        lay = _Layer("slice", [], _bounds_of(corners), field=field, cmap=cmap, values=valid)
        lay.vmin, lay.vmax = self._resolve_range(valid, range, field, cmap)
        lay.extra.update(axis=sd.axis, position=float(sd.position), extent=tuple(sd.extent))
        if self._rw is not None:
            lay.actors = self._slice_actors(lay, values, mask, corners, opacity, interpolate, outline)
        return self._add_layer("slice", lay)

    def _slice_actors(self, lay: _Layer, values, mask, corners, opacity, interpolate, outline) -> list:
        from vtkmodules.util.numpy_support import numpy_to_vtk
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkTexture

        nrow, ncol = values.shape
        img = vtkImageData()
        img.SetDimensions(ncol, nrow, 1)
        tex = vtkTexture()
        tex.SetInputData(img)
        tex.SetInterpolate(bool(interpolate))
        tex.EdgeClampOn()
        tex.RepeatOff()
        tex.SetColorModeToDirectScalars()

        def rebuild() -> None:
            rgb = colorize(values, lay.cmap, lay.vmin, lay.vmax, mask=mask,
                           mask_rgb=tuple(round(c * 255) for c in self.theme.solid))
            arr = numpy_to_vtk(np.ascontiguousarray(rgb.reshape(-1, 3)), deep=True)
            arr.SetName("rgb")
            img.GetPointData().SetScalars(arr)
            img.Modified()
            tex.Modified()

        rebuild()
        lay.rgb_rebuild = rebuild
        pd = polydata_to_vtk(corners, np.array([[0, 1, 2], [0, 2, 3]]))
        # texture coordinates at the outer cell edges; texel centres = cell centres
        tc = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        tca = numpy_to_vtk(tc, deep=True)
        tca.SetName("tcoords")
        pd.GetPointData().SetTCoords(tca)
        m = vtkPolyDataMapper()
        m.SetInputData(pd)
        m.ScalarVisibilityOff()
        a = vtkActor()
        a.SetMapper(m)
        a.SetTexture(tex)
        p = a.GetProperty()
        p.LightingOff()
        p.SetAmbient(1.0)
        p.SetDiffuse(0.0)
        p.SetColor(1.0, 1.0, 1.0)
        p.SetOpacity(float(opacity))
        acts = [a]
        if outline:
            segs = np.array([[corners[i], corners[(i + 1) % 4]] for i in range(4)])
            acts.append(_lines_actor(segs, self.theme.border, 1.0))
        return acts

    # ------------------------------------------------------------------
    # streamlines
    # ------------------------------------------------------------------
    def _seed_points(self, seeds, n, vb) -> tuple[np.ndarray, str]:
        """(points, default direction) for a seed argument: a preset name, a rake dict /
        SeedRake (:func:`.flow.resolve_rake`) or explicit points."""
        fb = self._mesh_bounds()
        if fb is None:
            cx, cy, cz = (0.5 * (vb[0] + vb[1]), 0.5 * (vb[2] + vb[3]), 0.5 * (vb[4] + vb[5]))
            q = [0.15 * (vb[1] - vb[0]), 0.15 * (vb[3] - vb[2]), 0.15 * (vb[5] - vb[4])]
            fb = (cx - q[0], cx + q[0], cy - q[1], cy + q[1], cz - q[2], cz + q[2])
        if isinstance(seeds, str):
            return rake_seeds(fb, vb, seeds, n), "forward"
        if isinstance(seeds, SeedRake):
            return np.asarray(seeds.points, dtype=np.float64), "forward"
        if isinstance(seeds, dict):
            spec = dict(seeds)
            if n is not None and "n" not in spec:
                spec["n"] = n
            return resolve_rake(spec, fb, vb).points, "forward"
        return np.asarray(seeds, dtype=np.float64).reshape(-1, 3), "both"

    def add_streamlines(self, volume, seeds="rake", *, n: int | None = None, color_by: str | None = "speed",
                        cmap: str | None = None, range=None, tubes: bool = True,
                        radius: float | None = None, direction: str | None = None,
                        max_steps: int = 4000, color=None) -> str:
        """Streamlines through a VolumeData's (u, v, w), or a traced :class:`.flow.StreamlineSet`.

        ``seeds``: "rake" / "plane" / "vertical" (see :func:`rake_seeds`; placed
        upstream of the meshes already in the scene, or of the middle of the
        volume when there are none), a rake dict or :class:`.flow.SeedRake`
        (:func:`.flow.resolve_rake`: the GUI's rake), or an (N, 3) array of points.
        Rake seeds integrate forward; explicit points both ways unless
        ``direction`` says otherwise. Velocity is zeroed inside solid cells so
        lines stop at walls instead of tunnelling through a Brinkman body.
        ``tubes`` draws shaded tubes (radius default 0.18 % of the lines' extent,
        then kept for updates), otherwise 1.5 px lines; ``color_by`` None draws
        them in one colour (``color`` or the theme's secondary text colour).
        Update in place with :meth:`update_streamlines`.
        """
        lay = _Layer("streamlines", [], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), field=color_by if color_by else None)
        lay.extra.update(seeds_arg=seeds, n=n, direction=direction, max_steps=int(max_steps), tubes=bool(tubes),
                         radius=radius, color=color, cmap_arg=cmap, range_arg=range)
        sset, values = self._streamline_data(lay, volume)
        lay.extra["seeds"] = sset.seeds
        if color_by and values is not None:
            lay.cmap = cmap or field_info(color_by).cmap
            lay.vmin, lay.vmax = self._resolve_range(values, range, color_by, lay.cmap)
            lay.values = values
        if self._rw is None:
            lay.extra["polydata"] = None
            return self._add_layer("streamlines", lay)
        lay.actors = self._line_actors(sset, lay)
        return self._add_layer("streamlines", lay)

    def _streamline_data(self, lay: _Layer, data):
        """Trace (or take) the lines of a streamline layer: (StreamlineSet, colour values)."""
        color_by = lay.field
        if isinstance(data, StreamlineSet):
            sset = data
            vals = sset.values.get(color_by) if color_by else None
            lay.bounds = _bounds_of(np.asarray(sset.points, dtype=np.float64)) if len(sset.points) else lay.bounds
            return sset, vals
        vb = volume_bounds(data)
        lay.bounds = vb
        pts, default_dir = self._seed_points(lay.extra["seeds_arg"], lay.extra["n"], vb)
        extra = (color_by,) if color_by and color_by != "speed" else ()
        sset = trace_streamlines(data, pts, direction=lay.extra["direction"] or default_dir,
                                 max_steps=lay.extra["max_steps"], extra=extra)
        vals = None
        if color_by:
            vals = _fluid_sample(data, color_by)
            if vals is None:
                vals = sset.values.get(color_by)
        return sset, vals

    def _line_actors(self, sset: StreamlineSet, lay: _Layer) -> list:
        from vtkmodules.vtkFiltersCore import vtkTubeFilter
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        lines = sset.to_polydata()
        lay.extra["polydata"] = lines
        lay.extra["n_points"] = lines.GetNumberOfPoints()
        lay.extra["n_lines"] = lines.GetNumberOfLines()
        mapper = vtkPolyDataMapper()
        tube = None
        if lay.extra["tubes"]:
            tube = vtkTubeFilter()
            # 8 sides with shared (smoothly shaded) vertices read as round even close up; each
            # side is ~10 % of the tube's build and draw time at 300 lines
            tube.SetNumberOfSides(8)
            tube.CappingOn()
            tube.SidesShareVerticesOn()
            lay.extra["tube"] = tube
            self._set_tube_input(lay, lines)
            mapper.SetInputData(tube.GetOutput())
        else:
            mapper.SetInputData(lines)
        lay.extra["mapper"] = mapper
        actor = vtkActor()
        actor.SetMapper(mapper)
        color = lay.extra["color"]
        if lay.field and lay.cmap is not None:
            mapper.SetLookupTable(lookup_table(lay.cmap, lay.vmin, lay.vmax))
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(lay.field)
            mapper.SetScalarRange(lay.vmin, lay.vmax)
            mapper.InterpolateScalarsBeforeMappingOn()
            mapper.ScalarVisibilityOn()
            lay.luts.append(mapper)
        else:
            lay.field = None
            mapper.ScalarVisibilityOff()
            lay.themed_color = color is None           # uncoloured lines follow the theme's text colour
        self._surface_property(actor.GetProperty(), _rgb(color) if color is not None else self.theme.text2,
                               1.0, False)
        # thin tubes are mostly silhouette: more ambient keeps their colour readable against the map
        actor.GetProperty().SetAmbient(0.35)
        actor.GetProperty().SetDiffuse(0.70)
        if not lay.extra["tubes"]:
            actor.GetProperty().SetLineWidth(1.5)
            actor.GetProperty().LightingOff()
        return [actor]

    def _set_tube_input(self, lay: _Layer, lines) -> None:
        tube = lay.extra["tube"]
        if lay.extra.get("tube_radius") is None:
            # thickness from the extent of the first lines, then kept: thin next to a car in a
            # 10 m domain, visible around a 1 m sphere, and steady while a live run updates them.
            # Capped at a fifth of the seed spacing, so neighbouring tubes never merge into a
            # ribbon: 30 seeds across a small sphere's rake sit closer than 0.36 % of the lines'
            # length, which is what a tube from the extent alone would be across
            r = lay.extra["radius"]
            if not r:
                lb = lines.GetBounds() if lines.GetNumberOfPoints() else (0, 1, 0, 1, 0, 1)
                r = 0.0018 * (math.dist((lb[0], lb[2], lb[4]), (lb[1], lb[3], lb[5])) or 1.0)
                gap = _seed_spacing(lay.extra.get("seeds"))
                if gap > 0:
                    r = min(r, 0.2 * gap)
            lay.extra["tube_radius"] = float(r)
        tube.SetRadius(lay.extra["tube_radius"])
        tube.SetInputData(lines)
        tube.Update()

    def update_streamlines(self, handle: str, data, *, seeds=None, rescale: bool = False) -> None:
        """Replace the lines of a streamline layer in place (same actor, no flashing).

        ``data`` is a VolumeData (re-traced from the layer's seeds, or ``seeds``
        when given) or a traced :class:`.flow.StreamlineSet`. The colour range
        and tube radius stay those of the first lines unless ``rescale``, so a
        live run's lines do not change colour from frame to frame.
        """
        lay = self._layers.get(handle)
        if lay is None or lay.kind != "streamlines":
            raise KeyError(f"no streamline layer {handle!r}")
        if seeds is not None:
            lay.extra["seeds_arg"] = seeds
        keep_field = lay.field
        sset, values = self._streamline_data(lay, data)
        lay.extra["seeds"] = sset.seeds
        if rescale and keep_field and values is not None:
            lay.vmin, lay.vmax = self._resolve_range(values, lay.extra["range_arg"], keep_field, lay.cmap)
            lay.values = values
        if self._rw is not None and lay.actors:
            lines = sset.to_polydata()
            lay.extra["polydata"] = lines
            lay.extra["n_points"] = lines.GetNumberOfPoints()
            lay.extra["n_lines"] = lines.GetNumberOfLines()
            if lay.extra.get("tube") is not None:
                self._set_tube_input(lay, lines)
                lay.extra["mapper"].SetInputData(lay.extra["tube"].GetOutput())
            else:
                lay.extra["mapper"].SetInputData(lines)
            if rescale and lay.cmap is not None:
                self._refresh_colors(lay)
        self._touch()

    # ------------------------------------------------------------------
    # smoke (volume rendering of the passive scalar, DESIGN.md 7.13)
    # ------------------------------------------------------------------
    def add_smoke(self, volume, field: str = "dye", *, opacity: float = 1.0, color="auto",
                  threshold: float = SMOKE_THRESHOLD, opacity_curve=None, gain: float = 1.0) -> str:
        """Smoke: a GPU volume rendering of ``field`` (0..1) of a VolumeData.

        Composited ray casting (``vtkSmartVolumeMapper``) of the smoke
        concentration through a fixed transfer function: no smoke below
        ``threshold``, then opacity rising with concentration to the theme's
        ``smoke_opacity`` x ``opacity`` per voxel (unit distance = one voxel, so
        the look does not depend on the grid) along a convex curve (the faint
        haze stays clear; ``opacity_curve`` replaces it with other (concentration,
        share of that opacity) points, concentrations in (0, 1], shares in [0, 1]),
        softly shaded by the scene's lights, in one colour --
        near-white wisps on the dark theme, a dark translucent grey on the light
        one (``color="auto"``), or ``color``. The transfer function never adapts to
        the data, so a live run's smoke does not flicker as it develops; the
        sample distance is fixed (half a voxel) and jittering is off for the
        same reason. Solid cells are cleared. Mirrored (symmetry-plane) volumes
        are shown whole. Update in place with :meth:`update_smoke`.

        ``gain`` multiplies the concentration before the transfer function
        (clipped at 1): a point emitter's thin stream diffuses to 0.05-0.15
        within a few cells (car run: 90 % of its smoky voxels below 0.15), where
        the convex curve is nearly clear, so it vanished; the caller lifts such a
        channel with :func:`smoke_gain`. The colour darkens towards the dense
        cores on light backgrounds and brightens on dark ones
        (:data:`SMOKE_CORE_SHADE`), so smoke reads as a shaded volume rather than
        a flat tint the floor swallows.
        """
        curve = _opacity_curve(opacity_curve)
        gain = float(gain) if math.isfinite(float(gain)) and float(gain) > 0 else 1.0
        origin, spacing, arr = self._smoke_array(volume, field, gain)
        nz, ny, nx = arr.shape
        vb = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
              origin[2], origin[2] + nz * spacing[2])
        lay = _Layer("smoke", [], vb)
        lay.extra.update(field=field, opacity=float(opacity), color=color, threshold=float(threshold), curve=curve,
                         gain=gain, shape=arr.shape, spacing=spacing, origin=origin,
                         max=float(arr.max()) / 255.0 if arr.size else 0)
        if self._rw is None:
            return self._add_layer("smoke", lay)
        from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolume, vtkVolumeProperty

        _init_volume_rendering()
        from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

        img = vtkImageData()
        lay.extra["image"] = img
        self._set_smoke_image(lay, origin, spacing, arr)
        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(img)
        mapper.SetBlendModeToComposite()
        mapper.SetRequestedRenderModeToDefault()          # GPU ray casting where available
        mapper.SetAutoAdjustSampleDistances(0)
        mapper.SetSampleDistance(0.5 * min(spacing))
        mapper.SetInteractiveAdjustSampleDistances(0)
        if hasattr(mapper, "SetUseJittering"):
            mapper.SetUseJittering(0)
        prop = vtkVolumeProperty()
        prop.SetInterpolationTypeToLinear()
        # soft gradient shading under the scene's light kit gives the wisps their volume; no
        # specular (smoke does not shine) and a high ambient term so the shadowed side stays smoke
        prop.ShadeOn()
        prop.SetAmbient(0.55)
        prop.SetDiffuse(0.55)
        prop.SetSpecular(0.0)
        prop.SetScalarOpacityUnitDistance(float(min(spacing)))
        lay.extra.update(mapper=mapper, prop=prop, ctf=vtkColorTransferFunction(), otf=vtkPiecewiseFunction())
        self._smoke_transfer(lay)
        prop.SetColor(lay.extra["ctf"])
        prop.SetScalarOpacity(lay.extra["otf"])
        actor = vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        lay.actors = [actor]
        return self._add_layer("smoke", lay)

    def _smoke_array(self, volume, field: str, gain: float = 1.0):
        """(origin, spacing, uint8 (nz, ny, nx)) of a volume's smoke times ``gain``, mirrored when
        needed, solids cleared. uint8 is plenty for a picture and uploads 4x faster than float."""
        if field not in volume.fields:
            raise KeyError(f"volume has no field {field!r}; available: {', '.join(sorted(volume.fields))}")
        origin = tuple(float(c) for c in volume.origin)
        spacing = tuple(float(c) for c in volume.spacing)
        a = np.asarray(volume.fields[field])
        if a.dtype == np.uint8 and gain == 1.0:
            q = a.copy()
        else:
            f = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32, copy=False)
            q = np.rint(np.clip(f * float(gain), 0.0, 1.0) * 255.0).astype(np.uint8)
        if volume.mask is not None:
            q[_wall_cleared(volume.mask)] = 0
        if getattr(volume, "mirrored", False):
            ny = q.shape[1]
            y_hi = origin[1] + ny * spacing[1]
            q = np.concatenate([q[:, ::-1, :], q], axis=1)
            origin = (origin[0], -y_hi, origin[2])
        return origin, spacing, np.ascontiguousarray(q)

    def _set_smoke_image(self, lay: _Layer, origin, spacing, arr: np.ndarray) -> None:
        from vtkmodules.util.numpy_support import numpy_to_vtk

        img = lay.extra["image"]
        nz, ny, nx = arr.shape
        if tuple(img.GetDimensions()) != (nx, ny, nz):
            img.SetDimensions(nx, ny, nz)
        img.SetSpacing(*spacing)
        img.SetOrigin(*(origin[i] + 0.5 * spacing[i] for i in range(3)))
        flat = arr.reshape(-1)
        va = numpy_to_vtk(flat, deep=False, array_type=3)     # VTK_UNSIGNED_CHAR, zero copy
        va.SetName("smoke")
        lay.extra["buffer"] = flat                            # keeps the zero-copy memory alive
        img.GetPointData().SetScalars(va)
        img.Modified()

    def _smoke_transfer(self, lay: _Layer) -> None:
        """(Re)build the smoke's colour and opacity functions for the current theme."""
        t = self.theme
        color = lay.extra["color"]
        rgb = t.smoke if color in (None, "auto") else smoke_display_rgb(_rgb(color))
        ctf, otf = lay.extra["ctf"], lay.extra["otf"]
        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(0.0, *rgb)
        # density-weighted shading: the wisps keep the emitter's colour, the dense cores go towards
        # black on a light background (towards white on a dark one), so a stream has an edge to read
        core = smoke_core_colour(rgb, bool(t.dark))
        ctf.AddRGBPoint(255.0 * 0.35, *rgb)
        ctf.AddRGBPoint(255.0, *core)
        a = max(0.0, min(1.0, t.smoke_opacity * lay.extra["opacity"]))
        c0 = 255.0 * lay.extra["threshold"]
        otf.RemoveAllPoints()
        otf.AddPoint(0.0, 0.0)
        otf.AddPoint(c0, 0.0)
        # convex rise: the faint haze numerical spreading leaves around a stream stays clear and the
        # streams' cores carry the picture; per voxel, so a stream seen edge-on is still translucent
        for c, f in lay.extra.get("curve") or SMOKE_OPACITY_CURVE:
            otf.AddPoint(max(c0 + 1.0, 255.0 * c), a * f)

    def update_smoke(self, handle: str, volume, field: str | None = None, *, gain: float | None = None) -> None:
        """Replace the smoke data in place (same volume actor and transfer function: a live
        run's smoke changes only where the flow changed). ``gain`` None keeps the layer's."""
        lay = self._layers.get(handle)
        if lay is None or lay.kind != "smoke":
            raise KeyError(f"no smoke layer {handle!r}")
        field = field or lay.extra["field"]
        if gain is not None and math.isfinite(float(gain)) and float(gain) > 0:
            lay.extra["gain"] = float(gain)
        origin, spacing, arr = self._smoke_array(volume, field, lay.extra.get("gain", 1.0))
        nz, ny, nx = arr.shape
        lay.bounds = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
                      origin[2], origin[2] + nz * spacing[2])
        lay.extra.update(field=field, shape=arr.shape, max=float(arr.max()) / 255.0 if arr.size else 0.0)
        if self._rw is not None and lay.actors:
            self._set_smoke_image(lay, origin, spacing, arr)
            if tuple(spacing) != tuple(lay.extra["spacing"]):
                lay.extra["mapper"].SetSampleDistance(0.5 * min(spacing))
                lay.extra["prop"].SetScalarOpacityUnitDistance(float(min(spacing)))
        lay.extra.update(spacing=spacing, origin=origin)
        self._touch()

    def set_smoke_opacity(self, handle: str, opacity: float) -> None:
        """Scale the smoke's density (1 = the theme's default)."""
        lay = self._layers.get(handle)
        if lay is None or lay.kind != "smoke":
            return
        lay.extra["opacity"] = float(opacity)
        if self._rw is not None and lay.actors:
            self._smoke_transfer(lay)
        self._touch()

    # ------------------------------------------------------------------
    # seed rake (the device smoke and streamlines start from)
    # ------------------------------------------------------------------
    def add_seed_rake(self, rake, *, body_bounds=None, domain_bounds=None, nozzles: bool = True,
                      handle_marker: bool = True) -> str:
        """Draw a seed rake: its bar with its end points (or the frame and corners of a grid /
        band; explicit points as dots) in the accent colour and, when ``handle_marker``, a
        round grab handle at its centre (the 3D view drags it). ``rake`` is a :class:`.flow.SeedRake`, or a rake dict
        resolved against ``body_bounds`` (default: the meshes) and ``domain_bounds``."""
        sr = self._as_rake(rake, body_bounds, domain_bounds)
        lay = _Layer("rake", [], _bounds_of(np.vstack([sr.points, sr.centre[None, :]])))
        lay.extra.update(rake=sr, nozzles=bool(nozzles), handle_marker=bool(handle_marker))
        if self._rw is not None:
            lay.actors = self._rake_actors(lay)
        return self._add_layer("rake", lay)

    def _as_rake(self, rake, body_bounds, domain_bounds) -> SeedRake:
        if isinstance(rake, SeedRake):
            return rake
        bb = body_bounds if body_bounds is not None else self._mesh_bounds()
        if bb is None:
            bb = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
        if domain_bounds is None:
            lx = bb[1] - bb[0]
            s = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
            domain_bounds = (bb[0] - 3 * lx, bb[1] + 6 * lx, bb[2] - 2 * s, bb[3] + 2 * s, bb[4] - 2 * s,
                             bb[5] + 2 * s)
        return resolve_rake(rake, bb, domain_bounds)

    def _rake_actors(self, lay: _Layer) -> list:
        from vtkmodules.util.numpy_support import numpy_to_vtk
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        sr: SeedRake = lay.extra["rake"]
        acc = self.theme.accent
        acts = []
        seg = sr.outline()
        if len(seg):
            bar = _lines_actor(seg, acc, 2.0)
            bar.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
            acts.append(bar)

        def dots(points: np.ndarray, size: float):
            pts = vtkPoints()
            pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
            verts = vtkCellArray()
            for i in range(len(points)):
                verts.InsertNextCell(1, [i])
            pd = vtkPolyData()
            pd.SetPoints(pts)
            pd.SetVerts(verts)
            m = vtkPolyDataMapper()
            m.SetInputData(pd)
            m.ScalarVisibilityOff()
            a = vtkActor()
            a.SetMapper(m)
            p = a.GetProperty()
            p.SetColor(*acc)
            p.SetPointSize(size)
            p.SetRenderPointsAsSpheres(True)
            p.SetAmbient(0.5)
            p.SetDiffuse(0.6)
            return a

        if lay.extra["nozzles"]:
            # the ParaView line-widget look: a bar with its two end points (a grid / band: its frame),
            # explicit points as dots; the seeds along a bar would only clutter it
            if sr.kind == "points":
                marks = np.asarray(sr.points, dtype=np.float64)[:400]
            elif sr.kind in ("rake", "vertical"):
                marks = sr.outline().reshape(-1, 3)
            else:
                marks = sr.outline()[:, 0, :]
            if len(marks):
                acts.append(dots(marks, 5.0))
                lay.extra["nozzle_actor"] = acts[-1]
        if lay.extra["handle_marker"]:
            acts.append(dots(sr.centre[None, :], 11.0))
            lay.extra["handle_actor"] = acts[-1]
        return acts

    def update_seed_rake(self, handle: str, rake, *, body_bounds=None, domain_bounds=None) -> None:
        """Move / reshape a rake layer (actors rebuilt: a rake is a few dozen points)."""
        lay = self._layers.get(handle)
        if lay is None or lay.kind != "rake":
            raise KeyError(f"no rake layer {handle!r}")
        sr = self._as_rake(rake, body_bounds, domain_bounds)
        lay.extra["rake"] = sr
        lay.bounds = _bounds_of(np.vstack([sr.points, sr.centre[None, :]]))
        if self._rw is not None:
            for a in lay.actors:
                self._ren.RemoveViewProp(a)
            lay.actors = self._rake_actors(lay)
            for a in lay.actors:
                a.SetVisibility(lay.visible)
                self._ren.AddViewProp(a)
        self._touch()

    def rake_of(self, handle: str) -> SeedRake | None:
        lay = self._layers.get(handle)
        return lay.extra.get("rake") if lay is not None and lay.kind == "rake" else None

    def pick_ray(self, px: float, py: float) -> tuple[np.ndarray, np.ndarray]:
        """The viewing ray (origin, unit direction) through pixel (px, py) of the last render
        (x right, y down; the inverse of :meth:`project`)."""
        self._ensure_camera()
        w, h = self._main_size()
        return cam.unproject(self._camera, px, py, w, h)

    # ------------------------------------------------------------------
    # surface streak lines (skin-friction lines, an oil-flow picture)
    # ------------------------------------------------------------------
    def add_surface_streaks(self, surfaces, *, n: int = 600, color=None, max_steps: int = 600,
                            seed: int = 1) -> str:
        """Skin-friction lines on the surface: streamlines of the wall-shear vector ``tau``
        constrained to the mesh (``vtkStreamTracer`` surface mode), from ``n`` area-weighted
        random seeds, both directions -- the picture an oil-flow test gives: attachment,
        separation and reattachment lines. Thin dark lines on the pale body in both themes
        (``color`` overrides)."""
        from vtkmodules.util.numpy_support import numpy_to_vtk
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

        surfaces = surfaces if isinstance(surfaces, (list, tuple)) else [surfaces]
        rng = np.random.default_rng(int(seed))
        usable = []
        for srf in surfaces:
            tau = getattr(srf, "tau", None)
            if tau is None:
                continue
            V = np.asarray(srf.vertices, dtype=np.float64)
            Fc = np.asarray(srf.faces, dtype=np.int64).reshape(-1, 3)
            T = np.asarray(tau, dtype=np.float64).reshape(-1, 3)
            if len(T) != len(V) or not len(Fc):
                continue
            area = 0.5 * np.linalg.norm(np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]]), axis=1)
            if area.sum() > 0:
                usable.append((V, Fc, T, area))
        total_area = sum(u[3].sum() for u in usable) or 1.0
        polys, seeds_all = [], []
        for V, Fc, T, area in usable:
            k = max(1, round(n * area.sum() / total_area))
            fi = rng.choice(len(Fc), size=k, p=area / area.sum())     # area-weighted triangles
            r1, r2 = rng.random(k), rng.random(k)
            s1 = np.sqrt(r1)                                           # uniform inside each triangle
            pts = (V[Fc[fi, 0]] * (1 - s1)[:, None] + V[Fc[fi, 1]] * (s1 * (1 - r2))[:, None]
                   + V[Fc[fi, 2]] * (s1 * r2)[:, None])
            pd = polydata_to_vtk(V, Fc)
            ta = numpy_to_vtk(np.ascontiguousarray(T, dtype=np.float32), deep=True)
            ta.SetName("tau")
            pd.GetPointData().AddArray(ta)
            pd.GetPointData().SetActiveVectors("tau")
            polys.append((pd, pts, float(np.percentile(np.linalg.norm(T, axis=1), 95)) or 1.0))
            seeds_all.append(pts)
        lay = _Layer("streaks", [], _bounds_of(np.vstack(seeds_all)) if seeds_all else (0.0,) * 6)
        if self._rw is None or not polys:
            lay.extra["n_lines"] = 0
            return self._add_layer("streaks", lay)
        from vtkmodules.vtkFiltersCore import vtkAppendPolyData
        app = vtkAppendPolyData()
        total = 0
        for pd, pts, tref in polys:
            vp = vtkPoints()
            vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
            verts = vtkCellArray()
            for i in range(len(pts)):
                verts.InsertNextCell(1, [i])
            sp = vtkPolyData()
            sp.SetPoints(vp)
            sp.SetVerts(verts)
            tr = vtkStreamTracer()
            tr.SetInputData(pd)
            tr.SetSourceData(sp)
            tr.SetInputArrayToProcess(0, 0, 0, 0, "tau")
            tr.SurfaceStreamlinesOn()
            tr.SetIntegratorTypeToRungeKutta4()
            tr.SetIntegrationStepUnit(2)
            tr.SetInitialIntegrationStep(0.3)
            tr.SetMaximumNumberOfSteps(int(max_steps))
            b = pd.GetBounds()
            tr.SetMaximumPropagation(0.6 * math.dist((b[0], b[2], b[4]), (b[1], b[3], b[5])))
            tr.SetTerminalSpeed(1e-3 * tref)
            tr.SetIntegrationDirectionToBoth()
            tr.SetComputeVorticity(False)
            tr.Update()
            total += tr.GetOutput().GetNumberOfLines()
            app.AddInputData(tr.GetOutput())
        app.Update()
        out = app.GetOutput()
        lay.extra.update(n_lines=total, polydata=out)
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        m = vtkPolyDataMapper()
        m.SetInputData(out)
        m.ScalarVisibilityOff()
        m.SetResolveCoincidentTopologyToPolygonOffset()
        m.SetRelativeCoincidentTopologyLineOffsetParameters(-2.0, -4.0)
        a = vtkActor()
        a.SetMapper(m)
        p = a.GetProperty()
        p.SetColor(*(_rgb(color) if color is not None else _streak_rgb(self.theme)))
        p.SetLineWidth(1.0)
        p.LightingOff()
        p.SetOpacity(0.85)
        lay.themed_color = color is None
        lay.actors = [a]
        return self._add_layer("streaks", lay)

    # ------------------------------------------------------------------
    # iso-surfaces
    # ------------------------------------------------------------------
    def add_isosurface(self, volume, field: str = "q", value: float | None = None, *, color_by: str | None = None,
                       cmap: str | None = None, range=None, color=None,
                       opacity: float = 1.0, wall_clearance: float | None = None, solid: str = "below") -> str:
        """An iso-surface of ``field`` (default the Q-criterion) at ``value``.

        ``value`` None: 10 % of the 99.9th percentile of the positive values,
        a robust "strong vortex cores" level for any flow speed or scale.
        Solid cells are set below the level first, so walls never produce a
        surface (``solid="above"`` sets them above it instead: around a heated
        body the fluid next to the wall is above the level, and solids below
        it drew a voxel skin on the body; above, the surface closes around the
        thermal boundary layer). ``color_by`` colours it by another field (e.g.
        "speed"); otherwise it is a uniform muted blue.

        ``wall_clearance`` (cells of the volume) removes the surface closer
        than that to the body; None means :data:`WALL_CLEARANCE_CELLS` for
        the vortex fields (:data:`VORTEX_FIELDS`) of a volume with a solid
        mask and 0 otherwise. Why: the diffuse immersed wall shears the flow
        over its last three cells or so, which gives a thin layer of large Q
        along every surface. That layer is not a vortex and would wrap the
        body in a skin. It is cut along a smooth distance to the body (see
        :meth:`_trim_near_wall`), never by masking whole voxels, which leaves a
        saw-tooth rim of voxel steps where the skin meets the real vortices.
        """
        from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        origin, spacing, fields, mask = full_volume(volume)
        arr = volume_field(fields, field).astype(np.float32).copy()
        finite = arr[np.isfinite(arr)]
        if value is None:
            pos = finite[finite > 0]
            value = 0.1 * float(np.percentile(pos, 99.9)) if pos.size else 0.0
        if solid not in ("below", "above"):
            raise ValueError(f"solid must be 'below' or 'above', not {solid!r}")
        if mask is not None:
            if solid == "above":
                arr[mask > 0.5] = max(float(finite.max()) if finite.size else 0.0, float(value)) + 1.0
            else:
                arr[mask > 0.5] = min(float(finite.min()) if finite.size else 0.0, float(value)) - 1.0
        arrays = {field: arr}
        if color_by:
            arrays[color_by] = volume_field(fields, color_by).astype(np.float32)
        nz, ny, nx = arr.shape
        vb = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
              origin[2], origin[2] + nz * spacing[2])
        lay = _Layer("isosurface", [], vb, field=color_by)
        lay.extra.update(value=float(value), iso_field=field)
        if self._rw is None:
            return self._add_layer("isosurface", lay)
        img = image_to_vtk(origin, spacing, arrays, vectors=None)
        img.GetPointData().SetActiveScalars(field)
        fe = vtkFlyingEdges3D()
        fe.SetInputData(img)
        fe.SetValue(0, float(value))
        fe.ComputeNormalsOn()
        fe.ComputeScalarsOff()
        if hasattr(fe, "SetInterpolateAttributes"):
            fe.SetInterpolateAttributes(bool(color_by))
        fe.Update()
        surf = fe.GetOutput()
        if wall_clearance is None:
            wall_clearance = WALL_CLEARANCE_CELLS if (field in VORTEX_FIELDS and mask is not None) else 0.0
        if wall_clearance > 0 and surf.GetNumberOfPoints() > 0:
            surf, how = self._trim_near_wall(surf, float(wall_clearance) * min(spacing), origin, spacing, mask)
            lay.extra.update(wall_clearance=float(wall_clearance), wall_distance=how)
        if surf.GetNumberOfPoints() > 0:
            lay.bounds = tuple(surf.GetBounds())
        lay.extra["n_triangles"] = surf.GetNumberOfPolys()
        lay.extra["polydata"] = surf
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(surf)
        if color_by and surf.GetPointData().GetArray(color_by) is not None:
            vals = arrays[color_by]
            lay.cmap = cmap or field_info(color_by).cmap
            lay.values = vals
            lay.vmin, lay.vmax = self._resolve_range(vals, range, color_by, lay.cmap)
            mapper.SetLookupTable(lookup_table(lay.cmap, lay.vmin, lay.vmax))
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(color_by)
            mapper.SetScalarRange(lay.vmin, lay.vmax)
            mapper.InterpolateScalarsBeforeMappingOn()
            mapper.ScalarVisibilityOn()
            lay.luts.append(mapper)
        else:
            lay.field = None
            mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        self._surface_property(actor.GetProperty(), _rgb(color) if color is not None else (0.60, 0.71, 0.84),
                               opacity, False)
        lay.actors = [actor]
        return self._add_layer("isosurface", lay)

    def _trim_near_wall(self, surf, clearance: float, origin, spacing, mask):
        """``surf`` without the part closer than ``clearance`` (metres) to the body.

        Returns (polydata, how) where ``how`` names the distance used:

        * ``"mesh"``: distance to the body meshes already in the scene (what the
          viewer sees), from a k-d tree of points spread over their triangles
          half a cell apart. A nearest sample is at most ~0.35 cells sideways,
          so at 3.5 cells out the cut deviates from the true offset surface by
          under 0.02 cell: it is as smooth as the iso-surface itself.
          Measured on the RTX 4060 Ti machine: 22 ms for a 0.2 m sphere at
          10 mm cells, 86 ms for a 13.7k-triangle car at 57 mm cells (an exact
          ``vtkImplicitPolyDataDistance`` took 8 s for 200k points).
        * ``"mask"``: no mesh in the scene; a Gaussian-smoothed Euclidean
          distance transform of the solid mask, interpolated trilinearly.
        * ``"none"``: neither (nothing trimmed).

        The distance becomes a point array and ``vtkClipPolyData`` cuts the
        triangles where it crosses ``clearance``, interpolating along edges, so
        the rim is a smooth curve instead of voxel steps.
        """
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        from vtkmodules.vtkFiltersCore import vtkClipPolyData

        pts = vtk_to_numpy(surf.GetPoints().GetData()).astype(np.float64)
        h = float(min(spacing))
        surfaces = [lay.extra["surface"] for lay in self._layers.values()
                    if lay.kind == "mesh" and lay.extra.get("n_faces", 0) > 0 and "surface" in lay.extra]
        dist = None
        how = "none"
        if surfaces:
            dist = _distance_to_triangles(surfaces, pts, step=h / 2.0, limit=clearance + 2.0 * h)
            how = "mesh"
        elif mask is not None:
            dist = _distance_from_mask(mask, origin, spacing, pts)
            how = "mask"
        if dist is None:
            return surf, "none"
        arr = numpy_to_vtk(np.ascontiguousarray(dist, dtype=np.float32), deep=True)
        arr.SetName("wall_distance")
        surf.GetPointData().AddArray(arr)
        surf.GetPointData().SetActiveScalars("wall_distance")
        clip = vtkClipPolyData()
        clip.SetInputData(surf)
        clip.SetValue(float(clearance))
        clip.InsideOutOff()                 # keep wall_distance > clearance
        clip.Update()
        out = clip.GetOutput()
        out.GetPointData().RemoveArray("wall_distance")
        return out, how

    # ------------------------------------------------------------------
    # placed physics (DESIGN.md 10): scalar volumes, tracer particles, device glyphs
    # ------------------------------------------------------------------
    def add_scalar_volume(self, volume, field: str = "temperature", *, cmap: str | None = None, range=None,
                          opacity: float = 1.0, floor: float = SCALAR_VOLUME_FLOOR, mode: str = "high") -> str:
        """Volume rendering of a scalar coloured by a colour map: a thermal plume around a heated body.

        ``field`` of a VolumeData is mapped through ``cmap`` over ``range``
        (default :func:`display_range` of the fluid cells). Opacity follows the
        value, not the data's statistics: ``mode="high"`` makes the bottom
        ``floor`` of the range clear and the rest increasingly dense towards
        the top (a hot plume in ambient air: the ambient fluid stays clear),
        ``mode="low"`` the mirror image (a cold plume), ``mode="both"`` keeps
        the middle clear and fills both ends (heating and cooling). The transfer function is fixed by the
        range, so a live run updated through :meth:`update_scalar_volume`
        never flickers as its plume grows. Solid cells are transparent; a
        mirrored volume is shown whole. The layer is colour-mapped
        (:meth:`scalar_info`, the colour bar, :meth:`set_cmap`,
        :meth:`set_scalar_range` work as for a slice).
        """
        if mode not in ("high", "low", "both"):
            raise ValueError(f"unknown scalar volume mode {mode!r}")
        origin, spacing, fields, mask = full_volume(volume)
        arr = volume_field(fields, field).astype(np.float32)
        fluid = arr[np.isfinite(arr) & (mask <= 0.5)] if mask is not None else arr[np.isfinite(arr)]
        cmap = cmap or field_info(field).cmap
        vmin, vmax = self._resolve_range(fluid if fluid.size else arr, range, field, cmap)
        nz, ny, nx = arr.shape
        vb = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
              origin[2], origin[2] + nz * spacing[2])
        lay = _Layer("volume", [], vb, field=field, cmap=cmap, vmin=vmin, vmax=vmax,
                     values=fluid[:: max(1, fluid.size // 200_000)] if fluid.size else fluid)
        lay.extra.update(field=field, opacity=float(opacity), floor=float(floor), mode=mode, origin=origin,
                         spacing=spacing, shape=arr.shape, data=arr, mask=mask)
        if self._rw is None:
            return self._add_layer("volume", lay)
        from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolume, vtkVolumeProperty

        _init_volume_rendering()
        from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

        lay.extra["image"] = vtkImageData()
        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(lay.extra["image"])
        mapper.SetBlendModeToComposite()
        mapper.SetRequestedRenderModeToDefault()
        mapper.SetAutoAdjustSampleDistances(0)
        mapper.SetSampleDistance(0.5 * min(spacing))
        mapper.SetInteractiveAdjustSampleDistances(0)
        if hasattr(mapper, "SetUseJittering"):
            mapper.SetUseJittering(0)
        prop = vtkVolumeProperty()
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOn()
        prop.SetAmbient(0.6)
        prop.SetDiffuse(0.45)
        prop.SetSpecular(0.0)
        prop.SetScalarOpacityUnitDistance(float(min(spacing)))
        lay.extra.update(mapper=mapper, prop=prop, ctf=vtkColorTransferFunction(), otf=vtkPiecewiseFunction())
        prop.SetColor(lay.extra["ctf"])
        prop.SetScalarOpacity(lay.extra["otf"])
        self._scalar_volume_image(lay)
        self._scalar_volume_transfer(lay)
        lay.rgb_rebuild = lambda: (self._scalar_volume_image(lay), self._scalar_volume_transfer(lay))
        actor = vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        lay.actors = [actor]
        return self._add_layer("volume", lay)

    def _scalar_volume_image(self, lay: _Layer) -> None:
        """Quantise the layer's values onto 0..255 over its range (solids at the clear end) and upload."""
        from vtkmodules.util.numpy_support import numpy_to_vtk

        arr, mask = lay.extra["data"], lay.extra["mask"]
        span = lay.vmax - lay.vmin if lay.vmax > lay.vmin else 1.0
        t = np.clip((arr - lay.vmin) / span, 0.0, 1.0)
        q = np.rint(np.nan_to_num(t, nan=0.0) * 255.0).astype(np.uint8)
        if mask is not None:
            q[mask > 0.5] = {"both": 128, "low": 255}.get(lay.extra["mode"], 0)
        img = lay.extra["image"]
        nz, ny, nx = q.shape
        origin, spacing = lay.extra["origin"], lay.extra["spacing"]
        if tuple(img.GetDimensions()) != (nx, ny, nz):
            img.SetDimensions(nx, ny, nz)
        img.SetSpacing(*spacing)
        img.SetOrigin(*(origin[i] + 0.5 * spacing[i] for i in range(3)))
        flat = np.ascontiguousarray(q).reshape(-1)
        va = numpy_to_vtk(flat, deep=False, array_type=3)
        va.SetName(lay.extra["field"])
        lay.extra["buffer"] = flat
        img.GetPointData().SetScalars(va)
        img.Modified()

    def _scalar_volume_transfer(self, lay: _Layer) -> None:
        """Colour from the 2D map's LUT; opacity by value (see :meth:`add_scalar_volume`)."""
        from .colors import get_lut
        lut = get_lut(lay.cmap)
        ctf, otf = lay.extra["ctf"], lay.extra["otf"]
        ctf.RemoveAllPoints()
        n = lut.shape[0]
        for i in range(0, n, 8):
            ctf.AddRGBPoint(255.0 * i / (n - 1), *(lut[i] / 255.0))
        ctf.AddRGBPoint(255.0, *(lut[-1] / 255.0))
        a = max(0.0, min(1.0, SCALAR_VOLUME_OPACITY * lay.extra["opacity"]))
        floor = min(max(lay.extra["floor"], 0.0), 0.95)
        otf.RemoveAllPoints()
        for t in np.linspace(0.0, 1.0, 33):
            mode = lay.extra["mode"]
            x = abs(2.0 * t - 1.0) if mode == "both" else (1.0 - t) if mode == "low" else t
            share = 0.0 if x <= floor else ((x - floor) / (1.0 - floor)) ** SCALAR_VOLUME_GAMMA
            otf.AddPoint(255.0 * t, a * share)

    def update_scalar_volume(self, handle: str, volume, *, rescale: bool = False) -> None:
        """Replace a scalar volume's data in place (same actor; the range stays unless ``rescale``)."""
        lay = self._layers.get(handle)
        if lay is None or lay.kind != "volume":
            raise KeyError(f"no scalar volume layer {handle!r}")
        origin, spacing, fields, mask = full_volume(volume)
        arr = volume_field(fields, lay.extra["field"]).astype(np.float32)
        fluid = arr[np.isfinite(arr) & (mask <= 0.5)] if mask is not None else arr[np.isfinite(arr)]
        lay.values = fluid[:: max(1, fluid.size // 200_000)] if fluid.size else fluid
        if rescale:
            lay.vmin, lay.vmax = self._resolve_range(fluid if fluid.size else arr, None, lay.field, lay.cmap)
        nz, ny, nx = arr.shape
        lay.bounds = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
                      origin[2], origin[2] + nz * spacing[2])
        lay.extra.update(data=arr, mask=mask, origin=origin, spacing=spacing, shape=arr.shape)
        if self._rw is not None and lay.actors:
            self._scalar_volume_image(lay)
            if rescale:
                self._scalar_volume_transfer(lay)
            lay.extra["mapper"].SetSampleDistance(0.5 * min(spacing))
            lay.extra["prop"].SetScalarOpacityUnitDistance(float(min(spacing)))
        self._touch()

    def add_particles(self, points, *, colors=None, color=None, size: float = PARTICLE_PX, opacity: float = 1.0,
                      trails=None, trail_opacity: float = TRAIL_OPACITY, mirror_y: bool = False) -> str:
        """Tracer particles as round points, optionally with their streak trails.

        ``points`` (N, 3) metres; ``colors`` (N, 3) uint8 per particle (an
        emitter's colour, ``LiveFrame.particle_colors``) or one ``color`` for
        all (default: the theme's streak colour, near-black on light and
        near-white on dark backgrounds). ``trails`` (N, K, 3), oldest first,
        NaN before release (``viz.EmitterParticles.trails``) draws each
        particle's recent path as a thin line in its colour: a continuous
        emitter's trails are its streaklines. ``size`` is in pixels (points
        do not grow when the camera zooms: they are markers). ``mirror_y``
        adds the y -> -y image of every particle for a symmetry-plane run.
        Update in place with :meth:`update_particles`.
        """
        lay = _Layer("particles", [], (0.0,) * 6)
        lay.extra.update(size=float(size), opacity=float(opacity), color=color, trail_opacity=float(trail_opacity),
                         mirror_y=bool(mirror_y))
        if self._rw is not None:
            from vtkmodules.vtkCommonDataModel import vtkPolyData
            from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
            for name in ("points", "trails"):
                pd = vtkPolyData()
                m = vtkPolyDataMapper()
                m.SetInputData(pd)
                a = vtkActor()
                a.SetMapper(m)
                p = a.GetProperty()
                p.LightingOff()
                if name == "points":
                    p.SetPointSize(float(size))
                    if hasattr(p, "SetRenderPointsAsSpheres"):
                        p.SetRenderPointsAsSpheres(True)
                    p.SetOpacity(float(opacity))
                else:
                    p.SetLineWidth(1.2)
                    p.SetOpacity(float(trail_opacity))
                lay.extra[f"{name}_pd"] = pd
                lay.actors.append(a)
        self._set_particles(lay, points, colors, trails)
        return self._add_layer("particles", lay)

    def _set_particles(self, lay: _Layer, points, colors, trails) -> None:
        pts = np.asarray(points if points is not None else np.zeros((0, 3)), dtype=np.float64).reshape(-1, 3)
        cols = None if colors is None else np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[:, :3]
        if cols is not None and len(cols) != len(pts):
            raise ValueError(f"{len(cols)} colours for {len(pts)} particles")
        tr = None if trails is None else np.asarray(trails, dtype=np.float64)
        if tr is not None and (tr.ndim != 3 or tr.shape[0] != len(pts) or tr.shape[-1] != 3):
            tr = None
        if lay.extra["mirror_y"] and len(pts):
            pts = np.vstack([pts, pts * (1.0, -1.0, 1.0)])
            cols = None if cols is None else np.vstack([cols, cols])
            tr = None if tr is None else np.concatenate([tr, tr * (1.0, -1.0, 1.0)])
        fin = np.isfinite(pts).all(axis=1)
        pts, cols = pts[fin], (cols[fin] if cols is not None else None)
        tr = tr[fin] if tr is not None else None
        lay.extra.update(n=len(pts), positions=pts, colors=cols, trails=tr)
        lay.bounds = _bounds_of(pts) if len(pts) else (0.0,) * 6
        if self._rw is None:
            return
        rgb = _rgb(lay.extra["color"]) if lay.extra["color"] is not None else _streak_rgb(self.theme)
        a_pts, a_tr = lay.actors
        _fill_points_polydata(lay.extra["points_pd"], pts, cols)
        _fill_trails_polydata(lay.extra["trails_pd"], tr, cols)
        for a in (a_pts, a_tr):
            m = a.GetMapper()
            if cols is not None:
                m.SetColorModeToDirectScalars()
                m.ScalarVisibilityOn()
            else:
                m.ScalarVisibilityOff()
                a.GetProperty().SetColor(*rgb)
        a_tr.SetVisibility(bool(lay.visible and tr is not None and tr.shape[1] > 1))

    def update_particles(self, handle: str, points, *, colors=None, trails=None) -> None:
        """New particle positions (and colours / trails) for the same layer: no actor is rebuilt."""
        lay = self._layers.get(handle)
        if lay is None or lay.kind != "particles":
            raise KeyError(f"no particle layer {handle!r}")
        self._set_particles(lay, points, colors, trails)
        self._touch()

    def add_glyph(self, mesh, *, color, opacity: float = 1.0, name: str = "") -> str:
        """A device marker (a fan disc, a heater box, an emitter bead): a lit mesh in one colour.

        Unlike :meth:`add_mesh` it is not a body: it takes no part in framing
        (:meth:`focus_bounds`), edges or the near-wall trimming of vortex
        iso-surfaces, so a fan disc never moves the camera or cuts a vortex.
        ``mesh`` is (vertices, faces); see :mod:`aerolume.viz3d.placed`.
        """
        v, f = mesh
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        lay = _Layer("glyph", [], _bounds_of(v))
        lay.extra.update(name=name, n_faces=len(f), color=color, opacity=float(opacity))
        if self._rw is not None and len(f):
            from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
            m = vtkPolyDataMapper()
            m.SetInputData(polydata_to_vtk(v, f))
            m.ScalarVisibilityOff()
            a = vtkActor()
            a.SetMapper(m)
            self._surface_property(a.GetProperty(), _rgb(color), float(opacity), False)
            lay.actors.append(a)
        return self._add_layer("glyph", lay)

    def layer_info(self, handle: str) -> dict:
        """Facts about a layer for tests and captions: its kind, bounds and kind-specific counts
        (``n`` particles, ``n_faces`` of a glyph, ``field`` / range of a scalar volume)."""
        lay = self._layers.get(handle)
        if lay is None:
            return {}
        out = {"kind": lay.kind, "bounds": tuple(lay.bounds), "visible": lay.visible}
        for k in ("n", "n_faces", "field", "mode", "name", "opacity", "color", "gain"):
            if k in lay.extra:
                out[k] = lay.extra[k]
        if lay.cmap is not None:
            out.update(cmap=lay.cmap, vmin=lay.vmin, vmax=lay.vmax)
        return out

    # ------------------------------------------------------------------
    # annotations
    # ------------------------------------------------------------------
    def add_colorbar(self, title: str | None = None, units: str | None = None, layer: str | None = None, *,
                     second: bool = False) -> None:
        """Show a vertical colour bar for ``layer`` (default: the newest colour-mapped layer).

        ``title`` / ``units`` default to the field's label and SI unit.

        ``second=True`` sets a second bar, drawn left of the first, for another
        colour-mapped layer: a view with two colour maps at once (a temperature
        plume and streamlines coloured by speed) needs a key for each, or one of
        them is read off the other's scale. The second bar needs its ``layer``
        (it never falls back to "the newest layer", which is usually the first
        bar's), and it is left out while that layer is hidden or gone, or when it
        would repeat the first bar.
        """
        entry = {"layer": layer, "title": title, "units": units}
        if second:
            self._colorbar2 = entry if layer else None
        else:
            self._colorbar = entry
        self._touch()

    def remove_colorbar(self, *, second: bool = False) -> None:
        """Remove the colour bars (``second=True``: only the second one)."""
        self._colorbar2 = None
        if not second:
            self._colorbar = None
        self._touch()

    def colorbar_spec(self) -> ColorbarSpec | None:
        """The (first) colour bar that the next render draws (None without one)."""
        return self._spec_of(self._colorbar, fallback=True)

    def colorbar_specs(self) -> list[ColorbarSpec]:
        """Every colour bar the next render draws, the first (rightmost) one first."""
        first = self.colorbar_spec()
        out = [first] if first is not None else []
        second = self._spec_of(self._colorbar2, fallback=False, visible_only=True)
        if second is not None and (first is None or (second.cmap, second.vmin, second.vmax, second.title) !=
                                   (first.cmap, first.vmin, first.vmax, first.title)):
            out.append(second)
        return out

    def _spec_of(self, entry: dict | None, *, fallback: bool, visible_only: bool = False) -> ColorbarSpec | None:
        """The bar of one :meth:`add_colorbar` entry: its layer's map, range, title, units and clipped ends."""
        if entry is None:
            return None
        h = entry.get("layer")
        lay = self._layers.get(h) if h else None
        if lay is not None and visible_only and not lay.visible:
            return None
        if lay is None and fallback:
            for cand in reversed(list(self._layers.values())):
                if cand.cmap is not None and cand.visible:
                    lay = cand
                    break
        if lay is None or lay.cmap is None:
            return None
        info = field_info(lay.field)
        title = entry.get("title") or info.label
        units = entry.get("units")
        units = info.unit if units is None else units
        ext = (False, False)
        if lay.values is not None and np.size(lay.values):
            v = np.asarray(lay.values)
            v = v[np.isfinite(v)]
            if v.size:
                tol = 1e-9 * max(abs(lay.vmax - lay.vmin), 1e-30)
                ext = (bool(v.min() < lay.vmin - tol), bool(v.max() > lay.vmax + tol))
        return ColorbarSpec(lay.cmap, float(lay.vmin), float(lay.vmax), title, units, ext)

    def add_axes(self, on: bool = True) -> None:
        """Show (or hide) the x / y / z triad in the bottom-left corner."""
        self._axes = bool(on)
        self._touch()

    def set_caption(self, text: str) -> None:
        """One line of text in the top-left corner ('' removes it)."""
        self._caption = str(text or "")
        self._touch()

    def caption(self) -> str:
        """The caption text set with :meth:`set_caption` (drawn or not)."""
        return self._caption

    def set_annotations(self, *, colorbar: bool | None = None, caption: bool | None = None) -> None:
        """Draw (True) or leave out (False) the colour bar and the caption; None keeps the setting.

        The bar and caption stay defined (:meth:`colorbar_spec` still answers),
        they are only not drawn. Why: in a split view the slice next to the 3D
        pane already shows the same legend and the same time label, and two
        copies side by side read as clutter. Without the bar the camera frames
        the model over the full width (nothing is reserved for it).
        """
        changed = False
        if colorbar is not None and bool(colorbar) != self._show_colorbar:
            self._show_colorbar, changed = bool(colorbar), True
        if caption is not None and bool(caption) != self._show_caption:
            self._show_caption, changed = bool(caption), True
        if changed:
            self._touch()

    def annotations(self) -> dict[str, bool]:
        """{"colorbar": drawn?, "caption": drawn?} as set by :meth:`set_annotations`."""
        return {"colorbar": self._show_colorbar, "caption": self._show_caption}

    def set_top_inset(self, pixels: float) -> None:
        """Keep the caption this many (1x) pixels below the top, e.g. under a floating toolbar."""
        self._top_inset = max(0.0, float(pixels))
        self._touch()

    def set_theme(self, theme: str | dict | SceneTheme, *, gradient: bool | None = None) -> None:
        """Switch colours (background, text, ground, body grey) without rebuilding data."""
        new = scene_theme(theme, gradient=self.theme.gradient if gradient is None else gradient)
        if new == self.theme:
            return
        self.theme = new
        if self._rw is not None:
            self._apply_background()
            for lay in self._layers.values():
                if lay.kind == "mesh":
                    for a in lay.actors:
                        a.GetProperty().SetEdgeColor(*new.edge)
                        if lay.themed_color:
                            a.GetProperty().SetColor(*new.body)
                elif lay.kind == "ground":
                    for a in lay.actors:
                        self._ren.RemoveViewProp(a)
                    lay.actors = self._ground_actors(lay)
                    for a in lay.actors:
                        a.SetVisibility(lay.visible)
                        self._ren.AddViewProp(a)
                elif lay.kind == "slice" and len(lay.actors) > 1:
                    lay.actors[1].GetProperty().SetColor(*new.border)
                elif lay.kind == "outline":
                    for a in lay.actors:
                        a.GetProperty().SetColor(*new.border)
                elif lay.kind == "streamlines" and lay.themed_color:
                    for a in lay.actors:
                        a.GetProperty().SetColor(*new.text2)
                elif lay.kind == "streaks" and lay.themed_color:
                    for a in lay.actors:
                        a.GetProperty().SetColor(*_streak_rgb(new))
                elif lay.kind == "smoke" and lay.actors:
                    self._smoke_transfer(lay)
                elif lay.kind == "particles" and lay.extra.get("color") is None and lay.extra.get("colors") is None:
                    for a in lay.actors:
                        a.GetProperty().SetColor(*_streak_rgb(new))
                elif lay.kind == "rake":
                    for a in lay.actors:
                        self._ren.RemoveViewProp(a)
                    lay.actors = self._rake_actors(lay)
                    for a in lay.actors:
                        a.SetVisibility(lay.visible)
                        self._ren.AddViewProp(a)
        self._touch()

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def _mesh_bounds(self):
        return _union([lay.bounds for lay in self._layers.values()
                       if lay.kind == "mesh" and lay.visible and lay.extra.get("n_faces", 1) > 0])

    def set_focus(self, bounds=None) -> None:
        """Override what 'fit' frames with a box (x0, x1, y0, y1, z0, z1); None restores the default.

        Pages showing flow (streamlines, a wake slice) frame the body plus its
        near wake this way, as the 2D views do.
        """
        self._focus = None if bounds is None else tuple(float(b) for b in bounds)
        self._touch()

    def focus_bounds(self) -> tuple[float, float, float, float, float, float]:
        """What 'fit' frames: the focus override, else the visible meshes, else every layer but the ground."""
        if self._focus is not None:
            return self._focus
        b = self._mesh_bounds()
        if b is None:
            b = _union([lay.bounds for lay in self._layers.values() if lay.visible and lay.kind != "ground"])
        if b is None:
            b = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
        return tuple(float(v) for v in b)  # type: ignore[return-value]

    def focus_points(self) -> np.ndarray | None:
        """Points 'fit' frames: samples of the visible meshes (None: use the focus box)."""
        if self._focus is not None:
            return None
        pts = [lay.extra["fit_points"] for lay in self._layers.values()
               if lay.kind == "mesh" and lay.visible and lay.extra.get("n_faces", 0) > 0]
        return np.concatenate(pts) if pts else None

    @property
    def camera(self) -> cam.CameraState:
        """The camera the next render uses (a pending auto-fit is resolved first)."""
        self._ensure_camera()
        return self._camera

    def set_camera_state(self, state: cam.CameraState) -> None:
        """Restore a saved camera (stops auto-fit)."""
        self._camera = state
        self._auto_fit = False
        self._touch()

    def set_camera(self, preset: str = "iso", fit: bool = True) -> None:
        """Look from a preset direction ('iso', 'front', 'back', 'side', 'top', 'bottom').

        With ``fit`` the view frames the bodies and keeps re-framing when the
        image aspect changes, until the user moves the camera.
        """
        d, up = cam.preset(preset)
        self._fit_dir = (d, up)
        if fit:
            self._auto_fit = True
        else:
            dist = self._camera.distance
            fp = np.asarray(self._camera.focal_point)
            self._camera = cam.CameraState(position=tuple(fp + d * dist), focal_point=tuple(fp),
                                           view_up=tuple(up), view_angle=self._camera.view_angle,
                                           parallel=self._camera.parallel,
                                           parallel_scale=self._camera.parallel_scale)
            self._auto_fit = False
        self._touch()

    def fit(self) -> None:
        """Frame the bodies, keeping the current view direction (double-click in the widget)."""
        self._fit_dir = None
        self._auto_fit = True
        self._touch()

    def set_parallel(self, on: bool) -> None:
        """Parallel (orthographic) projection on / off; keeps the framing."""
        if bool(on) == self._camera.parallel:
            return
        hh = self._camera.half_height()
        self._camera = cam.CameraState(self._camera.position, self._camera.focal_point, self._camera.view_up,
                                       self._camera.view_angle, bool(on), hh)
        if not on:
            d = hh / math.tan(math.radians(self._camera.view_angle) / 2)
            _, u, f = self._camera.frame()
            fp = np.asarray(self._camera.focal_point)
            self._camera = cam.CameraState(tuple(fp - f * d), tuple(fp), tuple(u), self._camera.view_angle,
                                           False, hh)
        self._touch()

    def _main_size(self) -> tuple[float, float]:
        w, h = self._size if self._size[0] > 0 else (800, 600)
        return max(1.0, float(w - self._strip_px(w))), float(max(h, 1))

    def _drawn_colorbars(self) -> list[ColorbarSpec]:
        """The colour bars the render draws (:meth:`colorbar_specs`), without the first one while
        :meth:`set_annotations` hid it: a companion view draws that one, but a second colour map in
        this view still needs its own key."""
        specs = self.colorbar_specs()
        if not self._show_colorbar and specs and specs[0] == self.colorbar_spec():
            specs = specs[1:]
        return specs

    def _strip_px(self, w: int) -> int:
        """Pixels reserved at the right for the colour bars (at most half the image)."""
        total = sum(colorbar_strip_width(s, self._scale) for s in self._drawn_colorbars())
        return round(min(total, w / 2)) if total else 0

    def _ensure_camera(self) -> None:
        if self._auto_fit:
            w, h = self._main_size()
            d, up = self._fit_dir if self._fit_dir is not None else (None, None)
            self._camera = cam.fit(self._camera, self.focus_bounds(), w / h, direction=d, view_up=up,
                                   points=self.focus_points())

    def _user_moves(self) -> None:
        self._ensure_camera()       # materialise the fitted view before changing it
        self._auto_fit = False

    def orbit(self, dx: float, dy: float) -> None:
        """Turntable orbit: ``dx`` degrees of azimuth, ``dy`` degrees of elevation."""
        self._user_moves()
        self._camera = cam.orbit(self._camera, dx, dy)
        self._touch()

    def pan(self, dx: float, dy: float) -> None:
        """Pan by (dx, dy) pixels of the last rendered image (y down)."""
        self._user_moves()
        self._camera = cam.pan(self._camera, dx, dy, self._main_size()[1])
        self._touch()

    def zoom(self, factor: float, at: tuple[float, float] | None = None) -> None:
        """Zoom by ``factor`` (> 1 closer), keeping pixel ``at`` (x, y down) fixed if given."""
        self._user_moves()
        self._camera = cam.zoom(self._camera, factor, at=at, size=self._main_size() if at else None)
        self._touch()

    def dolly(self, factor: float) -> None:
        """Move the camera along its axis towards the focal point (``factor`` > 1 closer)."""
        self.zoom(factor)

    def project(self, points) -> np.ndarray:
        """Pixel coordinates (x right, y down) of world ``points`` in the last rendered image.

        Accounts for the colour-bar shift of the projection centre; NaN for
        points behind the camera. Used for labels, picking and tests.
        """
        self._ensure_camera()
        w, h = self._main_size()
        return cam.project(self._camera, points, w, h)

    def view_size(self) -> tuple[int, int]:
        """(width, height) of the 3D part of the last render (the colour-bar strip excluded)."""
        w, h = self._main_size()
        return int(w), int(h)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(self, width: int, height: int, *, scale: float = 1.0) -> np.ndarray:
        """Render to an (height, width, 3) uint8 RGB image, row 0 at the top.

        ``scale`` is the device pixel ratio: overlay text, ticks and line
        widths grow with it, so a 2x render for a high-DPI screen or a report
        looks like the 1x one at double resolution.
        """
        w, h = max(1, int(width)), max(1, int(height))
        self._size, self._scale = (w, h), float(scale)
        if self._rw is None or self._closed:
            bg = np.round(np.asarray(self.theme.background) * 255).astype(np.uint8)
            return np.broadcast_to(bg, (h, w, 3)).copy()
        t0 = time.perf_counter()
        specs = self._drawn_colorbars()
        strip = self._strip_px(w)
        rw = self._rw
        if tuple(rw.GetSize()) != (w, h):
            rw.SetSize(w, h)
        # The 3D view keeps the full width (the floor runs on under the colour bar, no hard
        # edge), but its projection centre moves to the middle of the area left of the bar,
        # so pixels there behave exactly like a (w - strip) x h viewport: fit, pan and
        # zoom-at-cursor all work in that area's coordinates.
        self._window_cx = strip / w          # VTK: +x moves the projection centre left
        self._ensure_camera()
        self._apply_camera()
        line_w = max(1.0, float(scale))
        for lay in self._layers.values():
            if lay.kind in ("mesh",):
                for a in lay.actors:
                    a.GetProperty().SetLineWidth(line_w)
        self._overlay.set_colorbars(specs, w, h, scale, self.theme)
        self._overlay.set_triad(self._axes, self._camera.frame(), scale, self.theme)
        self._overlay.set_caption(self._caption if self._show_caption else "", h, scale, self.theme,
                                  top=self._top_inset)
        rw.Render()
        img = self._read_pixels(w, h)
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return img

    def _apply_camera(self) -> None:
        c = self._camera
        vc = self._ren.GetActiveCamera()
        vc.SetPosition(*c.position)
        vc.SetFocalPoint(*c.focal_point)
        vc.SetViewUp(*c.view_up)
        vc.SetWindowCenter(self._window_cx, 0.0)
        vc.SetViewAngle(c.view_angle)
        vc.SetParallelProjection(bool(c.parallel))
        vc.SetParallelScale(c.parallel_scale)
        self._ren.ResetCameraClippingRange()
        near, far = vc.GetClippingRange()
        vc.SetClippingRange(max(near, far * 1e-4), far)   # keep depth precision for close-ups

    def _read_pixels(self, w: int, h: int) -> np.ndarray:
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkCommonCore import vtkUnsignedCharArray

        arr = vtkUnsignedCharArray()
        self._rw.GetPixelData(0, 0, w - 1, h - 1, 0, arr, 0)
        a = vtk_to_numpy(arr)
        if a.size != w * h * 3:            # pragma: no cover - driver quirk: fall back to the filter
            from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
            f = vtkWindowToImageFilter()
            f.SetInput(self._rw)
            f.SetInputBufferTypeToRGB()
            f.ReadFrontBufferOff()
            f.Update()
            a = vtk_to_numpy(f.GetOutput().GetPointData().GetScalars())
        return np.ascontiguousarray(a.reshape(h, w, 3)[::-1])

    def save_png(self, path, width: int = 1600, height: int = 900, *, scale: float = 1.0) -> np.ndarray:
        """Render and write a PNG (Pillow); returns the image."""
        from PIL import Image
        img = self.render(width, height, scale=scale)
        Image.fromarray(img).save(str(path))
        return img

    def close(self) -> None:
        """Release the OpenGL context (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._rw is not None:
            try:
                self._ren.RemoveAllViewProps()
                self._ovl.RemoveAllViewProps()
                self._rw.Finalize()
            except Exception:
                log.debug("render window finalize failed", exc_info=True)
            self._rw = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        # no contextlib here: at interpreter shutdown module globals may already be None, and
        # ``contextlib.suppress`` would then raise a TypeError from inside the finaliser
        try:
            self.close()
        except BaseException:  # noqa: BLE001, S110 - a finaliser must never raise
            pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _fluid_sample(volume, name: str, n: int = 400_000) -> np.ndarray | None:
    """A strided sample of a volume field in fluid cells (solid fraction < 1/2), for colour
    ranges: the range then follows the field in the whole volume (a stable scale that does
    not depend on where the lines happen to go) at a fraction of the cost of the full array."""
    f = volume.fields
    if name in f:
        a = np.asarray(f[name])
        step = max(1, a.size // n)
        vals = a.reshape(-1)[::step].astype(np.float32)
    elif name == "speed" and all(k in f for k in ("u", "v", "w")):
        step = max(1, f["u"].size // n)
        u, v, w = (np.asarray(f[k]).reshape(-1)[::step].astype(np.float32) for k in ("u", "v", "w"))
        vals = np.sqrt(u * u + v * v + w * w)
    else:
        return None
    if volume.mask is not None:
        vals = vals[np.asarray(volume.mask).reshape(-1)[::step] < 0.5]
    return vals


def _streak_rgb(theme) -> tuple[float, float, float]:
    """Colour of skin-friction lines: dark on the body in both themes. The lines lie on the
    body, which is light grey in both themes, so the theme's text colour (light in the dark
    theme) would vanish on it; the oil-flow picture is dark lines on a pale model."""
    if not theme.dark:
        return tuple(theme.edge)
    return tuple(0.85 * e + 0.15 * b for e, b in zip(theme.edge, theme.body, strict=True))


_WALL_CACHE: dict[tuple, np.ndarray] = {}


def _wall_cleared(mask) -> np.ndarray:
    """Solid voxels and those within :data:`SMOKE_WALL_CLEAR_CELLS` cells of one (26-neighbourhood), as a
    boolean array: the smoke drawn there is cleared, so the body is not coated by its boundary layer's
    dye. Cached per mask (a live run sends the same mask with every frame; the dilation of a 1.9 M-voxel
    volume costs tens of milliseconds)."""
    m = np.asarray(mask)
    solid = m > 0.5
    n = int(SMOKE_WALL_CLEAR_CELLS)
    if n <= 0 or not solid.any():
        return solid
    key = (id(mask), solid.shape, int(np.count_nonzero(solid)))
    hit = _WALL_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        out = binary_dilation(solid, structure=generate_binary_structure(solid.ndim, solid.ndim), iterations=n)
    except ImportError:                                   # pragma: no cover - scipy is a dependency
        out = solid
    _WALL_CACHE.clear()                                   # one mask at a time: the view on screen
    _WALL_CACHE[key] = out
    return out


def _hsv(rgb) -> tuple[float, float, float]:
    import colorsys
    return colorsys.rgb_to_hsv(*(min(max(float(c), 0.0), 1.0) for c in rgb))


def smoke_display_rgb(rgb) -> tuple[float, float, float]:
    """The colour a smoke channel is drawn in: a coloured channel (HSV saturation at least
    :data:`SMOKE_CHROMA_MIN`) at full saturation, so thin orange and blue smoke stay orange and blue
    where they overlap; a white or grey one unchanged. The legend shows the same colour."""
    import colorsys
    h, s, v = _hsv(rgb)
    if s < SMOKE_CHROMA_MIN:
        return tuple(float(c) for c in rgb)  # type: ignore[return-value]
    return tuple(float(c) for c in colorsys.hsv_to_rgb(h, 1.0, v))  # type: ignore[return-value]


def smoke_core_colour(rgb, dark: bool) -> tuple[float, float, float]:
    """The colour a smoke channel's densest voxels are drawn in.

    A coloured channel keeps its hue at full saturation (darkened by :data:`SMOKE_CHROMA_DARKEN` on the
    light theme, at full value on the dark one): towards black or white, the car's orange and blue
    channels became one grey-brown haze. A neutral one moves :data:`SMOKE_CORE_SHADE` of the way towards
    black (light) or white (dark); the shift shrinks for a colour already near its target (the light
    theme's smoke grey #2b3038 taken the full way to black drew the white rake's cores as soot on the car
    roof)."""
    import colorsys
    h, s, v = _hsv(rgb)
    if s >= SMOKE_CHROMA_MIN:
        v2 = v * (1.0 - SMOKE_CHROMA_DARKEN) if not dark else max(v, 0.95)
        return tuple(float(c) for c in colorsys.hsv_to_rgb(h, 1.0, v2))  # type: ignore[return-value]
    target = 1.0 if dark else 0.0
    lum = 0.2126 * float(rgb[0]) + 0.7152 * float(rgb[1]) + 0.0722 * float(rgb[2])
    room = (1.0 - lum) if dark else lum                    # how far the colour is from black / white
    shade = SMOKE_CORE_SHADE * min(1.0, max(0.25, room / 0.5))
    return tuple(float(c) + (target - float(c)) * shade for c in rgb)  # type: ignore[return-value]


def smoke_gain(values, *, threshold: float = SMOKE_THRESHOLD) -> tuple[float, float]:
    """(concentration gain, opacity factor) that make one smoke channel visible.

    Measured on the stored car run (196 x 128 x 76): the two rakes hold smoke in
    4.3 % of the voxels with a 90th percentile of 0.26-0.37, the point emitter
    in 0.5 % with 0.15, and 90 % of its voxels sat where the convex opacity
    curve is nearly clear, so a blue stream that was really there drew nothing.
    The gain lifts the channel's 90th percentile to :data:`SMOKE_GAIN_TARGET`
    (never lowers it, at most :data:`SMOKE_GAIN_MAX`), and a channel in fewer
    than :data:`SMOKE_SPARSE_SHARE` of the voxels also gets up to
    :data:`SMOKE_SPARSE_BOOST` times the per-voxel opacity: a stream two cells
    wide crosses two voxels where a rake's sheet crosses twenty. Both are
    per-channel display scalings, not data: the colour bar and every number
    still read the stored concentration.
    """
    a = np.asarray(values)
    a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32, copy=False)
    smoky = a[a > threshold]
    if smoky.size == 0:
        return 1.0, 1.0
    p90 = float(np.percentile(smoky, 90))
    gain = float(np.clip(SMOKE_GAIN_TARGET / max(p90, 1e-6), 1.0, SMOKE_GAIN_MAX))
    share = smoky.size / max(a.size, 1)
    boost = float(np.clip(math.sqrt(SMOKE_SPARSE_SHARE / max(share, 1e-9)), 1.0, SMOKE_SPARSE_BOOST))
    return gain, boost


def _opacity_curve(curve) -> tuple[tuple[float, float], ...]:
    """A smoke opacity curve as sorted (concentration, share) pairs (None: the default).

    Checked here, so a script's typo fails with a readable message instead of a
    VTK warning and an invisible smoke."""
    if curve is None:
        return SMOKE_OPACITY_CURVE
    try:
        pts = sorted((float(c), float(f)) for c, f in curve)
    except (TypeError, ValueError):
        raise ValueError("opacity_curve must be (concentration, opacity share) pairs, "
                         "e.g. ((0.1, 0.05), (0.5, 0.4), (1.0, 1.0))") from None
    if not pts or any(not (0.0 < c <= 1.0) or not (0.0 <= f <= 1.0) for c, f in pts):
        raise ValueError("opacity_curve needs at least one pair with concentrations in (0, 1] "
                         "and opacity shares in [0, 1]")
    return tuple(pts)


def _seed_spacing(seeds, sample: int = 600) -> float:
    """Median distance from a seed to its nearest other seed (0 for fewer than two seeds).

    Brute force from at most ``sample`` query seeds (strided) to all seeds: a few
    hundred seeds cost well under a millisecond, and it needs no KD-tree (scipy's
    recurses badly when a solver left flush-denormal mode on in this process)."""
    if seeds is None:
        return 0.0
    p = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)[:20_000]
    if len(p) < 2:
        return 0.0
    idx = np.arange(0, len(p), max(1, len(p) // sample))
    d2 = ((p[idx, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    d2[np.arange(len(idx)), idx] = np.inf
    nn = np.sqrt(d2.min(axis=1))
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if len(nn) else 0.0


#: most points spread over one triangle for the wall distance (a 30-cell triangle edge)
_MAX_TRI_SPLIT = 90


def _triangle_samples(surfaces, step: float, box=None) -> np.ndarray:
    """Points spread over the triangles of ``surfaces`` [(vertices, faces), ...] at most ``step`` apart.

    Each triangle gets a barycentric grid with ``k = ceil(longest edge / step)``
    divisions (vertices, edges and interior). Triangles outside ``box`` (x0, x1,
    y0, y1, z0, z1) are skipped: only the walls near the iso-surface matter.
    """
    out = []
    for verts, faces in surfaces:
        v = np.asarray(verts, np.float64)
        f = np.asarray(faces, np.int64)
        if not len(f):
            continue
        tri = v[f]                                            # (m, 3, 3)
        if box is not None:
            lo, hi = tri.min(axis=1), tri.max(axis=1)
            keep = np.all((hi >= np.array(box[0::2])) & (lo <= np.array(box[1::2])), axis=1)
            tri = tri[keep]
            if not len(tri):
                continue
        edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
        k = np.clip(np.ceil(edge / max(step, 1e-12)), 1, _MAX_TRI_SPLIT).astype(int)
        for kk in np.unique(k):
            i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
            sel = (i + j) <= kk
            b = np.stack([(kk - i[sel] - j[sel]) / kk, i[sel] / kk, j[sel] / kk], axis=1)     # (p, 3)
            out.append(np.einsum("pb,tbx->tpx", b, tri[k == kk]).reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


def _distance_to_triangles(surfaces, points: np.ndarray, *, step: float, limit: float) -> np.ndarray | None:
    """Distance (metres) from each point to the nearest wall sample; ``limit`` and beyond are reported as ``2 * limit``.

    Only walls within ``limit`` of the points' bounding box are sampled, and the
    k-d tree query stops at ``limit``: the far wake costs nothing.
    """
    from scipy.spatial import cKDTree

    if not len(points):
        return np.zeros(0)
    lo, hi = points.min(axis=0) - limit, points.max(axis=0) + limit
    box = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    samples = _triangle_samples(surfaces, step, box)
    if not len(samples):
        return None
    d, _ = cKDTree(samples).query(points, k=1, distance_upper_bound=limit, workers=-1)
    return np.where(np.isfinite(d), d, 2.0 * limit)


def _distance_from_mask(mask: np.ndarray, origin, spacing, points: np.ndarray) -> np.ndarray:
    """Distance (metres) from each point to the solid of a cell mask, smoothed so its level sets are round.

    The Euclidean distance transform is measured between cell centres; the wall
    lies about half a cell beyond the last solid centre, hence the shift. A
    Gaussian of one cell removes the voxel facets of its level sets.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

    sx, sy, sz = (float(s) for s in spacing)
    solid = np.asarray(mask) > 0.5
    d = distance_transform_edt(~solid, sampling=(sz, sy, sx)).astype(np.float32)
    d = gaussian_filter(d, sigma=1.0) - 0.5 * min(sx, sy, sz)
    idx = [(points[:, 2] - origin[2]) / sz - 0.5, (points[:, 1] - origin[1]) / sy - 0.5,
           (points[:, 0] - origin[0]) / sx - 0.5]
    return map_coordinates(d, idx, order=1, mode="nearest")


def _rgb(c) -> tuple[float, float, float]:
    from .colors import hex_to_rgb
    return hex_to_rgb(c)


def _nice_step(target: float) -> float:
    if not target > 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(target))
    for m in (1, 2, 5, 10):
        if m * mag >= target:
            return m * mag
    return 10 * mag


def _fill_points_polydata(pd, pts: np.ndarray, cols: np.ndarray | None) -> None:
    """Points (one vertex cell each) with optional direct RGB colours, into an existing vtkPolyData."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    n = len(pts)
    p = vtkPoints()
    p.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64) if n else np.zeros((0, 3)), deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(n + 1, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(n, dtype=np.int64), deep=True))
    pd.Initialize()
    pd.SetPoints(p)
    pd.SetVerts(cells)
    if cols is not None and n:
        c = numpy_to_vtk(np.ascontiguousarray(cols, dtype=np.uint8), deep=True)
        c.SetName("rgb")
        pd.GetPointData().SetScalars(c)
    pd.Modified()


def _fill_trails_polydata(pd, trails: np.ndarray | None, cols: np.ndarray | None) -> None:
    """One polyline per particle trail (NaN samples dropped; trails shorter than 2 points skipped)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    pd.Initialize()
    if trails is None or trails.ndim != 3 or trails.shape[1] < 2:
        pd.Modified()
        return
    ok = np.isfinite(trails).all(axis=2)                     # (N, K)
    counts = ok.sum(axis=1)
    keep = counts >= 2
    pts = trails[keep][ok[keep]]                              # oldest first, per particle in order
    lens = counts[keep]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    p = vtkPoints()
    p.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64) if len(pts) else np.zeros((0, 3)),
                           deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(int(offsets[-1]), dtype=np.int64), deep=True))
    pd.SetPoints(p)
    pd.SetLines(cells)
    if cols is not None and len(pts):
        c = numpy_to_vtk(np.ascontiguousarray(np.repeat(cols[keep], lens, axis=0), dtype=np.uint8), deep=True)
        c.SetName("rgb")
        pd.GetPointData().SetScalars(c)
    pd.Modified()


def _lines_actor(segments: np.ndarray, rgb, width: float):
    """An unlit actor drawing (K, 2, 3) line segments in one colour."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

    seg = np.ascontiguousarray(segments.reshape(-1, 3), dtype=np.float64)
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(seg, deep=True))
    k = len(seg) // 2
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 2 * k + 1, 2, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(2 * k, dtype=np.int64), deep=True))
    pd = vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(cells)
    m = vtkPolyDataMapper()
    m.SetInputData(pd)
    m.ScalarVisibilityOff()
    a = vtkActor()
    a.SetMapper(m)
    p = a.GetProperty()
    p.SetColor(*rgb)
    p.LightingOff()
    p.SetLineWidth(width)
    return a
