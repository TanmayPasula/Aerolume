"""Flow visualisation in the volume around a 3D body: smoke, streamlines, slice planes (DESIGN.md 7.13).

Users asked to see the flow *around* the body in 3D, not only colour on its
surface. This module is the GUI half of that: the Simulate page (live runs and
explore) and the Results page (stored runs) both draw the same layers through
it, with the same controls, so a stored result looks like the last frame of
the run that made it.

* :class:`RakeSpec` -- the seed rake upstream of the body: the emitter chosen
  in Setup ▸ Flow visualisation (a rake across the flow, a vertical rake, a
  grid of nozzles or a plane band; :func:`setup_flow_viz`) placed by the
  solver's own resolver, then moved by the user (position x, height, lateral
  span, number of lines). Streamlines start on it; it is handed to the solver
  as ``ViewSettings.seed_rake`` so the smoke nozzles sit in the same place.
* :class:`RakeEditor` -- the rake's rows in a property-grid group (metres).
* :class:`FlowToggles` -- the checkable tool buttons Smoke, Streamlines,
  Slice plane, Vortex cores (Q) and Surface Cp, for any :class:`ToolStrip`.
* :class:`FlowScene` -- builds and updates those layers in one
  :class:`~aerolume.app.widgets.view3d.View3D`: body (grey or by Cp), smoke
  (the solver's passive scalar ``dye`` as a volume), streamlines through the
  velocity from the rake, the slice plane, the Q-criterion iso-surface and the
  rake itself, plus the one colour bar. Placed physics (DESIGN.md 10) adds the
  thermal plume (the temperature as a colour-mapped translucent volume or one
  iso-surface, :func:`heat_display_range`), smoke in one colour per emitter
  channel, tracer particles with their streak trails, and the device glyphs
  (fans, heaters, emitters, jets; :mod:`aerolume.viz3d.placed`).

Design notes (the WHY):

* **Smoke is the solver's dye, never a picture made up from the mean flow.**
  It needs a volume with a ``dye`` field: :attr:`LiveFrame.volume` during a
  run, the stored instantaneous snapshot afterwards (:func:`smoke_volume_of`).
  A result without one says so (the toggle is disabled with the reason);
  streamlines of the mean flow are offered instead, under their own name.
* **Rendering degrades, it does not fail.** ``Scene.add_smoke`` (a GPU volume
  renderer) is used when the viewer has it; otherwise the dye is drawn as two
  translucent iso-surfaces of its concentration, which reads as smoke at a
  fraction of the cost. Every layer is built in its own ``try``; a layer that
  cannot be drawn records why (:attr:`FlowScene.errors`) and the others stay.
* **Cheap updates.** Each layer is rebuilt only when its own data or setting
  changes, and the time it took is kept (:attr:`FlowScene.cost_ms`), so a live
  page can space heavy rebuilds (streamline tracing) to keep the window at
  display rate.
* **One colour bar.** The slice plane owns it when shown (streamlines then use
  the slice's field, map and range, so one legend reads both); otherwise the
  streamlines; otherwise the surface Cp. Smoke is greyscale and has none.
"""
from __future__ import annotations

import inspect
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QToolButton, QWidget

from ...core.units import sig
from .common import PropertyGroup, ToolStrip, button, label
from .plots import lazy_import, tokens

log = logging.getLogger(__name__)

__all__ = [
    "AUTO_SEEDS",
    "DEFAULT_SHOWN_LIVE",
    "DEFAULT_SHOWN_RESULT",
    "DYE_CHANNELS",
    "FLOW_LAYERS",
    "HEAT_STYLES",
    "HEAT_THRESHOLD_SHARE",
    "RAKE_KIND_TEXT",
    "STREAMLINE_OPACITY_WITH_SMOKE",
    "FlowScene",
    "FlowToggles",
    "RakeEditor",
    "RakeSpec",
    "SmokeLegend",
    "add_smoke_layers",
    "bounds_of",
    "colour_word",
    "dye_colours_of",
    "flow_focus_box",
    "has_velocity",
    "heat_display_range",
    "heat_volume_of",
    "keep_whole",
    "layer_toggle_qss",
    "near_field_box",
    "plane_window",
    "rake_mesh",
    "setup_flow_viz",
    "smoke_channel_labels",
    "smoke_channels",
    "smoke_colour",
    "smoke_volume_of",
    "temperature_field_of",
    "volume_box",
]

#: Toggle keys in strip order: (key, button text, tooltip)
FLOW_LAYERS: tuple[tuple[str, str, str], ...] = (
    ("smoke", "Smoke", "Smoke released from the rake and carried by the flow (the solver's passive dye)"),
    ("streamlines", "Streamlines", "Streamlines through the 3D velocity, starting on the rake"),
    ("slice", "Slice plane", "The slice plane, coloured by the field chosen above"),
    ("q", "Vortex cores", "Iso-surface of the Q-criterion: where rotation dominates strain"),
    ("heat", "Thermal plume", ("Where the fluid is warmer than the ambient air: the temperature as a translucent "
                               "volume (or an iso-surface), in the thermal colour map")),
    ("particles", "Particles", "Tracer particles from the particle emitters; their trails are the streaklines"),
    ("surface", "Surface Cp", "Colour the body by its time-averaged pressure coefficient"),
)
#: What a live run shows first: the body and its smoke in the volume around it. The slice plane is
#: one click away (and always shown by the Slice and Split views): drawn by default it covered half
#: of the smoke and turned the 3D view back into colour on a surface, which is what users asked to
#: get away from. Heat, particles and the device glyphs show whenever a run has them (a run with
#: heat transfer is about its plume; a fan nobody can see cannot be judged).
DEFAULT_SHOWN_LIVE = {"smoke": True, "streamlines": False, "slice": False, "q": False, "surface": False,
                      "heat": True, "particles": True, "devices": True}
#: What a stored result shows first: its smoke snapshot with thin, speed-coloured mean-flow
#: streamlines over it (DESIGN 7.13); streamlines drawn with smoke are translucent lines, not
#: tubes (STREAMLINE_OPACITY_WITH_SMOKE), so the plume and the body stay readable. A result
#: without a smoke snapshot opens on its streamlines alone (:meth:`Fields3DPanel.set_result`).
DEFAULT_SHOWN_RESULT = {"smoke": True, "streamlines": True, "slice": False, "q": False, "surface": False,
                        "heat": True, "particles": True, "devices": True}
#: Streamlines drawn together with smoke are thin lines at this opacity rather than shaded tubes, so
#: the plume stays readable through them (tubes are kept when the lines are the only flow layer).
STREAMLINE_OPACITY_WITH_SMOKE = 0.6
#: Order in which layers are (re)built; later layers draw over earlier translucent ones.
_BUILD_ORDER = ("body", "room", "devices", "slice", "streamlines", "q", "heat", "smoke", "particles", "rake")
#: How the thermal plume is drawn: a colour-mapped translucent volume, or one iso-surface.
HEAT_STYLES = (("plume", "Plume", "The temperature as a translucent volume, clear at the ambient temperature"),
               ("surface", "Iso-surface", "One surface of constant temperature at the threshold"))
#: Default threshold of the plume, as a share of the way from the ambient (bottom of the colour
#: range) to the hottest fluid: below it the volume is clear, so ambient air never fogs the view.
HEAT_THRESHOLD_SHARE = 0.12
#: Opacity of the temperature iso-surface (translucent: the body and the smoke stay visible in it).
HEAT_SURFACE_OPACITY = 0.55
#: Smoke channels in solver order (DESIGN.md 10.1); a white or default-coloured channel is drawn in
#: the theme's smoke grey (white smoke on a light background would be invisible).
DYE_CHANNELS = ("dye", "dye1", "dye2", "dye3")
#: Device glyphs of a disabled item are drawn at this share of their opacity.
DISABLED_GLYPH_OPACITY = 0.35
#: Smoke as iso-surfaces of the dye concentration when the viewer has no volume renderer:
#: (level as a fraction of the maximum, opacity). The faint outer shell reads as haze, the
#: inner one as the dense core of the plume.
SMOKE_SHELLS = ((0.08, 0.12), (0.30, 0.26))
#: Smoke grey for light and dark backgrounds (DESIGN 7.13: dark-grey smoke on light ones,
#: white-grey on dark ones).
SMOKE_RGB = {False: (0.40, 0.43, 0.48), True: (0.90, 0.91, 0.93)}
#: Gaussian smoothing [cells] of the dye before the fallback draws its shells: a passive scalar
#: advected without diffusion is speckled at the cell scale, and iso-surfaces of it look like
#: gravel instead of a plume.
SMOKE_SMOOTH_CELLS = 0.9
#: "argument not given" for :meth:`FlowScene.set_data` (None is a meaningful value there)
_KEEP = object()
#: Slice-plane opacity while smoke is shown (an opaque plane through the plume hides half of it).
SLICE_OPACITY_WITH_SMOKE = 0.55


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def bounds_of(bodies: Sequence) -> tuple[float, float, float, float, float, float] | None:
    """(x0, x1, y0, y1, z0, z1) of meshes / surfaces (objects with ``vertices``), None without any."""
    pts = [np.asarray(b.vertices, dtype=float) for b in bodies if len(getattr(b, "vertices", []))]
    if not pts:
        return None
    a = np.vstack(pts)
    lo, hi = a.min(axis=0), a.max(axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


def _box_around(bounds, ground, *, up: float, down: float, down_span: float, side: float, side_span: float,
                vert: float, vert_height: float, vert_span: float) -> tuple[float, ...] | None:
    if bounds is None:
        return None
    x0, x1, y0, y1, z0, z1 = (float(b) for b in bounds)
    length, width, height = max(x1 - x0, 1e-9), max(y1 - y0, 0.0), max(z1 - z0, 0.0)
    my = max(side * length, side_span * width)
    mz = max(vert * length, vert_height * height, vert_span * width)
    lo_z = float(ground) if ground is not None else z0 - mz
    return (x0 - up * length, x1 + max(down * length, down_span * width), y0 - my, y1 + my,
            min(lo_z, z0), z1 + mz)


def flow_focus_box(bounds, *, ground: float | None = None) -> tuple[float, ...] | None:
    """What a 3D flow view frames: the bodies and their near wake.

    Margins scale with each dimension rather than with the largest one: a
    wing (long span, short chord) gets its trailing vortices in the frame
    without a sky above it, a car its wake without a tall empty box, a sphere
    a square-ish frame.
    """
    return _box_around(bounds, ground, up=0.25, down=1.0, down_span=0.3, side=0.2, side_span=0.15,
                       vert=0.2, vert_height=0.25, vert_span=0.1)


def near_field_box(bounds, *, ground: float | None = None) -> tuple[float, ...] | None:
    """The region a slice plane covers in a 3D view: a margin beyond :func:`flow_focus_box`.

    The plane reads as "a cut through the flow around the body"; a plane the
    size of the domain dwarfs the body and hides everything behind it.
    """
    return _box_around(bounds, ground, up=0.4, down=1.5, down_span=0.5, side=0.3, side_span=0.25,
                       vert=0.3, vert_height=0.5, vert_span=0.15)


def plane_window(box, axis: str) -> tuple[float, float, float, float] | None:
    """(col0, col1, row0, row1) of a box seen on a slice normal to ``axis`` (SliceData orientation)."""
    if box is None:
        return None
    ci, ri = {"y": (0, 2), "z": (0, 1), "x": (1, 2)}[axis]
    return (box[2 * ci], box[2 * ci + 1], box[2 * ri], box[2 * ri + 1])


def volume_box(volume) -> tuple[float, float, float, float, float, float] | None:
    """(x0, x1, y0, y1, z0, z1) of a VolumeData's outer cell edges, mirrored to full width; None without data."""
    if volume is None or not getattr(volume, "fields", None):
        return None
    first = next(iter(volume.fields.values()))
    nz, ny, nx = np.shape(first)
    ox, oy, oz = (float(c) for c in volume.origin)
    dx, dy, dz = (float(c) for c in volume.spacing)
    y0, y1 = oy, oy + ny * dy
    if getattr(volume, "mirrored", False):
        y0 = -y1
    return (ox, ox + nx * dx, y0, y1, oz, oz + nz * dz)


def heat_display_range(values) -> tuple[float, float, str]:
    """(lo, hi, mode) colour range of a temperature field around a heated (or cooled) body.

    Most of a domain is at the ambient temperature, and a plume is a few per
    cent of its cells, so percentiles of all cells (the rule of every other
    field) end a hair above ambient and saturate the whole plume. The range
    here spans the tails *around the ambient median*: the 99.5th percentile
    of the warmer-than-ambient cells and the 0.5th of the cooler ones, each
    only when at least 100 cells are in it (a few stray cells do not make a
    plume). ``mode`` is "both" when the ambient sits inside the range with
    real tails on both sides (heating and cooling), else "high" (or "low"
    when only cooling): what :meth:`Scene.add_scalar_volume` keeps clear.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0, "high"
    if v.size > 400_000:
        v = v[:: v.size // 200_000]
    med = float(np.median(v))
    spread = float(v.max() - v.min())
    eps = max(1e-3 * spread, 1e-9 * max(abs(med), 1.0))
    warm, cool = v[v > med + eps], v[v < med - eps]
    # without a real tail on a side, that end is the field's own robust extreme on it (the inlet's
    # exact ambient, not the median a slowly decaying plume tail lifts a little above it)
    hi = float(np.percentile(warm, 99.5)) if warm.size >= 100 else max(med, float(np.percentile(v, 99.5)))
    lo = float(np.percentile(cool, 0.5)) if cool.size >= 100 else min(med, float(np.percentile(v, 0.5)))
    if hi - lo <= 1e-12 * max(1.0, abs(hi)):
        pad = max(abs(med) * 0.01, 1e-3)
        return med - pad, med + pad, "high"
    span = hi - lo
    if (med - lo) > 0.25 * span and (hi - med) > 0.25 * span:
        mode = "both"
    elif (med - lo) > 0.25 * span:
        mode = "low"
    else:
        mode = "high"
    return lo, hi, mode


def has_velocity(volume) -> bool:
    """Whether a volume carries u, v and w (what streamlines need)."""
    return volume is not None and all(k in getattr(volume, "fields", {}) for k in ("u", "v", "w"))


def smoke_volume_of(result):
    """The stored smoke (dye) snapshot of a 3D result, or None.

    ``CFD3DResult.find_volume("dye")`` (DESIGN 7.13: the instantaneous block
    in ``extra_volumes``, label "instant") when the result has it; otherwise
    the same search by hand, which also accepts a ``dye`` field of the mean
    volume and a ``smoke_volume`` attribute (results from other producers).
    """
    find = getattr(result, "find_volume", None)
    if callable(find):
        for ch in DYE_CHANNELS:              # the first channel is "dye"; a run may use only others
            try:
                v = find(ch)
            except Exception:  # noqa: BLE001 - a foreign result type: search by hand below
                v = None
            if v is not None:
                return v
    for name in ("smoke_volume", "dye_volume"):
        v = getattr(result, name, None)
        if v is not None and smoke_channels(v):
            return v
    extra = getattr(result, "extra_volumes", None)
    if isinstance(extra, dict):
        extra = list(extra.values())
    for v in extra or []:
        if v is not None and smoke_channels(v):
            return v
    vol = getattr(result, "volume", None)
    if vol is not None and smoke_channels(vol):
        return vol
    return None


#: Temperature fields a volume may carry (the stored °C field first; the solver's theta as a fallback).
TEMPERATURE_FIELDS = ("temperature", "theta")


def temperature_field_of(volume) -> str | None:
    """The temperature field of a volume ("temperature" in °C, else "theta"), or None."""
    fields = getattr(volume, "fields", None) or {}
    return next((f for f in TEMPERATURE_FIELDS if f in fields), None)


def heat_volume_of(result):
    """The block of a 3D result that holds its temperature: the time-averaged one when stored (the
    plume the mean slices and the Nusselt number describe), else the last snapshot; None without."""
    vols = []
    fn = getattr(result, "volumes", None)
    if callable(fn):
        try:
            vols = list(fn())
        except Exception:  # noqa: BLE001 - a foreign result type
            vols = []
    if not vols:
        vols = [v for v in [getattr(result, "volume", None), *(getattr(result, "extra_volumes", None) or [])] if v]
    with_t = [v for v in vols if temperature_field_of(v)]
    return next((v for v in with_t if getattr(v, "label", "") == "mean"), with_t[0] if with_t else None)


def _accepts(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call(fn, *args, **kw):
    """Call ``fn`` with only the keywords it accepts (the viewer API grows independently of this page)."""
    return fn(*args, **{k: v for k, v in kw.items() if _accepts(fn, k)})


# ---------------------------------------------------------------------------
# seed rake
# ---------------------------------------------------------------------------

#: Emitter kinds the rake editor handles (Setup ▸ Flow visualisation names them), in its words.
RAKE_KIND_TEXT = {"rake": "Rake", "vertical": "Vertical rake", "grid": "Grid", "plane": "Plane"}
#: Streamline seeds of a rake left on automatic counts (``aerolume.viz3d.flow.SEED_COUNTS``; the
#: fallback when that module is missing). A plane is capped for lines traced on the GUI thread:
#: the solver traces its 300 itself during a run, but a stored result is traced here, at ~4 ms
#: a line on a 2 M-cell volume.
AUTO_SEEDS = {"rake": 30, "vertical": 24, "grid": 120, "plane": 96}


def _set_layer_opacity(scene, handle: str, opacity: float) -> None:
    """Opacity of every actor of one scene layer.

    ``Scene.set_opacity`` is used when the viewer has it; otherwise the layer's
    actors are reached directly (a viewer without either keeps the lines opaque,
    which is still correct, only denser).
    """
    fn = getattr(scene, "set_opacity", None)
    if callable(fn):
        fn(handle, float(opacity))
        return
    lay = getattr(scene, "_layers", {}).get(handle)
    for actor in getattr(lay, "actors", []) or []:
        prop = getattr(actor, "GetProperty", None)
        if prop is not None:
            prop().SetOpacity(float(opacity))


def _flow_module():
    return lazy_import("aerolume.viz3d.flow")


def _split_counts(total: int, span: float, height: float) -> tuple[int, int]:
    """(across, up) counts of about ``total`` seeds on a span x height face, square-ish spacing
    (the resolver's rule for a grid or a band given one number)."""
    aspect = max(span, 1e-9) / max(height, 1e-9)
    up = max(2, round(math.sqrt(max(total, 4) / max(aspect, 1e-3))))
    return max(2, round(max(total, 4) / up)), up


def _auto_counts(kind: str, span: float, height: float) -> tuple[int, int]:
    """(across, up) seed counts of an automatic rake (what the resolver picks for streamlines)."""
    if kind == "rake":
        return AUTO_SEEDS["rake"], 1
    if kind == "vertical":
        return 1, AUTO_SEEDS["vertical"]
    return _split_counts(AUTO_SEEDS[kind], span, height)


def _extra_items(d: dict) -> tuple:
    """The rake dict's pass-through keys (nozzle radius, smoke pulses, a typed spacing) as a hashable tuple."""
    out = []
    for k in ("radius", "pulse", "spacing"):
        v = d.get(k)
        if v is None:
            continue
        out.append((k, tuple(float(t) for t in v) if isinstance(v, (list, tuple)) else float(v)))
    return tuple(out)


@dataclass(frozen=True)
class RakeSpec:
    """Where smoke is released and streamlines start: a seed rake in the flow (metres, wind frame).

    The kinds of :mod:`aerolume.viz3d.flow` the Setup page offers: a **rake**
    (a line across the flow at height ``z``, from ``y_min`` to ``y_max``), a
    **vertical** rake (a line from ``z`` up to ``z_top`` at ``y_centre``), a
    **grid** of nozzles and a **plane** band (both from ``y_min`` to ``y_max``
    and ``z`` to ``z_top``), all at ``x``. ``n`` counts the lines across (up a
    vertical rake), ``n_up`` the rows of a grid or plane. ``auto_n`` keeps the
    counts automatic: they are left out of the dict, so the solver spaces its
    smoke nozzles by its own rule (streams that stay distinct) while the
    streamlines use :data:`AUTO_SEEDS`. ``extra`` carries the nozzle radius,
    smoke pulses and a typed spacing through unchanged.

    One rake is the one definition of "where": :meth:`to_view` is
    ``ViewSettings.seed_rake``, which the solver releases its smoke from and
    traces its streamlines from, and :meth:`seeds` are the same points
    (``aerolume.viz3d.flow.resolve_rake`` order) for lines traced in the GUI.
    """
    x: float
    z: float
    y_min: float
    y_max: float
    n: int = 24
    kind: str = "rake"
    z_top: float | None = None
    n_up: int = 1
    auto_n: bool = False
    extra: tuple = ()

    @property
    def span(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def y_centre(self) -> float:
        return 0.5 * float(self.y_min + self.y_max)

    @property
    def z_max(self) -> float:
        """Upper edge (the rake's own height for a line rake)."""
        return float(self.z if self.z_top is None or self.kind == "rake" else self.z_top)

    @property
    def height(self) -> float:
        return self.z_max - float(self.z)

    @property
    def ranged(self) -> bool:
        """Whether the rake spans heights (vertical, grid, plane) rather than standing at one."""
        return self.kind != "rake"

    def counts(self) -> tuple[int, int]:
        """(across, up) numbers of seeds."""
        if self.kind == "rake":
            return max(int(self.n), 1), 1
        if self.kind == "vertical":
            return 1, max(int(self.n), 1)
        n, up = max(int(self.n), 1), max(int(self.n_up), 1)
        if self.auto_n and n * up > AUTO_SEEDS[self.kind]:
            return _auto_counts(self.kind, self.span, self.height)
        return n, up

    def seeds(self) -> np.ndarray:
        """(N, 3) seed points: a row across for each height, ends included (the resolver's order)."""
        na, nu = self.counts()
        ys = np.linspace(self.y_min, self.y_max, na) if na > 1 else np.array([self.y_centre])
        zs = np.linspace(self.z, self.z_max, nu) if nu > 1 else np.array([0.5 * (self.z + self.z_max)])
        Y, Z = np.meshgrid(ys, zs)
        return np.stack([np.full(Y.size, float(self.x)), Y.ravel(), Z.ravel()], axis=1)

    def with_span(self, span: float) -> RakeSpec:
        if self.kind == "vertical":
            return self
        half = 0.5 * max(float(span), 0.0)
        return replace(self, y_min=self.y_centre - half, y_max=self.y_centre + half)

    def moved_to(self, other: RakeSpec) -> RakeSpec:
        """This rake (kind, counts, extras) at ``other``'s place: what a drag of its handle does."""
        z_top = None if not self.ranged else (other.z_max if other.ranged else other.z + self.height)
        return replace(self, x=other.x, z=other.z, y_min=other.y_min, y_max=other.y_max, z_top=z_top)

    def clamped(self, box: Sequence[float] | None) -> RakeSpec:
        """The rake moved inside ``box`` (x0, x1, y0, y1, z0, z1), a hair from its faces, so every seed
        lies in the data (a seed outside the volume gives no line at all)."""
        if box is None:
            return self
        x0, x1, y0, y1, z0, z1 = (float(b) for b in box)
        eps = 1e-3 * max(x1 - x0, y1 - y0, z1 - z0, 1e-9)

        def cl(v: float, lo: float, hi: float) -> float:
            return min(max(float(v), lo + eps), hi - eps)

        return replace(self, x=cl(self.x, x0, x1), z=cl(self.z, z0, z1), y_min=cl(self.y_min, y0, y1),
                       y_max=cl(self.y_max, y0, y1), z_top=None if self.z_top is None else cl(self.z_top, z0, z1))

    def to_view(self) -> dict:
        """The rake as ``ViewSettings.seed_rake``: the rake dict of :mod:`aerolume.viz3d.flow` (metres,
        wind frame, plain JSON values). Counts only when the user chose them (``auto_n`` False)."""
        d: dict = {"kind": self.kind, "x": float(self.x)}
        d["y"] = float(self.y_centre) if self.kind == "vertical" else [float(self.y_min), float(self.y_max)]
        d["z"] = float(self.z) if self.kind == "rake" else [float(self.z), float(self.z_max)]
        if not self.auto_n:
            if self.kind in ("rake", "vertical"):
                d["n"] = int(self.n)
            else:
                d["n"] = [int(self.n), int(self.n_up)]
        for k, v in self.extra:
            d[k] = list(v) if isinstance(v, tuple) else v
        return d

    @classmethod
    def from_view(cls, d) -> RakeSpec | None:
        """Back from a rake dict that places itself (x, y, z given); None for anything else.

        Missing counts become automatic ones (:data:`AUTO_SEEDS`)."""
        try:
            kind = str(d.get("kind", "rake") or "rake")
            if kind not in RAKE_KIND_TEXT:
                return None
            x = float(d["x"])
            ys = sorted(float(v) for v in (list(d["y"]) if np.ndim(d["y"]) else [d["y"], d["y"]])[:2])
            zs = sorted(float(v) for v in (list(d["z"]) if np.ndim(d["z"]) else [d["z"], d["z"]])[:2])
            n_spec = d.get("n")
            auto = n_spec is None
            span, height = ys[1] - ys[0], zs[1] - zs[0]
            if auto:
                n, up = _auto_counts(kind, span, height)
            elif np.ndim(n_spec):
                n, up = int(n_spec[0]), int(n_spec[1])
            elif kind in ("grid", "plane"):              # one number: the total over the face
                n, up = _split_counts(int(n_spec), span, height)
            else:
                n, up = int(n_spec), int(n_spec)
            n = up if kind == "vertical" else n
            return cls(x=x, z=zs[0], y_min=ys[0], y_max=ys[1], n=max(int(n), 1), kind=kind,
                       z_top=None if kind == "rake" else zs[1], n_up=max(int(up), 1) if kind in ("grid", "plane") else 1,
                       auto_n=auto, extra=_extra_items(d))
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            return None

    @classmethod
    def from_seed_rake(cls, sr) -> RakeSpec | None:
        """A resolved :class:`aerolume.viz3d.flow.SeedRake` as a RakeSpec (None for loose seed points)."""
        kind = str(getattr(sr, "kind", "rake"))
        if kind not in RAKE_KIND_TEXT:
            return None
        na, nu = (int(v) for v in sr.n)
        auto = not bool((getattr(sr, "extra", None) or {}).get("n_given"))
        if auto:      # the GUI's automatic counts (a band traced here is capped; see AUTO_SEEDS)
            na, nu = _auto_counts(kind, float(sr.y[1]) - float(sr.y[0]), float(sr.z[1]) - float(sr.z[0]))
        d = {"radius": getattr(sr, "radius", None), "pulse": getattr(sr, "pulse", None)}
        return cls(x=float(sr.x), z=float(sr.z[0]), y_min=float(sr.y[0]), y_max=float(sr.y[1]),
                   n=max(nu if kind == "vertical" else na, 1), kind=kind,
                   z_top=None if kind == "rake" else float(sr.z[1]),
                   n_up=max(nu, 1) if kind in ("grid", "plane") else 1, auto_n=auto, extra=_extra_items(d))

    def to_seed_rake(self):
        """This rake as a :class:`aerolume.viz3d.flow.SeedRake` (what the viewer draws), or None
        when the viewer predates it."""
        flow = _flow_module()
        cls = getattr(flow, "SeedRake", None) if flow is not None else None
        if cls is None:
            return None
        ex = dict(self.extra)
        pulse = ex.get("pulse")
        return cls(self.kind, float(self.x), (float(self.y_min), float(self.y_max)), (float(self.z), self.z_max),
                   self.counts(), self.seeds(), ex.get("radius"), tuple(pulse) if pulse else None,
                   {"n_given": not self.auto_n})

    @classmethod
    def resolve(cls, spec: dict | None, bounds: Sequence[float], *, domain: Sequence[float] | None = None,
                cell: float | None = None) -> RakeSpec:
        """A rake dict (Setup's emitter, a saved rake, None for the default) placed for bodies in
        ``bounds`` and a flow domain, by :func:`aerolume.viz3d.flow.resolve_rake` -- the call the
        solver makes -- so the rake drawn here is where the run releases its smoke. Without that
        module (or for a dict it refuses) the default line rake of :meth:`around`."""
        x0, x1, y0, y1, z0, z1 = (float(b) for b in bounds)
        size = max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
        box = tuple(float(b) for b in domain) if domain is not None else (
            x0 - 2 * size, x1 + 3 * size, y0 - size, y1 + size, z0 - size, z1 + size)
        flow = _flow_module()
        fn = getattr(flow, "resolve_rake", None) if flow is not None else None
        if fn is not None:
            try:
                sr = fn(dict(spec or {}), tuple(float(b) for b in bounds), box, purpose="streamlines", h=cell)
                out = cls.from_seed_rake(sr)
                if out is not None:
                    return out
            except (ValueError, TypeError, KeyError):
                log.debug("rake %r could not be resolved; the default rake is used", spec, exc_info=True)
        return cls.around(bounds, domain=box)

    @classmethod
    def around(cls, bounds: Sequence[float], *, domain: Sequence[float] | None = None,
               n: int | None = None) -> RakeSpec:
        """The default line rake for bodies in ``bounds`` (x0, x1, y0, y1, z0, z1).

        The solver's own default (:func:`aerolume.viz3d.flow.resolve_rake`, a
        short way upstream of the nose, at 40 % of the body's height, 20 % of its
        width wider on each side) so a run that was never given a rake releases
        its smoke where this one is drawn; ``n`` fixes the number of lines
        (None: automatic). Without the flow module: a quarter of a length
        upstream at 40 % of the height.
        """
        flow = _flow_module()
        fn = getattr(flow, "resolve_rake", None) if flow is not None else None
        x0, x1, y0, y1, z0, z1 = (float(b) for b in bounds)
        size = max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
        box = tuple(float(b) for b in domain) if domain is not None else (
            x0 - 2 * size, x1 + 3 * size, y0 - size, y1 + size, z0 - size, z1 + size)
        if fn is not None:
            spec = {"kind": "rake"} if n is None else {"kind": "rake", "n": int(n)}
            try:
                out = cls.from_seed_rake(fn(spec, (x0, x1, y0, y1, z0, z1), box, purpose="streamlines"))
                if out is not None:
                    return out
            except (ValueError, TypeError):
                log.debug("default rake could not be resolved", exc_info=True)
        width, height = max(y1 - y0, 1e-9), max(z1 - z0, 1e-9)
        yc = 0.5 * (y0 + y1)
        spec = cls(x=x0 - 0.25 * max(x1 - x0, 1e-9), z=z0 + 0.4 * height, y_min=yc - 0.7 * width,
                   y_max=yc + 0.7 * width, n=AUTO_SEEDS["rake"] if n is None else int(n), auto_n=n is None)
        return spec.clamped(box)


def setup_flow_viz(sim) -> dict | None:
    """Setup ▸ Flow visualisation of a simulation: ``{"smoke": bool, "rake": rake dict}``, or None.

    The Setup page keeps those settings (whether smoke is released, the emitter
    and its heights and nozzle counts) wherever the data model allows
    (``aerolume.app.pages.setup.flow_viz_of``); the rake dict is in the format
    the solver reads (``stored_flow_viz``). None when the page or its helpers
    are unavailable: callers then use the default rake.
    """
    if sim is None:
        return None
    setup = lazy_import("aerolume.app.pages.setup")
    get = getattr(setup, "flow_viz_of", None) if setup is not None else None
    store = getattr(setup, "stored_flow_viz", None) if setup is not None else None
    if not callable(get) or not callable(store):
        return None
    try:
        out = store(get(sim))
    except Exception:  # the smoke settings are advisory; a run has its default rake
        log.debug("the Setup flow visualisation of %s could not be read", getattr(sim, "id", "?"), exc_info=True)
        return None
    if not isinstance(out, dict):
        return None
    rake = out.get("rake")
    return {"smoke": bool(out.get("smoke", True)), "rake": dict(rake) if isinstance(rake, dict) else None}


def _tube(p, q, r: float, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of a capped tube of radius ``r`` from ``p`` to ``q``."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = float(np.linalg.norm(d))
    t = d / length if length > 0 else np.array([0.0, 1.0, 0.0])
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(t, a)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    ang = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
    ring = r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    verts = np.vstack([ring + p, ring + q, p[None], q[None]])
    j = np.arange(k)
    jn = (j + 1) % k
    faces = np.vstack([np.stack([j, jn, k + jn], 1), np.stack([j, k + jn, k + j], 1),
                       np.stack([np.full(k, 2 * k), jn, j], 1), np.stack([np.full(k, 2 * k + 1), k + j, k + jn], 1)])
    return verts, faces


def rake_mesh(spec: RakeSpec, size: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of the drawn rake: thin bars (its line, or the frame of a grid or band) with a
    small bead at every nozzle (a plane band has none: it releases smoke over its whole face).

    ``size`` is the scene's length scale (the body length); the bar's radius is
    0.4 % of it, so the rake reads as an instrument, not as part of the model.
    """
    r = max(0.004 * float(size), 1e-6)
    x, y0, y1, z0, z1 = float(spec.x), float(spec.y_min), float(spec.y_max), float(spec.z), spec.z_max
    if spec.kind in ("grid", "plane"):
        c = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
        bars = [(c[i], c[(i + 1) % 4]) for i in range(4)]
    elif spec.kind == "vertical":
        bars = [((x, spec.y_centre, z0), (x, spec.y_centre, z1))]
    else:
        bars = [((x, y0, z0), (x, y1, z0))]
    verts, faces, off = [], [], 0
    for p, q in bars:
        v, f = _tube(p, q, r)
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    if spec.kind != "plane":
        # nozzles: small octahedra, 2.2 bar radii across
        oct_v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * 2.2 * r
        oct_f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        for p in spec.seeds():
            verts.append(oct_v + p)
            faces.append(oct_f + off)
            off += 6
    return np.vstack(verts).astype(np.float64), np.vstack(faces).astype(np.int64)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def keep_whole(widget: QWidget) -> QWidget:
    """Never squeeze ``widget``'s buttons below their text: a narrow strip reflows or gives the
    space up elsewhere instead of eliding "Smoke" to "S…e" (returns ``widget``)."""
    from PySide6.QtWidgets import QAbstractButton
    buttons = [widget] if isinstance(widget, QAbstractButton) else widget.findChildren(QAbstractButton)
    for b in buttons:
        b.setMinimumWidth(b.sizeHint().width())
    return widget


def layer_toggle_qss(tk: dict | None = None) -> str:
    """Style sheet of a layer toggle: on = selection fill with a 1 px ink border, off = text2.

    The application's flat ``variant="toggle"`` (on = text colour, off = muted)
    suits filter toggles in a title row, but for layers drawn in a viewport it
    made the off buttons look disabled and, in the light theme, left it hard to
    tell which layers were on; next to the View segmented control (a filled
    checked item) the two controls also disagreed. A disabled layer is faint, and
    a layer that is wanted but has no data yet (the smoke of a run before its
    first volume) keeps an outline without the fill. The border is ink (``text``)
    since v4, like every checked control in the chrome, and the corners follow
    the theme's control radius.
    """
    t = tk or tokens()
    radius = 6
    th = lazy_import("aerolume.app.theme")
    if th is not None:
        radius = int(getattr(th, "RADIUS", radius))
    return (f"QToolButton {{ padding: 0 8px; border: 1px solid transparent; border-radius: {radius}px; "
            f"background: transparent; color: {t['text2']}; }}\n"
            f"QToolButton:hover {{ background: {t['hover']}; color: {t['text']}; }}\n"
            f"QToolButton:pressed {{ background: {t['pressed']}; }}\n"
            f"QToolButton:checked {{ background: {t['accent_soft']}; border-color: {t['text']}; "
            f"color: {t['text']}; }}\n"
            f"QToolButton:disabled {{ color: {t['faint']}; }}\n"
            f"QToolButton:checked:disabled {{ background: transparent; border-color: {t['border_strong']}; "
            f"color: {t['muted']}; }}\n")


class FlowToggles(QObject):
    """Checkable tool buttons for the flow layers; ``changed(key, on)`` on a user change.

    ``keys`` picks and orders the buttons (a live run has no Q or surface data
    to show, for example). :meth:`set_available` disables a button and puts the
    reason in its tooltip, so a greyed "Smoke" always says why. The buttons are
    styled by :func:`layer_toggle_qss` (restyled on a theme change), so an "on"
    layer reads like the checked item of the View control beside them.

    In a narrow viewport the buttons that do not fit move into a "⋯" menu at
    the end of the row (:meth:`overflow_to`), as checkable entries: at 1280 px
    with the Live workspace the row was cut to "Sur". :meth:`set_shown_keys`
    says which layers the page offers at all (a run without heat transfer has
    no "Thermal plume"); visibility is decided here from both.
    """

    changed = Signal(str, bool)

    def __init__(self, keys: Sequence[str] = ("smoke", "streamlines", "slice", "surface"), *,
                 shown: dict | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        info = {k: (t, tip) for k, t, tip in FLOW_LAYERS}
        shown = dict(DEFAULT_SHOWN_RESULT if shown is None else shown)
        self.buttons: dict[str, QToolButton] = {}
        self._tips: dict[str, str] = {}
        self._present: set[str] = set(keys)       # layers the page offers (set_shown_keys)
        self._room: int | None = None             # the row's width for the buttons (overflow_to)
        self._overflow: list[str] = []
        self._strip = None
        self.more_btn: QToolButton | None = None
        for key in keys:
            text, tip = info[key]
            b = QToolButton()
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setCheckable(True)
            b.setChecked(bool(shown.get(key, False)))
            b.setProperty("variant", "layer")
            b.setToolTip(tip)
            b.setFixedHeight(24)
            b.toggled.connect(lambda on, k=key: self.changed.emit(k, bool(on)))
            self.buttons[key] = b
            self._tips[key] = tip
        self._restyle()
        theme = lazy_import("aerolume.app.theme")
        sig_ = getattr(theme, "changed", None) if theme is not None else None
        if sig_ is not None:
            sig_.connect(self._restyle)          # bound method of a QObject: dropped with this object

    def _restyle(self, *_args) -> None:
        qss = layer_toggle_qss()
        if qss == getattr(self, "_qss", None):
            return                               # the same theme re-applied: no re-polish of every button
        self._qss = qss
        for b in self.buttons.values():
            try:
                b.setStyleSheet(qss)
                b.setMinimumWidth(0)
                keep_whole(b)                    # its text at the restyled padding
            except RuntimeError:                 # a button already destroyed with its strip
                pass

    def add_to(self, strip: ToolStrip) -> None:
        """Append the buttons to a tool strip, in order, then the "⋯" overflow menu (hidden while all fit)."""
        from PySide6.QtWidgets import QMenu
        for b in self.buttons.values():
            strip.add_widget(b)
        more = QToolButton()
        theme = lazy_import("aerolume.app.theme")
        if theme is not None and hasattr(theme, "icon"):
            more.setIcon(theme.icon("more", "text2", active="accent"))
        else:
            more.setText("⋯")
        more.setToolTip("More layers: the ones that do not fit this row")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more.setFixedHeight(24)
        menu = QMenu(more)
        menu.aboutToShow.connect(self._fill_menu)
        more.setMenu(menu)
        more.hide()
        strip.add_widget(more)
        self.more_btn, self._strip = more, strip

    def set_shown_keys(self, keys) -> None:
        """The layers the page offers now (the others' buttons are hidden, never put in the menu)."""
        keys = {k for k in keys if k in self.buttons}
        if keys != self._present:
            self._present = keys
        self.relayout()

    def overflow_to(self, width: int | None) -> None:
        """Fit the buttons into ``width`` px, the rest into the "⋯" menu; None shows every offered button."""
        self._room = None if width is None else int(width)
        self.relayout()

    def overflowed(self) -> list[str]:
        """Keys of the layers that are in the "⋯" menu now (for tests and the page)."""
        return list(self._overflow)

    def relayout(self) -> None:
        keys = [k for k in self.buttons if k in self._present]
        over: list[str] = []
        if self._room is not None and self.more_btn is not None and keys:
            sp = max(self._strip.row.spacing(), 0) if self._strip is not None else 4
            widths = [max(self.buttons[k].minimumWidth(), self.buttons[k].sizeHint().width()) for k in keys]
            if sum(widths) + sp * (len(keys) - 1) > self._room:
                room = self._room - self.more_btn.sizeHint().width() - sp
                used = 0
                for k, w in zip(keys, widths):
                    if over or used + w > room:
                        over.append(k)              # in order: once one does not fit, the rest go too
                    else:
                        used += w + sp
        self._overflow = over
        for k, b in self.buttons.items():
            want = k in self._present and k not in over
            if b.isHidden() == want:
                b.setVisible(want)
        if self.more_btn is not None:
            self.more_btn.setVisible(bool(over))
            on = [k for k in over if self.buttons[k].isChecked()]
            self.more_btn.setToolTip("More layers: " + ", ".join(self.buttons[k].text() for k in over)
                                     + (f" ({', '.join(self.buttons[k].text() for k in on)} on)" if on else ""))

    def _fill_menu(self) -> None:
        menu = self.more_btn.menu() if self.more_btn is not None else None
        if menu is None:
            return
        menu.clear()
        for k in self._overflow:
            b = self.buttons[k]
            act = menu.addAction(b.text())
            act.setCheckable(True)
            act.setChecked(b.isChecked())
            act.setEnabled(b.isEnabled())
            act.setToolTip(b.toolTip())
            act.toggled.connect(b.setChecked)       # the button's toggled -> changed, as a click would

    def is_on(self, key: str) -> bool:
        b = self.buttons.get(key)
        return bool(b is not None and b.isChecked() and b.isEnabled())

    def wanted(self, key: str) -> bool:
        """Checked, whether or not the layer can be shown right now (a live run's smoke before its first volume)."""
        b = self.buttons.get(key)
        return bool(b is not None and b.isChecked())

    def set_on(self, key: str, on: bool) -> None:
        b = self.buttons.get(key)
        if b is not None and b.isChecked() != bool(on):
            b.blockSignals(True)
            b.setChecked(bool(on))
            b.blockSignals(False)

    def set_available(self, key: str, ok: bool, reason: str = "") -> None:
        b = self.buttons.get(key)
        if b is None:
            return
        b.setEnabled(bool(ok))
        b.setToolTip(self._tips[key] if ok or not reason else f"{self._tips[key]}.\nUnavailable: {reason}")

    def shown(self) -> dict[str, bool]:
        return {k: self.is_on(k) for k in self.buttons}


class RakeEditor(QObject):
    """The seed rake's rows in a property group, and Reset.

    Rows: the emitter (``show_kind``: Setup's choice, named, not edited here),
    position x, the height (one height for a rake; from / to for a vertical
    rake, a grid or a plane band), the lateral span (not for a vertical rake),
    the number of lines (and rows of a grid). Values are metres in the wind
    frame. ``changed`` fires once per user edit (spin boxes commit on Enter,
    focus-out or an arrow step, not per keystroke, so a rake is not re-traced
    for "1", "1.", "1.2"). Counts stay automatic (the solver spaces its smoke
    nozzles itself) until the user changes one.
    """

    changed = Signal()
    #: Edits that follow each other closer than this [ms] (a held arrow key, the mouse wheel)
    #: become one ``changed``: each one re-traces the streamlines.
    SETTLE_MS = 150

    def __init__(self, group: PropertyGroup, parent: QObject | None = None, *, show_kind: bool = False) -> None:
        super().__init__(parent)
        self._default: RakeSpec | None = None
        self._spec: RakeSpec | None = None           # the rake shown (kind, counts, extras)
        self._box: tuple[float, ...] | None = None
        self._lock = False
        self._auto = False                           # counts still automatic
        self._at_default = False                     # shows the default rake, unedited
        self._show_kind = bool(show_kind)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(self.SETTLE_MS)
        self._settle.timeout.connect(self.changed)
        self.kind_value = label("", "secondary")
        self._row_kind = group.add_row("Emitter", self.kind_value,
                                       tooltip="Chosen in Setup ▸ Flow visualisation; positioned here")
        self._row_kind.set_visible(False)
        self.x = self._spin("Position x", group, "Distance along the flow; the rake must stay upstream of the body")
        self.z = self._spin("Height z", group, "Height of the rake above the reference plane (the road for vehicles)")
        self._row_z = group.rows["Height z"]
        self.z_top = self._spin("Height to", group, "Upper edge of the emitter")
        self._row_z_top = group.rows["Height to"]
        self._row_z_top.set_visible(False)
        self.span = self._spin("Span", group, "Width of the rake across the flow, centred on the body")
        self._row_span = group.rows["Span"]
        self.n = QSpinBox()
        self.n.setRange(1, 200)
        self.n.setValue(24)
        self.n.setKeyboardTracking(False)
        self._row_n = group.add_row("Lines", self.n, unit="",
                                    tooltip="Number of nozzles on the rake: streamlines start at each")
        self.n_up = QSpinBox()
        self.n_up.setRange(1, 100)
        self.n_up.setValue(1)
        self.n_up.setKeyboardTracking(False)
        self._row_n_up = group.add_row("Rows", self.n_up, tooltip="Rows of nozzles from the lower to the upper edge")
        self._row_n_up.set_visible(False)
        self.reset_btn = button("Reset rake", "link")
        self.reset_btn.setToolTip("Back to where Setup places the emitter (by default a short way upstream of the "
                                  "nose, at 40 % of the body's height)")
        group.add_widget(self.reset_btn)
        for w in (self.x, self.z, self.z_top, self.span):
            w.valueChanged.connect(self._edited)
        self.n.valueChanged.connect(self._counts_edited)
        self.n_up.valueChanged.connect(self._counts_edited)
        self.reset_btn.clicked.connect(self.reset)
        self.set_enabled(False)

    @staticmethod
    def _spin(text: str, group: PropertyGroup, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(3)
        s.setRange(-1e6, 1e6)
        s.setKeyboardTracking(False)
        s.setAlignment(Qt.AlignmentFlag.AlignRight)
        s.setMinimumWidth(96)
        group.add_row(text, s, unit="m", tooltip=tip)
        return s

    def set_enabled(self, on: bool) -> None:
        for w in (self.x, self.z, self.z_top, self.span, self.n, self.n_up, self.reset_btn):
            w.setEnabled(bool(on))

    def kind(self) -> str:
        return self._spec.kind if self._spec is not None else "rake"

    def _layout_for(self, spec: RakeSpec) -> None:
        """Rows and labels for the emitter's kind (a vertical rake has no span, a band no nozzles)."""
        kind = spec.kind
        self.kind_value.setText(RAKE_KIND_TEXT.get(kind, kind.capitalize()))
        self._row_kind.set_visible(self._show_kind)
        self._row_z.set_label("Height from" if spec.ranged else "Height z")
        self._row_z_top.set_visible(spec.ranged)
        self._row_span.set_visible(kind != "vertical")
        self._row_n.set_label({"grid": "Nozzles across", "plane": "Lines across"}.get(kind, "Lines"))
        self._row_n_up.set_visible(kind in ("grid", "plane"))
        if kind == "plane":
            tip = "Streamlines start on a grid of points over the band; the band releases smoke over its whole face"
        elif spec.auto_n:
            tip = ("Streamlines start at each line; the solver spaces its smoke nozzles itself until a count is "
                   "set here")
        else:
            tip = "Number of nozzles: smoke is released, and a streamline starts, at each"
        self.n.setToolTip(tip)
        self._row_n.label.setToolTip(tip)
        if self._row_n.unit is not None:
            self._row_n.unit.setText("auto" if spec.auto_n else "")
            self._row_n.unit.setToolTip("Automatic until changed" if spec.auto_n else "")

    def set_rake(self, spec: RakeSpec | None, *, default: RakeSpec | None = None,
                 box: Sequence[float] | None = None) -> None:
        """Show ``spec`` (no signal); ``default`` is what Reset returns to, ``box`` bounds the values."""
        self._default = default if default is not None else spec
        self._box = tuple(float(b) for b in box) if box is not None else None
        self.set_enabled(spec is not None)
        self._spec = spec
        self._at_default = spec is not None and spec == self._default
        if spec is None:
            return
        self._lock = True
        try:
            self._layout_for(spec)
            if self._box is not None:
                x0, x1, y0, y1, z0, z1 = self._box
                self.x.setRange(x0, x1)
                self.z.setRange(z0, z1)
                self.z_top.setRange(z0, z1)
                self.span.setRange(0.0, y1 - y0)
                size = max(x1 - x0, y1 - y0, z1 - z0)
            else:
                size = max(abs(spec.span), abs(spec.height), 1.0)
            step = 10.0 ** math.floor(math.log10(max(size / 100.0, 1e-6)))
            dec = max(0, min(4, -math.floor(math.log10(step)) + 1))
            for w in (self.x, self.z, self.z_top, self.span):
                w.setSingleStep(step)
                w.setDecimals(dec)
            self.x.setValue(spec.x)
            self.z.setValue(spec.z)
            self.z_top.setValue(spec.z_max)
            self.span.setValue(spec.span)
            na, nu = spec.counts()
            self.n.setValue(nu if spec.kind == "vertical" else na)
            self.n_up.setValue(nu)
            self._auto = bool(spec.auto_n)
        finally:
            self._lock = False

    def show_rake(self, spec: RakeSpec) -> None:
        """Show ``spec`` without a signal, keeping the default and the bounds (a rake dragged in the view)."""
        self.set_rake(spec, default=self._default, box=self._box)

    def is_default(self) -> bool:
        """Whether the rake shown is the default one, unedited (never touched, or Reset)."""
        return bool(self._at_default and self._spec is not None)

    def rake(self) -> RakeSpec | None:
        """The rake the rows describe (the default itself, unrounded by the spin boxes, until edited)."""
        base = self._spec
        if base is None or not self.x.isEnabled():
            return None
        if self.is_default() and self._default is not None:
            return self._default.clamped(self._box)
        yc = base.y_centre
        half = 0.5 * self.span.value()
        z0, z1 = self.z.value(), self.z_top.value()
        if base.ranged and z1 < z0:
            z0, z1 = z1, z0
        vertical = base.kind == "vertical"
        spec = replace(base, x=self.x.value(), z=z0, z_top=z1 if base.ranged else None,
                       y_min=yc if vertical else yc - half, y_max=yc if vertical else yc + half,
                       n=self.n.value(), n_up=self.n_up.value() if base.kind in ("grid", "plane") else 1,
                       auto_n=self._auto)
        return spec.clamped(self._box)

    def reset(self) -> None:
        if self._default is not None:
            self._settle.stop()
            self.set_rake(self._default, default=self._default, box=self._box)
            self.changed.emit()

    def _edited(self, *_) -> None:
        if not self._lock:
            self._at_default = False
            self._settle.start()

    def _counts_edited(self, *_) -> None:
        if self._lock:
            return
        self._at_default = False
        if self._auto and self._spec is not None:
            # a count typed by the user: from now on it is the solver's nozzle count too
            self._auto = False
            self._spec = replace(self._spec, auto_n=False)
            self._layout_for(self._spec)
        self._settle.start()


# ---------------------------------------------------------------------------
# the scene controller
# ---------------------------------------------------------------------------

def smoke_colour(color: str | None) -> str:
    """The colour a smoke channel is drawn in: its emitter's, or "auto" (the theme's smoke grey) for
    white, near-white or unset colours, which would vanish on the light background."""
    if not color:
        return "auto"
    try:
        from ...viz.colormaps import hex_rgb
        r, g, b = hex_rgb(color)
    except (ValueError, ImportError):
        return "auto"
    return "auto" if min(r, g, b) >= 225 else f"#{r:02x}{g:02x}{b:02x}"


def dye_colours_of(sim) -> dict[str, str]:
    """{channel field: drawn colour} of a simulation's smoke emitters (``viz.colormaps.dye_channels``:
    one channel per distinct emitter colour, the order the solver uses)."""
    emitters = list(getattr(sim, "emitters", None) or []) if sim is not None else []
    if not emitters:
        return {}
    try:
        from ...viz.colormaps import dye_channels
        chans = dye_channels(emitters)
    except ImportError:
        return {}
    return {str(ch["field"]): smoke_colour(ch.get("color")) for ch in chans}


#: Words for the emitter colours users pick most (the legend names a channel by kind and colour when
#: its emitters carry the default name "Emitter", which told the three car rakes apart not at all).
_COLOUR_WORDS = ((0.0, "red"), (30.0, "orange"), (50.0, "yellow"), (90.0, "green"), (170.0, "teal"),
                 (200.0, "blue"), (260.0, "violet"), (300.0, "magenta"), (345.0, "red"))


def colour_word(hex_colour: str) -> str:
    """'white', 'grey', 'orange', 'blue' ...: a plain name for an emitter colour (legend text)."""
    try:
        from ...viz.colormaps import hex_rgb
        r, g, b = (c / 255.0 for c in hex_rgb(hex_colour))
    except (ValueError, ImportError):
        return ""
    import colorsys
    h, light, sat = colorsys.rgb_to_hls(r, g, b)
    if sat < 0.18 or light > 0.92 or light < 0.08:
        return "white" if light > 0.8 else "black" if light < 0.2 else "grey"
    deg = h * 360.0
    word = "red"
    for start, name in _COLOUR_WORDS:
        if deg >= start:
            word = name
    return word


def smoke_channel_labels(sim) -> dict[str, list[str]]:
    """{channel field: one label per smoke emitter} for the legend, in the solver's channel order.

    An emitter's own name when the user gave it one, else its kind and colour
    ("Rake, white", "Point, blue"): three emitters all called "Emitter" were
    indistinguishable in the legend.
    """
    emitters = [e for e in (getattr(sim, "emitters", None) or []) if getattr(e, "species", "dye") == "dye"
                and getattr(e, "enabled", True)] if sim is not None else []
    if not emitters:
        return {}
    try:
        from ...viz.colormaps import dye_channels, hex_rgb
    except ImportError:
        return {}
    chans = dye_channels(emitters)
    by_rgb = {}
    for ch in chans:
        try:
            by_rgb[hex_rgb(ch["color"])] = ch["field"]
        except ValueError:
            continue
    out: dict[str, list[str]] = {}
    for e in emitters:
        try:
            field = by_rgb.get(hex_rgb(getattr(e, "color", "") or ""), chans[-1]["field"] if chans else "dye")
        except ValueError:
            field = chans[-1]["field"] if chans else "dye"
        name = str(getattr(e, "name", "") or "").strip()
        if not name or name.lower() in ("emitter", "smoke emitter"):
            kind = str(getattr(e, "kind", "") or "emitter").replace("_", " ").capitalize()
            word = colour_word(getattr(e, "color", "") or "")
            name = f"{kind}, {word}" if word else kind
        out.setdefault(field, []).append(name)
    return out


class SmokeLegend(QWidget):
    """One row per smoke channel: an on / off box, a chip in the colour the view draws, the emitters.

    Several emitters in one picture (a white rake, an amber rake, a blue
    point) cannot be told apart by colour alone once they mix, and one of them
    may hide the others: each row switches its channel on and off
    (:meth:`FlowScene.set_channel_visible`). The chip shows the colour as
    drawn, so white smoke reads as the theme grey it is drawn in on the light
    background. Bind it to a :class:`FlowScene` and call :meth:`refresh` after
    the scene's smoke changed.
    """

    toggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout
        self._flow = None
        self._rows: dict[str, tuple] = {}
        self._key: tuple | None = None
        lay = self._lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        theme = lazy_import("aerolume.app.theme")
        sig_ = getattr(theme, "changed", None) if theme is not None else None
        if sig_ is not None:
            # a bound method (not a lambda holding the widget): Qt drops the connection with the legend, where
            # the lambda outlived it and raised "Internal C++ object (SmokeLegend) already deleted" per switch
            sig_.connect(self._on_theme)

    def _on_theme(self, *_args) -> None:
        QTimer.singleShot(0, self, self._recolour)

    def bind(self, flow) -> None:
        self._flow = flow
        self.refresh()

    def rows(self) -> list[str]:
        """Channel fields with a row, in order (for tests)."""
        return list(self._rows)

    def refresh(self) -> None:
        """Rebuild the rows when the channels changed; else update their colours and states."""
        from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel
        items = self._flow.smoke_legend() if self._flow is not None else []
        key = tuple((it["field"], tuple(it["names"]), it["empty"]) for it in items)
        if key != self._key:
            self._key = key
            while self._lay.count():
                it = self._lay.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
            self._rows = {}
            for it in items:
                row = QWidget()
                hl = QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(6)
                box = QCheckBox()
                box.setChecked(bool(it["visible"]))
                # a round legend dot, not a square: a 14 px light square beside the checkbox read as a
                # second, unchecked checkbox on the dark theme (restyle critic round 2)
                chip = QLabel()
                chip.setFixedSize(12, 12)
                chip.setAccessibleName("Smoke colour")
                names = it["names"] or [f"Channel {len(self._rows) + 1}"]
                text = names[0] + (f" + {len(names) - 1} more" if len(names) > 1 else "")
                text += " (no smoke yet)" if it["empty"] else ""
                lab = label(text, "secondary", wrap=True)
                lab.setToolTip("\n".join(names))
                hl.addWidget(box)
                hl.addWidget(chip)
                hl.addWidget(lab, 1)
                box.toggled.connect(lambda on, f=it["field"]: self._on_toggle(f, on))
                self._lay.addWidget(row)
                self._rows[it["field"]] = (box, chip, lab, text)
            self.setVisible(bool(items))
        for it in items:
            box = self._rows[it["field"]][0]
            if box.isChecked() != bool(it["visible"]):
                box.blockSignals(True)
                box.setChecked(bool(it["visible"]))
                box.blockSignals(False)
        self._recolour()

    def _recolour(self) -> None:
        if self._flow is None:
            return
        tk = tokens()
        dark = bool(getattr(getattr(self._flow.scene, "theme", None), "dark", False))
        for field, (box, chip, lab, text) in self._rows.items():
            c = self._flow.drawn_smoke_colour(field)
            raw = self._flow.dye_colors.get(field, "auto")
            chip_qss = (f"background:{c}; border:1px solid {tk.get('border_strong', tk.get('border', '#888'))};"
                        "border-radius:6px;")
            if chip.styleSheet() != chip_qss:
                chip.setStyleSheet(chip_qss)
            tip = "Show or hide this smoke"
            # white smoke would vanish on the light background, so the view draws it dark: the legend says
            # so, instead of a near-black chip beside "Rake, white"
            shown_dark = raw == "auto" and not dark
            lab.setText(text + (" · drawn dark on the light background" if shown_dark else ""))
            if raw == "auto":
                tip += " (white smoke is drawn dark on the light background, white on the dark one)"
            box.setToolTip(tip)
            chip.setToolTip(tip)

    def _on_toggle(self, field: str, on: bool) -> None:
        if self._flow is not None:
            self._flow.set_channel_visible(field, on)
        self.toggled.emit(field, bool(on))


def smoke_channels(volume) -> list[str]:
    """The smoke channels a volume carries (``dye``, ``dye1`` .. ``dye3``), in channel order."""
    fields = getattr(volume, "fields", None) or {}
    return [c for c in DYE_CHANNELS if c in fields]


def add_smoke_layers(scene, volume, *, field: str = "dye", opacity: float = 1.0, color: str = "auto",
                     gain: float = 1.0, opacity_curve=None) -> list[str]:
    """Draw a dye volume as smoke; returns the layer handles.

    ``Scene.add_smoke`` (DESIGN 7.13: a GPU volume renderer) when the viewer
    has it, else :data:`SMOKE_SHELLS` iso-surfaces of the concentration in the
    theme's smoke grey (or ``color``). Empty smoke (no dye anywhere yet) draws
    nothing. ``gain`` is the display gain of :func:`aerolume.viz3d.scene.smoke_gain`.
    """
    dye = np.asarray(volume.fields[field], dtype=np.float32)
    top = float(np.nanmax(dye)) if dye.size else 0.0
    if not top > 1e-6:
        return []
    fn = getattr(scene, "add_smoke", None)
    if callable(fn):
        kw = {"color": color} if color != "auto" and _accepts(fn, "color") else {}
        if gain != 1.0 and _accepts(fn, "gain"):
            kw["gain"] = float(gain)
        if opacity_curve is not None and _accepts(fn, "opacity_curve"):
            kw["opacity_curve"] = opacity_curve
        return [_call(fn, volume, field=field, opacity=opacity, **kw)]
    dark = bool(getattr(getattr(scene, "theme", None), "dark", False))
    rgb = SMOKE_RGB[dark] if color == "auto" else color
    try:
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(dye, SMOKE_SMOOTH_CELLS, mode="nearest")
    except ImportError:                                  # pragma: no cover - scipy is a dependency
        smooth = dye
    from ...model import VolumeData
    vol = VolumeData(origin=tuple(volume.origin), spacing=tuple(volume.spacing), fields={field: smooth},
                     mask=None, label=getattr(volume, "label", "instant"), mirrored=bool(getattr(volume, "mirrored", False)))
    top = float(np.nanmax(smooth)) or top
    out = []
    for frac, alpha in SMOKE_SHELLS:
        out.append(scene.add_isosurface(vol, field, frac * top, color=rgb, opacity=min(1.0, alpha * float(opacity))))
    return out


class FlowScene(QObject):
    """The flow-visualisation layers of one View3D (see the module docstring).

    Feed it data with ``set_bodies`` / ``set_volume`` / ``set_smoke`` /
    ``set_slice`` / ``set_rake`` (or several at once with :meth:`set_data`)
    and say what to show with :meth:`show`; each call rebuilds only the layers
    it affects and repaints the view. The object is a child of the view, so it
    lives and dies with it (and its theme connection with it).

    The rake is drawn through the view's ``set_seed_rake`` when it has one, so
    the user can drag it by its handle: ``rakeMoving(RakeSpec)`` reports it
    while dragged, ``rakeEdited(RakeSpec)`` on release (the streamlines are
    traced again from it then). Streamlines come from ``lines`` when a run
    traced them on its own thread (``LiveFrame.streamlines3d``), else they are
    traced here from the velocity volume.
    """

    rakeMoving = Signal(object)        # RakeSpec while the user drags the rake in the view
    rakeEdited = Signal(object)        # RakeSpec when the drag ends

    def __init__(self, view, *, shown: dict | None = None) -> None:
        super().__init__(view)
        self.view = view
        self.shown: dict[str, bool] = dict(DEFAULT_SHOWN_RESULT if shown is None else shown)
        self.handles: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}
        self.cost_ms: dict[str, float] = {}
        self.bodies: list = []
        self.ground: float | None = None
        self.volume = None             # u, v, w (+ q, mask): streamlines and vortex cores
        self.smoke = None              # a volume with "dye"
        self.slice = None              # SliceData already cropped for the 3D view
        self.slice_neutral = False     # a grey plane marking where the slice will be (no data yet)
        self.slice_field = "speed"
        self.slice_cmap: str | None = None
        self.slice_range: tuple[float, float] | None = None
        self.rake: RakeSpec | None = None
        self.rake_editable = True
        #: smoke is on its way (a run or explore that releases it): the rake is drawn before the first volume
        self.smoke_expected = False
        #: the rake is also where the smoke comes from (a run or explore releases it there), so it is
        #: drawn with smoke alone; False for a stored result, whose smoke snapshot is fixed and whose
        #: rake only seeds streamlines (it shows with them)
        self.rake_with_smoke = True
        self.lines = None              # streamlines traced elsewhere (a viz3d.flow.StreamlineSet)
        self.q_level: float | None = None
        self.q_color_by_slice = False
        self.caption = ""
        self.surface_field = "cp"
        self.size = 1.0                # the bodies' length scale, for the rake's thickness
        # placed physics (DESIGN.md 10)
        self.dye_colors: dict[str, str] = {}      # smoke channel -> drawn colour ("auto" = theme grey)
        self.channel_names: dict[str, list[str]] = {}   # smoke channel -> its emitters' names (the legend)
        self.hidden_channels: set[str] = set()    # smoke channels switched off in the legend
        self.smoke_gains: dict[str, tuple[float, float]] = {}   # channel -> (display gain, opacity factor)
        self.heat = None               # a volume with "temperature" (or "theta")
        self.heat_style = "plume"      # HEAT_STYLES key
        self.heat_level: float | None = None      # threshold [field units]; None = HEAT_THRESHOLD_SHARE
        self.heat_range: tuple[float, float] | None = None   # colour range; None = the data's display range
        self.heat_cmap: str | None = None
        self.particles: dict | None = None        # {"positions", "colors", "trails", "mirror_y"}
        self.devices: list = []        # viz3d.placed.Glyph
        self.room: tuple[float, ...] | None = None   # a closed domain's box, drawn as its edges
        self._body_key = None
        self._stream_thin_built = False   # the streamline layer on screen is thin lines (smoke shown with it)
        theme = lazy_import("aerolume.app.theme")
        sig_ = getattr(theme, "changed", None) if theme is not None else None
        if sig_ is not None:
            sig_.connect(self._on_theme)        # bound method of a QObject: dropped with this object
        for name, slot in (("rakeMoving", self._on_view_rake_moving), ("rakeChanged", self._on_view_rake)):
            signal = getattr(view, name, None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(slot)

    # -- data -------------------------------------------------------------
    @property
    def scene(self):
        return self.view.scene

    def set_bodies(self, bodies: Sequence, *, ground: float | None = None, key=None,
                   focus: Sequence[float] | None = None) -> None:
        """The solid bodies (Mesh3D or SurfaceData3D, wind frame); a no-op when ``key`` is unchanged."""
        if key is not None and key == self._body_key and ground == self.ground:
            return
        self._body_key = key
        self.bodies = [b for b in bodies if len(getattr(b, "faces", []))]
        self.ground = ground
        if self.bodies:
            pts = np.vstack([np.asarray(b.vertices, dtype=float) for b in self.bodies])
            ext = pts.max(axis=0) - pts.min(axis=0)
            self.size = float(max(ext.max(), 1e-6))
        if focus is not None:
            self.scene.set_focus(focus)
        self.rebuild("body", "rake")

    def body_bounds(self) -> tuple[float, ...] | None:
        return bounds_of(self.bodies)

    def set_focus(self, box: Sequence[float] | None) -> None:
        self.scene.set_focus(box)
        self.view.refresh()

    def set_volume(self, volume, *, rebuild: bool = True) -> None:
        """The velocity volume streamlines and vortex cores are traced in (mean or instantaneous)."""
        self.volume = volume
        if rebuild:
            self.rebuild("streamlines", "q")

    def set_smoke(self, volume, *, rebuild: bool = True) -> None:
        """A volume carrying smoke (``dye`` and / or ``dye1`` .. ``dye3``; None removes the smoke)."""
        self.smoke = volume if volume is not None and smoke_channels(volume) else None
        if rebuild:
            self.rebuild("smoke")

    def set_dye_colors(self, colors: dict[str, str] | None, *, rebuild: bool = True) -> None:
        """Colours of the smoke channels ({"dye1": "#e8a33d", ...}; :func:`dye_colours_of` of the run)."""
        new = {str(k): str(v) for k, v in (colors or {}).items()}
        if new == self.dye_colors:
            return
        self.dye_colors = new
        if rebuild:
            self.rebuild("smoke")

    def set_heat(self, volume, *, style: str | None = None, level=_KEEP, rng=_KEEP, cmap=_KEEP,
                 rebuild: bool = True) -> None:
        """The temperature volume of the thermal plume (None removes it) and how to draw it.

        ``style`` "plume" (a translucent volume, clear below ``level``) or
        "surface" (one iso-surface at ``level``); ``level`` in the field's
        units, None for :data:`HEAT_THRESHOLD_SHARE` of the colour range;
        ``rng`` the colour range (None: the data's display range; a live run
        passes a fixed one so the plume's colours do not drift frame to frame).
        A live update of the same style is applied in place (no flicker).
        """
        self.heat = volume if volume is not None and temperature_field_of(volume) else None
        if style is not None:
            if style not in {k for k, _t, _tip in HEAT_STYLES}:
                raise ValueError(f"unknown heat style {style!r}")
            restyle = style != self.heat_style
            self.heat_style = style
        else:
            restyle = False
        changed = restyle
        if level is not _KEEP and level != self.heat_level:
            self.heat_level, changed = level, True
        if rng is not _KEEP and rng != self.heat_range:
            self.heat_range, changed = (None if rng is None else (float(rng[0]), float(rng[1]))), True
        if cmap is not _KEEP and cmap != self.heat_cmap:
            self.heat_cmap, changed = cmap, True
        if rebuild and (changed or not self._update_in_place("heat")):
            self.rebuild("heat")

    def heat_colour_range(self) -> tuple[float, float] | None:
        """The plume's colour range: the one set, else :func:`heat_display_range` of the fluid's temperature."""
        if self.heat_range is not None:
            return self.heat_range
        vol = self.heat
        f = temperature_field_of(vol)
        if f is None:
            return None
        vals = np.asarray(vol.fields[f], dtype=float)
        if getattr(vol, "mask", None) is not None:
            vals = vals[np.asarray(vol.mask) <= 0.5]
        lo, hi, _mode = heat_display_range(vals)
        return lo, hi

    def heat_mode(self) -> str:
        """What the plume keeps clear: "high" (a warm plume in ambient fluid), "low" (a cold one) or
        "both" (heating and cooling); from where the ambient median sits in the colour range."""
        vol = self.heat
        f = temperature_field_of(vol)
        if f is None:
            return "high"
        vals = np.asarray(vol.fields[f], dtype=float)
        vals = vals[np.isfinite(vals)]
        rng = self.heat_colour_range()
        if rng is None or not vals.size:
            return "high"
        amb, span = float(np.median(vals)), max(rng[1] - rng[0], 1e-12)
        below, above = (amb - rng[0]) > 0.25 * span, (rng[1] - amb) > 0.25 * span
        return "both" if below and above else "low" if below else "high"

    def heat_threshold(self) -> float | None:
        """The threshold the plume / iso-surface is drawn at (field units)."""
        if self.heat_level is not None and _finite(self.heat_level):
            return float(self.heat_level)
        rng = self.heat_colour_range()
        return None if rng is None else rng[0] + HEAT_THRESHOLD_SHARE * (rng[1] - rng[0])

    def set_particles(self, positions, *, colors=None, trails=None, mirror_y: bool = False,
                      rebuild: bool = True) -> None:
        """Tracer particles (N, 3) metres, with per-particle colours (N, 3) uint8 and optional trails
        (N, K, 3); None removes them. Applied in place when the layer exists (a live run moves its
        particles every frame without rebuilding an actor)."""
        if positions is None or len(np.asarray(positions).reshape(-1, 3)) == 0:
            had = self.particles is not None
            self.particles = None
            if rebuild and had:
                self.rebuild("particles")
            return
        self.particles = {"positions": np.asarray(positions, dtype=float).reshape(-1, 3),
                          "colors": None if colors is None else np.asarray(colors, dtype=np.uint8),
                          "trails": trails, "mirror_y": bool(mirror_y)}
        if rebuild and not self._update_in_place("particles"):
            self.rebuild("particles")

    def set_room(self, box: Sequence[float] | None, *, rebuild: bool = True) -> None:
        """The box of a closed domain (x0, x1, y0, y1, z0, z1), drawn as its twelve edges; None for a
        tunnel. Simulate and Results draw it whenever ``domain_mode == "closed"``, as Setup does."""
        try:
            b = tuple(float(v) for v in box) if box is not None else None
        except (TypeError, ValueError):
            b = None
        if b is not None and (len(b) != 6 or not all(math.isfinite(v) for v in b)):
            b = None
        if b == self.room:
            return
        self.room = b
        if rebuild:
            self.rebuild("room")

    def set_devices(self, glyphs: Sequence, *, rebuild: bool = True) -> None:
        """The fans, heaters, emitters, jets and vents to draw (:func:`aerolume.viz3d.placed.placed_glyphs`)."""
        self.devices = list(glyphs or [])
        if rebuild:
            self.rebuild("devices")

    def set_slice(self, sd, *, field: str | None = None, cmap: str | None = None,
                  rng: tuple[float, float] | None = None, neutral: bool = False, rebuild: bool = True) -> None:
        """The slice drawn in 3D (already cropped); ``neutral`` marks a plain grey plane without data."""
        self.slice = sd
        self.slice_neutral = bool(neutral)
        if field:
            self.slice_field = field
        self.slice_cmap = cmap
        self.slice_range = rng
        if rebuild:
            # streamlines and vortex cores share the slice's colours: recoloured in place when
            # only the map or range changed (a live slice every frame must not re-trace lines)
            self.rebuild("slice", *self._recolour("streamlines", "q"))

    def set_rake(self, spec: RakeSpec | None, *, rebuild: bool = True) -> None:
        self.rake = spec
        if rebuild:
            self.rebuild("streamlines", "rake")

    def set_q(self, level: float | None, *, color_by_slice: bool = False, rebuild: bool = True) -> None:
        self.q_level, self.q_color_by_slice = level, bool(color_by_slice)
        if rebuild:
            self.rebuild("q")

    def set_data(self, *, bodies=_KEEP, body_key=None, ground=_KEEP, focus=None, volume=_KEEP, smoke=_KEEP,
                 slice=_KEEP, field: str | None = None, cmap: str | None = None, rng=None, neutral: bool = False,
                 rake=_KEEP, lines=_KEEP) -> list[str]:
        """Update several inputs at once and rebuild only the layers whose input really changed.

        Identity decides for the arrays (the same volume object is not traced
        again), equality for the rake, the body key for the bodies. A static
        result redrawn after every slider move therefore re-traces nothing;
        only the slice is replaced and the lines are recoloured. Returns the
        layers that were rebuilt.
        """
        layers: set[str] = set()
        if bodies is not _KEEP:
            g = self.ground if ground is _KEEP else ground
            if body_key is None or body_key != self._body_key or g != self.ground:
                self._body_key = body_key
                self.bodies = [b for b in bodies if len(getattr(b, "faces", []))]
                self.ground = g
                if self.bodies:
                    pts = np.vstack([np.asarray(b.vertices, dtype=float) for b in self.bodies])
                    self.size = float(max((pts.max(axis=0) - pts.min(axis=0)).max(), 1e-6))
                layers |= {"body", "rake"}
                if focus is not None:
                    self.scene.set_focus(focus)
        if lines is not _KEEP and lines is not self.lines:
            # traced lines and a volume traced here are different layers (a volume is traced
            # forward from the rake; a layer made from lines would re-trace both ways)
            switch = (lines is None) != (self.lines is None)
            self.lines = lines
            if switch or not self._update_in_place("streamlines"):
                layers.add("streamlines")
        if volume is not _KEEP and volume is not self.volume:
            self.volume = volume
            layers.add("q")
            if self.lines is None and "streamlines" not in layers and not self._update_in_place("streamlines"):
                layers.add("streamlines")
        if smoke is not _KEEP:
            sm = smoke if smoke is not None and smoke_channels(smoke) else None
            if sm is not self.smoke:
                had = self.smoke is not None
                self.smoke = sm
                if not (had and sm is not None and self._update_in_place("smoke")):
                    layers |= {"smoke", "rake", "slice"}
        if rake is not _KEEP and rake != self.rake:
            self.rake = rake
            layers |= {"streamlines", "rake"}
        if slice is not _KEEP:
            self.slice, self.slice_neutral = slice, bool(neutral)
            if field:
                self.slice_field = field
            self.slice_cmap, self.slice_range = cmap, rng
            layers.add("slice")
            layers |= set(self._recolour(*(k for k in ("streamlines", "q") if k not in layers)))
        if layers:
            self.rebuild(*layers)
        return sorted(layers)

    def _update_in_place(self, key: str) -> bool:
        """Swap the data of an existing smoke / streamline layer without rebuilding it.

        ``Scene.update_smoke`` / ``update_streamlines`` keep the actor, the
        transfer function and the colour range (a live run's smoke and lines do
        not flash or change colour from frame to frame). Returns False when the
        layer has to be built instead (none yet, hidden, other colouring, or a
        viewer without the update calls).
        """
        hs = self.handles.get(key) or []
        if not hs or not self.is_shown(key) or (len(hs) != 1 and key not in ("smoke", "particles")):
            return False
        sc = self.scene
        t0 = time.perf_counter()
        try:
            if key == "smoke":
                fn = getattr(sc, "update_smoke", None)
                info = getattr(sc, "layer_info", None)
                if fn is None or any(sc.layers().get(h) != "smoke" for h in hs):
                    return False
                # one layer per channel: the same channels (with smoke in them) must be on screen
                fields = [info(h).get("field", "dye") if callable(info) else "dye" for h in hs]
                if fields != self._smoke_fields_to_draw():
                    return False
                for h, f in zip(hs, fields):
                    # the display gain follows the data only when it moved by half again: re-scaling a
                    # live run's smoke every frame would make it pulse as it develops
                    old_g = self.smoke_gains.get(f, (1.0, 1.0))
                    g = self._channel_gain(f)
                    if max(g[0] / old_g[0], old_g[0] / g[0]) > 1.5 and _accepts(fn, "gain"):
                        self.smoke_gains[f] = (g[0], old_g[1])
                        fn(h, self.smoke, f, gain=g[0])
                    else:
                        fn(h, self.smoke, f)
            elif key == "heat":
                fn = getattr(sc, "update_scalar_volume", None)
                if fn is None or self.heat_style != "plume" or sc.layers().get(hs[0]) != "volume":
                    return False
                fn(hs[0], self.heat)
            elif key == "particles":
                fn = getattr(sc, "update_particles", None)
                p = self.particles
                if fn is None or p is None or sc.layers().get(hs[0]) != "particles":
                    return False
                fn(hs[0], p["positions"], colors=p["colors"], trails=p["trails"])
            elif key == "streamlines":
                if self._stream_style_stale():
                    return False                 # tubes <-> thin lines: a rebuild, not an update
                fn = getattr(sc, "update_streamlines", None)
                info = sc.scalar_info(hs[0])
                if fn is None or (info["field"] if info else None) != self._stream_colours()[0]:
                    return False
                if self.lines is not None:
                    fn(hs[0], self.lines)
                elif self.rake is not None:
                    fn(hs[0], self.volume, seeds=self.rake.clamped(volume_box(self.volume)).seeds())
                else:
                    return False
            else:
                return False
        except Exception:
            log.debug("in-place update of %s failed", key, exc_info=True)
            return False
        self.cost_ms[key] = 1000.0 * (time.perf_counter() - t0)
        self.view.refresh()
        return True

    def set_caption(self, text: str) -> None:
        self.caption = text or ""
        self.scene.set_caption(self.caption)
        self.view.refresh()

    def show(self, key: str, on: bool) -> None:
        """Show or hide one layer ("smoke", "streamlines", "slice", "q", "surface")."""
        if self.shown.get(key) == bool(on):
            return
        self.shown[key] = bool(on)
        if key == "slice":
            self.rebuild("slice", *self._recolour("streamlines", "q"))
            return
        affected = {"surface": ("body",), "smoke": ("smoke", "slice", "rake"),
                    "streamlines": ("streamlines", "rake"), "heat": ("heat", "slice")}.get(key, (key,))
        self.rebuild(*affected)

    def _recolour(self, *keys: str) -> list[str]:
        """Bring layers to the colours they should have now; returns those that need a rebuild.

        A layer coloured by the same field only gets its map and range swapped
        (cheap); one whose colouring field changed has to be traced again.
        """
        todo: list[str] = []
        from .field_view import default_cmap3d
        for key in keys:
            hs = self.handles.get(key) or []
            if not hs:
                if self.is_shown(key):
                    todo.append(key)
                continue
            field, cmap, rng = self._colours_for(key)
            info = self.scene.scalar_info(hs[0])
            if (info["field"] if info else None) != field:
                todo.append(key)
                continue
            if field is None:
                continue
            want = cmap or default_cmap3d(field)
            for h in hs:
                if info["cmap"] != want:
                    self.scene.set_cmap(h, want)
                if rng is not None:
                    self.scene.set_scalar_range(h, float(rng[0]), float(rng[1]))
        if keys and not todo:
            self._colorbar()
            self.view.refresh()
        return todo

    # -- availability -------------------------------------------------------
    def available(self, key: str) -> tuple[bool, str]:
        """Whether a layer has data to draw, and if not, why (for the toggle's tooltip)."""
        if key == "smoke":
            return (self.smoke is not None, "no smoke (dye) volume")
        if key == "streamlines":
            return (self.lines is not None or has_velocity(self.volume), "no 3D velocity volume")
        if key == "slice":
            return (self.slice is not None, "no slice")
        if key == "q":
            ok = self.volume is not None and "q" in self.volume.fields and _finite(self.q_level)
            return (ok, "the stored volume has no Q-criterion")
        if key == "surface":
            ok = any(getattr(b, self.surface_field, None) is not None for b in self.bodies)
            return (ok, "surface pressure is stored with the result when the run ends")
        if key == "heat":
            return (self.heat is not None, "no temperature field: heat transfer is off for this run")
        if key == "particles":
            return (self.particles is not None, "no tracer particles: the run has no particle emitters")
        if key == "devices":
            return (bool(self.devices), "no fans, heaters, emitters, jets or vents")
        return (True, "")

    def is_shown(self, key: str) -> bool:
        return bool(self.shown.get(key)) and self.available(key)[0]

    # -- building -----------------------------------------------------------
    def rebuild(self, *keys: str) -> None:
        """Rebuild the given layers (all when none are given), then the colour bar; repaints the view.

        Streamlines join any rebuild that changes whether smoke is shown, since
        they switch between tubes and thin translucent lines with it.
        """
        stale = self._stream_style_stale()
        todo = [k for k in _BUILD_ORDER if not keys or k in keys or (k == "streamlines" and stale)]
        for key in todo:
            t0 = time.perf_counter()
            for h in self.handles.pop(key, []):
                self.scene.remove(h)
            self.errors.pop(key, None)
            try:
                handles = getattr(self, f"_build_{key}")()
            except Exception as exc:
                log.exception("3D %s layer failed", key)
                self.errors[key] = f"{type(exc).__name__}: {exc}"
                handles = []
            if handles:
                self.handles[key] = [h for h in handles if h]
            self.cost_ms[key] = 1000.0 * (time.perf_counter() - t0)
        self._colorbar()
        self.view.refresh()

    def layer_handles(self) -> dict[str, list[str]]:
        """Scene handles of the flow layers, by layer ("rake" included when the view draws it)."""
        out = {k: list(v) for k, v in self.handles.items()}
        if "rake" not in out:
            rakes = [h for h, kind in self.scene.layers().items() if kind == "rake"]
            if rakes:
                out["rake"] = rakes
        return out

    def _build_body(self) -> list[str]:
        if not self.bodies:
            return []
        sc = self.scene
        out: list[str] = []
        field = self.surface_field
        coloured = [b for b in self.bodies if getattr(b, field, None) is not None] if self.shown.get("surface") \
            else []
        plain = [b for b in self.bodies if not any(b is c for c in coloured)]
        if coloured:
            from .results3d import surface_cp_range
            vals = np.concatenate([np.asarray(getattr(b, field), float) for b in coloured])
            from .field_view import default_cmap3d, display_range3d
            rng = surface_cp_range(vals) if field == "cp" else display_range3d(field, vals)
            out += sc.add_meshes(coloured, field, cmap=default_cmap3d(field), range=rng)
        for b in plain:
            out.append(sc.add_mesh(b))
        if self.ground is not None:
            out.append(sc.add_ground(float(self.ground)))
        return out

    def _build_slice(self) -> list[str]:
        if not self.is_shown("slice"):
            return []
        sd = self.slice
        if self.slice_neutral:
            f = next(iter(sd.fields))
            return [self.scene.add_slice(sd, field=f, cmap="grayscale", range=(0.0, 1.0), opacity=0.35)]
        field = self.slice_field if self.slice_field in sd.fields else next(iter(sd.fields))
        from .field_view import default_cmap3d
        opacity = SLICE_OPACITY_WITH_SMOKE if self.is_shown("smoke") else 1.0
        return [self.scene.add_slice(sd, field=field, cmap=self.slice_cmap or default_cmap3d(field),
                                     range=self.slice_range, opacity=opacity)]

    def _stream_colours(self) -> tuple[str | None, str | None, tuple[float, float] | None]:
        """(field, map, range) of streamlines: the chosen field when the volume has it, else speed.

        With the field's colours the slice's colour bar reads the lines too; a
        field the volume lacks (a live volume carries only velocity and dye)
        leaves the lines in one neutral colour while the slice owns the bar.
        """
        vol = self.volume
        f = self.slice_field
        if self.slice_neutral:
            return "speed", None, None
        if self.lines is not None:                      # traced elsewhere: the fields they carry
            if f in getattr(self.lines, "values", {}):
                return f, self.slice_cmap, self.slice_range
            return ("speed", None, None) if not self.is_shown("slice") else (None, None, None)
        if vol is not None and (f in vol.fields or (f == "speed" and has_velocity(vol))):
            return f, self.slice_cmap, self.slice_range
        if self.is_shown("slice"):
            return None, None, None
        return "speed", None, None

    def _colours_for(self, key: str) -> tuple[str | None, str | None, tuple[float, float] | None]:
        if key == "streamlines":
            return self._stream_colours()
        if key == "q" and self.q_color_by_slice:
            return self._stream_colours()
        return None, None, None

    def _stream_thin(self) -> bool:
        """Whether streamlines are drawn as thin translucent lines: they are when smoke is shown with them."""
        return self.is_shown("smoke")

    def _stream_style_stale(self) -> bool:
        """The streamlines on screen were built for the other style (smoke appeared or went away)."""
        return bool(self.handles.get("streamlines")) and self._stream_thin_built != self._stream_thin()

    def _build_streamlines(self) -> list[str]:
        if not self.is_shown("streamlines"):
            return []
        thin = self._stream_thin()
        field, cmap, rng = self._stream_colours()
        if self.lines is not None:
            h = self.scene.add_streamlines(self.lines, color_by=field, cmap=cmap, range=rng, tubes=not thin)
        elif self.rake is None or not has_velocity(self.volume):
            return []
        else:
            seeds = self.rake.clamped(volume_box(self.volume)).seeds()
            h = self.scene.add_streamlines(self.volume, seeds, direction="forward", color_by=field, cmap=cmap,
                                           range=rng, tubes=not thin)
        self._stream_thin_built = thin
        if thin:
            _set_layer_opacity(self.scene, h, STREAMLINE_OPACITY_WITH_SMOKE)
        return [h]

    def _build_q(self) -> list[str]:
        if not self.is_shown("q"):
            return []
        field, cmap, rng = self._colours_for("q")
        if field:
            return [self.scene.add_isosurface(self.volume, "q", float(self.q_level), color_by=field, cmap=cmap,
                                              range=rng)]
        return [self.scene.add_isosurface(self.volume, "q", float(self.q_level))]

    def _smoke_fields_to_draw(self) -> list[str]:
        """The smoke channels with smoke in them and not switched off, in channel order (empty channels
        draw nothing)."""
        sm = self.smoke
        out = []
        for ch in smoke_channels(sm):
            if ch in self.hidden_channels:
                continue
            a = np.asarray(sm.fields[ch])
            if a.size and float(np.nanmax(a)) > 1e-6:
                out.append(ch)
        return out

    def _channel_gain(self, ch: str) -> tuple[float, float]:
        """(concentration gain, opacity factor) of one channel (:func:`aerolume.viz3d.scene.smoke_gain`)."""
        try:
            from ...viz3d.scene import smoke_gain
        except ImportError:                              # pragma: no cover - the package ships it
            return 1.0, 1.0
        return smoke_gain(self.smoke.fields[ch])

    def _build_smoke(self) -> list[str]:
        """One smoke layer per channel, each in its emitter's colour (the theme grey for white smoke),
        each lifted to a visible density by its own display gain (a point emitter's thin stream)."""
        if not self.is_shown("smoke"):
            return []
        out: list[str] = []
        self.smoke_gains = {}
        for ch in self._smoke_fields_to_draw():
            gain, boost = self._channel_gain(ch)
            self.smoke_gains[ch] = (gain, boost)
            curve = None
            if boost > 1.2:                    # a sparse thread (a point emitter): the concave curve
                from ...viz3d.scene import SMOKE_SPARSE_CURVE
                curve = SMOKE_SPARSE_CURVE
            out += add_smoke_layers(self.scene, self.smoke, field=ch, color=self.dye_colors.get(ch, "auto"),
                                    opacity=boost, gain=gain, opacity_curve=curve)
        return out

    # -- smoke legend (one row per channel: its colour, its emitters, on / off) ----------------
    def smoke_legend(self) -> list[dict]:
        """The smoke channels of the data on screen: ``[{"field", "color" (as drawn, hex), "names",
        "visible", "empty"}]``, in channel order. The colour is the one the view draws, so the theme grey
        stands in for white smoke on the light background; names come from :meth:`set_channel_names`."""
        sm = self.smoke
        if sm is None:
            return []
        out = []
        for ch in smoke_channels(sm):
            a = np.asarray(sm.fields[ch])
            empty = not (a.size and float(np.nanmax(a)) > 1e-6)
            out.append({"field": ch, "color": self.drawn_smoke_colour(ch),
                        "names": list(self.channel_names.get(ch, [])), "visible": ch not in self.hidden_channels,
                        "empty": empty})
        return out

    def drawn_smoke_colour(self, ch: str) -> str:
        """Hex colour channel ``ch`` is drawn in now (its emitter's, or the theme's smoke grey)."""
        c = self.dye_colors.get(ch, "auto")
        if c and c != "auto":
            try:
                # the scene draws a coloured channel at full saturation (viz3d.scene.smoke_display_rgb)
                from ...viz.colormaps import hex_rgb
                from ...viz3d.scene import smoke_display_rgb
                rgb = smoke_display_rgb([v / 255.0 for v in hex_rgb(c)])
                return "#{:02x}{:02x}{:02x}".format(*(round(255 * float(v)) for v in rgb))
            except Exception:  # noqa: BLE001 - the emitter's own colour is a fine legend swatch
                return c
        th = getattr(self.scene, "theme", None)
        rgb = getattr(th, "smoke", None) or SMOKE_RGB[bool(getattr(th, "dark", False))]
        return "#{:02x}{:02x}{:02x}".format(*(round(255 * float(v)) for v in rgb))

    def set_channel_names(self, names: dict[str, Sequence[str]] | None) -> None:
        """Emitter names per smoke channel ({"dye": ["Rake 1"], ...}) for :meth:`smoke_legend`."""
        self.channel_names = {str(k): [str(n) for n in v] for k, v in (names or {}).items()}

    def set_channel_visible(self, ch: str, on: bool) -> None:
        """Show or hide one smoke channel (the others keep their layers' look)."""
        on = bool(on)
        if on == (ch not in self.hidden_channels):
            return
        if on:
            self.hidden_channels.discard(ch)
        else:
            self.hidden_channels.add(ch)
        self.rebuild("smoke")

    def channel_visible(self, ch: str) -> bool:
        return ch not in self.hidden_channels

    def _build_heat(self) -> list[str]:
        """The thermal plume: a colour-mapped translucent volume, or one iso-surface at the threshold."""
        if not self.is_shown("heat"):
            return []
        sc, vol = self.scene, self.heat
        f = temperature_field_of(vol)
        rng = self.heat_colour_range()
        level = self.heat_threshold()
        if f is None or rng is None or level is None:
            return []
        from .field_view import default_cmap3d
        cmap = self.heat_cmap or default_cmap3d(f)
        lo, hi = rng
        span = max(hi - lo, 1e-12)
        mode = self.heat_mode()
        if self.heat_style == "surface" or not callable(getattr(sc, "add_scalar_volume", None)):
            vals = np.asarray(vol.fields[f], dtype=float)
            if not (np.nanmin(vals) < level < np.nanmax(vals)):
                return []
            from ...viz3d.colors import get_lut
            lut = get_lut(cmap)
            rgb = lut[round(255 * min(max((level - lo) / span, 0.0), 1.0))] / 255.0
            # a heated body's solid counts as hot (a cooled one's as cold): the surface then closes around
            # the thermal boundary layer instead of drawing the solid's voxels
            return [_call(sc.add_isosurface, vol, f, float(level), color=tuple(float(c) for c in rgb),
                          opacity=HEAT_SURFACE_OPACITY, solid="below" if mode == "low" else "above")]
        if mode == "high":
            floor = (level - lo) / span
        elif mode == "low":
            floor = (hi - level) / span if self.heat_level is not None else HEAT_THRESHOLD_SHARE
        else:
            floor = HEAT_THRESHOLD_SHARE
        return [sc.add_scalar_volume(vol, f, cmap=cmap, range=(lo, hi), floor=min(max(floor, 0.0), 0.95),
                                     mode=mode)]

    def _build_particles(self) -> list[str]:
        p = self.particles
        if not self.is_shown("particles") or p is None:
            return []
        fn = getattr(self.scene, "add_particles", None)
        if fn is None:
            return []
        return [fn(p["positions"], colors=p["colors"], trails=p["trails"], mirror_y=p["mirror_y"])]

    def _glyph_colour(self, g) -> str:
        """A device's colour: accent (fans, ports), warn (heaters), the smoke's own colour for an emitter
        (the theme's ink when that is white), muted when the device is switched off. ``tokens()`` is the
        theme's table merged over the fallback one, so every key read here is present."""
        tk = tokens()
        if not getattr(g, "enabled", True):
            return tk["muted"]
        if g.role == "warn":
            return tk["warn"]
        if g.role == "own":
            c = smoke_colour(getattr(g, "color", ""))
            if c != "auto":
                return c
            # "auto" smoke: light ink on the dark scene, the secondary ink on the light one (v3's
            # blue-grey literals had a cool cast beside the v4 neutral chrome)
            dark = bool(getattr(getattr(self.scene, "theme", None), "dark", False))
            return tk["text"] if dark else tk["text2"]
        return tk["accent"]

    def _build_room(self) -> list[str]:
        sc = self.scene
        fn = getattr(sc, "add_box_outline", None) if sc is not None else None
        if self.room is None or not callable(fn):
            return []
        return [fn(self.room)]

    def _build_devices(self) -> list[str]:
        if not self.is_shown("devices"):
            return []
        fn = getattr(self.scene, "add_glyph", None)
        out: list[str] = []
        for g in self.devices:
            col = self._glyph_colour(g)
            k = 1.0 if getattr(g, "enabled", True) else DISABLED_GLYPH_OPACITY
            for v, f, a in g.parts:
                if fn is not None:
                    out.append(fn((v, f), color=col, opacity=a * k, name=g.name))
                else:
                    out.append(self.scene.add_mesh((v, f), color=col, opacity=a * k, smooth=False))
        return out

    def _build_rake(self) -> list[str]:
        # with smoke wanted the rake shows where it is (or will be) released, before the first volume too
        show = self.rake is not None and (self.is_shown("streamlines") or (
            self.rake_with_smoke and self.shown.get("smoke") and (self.smoke is not None or self.smoke_expected)))
        setter = getattr(self.view, "set_seed_rake", None)
        sr = self.rake.to_seed_rake() if show else None
        if setter is not None and (sr is not None or not show):
            # the view keeps its own rake layer (updated in place, draggable by its handle);
            # it is not in self.handles, so a rebuild never removes it under the user's mouse
            setter(sr, editable=self.rake_editable)
            return []
        if not show:
            return []
        fn = getattr(self.scene, "add_seed_rake", None)
        if fn is not None and sr is not None:
            return [_call(fn, sr, nozzles=True, handle_marker=False)]
        v, f = rake_mesh(self.rake, self.size)
        return [self.scene.add_mesh((v, f), color=tokens()["accent"], smooth=False)]

    def _on_view_rake_moving(self, d) -> None:
        spec = self._rake_from_view(d)
        if spec is not None:
            self.rakeMoving.emit(spec)

    def _on_view_rake(self, d) -> None:
        """The user dropped the rake somewhere else in the view: trace from there, tell the page."""
        spec = self._rake_from_view(d)
        if spec is None:
            return
        self.rake = spec
        self.rebuild("streamlines")
        self.rakeEdited.emit(spec)

    def _rake_from_view(self, d) -> RakeSpec | None:
        """The rake at the place the view reports (a drag moves it: kind, counts and extras stay)."""
        spec = RakeSpec.from_view(d) if isinstance(d, dict) else None
        if spec is None and d is not None and hasattr(d, "spec"):
            spec = RakeSpec.from_view(d.spec())
        if spec is not None and self.rake is not None:
            spec = self.rake.moved_to(spec)
        return spec

    #: Which layer's colour map gets the colour bar, first come first: the slice reads every field; the
    #: plume (temperature) comes next, as it is the picture a heat-transfer run is about; then the
    #: streamlines, the surface and the vortex cores.
    COLORBAR_ORDER = ("slice", "heat", "streamlines", "body", "q")

    def colorbar_layers(self) -> list[tuple[str, str]]:
        """(layer key, scene handle) of the colour maps that get a bar: the first in
        :data:`COLORBAR_ORDER`, and a second one when another layer on screen shows a different field
        or map (the plume in the thermal map with streamlines coloured by speed: without a second bar
        the streamlines' colours were read off the temperature scale)."""
        sc = self.scene
        out: list[tuple[str, str, tuple]] = []
        for key in self.COLORBAR_ORDER:
            if key == "slice" and self.slice_neutral:
                continue
            for h in self.handles.get(key, []):
                info = sc.scalar_info(h)
                if info is None:
                    continue
                sig_ = (info["field"], info["cmap"], info["vmin"], info["vmax"])
                if all(sig_ != s for _k, _h, s in out):
                    out.append((key, h, sig_))
                break
            if len(out) == 2:
                break
        return [(k, h) for k, h, _s in out]

    def _colorbar(self) -> None:
        sc = self.scene
        picked = self.colorbar_layers()
        if not picked:
            sc.remove_colorbar()
            return
        for i, (key, h) in enumerate(picked):
            title = units = None
            if key == "body" and self.surface_field == "cp":
                title, units = "Surface pressure coefficient Cp", ""
            if i == 0:
                sc.add_colorbar(title, units, layer=h)
            elif _accepts(sc.add_colorbar, "second"):          # a scene without second bars keeps one
                sc.add_colorbar(title, units, layer=h, second=True)
        if len(picked) < 2 and _accepts(getattr(sc, "remove_colorbar", None), "second"):
            sc.remove_colorbar(second=True)

    def _on_theme(self, *_args) -> None:
        # the rake (accent), the device glyphs (accent / warn / smoke grey) and the fallback smoke
        # (grey per background) carry theme colours
        try:
            self.rebuild("rake", "smoke", "devices")
        except RuntimeError:            # the view is being destroyed
            pass

    # -- facts ----------------------------------------------------------------
    def describe(self) -> str:
        """What is on screen, for captions and tests: "smoke · 24 streamlines · slice y = 0 m"."""
        parts = []
        if self.is_shown("smoke"):
            n = len(self._smoke_fields_to_draw()) if self.smoke is not None else 0
            parts.append("smoke" if n <= 1 else f"smoke in {n} colours")
        if self.is_shown("streamlines") and self.rake is not None:
            parts.append(f"{len(self.rake.seeds())} streamlines")
        if self.is_shown("slice") and self.slice is not None:
            parts.append(f"slice {self.slice.axis} = {sig(float(self.slice.position), 3)} m")
        if self.is_shown("q"):
            parts.append("vortex cores")
        if self.is_shown("heat"):
            t = self.heat_threshold()
            unit = "°C" if temperature_field_of(self.heat) == "temperature" else ""
            what = "thermal plume" if self.heat_style == "plume" else "iso-surface"
            parts.append(f"{what} T ≥ {sig(t, 3)} {unit}".rstrip() if t is not None and self.heat_style == "plume"
                         else f"{what} T = {sig(t, 3)} {unit}".rstrip() if t is not None else what)
        if self.is_shown("particles") and self.particles is not None:
            parts.append(f"{len(self.particles['positions'])} particles")
        return " · ".join(parts)


def flow_view(parent: QWidget | None = None, **kw):
    """A View3D (imported here: the module loads VTK only when its first scene is made)."""
    from .view3d import View3D
    return View3D(parent, **kw)
