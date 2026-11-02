"""Colours for the 3D viewer: the 2D colour maps, 3D field metadata and theme palettes.

The 3D viewer must paint a Cp of -0.5 in exactly the colour the 2D Cp plot, the
live field view and the PDF report use, otherwise a designer comparing a slice
with the surface map sees two different stories. So this module does not define
colour maps: it takes the 256-entry LUTs of :mod:`aerolume.viz.colormaps` and
only turns them into what VTK needs (:func:`lookup_table`) or applies them on the
CPU for textures (:func:`colorize`).

What *is* 3D-specific lives here too:

* :data:`FIELD_INFO_3D` -- labels, SI units and the diverging flag of the fields a
  3D result carries. 3D results are stored in SI already (DESIGN.md 7.7), so no
  non-dimensional scaling is involved, and two fields change meaning from 2D:
  ``vorticity`` is the magnitude |omega| (not the signed omega_z, so it is not
  diverging) and ``v`` is the lateral velocity; ``w``, ``q``, ``cf`` and
  ``sensitivity`` are new.
* :func:`display_range` -- one range rule: robust percentiles (a single bad cell
  must not wash out the map), speed from exactly 0, and diverging fields
  symmetric about 0 so white / grey always means "zero".
* :class:`SceneTheme` -- background, text and ground colours derived from the
  app's theme tokens (docs/ui_guidelines.md section 2). The defaults copy the
  token table so this module stays free of Qt; the widget passes the live
  tokens.

Only numpy is imported at module level; VTK is imported inside
:func:`lookup_table`, so the GUI and tests can use the rest without VTK.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from ..viz.colormaps import COLORMAPS, DIVERGING_CMAPS, LUT_SIZE, get_lut

__all__ = [
    "COLORMAPS", "DIVERGING_CMAPS", "DYE_CHANNELS", "FIELD_INFO_3D", "ZERO_BASED", "FieldInfo3D", "SceneTheme", "colorize",
    "default_cmap", "display_range", "field_info", "hex_to_rgb", "lookup_table", "scene_theme",
]


# ---------------------------------------------------------------------------
# field metadata
# ---------------------------------------------------------------------------

class FieldInfo3D(NamedTuple):
    """How to present one 3D field: label, SI unit symbol, diverging flag, default map."""
    label: str
    unit: str
    diverging: bool
    cmap: str


#: Fields of :class:`~aerolume.model.SurfaceData3D`, ``SliceData`` and ``VolumeData``.
#: Maps follow docs/ui_guidelines.md section 5: speed viridis, Cp coolwarm centred on 0,
#: signed rates RdBu_r centred; magnitudes that start at 0 use a sequential map.
FIELD_INFO_3D: dict[str, FieldInfo3D] = {
    "speed": FieldInfo3D("Speed |V|", "m/s", False, "viridis"),
    "u": FieldInfo3D("Velocity u", "m/s", False, "viridis"),
    "v": FieldInfo3D("Velocity v", "m/s", True, "coolwarm"),
    "w": FieldInfo3D("Velocity w", "m/s", True, "coolwarm"),
    "p": FieldInfo3D("Pressure (gauge)", "Pa", True, "coolwarm"),
    "cp": FieldInfo3D("Pressure coefficient Cp", "", True, "coolwarm"),
    "cf": FieldInfo3D("Skin friction coefficient Cf", "", False, "viridis"),
    "vorticity": FieldInfo3D("Vorticity |ω|", "1/s", False, "magma"),
    "q": FieldInfo3D("Q-criterion", "1/s²", True, "RdBu_r"),
    "sensitivity": FieldInfo3D("Sensitivity dJ/dn", "1/m", True, "RdBu_r"),
    "nut": FieldInfo3D("Eddy viscosity νt", "m²/s", False, "magma"),
    "mask": FieldInfo3D("Solid fraction", "", False, "grayscale"),
    # the passive smoke (DESIGN.md 7.13): concentration 0..1, the 2D smoke map on slices
    "dye": FieldInfo3D("Smoke", "", False, "grayscale"),
    # further smoke channels (DESIGN.md 10.1): one per emitter colour
    "dye1": FieldInfo3D("Smoke channel 2", "", False, "grayscale"),
    "dye2": FieldInfo3D("Smoke channel 3", "", False, "grayscale"),
    "dye3": FieldInfo3D("Smoke channel 4", "", False, "grayscale"),
    # heat transfer (DESIGN.md 10.2): the stored temperature in °C, its fluctuation, and the solver's
    # non-dimensional excess theta = (T - T_amb) / dT_ref. "thermal" is the perceptual
    # black-red-yellow-white map of the 2D views (docs/ui_guidelines.md section 5).
    "temperature": FieldInfo3D("Temperature T", "°C", False, "thermal"),
    "temperature_rms": FieldInfo3D("Temperature fluctuation T′ rms", "K", False, "magma"),
    "theta": FieldInfo3D("Temperature excess θ", "", False, "thermal"),
}


#: Non-negative magnitudes whose colour range always starts at 0.
ZERO_BASED = frozenset({"speed", "vorticity", "cf", "tau", "nut", "temperature_rms"})
#: Smoke channels in solver order (``dye`` is channel 1, the v0.2 smoke).
DYE_CHANNELS = ("dye", "dye1", "dye2", "dye3")


def field_info(name: str | None) -> FieldInfo3D:
    """Metadata for a field; unknown names get a neutral sequential entry."""
    if name and name in FIELD_INFO_3D:
        return FIELD_INFO_3D[name]
    return FieldInfo3D(str(name or ""), "", False, "viridis")


def default_cmap(name: str | None) -> str:
    """Default colour map of a field (viridis when unknown)."""
    return field_info(name).cmap


# ---------------------------------------------------------------------------
# ranges and colouring
# ---------------------------------------------------------------------------

def display_range(values: np.ndarray, *, field: str | None = None, diverging: bool | None = None,
                  cmap: str | None = None, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    """Colour range for ``values``: robust percentiles, centred when diverging.

    ``diverging`` defaults to the field's flag, or to the map being a diverging
    one. Percentiles rather than min/max because a single stagnation vertex or
    a voxel inside a body would otherwise compress the whole map into one
    colour. Magnitudes (speed, |omega|, Cf, nu_t: :data:`ZERO_BASED`) start at
    exactly 0 (the wake and stagnation regions should read as "slow", not as
    "the slowest value present"). NaNs are ignored; an
    empty or constant input gives a small range around the value so a LUT can
    always be built.
    """
    if diverging is None:
        diverging = field_info(field).diverging if field in FIELD_INFO_3D else (cmap in DIVERGING_CMAPS)
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0) if diverging else (0.0, 1.0)
    if v.size > 400_000:            # percentiles of a strided sample are as good and 10x cheaper
        v = v[:: v.size // 200_000]
    a, b = np.percentile(v, [lo, hi])
    if diverging:
        m = max(abs(a), abs(b))
        if m <= 0.0:
            m = max(abs(float(v.max())), abs(float(v.min())), 1e-12)
        return (-float(m), float(m))
    if field in ZERO_BASED and a >= 0.0:
        a = 0.0
    if b - a <= 1e-12 * max(1.0, abs(b)):
        pad = max(abs(b) * 0.05, 1e-6)
        return (float(a - pad), float(b + pad))
    return (float(a), float(b))


def colorize(values: np.ndarray, cmap: str | None, vmin: float, vmax: float, *,
             mask: np.ndarray | None = None, mask_rgb: tuple[int, int, int] = (143, 148, 156),
             nan_rgb: tuple[int, int, int] = (143, 148, 156)) -> np.ndarray:
    """Map ``values`` through a colour map: (..., 3) uint8.

    Used for slice textures, where colouring on the CPU and uploading an RGB
    texture is cheaper and more predictable than a scalar texture with VTK's
    colour-mapping-on-texture path. ``mask`` (solid fraction, > 0.5 = solid)
    and NaNs get the neutral solid grey of the 2D views.
    """
    lut = get_lut(cmap)
    v = np.asarray(values, dtype=np.float64)
    span = vmax - vmin if vmax > vmin else 1.0
    t = np.clip((v - vmin) / span, 0.0, 1.0)
    idx = np.round(np.nan_to_num(t, nan=0.0) * (LUT_SIZE - 1)).astype(np.intp)
    out = lut[idx]
    bad = ~np.isfinite(v)
    if bad.any():
        out[bad] = nan_rgb
    if mask is not None:
        out[np.asarray(mask) > 0.5] = mask_rgb
    return out


def lookup_table(cmap: str | None, vmin: float, vmax: float, *, nan_rgb=(143, 148, 156)):
    """A ``vtkLookupTable`` holding exactly the 256 colours of the 2D map.

    VTK's own presets differ from matplotlib's in the last digit of most
    entries; building the table from :data:`COLORMAPS` keeps the 2D and 3D
    views pixel-identical for the same value. Values outside the range clamp
    to the end colours (the colour bar shows that with extend triangles).
    """
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkLookupTable

    lut = get_lut(cmap)
    rgba = np.concatenate([lut, np.full((lut.shape[0], 1), 255, np.uint8)], axis=1)
    table = vtkLookupTable()
    table.SetNumberOfTableValues(lut.shape[0])
    arr = numpy_to_vtk(np.ascontiguousarray(rgba), deep=True)
    table.SetTable(arr)
    lo, hi = float(vmin), float(vmax)
    if not hi > lo:
        hi = lo + 1e-9
    table.SetRange(lo, hi)
    table.SetNanColor(*(c / 255.0 for c in nan_rgb), 1.0)
    return table


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------

def hex_to_rgb(c: str | tuple) -> tuple[float, float, float]:
    """'#rrggbb' (or an RGB tuple in 0..1 or 0..255) -> RGB floats in 0..1."""
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    t = tuple(float(x) for x in c[:3])
    if max(t) > 1.0:
        t = tuple(x / 255.0 for x in t)
    return t  # type: ignore[return-value]


def _mix(a: str, b: str, t: float) -> tuple[float, float, float]:
    ca, cb = hex_to_rgb(a), hex_to_rgb(b)
    return tuple(ca[i] * (1 - t) + cb[i] * t for i in range(3))  # type: ignore[return-value]


# Copies of the docs/ui_guidelines.md token table (v4, the website's neutral greys), so viz3d needs
# no Qt. The app passes its live tokens, which override these; the table serves name-only calls.
_TOKENS = {
    "light": {"surface": "#ffffff", "panel": "#f6f6f6", "border": "#d6d6d6", "border_strong": "#b8b8b8",
              "text": "#0e0e0e", "text2": "#474747", "muted": "#5c5c5c", "grid": "#e6e6e6", "accent": "#2463c4"},
    "dark": {"surface": "#1a1a1a", "panel": "#111111", "border": "#2b2b2b", "border_strong": "#3d3d3d",
             "text": "#f1f1f1", "text2": "#b1b1b1", "muted": "#a3a3a3", "grid": "#2a2a2a", "accent": "#7fb1f7"},
}


@dataclass(frozen=True)
class SceneTheme:
    """Every colour the 3D scene draws that is not a colour-map value (RGB 0..1).

    ``body`` is a light neutral grey in both themes: like the 2D solids it must
    never be confused with a colour-map value, and under the light kit it shades
    from near white to mid grey, which reads as "engineering model", not as a
    painted object. ``gradient`` (off by default; the guidelines forbid
    decorative gradients) darkens the bottom of the background by ~4 %, the
    ParaView default look, for users who prefer it.

    ``smoke`` is the colour of the 3D smoke (DESIGN.md 7.13): near-white wisps
    on the graphite background, as smoke photographs in a dark tunnel; on the
    white background white smoke would be invisible, so there it is a dark
    neutral grey drawn with less opacity (``smoke_opacity``), like ink in water.
    ``accent`` (the one UI accent) marks the editable seed rake.
    """
    name: str
    background: tuple[float, float, float]
    background2: tuple[float, float, float]
    text: tuple[float, float, float]
    text2: tuple[float, float, float]
    muted: tuple[float, float, float]
    border: tuple[float, float, float]
    ground: tuple[float, float, float]
    grid: tuple[float, float, float]
    grid_major: tuple[float, float, float]
    body: tuple[float, float, float]
    edge: tuple[float, float, float]
    solid: tuple[float, float, float]
    axis_x: tuple[float, float, float]
    axis_y: tuple[float, float, float]
    axis_z: tuple[float, float, float]
    gradient: bool = False
    dark: bool = False
    accent: tuple[float, float, float] = (0.141, 0.388, 0.769)
    smoke: tuple[float, float, float] = (0.18, 0.18, 0.18)
    smoke_opacity: float = 0.18

    def with_gradient(self, on: bool = True) -> SceneTheme:
        return replace(self, gradient=bool(on))


def scene_theme(theme: str | dict | SceneTheme | None = "light", *, gradient: bool = False) -> SceneTheme:
    """Build a :class:`SceneTheme` from a theme name, a token dict or a SceneTheme.

    A token dict (``aerolume.app.theme.tokens()``) is merged over the table for
    its brightness, so a partial dict works. Darkness is decided from the
    background luminance, not the name, so custom token sets behave.
    """
    if isinstance(theme, SceneTheme):
        return theme.with_gradient(gradient) if gradient != theme.gradient else theme
    if theme is None or isinstance(theme, str):
        name = theme or "light"
        if name not in _TOKENS:
            raise ValueError(f"unknown theme {name!r}; use 'light', 'dark' or a token dict")
        tk = dict(_TOKENS[name])
    else:
        bg = hex_to_rgb(theme.get("surface", "#ffffff"))
        name = "dark" if (0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]) < 0.4 else "light"
        tk = dict(_TOKENS[name])
        tk.update({k: v for k, v in theme.items() if isinstance(v, str) and v.startswith("#")})
    dark = name == "dark"
    surface = tk["surface"]
    # v4: neutral greys of the same lightness as v3's blue-greys (the floor and grid tint, the body
    # and the edge), so the 3D floor and grid carry no cool cast beside the neutral chrome
    if dark:
        ground = _mix(surface, "#000000", 0.07)
        grid = _mix(surface, tk["text"], 0.07)
        grid_major = _mix(surface, tk["text"], 0.13)
        body = hex_to_rgb("#b8b8b8")
        edge = hex_to_rgb("#1c1c1c")
        axes = ("#e0675f", "#57b37c", "#5b97ec")
        bg2 = _mix(surface, "#000000", 0.18)
        smoke, smoke_opacity = hex_to_rgb("#f0f0f0"), 0.20
    else:
        ground = _mix(surface, "#5c5c5c", 0.06)
        grid = _mix(surface, "#5c5c5c", 0.16)
        grid_major = _mix(surface, "#5c5c5c", 0.26)
        body = hex_to_rgb("#cbcbcb")
        edge = hex_to_rgb("#5a5a5a")
        axes = ("#c0392b", "#2f855a", "#2463c4")
        bg2 = _mix(surface, "#5c5c5c", 0.08)
        smoke, smoke_opacity = hex_to_rgb("#2e2e2e"), 0.18
    return SceneTheme(
        name=name, background=hex_to_rgb(surface), background2=bg2, text=hex_to_rgb(tk["text"]),
        text2=hex_to_rgb(tk["text2"]), muted=hex_to_rgb(tk["muted"]), border=hex_to_rgb(tk["border_strong"]),
        ground=ground, grid=grid, grid_major=grid_major, body=body, edge=edge,
        solid=hex_to_rgb("#8f8f8f"), axis_x=hex_to_rgb(axes[0]), axis_y=hex_to_rgb(axes[1]),
        axis_z=hex_to_rgb(axes[2]), gradient=bool(gradient), dark=dark, accent=hex_to_rgb(tk["accent"]),
        smoke=smoke, smoke_opacity=smoke_opacity)
