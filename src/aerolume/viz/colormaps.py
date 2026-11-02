"""Colour lookup tables and per-field display metadata.

Every image Aerolume draws of a flow field goes through a 256-entry RGB lookup
table (LUT). Keeping the maps as plain ``(256, 3) uint8`` arrays means the live
view can gather them on the GPU (:mod:`aerolume.viz.render`), pyqtgraph can use
them for colour bars, and the report can hand the very same colours to
matplotlib, so a screenshot and a PDF figure of one result always agree.

The perceptual maps (viridis, magma, ...) are sampled once at import from
matplotlib's registry -- ``matplotlib.colormaps`` only, never ``pyplot``, so no
GUI backend is ever touched. Maps matplotlib does not ship (the 'icefire'-like
vorticity map, 'grayscale') are built here by interpolating anchor colours in
CIELAB, which keeps lightness steps even. If matplotlib is not importable at
all, the same Lab interpolation over published anchor colours stands in for the
sampled maps; the result is visually indistinguishable at screen scale.

``FIELD_INFO`` says how each solver field should be presented: a label, the
kind of unit (matching :mod:`aerolume.core.units` kinds where one exists),
whether zero is a meaningful centre (diverging map, symmetric range), and how
to turn the solver's non-dimensional numbers into SI for a given speed, density
and reference length (DESIGN.md section 3: length by L, velocity by U, time by
L/U, pressure by rho U^2). This module imports only numpy (plus matplotlib,
guarded), so the GUI can use it without pulling in torch.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

LUT_SIZE = 256

# ---------------------------------------------------------------------------
# colour science helpers (sRGB <-> CIELAB, D65)
# ---------------------------------------------------------------------------

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of sRGB colours in 0..1, shape (..., 3). L* runs 0..100."""
    xyz = _srgb_to_linear(np.asarray(rgb, dtype=float)) @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`srgb_to_lab`; out-of-gamut colours are clipped to 0..1."""
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA) * _WHITE
    return np.clip(_linear_to_srgb(xyz @ _M_XYZ2RGB.T), 0.0, 1.0)


def _hex(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def lut_from_anchors(colors: list[str] | np.ndarray, positions: list[float] | None = None,
                     n: int = LUT_SIZE) -> np.ndarray:
    """Build an (n, 3) uint8 LUT by interpolating anchor colours in CIELAB.

    Interpolating in Lab rather than RGB keeps the lightness ramp smooth, which
    is what the eye reads as "value" -- RGB interpolation between saturated
    anchors produces muddy, darker midpoints and visible false contours.
    ``colors`` are '#rrggbb' strings or an (k, 3) float array in 0..1;
    ``positions`` default to evenly spaced in [0, 1].
    """
    rgb = np.array([_hex(c) for c in colors]) if isinstance(colors[0], str) else np.asarray(colors, float)
    pos = np.linspace(0.0, 1.0, len(rgb)) if positions is None else np.asarray(positions, float)
    lab = srgb_to_lab(rgb)
    t = np.linspace(0.0, 1.0, n)
    out = np.stack([np.interp(t, pos, lab[:, k]) for k in range(3)], axis=-1)
    return np.round(lab_to_srgb(out) * 255.0).astype(np.uint8)


def _lab_to_linear_unclipped(lab: np.ndarray) -> np.ndarray:
    """Linear-light RGB of CIELAB colours *without* clipping (for gamut tests)."""
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA) * _WHITE
    return xyz @ _M_XYZ2RGB.T


def lut_from_lch(lightness: list[float], hue_deg: list[float], chroma: list[float],
                 positions: list[float] | None = None, n: int = LUT_SIZE) -> np.ndarray:
    """Build an (n, 3) uint8 LUT along a path in CIELCh with the lightness exactly as given.

    This is how the CET-style maps below are made (after Kovesi, "Good colour
    maps: how to design them", 2015): the lightness ``L*`` -- which is what the
    eye reads as magnitude -- is piecewise linear between the knots, so a
    sequential map whose knots rise evenly has *uniform perceptual contrast*
    along its whole length and cannot show a false front. Hue (degrees,
    interpolated linearly, so a path may run past 360) and chroma give the
    colour. Where the requested chroma is outside the sRGB gamut at that
    lightness and hue it is reduced (bisection) until the colour fits; the
    lightness is never touched, which is what lab-then-clip would do and
    what makes clipped maps band.
    """
    pos = np.linspace(0.0, 1.0, len(lightness)) if positions is None else np.asarray(positions, float)
    t = np.linspace(0.0, 1.0, n)
    L = np.interp(t, pos, np.asarray(lightness, float))
    h = np.radians(np.interp(t, pos, np.asarray(hue_deg, float)))
    c_want = np.interp(t, pos, np.asarray(chroma, float))
    lo, hi = np.zeros(n), c_want.copy()

    def fits(c: np.ndarray) -> np.ndarray:
        rgb = _lab_to_linear_unclipped(np.stack([L, c * np.cos(h), c * np.sin(h)], axis=-1))
        return np.all((rgb >= -1e-9) & (rgb <= 1.0 + 1e-9), axis=-1)

    ok = fits(hi)
    for _ in range(40):                 # bisection on the chroma of the samples that do not fit
        mid = 0.5 * (lo + hi)
        good = fits(mid)
        lo = np.where(ok, lo, np.where(good, mid, lo))
        hi = np.where(ok, hi, np.where(good, hi, mid))
    c = np.where(ok, c_want, lo)
    lab = np.stack([L, c * np.cos(h), c * np.sin(h)], axis=-1)
    return np.round(lab_to_srgb(lab) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# the maps
# ---------------------------------------------------------------------------

#: Order in which maps are offered in the GUI (perceptual first, then diverging).
CMAP_NAMES: tuple[str, ...] = ("viridis", "magma", "inferno", "plasma", "cividis", "thermal", "fire", "turbo",
                               "coolwarm", "RdBu_r", "bwr", "icefire", "grayscale")

#: Maps whose centre is meant to sit at zero (used to pick symmetric ranges).
DIVERGING_CMAPS = frozenset({"coolwarm", "RdBu_r", "bwr", "icefire"})

#: The maps grouped as the colour-map menus list them: (family title, map names, what the family is for).
#: "Sequential" maps have lightness rising monotonically (magnitudes, temperature); "Diverging" maps
#: are symmetric about a neutral centre (signed fields, temperature above and below ambient);
#: turbo is a rainbow kept for users who ask for one -- never a default (docs/ui_guidelines.md 1).
CMAP_FAMILIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("Sequential", ("viridis", "cividis", "magma", "inferno", "plasma", "thermal", "fire"),
     "Lightness rises monotonically: magnitudes and temperature"),
    ("Diverging", ("coolwarm", "RdBu_r", "bwr", "icefire"),
     "Symmetric about a neutral centre: signed fields"),
    ("Other", ("turbo", "grayscale"), "Turbo is a rainbow (not perceptually uniform); greyscale for smoke"),
)

#: Maps built in CIELCh with the lightness linear between knots (CET-style, see :func:`lut_from_lch`):
#: lightness L*, hue in degrees and chroma at the knots (``pos``: knot positions, default even).
#: Not the published CET tables -- constructed here with the same rule, and checked by tests
#: (monotone or symmetric lightness, no gamut clipping).
_LCH_MAPS: dict[str, dict[str, list[float]]] = {
    # temperature: navy -> violet -> red -> orange -> pale yellow, L* 13 -> 94 (cmocean-thermal-like hues)
    "thermal": {"L": [13.0, 33.25, 53.5, 73.75, 94.0], "h": [262.0, 305.0, 372.0, 418.0, 452.0],
                "c": [34.0, 52.0, 72.0, 78.0, 82.0]},
    # heat (CET L3-style): black -> red -> orange -> yellow -> white, L* 3 -> 99
    "fire": {"L": [3.0, 27.0, 51.0, 75.0, 99.0], "h": [25.0, 32.0, 48.0, 75.0, 100.0],
             "c": [6.0, 78.0, 92.0, 88.0, 8.0]},
    # diverging blue - white - red (CET D1-style): equal lightness at equal distance from the
    # centre; the hue switches at the (achromatic) centre so neither half drifts through purple
    "bwr": {"L": [44.0, 70.5, 97.0, 97.0, 70.5, 44.0], "h": [262.0, 260.0, 258.0, 32.0, 30.0, 30.0],
            "c": [58.0, 36.0, 0.0, 0.0, 38.0, 62.0], "pos": [0.0, 0.25, 0.5, 0.5, 0.75, 1.0]},
}

# Anchors for every map. For the matplotlib maps these are only a fallback used
# when matplotlib cannot be imported: equally spaced samples of each map
# (11 points, 17 for turbo), which Lab interpolation reproduces to within ~2
# levels of 255 on average.
_ANCHORS: dict[str, list[str]] = {
    "viridis": ["#440154", "#482576", "#414487", "#355f8d", "#2a788e", "#21918c",
                "#22a884", "#42be71", "#7ad151", "#bddf26", "#fde725"],
    "magma": ["#000004", "#150e38", "#3b0f70", "#641a80", "#8c2981", "#b73779",
              "#de4968", "#f66e5c", "#fe9f6d", "#fecf92", "#fcfdbf"],
    "inferno": ["#000004", "#180c3c", "#420a68", "#6a176e", "#932667", "#bc3754",
                "#dd513a", "#f3761b", "#fca50a", "#f6d746", "#fcffa4"],
    "plasma": ["#0d0887", "#43039e", "#6a00a8", "#8f0da4", "#b12a90", "#cc4778",
               "#e16462", "#f1834c", "#fca636", "#fcce25", "#f0f921"],
    "cividis": ["#00224e", "#0c3470", "#35456c", "#4f576c", "#666970", "#7d7c78",
                "#948e77", "#ada272", "#c8b866", "#e5cf52", "#fee838"],
    "turbo": ["#30123b", "#4040a2", "#466be3", "#4294ff", "#28bceb", "#18ddc2",
              "#32f298", "#6dfe62", "#a4fc3c", "#cbed34", "#ecd13a", "#fdae35",
              "#fb8122", "#ec530f", "#d23105", "#ac1701", "#7a0403"],
    "coolwarm": ["#3b4cc0", "#5a78e4", "#7b9ff9", "#9ebeff", "#c0d4f5", "#dddcdc",
                 "#f2cbb7", "#f7ad90", "#ee8468", "#d65244", "#b40426"],
    "RdBu_r": ["#053061", "#2267ac", "#4393c3", "#90c4dd", "#d1e5f0", "#f7f6f6",
               "#fddbc7", "#f4a683", "#d6604d", "#b1182b", "#67001f"],
    # Vorticity: dark centre so the irrotational free stream recedes and the
    # shear layers / vortex cores glow -- cyan-blue for clockwise (negative),
    # red-orange-cream for counter-clockwise (positive). Lightness is V-shaped
    # and symmetric, so +w and -w of equal magnitude read as equally strong.
    "icefire": ["#bde7db", "#6fb8d8", "#3b7fcc", "#3a4c9b", "#252a4d", "#1d1d21",
                "#4c1f33", "#8e2a3e", "#d1404a", "#f08a4f", "#fbd9a6"],
}
# Positions for the hand-made maps (others are evenly spaced).
_ANCHOR_POS: dict[str, list[float]] = {
    "icefire": [0.0, 0.12, 0.25, 0.36, 0.45, 0.5, 0.55, 0.64, 0.75, 0.88, 1.0],
}
# Names under which matplotlib registers our maps.
_MPL_NAMES = {"viridis": "viridis", "magma": "magma", "inferno": "inferno", "plasma": "plasma",
              "cividis": "cividis", "turbo": "turbo", "coolwarm": "coolwarm", "RdBu_r": "RdBu_r"}


def _sample_matplotlib() -> dict[str, np.ndarray]:
    """Sample matplotlib's registered maps at 256 points; {} if unavailable.

    Only ``matplotlib.colormaps`` (the registry) is touched -- importing it
    does not select or start any backend.
    """
    try:
        from matplotlib import colormaps as registry
    except ImportError:     # pragma: no cover - matplotlib is a declared dependency
        return {}
    out = {}
    t = np.linspace(0.0, 1.0, LUT_SIZE)
    for ours, theirs in _MPL_NAMES.items():
        if theirs in registry:
            rgba = registry[theirs](t)
            out[ours] = np.round(np.asarray(rgba)[:, :3] * 255.0).astype(np.uint8)
    return out


def _build() -> dict[str, np.ndarray]:
    maps = _sample_matplotlib()
    for name in CMAP_NAMES:
        if name in maps:
            continue
        if name in _LCH_MAPS:
            k = _LCH_MAPS[name]
            maps[name] = lut_from_lch(k["L"], k["h"], k["c"], k.get("pos"))
        elif name == "grayscale":
            ramp = np.arange(LUT_SIZE, dtype=np.uint8)
            maps[name] = np.stack([ramp, ramp, ramp], axis=-1)
        else:
            maps[name] = lut_from_anchors(_ANCHORS[name], _ANCHOR_POS.get(name))
    for lut in maps.values():
        lut.setflags(write=False)       # shared, cached on devices: never mutate
    return {name: maps[name] for name in CMAP_NAMES}


#: name -> (256, 3) uint8 LUT, index 0 = low end. Read-only arrays.
COLORMAPS: dict[str, np.ndarray] = _build()


def get_lut(name: str | None) -> np.ndarray:
    """The (256, 3) uint8 LUT for a map name.

    Lookup is case-insensitive and a trailing ``_r`` reverses any map (so
    'viridis_r' works even though only 'viridis' is stored). ``None`` gives
    viridis. Unknown names raise ``KeyError`` listing the valid ones, because a
    silent fallback would hide typos in saved view settings.
    """
    if name is None:
        return COLORMAPS["viridis"]
    if name in COLORMAPS:
        return COLORMAPS[name]
    lower = {k.lower(): k for k in COLORMAPS}
    key = name.lower()
    if key in lower:
        return COLORMAPS[lower[key]]
    if key.endswith("_r") and key[:-2] in lower:
        rev = COLORMAPS[lower[key[:-2]]][::-1].copy()
        rev.setflags(write=False)
        return rev
    raise KeyError(f"unknown colormap {name!r}; available: {', '.join(COLORMAPS)}")


def lut_rgba(name: str | None, alpha: int = 255) -> np.ndarray:
    """(256, 4) uint8 LUT with a constant alpha column (pyqtgraph ``setLookupTable``)."""
    lut = get_lut(name)
    return np.concatenate([lut, np.full((lut.shape[0], 1), alpha, np.uint8)], axis=1)


def colorbar_image(name: str | None, length: int = 256, thickness: int = 14, *,
                   vertical: bool = True) -> np.ndarray:
    """A colour-bar strip as (H, W, 4) uint8, high values at the top (or right).

    Used for legends next to live and results views so they need not rebuild
    the LUT themselves.
    """
    lut = lut_rgba(name)
    idx = np.round(np.linspace(0, LUT_SIZE - 1, length)).astype(int)
    strip = lut[idx]                                   # low -> high
    if vertical:
        return np.ascontiguousarray(np.repeat(strip[::-1, None, :], thickness, axis=1))
    return np.ascontiguousarray(np.repeat(strip[None, :, :], thickness, axis=0))


# ---------------------------------------------------------------------------
# per-field display metadata
# ---------------------------------------------------------------------------

#: SI symbol for each units kind used in FIELD_INFO. Temperatures are shown in °C (the unit the
#: user types in Setup); a temperature *difference* is in K.
UNIT_SYMBOL: dict[str, str] = {
    "speed": "m/s",
    "pressure": "Pa",
    "rate": "1/s",
    "dimensionless": "",
    "kinematic_viscosity": "m²/s",
    "temperature": "°C",
    "temperature_difference": "K",
    "heat_flux": "W/m²",
}


class FieldInfo(NamedTuple):
    """How to present one solver field: ``(label, units_kind, diverging)``.

    It is a 3-tuple (the DESIGN.md contract, so ``label, kind, div = info``
    works) with helpers for the non-dimensional -> SI conversion. The units
    kind fixes the conversion, following the solver's scaling (DESIGN.md 3):
    velocities scale with U, pressure with rho U^2, rates (vorticity,
    divergence) with U/L and eddy viscosity with U L. Temperature is not a
    pure scaling (T = T_amb + theta * dT_ref has an offset), so a
    "temperature" field is expected in °C already and scales by 1; turn the
    solver's theta into °C with :func:`theta_to_celsius`.
    """
    label: str
    units_kind: str
    diverging: bool

    @property
    def unit(self) -> str:
        """SI unit symbol ('' for dimensionless fields)."""
        return UNIT_SYMBOL[self.units_kind]

    def scale(self, speed: float, density: float, length: float) -> float:
        """Factor turning the solver's non-dimensional value into SI."""
        k = self.units_kind
        if k == "speed":
            return float(speed)
        if k == "pressure":
            return float(density) * float(speed) ** 2
        if k == "rate":
            return float(speed) / float(length)
        if k == "kinematic_viscosity":
            return float(speed) * float(length)
        return 1.0


FIELD_INFO: dict[str, FieldInfo] = {
    "speed": FieldInfo("Speed |V|", "speed", False),
    "u": FieldInfo("Streamwise velocity u", "speed", False),
    "v": FieldInfo("Vertical velocity v", "speed", True),
    "p": FieldInfo("Pressure (gauge)", "pressure", True),
    "cp": FieldInfo("Pressure coefficient Cp", "dimensionless", True),
    "vorticity": FieldInfo("Vorticity ωz", "rate", True),
    "dye": FieldInfo("Smoke", "dimensionless", False),
    "mask": FieldInfo("Solid fraction", "dimensionless", False),
    "nut": FieldInfo("Eddy viscosity νt", "kinematic_viscosity", False),
    "divergence": FieldInfo("Divergence ∇·u", "rate", True),
    # heat transfer (DESIGN.md 10.2): T in °C, theta = (T - T_amb) / dT_ref
    "temperature": FieldInfo("Temperature T", "temperature", False),
    "theta": FieldInfo("Temperature excess θ", "dimensionless", False),
    "heat_flux": FieldInfo("Wall heat flux q″", "heat_flux", True),
    # further smoke channels (DESIGN.md 10.1: emitters map to up to four channels by colour)
    "dye1": FieldInfo("Smoke, channel 2", "dimensionless", False),
    "dye2": FieldInfo("Smoke, channel 3", "dimensionless", False),
    "dye3": FieldInfo("Smoke, channel 4", "dimensionless", False),
    # RMS fluctuations about the time average (the Results' time-averaged set): named, never raw keys
    "u_rms": FieldInfo("u RMS fluctuation", "speed", False),
    "v_rms": FieldInfo("v RMS fluctuation", "speed", False),
    "w_rms": FieldInfo("w RMS fluctuation", "speed", False),
    "p_rms": FieldInfo("Pressure RMS fluctuation", "pressure", False),
    "cp_rms": FieldInfo("Cp RMS fluctuation", "dimensionless", False),
    "temperature_rms": FieldInfo("Temperature RMS fluctuation", "temperature_difference", False),
    "theta_rms": FieldInfo("θ RMS fluctuation", "dimensionless", False),
}

#: Default colour map per field (docs/ui_guidelines.md section 5). Speed uses the
#: perceptually uniform viridis, never a rainbow: equal steps in value read as equal
#: steps in lightness, so a designer does not see fronts that are not there. Signed
#: fields use diverging maps centred on zero (coolwarm for pressure, RdBu_r for
#: vorticity and other rates); smoke is greyscale, the convention of a real smoke
#: tunnel; eddy viscosity uses magma, which keeps near-zero regions dark.
DEFAULT_CMAP: dict[str, str] = {
    "speed": "viridis",
    "u": "viridis",
    "v": "coolwarm",
    "p": "coolwarm",
    "cp": "coolwarm",
    "vorticity": "RdBu_r",
    "dye": "grayscale",
    "mask": "grayscale",
    "nut": "magma",
    "divergence": "RdBu_r",
    # temperature has its own map (not viridis), so a temperature field is never read as a speed
    "temperature": "thermal",
    "theta": "thermal",
    "heat_flux": "bwr",
    "dye1": "grayscale",
    "dye2": "grayscale",
    "dye3": "grayscale",
    # fluctuation magnitudes start at zero: magma keeps the quiet regions dark
    "u_rms": "magma",
    "v_rms": "magma",
    "w_rms": "magma",
    "p_rms": "magma",
    "cp_rms": "magma",
    "temperature_rms": "magma",
    "theta_rms": "magma",
}

#: Smoke-channel names in channel order (channel 0 is the original ``dye``).
DYE_FIELDS: tuple[str, ...] = ("dye", "dye1", "dye2", "dye3")
#: Default colours of up to four smoke channels when the emitters do not set their own: near-white
#: smoke, then Okabe-Ito orange, sky blue and reddish purple -- distinguishable with the common
#: colour-vision deficiencies and all light enough to read on the dark smoke background.
DYE_COLORS: tuple[str, ...] = ("#f5f5f5", "#e69f00", "#56b4e9", "#cc79a7")
#: Background of a smoke picture (a smoke tunnel is photographed against black); both themes.
DYE_BACKGROUND: tuple[int, int, int] = (24, 26, 30)


def hex_rgb(color: str | tuple | list) -> tuple[int, int, int]:
    """'#rrggbb' / '#rgb' / an (r, g, b) tuple (0..255 or 0..1 floats) -> (r, g, b) ints 0..255."""
    if isinstance(color, str):
        c = color.strip().lstrip("#")
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        if len(c) != 6:
            raise ValueError(f"not a colour: {color!r}")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    vals = [float(v) for v in color][:3]
    if len(vals) != 3:
        raise ValueError(f"not a colour: {color!r}")
    if max(vals) <= 1.0:
        vals = [v * 255.0 for v in vals]
    return tuple(round(min(max(v, 0.0), 255.0)) for v in vals)  # type: ignore[return-value]


def dye_channels(emitters, *, max_channels: int = len(DYE_FIELDS)) -> list[dict]:
    """Smoke channels of a list of emitters: one per distinct colour, in order of first use.

    DESIGN.md 10.1: smoke emitters map to up to four independent channels *by
    colour*, so two white emitters share channel 0 and an orange one gets
    channel 1. Returns ``[{"field": "dye", "color": "#f5f5f5", "names":
    ["Rake 1", ...]}, ...]``; colours beyond ``max_channels`` fold into the
    last channel (the solver does the same). Particle emitters and disabled
    ones carry no smoke and are skipped. Pure data: the GUI legend and the
    solver agree because both call this.
    """
    out: list[dict] = []
    by_colour: dict[tuple[int, int, int], dict] = {}
    for e in emitters or []:
        if getattr(e, "species", "dye") != "dye" or not getattr(e, "enabled", True):
            continue
        try:
            rgb = hex_rgb(getattr(e, "color", DYE_COLORS[0]) or DYE_COLORS[0])
        except ValueError:
            rgb = hex_rgb(DYE_COLORS[0])
        ch = by_colour.get(rgb)
        if ch is None:
            if len(out) >= max_channels:
                ch = out[-1]
            else:
                ch = {"field": DYE_FIELDS[len(out)], "color": "#{:02x}{:02x}{:02x}".format(*rgb), "names": []}
                out.append(ch)
            by_colour[rgb] = ch
        ch["names"].append(str(getattr(e, "name", "Emitter")))
    return out


#: A smoke channel has *filled* its domain when more than this share of the fluid holds over half the
#: source concentration. A closed room's dye cannot leave: after 240 L/U the fan room's orange rake read
#: 0.86 at the median cell, and drawn on the absolute 0..1 scale the whole room was one orange wash.
DYE_FILLED_SHARE = 0.25
#: Percentiles of a filled channel's own fluid cells mapped to transparent and fully opaque.
DYE_STRETCH_PCT: tuple[float, float] = (5.0, 99.5)


def dye_stretch(channel, mask=None) -> tuple[float, float] | None:
    """(lo, hi) to contrast-stretch a smoke channel that has filled its domain, else None.

    ``channel`` is a (ny, nx[, nz]) concentration 0..1, ``mask`` the solid fraction
    on the same grid (solids are left out). A channel is stretched only when more
    than :data:`DYE_FILLED_SHARE` of the fluid holds over half the source
    concentration; its :data:`DYE_STRETCH_PCT` percentiles then become 0 and 1, so
    the picture shows where the smoke is *thinner* -- the flow's structure -- at
    the price of the absolute scale, which the caller must say it gave up.
    """
    a = np.asarray(channel, dtype=np.float64)
    fluid = np.isfinite(a)
    if mask is not None:
        m = np.asarray(mask)
        if m.shape == a.shape:
            fluid &= m < 0.5
    vals = a[fluid]
    if vals.size < 16 or float(np.mean(vals > 0.5)) <= DYE_FILLED_SHARE:
        return None
    lo, hi = (float(x) for x in np.percentile(vals, DYE_STRETCH_PCT))
    if hi - lo < 1e-3:
        return None                                  # uniform: there is no structure to bring out
    return lo, hi


def composite_dye(channels, colors=None, *, background=DYE_BACKGROUND, gain: float = 1.0,
                  ranges=None) -> np.ndarray:
    """Numpy twin of :func:`aerolume.viz.render.render_dye` for stored results (no torch).

    ``channels``: a sequence of (ny, nx) concentrations 0..1 in solver
    orientation; ``colors``: one colour per channel (:data:`DYE_COLORS` by
    default). Each channel is laid *over* the picture in its colour with
    opacity ``clip(gain * c, 0, 1)``, in channel order. Returns (ny, nx, 4)
    uint8 in image orientation (row 0 = top), solids not painted (the caller's
    mask handling decides). ``ranges``: per channel None or a (lo, hi) from
    :func:`dye_stretch`, mapped to opacity 0 and 1 instead of 0 and ``1 / gain``.
    """
    chans = [np.nan_to_num(np.asarray(c, dtype=np.float32), nan=0.0) for c in channels]
    if not chans:
        raise ValueError("composite_dye needs at least one channel")
    cols = list(colors) if colors is not None else list(DYE_COLORS)
    while len(cols) < len(chans):
        cols.append(DYE_COLORS[min(len(cols), len(DYE_COLORS) - 1)])
    out = np.empty(chans[0].shape + (3,), np.float32)
    out[:] = np.asarray(background, np.float32)
    ranges = list(ranges) if ranges is not None else []
    for i, (c, col) in enumerate(zip(chans, cols)):
        rng = ranges[i] if i < len(ranges) else None
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
            a = np.clip((c - lo) / max(hi - lo, 1e-9), 0.0, 1.0)[..., None]
        else:
            a = np.clip(c * float(gain), 0.0, 1.0)[..., None]
        out += a * (np.asarray(hex_rgb(col), np.float32) - out)
    rgba = np.empty(out.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba[::-1])


def theta_to_celsius(theta, *, t_ambient: float, delta_t: float):
    """The solver's non-dimensional temperature theta = (T - T_amb) / dT_ref in °C (arrays or tensors)."""
    return theta * float(delta_t) + float(t_ambient)


def field_info(name: str) -> FieldInfo:
    """FIELD_INFO entry, or a generic dimensionless one for fields not listed."""
    return FIELD_INFO.get(name) or FieldInfo(name, "dimensionless", False)


def default_cmap(name: str) -> str:
    """Default map for a field ('viridis' for fields not listed)."""
    return DEFAULT_CMAP.get(name, "viridis")


def to_si(name: str, values, *, speed: float, density: float, length: float):
    """Convert a non-dimensional solver field (array, tensor or float) to SI.

    Works on anything that supports multiplication by a float, so a GPU
    tensor stays on the GPU.
    """
    return values * field_info(name).scale(speed, density, length)
