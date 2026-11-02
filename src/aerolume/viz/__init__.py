"""Visualisation primitives shared by the live view, the results pages and exports.

* :mod:`.colormaps` -- 256-entry LUTs, per-field labels / units / default maps
  (numpy only, safe to import anywhere, including the GUI thread at start-up).
* :mod:`.render`    -- field -> RGBA image on the field's device, robust ranges.
* :mod:`.streamlines` -- evenly spaced (Jobard-Lefer) streamlines.
* :mod:`.particles` -- GPU/CPU tracer particles released from an inlet rake.
* :mod:`.lic`       -- line-integral-convolution flow texture.

The torch-backed names are resolved lazily (PEP 562), so ``from aerolume.viz
import COLORMAPS`` does not import torch.
"""
from __future__ import annotations

from typing import Any

from .colormaps import (
    CMAP_FAMILIES,
    CMAP_NAMES,
    COLORMAPS,
    DEFAULT_CMAP,
    DIVERGING_CMAPS,
    DYE_COLORS,
    DYE_FIELDS,
    FIELD_INFO,
    FieldInfo,
    colorbar_image,
    default_cmap,
    dye_channels,
    field_info,
    get_lut,
    hex_rgb,
    lut_rgba,
    theta_to_celsius,
    to_si,
)

# Bound eagerly (numpy only): a lazily bound function would be shadowed by the
# submodule of the same name whenever ``aerolume.viz.streamlines`` is imported first.
from .streamlines import streamlines

_LAZY = {
    "render_field": ("render", "render_field"),
    "render_dye": ("render", "render_dye"),
    "auto_range": ("render", "auto_range"),
    "SOLID_RGBA": ("render", "SOLID_RGBA"),
    "Particles": ("particles", "Particles"),
    "EmitterParticles": ("particles", "EmitterParticles"),
    "lic_texture": ("lic", "lic"),        # not "lic": that name is the submodule
}

__all__ = [
    "CMAP_FAMILIES", "CMAP_NAMES", "COLORMAPS", "DEFAULT_CMAP", "DIVERGING_CMAPS", "DYE_COLORS", "DYE_FIELDS",
    "FIELD_INFO", "SOLID_RGBA", "EmitterParticles", "FieldInfo", "Particles", "auto_range", "colorbar_image",
    "default_cmap", "dye_channels", "field_info", "get_lut", "hex_rgb", "lic_texture", "lut_rgba",
    "render_dye", "render_field", "streamlines", "theta_to_celsius", "to_si",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(f"{__name__}.{mod}"), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
