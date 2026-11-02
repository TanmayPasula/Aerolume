"""Silhouette import from a picture (photo cut-out, scan, screenshot).

The quickest way for a vehicle designer to get a shape in is a side-view
silhouette. The pipeline:

1. Pillow loads the image. If it has a meaningful alpha channel (a cut-out PNG),
   opacity defines the object; otherwise the image is converted to grayscale.
2. A threshold separates object from background -- Otsu's method unless one is
   given. Which side is the object is decided from the image border (the class
   that dominates the border is background), so dark-on-light and light-on-dark
   both work; ``invert`` flips that decision.
3. contourpy traces the threshold iso-line of the *grayscale* field (not of the
   binarised mask), so anti-aliased edges give sub-pixel accurate outlines.
   Filled contours keep only the outer boundary of each region; holes (windows,
   wheel spokes) are discarded because the inside of a solid is irrelevant to the
   external flow.
4. Regions smaller than ``min_area`` are dropped (dust, text, logos) and the
   outlines are simplified with Ramer-Douglas-Peucker (``simplify`` pixels).

Coordinates are in pixels with y up (row 0 of the image is the top), so the
caller scales to metres -- :func:`aerolume.geometry.design_from_polygons` does
that from the real vehicle length.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .polygon import orient_ccw, remove_duplicates, signed_area, simplify_rdp

__all__ = ["otsu_threshold", "read_image_silhouette"]


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold (maximum between-class variance) of 8-bit-range values."""
    v = np.clip(np.asarray(values, dtype=float).ravel(), 0, 255)
    hist, edges = np.histogram(v, bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    sb[~np.isfinite(sb)] = -1
    k = int(np.argmax(sb))
    # threshold between bin k and k+1: values <= edges[k+1] belong to class 0
    return float(edges[k + 1])


def _load(path) -> tuple[np.ndarray, str]:
    """(field in 0..255 as float, source) with alpha preferred when it carries the shape."""
    from PIL import Image

    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = np.asarray(im.convert("RGBA"), dtype=float)
        alpha = rgba[..., 3]
        if alpha.min() < 128 <= alpha.max():
            return alpha, "alpha"
        im = im.convert("RGB")
    return np.asarray(im.convert("L"), dtype=float), "gray"


def read_image_silhouette(path, *, threshold: float | None = None, invert: bool = False,
                          simplify: float = 0.5, min_area: float | None = None,
                          notes: list[str] | None = None) -> list[np.ndarray]:
    """Trace the outer outlines of the object(s) in an image, in pixel units (y up).

    ``threshold``: gray level 0..255 separating object and background (default:
    Otsu). ``invert``: swap object and background after the automatic border-based
    choice. ``simplify``: Ramer-Douglas-Peucker tolerance in pixels (0 keeps every
    traced point). ``min_area``: smallest region kept, in pixels^2 (default 0.1 %
    of the image area, at least 16 px^2). Returns CCW polygons, largest first.
    """
    import contourpy

    z, source = _load(Path(path) if not hasattr(path, "read") else path)
    h, w = z.shape
    thr = float(threshold) if threshold is not None else (127.5 if source == "alpha" else otsu_threshold(z))
    if source == "alpha":
        obj_bright = True                     # opaque = object
    else:
        border = np.concatenate([z[0], z[-1], z[:, 0], z[:, -1]])
        obj_bright = (border > thr).mean() < 0.5   # bright border -> dark object
    if invert:
        obj_bright = not obj_bright
    f = (z - thr) if obj_bright else (thr - z)     # > 0 inside the object
    # pad with background so every region is closed by the tracer
    pad = -max(float(np.abs(f).max()), 1.0)
    f = np.pad(f, 1, constant_values=pad)
    cg = contourpy.contour_generator(z=f, fill_type=contourpy.FillType.OuterOffset)
    filled = cg.filled(0.0, np.inf)
    points_list, offsets_list = filled
    if min_area is None:
        min_area = max(16.0, 1e-3 * h * w)
    out = []
    for pts, offs in zip(points_list, offsets_list):
        outer = pts[offs[0]:offs[1]]            # first ring = outer boundary
        if len(outer) < 4:
            continue
        # contourpy x = column, y = row (of the padded array); image rows run down
        P = np.column_stack([outer[:, 0] - 1.0, (h - 1) - (outer[:, 1] - 1.0)])
        P = remove_duplicates(P)
        if abs(signed_area(P)) < min_area:
            continue
        if simplify and simplify > 0:
            P = simplify_rdp(P, simplify)
        if len(P) >= 3:
            out.append(orient_ccw(P))
    out.sort(key=lambda q: -abs(signed_area(q)))
    if notes is not None:
        notes.append(f"Traced {len(out)} outline(s) from a {w}x{h} image "
                     f"({'alpha channel' if source == 'alpha' else f'threshold {thr:.0f}'}); "
                     "coordinates are in pixels until scaled.")
    return out
