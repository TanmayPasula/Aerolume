"""One entry point for "Import shape..." in the GUI and the CLI.

:func:`import_file` picks the reader from the file extension, collects the
readers' notes (auto-closed gaps, units, traced image size) and suggests what kind
of design the file probably is, so the Geometry page can pre-fill the import
dialog: an airfoil .dat is an airfoil; a single long, low outline is most likely a
vehicle; anything else is generic. The caller then turns the result into a
:class:`~aerolume.model.Design` with :func:`design_from_polygons` (or
:func:`design_from_airfoil`) once the user has confirmed length and kind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import polygon as P

__all__ = ["SUPPORTED_EXTENSIONS", "ImportResult", "import_file"]

SUPPORTED_EXTENSIONS = {
    ".dat": "Airfoil coordinates (Selig / Lednicer)",
    ".svg": "SVG drawing",
    ".dxf": "DXF drawing",
    ".png": "Image silhouette", ".jpg": "Image silhouette", ".jpeg": "Image silhouette",
    ".bmp": "Image silhouette", ".gif": "Image silhouette", ".tif": "Image silhouette",
    ".tiff": "Image silhouette", ".webp": "Image silhouette",
    ".csv": "Point list", ".txt": "Point list or airfoil coordinates", ".xy": "Point list",
}


@dataclass
class ImportResult:
    """Outlines read from a file plus what the importer learned about them.

    ``units``: "m" when the file is known to be in metres, otherwise a hint such as
    "pixels", "drawing units" or "chord" -- the GUI then asks for a real length.
    """
    polygons: list[np.ndarray]
    name: str
    suggested_kind: str
    units: str
    notes: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)


def _guess_kind(polys: list[np.ndarray]) -> str:
    if not polys:
        return "generic"
    x0, x1, y0, y1 = P.bbox(polys)
    w, h = x1 - x0, y1 - y0
    # main outline long and thin, any others slender too: an airfoil, possibly
    # multi-element (a deflected flap or slat is less slender in x/y extent)
    def aspect(q: np.ndarray) -> float:
        return float(np.ptp(q[:, 0]) / max(np.ptp(q[:, 1]), 1e-300))
    main = max(polys, key=lambda q: abs(P.signed_area(q)))
    if len(polys) <= 4 and aspect(main) > 5.0 and all(aspect(q) > 2.0 for q in polys):
        return "airfoil"
    if len(polys) <= 5 and h > 0 and 1.8 <= w / h <= 6.0:
        return "vehicle"
    if len(polys) == 1 and h > 0 and w / h > 6.0:
        return "airfoil"
    return "generic"


def import_file(path, **options) -> ImportResult:
    """Read any supported shape file. ``options`` are passed to the specific reader.

    Raises ValueError for unsupported extensions or files without usable outlines.
    """
    p = Path(path)
    ext = p.suffix.lower()
    notes: list[str] = []
    name = p.stem
    if ext == ".dat" or (ext == ".txt" and _looks_like_airfoil(p)):
        from .airfoils import read_airfoil_dat
        name, coords = read_airfoil_dat(p)
        return ImportResult([coords], name, "airfoil", "chord", notes,
                            {"type": "import", "format": "dat", "file": p.name})
    if ext == ".svg":
        from .svg import read_svg
        polys = read_svg(p, notes=notes, **options)
        units = "m" if options.get("units") == "m" else "drawing units"
        fmt = "svg"
    elif ext == ".dxf":
        from .dxf import read_dxf
        polys = read_dxf(p, notes=notes, **options)
        units = "drawing units"
        fmt = "dxf"
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"):
        from .image import read_image_silhouette
        polys = read_image_silhouette(p, notes=notes, **options)
        units = "pixels"
        fmt = "image"
    elif ext in (".csv", ".txt", ".xy"):
        from .csvio import read_points_csv
        polys = read_points_csv(p, notes=notes)
        units = "file units"
        fmt = "csv"
    else:
        raise ValueError(f"unsupported file type {ext!r}; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if not polys:
        raise ValueError(f"no closed outline found in {p.name}")
    return ImportResult(polys, name, _guess_kind(polys), units, notes,
                        {"type": "import", "format": fmt, "file": p.name})


def _looks_like_airfoil(p: Path) -> bool:
    """A .txt that parses as an airfoil .dat with x spanning ~[0, 1]."""
    from .airfoils import read_airfoil_dat
    try:
        _, c = read_airfoil_dat(p)
    except (ValueError, OSError):
        return False
    return bool(-0.05 < c[:, 0].min() < 0.05 and 0.95 < c[:, 0].max() < 1.05)
