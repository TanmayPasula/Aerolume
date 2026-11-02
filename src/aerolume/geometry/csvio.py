"""Point-list import (CSV / TXT / spreadsheet export).

The lowest common denominator: a column of x and a column of y. The reader is
deliberately forgiving because these files are typed by hand or exported from
spreadsheets in any locale:

* separators: comma, semicolon, tab or spaces; decimal commas when the separator
  is ``;`` or a tab (e.g. a German Excel export) or when every value has one;
* a header row (any non-numeric line) is skipped;
* several outlines are separated by blank lines or non-numeric lines, or grouped
  by an optional third column (body id / name);
* a repeated closing point is removed.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

from .polygon import as_polygon, orient_ccw, remove_duplicates, signed_area

__all__ = ["read_points_csv"]

_NUM = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def _split(line: str) -> list[str]:
    s = line.strip()
    if ";" in s or "\t" in s:
        toks = [t.strip().replace(",", ".") for t in re.split(r"[;\t]", s)]
        return [t for t in toks if t != ""]
    if "," in s:
        toks = [t.strip() for t in s.split(",")]
        if len(toks) >= 2:
            return [t for t in toks if t != ""]
    toks = s.split()
    if toks and all(t.count(",") == 1 and "." not in t for t in toks[:2]):
        toks = [t.replace(",", ".") for t in toks]
    return toks


def read_points_csv(path, *, notes: list[str] | None = None) -> list[np.ndarray]:
    """Read one or more outlines from a point list. Returns CCW, implicitly closed polygons."""
    if hasattr(path, "read"):
        text = path.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
    elif isinstance(path, str) and "\n" in path:
        text = path
    else:
        text = Path(path).read_bytes().decode("utf-8", errors="replace")
    groups: dict[str, list[list[float]]] = {}
    order: list[str] = []
    block = 0
    in_block = False
    for raw in io.StringIO(text.lstrip("﻿")):
        toks = _split(raw)
        if len(toks) >= 2 and _NUM.match(toks[0]) and _NUM.match(toks[1]):
            # a third column groups points by body; otherwise blocks separate bodies
            # (an integer or a name); a fractional third column is a z coordinate: ignored
            grouped = len(toks) >= 3 and (not _NUM.match(toks[2]) or float(toks[2]).is_integer())
            key = f"#{block}/id:{toks[2]}" if grouped else f"#{block}"
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append([float(toks[0]), float(toks[1])])
            in_block = True
        else:
            if in_block:
                block += 1
            in_block = False
    out = []
    for k in order:
        P = remove_duplicates(as_polygon(groups[k]))
        if len(P) >= 3 and abs(signed_area(P)) > 0:
            out.append(orient_ccw(P))
        elif notes is not None:
            notes.append(f"Ignored a point group with {len(P)} usable point(s).")
    return out
