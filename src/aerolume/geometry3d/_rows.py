"""Fast ``np.unique(rows, axis=0)`` replacement used by the mesh topology code.

``np.unique(..., axis=0)`` views each row as an opaque structured scalar and sorts
those, which is ~10x slower than a ``lexsort`` of the columns. Mesh topology
(welding identical vertices, counting edges, finding duplicate faces) does this
on every import and every voxelisation, on arrays of a million rows, so it is
worth the few lines. Float rows are compared by their bit patterns after
``+ 0.0`` (which folds -0.0 into 0.0), i.e. exact equality, as the welders need.
The order of the unique rows is not numeric; callers only use the grouping.
"""
from __future__ import annotations

import numpy as np

__all__ = ["unique_rows"]


def unique_rows(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(unique rows, index of first occurrence, inverse, counts)`` of a 2D array.

    Like ``np.unique(a, axis=0, return_index=True, return_inverse=True,
    return_counts=True)`` except for the order of the groups. ``first`` is the
    smallest original index of each group (lexsort is stable).
    """
    a = np.ascontiguousarray(a)
    n = a.shape[0]
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return a.copy(), z, z, z
    if a.dtype.kind == "f":
        b = np.ascontiguousarray(a.astype(np.float64) + 0.0).view(np.int64)
    else:
        b = a.astype(np.int64, copy=False)
    b = b.reshape(n, -1)
    order = np.lexsort(b.T[::-1])
    sb = b[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.any(sb[1:] != sb[:-1], axis=1, out=new[1:])
    grp = np.cumsum(new) - 1
    inv = np.empty(n, dtype=np.int64)
    inv[order] = grp
    starts = np.flatnonzero(new)
    first = order[starts]
    counts = np.diff(np.append(starts, n))
    return a[first], first, inv, counts
