"""Derived 3D fields and point sampling (outputs only, no effect on the dynamics).

* cell-centred velocity components and speed from the staggered faces;
* velocity-gradient fields - vorticity magnitude, Q-criterion, eddy viscosity -
  evaluated on blocks of cells so a 12 M-cell field never needs more than
  ~``BLOCK_CELLS`` cells' worth of gradient temporaries at once;
* trilinear interpolation of a staggered field at arbitrary points (surface
  probes, wake lines).

Q-criterion: ``Q = (|Omega|^2 - |S|^2) / 2 = -1/2 g_ij g_ji`` with
``g_ij = du_i/dx_j`` (Hunt, Wray & Moin 1988); positive where rotation dominates
strain, the standard vortex-core indicator. Vorticity magnitude is ``|curl u|``.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import Ops3D, eddy_viscosity3d, pad_velocity, traceless, velocity_gradient

#: cells per block of gradient work (each block holds ~20 temporaries of its size)
BLOCK_CELLS = 1 << 21
GRADIENT_FIELDS = ("vorticity", "q", "nut")


def centre_u(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * (u[..., 1:] + u[..., :-1])


def centre_v(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * (v[:, 1:] + v[:, :-1])


def centre_w(w: torch.Tensor) -> torch.Tensor:
    return 0.5 * (w[1:] + w[:-1])


def _block(U, V, W, axis: int, a0: int, a1: int):
    """Padded sub-arrays covering cells a0..a1-1 along ``axis`` (0 = z, 1 = y, 2 = x)."""
    def cut(P, face: bool):
        return P.narrow(axis, a0, a1 - a0 + (5 if face else 4))
    return cut(U, axis == 2), cut(V, axis == 1), cut(W, axis == 0)


def gradient_field(name: str, u, v, w, ops: Ops3D, *, mask: torch.Tensor | None = None, axis: int = 0,
                   lo: int = 0, hi: int | None = None) -> torch.Tensor:
    """``"vorticity"`` (|curl u|), ``"q"`` or ``"nut"`` on cells lo..hi-1 along ``axis``
    (all cells along the other axes). ``mask`` (cell-centred solid fraction over
    the same block) switches the eddy viscosity off inside solids."""
    if name not in GRADIENT_FIELDS:
        raise KeyError(name)
    n_axis = (ops.nz, ops.ny, ops.nx)[axis]
    hi = n_axis if hi is None else hi
    U, V, W = pad_velocity(u, v, w, ops)
    plane = ops.nx * ops.ny * ops.nz // n_axis
    step = max(1, BLOCK_CELLS // max(plane, 1))
    out = []
    for a0 in range(lo, hi, step):
        a1 = min(hi, a0 + step)
        g = velocity_gradient(*_block(U, V, W, axis, a0, a1), ops.h)["g"]
        g = [[g[i][j][1:-1, 1:-1, 1:-1] for j in range(3)] for i in range(3)]
        if name == "q":
            f = -0.5 * sum(g[i][j] * g[j][i] for i in range(3) for j in range(3))
        elif name == "vorticity":
            ox, oy, oz = g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]
            f = torch.sqrt(ox * ox + oy * oy + oz * oz)
        else:
            if ops.turbulence == "none":
                shape = list(g[0][0].shape)
                f = torch.zeros(shape, device=u.device, dtype=u.dtype)
            else:
                f = eddy_viscosity3d(traceless(g), ops)
                if mask is not None:
                    f = f * (1.0 - mask.narrow(axis, a0 - lo, a1 - a0))
        out.append(f)
    return torch.cat(out, dim=axis)


def trilinear(field: torch.Tensor, pts: torch.Tensor, ops: Ops3D,
              off: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> torch.Tensor:
    """Trilinear interpolation of a grid field at points (P, 3) = (x, y, z).

    ``field[k, j, i]`` sits at ``(x0 + (i + ox) h, y0 + (j + oy) h, z0 + (k + oz) h)``:
    (0.5, 0.5, 0.5) cell centres, (0, .5, .5) u-faces, (.5, 0, .5) v-faces,
    (.5, .5, 0) w-faces - staggered velocities are interpolated where they live.
    Points outside are clamped to the edge.
    """
    nz, ny, nx = field.shape
    h = ops.h
    idx = []
    for c, origin, n, o in ((0, ops.x0, nx, off[0]), (1, ops.y0, ny, off[1]), (2, ops.z0, nz, off[2])):
        f = ((pts[:, c] - origin) / h - o).clamp(0.0, max(n - 1.0, 0.0))
        i0 = f.floor().clamp(max=max(n - 2, 0)).long()
        idx.append((i0, (f - i0).to(field.dtype), min(1, n - 1)))
    (i0, tx, sx), (j0, ty, sy), (k0, tz, sz) = idx

    def at(dk, dj, di):
        return field[k0 + dk * sz, j0 + dj * sy, i0 + di * sx]
    c00 = at(0, 0, 0) * (1 - tx) + at(0, 0, 1) * tx
    c01 = at(0, 1, 0) * (1 - tx) + at(0, 1, 1) * tx
    c10 = at(1, 0, 0) * (1 - tx) + at(1, 0, 1) * tx
    c11 = at(1, 1, 0) * (1 - tx) + at(1, 1, 1) * tx
    return (c00 * (1 - ty) + c01 * ty) * (1 - tz) + (c10 * (1 - ty) + c11 * ty) * tz


def vertex_normals(vertices, faces):
    """Area-weighted unit vertex normals (numpy, float64); zero for unused vertices."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    norm = np.linalg.norm(n, axis=1)
    return n / np.where(norm > 0, norm, 1.0)[:, None]


def probe_distances(ops: Ops3D) -> tuple[float, float]:
    """Wall-normal probe distances used by surface sampling (as in 2D: just
    outside the diffuse layer, which ends (w - shift) h = 0.68 h off the wall)."""
    d1 = max(1.5, ops.mask_width + 0.5) * ops.h
    return d1, d1 + ops.h


__all__ = [
    "BLOCK_CELLS",
    "GRADIENT_FIELDS",
    "centre_u",
    "centre_v",
    "centre_w",
    "gradient_field",
    "probe_distances",
    "trilinear",
    "vertex_normals",
]

