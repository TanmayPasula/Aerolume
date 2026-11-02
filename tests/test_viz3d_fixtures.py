"""Synthetic 3D data for the viz3d tests and sample renders (no solver needed).

Every field here is analytic, so a test can say exactly which colour must
appear where: potential flow around a sphere (stagnation Cp = 1 at the nose,
Cp = -1.25 at the equator), a Lamb-Oseen vortex pair (Q > 0 only inside the
cores), and a rounded "vehicle" body (a superellipsoid with four wheels) on
the road plane for realistic-looking samples.

The module name starts with ``test_`` only so it falls under the viz3d test
files; it defines no tests itself.
"""
from __future__ import annotations

import numpy as np

from aerolume.model import Mesh3D, SliceData, SurfaceData3D, VolumeData


def uv_sphere(d: float = 1.0, n_theta: int = 48, n_phi: int = 96, centre=(0.0, 0.0, 0.0)):
    """(vertices, faces) of a sphere with its poles on the x axis (pole 0 at -x, the nose)."""
    r = d / 2.0
    th = np.linspace(0.0, np.pi, n_theta + 1)[1:-1]
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack([-r * np.cos(t), r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p)], -1).reshape(-1, 3)
    v = np.vstack([[-r, 0.0, 0.0], ring, [r, 0.0, 0.0]]) + np.asarray(centre, float)
    nr = n_theta - 1
    j = np.arange(n_phi)
    jn = (j + 1) % n_phi

    def idx(i, jj):
        return 1 + i * n_phi + jj

    faces = [np.stack([np.zeros(n_phi, int), idx(0, jn), idx(0, j)], 1),
             np.stack([np.full(n_phi, len(v) - 1), idx(nr - 1, j), idx(nr - 1, jn)], 1)]
    for i in range(nr - 1):
        a, b, c, e = idx(i, j), idx(i, jn), idx(i + 1, jn), idx(i + 1, j)
        faces += [np.stack([a, b, c], 1), np.stack([a, c, e], 1)]
    return v, np.concatenate(faces).astype(np.int64)


def sphere_cp(vertices: np.ndarray, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Potential-flow surface Cp on a sphere in a +x stream: 1 - 9/4 sin^2(theta)."""
    rel = vertices - np.asarray(centre)
    r = np.linalg.norm(rel, axis=1)
    cos_t = -rel[:, 0] / np.maximum(r, 1e-12)
    return 1.0 - 2.25 * (1.0 - cos_t ** 2)


def sphere_surface(d: float = 1.0, n_theta: int = 48, n_phi: int = 96) -> SurfaceData3D:
    v, f = uv_sphere(d, n_theta, n_phi)
    return SurfaceData3D(body="sphere", vertices=v, faces=f, cp=sphere_cp(v))


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def superellipsoid(a: float, b: float, c: float, e: float = 0.35, n_u: int = 96, n_v: int = 192,
                   centre=(0.0, 0.0, 0.0)):
    """(vertices, faces) of a rounded box |x/a|^(2/e) + |y/b|^(2/e) + |z/c|^(2/e) = 1."""
    v, f = uv_sphere(2.0, n_u, n_v)

    def spow(x, p):
        return np.sign(x) * np.abs(x) ** p
    out = np.stack([a * spow(v[:, 0], e), b * spow(v[:, 1], e), c * spow(v[:, 2], e)], 1)
    return out + np.asarray(centre, float), f


def cylinder(radius: float, width: float, centre, n: int = 48):
    """(vertices, faces) of a closed cylinder with its axis along y (a wheel)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy, cz = centre
    ring = np.stack([cx + radius * np.cos(ang), np.zeros(n), cz + radius * np.sin(ang)], 1)
    left, right = ring + [0, cy - width / 2, 0], ring + [0, cy + width / 2, 0]
    v = np.vstack([left, right, [[cx, cy - width / 2, cz], [cx, cy + width / 2, cz]]])
    j = np.arange(n)
    jn = (j + 1) % n
    faces = np.concatenate([np.stack([j, jn, n + jn], 1), np.stack([j, n + jn, n + j], 1),
                            np.stack([np.full(n, 2 * n), jn, j], 1),
                            np.stack([np.full(n, 2 * n + 1), n + j, n + jn], 1)])
    return v, faces.astype(np.int64)


def vehicle_surfaces() -> list[SurfaceData3D]:
    """A 4.4 m rounded body on four wheels, with a plausible Cp from the surface normals."""
    out = []
    v, f = superellipsoid(2.2, 0.88, 0.62, e=0.32, centre=(0.0, 0.0, 0.92))
    nx = vertex_normals(v, f)[:, 0]
    cp = np.where(nx < 0, 1.0 - 2.25 * (1.0 - nx ** 2), -0.2 - 1.05 * (1.0 - nx) ** 2)
    out.append(SurfaceData3D(body="body", vertices=v, faces=f, cp=cp))
    for i, (x, y) in enumerate(((-1.35, -0.78), (-1.35, 0.78), (1.35, -0.78), (1.35, 0.78))):
        wv, wf = cylinder(0.33, 0.24, (x, y, 0.33))
        wn = vertex_normals(wv, wf)[:, 0]
        out.append(SurfaceData3D(body=f"wheel{i}", vertices=wv, faces=wf, cp=np.clip(0.4 - wn, -1, 1) * 0.5))
    return out


def vehicle_meshes() -> list[Mesh3D]:
    return [Mesh3D(s.body, s.vertices, s.faces, role="wheel" if s.body.startswith("wheel") else "body")
            for s in vehicle_surfaces()]


def grid(n=(64, 40, 40), lo=(-1.5, -1.0, -1.0), hi=(2.5, 1.0, 1.0)):
    nx, ny, nz = n
    h = tuple((hi[i] - lo[i]) / n[i] for i in range(3))
    x = lo[0] + (np.arange(nx) + 0.5) * h[0]
    y = lo[1] + (np.arange(ny) + 0.5) * h[1]
    z = lo[2] + (np.arange(nz) + 0.5) * h[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz, h


def sphere_flow_volume(n=(64, 40, 40), lo=(-1.5, -1.0, -1.0), hi=(2.5, 1.0, 1.0), d: float = 1.0,
                       speed: float = 10.0, mirrored: bool = False) -> VolumeData:
    """Potential flow past a sphere of diameter d at the origin, solid cells masked.

    With ``mirrored`` only the y >= 0 half is returned (lo[1] is set to 0),
    as a symmetry-plane run stores it.
    """
    if mirrored:
        n = (n[0], n[1] // 2, n[2])
        lo = (lo[0], 0.0, lo[2])
    xx, yy, zz, h = grid(n, lo, hi)
    R = d / 2.0
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    r = np.maximum(r, 1e-6)
    k = R ** 3 / (2 * r ** 5)
    u = speed * (1 + R ** 3 / (2 * r ** 3) - 3 * k * xx ** 2)
    v = -speed * 3 * k * xx * yy
    w = -speed * 3 * k * xx * zz
    solid = r < R
    for a in (u, v, w):
        a[solid] = 0.0
    sp = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    cp = 1.0 - (sp / speed) ** 2
    fields = {k_: a.astype(np.float32) for k_, a in {"u": u, "v": v, "w": w, "speed": sp, "cp": cp}.items()}
    return VolumeData(origin=lo, spacing=h, fields=fields, mask=solid.astype(np.float32), mirrored=mirrored)


def vortex_pair_volume(n=(72, 48, 48), lo=(-1.0, -1.0, -1.0), hi=(3.0, 1.0, 1.0), speed: float = 10.0,
                       gamma: float = 9.0, rc: float = 0.12, half_span: float = 0.45) -> VolumeData:
    """Two counter-rotating Lamb-Oseen vortices along x (a wing's trailing pair) in a uniform stream.

    The cores sit at y = +-half_span, z = 0; Q is computed analytically from
    the in-plane velocity gradient, so Q > 0 exactly in the cores.
    """
    xx, yy, zz, h = grid(n, lo, hi)
    v = np.zeros_like(xx)
    w = np.zeros_like(xx)
    for yc, sgn in ((half_span, 1.0), (-half_span, -1.0)):
        dy, dz = yy - yc, zz
        r2 = dy ** 2 + dz ** 2 + 1e-12
        f = sgn * gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / rc ** 2))
        v += -f * dz
        w += f * dy
    u = np.full_like(xx, speed)
    dv_dz, dv_dy = np.gradient(v, h[2], h[1], axis=(0, 1))
    dw_dz, dw_dy = np.gradient(w, h[2], h[1], axis=(0, 1))
    # Q = -1/2 tr(A^2) of the in-plane velocity gradient A (u is uniform)
    q = -0.5 * (dv_dy ** 2 + dw_dz ** 2 + 2.0 * dv_dz * dw_dy)
    sp = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    vort = np.abs(dw_dy - dv_dz)
    fields = {k: a.astype(np.float32) for k, a in {"u": u, "v": v, "w": w, "speed": sp, "q": q, "vorticity": vort}.items()}
    return VolumeData(origin=lo, spacing=h, fields=fields)


def gradient_slice(axis: str = "y", position: float = 0.0, shape=(40, 80),
                   extent=(-2.0, 2.0, 0.0, 2.0)) -> SliceData:
    """A slice whose field equals the column coordinate: low on the left, high on the right."""
    nrow, ncol = shape
    c0, c1 = extent[0], extent[1]
    col = c0 + (np.arange(ncol) + 0.5) * (c1 - c0) / ncol
    vals = np.broadcast_to(col, (nrow, ncol)).astype(np.float32).copy()
    return SliceData(axis=axis, position=position, extent=extent, fields={"u": vals})
