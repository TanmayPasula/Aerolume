"""3D field exports: VTK ImageData ``.vti`` and NumPy ``.npz`` of a :class:`VolumeData`.

A volume block is a uniform grid of (nz, ny, nx) cells, which is VTK's
ImageData: the point extent is ``0 nx 0 ny 0 nz``, ``Origin`` the min corner of
the outer cell edges and ``Spacing`` the cell size, all in metres, wind frame.
Flattening an (nz, ny, nx) array row-major gives VTK's x-fastest cell order.
Every field becomes a Float32 cell-data array; ``velocity`` (u, v, w) is added
for ParaView's Glyph and Stream Tracer filters, and the solid fraction as
``mask``.

**Symmetry plane.** A run that used the symmetry plane y = 0 holds only the
y >= 0 half (``VolumeData.mirrored`` True). ``mirror=None`` (default) writes the
full body by reflecting it, so ParaView shows what the 3D viewer shows; the
reflection flips the sign of the fields that are odd under y -> -y (the
y-velocity ``v`` and the x / z vorticity components), see
:data:`ODD_IN_Y`. ``mirror=False`` writes the simulated half as it is.
"""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from ..model import VolumeData
from ..project.fileio import atomic_writer
from .vtkxml import VtkXmlWriter

#: fields whose sign flips under the reflection y -> -y (velocity v; pseudo-vector components)
ODD_IN_Y = frozenset({"v", "omega_x", "omega_z", "vorticity_x", "vorticity_z", "tau_y"})


def _shape(vol: VolumeData) -> tuple[int, int, int]:
    shapes = {np.shape(a) for a in vol.fields.values()}
    if vol.mask is not None:
        shapes.add(np.shape(vol.mask))
    if len(shapes) != 1:
        raise ValueError(f"volume fields have different shapes: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 3:
        raise ValueError(f"volume fields must be 3D (nz, ny, nx), got shape {shape}")
    return shape  # type: ignore[return-value]


def mirror_volume(vol: VolumeData) -> VolumeData:
    """The full-body volume of a symmetry-plane run (see the module docstring).

    The simulated half must start at the plane: its y origin within half a cell of
    0. A volume that is not ``mirrored`` is returned unchanged.
    """
    if not vol.mirrored:
        return vol
    _, ny, _ = _shape(vol)
    dy = float(vol.spacing[1])
    if abs(float(vol.origin[1])) > 0.5 * dy:
        raise ValueError(f"cannot mirror: the volume starts at y = {vol.origin[1]:.6g} m, not on the "
                         "symmetry plane y = 0")

    def refl(name: str, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        half = -a[:, ::-1, :] if name in ODD_IN_Y else a[:, ::-1, :]
        return np.concatenate([half, a], axis=1)

    fields = {k: refl(k, a) for k, a in vol.fields.items()}
    mask = refl("mask", vol.mask) if vol.mask is not None else None
    origin = (float(vol.origin[0]), -ny * dy, float(vol.origin[2]))
    return VolumeData(origin=origin, spacing=tuple(vol.spacing), fields=fields, mask=mask,  # type: ignore[arg-type]
                      label=vol.label, mirrored=False)


def cell_centres3(vol: VolumeData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D x, y, z coordinates [m] of the volume's cell centres."""
    nz, ny, nx = _shape(vol)
    (x0, y0, z0), (dx, dy, dz) = vol.origin, vol.spacing
    return (x0 + dx * (np.arange(nx) + 0.5), y0 + dy * (np.arange(ny) + 0.5), z0 + dz * (np.arange(nz) + 0.5))


def volume_vti(volume: VolumeData, path: str | os.PathLike, *, mirror: bool | None = None,
               encoding: str = "appended", compress: bool = True, velocity: bool = True) -> Path:
    """Write a :class:`VolumeData` as VTK XML ImageData (``.vti``) for ParaView.

    ``mirror``: None (default) reflects a symmetry-plane half to the full body,
    False writes it as simulated, True requires a ``mirrored`` volume and
    reflects it. ``compress`` (default) deflates the blocks with VTK's zlib
    compressor (a 2 M-cell volume is ~50 MB of float32 otherwise);
    ``velocity`` adds the (u, v, w) vector when all three exist.
    """
    if mirror is True and not volume.mirrored:
        raise ValueError("mirror=True needs a volume from a symmetry-plane run (VolumeData.mirrored)")
    vol = mirror_volume(volume) if (mirror is None or mirror) else volume
    nz, ny, nx = _shape(vol)
    (x0, y0, z0), (dx, dy, dz) = vol.origin, vol.spacing
    arrays: list[tuple[str, int, np.ndarray]] = [(n, 1, np.asarray(a)) for n, a in vol.fields.items()]
    if vol.mask is not None and "mask" not in vol.fields:
        arrays.append(("mask", 1, np.asarray(vol.mask)))
    if velocity and all(c in vol.fields for c in ("u", "v", "w")) and "velocity" not in vol.fields:
        arrays.append(("velocity", 3, np.stack([np.asarray(vol.fields[c]).ravel() for c in ("u", "v", "w")],
                                               axis=1)))
    scalars = next((n for n in ("cp", "speed", "p", "u") if n in vol.fields), None)
    attrs = (f" Scalars={quoteattr(scalars)}" if scalars else "") + \
        (' Vectors="velocity"' if any(n == "velocity" for n, _, _ in arrays) else "")
    w = VtkXmlWriter("ImageData", encoding=encoding, compress=compress)
    w.line(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="{float(x0)!r} {float(y0)!r} {float(z0)!r}" '
           f'Spacing="{float(dx)!r} {float(dy)!r} {float(dz)!r}">')
    w.line(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">')
    w.line(f"      <CellData{attrs}>")
    for name, ncomp, a in arrays:
        w.array(name, a, ncomp=ncomp)
    for t in ("      </CellData>", "    </Piece>", "  </ImageData>"):
        w.line(t)
    return w.write(path)


def volume_npz(volume: VolumeData, path: str | os.PathLike, *, mirror: bool | None = None) -> Path:
    """All fields (float32, (nz, ny, nx)) plus ``mask``, ``origin``, ``spacing``, cell-centre
    ``x`` / ``y`` / ``z``, ``label`` and ``mirrored``; ``mirror`` as in :func:`volume_vti`."""
    if mirror is True and not volume.mirrored:
        raise ValueError("mirror=True needs a volume from a symmetry-plane run (VolumeData.mirrored)")
    vol = mirror_volume(volume) if (mirror is None or mirror) else volume
    x, y, z = cell_centres3(vol)
    arrays = {name: np.asarray(a, dtype=np.float32) for name, a in vol.fields.items()}
    for reserved in ("origin", "spacing", "x", "y", "z", "label", "mask", "mirrored"):
        if reserved in arrays:
            raise ValueError(f"field name {reserved!r} is reserved in the npz export")
    arrays.update(origin=np.asarray(vol.origin, dtype=np.float64), spacing=np.asarray(vol.spacing, dtype=np.float64),
                  x=x, y=y, z=z, label=np.array(vol.label), mirrored=np.array(bool(vol.mirrored)))
    if vol.mask is not None:
        arrays["mask"] = np.asarray(vol.mask, dtype=np.float32)
    with atomic_writer(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    return Path(path)


__all__ = ["ODD_IN_Y", "cell_centres3", "mirror_volume", "volume_npz", "volume_vti"]
