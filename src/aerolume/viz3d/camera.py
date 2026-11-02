"""Camera state and the CAD-style camera operations of the 3D viewer (numpy only).

The camera is kept as a small value object (:class:`CameraState`) and every
operation is a pure function returning a new state. The scene copies the state
into its ``vtkCamera`` right before rendering. Keeping the maths out of VTK makes
the behaviour testable without an OpenGL context and keeps it identical for the
widget, for scripted report figures and for tests.

Conventions (DESIGN.md 7.2): x downstream, y to the left, z up; the flow runs
left to right in the side and top views, like every 2D view in the app.

Interaction model (the WHY):

* **Turntable orbit about world z.** Vehicles and wings have a natural "up";
  a free trackball lets the model roll and users lose the horizon, which CAE
  tools avoid. Elevation is clamped to +-90 deg so the view never flips.
* **Pan in the view plane at the focal distance**, scaled so the point under
  the cursor follows the cursor exactly.
* **Zoom towards the cursor.** Wheel zoom moves the camera towards the point
  under the cursor on the focal plane (keeping it fixed on screen), the
  behaviour of ParaView, Fusion and most CAD tools.
* **Fit is exact for the current aspect ratio.** ``vtkRenderer.ResetCamera``
  fits a bounding *sphere*, which leaves a long car small in a wide window.
  :func:`fit` instead solves for the distance at which every corner of the box
  lies inside the perspective frustum with a small margin, and re-centres the
  projected box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["PRESETS", "CameraState", "dolly", "fit", "orbit", "pan", "preset", "project", "unproject", "zoom"]

_Z = np.array([0.0, 0.0, 1.0])

#: name -> (unit vector from the focal point to the camera, view-up)
PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # from upstream-left-above: front, -y side and top visible, nose on the left
    "iso": ((-1.0, -1.15, 0.85), (0.0, 0.0, 1.0)),
    "front": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),       # looking downstream at the frontal area
    "back": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "side": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),        # flow left to right, z up
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),          # flow left to right, +y up the screen
    "bottom": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass(frozen=True)
class CameraState:
    """A perspective (or parallel) camera: position, focal point, view-up, angle.

    ``view_angle`` is the vertical field of view in degrees (VTK's meaning);
    ``parallel_scale`` is half the view height in metres when ``parallel``.
    """
    position: tuple[float, float, float] = (-3.0, -3.0, 2.0)
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    view_angle: float = 25.0
    parallel: bool = False
    parallel_scale: float = 1.0

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(np.subtract(self.position, self.focal_point)))

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal (right, up, forward) of the view; forward points into the screen."""
        f = _unit(np.subtract(self.focal_point, self.position))
        r = np.cross(f, self.view_up)
        if np.linalg.norm(r) < 1e-9:            # view-up parallel to the view direction
            r = np.cross(f, [0.0, 1.0, 0.0] if abs(f[1]) < 0.9 else [1.0, 0.0, 0.0])
        r = _unit(r)
        u = np.cross(r, f)
        return r, u, f

    def half_height(self) -> float:
        """Half the view height at the focal distance [m] (world units per half screen)."""
        if self.parallel:
            return float(self.parallel_scale)
        return self.distance * math.tan(math.radians(self.view_angle) / 2.0)


def preset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(direction to camera, view-up) of a named preset ('iso', 'front', 'side', ...)."""
    try:
        d, up = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown camera preset {name!r}; choose from {', '.join(PRESETS)}") from None
    return _unit(d), _unit(up)


def _corners(bounds) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = (float(b) for b in bounds)
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def fit(state: CameraState, bounds, aspect: float, *, margin: float = 0.12,
        direction=None, view_up=None, points=None) -> CameraState:
    """Frame the box ``bounds`` = (x0, x1, y0, y1, z0, z1) for a ``aspect`` = W / H view.

    ``points`` (N, 3), when given, are framed instead of the box corners: a
    sample of the actual mesh vertices frames a sphere or a car tightly, where
    the eight corners of its box would leave it small in an oblique view.

    Keeps the view direction (or uses ``direction``/``view_up`` from a preset)
    and solves for the camera distance at which all eight corners fit inside
    the frustum with ``margin`` (fraction of the half-size) to spare, then
    shifts the focal point so the projected box is centred. Three fixed-point
    iterations converge to well under a pixel for any box.
    """
    aspect = max(float(aspect), 1e-3)
    if direction is None:
        _, u0, f0 = state.frame()
        back = -f0
        up = u0 if view_up is None else _unit(view_up)
    else:
        back = _unit(direction)
        up = _unit(view_up if view_up is not None else _Z)
    corners = _corners(bounds) if points is None or len(points) == 0 else np.asarray(points, float)
    centre = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
    size = float(np.linalg.norm(corners.max(0) - corners.min(0)))
    if size <= 0:
        size = 1.0
    f = -back
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, [0.0, 1.0, 0.0] if abs(f[1]) < 0.9 else [1.0, 0.0, 0.0])
    r = _unit(r)
    u = np.cross(r, f)
    tan_v = math.tan(math.radians(state.view_angle) / 2.0)
    tan_h = tan_v * aspect
    scale = 1.0 + margin
    if state.parallel:
        rel = corners - centre
        xs, ys = rel @ r, rel @ u
        cx, cy = 0.5 * (xs.max() + xs.min()), 0.5 * (ys.max() + ys.min())
        centre = centre + cx * r + cy * u
        half = max(0.5 * (xs.max() - xs.min()) / aspect, 0.5 * (ys.max() - ys.min())) * scale
        pos = centre + back * size * 2.0
        return replace(state, position=tuple(pos), focal_point=tuple(centre), view_up=tuple(u),
                       parallel_scale=float(max(half, 1e-9)))
    d = size
    for _ in range(3):
        rel = corners - centre
        xs, ys, zs = rel @ r, rel @ u, rel @ back          # zs > 0: towards the camera
        d = float(np.max(np.maximum(zs + np.abs(xs) * scale / tan_h, zs + np.abs(ys) * scale / tan_v)))
        depth = np.maximum(d - zs, 1e-9)
        nx, ny = xs / depth, ys / depth                    # tangent-space screen coordinates
        cx, cy = 0.5 * (nx.max() + nx.min()), 0.5 * (ny.max() + ny.min())
        centre = centre + (cx * d) * r + (cy * d) * u
    pos = centre + back * d
    return replace(state, position=tuple(pos), focal_point=tuple(centre), view_up=tuple(u))


def _rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` about the unit ``axis``."""
    k = _unit(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1 - c)


def orbit(state: CameraState, d_azimuth_deg: float, d_elevation_deg: float) -> CameraState:
    """Turntable orbit about the focal point: azimuth about world z, then elevation.

    Positive azimuth moves the camera counter-clockwise seen from above
    (dragging the mouse right turns the model to the right); positive
    elevation raises the camera. Elevation is clamped at +-90 deg.
    """
    fp = np.asarray(state.focal_point, float)
    off = np.subtract(state.position, fp)
    up = np.asarray(state.view_up, float)
    a = math.radians(-float(d_azimuth_deg))
    off, up = _rotate(off, _Z, a), _rotate(up, _Z, a)
    dist = float(np.linalg.norm(off))
    if dist > 0 and d_elevation_deg:
        elev = math.degrees(math.asin(max(-1.0, min(1.0, off[2] / dist))))
        new = max(-90.0, min(90.0, elev + float(d_elevation_deg)))
        delta = math.radians(new - elev)
        if abs(delta) > 0:
            right = np.cross(-off, up)
            if np.linalg.norm(right) < 1e-12:
                right = np.cross(-off, _Z) if abs(off[2]) < 0.999 * dist else np.array([0.0, 1.0, 0.0])
            # rotating about "right" by -delta raises the camera
            off, up = _rotate(off, right, -delta), _rotate(up, right, -delta)
    return replace(state, position=tuple(fp + off), view_up=tuple(_unit(up)))


def pan(state: CameraState, dx_px: float, dy_px: float, height_px: float) -> CameraState:
    """Move the view so the scene follows a drag of (dx, dy) screen pixels (y down)."""
    r, u, _ = state.frame()
    wpp = 2.0 * state.half_height() / max(float(height_px), 1.0)
    shift = (-float(dx_px) * r + float(dy_px) * u) * wpp
    return replace(state, position=tuple(np.add(state.position, shift)),
                   focal_point=tuple(np.add(state.focal_point, shift)))


def zoom(state: CameraState, factor: float, *, at: tuple[float, float] | None = None,
         size: tuple[float, float] | None = None, min_distance: float = 1e-6) -> CameraState:
    """Zoom by ``factor`` (> 1 closer), optionally keeping screen pixel ``at`` fixed.

    In perspective this dollies the camera (the field of view never changes,
    so there is no fish-eye at high zoom); in parallel projection it scales the
    view height. ``at`` = (x, y) pixels with y down, ``size`` = (W, H) pixels.
    """
    factor = float(factor)
    if not factor > 0:
        return state
    fp = np.asarray(state.focal_point, float)
    r, u, f = state.frame()
    target = fp
    if at is not None and size is not None:
        w, h = max(size[0], 1.0), max(size[1], 1.0)
        hh = state.half_height()
        sx = (2.0 * at[0] / w - 1.0) * hh * (w / h)
        sy = (1.0 - 2.0 * at[1] / h) * hh
        target = fp + sx * r + sy * u
    keep = 1.0 / factor
    new_fp = target + (fp - target) * keep
    if state.parallel:
        shift = new_fp - fp
        return replace(state, focal_point=tuple(new_fp), position=tuple(np.add(state.position, shift)),
                       parallel_scale=float(state.parallel_scale) * keep)
    dist = max(state.distance * keep, min_distance)
    return replace(state, focal_point=tuple(new_fp), position=tuple(new_fp - f * dist))


def dolly(state: CameraState, factor: float) -> CameraState:
    """Move the camera towards the focal point (``factor`` > 1 closer) along the view axis."""
    return zoom(state, factor)


def project(state: CameraState, points, width: float, height: float) -> np.ndarray:
    """Pixel coordinates (x right, y down) of world ``points`` (N, 3); NaN behind the camera."""
    p = np.atleast_2d(np.asarray(points, float))
    r, u, f = state.frame()
    rel = p - np.asarray(state.position, float)
    x, y, z = rel @ r, rel @ u, rel @ f
    hh = math.tan(math.radians(state.view_angle) / 2.0)
    aspect = width / max(height, 1.0)
    if state.parallel:
        sx = x / (state.parallel_scale * aspect)
        sy = y / state.parallel_scale
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = np.where(z > 0, x / (z * hh * aspect), np.nan)
            sy = np.where(z > 0, y / (z * hh), np.nan)
    return np.stack([(sx + 1.0) * 0.5 * width, (1.0 - sy) * 0.5 * height], axis=-1)


def unproject(state: CameraState, px: float, py: float, width: float, height: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """The viewing ray through pixel (px, py) (x right, y down): (origin, unit direction).

    The inverse of :func:`project`; used to drag things (the seed rake) so the
    grabbed point stays exactly under the cursor.
    """
    r, u, f = state.frame()
    sx = 2.0 * float(px) / max(width, 1.0) - 1.0
    sy = 1.0 - 2.0 * float(py) / max(height, 1.0)
    aspect = width / max(height, 1.0)
    pos = np.asarray(state.position, float)
    if state.parallel:
        origin = pos + sx * state.parallel_scale * aspect * r + sy * state.parallel_scale * u
        return origin, f.copy()
    hh = math.tan(math.radians(state.view_angle) / 2.0)
    d = f + sx * hh * aspect * r + sy * hh * u
    return pos, d / np.linalg.norm(d)
