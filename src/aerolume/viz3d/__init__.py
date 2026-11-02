"""3D visualisation: VTK scene building and off-screen rendering (DESIGN.md 7.9). No Qt.

``from aerolume.viz3d import Scene`` gives the renderer; importing the package
itself does not import VTK (about 0.3 s and 100 MB of DLLs), so modules that
only need :mod:`aerolume.viz3d.colors` or the camera maths stay light.

Typical use (a report figure)::

    from aerolume.viz3d import Scene
    s = Scene("light")
    s.add_meshes(result.surface, "cp")
    s.add_colorbar()
    s.set_camera("iso")
    image = s.render(1600, 900)        # (900, 1600, 3) uint8
    s.close()
"""
from __future__ import annotations

from typing import Any

__all__ = ["CameraState", "Scene", "SceneTheme", "backend_info", "rake_seeds", "scene_theme"]


def __getattr__(name: str) -> Any:
    if name in ("Scene", "backend_info", "rake_seeds"):
        from . import scene
        return getattr(scene, name)
    if name in ("SceneTheme", "scene_theme"):
        from . import colors
        return getattr(colors, name)
    if name == "CameraState":
        from .camera import CameraState
        return CameraState
    raise AttributeError(f"module 'aerolume.viz3d' has no attribute {name!r}")
