"""Modal dialogs of the desktop application.

Dialogs here only *build* things (a :class:`~aerolume.model.Design` from a file,
a NACA code or a template); adding the result to the project is always done by
the calling page through :class:`~aerolume.app.state.AppState`, so dialogs stay
testable without a project and never touch persistence. The one helper that
adds designs itself, :func:`~.mesh_import.import_mesh_dialog`, does so through
``AppState`` too.
"""
from __future__ import annotations

from .import_dialogs import (
    FILE_FILTER,
    ImageSilhouetteDialog,
    NacaDialog,
    PolygonImportDialog,
    TemplateGalleryDialog,
    classify_path,
)
from .mesh_import import MeshImportDialog, import_mesh_dialog, is_mesh_path

__all__ = [
    "FILE_FILTER",
    "ImageSilhouetteDialog",
    "MeshImportDialog",
    "NacaDialog",
    "PolygonImportDialog",
    "TemplateGalleryDialog",
    "classify_path",
    "import_mesh_dialog",
    "is_mesh_path",
]
