"""The project store: a folder of small JSON files plus one ``.npz`` per result.

Layout (DESIGN.md 5.7)::

    <Name>.aerolume/
      project.json                 name, schema, timestamps, design + simulation ids (order)
      designs/<id>.json            Design.to_dict() (+ "mesh_file" pairing record for 3D designs)
      designs/<id>.npz             3D designs: mesh vertex / face arrays (project/meshes.py)
      sims/<id>/simulation.json    Simulation.to_dict()
      sims/<id>/result.json        result scalars (see project/results.py)
      sims/<id>/arrays.npz         histories, surfaces, fields
      reports/                     generated reports
      scripts/                     the user's Python scripts (Script console, DESIGN.md 8)
      experiments/<id>/            AI assistant experiment logs (DESIGN.md 9)

The store owns project.json, designs/ and sims/; reports/, scripts/ and
experiments/ belong to their tools and are never rewritten or pruned by it.

Why a folder of files rather than one archive: a result is written by a worker
thread while the user keeps editing designs, a crash must never take the whole
project with it, and git / OneDrive / Dropbox sync small text files well.

Consistency rules:

* Every write is atomic (``fileio.atomic_writer``). ``save()`` only rewrites files
  whose serialised content changed, so calling it after every edit (as
  ``app/state.py`` does) costs a few JSON encodes, not a rewrite of the project.
* Removals and results are written *immediately*: removing a simulation deletes its
  folder and rewrites ``project.json`` at once, and ``save_result`` is durable when
  it returns. Additions and edits become durable on the next ``save()``.
* 3D designs (DESIGN.md 7.10) keep their mesh arrays in ``designs/<id>.npz``,
  written before the JSON and only when the arrays changed (a SHA-1 of the arrays
  is kept, see ``project/meshes.py``). ``Project.open`` reads them back, so
  ``designs[id].meshes`` carry their vertices and faces like any in-memory design;
  a 3D design whose mesh file is missing or does not fit is skipped with a load
  warning and its files are left untouched (loading it without geometry and saving
  would make the loss permanent).
* One ``threading.RLock`` guards every method, because the job manager saves results
  from a worker thread while the GUI thread edits the same project.
* Loading is forward compatible: unknown keys are ignored (and unknown top-level keys
  of ``project.json`` are written back unchanged), a newer schema opens with a
  warning, and design / simulation files missing from the index (a crash between
  writing a file and saving the index) are recovered. A broken individual design or
  simulation file is skipped with a message in ``load_warnings``; a broken
  ``project.json`` or result raises :class:`ProjectCorruptError`.

Schema policy (``model.SCHEMA_VERSION``): a project is written as schema 1 while it
holds 2D designs only -- byte for byte what Aerolume 0.1 wrote, so 0.1 keeps opening
it -- and as schema 2 as soon as it holds a 3D design. Aerolume 0.1 warns about
schema 2 ("saved by a newer Aerolume") but would otherwise read a 3D design file as a
2D design without bodies and, after an edit, write it back without its dimension and
meshes. A 3D design file therefore leaves out the ``bodies`` key that 0.1 requires
(3D designs have none): 0.1 skips such a file with a load warning and never rewrites
it, and this version, which recovers design files missing from the index, gets it
back intact even after 0.1 has saved the project.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..model import SCHEMA_VERSION, SCHEMA_VERSION_2D, Design, Simulation, new_id, now_iso
from . import meshes as _meshes
from . import provenance as _prov
from .fileio import (
    ProjectCorruptError,
    ProjectError,
    ProjectExistsError,
    ProjectNotFoundError,
    atomic_writer,
    dumps_json,
    npz_bytes,
    read_json,
    remove_tree,
    sweep_trash,
    write_json,
)
from .results import AnyResult, pack_result, result_type, unpack_result

PROJECT_FORMAT = "aerolume-project"
SUFFIX = ".aerolume"
_INDEX = "project.json"
_KNOWN_KEYS = {"format", "schema", "app_version", "name", "description", "created", "modified", "designs",
               "simulations", "comparisons", "settings"}


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _why(exc: BaseException) -> str:
    """A short reason for a load warning (the exception text, or its type when it has none)."""
    text = str(exc).strip()
    return text if text else type(exc).__name__


def _id_list(value, what: str, warnings: list[str]) -> list[str]:
    """The design / simulation ids of project.json, keeping only strings.

    A damaged index must not be split into characters ("abc" -> a, b, c) or
    stop the project from opening; the item files on disk are recovered anyway.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append(f"The project's list of {what}s was unreadable; {what}s found on disk are recovered.")
        return []
    ids = [v for v in value if isinstance(v, str) and v]
    if len(ids) != len(value):
        warnings.append(f"Some entries of the project's list of {what}s were unreadable and were skipped.")
    return ids


class Project:
    """An open project: designs, simulations and their results on disk.

    Build one with :meth:`create` or :meth:`open`. ``designs`` and ``simulations``
    are ordinary dicts in insertion order (the order the GUI lists them); mutate them
    through the methods so the files follow. ``load_warnings`` lists anything that
    was skipped or repaired while opening, for the GUI to show once.
    """

    def __init__(self, path: Path, name: str, *, created: str | None = None) -> None:
        self._path = Path(path)
        self._name = name
        self.created: str = created or now_iso()
        self.modified: str = self.created
        self.designs: dict[str, Design] = {}
        self.simulations: dict[str, Simulation] = {}
        self.comparisons: list[dict] = []     # saved comparison sets, free-form for the app
        self.settings: dict = {}              # per-project app settings (units, last page, ...)
        #: one-line summary shown next to the name in the Start page's lists (examples use it)
        self.description: str = ""
        self.load_warnings: list[str] = []
        #: the schema of project.json as opened (or as last written); see the module docstring
        self.schema: int = SCHEMA_VERSION_2D
        self._extra: dict = {}                # unknown project.json keys from newer versions
        self._written: dict[str, str] = {}    # relative path -> digest of what is on disk
        self._summaries: dict[str, tuple[float, dict]] = {}
        self._lock = threading.RLock()

    # -- construction -----------------------------------------------------
    @classmethod
    def create(cls, path: str | os.PathLike, name: str | None = None) -> Project:
        """Create a new, empty project folder and write its index.

        ``.aerolume`` is appended to the folder name when missing. Creating inside an
        existing *empty* folder is allowed; a non-empty folder or an existing file is
        refused with :class:`ProjectExistsError`, so a mistyped path can never mix a
        project into someone's documents.
        """
        path = Path(path).expanduser()
        if path.suffix.lower() != SUFFIX:
            path = path.with_name(path.name + SUFFIX)
        if path.exists():
            if not path.is_dir():
                raise ProjectExistsError(f"a file already exists at {path}")
            if any(path.iterdir()):
                raise ProjectExistsError(f"refusing to create a project inside a non-empty folder: {path}")
        name = (name or path.stem).strip() or "Untitled"
        try:
            for sub in ("designs", "sims", "reports"):
                (path / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProjectError(f"cannot create project folder {path}: {exc}") from exc
        proj = cls(path.resolve(), name)
        proj.save()
        return proj

    @classmethod
    def open(cls, path: str | os.PathLike) -> Project:
        """Open an existing project (the folder, or its ``project.json``)."""
        path = Path(path).expanduser()
        if path.is_file() and path.name == _INDEX:
            path = path.parent
        if not path.exists() and path.suffix.lower() != SUFFIX and path.with_name(path.name + SUFFIX).exists():
            path = path.with_name(path.name + SUFFIX)
        if not path.is_dir():
            raise ProjectNotFoundError(f"no project folder at {path}")
        meta = read_json(path / _INDEX, "project index (project.json)")
        if not isinstance(meta, dict) or meta.get("format") != PROJECT_FORMAT:
            raise ProjectCorruptError(f"not an Aerolume project index: {path / _INDEX}")
        proj = cls(path.resolve(), str(meta.get("name") or path.stem), created=meta.get("created"))
        proj.modified = str(meta.get("modified") or proj.created)
        schema = meta.get("schema", 1)
        if isinstance(schema, int) and not isinstance(schema, bool):
            proj.schema = schema
        if isinstance(schema, int) and schema > SCHEMA_VERSION:
            proj.load_warnings.append(
                f"This project was saved by a newer Aerolume (schema {schema}, this version reads "
                f"{SCHEMA_VERSION}); features it does not know about are ignored, and saving it here "
                "may drop them. Keep a copy if you still use the newer version.")
        # Every field is type-checked: one damaged entry must cost that entry, not the
        # project ('settings': [1, 2] made dict() raise; 'comparisons': 'abc' became
        # ['a', 'b', 'c'] without a word).
        comps = meta.get("comparisons")
        if isinstance(comps, list):
            proj.comparisons = [c for c in comps if isinstance(c, dict)]
            if len(proj.comparisons) != len(comps):
                proj.load_warnings.append("Some saved comparisons were unreadable and were dropped.")
        elif comps:
            proj.load_warnings.append("The saved comparisons were unreadable and were reset.")
        settings = meta.get("settings")
        if isinstance(settings, dict):
            proj.settings = dict(settings)
        elif settings:
            proj.load_warnings.append("The project settings were unreadable and were reset.")
        proj.description = str(meta.get("description") or "")
        proj._extra = {k: v for k, v in meta.items() if k not in _KNOWN_KEYS}
        sweep_trash(path / "sims")
        sweep_trash(path / "designs")
        sweep_trash(path)
        proj._load_items(_id_list(meta.get("designs"), "design", proj.load_warnings),
                         _id_list(meta.get("simulations"), "simulation", proj.load_warnings))
        proj._pin_reynolds_caps()
        return proj

    def _pin_reynolds_caps(self) -> None:
        """Give capped simulations saved without a cap the cap their result was run at.

        Projects from before the cap was stored with the simulation let the
        planner pick it from the grid, so running such a simulation at another
        quality (or on another machine) would silently simulate a different
        Reynolds number. Pinning it to the stored result's ``re_sim`` keeps every
        later run on the flow that result describes. In memory only; written with
        the next ``save()``.
        """
        for sid, sim in self.simulations.items():
            cfd = sim.cfd
            if sim.analysis != "cfd" or cfd.reynolds_mode != "capped" or cfd.reynolds_cap:
                continue
            try:
                summ = self.result_summary(sid)
            except (ProjectError, OSError, ValueError):
                continue
            re_sim = (summ or {}).get("re_sim")
            if isinstance(re_sim, (int, float)) and math.isfinite(re_sim) and re_sim > 0:
                cfd.reynolds_cap = float(round(re_sim))

    def _load_items(self, design_ids: list[str], sim_ids: list[str]) -> None:
        ddir, sdir = self._path / "designs", self._path / "sims"
        on_disk_d = sorted(p.stem for p in ddir.glob("*.json")) if ddir.is_dir() else []
        on_disk_s = sorted(p.name for p in sdir.iterdir()
                           if (p / "simulation.json").is_file()) if sdir.is_dir() else []
        for did in design_ids + [d for d in on_disk_d if d not in design_ids]:
            f = ddir / f"{did}.json"
            if not f.is_file():
                self.load_warnings.append(f"Design {did} is listed but its file is missing; skipped.")
                continue
            try:
                raw = read_json(f, f"design {did}")
                d = _meshes.design_from_document(raw)
            except Exception as exc:  # noqa: BLE001 - any damaged file costs that file only
                self.load_warnings.append(f"Design file {f.name} could not be read ({_why(exc)}); skipped.")
                continue
            if _meshes.has_mesh_data(d) and self._attach_meshes(did, d, raw) is None:
                continue
            if did not in design_ids:
                self.load_warnings.append(f"Recovered design '{d.name}' that was not in the project index.")
            d.id = did
            self.designs[did] = d
            # the digest of what is on disk (with the mesh digest the file records), so a file
            # whose record is stale or missing is rewritten by the next save() -- as is a 3D
            # design file from before the schema policy, which still carries "bodies"
            doc = _meshes.design_document(d, _meshes.recorded_digest(raw))
            if "bodies" in raw and "bodies" not in doc:
                doc = raw
            self._written[f"designs/{did}.json"] = _digest(dumps_json(doc))
        for sid in sim_ids + [s for s in on_disk_s if s not in sim_ids]:
            f = sdir / sid / "simulation.json"
            if not f.is_file():
                self.load_warnings.append(f"Simulation {sid} is listed but its folder is missing; skipped.")
                continue
            try:
                s = Simulation.from_dict(read_json(f, f"simulation {sid}"))
            except Exception as exc:  # noqa: BLE001 - any damaged file costs that file only
                self.load_warnings.append(f"Simulation file {sid}/simulation.json could not be read "
                                          f"({_why(exc)}); skipped.")
                continue
            if sid not in sim_ids:
                self.load_warnings.append(f"Recovered simulation '{s.name}' that was not in the project index.")
            if s.design_id not in self.designs:
                self.load_warnings.append(f"Simulation '{s.name}' refers to a design that no longer exists.")
            s.id = sid
            self._written[f"sims/{sid}/simulation.json"] = _digest(dumps_json(s.to_dict()))
            if s.status in ("queued", "running"):
                # The process that was running it is gone. Changed after taking the digest,
                # so the next save() writes the corrected status.
                s.status = "done" if (sdir / sid / "result.json").is_file() else "new"
            self.simulations[sid] = s

    def _attach_meshes(self, did: str, d: Design, raw: dict) -> str | None:
        """Read ``designs/<did>.npz`` onto the design's meshes; the arrays' digest, or
        None (with a load warning) when the design must be skipped."""
        npz = self._path / "designs" / f"{did}.npz"
        counts = _meshes.json_counts(raw)
        if not npz.is_file():
            if counts is not None and all(nv == 0 and nf == 0 for nv, nf in counts):
                return _meshes.mesh_digest(d)            # meshes recorded as empty: nothing to read
            self.load_warnings.append(f"Design '{d.name}': its mesh file {npz.name} is missing; skipped.")
            return None
        try:
            arrays, stored = _meshes.read_meshes(npz)
        except Exception as exc:  # noqa: BLE001 - a damaged mesh file costs that design only
            self.load_warnings.append(f"Design '{d.name}': mesh file {npz.name} could not be read "
                                      f"({_why(exc)}); skipped.")
            return None
        why = _meshes.check_shapes(d, arrays, counts)
        if why is not None:
            self.load_warnings.append(f"Design '{d.name}': mesh file {npz.name} does not fit the design "
                                      f"({why}); skipped.")
            return None
        expected = _meshes.recorded_digest(raw)
        d.attach_mesh_arrays(arrays)
        digest = _meshes.mesh_digest(d) if stored is None else stored
        if expected is not None and stored is not None and expected != stored:
            self.load_warnings.append(
                f"Design '{d.name}': the mesh file was written by a different save than the design file "
                "(interrupted save?). The meshes fit the design and were loaded; check the geometry.")
        if stored is not None:
            self._written[f"designs/{did}.npz"] = digest   # a plain npz is rewritten in the store's format
        return digest

    # -- properties ---------------------------------------------------------
    @property
    def path(self) -> Path:
        """The project folder (``...\\Name.aerolume``)."""
        return self._path

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Rename the project (the folder keeps its name; saved on the next ``save()``)."""
        value = str(value).strip()
        if not value:
            raise ValueError("project name cannot be empty")
        with self._lock:
            self._name = value

    @property
    def reports_dir(self) -> Path:
        """Folder for generated reports (created on demand)."""
        p = self._path / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def scripts_dir(self) -> Path:
        """Folder for the project's Python scripts (DESIGN.md 8; created on demand).

        The store never reads or writes inside it: scripts belong to the user and
        to the Script console, and travel with the project folder."""
        p = self._path / "scripts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def experiments_dir(self) -> Path:
        """Folder for the AI assistant's experiment logs, ``experiments/<id>/`` (DESIGN.md 9;
        created on demand). Like :attr:`scripts_dir` it is left alone by ``save()``."""
        p = self._path / "experiments"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def sim_dir(self, sim_id: str) -> Path:
        """Folder of one simulation (may not exist yet)."""
        return self._path / "sims" / sim_id

    def __repr__(self) -> str:
        return (f"Project({self._name!r}, {str(self._path)!r}, {len(self.designs)} designs, "
                f"{len(self.simulations)} simulations)")

    # -- persistence --------------------------------------------------------
    def _write_if_changed(self, rel: str, obj: dict) -> bool:
        text = dumps_json(obj)
        dig = _digest(text)
        if self._written.get(rel) == dig and (self._path / rel).is_file():
            return False
        with atomic_writer(self._path / rel, "w") as fh:
            fh.write(text)
        self._written[rel] = dig
        return True

    def _write_design(self, did: str, d: Design) -> bool:
        """Write a design's JSON (and, for 3D designs, its mesh npz first) when changed.

        The npz is rewritten only when the arrays' digest changed, and before the
        JSON that names that digest, so the pair on disk is always either both old,
        both new, or detectably mixed (see ``project/meshes.py``).
        """
        rel_npz = f"designs/{did}.npz"
        digest = None
        changed = False
        if _meshes.has_mesh_data(d):
            digest = _meshes.mesh_digest(d)
            if self._written.get(rel_npz) != digest or not (self._path / rel_npz).is_file():
                payload, _ = _meshes.pack_meshes(d, digest)
                with atomic_writer(self._path / rel_npz, "wb") as fh:
                    fh.write(payload)
                self._written[rel_npz] = digest
                changed = True
        elif rel_npz in self._written or (self._path / rel_npz).is_file():
            # the design no longer has meshes (edited to 2D or cleared): drop the stale arrays
            (self._path / rel_npz).unlink(missing_ok=True)
            self._written.pop(rel_npz, None)
            changed = True
        changed |= self._write_if_changed(f"designs/{did}.json", _meshes.design_document(d, digest))
        return changed

    def schema_needed(self) -> int:
        """The schema this project is written with: 2 with any 3D design, else 1 (module docstring)."""
        with self._lock:
            return SCHEMA_VERSION if any(d.dimension == 3 for d in self.designs.values()) else SCHEMA_VERSION_2D

    def _index(self) -> dict:
        meta = dict(self._extra)
        self.schema = self.schema_needed()
        meta.update({"format": PROJECT_FORMAT, "schema": self.schema, "app_version": __version__,
                     "name": self._name, "description": self.description,
                     "created": self.created, "modified": self.modified,
                     "designs": list(self.designs), "simulations": list(self.simulations),
                     "comparisons": self.comparisons, "settings": self.settings})
        return meta

    def _write_index(self) -> None:
        self.modified = now_iso()
        write_json(self._path / _INDEX, self._index())

    def save(self) -> None:
        """Write the index and every design / simulation whose content changed.

        Objects edited in place (without ``update_*``) are picked up too, because the
        comparison is on serialised content.
        """
        with self._lock:
            if not self._path.is_dir():
                raise ProjectNotFoundError(f"project folder has disappeared: {self._path}")
            changed = False
            for did, d in self.designs.items():
                changed |= self._write_design(did, d)
            for sid, s in self.simulations.items():
                changed |= self._write_if_changed(f"sims/{sid}/simulation.json", s.to_dict())
            index = self._index()
            index.pop("modified")
            dig = _digest(dumps_json(index))
            if changed or self._written.get(_INDEX) != dig or not (self._path / _INDEX).is_file():
                self._write_index()
                self._written[_INDEX] = dig

    # -- designs ------------------------------------------------------------
    def add_design(self, design: Design) -> Design:
        """Add a design (its ``id`` must be new). Durable on the next ``save()``."""
        with self._lock:
            if design.id in self.designs:
                raise ValueError(f"a design with id {design.id!r} already exists")
            self.designs[design.id] = design
            return design

    def update_design(self, design: Design) -> None:
        """Replace the stored design with the same id."""
        with self._lock:
            if design.id not in self.designs:
                raise KeyError(f"unknown design {design.id!r}")
            self.designs[design.id] = design

    def remove_design(self, design_id: str) -> list[str]:
        """Remove a design and every simulation of it (results included); returns the
        removed simulation ids. Takes effect on disk immediately."""
        with self._lock:
            if design_id not in self.designs:
                raise KeyError(f"unknown design {design_id!r}")
            removed = [sid for sid, s in self.simulations.items() if s.design_id == design_id]
            for sid in removed:
                self._drop_simulation(sid)
            del self.designs[design_id]
            for ext in ("json", "npz"):
                (self._path / "designs" / f"{design_id}.{ext}").unlink(missing_ok=True)
                self._written.pop(f"designs/{design_id}.{ext}", None)
            self._write_index()
            return removed

    def simulations_of(self, design_id: str) -> list[Simulation]:
        """Simulations of one design, in project order."""
        with self._lock:
            return [s for s in self.simulations.values() if s.design_id == design_id]

    # -- simulations ----------------------------------------------------------
    def add_simulation(self, sim: Simulation) -> Simulation:
        """Add a simulation of an existing design. Durable on the next ``save()``."""
        with self._lock:
            if sim.id in self.simulations:
                raise ValueError(f"a simulation with id {sim.id!r} already exists")
            if sim.design_id not in self.designs:
                raise KeyError(f"simulation refers to unknown design {sim.design_id!r}")
            self.simulations[sim.id] = sim
            return sim

    def update_simulation(self, sim: Simulation) -> None:
        """Replace the stored simulation with the same id."""
        with self._lock:
            if sim.id not in self.simulations:
                raise KeyError(f"unknown simulation {sim.id!r}")
            self.simulations[sim.id] = sim

    def _drop_simulation(self, sim_id: str) -> None:
        del self.simulations[sim_id]
        self._summaries.pop(sim_id, None)
        for rel in [r for r in self._written if r.startswith(f"sims/{sim_id}/")]:
            del self._written[rel]
        remove_tree(self.sim_dir(sim_id))
        for c in self.comparisons:
            ids = c.get("sim_ids") if isinstance(c, dict) else None
            if isinstance(ids, list) and sim_id in ids:
                ids.remove(sim_id)

    def remove_simulation(self, sim_id: str) -> None:
        """Remove a simulation and delete its folder (result included), immediately."""
        with self._lock:
            if sim_id not in self.simulations:
                raise KeyError(f"unknown simulation {sim_id!r}")
            self._drop_simulation(sim_id)
            self._write_index()

    def duplicate_simulation(self, sim_id: str, name: str | None = None) -> Simulation:
        """Copy a simulation's setup (not its result) under a new id, placed right after it."""
        with self._lock:
            src = self.simulations[sim_id]
            dup = Simulation.from_dict(copy.deepcopy(src.to_dict()))
            dup.id = new_id()
            dup.name = name or f"{src.name} (copy)"
            dup.status, dup.finished, dup.error, dup.created = "new", None, None, now_iso()
            items = list(self.simulations.items())
            pos = [k for k, _ in items].index(sim_id) + 1
            items.insert(pos, (dup.id, dup))
            self.simulations.clear()
            self.simulations.update(items)
            return dup

    # -- results --------------------------------------------------------------
    def _provenance_for(self, sim_id: str, result: AnyResult, given: dict | None) -> dict:
        """The provenance record saved with a result (see ``project/provenance.py``).

        The record the caller gives and the one the runner attached to the result
        (``result.provenance``, captured with the plan at run start) win; anything they
        lack is captured here from the project's design and simulation. A CFD result
        without the runner's plan gets one re-planned here when that reproduces the run
        (``provenance.replan_record``: same grid and simulated Reynolds number), else a
        ``plan_note`` saying why there is none. Never raises: a record that cannot be
        built is saved as its error, and the result still saves.
        """
        try:
            sim = self.simulations.get(sim_id)
            design = self.designs.get(sim.design_id) if sim is not None else None
            here = _prov.capture(design, sim, result=result, source="store")
            rec = _prov.merge(_prov.merge(given, getattr(result, "provenance", None)), here)
            if not rec.get("plan"):
                plan, note = _prov.replan_record(design, sim, result, airfoil_coords=self._airfoil_resolver())
                if plan is not None:
                    rec["plan"] = plan
                    rec.setdefault("source", {})["plan"] = _prov.REPLANNED
                elif note:
                    rec["plan_note"] = note
        except Exception as exc:  # noqa: BLE001 - provenance must never cost a result
            rec = {"format": _prov.PROVENANCE_FORMAT, "schema": _prov.PROVENANCE_SCHEMA,
                   "errors": [f"provenance could not be recorded: {type(exc).__name__}: {exc}"]}
        rec["saved"] = now_iso()
        return rec

    def _airfoil_resolver(self) -> Any:
        """The wing-section airfoil resolver of this project for a re-plan (None when the
        simulation package is not loaded: then nothing is re-planned anyway)."""
        mod = sys.modules.get("aerolume.simulation.wing")
        if mod is None:
            return None
        try:
            return mod.airfoil_resolver(self, None)
        except Exception:  # noqa: BLE001 - only a 3D lofted wing needs it
            return None

    def save_result(self, sim_id: str, result: AnyResult, *, provenance: dict | None = None) -> None:
        """Write a simulation's result durably (arrays first, then the scalars).

        Accepts :class:`CFDResult`, :class:`PolarResult`, :class:`CFD3DResult` and
        :class:`WingResult`. Safe to call from a worker thread. Raises ``KeyError`` if
        the simulation was removed while it was running. Does not change
        ``sim.status``; the caller does.

        Every result is saved with its provenance record (software versions, the
        settings as run, configuration and solver hashes, device, seed; read it back
        with :meth:`provenance`). ``provenance`` (or ``result.provenance``, which the
        simulation runner attaches with the plan it used) is merged over what the store
        captures itself.
        """
        # Packing and zlib compression take ~0.5 s for a 1024x512 run (about 1 s for a
        # 2 M-cell 3D result); do them before taking the lock so the GUI thread's save()
        # never waits on them. Only the file writes (tens of ms) happen under the lock.
        doc, arrays = pack_result(result)
        if result_type(result) in ("cfd", "polar"):
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)          # the v0.1 format, unchanged
            payload: bytes | memoryview = buf.getbuffer()
        else:
            payload = npz_bytes(arrays, level=1)        # 3D: shuffled arrays, fast deflate
        # the provenance record only reads the design and the simulation (and may re-plan the
        # run, milliseconds to a fraction of a second for a 3D mesh), so it is built before the
        # lock as well
        prov_rec = self._provenance_for(sim_id, result, provenance)
        with self._lock:
            if sim_id not in self.simulations:
                raise KeyError(f"simulation {sim_id!r} no longer exists in the project")
            doc["provenance"] = prov_rec
            d = self.sim_dir(sim_id)
            d.mkdir(parents=True, exist_ok=True)
            # the simulation file must exist so the folder is recognisable on recovery
            self._write_if_changed(f"sims/{sim_id}/simulation.json", self.simulations[sim_id].to_dict())
            with atomic_writer(d / "arrays.npz", "wb") as fh:
                fh.write(payload)
            write_json(d / "result.json", doc)
            self._summaries.pop(sim_id, None)

    def provenance(self, sim_id: str) -> dict | None:
        """The provenance record saved with a simulation's result, or ``None``.

        ``None`` also for a result saved before Aerolume recorded provenance (0.2 and
        earlier). See ``aerolume.project.provenance`` for the record's content.
        """
        with self._lock:
            doc = self._read_doc(sim_id)
        if not isinstance(doc, dict):
            return None
        rec = doc.get("provenance")
        return dict(rec) if isinstance(rec, dict) else None

    def export_reproduction_script(self, sim_id: str, path: str | os.PathLike | None = None) -> Path:
        """Write a Python script that rebuilds and re-runs a simulation through ``aerolume.api``.

        The script embeds the design and the settings *as they were run* (from the
        provenance record, else the current ones, which it says), creates its own
        project next to itself, runs the simulation through the same dispatcher the
        app uses and compares the new numbers with the recorded ones (CFD means within
        three combined standard errors; deterministic polars and wings to 10⁻⁵).
        Default path: ``<project>/scripts/reproduce_<name>.py``; 3D meshes go into a
        ``<script>_mesh.npz`` beside it. Returns the script path. ``KeyError`` for an
        unknown simulation.
        """
        from .reproduce import write_reproduction_script
        return write_reproduction_script(self, sim_id, path)

    def grid_studies(self, sim_ids: list[str] | None = None) -> list[list[str]]:
        """Groups of simulations that form grid-convergence studies (``analysis.gci``).

        Same design and settings apart from the resolution settings
        (``analysis.gci.RESOLUTION_SETTINGS``), three or more resolutions, each with a
        CFD result; ids fine to coarse. ``sim_ids`` restricts the search (a report
        selection). Only ``result.json`` is read, so this is cheap.
        """
        from ..analysis.gci import find_grid_studies
        with self._lock:
            ids = [s for s in (sim_ids if sim_ids is not None else list(self.simulations)) if s in self.simulations]
            sims = {s: self.simulations[s] for s in ids}
        results = {}
        for s in ids:
            try:
                r = self.load_result(s, arrays=False)
            except (ProjectError, OSError, ValueError):
                continue
            if r is not None and type(r).__name__ in ("CFDResult", "CFD3DResult"):
                results[s] = r
        return find_grid_studies(sims, results)

    def has_result(self, sim_id: str) -> bool:
        """Whether a result has been saved for the simulation."""
        return (self.sim_dir(sim_id) / "result.json").is_file()

    def delete_result(self, sim_id: str) -> None:
        """Forget a simulation's result (e.g. before re-running with new settings)."""
        with self._lock:
            d = self.sim_dir(sim_id)
            (d / "result.json").unlink(missing_ok=True)
            (d / "arrays.npz").unlink(missing_ok=True)
            self._summaries.pop(sim_id, None)

    def _read_doc(self, sim_id: str) -> dict | None:
        f = self.sim_dir(sim_id) / "result.json"
        if not f.is_file():
            return None
        return read_json(f, f"result of simulation {sim_id}")

    def load_result(self, sim_id: str, *, arrays: bool = True) -> AnyResult | None:
        """The saved result, or ``None`` if there is none.

        With ``arrays=True`` (default) a CFD result comes back complete: history
        (float64), surface data and mean / instant fields (float32 from float16
        storage); a 3D CFD result likewise with its surfaces (float32), slices and
        volume (float32 from float16); polar and wing points with their
        distributions. ``arrays=False`` reads only ``result.json``. A damaged or
        mismatched file, or a result type this version does not know, raises
        :class:`ProjectCorruptError`.
        """
        with self._lock:
            doc = self._read_doc(sim_id)
            if doc is None:
                return None
            where = str(self.sim_dir(sim_id))
            if not arrays:
                return unpack_result(doc, None, where)
            npz = self.sim_dir(sim_id) / "arrays.npz"
            if not npz.is_file():
                raise ProjectCorruptError(f"result arrays are missing (arrays.npz): {where}")
            try:
                with np.load(npz, allow_pickle=False) as z:
                    data = {k: z[k] for k in z.files}
            except (zipfile.BadZipFile, ValueError, OSError, EOFError, KeyError) as exc:
                raise ProjectCorruptError(f"result arrays cannot be read ({exc}): {npz}") from None
            return unpack_result(doc, data, where)

    def result_summary(self, sim_id: str) -> dict | None:
        """Cheap scalar view of a result for lists and tables, or ``None``.

        ``{"type": "cfd" | "polar" | "cfd3d" | "wing", **result.to_dict()}`` straight
        from ``result.json``, cached until the file changes. Polar and wing point
        distributions are not included (they live in the arrays, as do 3D surfaces,
        slices and volumes). A type from a newer version is returned as it is.
        """
        with self._lock:
            f = self.sim_dir(sim_id) / "result.json"
            try:
                mtime = f.stat().st_mtime_ns
            except FileNotFoundError:
                self._summaries.pop(sim_id, None)
                return None
            hit = self._summaries.get(sim_id)
            if hit is not None and hit[0] == mtime:
                return dict(hit[1])
            doc = read_json(f, f"result of simulation {sim_id}")
            if not isinstance(doc, dict) or not isinstance(doc.get("result"), dict):
                raise ProjectCorruptError(f"not an Aerolume result file: {f}")
            summary = {"type": doc.get("type"), **doc["result"]}
            self._summaries[sim_id] = (mtime, summary)
            return dict(summary)
