"""Cloud projects, sync status and conflicts, share links, profile and avatar (mock Supabase).

"Two computers" are two AccountService instances with separate session and
sync stores, signed in as the same user, working on two folders.
"""
from __future__ import annotations

import io
import json
import os
import random
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta

import pytest
from test_accounts_fakes import signed_in

from aerolume.accounts import AccountError, RemoteError, SyncConflictError, TooLargeError, cloud
from aerolume.accounts.mock import MockSupabase
from aerolume.project.store import Project

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15"
       b"\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00"
       b"\x00IEND\xaeB`\x82")


@pytest.fixture()
def mock():
    with MockSupabase() as m:
        yield m


def _project(folder, name="Wing study"):
    p = Project.create(folder / f"{name}.aerolume", name)
    p.save()
    (p.reports_dir / "report.html").write_text("<!doctype html><meta charset='utf-8'><h1>Report</h1>",
                                               encoding="utf-8")
    return p.path


def _uploads(mock):
    return [r for r in mock.requests if r[0] == "POST" and r[1].startswith("/storage/v1/object/projects/")]


# ---------------------------------------------------------------------------
# hashing and packing
# ---------------------------------------------------------------------------

def test_content_hash_ignores_location_time_and_litter(tmp_path):
    a = _project(tmp_path / "a")
    b = tmp_path / "b" / "Copy.aerolume"
    b.parent.mkdir()
    import shutil
    shutil.copytree(a, b)
    os.utime(b / "project.json", (1, 1))
    for junk in (".trash-sims-1-2/x.json", "__pycache__/m.pyc", ".project.json.123.tmp", "Thumbs.db",
                 "reports/~$draft.docx"):
        (b / junk).parent.mkdir(parents=True, exist_ok=True)
        (b / junk).write_bytes(b"junk")
    assert cloud.content_hash(a) == cloud.content_hash(b)
    (b / "reports" / "report.html").write_text("<h1>Changed</h1>", encoding="utf-8")
    assert cloud.content_hash(a) != cloud.content_hash(b)
    with pytest.raises(AccountError, match="not an Aerolume project"):
        cloud.content_hash(tmp_path)


def test_save_timestamps_do_not_count_as_changes(tmp_path):
    """Identical edits saved at different times (or by another version) hash the same."""
    folder = _project(tmp_path / "a")
    h0 = cloud.content_hash(folder)
    doc = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    doc["modified"], doc["app_version"] = "2031-01-01T00:00:00+00:00", "9.9.9"
    (folder / "project.json").write_text(json.dumps(doc, indent=4), encoding="utf-8")      # other layout too
    assert cloud.content_hash(folder) == h0 == cloud.pack_project(folder).content_hash
    doc["name"] = "Renamed"
    (folder / "project.json").write_text(json.dumps(doc), encoding="utf-8")
    assert cloud.content_hash(folder) != h0


def test_pack_unpack_round_trip_is_deterministic(tmp_path):
    src = _project(tmp_path / "src")
    p1, p2 = cloud.pack_project(src), cloud.pack_project(src)
    assert p1.data == p2.data and p1.content_hash == cloud.content_hash(src) and p1.file_count >= 2
    out = cloud.unpack_project(p1.data, tmp_path / "out" / "W.aerolume", expected_hash=p1.content_hash)
    assert cloud.content_hash(out) == p1.content_hash
    assert Project.open(out).name == "Wing study"
    with pytest.raises(AccountError, match="already exists"):
        cloud.unpack_project(p1.data, out)
    with pytest.raises(AccountError, match="checksum"):
        cloud.unpack_project(p1.data, tmp_path / "out" / "X.aerolume", expected_hash="0" * 64)
    assert not (tmp_path / "out" / "X.aerolume").exists()
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["W.aerolume"]      # no temp left


@pytest.mark.parametrize("member", ["../evil.txt", "/abs.txt", "C:/win.txt", "a/../../b.txt", "con.json"])
def test_unpack_rejects_unsafe_members(tmp_path, member):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("project.json", "{}")
        zf.writestr(member, "x")
    with pytest.raises(AccountError) as ei:
        cloud.unpack_project(buf.getvalue(), tmp_path / "dest" / "P.aerolume")
    assert ei.value.code == "unsafe_archive"
    assert not (tmp_path / "evil.txt").exists() and not any((tmp_path / "dest").iterdir())


@pytest.mark.parametrize("local,base,remote,status", [
    ("a", "a", "a", cloud.SYNCED), ("b", "a", "a", cloud.LOCAL_CHANGES), ("a", "a", "b", cloud.REMOTE_CHANGES),
    ("b", "a", "c", cloud.CONFLICT), ("b", "a", "b", cloud.SYNCED), ("a", None, None, cloud.LOCAL_ONLY),
    ("a", None, "a", cloud.SYNCED), ("a", None, "b", cloud.CONFLICT), ("a", "a", None, cloud.REMOTE_MISSING),
    (None, None, "a", cloud.REMOTE_ONLY),
])
def test_classify_truth_table(local, base, remote, status):
    assert cloud.classify(local, base, remote) == status


# ---------------------------------------------------------------------------
# upload / download / sync
# ---------------------------------------------------------------------------

def test_upload_status_and_versioned_objects(mock, tmp_path):
    svc = signed_in(mock, tmp_path / "pc1")
    folder = _project(tmp_path / "work")
    assert svc.project_status(folder).status == cloud.LOCAL_ONLY
    st = svc.upload_project(folder)
    uid = svc.user.id
    assert st.status == cloud.SYNCED and st.cloud.object_path == f"{uid}/{st.cloud.id}/{st.local_hash}.zip"
    assert list(mock.objects) == [("projects", st.cloud.object_path)]
    assert svc.project_status(folder).status == cloud.SYNCED
    n = len(_uploads(mock))
    assert svc.upload_project(folder).status == cloud.SYNCED and len(_uploads(mock)) == n    # nothing re-sent
    (folder / "reports" / "report.html").write_text("<h1>v2</h1>", encoding="utf-8")
    assert svc.project_status(folder).status == cloud.LOCAL_CHANGES
    st2 = svc.upload_project(folder)
    assert st2.cloud.id == st.cloud.id and st2.local_hash != st.local_hash
    assert list(mock.objects) == [("projects", st2.cloud.object_path)]        # the old version was removed
    [row] = svc.list_cloud_projects()
    assert (row.name, row.content_hash, row.file_count) == ("Wing study", st2.local_hash, st2.cloud.file_count)


def test_second_computer_downloads_identical_project(mock, tmp_path):
    pc1 = signed_in(mock, tmp_path / "pc1")
    folder = _project(tmp_path / "work1")
    st = pc1.upload_project(folder)
    pc2 = signed_in(mock, tmp_path / "pc2")
    [entry] = pc2.sync_overview()
    assert entry.status == cloud.REMOTE_ONLY and entry.label == "Cloud only"
    dest = pc2.download_project(st.cloud.id, tmp_path / "work2")
    assert dest.name == "Wing study.aerolume" and cloud.content_hash(dest) == st.local_hash
    assert pc2.project_status(dest).status == cloud.SYNCED
    assert Project.open(dest).name == "Wing study"
    again = pc2.download_project(st.cloud.id, tmp_path / "work2")             # never overwrites
    assert again.name == "Wing study (2).aerolume"


def test_conflict_is_detected_and_nothing_is_overwritten(mock, tmp_path):
    pc1, pc2 = signed_in(mock, tmp_path / "pc1"), signed_in(mock, tmp_path / "pc2")
    f1 = _project(tmp_path / "w1")
    pid = pc1.upload_project(f1).cloud.id
    f2 = pc2.download_project(pid, tmp_path / "w2")
    (f2 / "reports" / "report.html").write_text("<h1>from pc2</h1>", encoding="utf-8")
    pc2_hash = pc2.upload_project(f2).local_hash
    (f1 / "reports" / "report.html").write_text("<h1>from pc1</h1>", encoding="utf-8")
    st = pc1.project_status(f1)
    assert st.status == cloud.CONFLICT and st.label == "Conflict"
    with pytest.raises(SyncConflictError) as ei:
        pc1.upload_project(f1)
    assert ei.value.remote_hash == pc2_hash and ei.value.project_id == pid
    assert pc1.list_cloud_projects()[0].content_hash == pc2_hash                # pc2's work is intact
    # "keep mine" is an explicit choice
    st = pc1.upload_project(f1, force=True)
    assert st.status == cloud.SYNCED and pc2.project_status(f2).status == cloud.REMOTE_CHANGES


def test_newer_in_cloud_then_replace_keeps_unsynced_work(mock, tmp_path):
    pc1, pc2 = signed_in(mock, tmp_path / "pc1"), signed_in(mock, tmp_path / "pc2")
    f1 = _project(tmp_path / "w1")
    pid = pc1.upload_project(f1).cloud.id
    f2 = pc2.download_project(pid, tmp_path / "w2")
    (f2 / "reports" / "report.html").write_text("<h1>v2</h1>", encoding="utf-8")
    v2 = pc2.upload_project(f2).local_hash
    assert pc1.project_status(f1).status == cloud.REMOTE_CHANGES
    with pytest.raises(SyncConflictError):
        pc1.upload_project(f1)                                   # would overwrite newer cloud work
    # pc1 had no local edits: the folder is simply replaced
    out = pc1.download_project(pid, None, replace=f1)
    assert out == f1 and cloud.content_hash(f1) == v2
    assert sorted(p.name for p in f1.parent.iterdir()) == ["Wing study.aerolume"]
    # with unsynced local edits the old folder is kept beside the new one
    (f2 / "reports" / "report.html").write_text("<h1>v3</h1>", encoding="utf-8")
    pc2.upload_project(f2)
    (f1 / "notes.txt").write_text("unsynced thought", encoding="utf-8")
    pc1.download_project(pid, None, replace=f1)
    names = sorted(p.name for p in f1.parent.iterdir())
    assert len(names) == 2 and any("local copy" in n for n in names)
    backup = next(p for p in f1.parent.iterdir() if "local copy" in p.name)
    assert (backup / "notes.txt").read_text(encoding="utf-8") == "unsynced thought"


def test_upload_race_is_a_conflict_not_an_overwrite(mock, tmp_path, monkeypatch):
    pc1, pc2 = signed_in(mock, tmp_path / "pc1"), signed_in(mock, tmp_path / "pc2")
    f1 = _project(tmp_path / "w1")
    pid = pc1.upload_project(f1).cloud.id
    f2 = pc2.download_project(pid, tmp_path / "w2")
    (f1 / "a.txt").write_text("pc1", encoding="utf-8")
    (f2 / "b.txt").write_text("pc2", encoding="utf-8")
    stale = pc1._cloud_row(pid)
    pc2.upload_project(f2)                 # lands between pc1's read and pc1's write
    real = pc1._cloud_row
    monkeypatch.setattr(pc1, "_cloud_row", lambda p: stale if p == pid and stale else real(p))
    with pytest.raises(SyncConflictError, match="moment ago"):
        pc1.upload_project(f1)
    monkeypatch.setattr(pc1, "_cloud_row", real)
    [row] = pc2.list_cloud_projects()
    assert row.content_hash == cloud.content_hash(f2)
    assert [k for k in mock.objects if k[0] == "projects"] == [("projects", row.object_path)]   # orphan removed


def _project_objects(mock):
    return sorted(p for (b, p) in mock.objects if b == "projects")


def _add_run(folder, sim_id, name, kind, nbytes, seed=1):
    """A fake run folder: the files the size breakdown reads, plus incompressible "fields"."""
    d = folder / "sims" / sim_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "simulation.json").write_text(json.dumps({"name": name, "id": sim_id}), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({"type": kind}), encoding="utf-8")
    (d / "arrays.npz").write_bytes(random.Random(seed).randbytes(nbytes))      # .npz is stored, not deflated


def test_project_over_one_part_uploads_in_parts_and_round_trips(mock, tmp_path):
    """max_upload_mb=1 makes parts 0.9 MiB, so a 2.5 MB project needs three of them."""
    svc = signed_in(mock, tmp_path / "pc1", max_upload_mb=1)
    folder = _project(tmp_path / "w")
    _add_run(folder, "aaaa", "Hot sphere", "cfd3d", 2_500_000)
    st = svc.upload_project(folder)
    part = svc.config.part_bytes
    assert part == int(0.9 * 1024 * 1024)
    n = st.cloud.part_count
    assert n == -(-st.cloud.size_bytes // part) == 3
    base = f"{svc.user.id}/{st.cloud.id}/{st.local_hash}.zip"
    assert st.cloud.object_path == base
    assert _project_objects(mock) == [f"{base}.part{i:03d}" for i in (1, 2, 3)] == st.cloud.object_paths
    sizes = [len(mock.objects[("projects", p)][0]) for p in st.cloud.object_paths]
    assert max(sizes) <= part and sum(sizes) == st.cloud.size_bytes
    assert {mock.objects[("projects", p)][1] for p in st.cloud.object_paths} == {"application/octet-stream"}
    # a second computer reassembles the parts and verifies the checksum
    pc2 = signed_in(mock, tmp_path / "pc2")
    dest = pc2.download_project(st.cloud.id, tmp_path / "w2")
    assert cloud.content_hash(dest) == st.local_hash and pc2.project_status(dest).status == cloud.SYNCED
    assert (dest / "sims" / "aaaa" / "arrays.npz").read_bytes() == (folder / "sims/aaaa/arrays.npz").read_bytes()
    # a smaller new version is one plain zip; every old part is removed
    from aerolume.project.fileio import remove_tree
    remove_tree(folder / "sims" / "aaaa")
    st2 = svc.upload_project(folder)
    assert st2.cloud.part_count == 1 and _project_objects(mock) == [st2.cloud.object_path]
    assert mock.objects[("projects", st2.cloud.object_path)][1] == "application/zip"
    # deleting removes every part of a multi-part version too
    _add_run(folder, "bbbb", "Car smoke", "cfd3d", 2_000_000, seed=2)
    st3 = svc.upload_project(folder)
    assert st3.cloud.part_count == 3 and len(_project_objects(mock)) == 3
    svc.delete_cloud_project(st3.cloud.id)
    assert _project_objects(mock) == []


def test_sixty_mb_project_on_the_default_50_mb_limit(mock, tmp_path):
    """The critic's case at full size: a normal project over 50 MB, default settings."""
    import time
    svc = signed_in(mock, tmp_path / "pc1")                  # max_upload_mb=50: parts of 45 MiB
    assert mock.file_size_limit == 50 * 1024 * 1024
    folder = _project(tmp_path / "w", "Room study")
    for i, (kind, mb) in enumerate((("cfd3d", 16), ("cfd3d", 15), ("cfd3d", 14), ("cfd", 9), ("cfd", 6))):
        _add_run(folder, f"run{i}", f"Run {i}", kind, mb * 1_000_000, seed=i)
    preview = svc.upload_preview(folder)
    assert preview.size_bytes > 60_000_000 and preview.parts == 2 and preview.runs == 5
    assert preview.groups[0].label == "Run 0 (3D run)" and preview.groups[0].size_bytes > 16_000_000
    said: list[str] = []
    t0 = time.perf_counter()
    st = svc.upload_project(folder, progress=said.append)
    upload_s = time.perf_counter() - t0
    assert st.status == cloud.SYNCED and st.cloud.part_count == 2
    assert st.cloud.size_bytes > 60_000_000 and st.cloud.size_bytes == preview.size_bytes   # shown = sent
    assert all(len(mock.objects[("projects", p)][0]) <= 45 * 1024 * 1024 for p in st.cloud.object_paths)
    assert said[0] == "Packing the project" and any(m.startswith("Uploading part 2 of 2") for m in said)
    t0 = time.perf_counter()
    dest = signed_in(mock, tmp_path / "pc2").download_project(st.cloud.id, tmp_path / "w2")
    download_s = time.perf_counter() - t0
    assert cloud.content_hash(dest) == st.local_hash
    print(f"60 MB project: packed {st.cloud.size_bytes / 1e6:.1f} MB in {st.cloud.part_count} parts, "
          f"upload {upload_s:.2f} s, download {download_s:.2f} s (mock server on 127.0.0.1)")


def test_project_limit_names_what_takes_the_space(mock, tmp_path):
    svc = signed_in(mock, tmp_path / "pc1", max_project_mb=1)
    folder = _project(tmp_path / "w")
    _add_run(folder, "aaaa", "Car smoke", "cfd3d", 1_500_000)
    (folder / "exports").mkdir(exist_ok=True)
    (folder / "exports" / "field.npz").write_bytes(random.Random(3).randbytes(700_000))
    with pytest.raises(TooLargeError) as ei:
        svc.upload_project(folder)
    msg = str(ei.value)
    assert "over this server's project limit of 1 MB" in msg
    assert "Car smoke (3D run) 1.4 MB" in msg and "Exported files" in msg and "written again" in msg
    assert "delete" not in msg.lower()                    # never the first remedy
    assert not _uploads(mock)


def test_server_with_a_lower_file_limit_fails_cleanly(mock, tmp_path):
    """The bucket refuses a part (its limit is below max_upload_mb): parts already sent are removed."""
    svc = signed_in(mock, tmp_path / "pc1", max_upload_mb=1)
    folder = _project(tmp_path / "w")
    _add_run(folder, "aaaa", "Run", "cfd", 2_000_000)
    real = mock.file_size_limit
    mock.file_size_limit = 100
    with pytest.raises(TooLargeError, match="max_upload_mb") as ei:
        svc.upload_project(folder)
    assert "delete" not in str(ei.value).lower()
    assert _project_objects(mock) == [] and svc.list_cloud_projects() == []
    mock.file_size_limit = real
    assert svc.upload_project(folder).status == cloud.SYNCED      # nothing stale got in the way


def test_interrupted_upload_leaves_the_cloud_copy_intact(mock, tmp_path, monkeypatch):
    svc = signed_in(mock, tmp_path / "pc1", max_upload_mb=1)
    folder = _project(tmp_path / "w")
    _add_run(folder, "aaaa", "Run", "cfd", 2_000_000)
    first = svc.upload_project(folder)
    before = _project_objects(mock)
    _add_run(folder, "bbbb", "Second", "cfd", 1_000_000, seed=5)
    backend = svc._backend()
    real_upload, calls = backend.upload, []

    def flaky(bucket, path, data, token, **kw):
        calls.append(path)
        if len(calls) == 2:
            raise RemoteError("Uploading the project failed (HTTP 503).", code="http_503", status=503)
        return real_upload(bucket, path, data, token, **kw)
    monkeypatch.setattr(backend, "upload", flaky)
    monkeypatch.setattr(svc, "_backend", lambda: backend)
    with pytest.raises(RemoteError, match="503"):
        svc.upload_project(folder)
    assert _project_objects(mock) == before                   # the new part 1 was removed again
    [row] = svc.list_cloud_projects()
    assert row.content_hash == first.local_hash and row.part_count == first.cloud.part_count
    assert svc.project_status(folder).status == cloud.LOCAL_CHANGES


def test_race_with_parts_removes_only_our_parts(mock, tmp_path, monkeypatch):
    pc1, pc2 = signed_in(mock, tmp_path / "pc1", max_upload_mb=1), signed_in(mock, tmp_path / "pc2", max_upload_mb=1)
    f1 = _project(tmp_path / "w1")
    _add_run(f1, "aaaa", "Run", "cfd", 2_000_000)
    pid = pc1.upload_project(f1).cloud.id
    f2 = pc2.download_project(pid, tmp_path / "w2")
    _add_run(f1, "bbbb", "pc1 run", "cfd", 1_000_000, seed=7)
    _add_run(f2, "cccc", "pc2 run", "cfd", 1_000_000, seed=8)
    stale = pc1._cloud_row(pid)
    pc2.upload_project(f2)
    real = pc1._cloud_row
    monkeypatch.setattr(pc1, "_cloud_row", lambda p: stale if p == pid and stale else real(p))
    with pytest.raises(SyncConflictError, match="moment ago"):
        pc1.upload_project(f1)
    [row] = pc2.list_cloud_projects()
    assert _project_objects(mock) == sorted(row.object_paths)


def test_database_without_schema_2_gets_a_clear_message(mock, tmp_path):
    mock.schema_version = 1
    svc = signed_in(mock, tmp_path / "pc1", max_upload_mb=1)
    folder = _project(tmp_path / "w")
    _add_run(folder, "aaaa", "Run", "cfd", 2_000_000)
    with pytest.raises(RemoteError) as ei:                 # the part MIME type is refused
        svc.upload_project(folder)
    assert ei.value.code == "schema_outdated" and "20261030120000_aerolume_project_parts.sql" in str(ei.value)
    small = _project(tmp_path / "w2", "Small")              # one-part zip: the part_count column is refused
    with pytest.raises(RemoteError) as ei:
        svc.upload_project(small)
    assert ei.value.code == "schema_outdated"
    assert _project_objects(mock) == []
    assert svc.test_connection().schema_version == 1        # Test connection says the same


def test_size_breakdown_groups_runs_and_folders(tmp_path):
    folder = _project(tmp_path / "w")
    _add_run(folder, "aaaa", "Hot sphere", "cfd3d", 3000)
    _add_run(folder, "bbbb", "Cylinder", "cfd", 2000)
    (folder / "sims" / "cccc").mkdir()
    (folder / "sims" / "cccc" / "arrays.npz").write_bytes(b"x" * 1000)     # unreadable run: folder name
    groups = cloud.size_breakdown(folder)
    labels = [g.label for g in groups]
    assert labels[:2] == ["Hot sphere (3D run)", "Cylinder (2D run)"] and "cccc (run)" in labels
    assert "Reports" in labels and "Project settings and other files" in labels
    assert sum(g.file_count for g in groups) == len(cloud.project_files(folder))
    assert sum(g.size_bytes for g in groups) == sum(p.stat().st_size for _r, p in cloud.project_files(folder))
    buf = io.BytesIO()
    info = cloud.pack_project_to(folder, buf)
    packed = cloud.size_breakdown(folder, info.entry_bytes)
    assert {g.label for g in packed} == set(labels)
    assert info.packed_bytes == len(buf.getvalue()) and info.content_hash == cloud.content_hash(folder)
    # the size shown before an upload is the archive's exact size, deflated files included
    (folder / "exports").mkdir(exist_ok=True)
    (folder / "exports" / "smoke.vti").write_bytes(b"<VTKFile> 0 0 0 1 1 1 </VTKFile>\n" * 30000)
    (folder / "exports" / "empty.csv").write_bytes(b"")
    (folder / "notes ü.txt").write_text("non-ASCII name", encoding="utf-8")
    entries, total = cloud.packed_sizes(folder)
    buf = io.BytesIO()
    info = cloud.pack_project_to(folder, buf)
    assert total == info.packed_bytes and entries == info.entry_bytes
    assert entries["exports/smoke.vti"] < (folder / "exports/smoke.vti").stat().st_size / 50
    assert cloud.packed_sizes(folder) == (entries, total)           # cached second time, same answer
    assert cloud.part_paths("u/p/h.zip", 1) == ["u/p/h.zip"]
    assert cloud.part_paths("u/p/h.zip", 2) == ["u/p/h.zip.part001", "u/p/h.zip.part002"]
    assert [cloud.part_count_for(n, 10) for n in (0, 1, 10, 11, 25)] == [1, 1, 1, 2, 3]


def test_users_cannot_see_or_touch_each_other(mock, tmp_path):
    ada = signed_in(mock, tmp_path / "ada", "ada@example.org")
    bob = signed_in(mock, tmp_path / "bob", "bob@example.org")
    st = ada.upload_project(_project(tmp_path / "w"))
    assert bob.list_cloud_projects() == []
    with pytest.raises(RemoteError) as ei:
        bob._backend().download("projects", st.cloud.object_path, bob.access_token())
    assert ei.value.code == "not_found"
    with pytest.raises(RemoteError) as ei:
        bob._backend().upload("projects", f"{ada.user.id}/x/y.zip", b"PK", bob.access_token(),
                              content_type="application/zip")
    assert ei.value.code == "forbidden"
    with pytest.raises(RemoteError):
        bob._backend().insert("projects", bob.access_token(), {"id": st.cloud.id, "owner": ada.user.id,
                                                               "name": "x", "content_hash": "0" * 64,
                                                               "object_path": f"{ada.user.id}/{st.cloud.id}/a"})
    assert bob._backend().delete("projects", bob.access_token(), {"id": st.cloud.id}) == []
    assert len(ada.list_cloud_projects()) == 1


def test_delete_cloud_project_keeps_the_local_folder(mock, tmp_path):
    svc = signed_in(mock, tmp_path / "pc1")
    folder = _project(tmp_path / "w")
    pid = svc.upload_project(folder).cloud.id
    svc.delete_cloud_project(pid)
    assert svc.list_cloud_projects() == [] and not [k for k in mock.objects if k[0] == "projects"]
    assert (folder / "project.json").is_file() and svc.project_status(folder).status == cloud.LOCAL_ONLY


# ---------------------------------------------------------------------------
# shares
# ---------------------------------------------------------------------------

def test_share_links_work_without_sign_in_and_die_when_revoked(mock, tmp_path):
    svc = signed_in(mock, tmp_path / "pc1")
    folder = _project(tmp_path / "w")
    pdf = folder / "reports" / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    share = svc.create_share([folder / "reports" / "report.html", pdf], title="NACA 2412 at 4 deg",
                             expires_days=7)
    assert len(share.token) >= 43 and share.files == ("index.html", "01-report.pdf")
    html = urllib.request.urlopen(share.url).read()                      # no apikey, no sign-in
    assert html == (folder / "reports" / "report.html").read_bytes()
    assert urllib.request.urlopen(share.links["01-report.pdf"]).read() == b"%PDF-1.4 test"
    anon = signed_in(mock, tmp_path / "anon", "eve@example.org")
    anon.sign_out()
    facts = anon.resolve_share(share.token)                             # by token, signed out
    assert facts["title"] == "NACA 2412 at 4 deg" and "owner" not in facts and "links" not in facts
    exp = datetime.fromisoformat(share.expires_at)
    assert timedelta(days=6.9) < exp - datetime.now(UTC) < timedelta(days=7.1)
    assert [s.token for s in svc.list_shares()] == [share.token]
    svc.revoke_share(share.token)
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(share.url)
    assert anon.resolve_share(share.token) is None and svc.list_shares() == []


def test_share_rules(mock, tmp_path):
    svc = signed_in(mock, tmp_path / "pc1")
    folder = _project(tmp_path / "w")
    with pytest.raises(AccountError, match="HTML"):
        svc.create_share([folder / "project.json"], title="x")
    with pytest.raises(AccountError, match="expiry"):
        svc.create_share([folder / "reports" / "report.html"], title="x", expires_days=0)
    share = svc.create_share([folder / "reports" / "report.html"], title="")
    assert share.title == "Aerolume result"
    # expired shares are not resolvable even though the row still exists
    mock.rows["shares"][share.token]["expires_at"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    assert svc.resolve_share(share.token) is None
    # nobody else can write into a share's folder
    bob = signed_in(mock, tmp_path / "bob", "bob@example.org")
    with pytest.raises(RemoteError) as ei:
        bob._backend().upload("shares", f"{share.token}/evil.html", b"<x>", bob.access_token(),
                              content_type="text/html")
    assert ei.value.code == "forbidden"


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

def test_profile_and_avatar(mock, tmp_path):
    svc = signed_in(mock, tmp_path / "pc1")
    events = []
    svc.subscribe(events.append)
    prof = svc.update_profile(display_name="Ada Lovelace", affiliation="Analytical Engines Ltd",
                              preferences={"unit_system": "SI", "theme": "dark"})
    assert (prof.display_name, prof.affiliation, prof.preferences["theme"]) == (
        "Ada Lovelace", "Analytical Engines Ltd", "dark")
    assert svc.user.display_name == "Ada Lovelace" and svc.user.initials == "AL"
    assert "profile" in events
    with pytest.raises(AccountError, match="80 characters"):
        svc.update_profile(display_name="x" * 81)
    with pytest.raises(AccountError, match="PNG or JPEG"):
        svc.set_avatar(b"GIF89a....")
    with pytest.raises(TooLargeError):
        svc.set_avatar(PNG + b"\0" * (1024 * 1024))
    prof = svc.set_avatar(PNG)
    assert prof.avatar_path == f"{svc.user.id}/avatar.png" and svc.avatar_bytes() == PNG
    prof = svc.remove_avatar()
    assert prof.avatar_path == "" and svc.avatar_bytes() is None
    assert not [k for k in mock.objects if k[0] == "avatars"]
    # a fresh service reads the same profile back
    other = signed_in(mock, tmp_path / "pc2")
    assert other.get_profile().affiliation == "Analytical Engines Ltd"
    assert json.dumps(other.get_profile().preferences) == '{"unit_system": "SI", "theme": "dark"}'
