"""Generate the sample 3D mesh files in this folder.

Every file is produced programmatically (no third-party models), small, and
chosen to exercise one reader feature or one robustness case:

* ``sphere_d100mm_binary.stl``  binary STL, a 100 mm sphere in millimetres (units guess: mm)
* ``box_and_cylinder_ascii.stl`` ASCII STL with two ``solid`` blocks (two meshes), metres
* ``body_and_wheel.obj``        OBJ with two groups: "body" written as quads (fan
                                triangulation) and "wheel" written with negative indices
                                and ``v/vt/vn`` triples, plus texture / normal records
* ``wedge_binary.ply``          binary little-endian PLY: quads and triangles mixed, an
                                extra per-vertex colour and an extra element to skip
* ``leaky_sphere.stl``          a 1 m sphere with a patch of triangles missing (a hole of
                                about 20 degrees) -- voxelises like the closed sphere

Run: ``python examples/geometry3d/make_examples.py``
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def make_stl() -> None:
    from aerolume.geometry3d import box, cylinder, sphere, write_stl

    s = sphere(100.0, 24, name="sphere_mm")
    write_stl(s, HERE / "sphere_d100mm_binary.stl", binary=True)
    b = box(0.4, 0.3, 0.2, center=(0.0, 0.0, 0.1), name="box")
    c = cylinder(0.1, 0.3, axis="y", n=12, center=(0.5, 0.0, 0.05), name="cylinder")
    write_stl([b, c], HERE / "box_and_cylinder_ascii.stl", binary=False)


def make_leaky() -> None:
    from aerolume.geometry3d import sphere, write_stl
    from aerolume.model import Mesh3D

    s = sphere(1.0, 32, name="leaky_sphere")
    c = s.vertices[s.faces].mean(1)
    d = np.array([0.35, 0.45, 0.82])
    d /= np.linalg.norm(d)
    keep = (c / np.linalg.norm(c, axis=1, keepdims=True)) @ d < np.cos(np.radians(20.0))
    write_stl(Mesh3D("leaky_sphere", s.vertices, s.faces[keep]), HERE / "leaky_sphere.stl", binary=True)


def make_obj() -> None:
    from aerolume.geometry3d import cylinder

    # body: a 2 m x 0.8 m x 0.6 m block with a sloped nose (front top edge set back),
    # nose towards -x, standing 0.15 m above the road; faces written as outward quads
    bv = np.array([[0.0, -0.4, 0.15], [2.0, -0.4, 0.15], [2.0, 0.4, 0.15], [0.0, 0.4, 0.15],
                   [0.5, -0.4, 0.75], [2.0, -0.4, 0.75], [2.0, 0.4, 0.75], [0.5, 0.4, 0.75]])
    quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]
    w = cylinder(0.3, 0.12, axis="y", n=16, center=(0.45, -0.47, 0.15), name="wheel")
    lines = ["# Aerolume sample: two groups, quads, negative indices, vt/vn records",
             "mtllib none.mtl", "o sample_vehicle", "g body", "usemtl paint"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in bv]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vn 0 0 1"]
    lines += ["f " + " ".join(str(i + 1) for i in q) for q in quads]
    lines += ["g wheel", "usemtl rubber"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in w.vertices]
    nw = len(w.vertices)
    for f in w.faces:                                   # negative indices, v/vt/vn triples
        lines.append("f " + " ".join(f"{int(i) - nw}/1/1" for i in f))
    (HERE / "body_and_wheel.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_ply() -> None:
    # a wedge (ramp) 1 m long, 0.5 m wide, 0.3 m high: two quads, three more faces
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 0.0],
                  [1.0, 0.0, 0.3], [1.0, 0.5, 0.3]], dtype="<f4")
    faces = [[0, 3, 2, 1], [0, 1, 4], [3, 5, 2], [1, 2, 5, 4], [0, 4, 5, 3]]
    header = ("ply\nformat binary_little_endian 1.0\ncomment Aerolume sample wedge\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
              "element material 1\nproperty float shininess\nend_header\n")
    body = b"".join(struct.pack("<3f3B", *row, 200, 120, 40) for row in v)
    for f in faces:
        body += struct.pack("<B", len(f)) + struct.pack(f"<{len(f)}i", *f)
    body += struct.pack("<f", 0.5)
    (HERE / "wedge_binary.ply").write_bytes(header.encode("ascii") + body)


if __name__ == "__main__":
    make_stl()
    make_leaky()
    make_obj()
    make_ply()
    for p in sorted(HERE.iterdir()):
        if p.suffix in (".stl", ".obj", ".ply"):
            print(f"{p.name:30s} {p.stat().st_size:8d} bytes")
