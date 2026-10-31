"""Tests for P2 mesh validation/export. Run from the repo root:

    python scripts/test_export.py                 # synthetic meshes
    python scripts/test_export.py some_model.glb  # validate/re-export a real mesh from P3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from src.export import MeshValidationError, mesh_report, validate_and_export  # noqa: E402

OUT = Path("outputs")


def expect_error(fn, label):
    try:
        fn()
    except MeshValidationError as e:
        print(f"  ok   {label}: {e}")
        return
    raise AssertionError(f"{label}: expected MeshValidationError")


def main():
    OUT.mkdir(exist_ok=True)

    sphere = trimesh.creation.icosphere(subdivisions=3)
    sphere.visual.vertex_colors = np.tile([200, 60, 60, 255], (len(sphere.vertices), 1))
    path = validate_and_export(sphere, str(OUT / "test_sphere.glb"))
    assert Path(path).exists() and Path(path).with_suffix(".obj").exists()
    rep = mesh_report(trimesh.load(path))
    assert rep["has_color_or_texture"], "vertex colours were lost in the .glb"
    print("  ok   coloured sphere exported to .glb + .obj, colours preserved:", rep)

    box = trimesh.creation.box(extents=(2, 1, 0.5))
    validate_and_export(box, str(OUT / "test_box.glb"))
    print("  ok   plain box without colours")

    scene = trimesh.Scene([trimesh.creation.box(), trimesh.creation.icosphere().apply_translation([2, 0, 0])])
    validate_and_export(scene, str(OUT / "test_scene.glb"))
    print("  ok   Scene input is merged into one mesh")

    expect_error(lambda: validate_and_export(None, str(OUT / "x.glb")), "None mesh")
    expect_error(lambda: validate_and_export(trimesh.Trimesh(), str(OUT / "x.glb")), "empty mesh")
    flat = trimesh.creation.box(extents=(2, 2, 0.001))
    expect_error(lambda: validate_and_export(flat, str(OUT / "x.glb")), "flat plate (like a textured plane)")
    bad = trimesh.creation.box()
    bad.vertices[0, 0] = np.nan
    expect_error(lambda: validate_and_export(bad, str(OUT / "x.glb")), "NaN coordinates")

    if len(sys.argv) > 1:
        loaded = trimesh.load(sys.argv[1])
        out = validate_and_export(loaded, str(OUT / "reexported.glb"))
        print("real mesh OK ->", out, mesh_report(trimesh.load(out)))
    print("ALL EXPORT TESTS PASSED")


if __name__ == "__main__":
    main()
