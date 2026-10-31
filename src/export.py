"""P2: Mesh validation and export.

    validate_and_export(mesh, out_path) -> path of the exported .glb

Shared contract:
    def validate_and_export(mesh, out_path: str) -> str      # raises if empty/flat

Checks: non-zero vertices/faces, finite coordinates, not flat on any axis.
Exports .glb (main output) and a matching .obj, then RELOADS the files to prove they open.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)

MIN_FLATNESS = 0.02   # smallest bounding-box extent / largest extent; below this the mesh is "flat"


class MeshValidationError(ValueError):
    """The reconstructed mesh is unusable (empty, broken or flat)."""


def _as_mesh(obj) -> trimesh.Trimesh:
    """Accept a Trimesh or a Scene (what trimesh.load returns for .glb) and return one Trimesh."""
    if obj is None:
        raise MeshValidationError("No mesh was produced.")
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        if len(obj.geometry) == 0:
            raise MeshValidationError("The 3D scene is empty.")
        try:
            merged = obj.dump(concatenate=True)
        except Exception:
            merged = obj.to_mesh()
        if isinstance(merged, trimesh.Trimesh):
            return merged
    raise MeshValidationError(f"Unsupported mesh type: {type(obj).__name__}")


def _has_color(mesh: trimesh.Trimesh) -> bool:
    try:
        return mesh.visual.kind in ("vertex", "face", "texture")
    except Exception:
        return False


def validate_mesh(mesh) -> trimesh.Trimesh:
    """Clean and validate. Returns the (cleaned) mesh or raises MeshValidationError."""
    mesh = _as_mesh(mesh)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise MeshValidationError("The mesh is empty (no vertices or faces).")
    if not np.isfinite(mesh.vertices).all():
        raise MeshValidationError("The mesh contains invalid (NaN/inf) coordinates.")

    # drop degenerate faces and unused vertices (keeps colours/textures aligned)
    try:
        ok = mesh.nondegenerate_faces()
        if not ok.all():
            mesh.update_faces(ok)
        mesh.remove_unreferenced_vertices()
    except Exception as e:  # cleanup is best effort
        log.warning("Mesh cleanup skipped (%s)", e)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise MeshValidationError("The mesh has no valid faces after cleanup.")

    ext = np.asarray(mesh.extents, dtype=float)
    if ext.max() <= 0:
        raise MeshValidationError("The mesh has zero size.")
    if ext.min() / ext.max() < MIN_FLATNESS:
        raise MeshValidationError(
            "The mesh is flat (almost no depth on one axis). The 3D reconstruction probably failed.")
    return mesh


def mesh_report(mesh) -> dict:
    """Numbers for the README / benchmarks / UI."""
    mesh = _as_mesh(mesh)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents": [round(float(x), 4) for x in mesh.extents],
        "watertight": bool(mesh.is_watertight),
        "has_color_or_texture": _has_color(mesh),
    }


def _reload_check(path: Path, expect_color: bool) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise MeshValidationError(f"Export failed: {path.name} is missing or empty.")
    try:
        loaded = _as_mesh(trimesh.load(str(path)))
    except MeshValidationError:
        raise
    except Exception as e:
        raise MeshValidationError(f"Exported file {path.name} cannot be opened: {e}") from e
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise MeshValidationError(f"Exported file {path.name} opens but contains no geometry.")
    if expect_color and not _has_color(loaded):
        log.warning("%s was reloaded without colours/texture (format limitation?)", path.name)
    return mesh_report(loaded)


def validate_and_export(mesh, out_path: str, also_obj: bool = True, normalize: bool = True) -> str:
    """Validate `mesh`, export it as .glb (plus .obj), reload both, return the .glb path.

    normalize: centre the mesh at the origin and scale its longest side to 1 (nicer in viewers).
    The caller's mesh object is not modified.
    """
    mesh = validate_mesh(_as_mesh(mesh).copy())
    had_color = _has_color(mesh)

    if normalize:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(1.0 / float(mesh.extents.max()))

    out = Path(out_path)
    if out.suffix.lower() not in (".glb", ".obj"):
        out = out.with_suffix(".glb")
    glb_path = out.with_suffix(".glb")
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(str(glb_path))
    report = _reload_check(glb_path, had_color)

    if also_obj:
        obj_path = glb_path.with_suffix(".obj")
        try:
            mesh.export(str(obj_path), include_color=True)
        except TypeError:
            mesh.export(str(obj_path))
        _reload_check(obj_path, False)  # OBJ cannot always keep colours; only check geometry

    log.info("Exported %s | %s", glb_path, report)
    return str(glb_path)
