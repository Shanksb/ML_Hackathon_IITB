"""Pipeline: sketch -> prepare_sketch -> generate_image -> prepare_for_3d
-> reconstruct -> validate_and_export. Models stay cached between calls."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import config
from src.export import MeshValidationError, mesh_report, validate_and_export
from src.preprocess import (EmptySketchError, SegmentationError,
                            prepare_for_3d, prepare_sketch)
from src.reconstruct import ReconstructionError, reconstruct
from src.sketch_to_image import generate_image, infer_label

EXPECTED_ERRORS = (EmptySketchError, SegmentationError,
                   ReconstructionError, MeshValidationError)


class PipelineError(Exception):
    """Wraps any expected failure; str(e) is safe to show to users."""


def warmup() -> None:
    """Load all models once so the first user request is not slow."""
    from src import reconstruct as R
    from src import sketch_to_image as S
    S._load_clip()
    S._load_pipe()
    R._load_model()


def run_pipeline(sketch, prompt=None, seed=42, mc_resolution=None, progress=None) -> dict:
    """sketch: PIL image. Returns glb_path, generated_image, label, timings, report."""
    def step(frac, msg):
        if progress is not None:
            progress(frac, desc=msg)

    mc_resolution = mc_resolution or config.MC_RESOLUTION
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:8]
    t, label = {}, None

    try:
        step(0.05, "Cleaning the sketch...")
        t0 = time.time()
        clean = prepare_sketch(sketch)
        t["prepare_sketch"] = time.time() - t0

        step(0.15, "Recognising and generating a product image...")
        t0 = time.time()
        try:
            label, conf = infer_label(clean)
            label = f"{label} ({conf:.0%})"
        except Exception:
            label = "unknown"
        generated = generate_image(clean, (prompt or "").strip() or None, seed=seed)
        t["generate_image"] = time.time() - t0

        step(0.65, "Removing the background...")
        t0 = time.time()
        rgba = prepare_for_3d(generated)
        t["prepare_for_3d"] = time.time() - t0

        step(0.75, "Building the 3D model...")
        t0 = time.time()
        mesh = reconstruct(rgba, mc_resolution=mc_resolution)
        t["reconstruct"] = time.time() - t0

        step(0.95, "Validating and exporting...")
        t0 = time.time()
        glb = validate_and_export(mesh, str(out_dir / f"{name}.glb"))
        t["export"] = time.time() - t0
    except EXPECTED_ERRORS as e:
        raise PipelineError(str(e)) from e

    t["total"] = sum(t.values())
    return {
        "glb_path": glb,
        "generated_image": generated,
        "label": label,
        "timings": {k: round(v, 1) for k, v in t.items()},
        "report": mesh_report(mesh),
    }