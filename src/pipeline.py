"""Pipeline: sketch -> prepare_sketch -> generate_best_image -> prepare_for_3d
-> reconstruct -> validate_and_export. Models stay cached between calls.
Photos (many colours) skip ControlNet and go straight to 3D."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src.export import MeshValidationError, mesh_report, validate_and_export
from src.preprocess import (EmptySketchError, SegmentationError,
                            prepare_for_3d, prepare_sketch)
from src.reconstruct import ReconstructionError, reconstruct
from src.sketch_to_image import generate_best_image, infer_label

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


def looks_like_photo(img: Image.Image) -> bool:
    """Line art is nearly two-tone; photos have many colours and mid-tones."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    g = np.array(img.convert("L").resize((128, 128)))
    mid = ((g > 60) & (g < 195)).mean()      # share of mid-gray pixels
    return bool(mid > 0.25)


def run_pipeline(sketch, prompt=None, seed=42, mc_resolution=None, progress=None) -> dict:
    """sketch: PIL image. Returns glb_path, generated_image, label, timings,
    report and generation (candidate scores, empty for photos)."""
    def step(frac, msg):
        if progress is not None:
            progress(frac, desc=msg)

    mc_resolution = mc_resolution or config.MC_RESOLUTION
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:8]
    t, label, gen_scores = {}, None, []
    prompt = (prompt or "").strip() or None

    try:
        if looks_like_photo(sketch):
            # Photo: skip sketch cleaning and ControlNet, cut it out and go to 3D
            step(0.15, "Photo detected, skipping generation...")
            label = "photo (generation skipped)"
            # keep RGBA so an existing cut-out is preserved; otherwise use RGB
            generated = sketch if sketch.mode == "RGBA" else sketch.convert("RGB")
            t["prepare_sketch"] = 0.0
            t["generate_image"] = 0.0
        else:
            step(0.05, "Cleaning the sketch...")
            t0 = time.time()
            clean = prepare_sketch(sketch)
            t["prepare_sketch"] = time.time() - t0

            step(0.15, "Recognising and generating a product image...")
            t0 = time.time()
            try:
                lab, conf = infer_label(clean)
                label = f"{lab} ({conf:.0%})"
            except Exception:
                label = "unknown"
            generated, gen_info = generate_best_image(
                clean, prompt, seed=seed, n_candidates=config.N_CANDIDATES)
            gen_scores = gen_info["scores"]
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
        "generated_image": generated.convert("RGB") if generated.mode != "RGB" else generated,
        "label": label,
        "timings": {k: round(v, 1) for k, v in t.items()},
        "report": mesh_report(mesh),
        "generation": gen_scores,
    }