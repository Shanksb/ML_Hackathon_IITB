"""Pipeline: sketch -> prepare_sketch -> generate_best_image -> prepare_for_3d
-> reconstruct -> validate_and_export. Models stay cached between calls.
Photos: no prompt -> straight to 3D; with a prompt -> restyled via ControlNet edges."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

import config
from src.export import MeshValidationError, mesh_report, validate_and_export
from src.preprocess import (EmptySketchError, SegmentationError,
                            prepare_for_3d, prepare_sketch)
from src.reconstruct import ReconstructionError, reconstruct
from src.sketch_to_image import generate_best_image, generate_image, infer_label

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


def _on_white(img: Image.Image) -> Image.Image:
    """Composite any image onto white and return RGB."""
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def looks_like_photo(img: Image.Image) -> bool:
    """Line art is nearly two-tone; photos have many colours and mid-tones."""
    g = np.array(_on_white(img).convert("L").resize((128, 128)))
    mid = ((g > 60) & (g < 195)).mean()
    return bool(mid > 0.25)


def photo_to_edges(img: Image.Image) -> Image.Image:
    """Photo -> white edges on black (ControlNet scribble format)."""
    g = cv2.GaussianBlur(np.array(_on_white(img).convert("L")), (5, 5), 0)
    edges = cv2.Canny(g, 50, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(edges).convert("RGB")


def boost_colour(img: Image.Image, factor: float = 1.4) -> Image.Image:
    """Counteract TripoSR's washed-out colours. Skips transparent cut-outs."""
    if img.mode == "RGBA":
        return img
    return ImageEnhance.Color(img.convert("RGB")).enhance(factor)


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
            if prompt:
                step(0.15, "Photo + prompt: restyling with your prompt...")
                label = "photo + prompt"
                t0 = time.time()
                generated = generate_image(photo_to_edges(sketch), prompt, seed=seed,
                                           conditioning_scale=0.9, auto_label=False)
                t["prepare_sketch"] = 0.0
                t["generate_image"] = time.time() - t0
            else:
                step(0.15, "Photo detected, skipping generation...")
                label = "photo (generation skipped)"
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
        rgba = prepare_for_3d(boost_colour(generated))
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
        "generated_image": _on_white(generated),
        "label": label,
        "timings": {k: round(v, 1) for k, v in t.items()},
        "report": mesh_report(mesh),
        "generation": gen_scores,
    }