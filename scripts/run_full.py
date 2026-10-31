"""Full pipeline: sketch -> P2 prepare_sketch -> P1 generate_image
-> P2 prepare_for_3d -> P3 reconstruct -> P2 validate_and_export.

    python scripts/run_full.py samples/sketches/mug.png
    python scripts/run_full.py samples/sketches/mug.png --prompt "red metal" --seed 7
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.export import MeshValidationError, mesh_report, validate_and_export  # noqa: E402
from src.preprocess import (EmptySketchError, SegmentationError,  # noqa: E402
                            prepare_for_3d, prepare_sketch)
from src.reconstruct import ReconstructionError, reconstruct  # noqa: E402
from src.sketch_to_image import generate_image, infer_label, unload_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

EXPECTED_ERRORS = (EmptySketchError, SegmentationError, ReconstructionError, MeshValidationError)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--res", type=int, default=256, help="marching-cubes resolution")
    ap.add_argument("--outdir", default="outputs/full")
    ap.add_argument("--unload", action="store_true",
                    help="free the diffusion model before 3D (only needed on small GPUs)")
    args = ap.parse_args()

    sketch_path = Path(args.sketch).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = sketch_path.stem
    t = {}

    try:
        t0 = time.time()
        sketch = prepare_sketch(Image.open(sketch_path))
        sketch.save(out / f"{name}_1_sketch.png")
        t["prepare_sketch"] = time.time() - t0

        t0 = time.time()
        label, conf = infer_label(sketch)
        print(f"Detected: {label} ({conf:.2f})")
        image = generate_image(sketch, args.prompt, seed=args.seed)
        image.save(out / f"{name}_2_generated.png")
        t["generate_image"] = time.time() - t0
        if args.unload:
            unload_model()

        t0 = time.time()
        rgba = prepare_for_3d(image)
        rgba.save(out / f"{name}_3_for_3d.png")
        t["prepare_for_3d"] = time.time() - t0

        t0 = time.time()
        mesh = reconstruct(rgba, mc_resolution=args.res)
        t["reconstruct"] = time.time() - t0

        t0 = time.time()
        glb = validate_and_export(mesh, str(out / f"{name}.glb"))
        t["export"] = time.time() - t0

    except EXPECTED_ERRORS as e:
        print(f"\nFAILED ({type(e).__name__}): {e}")
        sys.exit(1)

    t["total"] = sum(t.values())
    report = mesh_report(mesh)
    print("\n=== RESULT ===")
    print("GLB:", glb)
    print("Mesh:", report)
    print("Timings (s):", {k: round(v, 1) for k, v in t.items()})
    (out / f"{name}_report.json").write_text(
        json.dumps({"mesh": report, "timings": t, "prompt": args.prompt,
                    "seed": args.seed, "label": label}, indent=2))


if __name__ == "__main__":
    main()