"""P1 + P2 together: sketch -> prepare_sketch -> generate_image -> prepare_for_3d.
(P3's reconstruct() and P2's validate_and_export() then take it from there.)

    python scripts/run_p1_p2.py samples/sketches/mug.png
    python scripts/run_p1_p2.py samples/sketches/mug.png --prompt "red metal"
Windows PowerShell low-VRAM mode:  $env:SKETCH2SHAPE_LOW_VRAM="1"
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.preprocess import prepare_for_3d, prepare_sketch  # noqa: E402
from src.sketch_to_image import generate_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sketch = prepare_sketch(Image.open(args.sketch))            # P2
    sketch.save(out / "step1_sketch.png")
    image = generate_image(sketch, args.prompt, seed=args.seed)  # P1
    image.save(out / "step2_generated.png")
    ready = prepare_for_3d(image)                                # P2 (transparent RGBA)
    ready.save(out / "step3_for_3d.png")
    prepare_for_3d(image, background=(128, 128, 128)).save(out / "step3_for_3d_gray.png")

    print(f"Done in {time.time() - t0:.1f}s -> {out}/step1_sketch.png, step2_generated.png, step3_for_3d*.png")


if __name__ == "__main__":
    main()
