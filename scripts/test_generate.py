"""Standalone test for P1's stage.

Examples (run from the repo root):
    python scripts/test_generate.py samples/sketches/chair.png
    python scripts/test_generate.py samples/sketches/chair.png --prompt "wooden chair"
    python scripts/test_generate.py samples/sketches/chair.png --sweep      # compare conditioning scales
    SKETCH2SHAPE_LOW_VRAM=1 python scripts/test_generate.py ...            # 4-6 GB GPUs
    SKETCH2SHAPE_MOCK=1 python scripts/test_generate.py ...                # no GPU
(On Windows PowerShell: $env:SKETCH2SHAPE_LOW_VRAM="1"; python scripts/...)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.sketch_to_image import generate_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/p1_test.png")
    ap.add_argument("--sweep", action="store_true",
                    help="try several ControlNet conditioning scales side by side")
    args = ap.parse_args()

    sketch = Image.open(args.sketch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        scales = [0.5, 0.6, 0.7, 0.8]
        images = []
        for s in scales:
            t0 = time.time()
            images.append(generate_image(sketch, args.prompt, seed=args.seed, conditioning_scale=s))
            print(f"conditioning_scale={s}: {time.time() - t0:.1f}s")
        w, h = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, im in enumerate(images):
            grid.paste(im, (i * w, 0))
        grid.save(out)
    else:
        t0 = time.time()
        generate_image(sketch, args.prompt, seed=args.seed).save(out)
        print(f"Generated in {time.time() - t0:.1f}s")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
                  f"on {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
