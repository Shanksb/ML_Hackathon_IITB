"""Compare single generation with best-of-N on a sketch; saves all candidates.
    python scripts/test_best_of_n.py samples/sketches/mug.png --n 3"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.preprocess import prepare_sketch
from src.sketch_to_image import generate_best_image

ap = argparse.ArgumentParser()
ap.add_argument("sketch"); ap.add_argument("--n", type=int, default=3)
ap.add_argument("--prompt", default=None); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

out = Path("outputs/best_of_n"); out.mkdir(parents=True, exist_ok=True)
sketch = prepare_sketch(Image.open(a.sketch))
best, info = generate_best_image(sketch, a.prompt, seed=a.seed, n_candidates=a.n)
name = Path(a.sketch).stem
for i, img in enumerate(info["candidates"]):
    img.save(out / f"{name}_cand{i}.png")
best.save(out / f"{name}_BEST.png")
print("label:", info["label"], round(info["confidence"] or 0, 2))
for i, s in enumerate(info["scores"]):
    print(("-> " if i == info["best_index"] else "   ") + str(s))