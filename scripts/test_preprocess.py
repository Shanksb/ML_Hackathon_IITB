"""Tests for P2 preprocessing. Run from the repo root:

    python scripts/test_preprocess.py                         # synthetic edge-case tests
    python scripts/test_preprocess.py samples/sketches/mug.png          # + save processed sketch
    python scripts/test_preprocess.py samples/sketches/mug.png outputs/p1_test.png   # + cut-out of a generated image
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.preprocess import (EmptySketchError, SegmentationError,  # noqa: E402
                            prepare_for_3d, prepare_sketch)


def expect_error(exc, fn, label):
    try:
        fn()
    except exc as e:
        print(f"  ok   {label}: {e}")
        return
    raise AssertionError(f"{label}: expected {exc.__name__}")


def mug_on_white(size=600):
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([200, 220, 380, 430], fill=(40, 90, 200))
    d.ellipse([370, 260, 450, 380], outline=(40, 90, 200), width=18)
    return im


def sketch_on_white(size=600, scale=1.0, color="black", bg="white", mode="RGB"):
    im = Image.new(mode, (size, size), bg)
    d = ImageDraw.Draw(im)
    s = scale
    d.rectangle([200, 220, 200 + 180 * s, 220 + 210 * s], outline=color, width=4)
    d.arc([370, 260, 370 + 80 * s, 260 + 120 * s], 270, 90, fill=color, width=4)
    return im


def check_sketch_format(out, label):
    a = np.array(out)
    assert out.size == (512, 512) and out.mode == "RGB", label
    assert set(np.unique(a)) <= {0, 255}, f"{label}: not binary"
    assert 0.002 < (a[..., 0] > 0).mean() < 0.4, f"{label}: stroke coverage looks wrong"
    assert a[0, 0, 0] == 0 and a[-1, -1, 0] == 0, f"{label}: corners must be black"
    print(f"  ok   {label}")


def main():
    print("prepare_sketch")
    check_sketch_format(prepare_sketch(sketch_on_white()), "dark strokes on white")
    check_sketch_format(prepare_sketch(sketch_on_white(color="white", bg="black")), "white strokes on black")
    check_sketch_format(prepare_sketch(sketch_on_white(color=(200, 30, 30))), "coloured strokes")
    check_sketch_format(prepare_sketch(sketch_on_white(scale=0.15)), "tiny drawing is enlarged")
    # transparent canvas, strokes only in alpha (like a Gradio sketchpad layer)
    rgba = sketch_on_white(color=(255, 255, 255, 255), bg=(0, 0, 0, 0), mode="RGBA")
    check_sketch_format(prepare_sketch(rgba), "transparent canvas")
    # noisy upload
    noisy = np.array(sketch_on_white()).astype(np.int16)
    noisy = np.clip(noisy + np.random.default_rng(0).normal(0, 12, noisy.shape), 0, 255).astype(np.uint8)
    check_sketch_format(prepare_sketch(Image.fromarray(noisy)), "noisy upload")
    expect_error(EmptySketchError, lambda: prepare_sketch(Image.new("RGB", (400, 400), "white")), "blank white canvas")
    expect_error(EmptySketchError, lambda: prepare_sketch(Image.new("RGBA", (400, 400), (0, 0, 0, 0))), "blank transparent canvas")
    dot = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(dot).ellipse([200, 200, 204, 204], fill="black")
    expect_error(EmptySketchError, lambda: prepare_sketch(dot), "single tiny dot")

    print("prepare_for_3d")
    out = prepare_for_3d(mug_on_white(), use_rembg=False)
    a = np.array(out.getchannel("A"))
    ys, xs = np.where(a > 128)
    assert out.size == (512, 512) and out.mode == "RGBA"
    assert abs((xs.min() + xs.max()) / 2 - 256) < 6 and abs((ys.min() + ys.max()) / 2 - 256) < 6, "not centred"
    longest = max(xs.max() - xs.min(), ys.max() - ys.min())
    assert abs(longest - 0.85 * 512) < 8, f"fill ratio off: {longest}"
    print("  ok   object centred, fills 85% of the frame, RGBA")
    rgb = prepare_for_3d(mug_on_white(), use_rembg=False, background=(128, 128, 128))
    assert rgb.mode == "RGB" and rgb.getpixel((2, 2)) == (128, 128, 128)
    print("  ok   grey background variant (for TripoSR)")
    expect_error(SegmentationError, lambda: prepare_for_3d(Image.new("RGB", (400, 400), "white"), use_rembg=False),
                 "blank image / nothing to cut out")

    if len(sys.argv) > 1:
        Path("outputs").mkdir(exist_ok=True)
        prepare_sketch(Image.open(sys.argv[1])).save("outputs/p2_sketch.png")
        print("saved outputs/p2_sketch.png")
    if len(sys.argv) > 2:
        prepare_for_3d(Image.open(sys.argv[2])).save("outputs/p2_cutout.png")
        prepare_for_3d(Image.open(sys.argv[2]), background=(128, 128, 128)).save("outputs/p2_cutout_gray.png")
        print("saved outputs/p2_cutout.png and p2_cutout_gray.png")
    print("ALL PREPROCESS TESTS PASSED")


if __name__ == "__main__":
    main()
