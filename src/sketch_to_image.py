"""P1: Sketch understanding and generation.

generate_image(sketch, prompt) -> clean, centred, product-style PIL image
using Stable Diffusion 1.5 + ControlNet (scribble) via Hugging Face diffusers.

Contract (shared with the team):
    def generate_image(sketch: Image.Image, prompt: str | None = None) -> Image.Image

Extras for P3/P4:
    - MOCK_MODE (config.MOCK_MODE or env SKETCH2SHAPE_MOCK=1): no GPU needed
    - LOW_VRAM  (config.LOW_VRAM  or env SKETCH2SHAPE_LOW_VRAM=1): for 4-6 GB GPUs
    - unload_model(): frees VRAM so the 3D model can load afterwards
"""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:  # config.py is owned by P4; fall back to defaults if it is not there yet
    import config  # type: ignore
except ImportError:  # pragma: no cover
    config = None

log = logging.getLogger(__name__)

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_scribble"

DEFAULT_SUBJECT = "a single object"
STYLE_SUFFIX = (
    "centered, product render, plain white background, "
    "soft studio lighting, high detail, sharp focus"
)
NEGATIVE_PROMPT = (
    "multiple objects, text, watermark, logo, blurry, low quality, cropped, "
    "out of frame, cluttered background, people, hands"
)

# Defaults to tune (log results in docs/p1_experiments.md)
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
DEFAULT_CONDITIONING_SCALE = 0.9
IMG_SIZE = 512

_pipe = None


def _flag(name: str, env: str, default: bool = False) -> bool:
    if os.environ.get(env) is not None:
        return os.environ[env] == "1"
    return bool(getattr(config, name, default))


def _mock_mode() -> bool:
    return _flag("MOCK_MODE", "SKETCH2SHAPE_MOCK")


def _low_vram() -> bool:
    return _flag("LOW_VRAM", "SKETCH2SHAPE_LOW_VRAM")


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
def _to_scribble(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Make any sketch look like what scribble ControlNet expects:
    WHITE strokes on a BLACK background, square, size x size, RGB.
    (P2's prepare_sketch can replace/extend this later.)"""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = np.array(img.convert("L"))
    if g.mean() > 127:  # drawn as dark strokes on light paper -> invert
        g = 255 - g
    g = ((g > 40) * 255).astype(np.uint8)

    if (g > 0).mean() < 0.0005:
        raise ValueError("The sketch looks empty. Please draw something first.")

    # pad to square on black, then resize
    h, w = g.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
    out = Image.fromarray(canvas).resize((size, size), Image.LANCZOS)
    out = out.filter(ImageFilter.MaxFilter(3))  # thicken thin strokes
    return out.convert("RGB")


def build_prompt(prompt: str | None) -> str:
    """User prompt is optional. Always append the style suffix so the output is
    a clean single object that image-to-3D models handle well."""
    subject = (prompt or "").strip() or DEFAULT_SUBJECT
    return f"{subject}, {STYLE_SUFFIX}"


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    t0 = time.time()
    log.info("Loading SD1.5 + ControlNet (cuda=%s, low_vram=%s)...", use_cuda, _low_vram())

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if use_cuda:
        pipe.enable_attention_slicing()
        if _low_vram():
            pipe.enable_model_cpu_offload()  # needs `accelerate`
            pipe.enable_vae_slicing()
        else:
            pipe.to("cuda")
    # on CPU: works but very slow (minutes); use Kaggle/Colab instead

    _pipe = pipe
    log.info("Model loaded in %.1fs", time.time() - t0)
    return _pipe


def unload_model() -> None:
    """Free GPU memory (call before loading the 3D model on small GPUs)."""
    global _pipe
    _pipe = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def generate_image(
    sketch: Image.Image,
    prompt: str | None = None,
    seed: int | None = 42,
    steps: int | None = None,
    guidance_scale: float | None = None,
    conditioning_scale: float | None = None,
) -> Image.Image:
    """Sketch (+ optional prompt) -> refined 512x512 RGB product-style image."""
    cond = _to_scribble(sketch)

    if _mock_mode():
        mock = Path(__file__).resolve().parents[1] / "samples" / "outputs" / "mock_generated.png"
        if mock.exists():
            return Image.open(mock).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        # no mock file yet: show the processed sketch as black strokes on white
        return Image.eval(cond, lambda v: 255 - v)

    import torch

    pipe = _load_pipe()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=build_prompt(prompt),
            negative_prompt=NEGATIVE_PROMPT,
            image=cond,
            num_inference_steps=steps or DEFAULT_STEPS,
            guidance_scale=guidance_scale or DEFAULT_GUIDANCE,
            controlnet_conditioning_scale=(
                conditioning_scale if conditioning_scale is not None
                else DEFAULT_CONDITIONING_SCALE
            ),
            generator=generator,
        )
    log.info("generate_image took %.1fs", time.time() - t0)
    return result.images[0].convert("RGB")
