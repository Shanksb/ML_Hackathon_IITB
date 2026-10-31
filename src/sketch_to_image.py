"""P1: Sketch understanding and generation (v2: works from the sketch alone).

generate_image(sketch, prompt) -> clean, centred, single-object image on a
PURE WHITE background, ready for background removal / image-to-3D.

How it works without a prompt:
    1. The sketch is cleaned and converted to white-strokes-on-black (ControlNet format).
    2. If no prompt is given, CLIP (zero-shot, runs on CPU) recognises WHAT the sketch
       shows ("mug", "chair", "car", ...) and builds the prompt "a mug, ..." itself.
    3. SD 1.5 + ControlNet (scribble) generates a product-style image.
    4. rembg cuts out the object and composites it on pure white (255,255,255).

Contract (shared with the team):
    def generate_image(sketch: Image.Image, prompt: str | None = None) -> Image.Image

Extras:
    infer_label(sketch) -> (label, confidence)   # P4 can show "Detected: mug"
    unload_model()                               # frees VRAM before the 3D model loads
    MOCK_MODE (config.MOCK_MODE or env SKETCH2SHAPE_MOCK=1)
    LOW_VRAM  (config.LOW_VRAM  or env SKETCH2SHAPE_LOW_VRAM=1)
"""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # config.py is owned by P4; fall back to defaults if it is not there yet
    import config  # type: ignore
except ImportError:  # pragma: no cover
    config = None

log = logging.getLogger(__name__)

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_scribble"
CLIP_ID = "openai/clip-vit-base-patch32"

STYLE_SUFFIX = (
    "single isolated object, front three-quarter view, centered, "
    "matte finish, flat even diffuse lighting, soft uniform colors, "
    "no reflections, no highlights, no shadows, pure white background, high detail"
)
NEGATIVE_PROMPT = (
    "multiple objects, extra objects, saucer, plate, coaster, table, floor, "
    "shadow, harsh shadows, glossy, shiny, reflective, specular highlights, "
    "dark, gray background, colored background, text, watermark, logo, "
    "blurry, low quality, cropped, out of frame, cluttered background, people, hands"
)
FALLBACK_SUBJECT = "a simple object"
CLIP_MIN_CONF = 0.15  # below this we do not trust the detected label

# Tune these (log results in docs/p1_experiments.md)
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
DEFAULT_CONDITIONING_SCALE = 0.7
IMG_SIZE = 512

# Objects CLIP can recognise from a sketch. Add your own categories here.
LABELS = [
    "mug", "cup", "teapot", "kettle", "bottle", "wine glass", "bowl", "plate", "vase",
    "chair", "armchair", "sofa", "stool", "table", "bench", "bed", "lamp", "desk lamp",
    "floor lamp", "candle", "clock", "alarm clock", "wristwatch", "backpack", "handbag",
    "suitcase", "umbrella", "hat", "shoe", "sneaker", "boot", "sandal", "glasses",
    "sunglasses", "headphones", "camera", "laptop", "computer monitor", "keyboard",
    "computer mouse", "smartphone", "tablet", "television", "radio", "speaker", "guitar",
    "violin", "piano", "drum", "trumpet", "car", "sports car", "truck", "bus", "van",
    "motorcycle", "bicycle", "scooter", "airplane", "helicopter", "rocket", "boat",
    "sailboat", "ship", "train", "tractor", "house", "castle", "lighthouse", "tent",
    "tower", "bridge", "door", "window", "key", "scissors", "hammer", "screwdriver",
    "wrench", "axe", "saw", "knife", "fork", "spoon", "pencil", "pen", "book", "envelope",
    "trophy", "crown", "sword", "shield", "robot", "teddy bear", "toy car", "rubber duck",
    "doll", "cat", "dog", "horse", "cow", "pig", "sheep", "elephant", "lion", "giraffe",
    "rabbit", "bird", "owl", "penguin", "duck", "fish", "shark", "dolphin", "whale",
    "turtle", "snake", "frog", "butterfly", "bee", "spider", "dinosaur", "dragon",
    "apple", "banana", "pineapple", "strawberry", "watermelon", "pear", "carrot",
    "mushroom", "pumpkin", "cake", "ice cream cone", "pizza", "hamburger", "cupcake",
    "flower", "rose", "tree", "cactus", "potted plant", "star", "heart", "moon",
    "light bulb", "bell", "anchor", "balloon", "gift box", "treasure chest", "basket",
    "bucket", "barrel", "fan", "toaster", "microwave", "refrigerator", "washing machine",
    "bathtub", "toilet", "sink", "faucet", "fireplace", "ladder", "skateboard",
    "football", "basketball", "tennis racket", "dumbbell", "helmet", "glove",
    "t-shirt", "dress", "jacket",
]

_pipe = None
_clip = None
_rembg_session = None


def _flag(name: str, env: str, default: bool = False) -> bool:
    if os.environ.get(env) is not None:
        return os.environ[env] == "1"
    return bool(getattr(config, name, default))


def _mock_mode() -> bool:
    return _flag("MOCK_MODE", "SKETCH2SHAPE_MOCK")


def _low_vram() -> bool:
    return _flag("LOW_VRAM", "SKETCH2SHAPE_LOW_VRAM")


# ----------------------------------------------------------------------------
# 1. Input handling
# ----------------------------------------------------------------------------
def _to_scribble(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Make any sketch look like what scribble ControlNet expects:
    WHITE strokes on a BLACK background, square, size x size, RGB."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = np.array(img.convert("L"))
    if g.mean() > 127:  # dark strokes on light paper -> invert
        g = 255 - g
    g = ((g > 40) * 255).astype(np.uint8)

    if (g > 0).mean() < 0.0005:
        raise ValueError("The sketch looks empty. Please draw something first.")

    # crop to the drawn strokes, then pad to a square with a margin
    ys, xs = np.where(g > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = g[y0:y1, x0:x1]
    h, w = g.shape
    side = int(max(h, w) * 1.3)
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = g

    out = Image.fromarray(canvas).resize((size, size), Image.LANCZOS)
    out = out.filter(ImageFilter.MaxFilter(3))  # thicken thin strokes
    return out.convert("RGB")


# ----------------------------------------------------------------------------
# 2. Sketch understanding (CLIP zero-shot, CPU) -> prompt without user input
# ----------------------------------------------------------------------------
def _load_clip():
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        log.info("Loading CLIP (%s) on CPU for sketch recognition...", CLIP_ID)
        _clip = (CLIPModel.from_pretrained(CLIP_ID).eval(),
                 CLIPProcessor.from_pretrained(CLIP_ID))
    return _clip


def infer_label(sketch: Image.Image) -> tuple[str, float]:
    """What does the sketch show? Returns (label, confidence 0..1)."""
    import torch

    cond = _to_scribble(sketch)
    view = ImageOps.invert(cond)  # black strokes on white = natural for CLIP
    model, proc = _load_clip()
    texts = [f"a simple line drawing of a {label}" for label in LABELS]
    inputs = proc(text=texts, images=view, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
    probs = logits.softmax(dim=-1)
    top = int(probs.argmax())
    return LABELS[top], float(probs[top])


def build_prompt(prompt: str | None, label: str | None = None) -> str:
    """Prompt is optional and only adds detail (material, colour, style).
    - no prompt            -> "a mug"            (from the sketch)
    - short prompt "red"   -> "red mug"          (attribute + detected object)
    - full prompt          -> used as written
    """
    user = (prompt or "").strip()
    if user:
        if label and len(user.split()) <= 3 and label.lower() not in user.lower():
            subject = f"{user} {label}"
        else:
            subject = user
    else:
        subject = f"a {label}" if label else FALLBACK_SUBJECT
    return f"{subject}, {STYLE_SUFFIX}"

# ----------------------------------------------------------------------------
# 3. Stable Diffusion + ControlNet
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
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        if _low_vram():
            pipe.enable_model_cpu_offload()  # needs `accelerate`
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
        else:
            pipe.to("cuda")
    # on CPU it works but is very slow (minutes): use Kaggle/Colab instead

    _pipe = pipe
    log.info("Model loaded in %.1fs", time.time() - t0)
    return _pipe


def unload_model() -> None:
    """Free memory (call before loading the 3D model on small GPUs)."""
    global _pipe, _clip, _rembg_session
    _pipe = None
    _clip = None
    _rembg_session = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# 4. Pure white background
# ----------------------------------------------------------------------------
def _force_white_background(img: Image.Image) -> Image.Image:
    """Cut the object out with rembg and place it on pure white (255,255,255).
    Falls back to the original image if rembg is missing or removes everything."""
    global _rembg_session
    try:
        from rembg import new_session, remove
        if _rembg_session is None:
            _rembg_session = new_session("u2net")
        fg = remove(img.convert("RGB"), session=_rembg_session).convert("RGBA")
        alpha = np.array(fg.getchannel("A"))
        if (alpha > 128).mean() < 0.02:  # nothing found: do not return a blank image
            log.warning("rembg removed (almost) everything; keeping original image")
            return img
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, fg).convert("RGB")
    except ImportError:
        log.warning("rembg not installed (pip install rembg onnxruntime): "
                    "background not forced to white")
        return img
    except Exception as e:  # never break the pipeline because of cleanup
        log.warning("Background cleanup failed (%s); keeping original image", e)
        return img


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
    white_background: bool = True,
    auto_label: bool = True,
) -> Image.Image:
    """Sketch (+ optional prompt) -> 512x512 RGB product-style image on white."""
    cond = _to_scribble(sketch)

    if _mock_mode():
        mock = Path(__file__).resolve().parents[1] / "samples" / "outputs" / "mock_generated.png"
        if mock.exists():
            return Image.open(mock).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return ImageOps.invert(cond)  # processed sketch as black strokes on white

    import torch

    # --- understand the sketch when the user gave no prompt
        # --- understand the sketch (also used to complete short prompts)
    label = None
    if auto_label:
        try:
            lab, conf = infer_label(sketch)
            log.info("Detected from sketch: %s (confidence %.2f)", lab, conf)
            has_prompt = bool((prompt or "").strip())
            # be stricter when the user typed something, so a wrong guess cannot hijack it
            if conf >= (0.30 if has_prompt else CLIP_MIN_CONF):
                label = lab
        except Exception as e:
            log.warning("Sketch recognition failed (%s); using neutral prompt", e)

    full_prompt = build_prompt(prompt, label)
    log.info("Prompt: %s", full_prompt)

    pipe = _load_pipe()
    scale = DEFAULT_CONDITIONING_SCALE if conditioning_scale is None else conditioning_scale

    def _run():
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.inference_mode():
            return pipe(
                prompt=full_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=cond,
                num_inference_steps=steps or DEFAULT_STEPS,
                guidance_scale=guidance_scale or DEFAULT_GUIDANCE,
                controlnet_conditioning_scale=scale,
                generator=generator,
            ).images[0]

    t0 = time.time()
    try:
        img = _run()
    except torch.cuda.OutOfMemoryError:  # one retry after freeing cached memory
        log.warning("CUDA out of memory; clearing cache and retrying once")
        gc.collect()
        torch.cuda.empty_cache()
        img = _run()
    log.info("Diffusion took %.1fs", time.time() - t0)

    img = img.convert("RGB")
    if white_background:
        img = _force_white_background(img)
    return img

# ----------------------------------------------------------------------------
# Best-of-N generation (candidate scoring)
# ----------------------------------------------------------------------------
def _diffuse(pipe, cond, full_prompt, seed, steps, guidance, scale):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.inference_mode():
        return pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=cond,
            num_inference_steps=steps or DEFAULT_STEPS,
            guidance_scale=guidance or DEFAULT_GUIDANCE,
            controlnet_conditioning_scale=scale,
            generator=generator,
        ).images[0].convert("RGB")


def _clip_label_probs(images, label):
    """Probability CLIP assigns to `label` for each image (0..1)."""
    import torch
    model, proc = _load_clip()
    texts = [f"a product photo of a {l}" for l in LABELS]
    inputs = proc(text=texts, images=images, return_tensors="pt", padding=True)
    with torch.inference_mode():
        sims = logits[:, LABELS.index(label)]/100.0
    return sims.tolist()


def _fidelity(candidate: Image.Image, cond: Image.Image) -> float:
    """How closely the candidate's edges follow the user's strokes (0..1)."""
    import cv2
    gray = cv2.cvtColor(np.array(candidate.resize(cond.size)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    strokes = np.array(cond.convert("L")) > 127
    if edges.sum() == 0 or not strokes.any():
        return 0.0
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.exp(-dist[strokes] / 20.0).mean())


def generate_best_image(
    sketch: Image.Image,
    prompt: str | None = None,
    seed: int = 42,
    n_candidates: int = 3,
    w_clip: float = 0.6,
    steps: int | None = None,
    guidance_scale: float | None = None,
    conditioning_scale: float | None = None,
) -> tuple[Image.Image, dict]:
    """Generate n_candidates images (seeds seed, seed+1, ...) and return the best one.

    Returns (best_image, info). info has label, confidence, scores per candidate,
    best_index and the candidate images (for saving or showing in the UI).
    With n_candidates=1 it behaves like generate_image().
    """
    cond = _to_scribble(sketch)

    if _mock_mode() or n_candidates <= 1:
        img = generate_image(sketch, prompt, seed=seed, steps=steps,
                             guidance_scale=guidance_scale,
                             conditioning_scale=conditioning_scale)
        return img, {"label": None, "confidence": None, "scores": [],
                     "best_index": 0, "candidates": [img]}

    import torch

    # understand the sketch (same rules as generate_image)
    label, conf = None, 0.0
    try:
        lab, conf = infer_label(sketch)
        has_prompt = bool((prompt or "").strip())
        if conf >= (0.30 if has_prompt else CLIP_MIN_CONF):
            label = lab
    except Exception as e:
        log.warning("Sketch recognition failed (%s)", e)

    full_prompt = build_prompt(prompt, label)
    pipe = _load_pipe()
    scale = DEFAULT_CONDITIONING_SCALE if conditioning_scale is None else conditioning_scale

    candidates = []
    for i in range(n_candidates):
        s = int(seed) + i
        t0 = time.time()
        try:
            img = _diffuse(pipe, cond, full_prompt, s, steps, guidance_scale, scale)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            img = _diffuse(pipe, cond, full_prompt, s, steps, guidance_scale, scale)
        candidates.append(_force_white_background(img))
        log.info("Candidate %d/%d (seed %d) in %.1fs", i + 1, n_candidates, s, time.time() - t0)

    # score every candidate
    if label is not None:
        clip_scores = _clip_label_probs(candidates, label)
    else:
        clip_scores = [0.0] * len(candidates)
    w = w_clip if label is not None else 0.0          # no label -> fidelity only
    scores = []
    for i, (img, c) in enumerate(zip(candidates, clip_scores)):
        f = _fidelity(img, cond)
        scores.append({"seed": int(seed) + i, "clip": round(c, 3),
                       "fidelity": round(f, 3), "total": round(w * c + (1 - w) * f, 3)})
    best = int(np.argmax([s["total"] for s in scores]))
    log.info("Scores: %s -> best candidate %d", scores, best)

    return candidates[best], {"label": label, "confidence": conf, "scores": scores,
                              "best_index": best, "candidates": candidates}