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
        logits = model(**inputs).logits_per_image          # [n_images, n_labels]
    return logits.softmax(dim=-1)[:, LABELS.index(label)].tolist()


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