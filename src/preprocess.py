"""P2: Preprocessing.

    prepare_sketch(img)  -> RGB 512x512, WHITE strokes on BLACK (ControlNet scribble format)
    prepare_for_3d(img)  -> RGBA 512x512, background removed, object centred (for image-to-3D)

Shared contract:
    def prepare_sketch(img: Image.Image) -> Image.Image
    def prepare_for_3d(img: Image.Image) -> Image.Image

Edge cases raise readable exceptions (both subclass ValueError) that P4 can show in the UI:
    EmptySketchError   blank canvas / invisible strokes / drawing far too small
    SegmentationError  no object could be found in the image
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

SKETCH_SIZE = 512
SKETCH_MARGIN = 0.15          # empty margin around the drawing, per side (fraction of drawing size)
MIN_CONTRAST = 30             # a sketch needs at least this much gray-level contrast
MIN_STROKE_FRACTION = 0.0003  # strokes must cover at least this fraction of the canvas
MIN_STROKE_EXTENT_PX = 12     # drawing must be at least this many pixels wide or tall
MIN_FOREGROUND_FRACTION = 0.01  # a cut-out smaller than 1% of the image counts as "failed"


class EmptySketchError(ValueError):
    """The sketch is blank, too faint, or too small to use."""


class SegmentationError(ValueError):
    """No foreground object could be isolated from the image."""


_rembg_session = None


def _to_pil(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if not isinstance(img, Image.Image):
        raise TypeError("Expected a PIL image or numpy array.")
    return img


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Composite transparency onto white and return RGB."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


# ----------------------------------------------------------------------------
# prepare_sketch
# ----------------------------------------------------------------------------
def _border_median(gray: np.ndarray) -> float:
    b = max(2, int(0.02 * min(gray.shape)))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(),
                             gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return float(np.median(border))


def _stroke_mask(img: Image.Image) -> np.ndarray:
    """uint8 mask, 255 where there are strokes, for any kind of user sketch."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        alpha = np.array(rgba.getchannel("A"))
        if (alpha < 10).mean() > 0.30:
            # transparent canvas (e.g. Gradio sketchpad): the strokes live in the alpha channel,
            # whatever colour the brush had
            return np.where(alpha > 40, 255, 0).astype(np.uint8)
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)

    gray = np.array(img.convert("L"))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                      # noise reduction
    if int(gray.max()) - int(gray.min()) < MIN_CONTRAST:
        raise EmptySketchError("The sketch looks empty. Please draw something first.")
    gray = np.array(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1))  # contrast

    if _border_median(gray) > 127:
        # light paper, dark strokes: remove uneven lighting (e.g. a photo of paper), then invert
        k = max(15, (min(gray.shape) // 30) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = 255 - cv2.divide(gray, np.maximum(paper, 1), scale=255)
    # now strokes are bright
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _remove_specks(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(12, int(5e-5 * mask.size))
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def prepare_sketch(img, size: int = SKETCH_SIZE, white_on_black: bool = True) -> Image.Image:
    """Any sketch (drawn or uploaded) -> clean, centred, square scribble image.

    Steps: transparency handling, noise reduction, contrast adjustment, polarity detection
    (dark-on-light or light-on-dark), thresholding, speck removal, crop to the drawing,
    square padding with margin, resize, stroke thickening.
    """
    img = _to_pil(img)
    mask = _remove_specks(_stroke_mask(img))

    if (mask > 0).mean() < MIN_STROKE_FRACTION:
        raise EmptySketchError("The sketch looks empty or too faint. Please draw with a darker brush.")

    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if max(h, w) < MIN_STROKE_EXTENT_PX:
        raise EmptySketchError("The drawing is too small. Please draw larger.")

    crop = mask[y0:y1, x0:x1]
    side = int(round(max(h, w) * (1 + 2 * SKETCH_MARGIN)))
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop

    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    out = cv2.resize(canvas, (size, size), interpolation=interp)
    out = ((out > 40) * 255).astype(np.uint8)          # low threshold keeps thin lines alive
    out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=1)
    if not white_on_black:
        out = 255 - out
    return Image.fromarray(out).convert("RGB")


# ----------------------------------------------------------------------------
# prepare_for_3d
# ----------------------------------------------------------------------------
def _coverage(rgba: Image.Image) -> float:
    return float((np.array(rgba.getchannel("A")) > 128).mean())


def _rembg_cutout(rgb: Image.Image):
    global _rembg_session
    try:
        from rembg import new_session, remove
        if _rembg_session is None:
            _rembg_session = new_session("u2net")
        return remove(rgb, session=_rembg_session).convert("RGBA")
    except ImportError:
        log.warning("rembg not installed (pip install rembg onnxruntime); using colour-difference fallback")
    except Exception as e:  # never crash the pipeline on a segmentation hiccup
        log.warning("rembg failed (%s); using colour-difference fallback", e)
    return None


def _background_difference_cutout(rgb: Image.Image) -> Image.Image:
    """Fallback segmentation: everything that differs from the border colour is foreground."""
    arr = np.array(rgb).astype(np.int16)
    h, w, _ = arr.shape
    b = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                             arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = np.abs(arr - bg).max(axis=2)
    mask = ((diff > 28) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask))
    return rgba


def _cutout(img: Image.Image, use_rembg: bool) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        transparent = (np.array(rgba.getchannel("A")) < 10).mean()
        if 0.05 < transparent < 0.95:       # the image already has a real cut-out
            return rgba
    rgb = _flatten_to_rgb(img)
    if use_rembg:
        out = _rembg_cutout(rgb)
        if out is not None and _coverage(out) >= MIN_FOREGROUND_FRACTION:
            return out
        if out is not None:
            log.warning("rembg removed (almost) everything; trying colour-difference fallback")
    out = _background_difference_cutout(rgb)
    if _coverage(out) < MIN_FOREGROUND_FRACTION:
        raise SegmentationError(
            "Could not find an object in the generated image. Try a clearer sketch or add a text prompt.")
    return out


def _keep_main_components(rgba: Image.Image, rel: float = 0.05) -> Image.Image:
    """Drop stray specks: keep only parts at least `rel` times the size of the largest part."""
    alpha = np.array(rgba.getchannel("A"))
    mask = (alpha > 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return rgba
    areas = stats[1:, cv2.CC_STAT_AREA]
    thresh = rel * areas.max()
    keep = np.zeros_like(mask)
    for i, a in enumerate(areas, start=1):
        if a >= thresh:
            keep[labels == i] = 1
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))   # preserve soft edges
    out = rgba.copy()
    out.putalpha(Image.fromarray((alpha * keep).astype(np.uint8)))
    return out


def prepare_for_3d(img, size: int = 512, fill_ratio: float = 0.85,
                   background=None, use_rembg: bool = True) -> Image.Image:
    """Generated image -> object-only RGBA, cropped to the object and centred.

    size        output is size x size
    fill_ratio  the object's longest side fills this fraction of the frame (0.85 suits TripoSR)
    background  None -> transparent RGBA (default, per contract);
                (r, g, b) -> RGB image on that colour, e.g. (128, 128, 128) for TripoSR
    """
    img = _to_pil(img)
    rgba = _keep_main_components(_cutout(img, use_rembg))

    mask = rgba.getchannel("A").point(lambda v: 255 if v > 16 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise SegmentationError("No foreground object found.")

    obj = rgba.crop(bbox)
    w, h = obj.size
    scale = int(size * fill_ratio) / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # resize in premultiplied alpha to avoid dark/white halos at the edges
    obj = obj.convert("RGBa").resize(new_size, Image.LANCZOS).convert("RGBA")

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.paste(obj, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))

    if background is not None:
        bg = Image.new("RGBA", (size, size), tuple(background) + (255,))
        return Image.alpha_composite(bg, canvas).convert("RGB")
    return canvas
