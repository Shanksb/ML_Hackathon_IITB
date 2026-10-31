import sys
import time
import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.append("third_party/TripoSR")
from tsr.system import TSR
from tsr.utils import resize_foreground

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

# Resolutions to try in order if we run out of GPU memory
RESOLUTION_LADDER = [256, 192, 128]


class ReconstructionError(Exception):
    """Raised when 3D reconstruction fails. The message is safe to show to users."""


def _load_model():
    """Load weights once and reuse them."""
    global _model
    if _model is None:
        m = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        m.renderer.set_chunk_size(8192)
        m.to(DEVICE)
        _model = m
    return _model


def _orient(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate TripoSR's native frame so the object stands upright (Y-up).
    The third rotation was found by experiment (x+90 made the chair upright)."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def _check_input(img: Image.Image) -> None:
    """Reject inputs that cannot produce a sensible mesh."""
    if not isinstance(img, Image.Image):
        raise ReconstructionError("Input must be a PIL image.")
    if img.mode != "RGBA":
        raise ReconstructionError(
            "Input must be an RGBA image with the background removed "
            f"(got mode {img.mode}). Run prepare_for_3d() first."
        )
    alpha = np.array(img)[..., 3]
    coverage = (alpha > 127).mean()
    if coverage < 0.01:
        raise ReconstructionError(
            "No object found in the image (background removal removed everything). "
            "Try a clearer, larger drawing."
        )


def _run_once(img: Image.Image, resolution: int, foreground_ratio: float) -> trimesh.Trimesh:
    """One reconstruction attempt at a given resolution. May raise OOM or ValueError."""
    model = _load_model()

    img = resize_foreground(img, foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + 0.5 * (1.0 - arr[..., 3:4])  # grey background
    image = Image.fromarray((rgb * 255).astype(np.uint8))

    with torch.no_grad():
        codes = model([image], device=DEVICE)
        meshes = model.extract_mesh(codes, True, resolution=resolution)
    return meshes[0]


def reconstruct(img: Image.Image, mc_resolution: int = 256,
                foreground_ratio: float = 0.85) -> trimesh.Trimesh:
    """RGBA image (background removed) -> coloured, upright trimesh.

    On GPU out-of-memory, retries at lower resolutions.
    Raises ReconstructionError with a user-friendly message on any failure.
    Run info is stored in mesh.metadata (resolution_used, seconds).
    """
    _check_input(img)

    # Start the ladder at the requested resolution
    ladder = [r for r in RESOLUTION_LADDER if r <= mc_resolution] or [mc_resolution]
    if mc_resolution not in ladder:
        ladder.insert(0, mc_resolution)

    last_error = None
    for res in ladder:
        t0 = time.time()
        try:
            mesh = _run_once(img, res, foreground_ratio)
        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print(f"[reconstruct] out of GPU memory at resolution {res}, retrying lower...")
            torch.cuda.empty_cache()
            continue
        except ValueError as e:
            # scikit-image raises ValueError when marching cubes finds no surface
            raise ReconstructionError(
                "The model could not find a 3D surface for this image. "
                "Try a simpler, clearer object on a plain background."
            ) from e
        except Exception as e:
            raise ReconstructionError(f"Reconstruction failed unexpectedly: {e}") from e
        finally:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ReconstructionError("Reconstruction produced an empty mesh.")

        mesh = _orient(mesh)
        mesh.metadata["resolution_used"] = res
        mesh.metadata["seconds"] = round(time.time() - t0, 2)
        return mesh

    raise ReconstructionError(
        "Ran out of GPU memory even at the lowest resolution. "
        "Close other GPU jobs or use a GPU with more memory."
    ) from last_error