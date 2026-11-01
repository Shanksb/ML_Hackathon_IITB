import sys
import time
import numpy as np
import torch
import trimesh
from PIL import Image
from PIL import ImageEnhance

sys.path.append("third_party/TripoSR")
from tsr.system import TSR
from tsr.utils import resize_foreground

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

# Resolutions to try in order if we run out of GPU memory
RESOLUTION_LADDER = [256, 192, 128]


class ReconstructionError(Exception):
    """Raised when 3D reconstruction fails. The message is safe to show to users."""

import re

def _convert_key(k: str) -> str:
    """Old HF ViT key names -> new (transformers 5.x) names."""
    k = k.replace(".encoder.layer.", ".layers.")
    k = k.replace(".attention.attention.query.", ".attention.q_proj.")
    k = k.replace(".attention.attention.key.", ".attention.k_proj.")
    k = k.replace(".attention.attention.value.", ".attention.v_proj.")
    k = k.replace(".attention.output.dense.", ".attention.o_proj.")
    k = k.replace(".intermediate.dense.", ".mlp.fc1.")
    # the MLP output projection: "layers.N.output.dense" -> "layers.N.mlp.fc2"
    k = re.sub(r"\.layers\.(\d+)\.output\.dense\.", r".layers.\1.mlp.fc2.", k)
    return k


def _load_model():
    """Load weights once and reuse them."""
    global _model
    if _model is None:
        from omegaconf import OmegaConf
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download("stabilityai/TripoSR", "config.yaml")
        ckpt_path = hf_hub_download("stabilityai/TripoSR", "model.ckpt")
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(cfg)
        m = TSR(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = {_convert_key(k): v for k, v in ckpt.items()}
        m.load_state_dict(ckpt)
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
    trimesh.smoothing.filter_laplacian(mesh, iterations=3)
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


def _run_once(img: Image.Image, resolution: int, foreground_ratio: float,
              bake: bool = False, texture_resolution: int = 1024) -> trimesh.Trimesh:
    """One reconstruction attempt. Tries a baked texture, falls back to vertex colours."""
    model = _load_model()

    img = resize_foreground(img, foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + 0.5 * (1.0 - arr[..., 3:4])  # grey background
    image = Image.fromarray((rgb * 255).astype(np.uint8))

    with torch.no_grad():
        codes = model([image], device=DEVICE)

        if bake:
            try:
                from tsr.bake_texture import bake_texture
                raw = model.extract_mesh(codes, False, resolution=resolution)[0]
                out = bake_texture(raw, model, codes[0], texture_resolution)
                tex = (np.clip(out["colors"], 0, 1) * 255.0).astype(np.uint8)
                tex_img = Image.fromarray(tex).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
                mesh = trimesh.Trimesh(
                    vertices=raw.vertices[out["vmapping"]],
                    faces=out["indices"],
                    vertex_normals=raw.vertex_normals[out["vmapping"]],
                    visual=trimesh.visual.TextureVisuals(uv=out["uvs"], image=tex_img),
                    process=False,
                )
                mesh.metadata["textured"] = True
                return mesh
            except Exception as e:
                print(f"[reconstruct] texture baking failed ({type(e).__name__}: {e}); "
                      "using vertex colours")
                torch.cuda.empty_cache()

        mesh = model.extract_mesh(codes, True, resolution=resolution)[0]
        mesh.metadata["textured"] = False
        return mesh

def _recolor_from_image(mesh, rgba: Image.Image, blend: float = 0.85):
    """Pull vertex colours from the reference image for camera-facing vertices."""
    img = resize_foreground(rgba, 0.85)
    arr = np.array(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    v = np.asarray(mesh.vertices)
    # TripoSR frame: camera looks along -z/+z, image x = v[:,0], image y = -v[:,2]... see note
    u = (v[:, 0] / (np.abs(v[:, 0]).max() + 1e-8) * 0.5 + 0.5) * (w - 1)
    t = (0.5 - v[:, 1] / (np.abs(v[:, 1]).max() + 1e-8) * 0.5) * (h - 1)
    px = arr[np.clip(t.astype(int), 0, h - 1), np.clip(u.astype(int), 0, w - 1)]
    facing = np.asarray(mesh.vertex_normals)[:, 2] > 0.2
    ok = facing & (px[:, 3] > 0.5)
    cols = mesh.visual.vertex_colors.copy().astype(np.float32)
    cols[ok, :3] = (1 - blend) * cols[ok, :3] + blend * px[ok, :3] * 255.0
    mesh.visual.vertex_colors = cols.astype(np.uint8)
    return mesh


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