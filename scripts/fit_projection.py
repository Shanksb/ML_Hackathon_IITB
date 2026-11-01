"""Fit the image->mesh projection and save a recoloured test mesh.
    python scripts/fit_projection.py samples/outputs/chair_ref.png"""
import itertools, sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import reconstruct as R
from src.preprocess import prepare_for_3d
from tsr.utils import resize_foreground


def fit(verts, alpha):
    H, W = alpha.shape
    ys, xs = np.where(alpha)
    bb = np.array([xs.min(), xs.max(), ys.min(), ys.max()], float)
    best = (-1, None)
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            p = verts[:, perm] * np.array(sg)
            u, v = p[:, 0], p[:, 1]
            s = min((bb[1] - bb[0]) / (np.ptp(u) + 1e-8), (bb[3] - bb[2]) / (np.ptp(v) + 1e-8))
            px = ((u - u.mean()) * s + (bb[0] + bb[1]) / 2).round().astype(int)
            py = ((-(v - v.mean())) * s + (bb[2] + bb[3]) / 2).round().astype(int)
            m = np.zeros((H, W), np.uint8)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            m[py[ok], px[ok]] = 1
            m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            iou = (m.astype(bool) & alpha).sum() / max(1, (m.astype(bool) | alpha).sum())
            if iou > best[0]:
                best = (iou, (perm, sg, s, u.mean(), v.mean(), bb))
    return best


img = Image.open(sys.argv[1])
rgba = img if img.mode == "RGBA" else prepare_for_3d(img)
rgba = resize_foreground(rgba, 0.85)
arr = np.array(rgba)
alpha = arr[..., 3] > 127

mesh = R._run_once(rgba, 256, 0.85, bake=False)      # native frame, before _orient
iou, (perm, sg, s, um, vm, bb) = fit(np.asarray(mesh.vertices), alpha)
print(f"best silhouette IoU = {iou:.3f}  perm={perm} signs={sg}")

# try both depth directions; keep the one whose colours agree with TripoSR's own
H, W = alpha.shape
base = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
best_err, best_cols = 1e9, None
for dsign in (1, -1):
    p = np.asarray(mesh.vertices)[:, perm] * np.array(sg)
    px = ((p[:, 0] - um) * s + (bb[0] + bb[1]) / 2).round().astype(int).clip(0, W - 1)
    py = ((-(p[:, 1] - vm)) * s + (bb[2] + bb[3]) / 2).round().astype(int).clip(0, H - 1)
    d = p[:, 2] * dsign                       # larger = closer to camera
    zbuf = np.full((H, W), -1e9)
    np.maximum.at(zbuf, (py, px), d)
    vis = d >= zbuf[py, px] - 0.02 * np.ptp(d)
    col = arr[py, px, :3].astype(float)
    err = np.abs(col[vis] - base[vis]).mean()
    print(f"depth sign {dsign}: visible={vis.mean():.2f} colour error={err:.1f}")
    if err < best_err:
        best_err = err
        new = base.copy(); new[vis] = 0.15 * base[vis] + 0.85 * col[vis]
        best_cols = new
mesh.visual.vertex_colors = np.clip(best_cols, 0, 255).astype(np.uint8)
mesh.apply_transform(__import__("trimesh").transformations.rotation_matrix(0, [1, 0, 0]))
Path("outputs").mkdir(exist_ok=True)
mesh.export("outputs/recolour_test.glb")
print("saved outputs/recolour_test.glb  (not oriented; just check the colours)")