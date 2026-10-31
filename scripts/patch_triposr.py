import pathlib

p = pathlib.Path("third_party/TripoSR/tsr/models/isosurface.py")
src = p.read_text()
if "skimage" in src:
    print("already patched")
else:
    patch = '''import numpy as np
import torch
from skimage import measure

def marching_cubes(level, threshold):
    arr = level.detach().cpu().numpy()
    verts, faces, _, _ = measure.marching_cubes(arr, threshold)
    return (torch.from_numpy(verts.copy()).float(),
            torch.from_numpy(faces.astype(np.int64).copy()))
'''
    p.write_text(src.replace("from torchmcubes import marching_cubes", patch))
    print("patched")