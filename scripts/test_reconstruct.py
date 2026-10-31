import sys, time
sys.path.insert(0, ".")
import torch
from PIL import Image
from src.reconstruct import reconstruct, _load_model

path = sys.argv[1] if len(sys.argv) > 1 else "third_party/TripoSR/examples/chair.png"
res = int(sys.argv[2]) if len(sys.argv) > 2 else 256

# Test input must be RGBA with the background removed
img = Image.open(path)
if img.mode != "RGBA":
    from rembg import remove
    img = remove(img.convert("RGB"))

t = time.time(); _load_model(); print(f"model load: {time.time()-t:.1f}s")

torch.cuda.reset_peak_memory_stats()
t = time.time()
mesh = reconstruct(img, mc_resolution=res)
print(f"reconstruct @res {res}: {time.time()-t:.1f}s")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
print(f"bounds extents: {mesh.extents}")

mesh.export("outputs/test_mesh.glb")
print("saved outputs/test_mesh.glb")