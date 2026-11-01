"""python scripts/test_trellis_api.py outputs/full/chair_3_for_3d.png"""
import sys, os, shutil
from gradio_client import Client, handle_file

tok = os.environ.get("HF_TOKEN")
c = Client("trellis-community/TRELLIS", token=tok)

r = c.predict(
    image=handle_file(sys.argv[1]),
    multiimages=[],
    seed=0,
    ss_guidance_strength=7.5,
    ss_sampling_steps=12,
    slat_guidance_strength=3.0,
    slat_sampling_steps=12,
    multiimage_algo="stochastic",
    mesh_simplify=0.95,
    texture_size=1024,
    api_name="/generate_and_extract_glb",
)
print("RESULT:", r)
glb = r[2]   # download_glb filepath
os.makedirs("outputs", exist_ok=True)
shutil.copy(glb, "outputs/trellis_test.glb")
print("saved outputs/trellis_test.glb")