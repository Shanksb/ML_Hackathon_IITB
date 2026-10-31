import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from src.reconstruct import reconstruct, ReconstructionError

def expect_error(name, img):
    try:
        reconstruct(img)
        print(f"[FAIL] {name}: no error raised")
    except ReconstructionError as e:
        print(f"[ OK ] {name}: {e}")
    except Exception as e:
        print(f"[FAIL] {name}: wrong exception type {type(e).__name__}: {e}")

# 1. Wrong mode (RGB instead of RGBA)
expect_error("RGB input", Image.new("RGB", (256, 256), "white"))

# 2. Fully transparent (nothing in the image)
expect_error("blank RGBA", Image.new("RGBA", (256, 256), (0, 0, 0, 0)))

# 3. Tiny speck (under 1% coverage)
speck = np.zeros((256, 256, 4), dtype=np.uint8)
speck[100:104, 100:104] = (255, 0, 0, 255)
expect_error("tiny speck", Image.fromarray(speck, "RGBA"))

# 4. Not an image at all
expect_error("not an image", "hello")

# 5. A valid input should succeed
chair = Image.open("third_party/TripoSR/examples/chair.png")
if chair.mode != "RGBA":
    from rembg import remove
    chair = remove(chair.convert("RGB"))
mesh = reconstruct(chair)
print(f"[ OK ] valid chair: {len(mesh.vertices)} vertices, "
      f"resolution {mesh.metadata['resolution_used']}, {mesh.metadata['seconds']}s, "
      f"extents {np.round(mesh.extents, 2)}")