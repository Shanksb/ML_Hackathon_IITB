"""Shared settings. Environment variables override these defaults."""
import os

MOCK_MODE = os.environ.get("SKETCH2SHAPE_MOCK", "0") == "1"      # skip SD, return the sketch
LOW_VRAM = os.environ.get("SKETCH2SHAPE_LOW_VRAM", "0") == "1"   # CPU offload on small GPUs
OUTPUT_DIR = os.environ.get("SKETCH2SHAPE_OUT", "outputs/app")
MC_RESOLUTION = 256   # mesh detail: 128 fast, 256 default, 320 finer
N_CANDIDATES = int(os.environ.get("SKETCH2SHAPE_N_CANDIDATES", "3"))  # how many images to generate and pick the best