"""One-command setup for the 3D reconstruction stage (P3).

Usage (from the project root):
    python scripts/download_weights.py

What it does:
    1. Clones TripoSR into third_party/ if missing
    2. Applies the scikit-image patch (replaces torchmcubes)
    3. Downloads the TripoSR weights from Hugging Face
P1 can add their own model downloads at the bottom.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRIPOSR_DIR = ROOT / "third_party" / "TripoSR"


def setup_triposr():
    if TRIPOSR_DIR.exists():
        print("[1/3] TripoSR already cloned")
    else:
        print("[1/3] Cloning TripoSR ...")
        TRIPOSR_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/VAST-AI-Research/TripoSR", str(TRIPOSR_DIR)],
            check=True,
        )

    print("[2/3] Applying scikit-image patch ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "patch_triposr.py")],
                   check=True, cwd=ROOT)

    print("[3/3] Downloading TripoSR weights (about 1.7 GB, first time only) ...")
    from huggingface_hub import hf_hub_download
    for fname in ["config.yaml", "model.ckpt"]:
        path = hf_hub_download("stabilityai/TripoSR", fname)
        print("      ok:", path)


if __name__ == "__main__":
    setup_triposr()
    # --- P1: add your model downloads below, e.g. ---
    # from huggingface_hub import snapshot_download
    # snapshot_download("lllyasviel/control_v11p_sd15_scribble")
    print("Done.")