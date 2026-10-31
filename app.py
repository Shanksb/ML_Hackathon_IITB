"""Sketch2Shape AI: Gradio UI.   python app.py   (SHARE=1 on Colab)"""
import logging
import os

import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline import PipelineError, run_pipeline, warmup

logging.basicConfig(level=logging.INFO)


def to_pil(value):
    """Gradio gives a dict (ImageEditor), an array, or a PIL image."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite")
        if value is None:
            return None
    if isinstance(value, np.ndarray):
        value = Image.fromarray(value)
    return value


def generate(canvas, upload, prompt, seed, progress=gr.Progress()):
    sketch = to_pil(upload) or to_pil(canvas)   # an uploaded image wins over the canvas
    if sketch is None:
        return None, None, None, "Please draw a sketch or upload one first.", ""
    try:
        r = run_pipeline(sketch, prompt, seed=int(seed), progress=progress)
    except PipelineError as e:
        return None, None, None, f"Could not generate: {e}", ""
    except Exception as e:  # last resort, keeps the UI alive
        return None, None, None, f"Unexpected error: {e}", ""
    info = (f"Detected: {r['label']}  \n"
            f"Time: {r['timings']['total']} s  \n"
            f"Mesh: {r['report']['vertices']} vertices, {r['report']['faces']} faces")
    return r["glb_path"], r["glb_path"], r["generated_image"], "", info


with gr.Blocks(title="Sketch2Shape AI") as demo:
    gr.Markdown("# Sketch2Shape AI\nDraw or upload a sketch, get a textured 3D model. "
                "*The first run is slower while models load.*")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                canvas = gr.ImageEditor(label="Sketch canvas", type="numpy",
                                        canvas_size=(512, 512), brush=gr.Brush(color=["000000"], default_color=["000000"], default_size=6))
            with gr.Tab("Upload"):
                upload = gr.Image(label="Upload a sketch", type="pil")
            prompt = gr.Textbox(label="Optional prompt",
                                placeholder="e.g. red metal, wooden, glossy")
            seed = gr.Number(label="Seed", value=42, precision=0)
            btn = gr.Button("Generate 3D model", variant="primary")
            error = gr.Markdown()
        with gr.Column():
            viewer = gr.Model3D(label="3D preview", height=420)
            download = gr.File(label="Download .glb")
            generated = gr.Image(label="Generated reference image", type="pil")
            info = gr.Markdown()

    btn.click(generate, [canvas, upload, prompt, seed],
              [viewer, download, generated, error, info])

if __name__ == "__main__":
    if os.environ.get("WARMUP", "1") == "1":
        warmup()
    demo.launch(share=os.environ.get("SHARE", "0") == "1")