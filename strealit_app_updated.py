import os
import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Optional imports that may not be available in every environment
try:
    import torch
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False


# ----------------------
# Helper utilities
# ----------------------

def get_device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # MPS (Apple silicon)
    if hasattr(torch, "has_mps") and torch.has_mps:
        return "mps"
    return "cpu"


def add_watermark(image: Image.Image, watermark_text: str = "AI-Generated") -> Image.Image:
    """Add a small semi-transparent watermark in the lower-right corner."""
    img = image.convert("RGBA")
    txt = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    # choose a font size relative to image size
    fontsize = max(14, img.size[0] // 40)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    text = watermark_text
    textwidth, textheight = draw.textsize(text, font=font)
    # position
    margin = 10
    x = img.size[0] - textwidth - margin
    y = img.size[1] - textheight - margin
    # Draw semi-transparent rectangle behind text for readability
    rect_padding = 6
    draw.rectangle([x-rect_padding, y-rect_padding, x+textwidth+rect_padding, y+textheight+rect_padding], fill=(0,0,0,100))
    draw.text((x, y), text, font=font, fill=(255,255,255,200))
    combined = Image.alpha_composite(img, txt)
    return combined.convert("RGB")


# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")
st.title("🎨 AI-Powered Text-to-Image Generator (Stable Diffusion)")
st.write("Note: This app requires the `diffusers` library and a Hugging Face token with model access.")

with st.expander("Requirements / Quick notes", expanded=False):
    st.markdown(
        """
- Install requirements: `pip install torch diffusers accelerate transformers safetensors --upgrade` (and optionally `xformers` for speed).
- You must provide a Hugging Face token with access to the model (e.g. `runwayml/stable-diffusion-v1-5`) via `st.secrets['hf_token']` or environment variable `HUGGINGFACE_HUB_TOKEN`.
- GPU is strongly recommended for speed. On CPU, generation will be very slow.
"""
    )

# Sidebar: configuration
with st.sidebar:
    st.header("Generation Settings")
    prompt = st.text_area("Enter image prompt", height=110, placeholder="A futuristic city at sunset with flying cars...")
    style = st.selectbox("Style (informal guidance)", ["photorealistic", "artistic", "cartoon", "cinematic", "detailed"], index=0)
    num_images = st.slider("Number of images", 1, 4, 1)
    num_steps = st.slider("Inference steps", 20, 100, 50, step=5)
    guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, step=0.5)
    height = st.selectbox("Height", [512, 640, 768], index=0)
    width = st.selectbox("Width", [512, 640, 768], index=0)
    export_formats = st.multiselect("Export formats", ["PNG", "JPEG"], default=["PNG"])
    add_watermark_opt = st.checkbox("Add watermark (AI-Generated)", value=True)
    save_metadata = st.checkbox("Save metadata (JSON)", value=True)
    model_id = st.text_input("Hugging Face model id", value="runwayml/stable-diffusion-v1-5")

    st.markdown("---")
    st.subheader("Auth / Device")
    hf_token = st.text_input("Hugging Face token (or set HUGGINGFACE_HUB_TOKEN env)", type="password")
    device_str = st.selectbox("Device (auto-detect recommended)", ["auto", "cuda", "cpu", "mps"], index=0)
    run_button = st.button("🚀 Generate")

# Early checks
if run_button:
    if not prompt:
        st.error("Please provide a prompt before generating.")
    else:
        # Check diffusers availability
        if not DIFFUSERS_AVAILABLE:
            st.error("The 'diffusers' or 'torch' library is not available in this environment.\nInstall requirements and restart the app.")
        else:
            # Determine device
            if device_str == "auto":
                device = get_device()
            else:
                device = device_str

            if not hf_token:
                # Try environment
                hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

            if not hf_token:
                st.warning("No Hugging Face token provided. If the model requires authentication you will see an error.\nSet `st.secrets['hf_token']` or the HUGGINGFACE_HUB_TOKEN environment variable.")

            # Load model
            progress_text = st.empty()
            progress_text.info(f"Loading model {model_id} to device={device} ... This may take a while.")

            try:
                # Prefer low_cpu_mem_usage and torch_dtype when possible
                kwargs = {}
                if torch and device == "cuda":
                    kwargs.update({"torch_dtype": torch.float16})

                # The `use_auth_token` argument was removed in recent diffusers; instead, pass the token via the `token` arg or login before.
                token_arg = hf_token if hf_token else True

                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    revision="fp16" if device == "cuda" else None,
                    safety_checker=None,
                    **kwargs
                )

                # Move to device
                pipe = pipe.to(device)

            except Exception as e:
                st.exception(e)
                st.error("Failed to load the model. Check that you have the model ID correct, internet access, and a valid Hugging Face token if required.")
                raise st.stop()

            progress_text.success("Model loaded — starting generation...")

            # Image generation loop
            cols = st.columns(num_images)
            generated_images = []
            metadata = {
                "prompt": prompt,
                "style": style,
                "num_images": num_images,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "model_id": model_id,
                "device": device,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }

            for i in range(num_images):
                with st.spinner(f"Generating image {i+1}/{num_images}..."):
                    # Some pipelines accept width/height through .__call__
                    try:
                        out = pipe(
                            prompt,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width
                        )
                        image = out.images[0]
                    except TypeError:
                        # Fallback if the pipeline call signature differs
                        out = pipe(prompt)
                        image = out.images[0]

                    if add_watermark_opt:
                        image = add_watermark(image, watermark_text="AI-Generated")

                    generated_images.append(image)
                    cols[i % len(cols)].image(image, caption=f"Image {i+1}")

            st.success(f"Generated {len(generated_images)} image(s)")

            # Show download buttons
            for idx, img in enumerate(generated_images, start=1):
                buf = io.BytesIO()
                fmt = "PNG" if "PNG" in export_formats else "JPEG"
                img.save(buf, format=fmt)
                byte_im = buf.getvalue()
                st.download_button(label=f"Download Image {idx} ({fmt})", data=byte_im, file_name=f"image_{idx}.{fmt.lower()}")

            # Save metadata if requested
            if save_metadata:
                metadata["images"] = [f"image_{i+1}.{('png' if 'PNG' in export_formats else 'jpg')}" for i in range(len(generated_images))]
                metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
                st.download_button(label="Download metadata (JSON)", data=metadata_bytes, file_name="metadata.json")


# Footer / instructions
st.markdown("---")
st.markdown(
    """
### Notes
- If you get errors while loading the model, make sure you have network access and a valid Hugging Face token with access to the chosen model.
- On machines without GPUs, generation will be very slow — consider using smaller models or hosting the generation elsewhere (Colab/GCP/AWS).

**If you want, I can also provide a Colab-ready notebook or a CPU-only simplified version.**
"""
)
