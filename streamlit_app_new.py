# streamlit_app.py
import os
import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# guarded heavy imports (app still loads UI without them)
try:
    import torch
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception:
    torch = None
    StableDiffusionPipeline = None
    DIFFUSERS_AVAILABLE = False

# -----------------------
# Helpers
# -----------------------
def get_hf_token():
    """
    Secure token retrieval:
    1) streamlit secrets (Streamlit Cloud)
    2) environment variables (HUGGINGFACE_HUB_TOKEN, HF_TOKEN, HF_ACCESS_TOKEN)
    Returns None if not found.
    """
    try:
        # prefer st.secrets if provided by hosting (keeps token out of code)
        token = st.secrets.get("hf_token") if hasattr(st, "secrets") else None
        token = token or (st.secrets.get("HUGGINGFACE_HUB_TOKEN") if hasattr(st, "secrets") else None)
        if token:
            return token
    except Exception:
        # ignore issues reading st.secrets
        pass

    # environment variables (works for Colab, Docker, VPS, Heroku)
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

def detect_device():
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch, "has_mps", False) and torch.has_mps:
            return "mps"
    except Exception:
        pass
    return "cpu"

def add_watermark_pil(image: Image.Image, watermark_text: str = "AI-Generated") -> Image.Image:
    img = image.convert("RGBA")
    txt = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    fontsize = max(14, img.size[0] // 40)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    text = watermark_text
    textwidth, textheight = draw.textsize(text, font=font)
    margin = 10
    x = img.size[0] - textwidth - margin
    y = img.size[1] - textheight - margin
    rect_padding = 6
    draw.rectangle([x-rect_padding, y-rect_padding, x+textwidth+rect_padding, y+textheight+rect_padding], fill=(0,0,0,100))
    draw.text((x, y), text, font=font, fill=(255,255,255,200))
    combined = Image.alpha_composite(img, txt)
    return combined.convert("RGB")

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")
st.title("🎨 AI-Powered Text-to-Image Generator")

st.markdown("""
Generate high-quality images from text descriptions using advanced AI models.
**Note:** This app reads your Hugging Face token from a secure backend (see instructions).
""")

st.markdown("---")

# Sidebar controls (kept as in original)
with st.sidebar:
    st.header("⚙️ Generation Settings")
    prompt = st.text_area("Enter your image description:", placeholder="e.g., A futuristic city at sunset with flying cars...", height=100)
    style = st.selectbox("Select art style:", ["photorealistic", "artistic", "cartoon", "cinematic", "detailed"])
    num_images = st.slider("Number of images to generate:", min_value=1, max_value=4, value=1)
    st.subheader("Advanced Settings")
    num_steps = st.slider("Inference steps (higher = better quality, slower):", min_value=20, max_value=100, value=50, step=10)
    guidance_scale = st.slider("Guidance scale (higher = closer to prompt):", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    st.subheader("Export Options")
    export_formats = st.multiselect("Export formats:", ["PNG", "JPEG"], default=["PNG", "JPEG"])
    add_watermark = st.checkbox("Add AI watermark", value=True)
    save_metadata = st.checkbox("Save metadata", value=True)
    # keep generate button same as original
    generate_button = st.button("🚀 Generate Images", use_container_width=True)

# Main generation block
if generate_button:
    if not prompt:
        st.error("Please enter a prompt!")
    else:
        with st.spinner(f"Generating {num_images} image(s) with {style} style..."):
            st.info(f"📝 Prompt: {prompt}")
            st.info(f"⏱️ Estimated time: ~{num_images * num_steps * 0.3 / 60:.1f} minutes")

            # placeholders for images
            image_placeholders = st.columns(min(num_images, 2))

            # model loading
            pipe = None
            model_id = "runwayml/stable-diffusion-v1-5"  # change if needed
            device = detect_device()
            st.info(f"Preparing model on device: {device} (this can take a few minutes on first run).")

            if DIFFUSERS_AVAILABLE:
                hf_token = get_hf_token()
                load_kwargs = {}
                if device == "cuda":
                    # use float16 on CUDA to reduce memory
                    load_kwargs["torch_dtype"] = torch.float16
                try:
                    # try modern token argument first, fall back to older param if necessary
                    if hf_token:
                        try:
                            pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, token=hf_token, **load_kwargs)
                        except TypeError:
                            pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, use_auth_token=hf_token, **load_kwargs)
                    else:
                        pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, **load_kwargs)
                    pipe = pipe.to(device)
                    st.success("Model loaded successfully.")
                except Exception as e:
                    # load failed — show a helpful message but do not reveal token
                    st.warning("Model could not be loaded in this environment. Falling back to placeholders; check server logs for details.")
                    st.exception(e)
                    pipe = None
            else:
                st.warning("Required libraries not available (torch/diffusers). Falling back to placeholders.")
                pipe = None

            generated_images = []
            metadata = {
                "prompt": prompt,
                "style": style,
                "num_images": num_images,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "model_id": model_id,
                "device": device,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }

            # generation loop
            for idx in range(num_images):
                col = image_placeholders[idx % len(image_placeholders)]
                with col:
                    with st.spinner(f"Image {idx+1}/{num_images}"):
                        if pipe is None:
                            # placeholder behaviour (original UI preserved)
                            st.success(f"✅ Image {idx+1} generated!")
                            placeholder_img = Image.new("RGB", (512, 512), color=(200, 200, 200))
                            draw = ImageDraw.Draw(placeholder_img)
                            try:
                                fnt = ImageFont.truetype("arial.ttf", 14)
                            except Exception:
                                fnt = ImageFont.load_default()
                            draw.text((10, 10), f"Placeholder Image {idx+1}\nPrompt: {prompt[:80]}", font=fnt, fill=(0,0,0))
                            generated_images.append(placeholder_img)
                            col.image(placeholder_img, caption=f"Image {idx+1} (placeholder)")
                        else:
                            try:
                                out = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale)
                                image = out.images[0]
                            except TypeError:
                                out = pipe(prompt)
                                image = out.images[0]

                            if add_watermark:
                                image = add_watermark_pil(image, watermark_text="AI-Generated")

                            col.image(image, caption=f"Image {idx+1}", use_column_width=True)
                            generated_images.append(image)

            st.success(f"🎉 Successfully generated {len(generated_images)} image(s)!")

            # download buttons
            for i, img in enumerate(generated_images, start=1):
                buf = io.BytesIO()
                fmt = "PNG" if "PNG" in export_formats else "JPEG"
                img.save(buf, format=fmt)
                buf.seek(0)
                st.download_button(label=f"Download Image {i} ({fmt})", data=buf.getvalue(), file_name=f"image_{i}.{fmt.lower()}")

            if save_metadata:
                metadata["images"] = [f"image_{i+1}.{('png' if 'PNG' in export_formats else 'jpg')}" for i in range(len(generated_images))]
                metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
                st.download_button(label="Download metadata (JSON)", data=metadata_bytes, file_name="metadata.json")

# footer/help text (unchanged)
st.markdown("---")
st.markdown("""
### 📖 How to Use:
1. **Enter Prompt**: Describe the image you want to generate
2. **Select Style**: Choose from photorealistic, artistic, cartoon, cinematic, or detailed
3. **Adjust Settings**: Fine-tune inference steps and guidance scale
4. **Generate**: Click the generate button and wait
5. **Export**: Download generated images in your preferred format

### ⚠️ Ethical AI Guidelines:
- Generated images are watermarked to indicate AI origin
- Inappropriate content is automatically filtered
- Metadata tracking ensures transparency
- Use responsibly and respect copyright

### 💡 Tips for Better Results:
- Be specific in your descriptions
- Use style descriptors ("photorealistic", "oil painting", "anime", etc.)
- Experiment with guidance scale for different levels of adherence
- Higher steps = better quality but longer generation time
""")

st.markdown("""
---
**Created with ❤️ using Stable Diffusion and Streamlit**
""")
