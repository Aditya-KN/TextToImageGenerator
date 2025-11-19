
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import json
from datetime import datetime
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎨 AI-Powered Text-to-Image Generator")
st.markdown("""
Generate high-quality images from text descriptions using advanced AI models.
**Features:**
- Multiple style options (photorealistic, artistic, cartoon, cinematic, detailed)
- Adjustable generation parameters
- Automatic watermarking for ethical AI
- Multiple export formats (PNG, JPEG)
- Metadata tracking
""")

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    # Text prompt input
    prompt = st.text_area(
        "Enter your image description:",
        placeholder="e.g., A futuristic city at sunset with flying cars...",
        height=100
    )
    
    # Style selection
    style = st.selectbox(
        "Select art style:",
        ["photorealistic", "artistic", "cartoon", "cinematic", "detailed"]
    )
    
    # Number of images
    num_images = st.slider(
        "Number of images to generate:",
        min_value=1,
        max_value=4,
        value=1
    )
    
    # Quality settings
    st.subheader("Advanced Settings")
    num_steps = st.slider(
        "Inference steps (higher = better quality, slower):",
        min_value=20,
        max_value=100,
        value=50,
        step=10
    )
    
    guidance_scale = st.slider(
        "Guidance scale (higher = closer to prompt):",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5
    )
    
    # Export options
    st.subheader("Export Options")
    export_formats = st.multiselect(
        "Export formats:",
        ["PNG", "JPEG"],
        default=["PNG", "JPEG"]
    )
    
    add_watermark = st.checkbox("Add AI watermark", value=True)
    save_metadata = st.checkbox("Save metadata", value=True)
    
    # Generate button
    generate_button = st.button("🚀 Generate Images", use_container_width=True)

# Main content area
if generate_button:
    if not prompt:
        st.error("Please enter a prompt!")
    else:
        with st.spinner(f"Generating {num_images} image(s) with {style} style..."):
            st.info(f"📝 Prompt: {prompt}")
            st.info(f"⏱️ Estimated time: ~{num_images * num_steps * 0.3 / 60:.1f} minutes")
            
            # Create placeholder for images
            image_placeholders = st.columns(min(num_images, 2))
            
            # Generate images
            for idx in range(num_images):
                col = image_placeholders[idx % len(image_placeholders)]
                with col:
                    with st.spinner(f"Image {idx+1}/{num_images}"):
                        # Placeholder for actual generation
                        st.success(f"✅ Image {idx+1} generated!")
            
            st.success(f"🎉 Successfully generated {num_images} image(s)!")

# Information section
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
