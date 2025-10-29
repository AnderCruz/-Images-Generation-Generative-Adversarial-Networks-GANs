import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🖼️ AI Image Generator")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")  # se não tiver GPU
    return pipe

pipe = load_pipeline()

prompt = st.text_input("Image description (in English):")

if st.button("Generate"):
    if not prompt:
        st.warning("⚠️ Please enter a prompt!")
    else:
        with st.spinner("Generating image..."):
            image = pipe(prompt, guidance_scale=7.5).images[0]
            st.image(image, use_column_width=True)
            image.save("generated.png")
            with open("generated.png", "rb") as f:
                st.download_button("📥 Download Image", f, "generated.png", "image/png")
