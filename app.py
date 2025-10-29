import streamlit as st
import torch
from diffusers import StableDiffusionTurboPipeline
from PIL import Image

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🧠 AI Image Generator (Stable Diffusion Turbo)")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionTurboPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")
    return pipe

pipe = load_pipeline()

st.markdown("Enter a text prompt to generate an image!")

prompt = st.text_input("Image description (in English):")

if st.button("Generate"):
    if not prompt:
        st.warning("⚠️ Please enter a prompt!")
    else:
        with st.spinner("✨ Generating image, please wait..."):
            image = pipe(prompt, guidance_scale=0.0, num_inference_steps=4).images[0]

            st.image(image, caption="✅ Generated Image", use_column_width=True)

            # save image
            img_path = "generated.png"
            image.save(img_path)

            with open(img_path, "rb") as file:
                st.download_button(
                    label="📥 Download Image",
                    data=file,
                    file_name="generated.png",
                    mime="image/png"
                )