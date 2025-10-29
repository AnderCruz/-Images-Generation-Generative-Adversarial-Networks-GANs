import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🖼️ AI Image Generator - CPU Mode (Streamlit Cloud)")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")
    return pipe

pipe = load_pipeline()

st.markdown("Enter a text prompt to generate an image!")

prompt = st.text_input("Image description (in English):")
steps = st.slider("Inference Steps", 10, 40, 20)

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating image... This may take ~20s ⏳"):
            image = pipe(
                prompt,
                guidance_scale=7.5,
                num_inference_steps=steps
            ).images[0]

            st.image(image, caption="Generated Image", use_column_width=True)
            st.success("Image generated successfully ✅")

            img_path = "generated_image.png"
            image.save(img_path)

            with open(img_path, "rb") as file:
                st.download_button(
                    label="📥 Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
