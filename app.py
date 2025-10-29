import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🖼️ AI Image Generator with Diffusers")

@st.cache_resource
def load_pipeline():
    return AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )

pipe = load_pipeline()

st.markdown("Enter a text prompt to generate an image!")

prompt = st.text_input("Image description (in English):")

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating, please wait..."):
            image = pipe(prompt, guidance_scale=0.0).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
            st.success("Image generated successfully ✅")

            # Saving option
            img_path = "generated_image.png"
            image.save(img_path)

            with open(img_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
