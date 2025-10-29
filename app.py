import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Configuração da página
st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🖼️ AI Image Generator with Diffusers")

# Carregando pipeline com cache
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32
    )
    # Mover para GPU se disponível
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

pipe = load_pipeline()

st.markdown("Enter a text prompt to generate an image!")

# Input de prompt
prompt = st.text_input("Image description (in English):")

# Botão para gerar imagem
if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating, please wait..."):
            # Gerar imagem
            image = pipe(prompt).images[0]

            # Mostrar imagem
            st.image(image, caption="Generated Image", use_column_width=True)
            st.success("Image generated successfully ✅")

            # Salvar e oferecer download
            img_path = "generated_image.png"
            image.save(img_path)
            with open(img_path, "rb") as file:
                st.download_button(
                    label="📥 Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )

