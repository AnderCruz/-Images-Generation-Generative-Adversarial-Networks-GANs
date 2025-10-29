import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st

def plot_images(imagens):
    plt.figure(figsize=(20, 20))
    for i in range(len(imagens)):
        ax = plt.subplot(1, len(imagens), i + 1)
        plt.imshow(imagens[i])
        plt.axis("off")
    st.pyplot(plt.gcf())  # Exibe a figura no Streamlit

def main():
    prompt = st.text_input('Digite a descrição da imagem em inglês')  # Corrigido o nome da função
    estilo = '''dark fantasy art,
    high quality, highly detailed, elegant, sharp focus,
    concept art, character concepts, digital painting, mystery, adventure'''
    
    keras.mixed_precision.set_global_policy("mixed_float16")
    modelo = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    
    if prompt:
        imagens = modelo.text_to_image(prompt + estilo, batch_size=1)  # batch_size especificado
        plot_images(imagens)  # Exibe as imagens

if __name__ == "__main__":
    main()
