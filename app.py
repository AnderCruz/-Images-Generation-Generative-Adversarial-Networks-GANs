import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    st.pyplot(plt.gcf())  # Displays the figure in Streamlit

def main():
    prompt = st.text_input('Enter the image description in English')  # Corrected function name
    style = '''dark fantasy art,
    high quality, highly detailed, elegant, sharp focus,
    concept art, character concepts, digital painting, mystery, adventure'''
    
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    
    if prompt:
        images = model.text_to_image(prompt + style, batch_size=1)  # Specified batch_size
        plot_images(images)  # Displays the images

if __name__ == "__main__":
    main()
