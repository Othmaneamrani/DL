import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.utils import img_to_array
from PIL import Image
from tensorflow.python.keras.models import load_model

# Try loading the model with custom_objects
try:
    model = load_model("model/fruits_model.h5", compile=False)
except:
    # Alternative loading method if the first fails
    model = tf.keras.models.load_model(
        "model/fruits_model.h5",
        custom_objects=None,
        compile=False
    )

# Rest of your code remains the same
class_names = ['apple', 'banana', 'orange']


def predict(image):
    # Convertir l'image en RGB (au cas où elle serait RGBA ou en mode autre que RGB)
    image = image.convert('RGB')
    img = image.resize((32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[class_index], confidence


st.title("CNN model ~ classification des fruits")

uploaded_file = st.file_uploader("Telecharger l'image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image telechargé", use_container_width=True)

    with st.spinner("Analyse en cours"):
        class_name, confidence = predict(image)
        st.success(f"Résultat : {class_name} ({confidence * 100:.2f}%)")