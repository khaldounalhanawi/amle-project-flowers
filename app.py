"""Classify a flower photo with a model saved in step 5 of the project.

Run it from the repository folder: uv run streamlit run app.py

The model must take the raw photos (pixel values 0 to 255) and do its own preprocessing, for example with
mobilenet_v2.preprocess_input inside the model, and it must output one probability per flower.
"""

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

CLASSES = ["rose", "sunflower", "tulip"]  # the folder names, in alphabetical order, as in the training data


@st.cache_resource  # load each model only once, not again every time the page changes
def load_model(path):
    return keras.models.load_model(path)


def prepare_photo(photo_bytes, model):
    """Decode and resize a photo with TensorFlow, as image_dataset_from_directory did for the training photos."""
    image = tf.io.decode_image(photo_bytes, channels=3, expand_animations=False)
    height, width = model.input_shape[1:3]
    return tf.image.resize(image, (height, width))  # bilinear, the default of image_dataset_from_directory


st.title("Flower Classifier")

model_files = sorted(Path("models").glob("*.keras"))
if not model_files:
    st.error("No model found in `models/`. Save a model there in step 5 of the notebook first.")
    st.stop()

model_path = st.selectbox("Model", model_files, format_func=lambda path: path.name)
model = load_model(str(model_path))

photo = st.file_uploader("Upload a photo of a rose, a sunflower or a tulip", type=["jpg", "jpeg", "png"])
if photo is not None:
    st.image(photo, width=300)

    pixels = prepare_photo(photo.getvalue(), model)
    probabilities = model.predict(pixels[np.newaxis], verbose=0)[0]  # a batch of one photo

    st.subheader(f"Prediction: {CLASSES[probabilities.argmax()]} ({probabilities.max():.0%})")
    st.bar_chart(pd.Series(probabilities, index=CLASSES, name="probability"))
