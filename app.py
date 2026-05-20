import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to predict the tumor type.")

# -----------------------------
# Class Names
# -----------------------------
classes = [
    "Glioma",
    "Meningioma",
    "Pituitary Tumor",
    "No Tumor"
]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model("Brain_tumer_cnn.keras")
    return model

model = load_model()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction Function
# -----------------------------
def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# -----------------------------
# Main Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    probabilities = prediction[0]

    predicted_class = np.argmax(probabilities)

    predicted_label = classes[predicted_class]

    confidence = probabilities[predicted_class] * 100

    # -----------------------------
    # Show Prediction
    # -----------------------------
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Class: {predicted_label}"
    )

    st.write(f"Confidence: {confidence:.2f}%")

    # -----------------------------
    # Show All Class Probabilities
    # -----------------------------
    st.subheader("Class Probabilities")

    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probabilities[i] * 100:.2f}%")

    # -----------------------------
    # Probability Graph
    # -----------------------------
    st.subheader("Prediction Graph")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(classes, probabilities * 100)

    ax.set_ylabel("Probability (%)")

    ax.set_xlabel("Classes")

    ax.set_title("Class Prediction Probabilities")

    st.pyplot(fig)