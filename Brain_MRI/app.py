import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Class labels from your notebook
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(img):
    """Preprocess the image to match the model's input requirements."""
    img = img.resize((224, 224))  # Resize to model input size
    img = img.convert("RGB")  # Ensure RGB mode
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Brain Tumor Detection with Deep Learning")
st.write("Upload an MRI scan (JPG, PNG) to check for a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Display result
    st.subheader("🩺 Prediction Results")
    st.write(f"**Predicted Class:** {CLASS_LABELS[predicted_class]}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    # Provide interpretation
    if predicted_class == 2:
        st.success("✅ No Brain Tumor Detected.")
    else:
        st.error(f"🚨 Brain Tumor Detected: **{CLASS_LABELS[predicted_class]}**. Consult a doctor.")

