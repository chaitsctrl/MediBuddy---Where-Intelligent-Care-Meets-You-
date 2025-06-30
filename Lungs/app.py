import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

# Define class labels
CLASS_LABELS = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_disease_model.h5")  # Update filename if needed
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Match your model's input size
    image = np.array(image)  # Convert to NumPy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Apply DenseNet preprocessing
    return image

# Streamlit UI
st.title("Lung Disease Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Get the predicted class index
    predicted_class = np.argmax(prediction)
    predicted_label = CLASS_LABELS[predicted_class]
    
    # Display result
    st.write(f"### Prediction: **{predicted_label}**")