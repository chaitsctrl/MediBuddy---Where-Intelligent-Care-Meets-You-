import streamlit as st
import numpy as np
import joblib
import requests
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC

# Load the model (SVM and VGG16)
@st.cache_resource
def load_model():
    # Load the VGG16 base model (without top)
    vgg16_base = VGG16(weights='vgg16_weights.h5', include_top=False, input_shape=(224, 224, 3))
    
    # Load your SVM model
    svm_vgg16 = joblib.load('svm_model_vgg16.joblib')
    return vgg16_base, svm_vgg16

# Feature extraction function
def extract_features(model, img):
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess image for VGG16
    features = model.predict(img)  # Extract features using VGG16
    features = features.flatten()  # Flatten the features for SVM input
    return features

# Prediction function
def predict_label(model, svm, img):
    features = extract_features(model, img)  # Extract features
    prediction = svm.predict([features])  # Get prediction (which is an integer index)
    labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    return labels[int(prediction[0])]

# Function to get response from LLM model
def get_llm_response(prediction):
    url = "https://rohitashva-healthcare-chatbot.hf.space"
    payload = {"query": prediction}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    return "Error communicating with LLM."

# Streamlit interface
st.title('Dementia Severity Prediction')
st.write('Upload an MRI scan image to predict the dementia severity.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read image using Keras preprocessing utility (resize automatically)
    img = np.array(image.load_img(uploaded_file, target_size=(224, 224)))
    
    # Load model
    vgg16_base, svm_vgg16 = load_model()

    # Make prediction
    prediction = predict_label(vgg16_base, svm_vgg16, img)

    # Get response from LLM
    llm_response = get_llm_response(prediction)

    # Show the result
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {prediction}")
    st.write(f"LLM Response: {llm_response}")