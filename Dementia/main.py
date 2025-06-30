from flask import Flask, request, jsonify
import numpy as np
import joblib
import requests
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Load models
vgg16_base = VGG16(weights='vgg16_weights.h5', include_top=False, input_shape=(224, 224, 3))
svm_model = joblib.load('svm_model_vgg16.joblib')

# Class labels
labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# Feature extraction
def extract_features(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = vgg16_base.predict(img_array)
    return features.flatten()

# LLM response
def get_llm_response(prediction):
    url = "https://rohitashva-healthcare-chatbot.hf.space"
    payload = {"query": prediction}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "No response received.")
        return "Error from LLM API."
    except:
        return "Failed to connect to LLM server."

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Extract features and predict
    features = extract_features(img_array)
    prediction_idx = svm_model.predict([features])[0]
    predicted_label = labels[int(prediction_idx)]

    # Get LLM response
    llm_response = get_llm_response(predicted_label)

    return jsonify({
        'prediction': predicted_label,
        'llm_response': llm_response
    })

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)

# url = "http://127.0.0.1:5002/predict"
# files = {'image': open('test_image.jpg', 'rb')}
# response = requests.post(url, files=files)
# print(response.json())