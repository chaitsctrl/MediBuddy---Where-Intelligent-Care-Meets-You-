from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)

# Define class labels
CLASS_LABELS = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

# Load model
model = tf.keras.models.load_model("lung_disease_model.h5")

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        img = Image.open(file)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        predicted_label = CLASS_LABELS[predicted_class]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5004)


# import requests

# url = "http://127.0.0.1:5004/predict"
# files = {'image': open('test_image.jpg', 'rb')}
# response = requests.post(url, files=files)
# print(response.json())