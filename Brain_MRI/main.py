from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define class labels
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# Image preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_array = preprocess_image(file.read())
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': CLASS_LABELS[predicted_class],
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


# url = "http://127.0.0.1:5001/predict"
# files = {'image': open('test_image.jpg', 'rb')}
# response = requests.post(url, files=files)
# print(response.json())