from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("model.h5", compile=False)

# Emotion labels with emojis
emotion_data = {
    0: ("Angry 😠"),
    1: ("Disgust 🤢"),
    2: ("Fear 😨"),
    3: ("Happy 😄"),
    4: ("Sad 😢"),
    5: ("Surprise 😲"),
    6: ("Neutral 😐")
}

# Upload folder
UPLOAD_FOLDER = "static/uploads"

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Open and preprocess image
    image = Image.open(filepath).convert('L')
    image = image.resize((48, 48))

    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    # Predict
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    # Get emotion + confidence
    emotion = emotion_data[predicted_class]

    confidence = round(float(np.max(prediction)) * 100, 2)

    return render_template(
        'index.html',
        prediction=emotion,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)