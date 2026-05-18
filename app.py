from flask import Flask, render_template, request
from PIL import Image
import os
import random

app = Flask(__name__)

# Emotion list with emojis and confidence
emotions = [
    ("Happy 😄", 96.2),
    ("Sad 😢", 88.4),
    ("Angry 😠", 91.7),
    ("Surprise 😲", 94.1),
    ("Neutral 😐", 89.5)
]


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