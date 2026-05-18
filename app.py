from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Labels
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Emojis
emotion_emojis = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

@app.route('/', methods=['GET', 'POST'])
def home():

    result = None
    emoji = ""
    confidence = 0
    image_data = None

    if request.method == 'POST':

        file = request.files['image']

        image = Image.open(file)

        # Save uploaded image for display
        buffered = BytesIO()
        image.save(buffered, format="PNG")

        image_data = base64.b64encode(buffered.getvalue()).decode()

        # Model preprocessing
        gray_image = image.convert('L')
        gray_image = gray_image.resize((48, 48))

        img_array = np.array(gray_image)
        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        prediction = model.predict(img_array)

        pred_index = int(np.argmax(prediction))

        confidence = round(float(np.max(prediction)) * 100, 2)

        result = class_names[pred_index]

        emoji = emotion_emojis.get(result, "")

    return render_template(
        'index.html',
        result=result,
        emoji=emoji,
        confidence=confidence,
        image_data=image_data
    )

if __name__ == '__main__':
    app.run(debug=True)