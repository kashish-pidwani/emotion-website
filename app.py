from flask import Flask, render_template, request
import os
import random

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

emotions = [
    ("Happy 😄", 96),
    ("Sad 😢", 88),
    ("Angry 😠", 91),
    ("Surprise 😲", 94),
    ("Neutral 😐", 89)
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    prediction, confidence = random.choice(emotions)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)