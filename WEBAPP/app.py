from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import string
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = load_model("model.h5")

def predict_sign(image_input):
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_input.copy()

    image = cv2.resize(image, (28, 28))

    if np.mean(image) > 127:
        image = 255 - image

    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    labels = list(string.ascii_uppercase)
    labels.remove('J')

    return labels[predicted_class]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json.get("image")
    encoded_data = data_url.split(',')[1]
    img_data = base64.b64decode(encoded_data)
    img_array = np.array(Image.open(BytesIO(img_data)).convert('L'))

    predicted_letter = predict_sign(img_array)
    return jsonify({"letter": predicted_letter})

if __name__ == '__main__':
    app.run(debug=True)
