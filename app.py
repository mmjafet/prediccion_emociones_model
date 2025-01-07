from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('emotion_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = Image.open(request.files['file'].stream)
    img = img.resize((48, 48))
    img = img.convert('L')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    emotion = np.argmax(prediction)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    result = emotions[emotion]

    return jsonify({'emotion': result})

if __name__ == '__main__':
    app.run(debug=True)