import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
model = load_model('emotion_model.h5')

executor = ThreadPoolExecutor(max_workers=2)  # Ajusta el número de trabajadores según tus necesidades

def process_image(img):
    img = img.resize((48, 48))
    img = img.convert('L')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    return img_array

def predict_emotion(img_array):
    prediction = model.predict(img_array)
    emotion = np.argmax(prediction)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[emotion]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = Image.open(request.files['file'].stream)
    
    # Procesar la imagen en un hilo separado
    future = executor.submit(process_image, img)
    img_array = future.result()
    
    # Predecir la emoción en un hilo separado
    future = executor.submit(predict_emotion, img_array)
    result = future.result()

    return jsonify({'emotion': result})

if __name__ == '__main__':
    app.run(debug=True)
