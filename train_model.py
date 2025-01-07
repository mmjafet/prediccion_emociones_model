import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos FER-2013
data = pd.read_csv('fer2013.csv')

# Preprocesar los datos
def preprocess_data(data):
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = face / 255.0
        faces.append(face)
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = to_categorical(data['emotion'], num_classes=7)
    return faces, emotions

train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']

X_train, y_train = preprocess_data(train_data)
X_val, y_val = preprocess_data(val_data)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Guardar el modelo
model.save('emotion_model.h5')