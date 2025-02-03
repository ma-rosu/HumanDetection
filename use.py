import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

print(tf.keras.__version__)

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error: {e}")
        return None


def predict_image(model, image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return None
        prediction = model.predict(img)[0][0]
        return prediction
    except Exception as e:
        print(f"Error: {e}")
        return None


def detect_humans(model_path, image_folder):
    model = load_model(model_path)
    if model is None:
        return

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            prediction = predict_image(model, image_path)

            if prediction is None:
                continue
            if prediction > 0.5:
                print(f"Image {filename}: Human detected")
            else:
                print(f"Image {filename}: Human not detected")


if __name__ == "__main__":
    model_path = 'human_detector.h5'
    image_folder = 'evaluate'
    detect_humans(model_path, image_folder)