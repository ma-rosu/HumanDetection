import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def preprocess_image(image, label):
    image = tf.image.resize_with_pad(image, 256, 256)
    image = image / 255.0
    return image, label


data = tf.keras.utils.image_dataset_from_directory('data', shuffle=True)
data = data.map(preprocess_image)

train_size = int(len(data) * .8)
val_size = int(len(data) * .2)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)

AUTOTUNE = tf.data.AUTOTUNE
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.prefetch(buffer_size=AUTOTUNE)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train, epochs=10)

model.save('human_detector.h5')

