import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained VGG16 model
model = VGG16(weights="imagenet")

# Load the Stanford Dogs dataset
dataset = tfds.load("stanford_dogs", split="test", shuffle_files=False)
dataset = dataset.take(1000)

function process_dataset(data):
    # Preprocess and make predictions on the images
    proc_dataset = []
    for example in data:
        image = example["image"]
        image = tf.image.resize(image, (224, 224))
        image = preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        proc_dataset.append(image)

    proc_dataset = tf.data.Dataset.from_tensor_slices(proc_dataset)

    return(proc_dataset)

for example in proc_dataset:
    print(".")
