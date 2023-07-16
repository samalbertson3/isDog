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
dataset = dataset.take(1)  # Take one example from the test split

# Preprocess and make predictions on the image
for example in dataset:
    img = example["image"]
    img = tf.image.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(np.copy(img))

    # Make predictions on the image
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

    # Display the predictions
    print("Predictions:")
    for _, label, probability in decoded_preds:
        print(f"{label}: {probability * 100:.2f}%")
