import tensorflow as tf
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained VGG16 model
model = VGG16(weights="imagenet")

# Load and preprocess the test image
img_path = "test.jpg"  # Replace with the path to your test image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions on the image
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

# Display the predictions
print("Predictions:")
for _, label, probability in decoded_preds:
    print(f"{label}: {probability*100:.2f}%")
