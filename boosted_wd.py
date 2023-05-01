from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

IMG_SIZE = 224
# Define the input size of the images
input_shape = (224, 224, 3)
# Load the dataset
print("loading dogs")
dataset, info = tfds.load('stanford_dogs', with_info=True)
print("loaded dogs")

def preprocess(features):
    image = features['image']
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = features['label']
    return {'image': image, 'label': label}


# Preprocess the data
train_data = dataset['train'].map(preprocess)
test_data = dataset['test'].map(preprocess)

# Split the data into features and labels
train_features = np.array([x['image'] for x in train_data])
train_labels = np.array([x['label'] for x in train_data])
test_features = np.array([x['image'] for x in test_data])
test_labels = np.array([x['label'] for x in test_data])
print("training first model")

# build model 
model = models.Sequential()

# Add the convolutional layers
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=input_shape,
        kernel_regularizer=regularizers.L1(0.01),
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(
    layers.Conv2D(
        64, (3, 3), activation="relu", kernel_regularizer=regularizers.L1(0.01)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))

# Add the flatten layer
model.add(layers.Flatten())

# Add the dense layers
model.add(
    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.L1(0.01))
)
model.add(layers.Dropout(0.5))
model.add(
    layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L1(0.01))
)

print("Training model...")
# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the TensorFlow model on the training set
model.fit(train_features, train_labels)

print("training second model")
# Use the trained TensorFlow model as a weak classifier in AdaBoost
ada_boost_model = AdaBoostClassifier(estimator=model, n_estimators=50, algorithm = 'SAMME')

# Train the AdaBoost model on the training set
ada_boost_model.fit(train_features, train_labels)

# Evaluate the performance of the AdaBoost model on the testing set
test_pred = ada_boost_model.predict(test_features)
accuracy = accuracy_score(test_labels, test_pred)
print("Accuracy:", accuracy)

def predict_image_class(image_path, model):
    # Preprocess the input image
    img_array = preprocess_image(image_path)
    # Use the model to make a prediction
    predictions = model.predict(img_array)
    # Get the predicted class (1 for dog, 0 for non-dog)
    raw_prediction = predictions[0]
    predicted_class = int(np.round(raw_prediction))
    # Return the predicted class
    return predicted_class, raw_prediction

def preprocess_image(image_path):
    # Load the image using Pillow
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(input_shape[:2])
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Expand the dimensions of the image to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the pixel values to be between 0 and 1
    img_array = img_array / 255.0
    return img_array

print(predict_image_class("akita.jpg", ada_boost_model))