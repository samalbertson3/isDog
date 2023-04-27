import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

# Define the input size of the images
input_shape = (224, 224, 3)


# test
def train_model(input_shape, lamby):
    # Load the Stanford Dogs dataset
    print("Loading dogs...")
    dogs_ds, dogs_info = tfds.load(
        "stanford_dogs", with_info=True, split="train[:100%]"
    )

    print("Loading non-dogs...")
    # Load the Caltech 101 dataset
    non_dogs_ds, non_dogs_info = tfds.load(
        "caltech101", with_info=True, split="train[:100%]"
    )

    print("Processing dogs...")
    # Preprocess the dog images
    dogs_ds = dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
    )

    print("Processing non-dogs...")
    # Preprocess the non-dog images
    non_dogs_ds = non_dogs_ds.filter(
        lambda x: x["label"] != 37
    )  # exclude the "dog" class from Caltech 101
    non_dogs_ds = non_dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
    )

    # Shuffle and batch the dataset
    dogs_ds = dogs_ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    non_dogs_ds = non_dogs_ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

    print("Subsetting data...")
    # Subset both datasets
    dogs_ds = dogs_ds.take(50)
    non_dogs_ds = non_dogs_ds.take(300)

    print("Finalizing image processing...")
    # Concatenate the dog and non-dog datasets
    dataset = dogs_ds.concatenate(non_dogs_ds)

    print("Building model...")
    # Create the model
    model = models.Sequential()

    # Add the convolutional layers
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=regularizers.L1(lamby),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(
        layers.Conv2D(
            64, (3, 3), activation="relu", kernel_regularizer=regularizers.L1(lamby)
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
        layers.Dense(512, activation="relu", kernel_regularizer=regularizers.L1(lamby))
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L1(lamby))
    )

    print("Training model...")
    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(dataset, epochs=1)

    print("Done!")
    return model


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


model = train_model(input_shape, 10)
print(predict_image_class("C:/Users/Sam/Desktop/dog.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/dog2.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/dog3.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/non-dog.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/non-dog2.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/non-dog3.jpg", model))
