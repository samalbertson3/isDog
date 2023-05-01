import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import matplotlib as plt

# Define the input size of the images
input_shape = (224, 224, 3)


# test
def train_model(input_shape):
    # Load the Stanford Dogs dataset
    print("Loading dogs...")
    dogs_ds, test_dogs_ds = tfds.load(
        "stanford_dogs", with_info=False, split=["train[:80%]", "test"]
    )

    print("Loading non-dogs...")
    # Load the Caltech 101 dataset
    non_dogs_ds, test_non_dogs_ds = tfds.load(
        "caltech101", with_info=False, split=["train[:80%]", "test"]
    )

    print("Processing dogs...")
    # Preprocess the dog images
    dogs_ds = dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
    )
    dogs_ds = dogs_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    test_dogs_ds = test_dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
    )
    test_dogs_ds = test_dogs_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    print("Processing non-dogs...")
    # Preprocess the non-dog images
    non_dogs_ds = non_dogs_ds.filter(
        lambda x: x["label"] != 37
    )  # exclude the "dog" class from Caltech 101
    non_dogs_ds = non_dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
    )
    non_dogs_ds = non_dogs_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    test_non_dogs_ds = test_non_dogs_ds.filter(
        lambda x: x["label"] != 37
    )  # exclude the "dog" class from Caltech 101
    test_non_dogs_ds = test_non_dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
    )
    test_non_dogs_ds = test_non_dogs_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    )

    # Shuffle and batch the dataset
    dogs_ds = dogs_ds.shuffle(1024).prefetch(tf.data.AUTOTUNE)
    non_dogs_ds = non_dogs_ds.shuffle(1024).prefetch(tf.data.AUTOTUNE)
    test_dogs_ds = test_dogs_ds.shuffle(1024).prefetch(tf.data.AUTOTUNE)
    test_non_dogs_ds = test_non_dogs_ds.shuffle(1024).prefetch(tf.data.AUTOTUNE)

    print("Subsetting data...")
    # Subset both datasets
    dogs_ds = dogs_ds.take(1000)
    non_dogs_ds = non_dogs_ds.take(1000)
    test_dogs_ds = test_dogs_ds.take(1000)
    test_non_dogs_ds = test_non_dogs_ds.take(1000)

    print("Finalizing image processing...")
    # Concatenate the dog and non-dog datasets
    dataset = dogs_ds.concatenate(non_dogs_ds)
    dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dogs_ds.concatenate(test_non_dogs_ds)
    test_dataset = test_dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

    # for image, label in dataset.take(10):
    #     # Convert the image to a numpy array
    #     image = image[0].numpy()
    #     # Show the image with the label as the title
    #     plt.imshow(image[0])
    #     # plt.title(label[0])
    #     plt.show()

    print("Building model...")
    # Create the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    print("Training model...")
    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(dataset, epochs=5, validation_data=test_dataset)

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


model = train_model(input_shape)
print(predict_image_class("C:/Users/Sam/Desktop/dog.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/dog2.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/dog3.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/non-dog.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/non-dog2.jpg", model))
print(predict_image_class("C:/Users/Sam/Desktop/non-dog3.jpg", model))

# for image_data in sample:
#     label = image_data["label"].numpy()
#     print("Label:", label)
