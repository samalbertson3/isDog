import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds


def train_model():
    # Load the Stanford Dogs dataset
    print("Loading dogs...")
    dogs_ds, dogs_info = tfds.load("stanford_dogs", with_info=True, split="train[:80%]")

    print("Loading non-dogs...")
    # Load the Caltech 101 dataset
    non_dogs_ds, non_dogs_info = tfds.load(
        "caltech101", with_info=True, split="train[:80%]"
    )

    print("Subsetting data...")
    # Subset both datasets
    dogs_ds = dogs_ds.take(1000)
    non_dogs_ds = non_dogs_ds.take(1000)

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

    print("Finalizing image processing...")
    # Concatenate the dog and non-dog datasets
    dataset = dogs_ds.concatenate(non_dogs_ds)

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

    # Define the input size of the images
    input_shape = (224, 224, 3)

    print("Building model...")
    # Create the model
    model = models.Sequential()

    # Add the convolutional layers
    model.add(layers.Conv2D(3, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add the flatten layer
    model.add(layers.Flatten())

    # Add the dense layers
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    print("Training model...")
    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(dataset, epochs=10)

    print("Done!")
    return None
