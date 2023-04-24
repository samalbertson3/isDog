import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold


# Define the input size of the images
input_shape = (224, 224, 3)


def kfold_crossval(k_folds=5, dataset, num_epochs = 5):
    # Create a list to store the history objects from each fold
    histories = []
    dataset_size = len(dataset)
    # Generate array of indices, shuffle and split them into k parts
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    splits = np.array_split(indices, k_folds)
    # Perform K-fold cross-validation
    for i in range(k_folds):
        # Get the training and validation sets for this fold
        val_indices = splits[i]
        train_indices = np.concatenate(splits[:i] + splits[i + 1 :])

        val_dataset = dataset.take(val_indices)
        train_dataset = dataset.take(train_indices)

        # Train the model on the training set
        model.fit(train_dataset, epochs=5)

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(val_dataset)
        print(
            "Fold {}: Validation Loss = {}, Validation Accuracy = {}".format(
                i + 1, val_loss, val_acc
            )
        )
        val_losses = []
        val_accs = []

    for i in range(k_folds):
        val_indices = splits[i]
        train_indices = np.concatenate(splits[:i] + splits[i + 1 :])

        val_dataset = dataset.take(val_indices)
        train_dataset = dataset.take(train_indices)

        # Train the model on the training set
        model.fit(train_dataset, epochs=num_epochs)

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(val_dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    print(
        "Mean Validation Loss = {}, Mean Validation Accuracy = {}".format(
            np.mean(val_losses), np.mean(val_accs)
        )
    )


def train_model(input_shape):
    # Load the Stanford Dogs dataset
    print("Loading dogs...")
    dogs_ds, dogs_info = tfds.load("stanford_dogs", with_info=True, split="train[:60%]")

    print("Loading non-dogs...")
    # Load the Caltech 101 dataset
    non_dogs_ds, non_dogs_info = tfds.load(
        "caltech101", with_info=True, split="train[:60%]"
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
    
    #added crossval step to building model 
    kfold_crossval(k_folds=5, dataset, num_epochs = 5)

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

    # Train the model
    history = model.fit(dataset, epochs=5)

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
# print(predict_image_class("C:/Users/Sam/Desktop/dog.jpg", model))
# print(predict_image_class("C:/Users/Sam/Desktop/dog2.jpg", model))
# print(predict_image_class("C:/Users/Sam/Desktop/dog3.jpg", model))
# print(predict_image_class("C:/Users/Sam/Desktop/non-dog.jpg", model))
# print(predict_image_class("C:/Users/Sam/Desktop/non-dog2.jpg", model))
# print(predict_image_class("C:/Users/Sam/Desktop/non-dog3.jpg", model))
