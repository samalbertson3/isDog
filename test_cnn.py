import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold


# Define the input size of the images
input_shape = (224, 224, 3)


def kfold_crossval(model, k_folds=2, num_epochs=1, num_lamby=2):
    dogs_ds, non_dogs_ds = create_training_data()
    # make a vector of lambdas -> interate through lambdas, train model k tims
    lambs = np.logspace(-1, 7, num=num_lamby)
    print("Beginning CrossValidation")
    # Create a list to store the history objects from each fold
    histories = []
    dataset_size = 1000
    # Generate array of indices, shuffle and split them into k parts
    dog_indices = np.arange(dataset_size)
    nd_indices = np.arange(dataset_size)
    np.random.shuffle(dog_indices)
    np.random.shuffle(nd_indices)
    dog_splits = np.array_split(dog_indices, k_folds)
    nd_splits = np.array_split(nd_indices, k_folds)

    # Preprocess the dog images
    dogs_ds = dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
    )

    # Preprocess the non-dog images
    non_dogs_ds = non_dogs_ds.map(
        lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
    )

    out_lambda = []
    out_loss = []
    for lamby in lambs:
        out_lambda.append(lamby)
        # Perform K-fold cross-validation
        for i in range(k_folds):
            # Get the training and validation sets for this fold
            dog_val_indices = dog_splits[i]
            dog_train_indices = np.concatenate(dog_splits[:i] + dog_splits[i + 1 :])

            dog_val_dataset = dogs_ds.take(dog_val_indices)
            dog_train_dataset = dogs_ds.take(dog_train_indices)

            nd_val_indices = nd_splits[i]
            nd_train_indices = np.concatenate(nd_splits[:i] + nd_splits[i + 1 :])

            nd_val_dataset = non_dogs_ds.take(nd_val_indices)
            nd_train_dataset = non_dogs_ds.take(nd_train_indices)

            # Concatenate the dog and non-dog datasets
            val_dataset = dog_val_dataset.concatenate(nd_val_dataset)
            train_dataset = dog_train_dataset.concatenate(nd_train_dataset)

            # Train the model on the training set
            # call train model
            model = train_model(input_shape, lamby, train_dataset)
            # don't need model.fit(train_dataset, epochs=num_epochs)

            # Evaluate the model on the validation set
            val_loss, val_acc = model.evaluate(val_dataset)
            print(
                "Fold {}: Validation Loss = {}, Validation Accuracy = {}".format(
                    i + 1, val_loss, val_acc
                )
            )
        val_losses = []
        val_accs = []
        print(
            "Mean Validation Loss = {}, Mean Validation Accuracy = {}".format(
                np.mean(val_losses), np.mean(val_accs)
            )
        )
        out_loss.append(np.mean(val_losses))
    return (out_lambda, out_loss)


def create_training_data():
    # does first 3 sections of train model function
    # Load the Stanford Dogs dataset
    print("Loading dogs...")
    dogs_ds, dogs_info = tfds.load("stanford_dogs", with_info=True)

    print("Loading non-dogs...")
    # Load the Caltech 101 dataset
    non_dogs_ds, non_dogs_info = tfds.load("caltech101", with_info=True)

    print("Subsetting data...")
    # Subset both datasets
    dogs_ds = dogs_ds.take(1000)
    non_dogs_ds = non_dogs_ds.take(1000)
    return (dogs_ds, non_dogs_ds)


def train_model(input_shape, lamby, training_data):
    # Load the Stanford Dogs dataset
    # dogs_ds, non_dogs_ds = training_data

    print("Finalizing image processing...")

    # Shuffle and batch the dataset
    training_data = training_data.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

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
