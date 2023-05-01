import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

# https://pythonspot.com/polymorphism/
# "/Users/ellieanderson/Downloads/golden-retriever.jpg"


class IsDog:
    """Determine if an image is of a dog."""

    def __init__(self, input_shape: tuple = (224, 224, 3)) -> None:
        self.input_shape = input_shape

    @property
    def model(self):
        print("Building model...")
        # Create the model
        model = models.Sequential()

        # Add the convolutional layers
        model.add(
            layers.Conv2D(
                32,
                (3, 3),
                input_shape=self.input_shape,
            )
        )
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(
            layers.Conv2D(
                64,
                (3, 3),
            )
        )
        model.add(layers.MaxPooling2D((2, 2)))

        # Add the flatten layer
        model.add(layers.Flatten())

        # Add the dense layers
        model.add(layers.Dense(2048))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    def train_model(self, suppress_print = False):
        # Load the Stanford Dogs dataset
        if !suppress_print:
            print("Loading dogs...")
        dogs_ds, val_dogs_ds, test_dogs_ds = tfds.load(
            "stanford_dogs",
            with_info=False,
            split=["train[:70%]", "train[70%:]", "test"],
        )

        if !suppress_print:
            print("Loading non-dogs...")
        # Load the Caltech 101 dataset
        non_dogs_ds, val_non_dogs_ds, test_non_dogs_ds = tfds.load(
            "caltech101", with_info=False, split=["train[:70%]", "train[70%:]", "test"]
        )

        if !suppress_print:
            print("Subsetting data...")
        # Subset both datasets
        dogs_ds = dogs_ds.take(1000)
        non_dogs_ds = non_dogs_ds.take(1000)
        val_dogs_ds = dogs_ds.take(1000)
        val_non_dogs_ds = non_dogs_ds.take(1000)
        test_dogs_ds = dogs_ds.take(1000)
        test_non_dogs_ds = non_dogs_ds.take(1000)

        if !suppress_print:
            print("Processing dogs...")
        # Preprocess the dog images
        dogs_ds = dogs_ds.map(
            lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
        )
        val_dogs_ds = val_dogs_ds.map(
            lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
        )
        test_dogs_ds = test_dogs_ds.map(
            lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(1))
        )

        if !suppress_print:
            print("Processing non-dogs...")
        # Preprocess the non-dog images
        non_dogs_ds = non_dogs_ds.filter(
            lambda x: x["label"] != 37
        )  # exclude the "dog" class from Caltech 101
        val_non_dogs_ds = val_non_dogs_ds.filter(lambda x: x["label"] != 37)
        test_non_dogs_ds = test_non_dogs_ds.filter(lambda x: x["label"] != 37)
        non_dogs_ds = non_dogs_ds.map(
            lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
        )
        if !suppress_print:
            print("Finalizing image processing...")
        val_non_dogs_ds = val_non_dogs_ds.map(
            lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
        )
        test_non_dogs_ds = test_non_dogs_ds.map(
            lambda x: (tf.image.resize(x["image"], (224, 224)), tf.constant(0))
        )

        print("Finalizing image processing...")
        # Concatenate the dog and non-dog datasets
        dataset = dogs_ds.concatenate(non_dogs_ds)
        val_dataset = val_dogs_ds.concatenate(val_non_dogs_ds)
        test_dataset = test_dogs_ds.concatenate(test_non_dogs_ds)

        # Shuffle and batch the dataset
        dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
        if print_on:
            print("Training model...")
        # Compile the model
        model = self.model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        history = model.fit(dataset, epochs=3, validation_data=val_dataset)
        results = model.evaluate(test_dataset)

        if !suppress_print:
            print("Done!")
            print("Done!")
            print("Test loss: ", results[0])
            print("Test accuracy: ", results[1])
        return model

    def save_model(self, filepath: str = "saved_models/isDog") -> None:
        """Save model to disk"""
        self.model.save(filepath)

    def preprocess_image(self, image_path):
        # Load the image using Pillow
        img = Image.open(image_path)
        # Resize the image
        img = img.resize(self.input_shape[:2])
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Expand the dimensions of the image to match the input shape of the model
        img_array = np.expand_dims(img_array, axis=0)
        # Scale the pixel values to be between 0 and 1
        img_array = img_array / 255.0
        return img_array

    def predict_dog(self, image_path, model_filepath="saved_model/isDog"):
        # Preprocess the input image
        img_array = self.preprocess_image(image_path)
        # Use the model to make a prediction
        model = tf.keras.models.load_model(model_filepath)
        predictions = model.predict(img_array)
        # Get the predicted class (1 for dog, 0 for non-dog)
        raw_prediction = predictions[0]
        predicted_class = int(np.round(raw_prediction))
        # Return the predicted class
        if predicted_class == 1:
            print(f"The image is a dog. ({round(raw_prediction[0]*100, 2)}% confident)")
        else:
            print(
                f"The image is not a dog. ({round(raw_prediction[1]*100, 2)}% confident)"
            )
        # return predicted_class, raw_prediction

    # model = train_model(self.input_shape)


class DogClassifier:
    def __init__(self, batchsize=32, imgsize=(224, 224)) -> None:
        # Define the input size of the images
        self.batchsize = batchsize
        self.imgsize = imgsize

        # Define list of breeds
        self.breedslist = [
            "chihuahua",
            "japanese spaniel",
            "maltese dog",
            "pekinese",
            "shih-tzu",
            "blenheim spaniel",
            "papillon",
            "toy terrier",
            "rhodesian ridgeback",
            "afghan hound",
            "basset",
            "beagle",
            "bloodhound",
            "bluetick",
            "black-and-tan coonhound",
            "walker hound",
            "english foxhound",
            "redbone",
            "borzoi",
            "irish wolfhound",
            "italian greyhound",
            "whippet",
            "ibizan hound",
            "norwegian elkhound",
            "otterhound",
            "saluki",
            "scottish deerhound",
            "weimaraner",
            "staffordshire bullterrier",
            "american staffordshire terrier",
            "bedlington terrier",
            "border terrier",
            "kerry blue terrier",
            "irish terrier",
            "norfolk terrier",
            "norwich terrier",
            "yorkshire terrier",
            "wire-haired fox terrier",
            "lakeland terrier",
            "sealyham terrier",
            "airedale",
            "cairn",
            "australian terrier",
            "dandie dinmont",
            "boston bull",
            "miniature schnauzer",
            "giant schnauzer",
            "standard schnauzer",
            "scotch terrier",
            "tibetan terrier",
            "silky terrier",
            "soft-coated wheaten terrier",
            "west highland white terrier",
            "lhasa",
            "flat-coated retriever",
            "curly-coated retriever",
            "golden retriever",
            "labrador retriever",
            "chesapeake bay retriever",
            "german short-haired pointer",
            "vizsla",
            "english setter",
            "irish setter",
            "gordon setter",
            "brittany spaniel",
            "clumber",
            "english springer",
            "welsh springer spaniel",
            "cocker spaniel",
            "sussex spaniel",
            "irish water spaniel",
            "kuvasz",
            "schipperke",
            "groenendael",
            "malinois",
            "briard",
            "kelpie",
            "komondor",
            "old english sheepdog",
            "shetland sheepdog",
            "collie",
            "border collie",
            "bouvier des flandres",
            "rottweiler",
            "german shepherd",
            "doberman",
            "miniature pinscher",
            "greater swiss mountain dog",
            "bernese mountain dog",
            "appenzeller",
            "entlebucher",
            "boxer",
            "bull mastiff",
            "tibetan mastiff",
            "french bulldog",
            "great dane",
            "saint bernard",
            "eskimo dog",
            "malamute",
            "siberian husky",
            "affenpinscher",
            "basenji",
            "pug",
            "leonberg",
            "newfoundland",
            "great pyrenees",
            "samoyed",
            "pomeranian",
            "chow",
            "keeshond",
            "brabancon griffon",
            "pembroke",
            "cardigan",
            "toy poodle",
            "miniature poodle",
            "standard poodle",
            "mexican hairless",
            "dingo",
            "dhole",
            "african hunting dog",
        ]

    def preprocess_data(self, example):
        """Preprocesses data to be used in training of the model."""
        # resize the image and fetch its label
        image = tf.image.resize(example["image"], self.imgsize)
        image = tf.cast(image, tf.float32) / 255.0
        label = example["label"]
        return image, label

    @property
    def model(self):
        """Builds the CNN model and returns model."""
        # Build model archetecture
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2048),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.breedslist), activation="relu"),
            ]
        )
        return model

    def train_model(self):
        """Performs training of CNN model."""
        # Load the dataset
        dataset, info = tfds.load("stanford_dogs", with_info=True, split="train[:60%]")
        # shuffle the dataset so we don't just get a subset of dog breeds
        dataset = dataset.shuffle(1024)
        # Apply preprocessing to the dataset
        dataset = dataset.map(self.preprocess_data)
        # Batch the dataset
        dataset = dataset.batch(self.batchsize)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Define the training and validation data
        train = dataset.take(5000)
        val = dataset.skip(5000)

        model = self.model

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train the model
        history = model.fit(train, epochs=2, validation_data=val)

        return model

    def save_model(self, filepath="saved_model/whichDog") -> None:
        """Save the model to the filepath specified"""
        model = self.train_model()
        model.save(filepath)


class WhichDog(DogClassifier):
    def __init__(self, batchsize=32, imgsize=(224, 224)) -> None:
        super().__init__(batchsize, imgsize)

    def predict_dog(self, img_path, model_filepath="saved_model/whichDog"):
        """Uses saved whichDog model to predict dog breed of image."""
        # Load the model

        # if os.path.isfile(model_filepath):
        model = tf.keras.models.load_model(model_filepath)
        # else:
        #     super().train_model()
        #     super().save_model()
        #     model = tf.keras.models.load_model(model_filepath)

        # Load the image to specified target size
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

        # Process the image
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Get predicted breed
        predicted_breed = self.breedslist[predicted_class]
        # Print the predicted breed
        print(predicted_breed)
        return predicted_breed


class BigDog(WhichDog):
    def __init__(self, batchsize=32, imgsize=(224, 224)) -> None:
        super().__init__(batchsize, imgsize)

    def predict(self, img_path):
        big_boys = [
            "Rhodesian Ridgeback",
            "Afghan Hound",
            "Basset",
            "Bloodhound",
            "Bluetick",
            "Black-and-tan Coonhound",
            "Walker Hound",
            "Borzoi",
            "Irish Wolfhound",
            "Weimaraner",
            "Staffordshire Bullterrier",
            "American Staffordshire Terrier",
            "Airedale",
            "Chesapeake Bay Retriever",
            "German Short-haired Pointer",
            "Vizsla",
            "Kuvasz",
            "Bouvier des Flandres",
            "Rottweiler",
            "German Shepherd",
            "Doberman",
            "Bull Mastiff",
            "Tibetan Mastiff",
            "Great Dane",
            "Saint Bernard",
            "Eskimo Dog",
            "Malamute",
            "Siberian Husky",
            "Great Pyrenees",
            "Samoyed",
            "Chow",
        ]
        lil_guys = [
            "Chihuahua",
            "Japanese Spaniel",
            "Maltese Dog",
            "Pekinese",
            "Shih-tzu",
            "Blenheim Spaniel",
            "Papillon",
            "Toy Terrier",
            "Italian Greyhound",
            "Whippet",
            "Ibizan Hound",
            "Norwegian Elkhound",
            "Otterhound",
            "Scottish Deerhound",
            "Bedlington Terrier",
            "Border Terrier",
            "Kerry Blue Terrier",
            "Irish Terrier",
            "Norfolk Terrier",
            "Norwich Terrier",
            "Yorkshire Terrier",
            "Wire-haired Fox Terrier",
            "Lakeland Terrier",
            "Sealyham Terrier",
            "Cairn",
            "Australian Terrier",
            "Dandie Dinmont",
            "Boston Bull",
            "Miniature Schnauzer",
            "Giant Schnauzer",
            "Standard Schnauzer",
            "Scotch Terrier",
            "Tibetan Terrier",
            "Silky Terrier",
            "Soft-coated Wheaten Terrier",
            "West Highland White Terrier",
            "Lhasa",
            "Flat-coated Retriever",
            "Curly-coated Retriever",
            "Golden Retriever",
            "Labrador Retriever",
            "Brittany Spaniel",
            "Clumber",
            "English Springer",
            "Welsh Springer Spaniel",
            "Cocker Spaniel",
            "Sussex Spaniel",
            "Irish Water Spaniel",
            "Schipperke",
            "Groenendael",
            "Malinois",
            "Briard",
            "Kelpie",
            "Komondor",
            "Old English Sheepdog",
            "Shetland Sheepdog",
            "Collie",
            "Border Collie",
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Appenzeller",
            "Entlebucher",
            "Boxer",
            "Pug",
            "Leonberg",
            "Newfoundland",
            "Toy Poodle",
            "Miniature Poodle",
            "Standard Poodle",
            "Mexican Hairless",
            "Dingo",
            "Dhole",
            "African Hunting Dog",
            "Basenji",
            "Keeshond",
            "Brabancon Griffon",
            "Pembroke",
            "Cardigan",
        ]
        breed = self.predict_dog(img_path)
        if breed in big_boys:
            dogType = "Big Boy"
        elif breed in lil_guys:
            dogType = "lil guy"
        else:
            dogType = "This dog is neither a Big Boy or a lil guy"
        return dogType
