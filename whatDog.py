"""The classifier for whatDog and bigBoy."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# Define the input size of the images
BATCHSIZE = 32
IMGSIZE = (224, 224)

# Define list of breeds
BREEDSLIST = [
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


def preprocess_data(example):
    """Preprocesses data to be used in training of the model."""
    # resize the image and fetch its label
    image = tf.image.resize(example["image"], IMGSIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = example["label"]
    return image, label


def train_model(dataset, filepath="saved_model/my_model"):
    """Performs training of CNN model."""
    # Define the training and validation data
    train = dataset.take(5000)
    val = dataset.skip(5000)
    # Build model archetecture
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(224, 224, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(BREEDSLIST), activation="softmax"),
        ]
    )
    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(train, epochs=2, validation_data=val)

    # Save the model to the filepath specified
    model.save(filepath)
    return model


def buildWhichDog(model_filepath="saved_model/whichDog"):
    """Builds the CNN model and returns model."""
    # print("Loading dogs...")
    # Load the dataset
    dataset, info = tfds.load("stanford_dogs", with_info=True, split="train[:60%]")
    # print("Dogs Loaded!")
    # shuffle the dataset so we don't just get a subset of dog breeds
    dataset = dataset.shuffle(1024)
    # Apply preprocessing to the dataset
    dataset = dataset.map(preprocess_data)
    # Batch the dataset
    dataset = dataset.batch(BATCHSIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    model = train_model(dataset, model_filepath)
    return model


def predict_dog(img_path, model_filepath="saved_model/whichDog"):
    """Uses saved whichDog model to predict dog breed of image."""
    # Load the model
    model = tf.keras.models.load_model(model_filepath)

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
    predicted_breed = BREEDSLIST[predicted_class]
    # Print the predicted breed
    print(predicted_breed)
    return predicted_breed


def predict_biglil(img_path, model_filepath="saved_model/whichDog"):
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
    breed = predict_dog(img_path, model_filepath)
    if breed in big_boys:
        dogType = "Big Boy"
    elif breed in lil_guys:
        dogType = "lil guy"
    else:
        dogType = "This dog is neither a Big Boy or a lil guy"
    return dogType


def main() -> None:
    """Do the thing."""
    buildWhichDog(model_filepath="saved_model/whichDog")
    predict_dog("dog.jpg")
    predict_biglil("dog.jpg")
    predict_dog("dog2.jpg")
    predict_biglil("dog2.jpg")
    predict_dog("non-dog.jpg")
    predict_dog("non-dog2.jpg")


if __name__ == "__main__":
    main()
