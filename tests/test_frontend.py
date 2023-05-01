import pytest
import front_end as cnn

# Breeds List
Breed_list = [
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


# Create Classifiers
test_isDog = cnn.IsDog()

test_dogClassifier = cnn.DogClassifier()

test_whichDog = cnn.WhichDog()

test_bigDog = cnn.BigDog()

# Train Classifiers

# test_isDog.train_model()
# test_isDog.save_model()

test_dogClassifier.train_model()
test_dogClassifier.save_model()

# Test Image Paths
test_dog = "dog.jpg"
test_notdog = "non-dog.jpg"


# whichDog Tests
def test_correct_input_accepted():
    """Accepts jpg image and gives output."""
    test_whichDog.predict(test_dog)


def test_correct_output_provided():
    """Checks if output provided is in correct format."""
    prediction = test_whichDog.predict(test_dog)
    assert prediction in Breed_list


# bigDog Tests
def test_correct_input_accepted():
    """Accepts jpg image and gives output."""
    test_whichDog.predict(test_dog)


def test_correct_output_provided():
    """Checks if output provided is in correct format."""
    prediction = test_whichDog.predict(test_dog)
    assert prediction in [
        "Big Boy",
        "lil guy",
        "This dog is neither a Big Boy or a lil guy",
    ]
