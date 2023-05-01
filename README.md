# isDog
Final project: utilities package for identifying and classifying images of dogs (and not dogs). 

# Structure
There are 2 separate convolutional neural networks: one trained on images of both dogs and non-dogs and another trained to identify different breeds of dogs (with no training data that does not contain dogs.)

# Classes

## IsDog

Determines if an image contains a dog or not.

 - train_model()
 - save_model(model_filepath = "saved_models/isDog")
    - model_filepath (str): directory to where the model will be saved
 - predict_dog(img_path, model_path)
    - img_path(str): path to image being predicted. The image must be in .jpg/.jpeg format.
    - model_path(str): path to model

WhichDog and BigDog are both based off of the same classifier.

## WhichDog

Determines the breed of a dog in a given image, assuming there is a dog in the frame.

 - train_model()
 - save_model(model_filepath = "saved_models/whichDog")
    - model_filepath (str): directory to where the model will be saved
 - predict_dog(img_path, model_path = "saved_models/whichDog")
    - img_path (str): path to image being predicted. The image must be in .jpg/.jpeg format.
    - model_path (str): path to model
## BigDog

Assuming the image contains a dog, determine whether the dog is a big dog or a lil dog.

 - train_model()
 - save_model(model_filepath = "saved_models/bigDog")
    - model_filepath (str): directory to where the model will be saved
 - predict_dog(img_path, model_path = "saved_models/bigDog")
    - img_path (str): path to image being predicted. The image must be in .jpg/.jpeg format.
    - model_path (str): path to model  
