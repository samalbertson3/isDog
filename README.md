# isDog
Final project: utilities package for identifying and classifying images of dogs (and not dogs). 

# Structure
There are 2 separate convolutional neural networks: one trained on images of both dogs and non-dogs and another trained to identify different breeds of dogs (with no training data that does not contain dogs.)

# Classes

## IsDog

Determines if an image contains a dog or not.

 - **train_model**(optional args: suppress_print, test_mode)
   - This function trains the model and must be called before using the predict_dog function. No arguments are needed to run the function as intended for a regular user. 
   - This function has two optional arguments that can be used. 
      - *test_mode*: set to "True" to use fewer epochs and less training data to run the function more quickly. Warning: this mode is for testing purposes only and will have extremely low prediction accuracy. 
      - *suppress_print*: Set to "True" to suppress print statements that tell the user what step of the training process the model is on and providing test loss and test accuracy after training has completed.
 - **save_model**(model_filepath = "saved_models/isDog")
    - model_filepath (str): directory to where the model will be saved
    - This function must be run before using the predict_dog function.
 - **predict_dog**(img_path, model_filepath = "saved_models/isDog")
    - img_path(str): path to image being predicted. The image must be in .jpg/.jpeg format.
    - model_path(str): path to model. this argument is optional if you did not change the filepath from the default value.

**Note: WhichDog and BigDog are both based off of the same classifier. The classifier only needs to be trained and saved once to use both predict functions.**
## WhichDog

Determines the breed of a dog in a given image, assuming there is a dog in the frame.

 - **train_model**(optional arguments: test_mode)
   - This function trains the model and must be called before using the predict_dog function. No arguments are needed to run the function as intended for a regular user. 
   - This function has an optional argument that can be used. 
      - *test_mode*: set to "True" to use fewer epochs and less training data to run the function more quickly. Warning: this mode is for testing purposes only and will have extremely low prediction accuracy. 

 - **save_model**(model_filepath = "saved_models/whichDog")
    - model_filepath (str): directory to where the model will be saved
 - **predict_dog**(img_path, model_path = "saved_models/whichDog")
    - img_path (str): path to image being predicted. The image must be in .jpg/.jpeg format.
    - model_path (str): path to model

## BigDog

Assuming the image contains a dog, determine whether the dog is a Big Boy or a lil guy. Please note that this determination is based purely on vibes and has no bearing on the gender of the dog or non-dog image subject. 

 - **train_model**(optional arguments: test_mode)
   - This function trains the model and must be called before using the predict_dog function. No arguments are needed to run the function as intended for a regular user. 
      - This function has an optional argument that can be used. 
         - *test_mode*: set to "True" to use fewer epochs and less training data to run the function more quickly. Warning: this mode is for testing purposes only and will have extremely low prediction accuracy. 
 - **save_model**(model_filepath = "saved_models/whichDog")
    - model_filepath (str): directory to where the model will be saved
 - **predict_dog**(img_path, model_path = "saved_models/whichDog")
    - img_path (str): path to image being predicted. The image must be in .jpg/.jpeg format.
    - model_path (str): path to model  

# Testing Module

## Setup

To make sure that tests will run as expected, please have the following folders set up: src, tests

Additionally, if you have run the classifiers once already and they have trained and been saved, there should be a folder titled **"saved_models"** containing two sub-folders "isDog" and "whichDog" that contained the trained models.  

These folders should be set up correctly when you clone the repository from github. 
### src
In the src folder should be the file "front_end.py". 

### tests
In the test folder should be the file "test_frontend.py" which contains the tests and a jpg file called "test_dog.jpg", which is the test imaged used in the functions. Please use pytest to run the tests by calling "pytest tests/" in the terminal from the directory that you cloned the repository into. 
