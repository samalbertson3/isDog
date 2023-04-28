# isDog
Final project: utilities package for identifying and classifying images of dogs (and not dogs). 
Task breakdown: 
    Updated task breakdown as of 26APR2023

# Functions 
    *Updated as of 26APR2023*
**isDog:(img_path)** given an image, determines if image is a dog or not  
    - Inputs: image file path
    - Outputs: _
**bigBoy:(img_path, model_filepath)** given an image, whether it is a dog or not, its it a big boy or a lil guy  
    - Inputs: image file path, (optional) model storage file path
    - Outputs: predicted category of dog (big boy or lil guy)
**whatDog:(img_path, model_filepath)** given an image, whether it is a dog or not, what type of dog characteristics does it have  
    - Inputs: image file path, (optional) model storage file path
    - Outputs: predicted breed of dog

# Structure
There are 2 separate convolutional neural networks: one trained on images of both dogs and non-dogs and another trained to identify different breeds of dogs (with no training data that does not contain dogs.)

isDog is built off of the first classifier and both bigBoy and whatDog are built off of the second. 
