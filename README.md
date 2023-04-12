# isDog
Final project: utilities package for identifying and classifying images of dogs (and not dogs). 
Task breakdown: Back end -> Sam, Front end -> Ellie, Testing -> Dani

# Functions
isDog: given an image, determines if image is a dog or not  
bigBoy: given an image, whether it is a dog or not, its it a big boy or a lil guy  
whatDog: given an image, whether it is a dog or not, what type of dog characteristics does it have  

# Structure
The backend is a KNN classifier -> built primarily by Sam that classifies user input images with regards to traits (dog vs not dog, bigboy vs lilguy, yes/no for series of additional descriptor traits/breeds to be determined  

The front end functions -> built primarily by Ellie allow the user access to functionality in a way that makes sense to the user  

The testing suite -> built primarily by Dani ensures that everything is working as it should

# Image Guidelines

Please use the following guidelines for user images to ensure maximum accuracy. 
- .png or .jpg filetype only
- images should be square (1:1 aspect ratio). Non-square images will be cropped/resized
- Only one target per image for maximum effectiveness. Target should be approximately centered in image. 
