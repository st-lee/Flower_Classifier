# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


# Project Breakdown

1. Load the data and define transforms for traning, valdation, and testing sets.

2. Buliding and training the classifier: creating the architecture with VGG model and my own classifier. We train the classifier with help from Udacity's GPU-enabled plstform.

3. Saving and loading the model to a `checkpoint.pth`.

4. Inference for classification: With the loading model, we randomly choose a image from test set and predict the category of it.


# Files

1. `Image Classifier Project.ipynb/.html`: This is the Jupyter Notebook where I go through the whole flow from loading data to predicting flower category. The HTML form is convert from the ipynb, which is ease for mentor reviewing.

2. `train.py`: This is the first half code of the project, which contains loading data, buliding and training the classifier and saving the model.

3. `predict.py`: This is the last half code of the project, which contains loading the model and making predicting.

4. `*_args.py`: Define all the Argument Parser for the command line application. 