# ASL_detection

This Jupyter notebook implements an American Sign Language (ASL) alphabet recognition system using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Project Goal:
The primary goal is to classify ASL alphabet signs from images.

Key Libraries Used:

tensorflow

keras

os

zipfile

numpy

matplotlib.pyplot

pandas (although not explicitly used in the provided snippets for data manipulation, it is commonly imported alongside these libraries for data handling).

## Dataset:
The notebook uses the asl-alphabet dataset from Kaggle, downloading and unzipping it if not already present. The dataset contains 87,000 images belonging to 29 classes (26 alphabet letters + 'del', 'nothing', 'space').

## Model Architecture:
The CNN model consists of:

A Rescaling layer to normalize pixel values.

Multiple Conv2D and MaxPool2D layers for feature extraction.

BatchNormalization layers to stabilize and accelerate training.

Dropout layers for regularization.

A Flatten layer to prepare data for the dense layers.

Dense layers with 'relu' and 'softmax' activation for classification.

## Training and Evaluation:
The model is compiled with the 'adam' optimizer and 'categorical_crossentropy' loss. Training is performed for 15 epochs, with ModelCheckpoint to save the best model based on validation loss. The best validation accuracy achieved was 98.94%.

## Prediction:
A predict_image function is defined to load and preprocess a single image and predict its ASL class. The notebook demonstrates predictions on sample test images.

heart_disease.ipynb

This Jupyter notebook focuses on predicting heart disease using an XGBoost classifier.

## Key Libraries Used:

pandas

sklearn.model_selection

sklearn.preprocessing

sklearn.metrics

xgboost

Data Processing:
The notebook reads a CSV dataset, displays its head, describes its statistics, and checks for missing values. It separates features (X) from the target variable (y, 'target' column). Numerical features are scaled using StandardScaler. The data is then split into training and testing sets.

Model:
An XGBClassifier is used for the prediction task.

Prediction Example:
The notebook includes an example of predicting heart disease for a new patient, demonstrating the use of the trained XGBoost model.
