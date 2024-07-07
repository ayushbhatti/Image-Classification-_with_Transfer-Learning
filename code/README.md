# Project Title: Image Classification and Transfer Learning with VGG16 on MNIST Dataset

## Overview

This project is for an image classification task using transfer learning with the VGG16 model on the MNIST dataset. The goal is to leverage the powerful feature extraction capabilities of VGG16, pre-trained on ImageNet, to classify MNIST digits after converting the grayscale images to RGB and resizing them to fit VGG16's input requirements.

## Models Implemented

-   **Model:** VGG16 Base Model

## Dataset

Dataset used for the Image classification is MNIST Dataset

## Results

Average Metrics over 10 Epochs \
**VGG16 Base Model** \
- **Average Loss:** 0.7288

\- **Average Accuracy:** 0.8081,

\- **Average Validation Loss:** 0.3054,

\- **Average Validation Accuracy:** 0.9195

These averages indicate that over the 10 epochs:

The model's average training loss was approximately 72.88%, suggesting it effectively minimized the error during training. The average training accuracy was about 80.81%, indicating good performance on the training set. The average validation loss was about 0.3054, showing how well the model performed on the validation set. The average validation accuracy was approximately 91.95%, demonstrating strong generalization to unseen data.
