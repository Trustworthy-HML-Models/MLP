# Multi-Layer Perceptron from Scratch for Multi-Class Classification

## Overview

This project implements a **multi-layer perceptron (MLP)** with **backpropagation** from scratch (without libraries like TensorFlow, Keras, or PyTorch). The goal is to classify a dataset of images into one of **four classes**. The dataset consists of **24,754 samples**, each with **784 features** representing an image, divided into 4 classes (0, 1, 2, 3).

## Problem Statement

We are tasked with training a neural network to distinguish between 4 classes using the **backpropagation algorithm**. The network is a simple feedforward model with:

- **One input layer** (784 features)
- **One hidden layer** (customizable number of nodes)
- **One output layer** (4 classes, one-hot encoded labels)

The network is trained and validated on the provided dataset to avoid overfitting. Finally, we use this model to predict labels for a test set.

## Dataset

- **Training data**: `train_data.csv` contains 24,754 samples with 784 features per sample.
- **Labels**: `train_labels.csv` contains the corresponding labels for each sample, represented in a one-hot encoded format.

## Model Architecture

The network consists of:
- **Input layer**: 784 input features (corresponding to pixel values).
- **Hidden layer**: Customizable number of neurons, which you can experiment with.
- **Output layer**: 4 output neurons for classification (one for each class), using softmax activation.

## Implementation Details

- **Activation Functions**: ReLU for the hidden layer, and Softmax for the output layer.
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Gradient descent with backpropagation to update weights.
- **Performance Evaluation**: The model is evaluated on unseen test data, and its accuracy is measured.
- **Validation Set**: Used to ensure the model does not overfit during training.

## Key Features

- **Backpropagation Algorithm**: Implemented from scratch to update weights and biases in each layer.
- **Customizable Hidden Layer**: The number of neurons in the hidden layer can be adjusted for experimentation.
- **One-hot Encoded Labels**: Output labels are in a one-hot encoded format like `[1, 0, 0, 0]` for class 0.
- **Validation Set**: Splitting the training data into training and validation sets helps to avoid overfitting.
- **Final Function**: The model includes a prediction function that outputs the labels for a test set as a numpy array.

## Results
- **Training Accuracy**: The model achieved a training accuracy of 96.445%.
- **Training Loss**: The model's training loss decreased over 500 epochs to 0.113.
- **Validation Accuracy**: The model achieved a Validation accuracy of 95.314.
## Conclusion
This project demonstrates a fully implemented MLP from scratch using backpropagation to solve a multi-class classification problem. It provided hands-on experience with deep learning fundamentals and solidified understanding of the backpropagation algorithm.
