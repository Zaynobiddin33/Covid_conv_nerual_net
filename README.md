# Chest X-Ray Classifier: COVID-19, Pneumonia, or Healthy

![Example Image](./images_model_didnt_see/normal.png)

This project is a deep learning-based classifier that predicts whether a patient's chest X-ray shows signs of COVID-19, viral pneumonia, or a healthy lung. The model was trained using a convolutional neural network (CNN) to classify X-ray images into one of these three categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)

## Project Overview
Chest X-rays are an essential tool in diagnosing respiratory conditions such as COVID-19 and pneumonia. This model uses a CNN to analyze chest X-ray images and predict whether the patient has:
- **COVID-19 (coronavirus)**,
- **Viral Pneumonia**, or 
- **Healthy Lungs**.

The model was trained on a labeled dataset of chest X-rays and demonstrates strong performance on the test set. This can potentially assist radiologists in screening and diagnosis.

## Model Architecture
The neural network was built using PyTorch and consists of the following layers:
- Convolutional layers to extract features from X-ray images.
- Max-pooling layers to downsample the feature maps.
- Fully connected layers to classify the extracted features into one of three classes (COVID-19, Pneumonia, or Healthy).

### Layers:
1. **Conv2d Layer 1**: Input channels: 1 (grayscale), Output channels: 6, Kernel size: 3
2. **Conv2d Layer 2**: Output channels: 10, Kernel size: 3
3. **Fully connected layer 1**: 10 * 61 * 61 input neurons, 1000 output neurons
4. **Fully connected layer 2**: 1000 input neurons, 200 output neurons
5. **Fully connected layer 3**: 200 input neurons, 35 output neurons

The model was trained using cross-entropy loss and optimized using the Adam optimizer.


