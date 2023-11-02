# Machine Learning Algorithms for Crack Detection on Edge Devices

## Overview

This repository contains machine learning algorithms for detecting cracked images on edge devices. The algorithms included in this project are Support Vector Machines (SVM), Logistic Regression, 1D Convolutional Neural Network (1D CNN), k-Nearest Neighbors (KNN), and Convolutional Neural Network (CNN). These algorithms are designed to predict whether input images contain cracks or not.

## Table of Contents

1. [Introduction](#introduction): Welcome to the "Machine Learning Algorithms for Crack Detection on Edge Devices" project. In this repository, we present a comprehensive solution for the detection of cracks in images using various machine learning algorithms. Our goal is to enable the deployment of these algorithms on edge devices, making it possible to perform real-time crack detection with minimal computational resources.

2. [Usage](#usage)
    - [Requirements](#requirements): Python 3.9 or above, Raspberry Pi
    - [Installation](#installation): Need to deploy Amazon Greengrass in the raspberry pi
    - [Running the Models](#running-the-models): Run the models on the edge Device to compute it specification like( Bateery, CPU Utilization and Memory)
3. [Model Details](#model-details)
    - [SVM](#svm):
    - Algorithm Description:
Support Vector Machines (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks. In the context of crack detection, SVM is employed to classify input images as either containing cracks (positive class) or not (negative class). It does this by finding the optimal hyperplane that best separates the two classes in the feature space. SVM can handle high-dimensional data and is particularly effective when there is a clear margin of separation between classes.
    - [Logistic Regression](#logistic-regression)
    - Logistic Regression is a simple yet effective algorithm used for binary classification tasks, making it suitable for problems like crack detection. It models the probability that a given input image belongs to the positive (cracked) class. The algorithm uses the logistic function (sigmoid) to map the output to a probability between 0 and 1. If the probability is greater than a predefined threshold (e.g., 0.5), the image is classified as cracked; otherwise, it's classified as not cracked.
    - [1D CNN](#1d-cnn)
    - The 1D Convolutional Neural Network (1D CNN) is a deep learning architecture used for sequence-based data, such as time series or one-dimensional signals. In this project, 1D CNNs are applied to crack detection by treating the pixel values of the input images as a one-dimensional signal. The network uses convolutional layers to learn features from the signal and make predictions.
    - [KNN](#knn)
    - k-Nearest Neighbors (KNN) is a simple yet effective instance-based machine learning algorithm used for classification tasks. It classifies input images based on the majority class among their k-nearest neighbors in the feature space. KNN does not require explicit model training; instead, it stores the training data and computes predictions at runtime based on the nearest neighbors.
    - [CNN](#cnn)
    - Convolutional Neural Networks (CNNs) are deep learning models designed for image-related tasks. In the context of crack detection, a CNN is used to automatically learn hierarchical features from input images. The network typically consists of convolutional layers, pooling layers, and fully connected layers. It excels at capturing spatial patterns in images.
4. [Data](#data): Will provide soon
6. [Results](#results): Will update soon
7. [Contributing](#contributing)  : Sandeep, Farjad
8. [License](#license)

## Goal

The goal of this project is to provide machine learning models for crack detection on edge devices. These models can be used to determine whether input images are cracked or not, which can be valuable in various applications, such as structural integrity monitoring.

## Usage

### Requirements

To run the machine learning models, you'll need the following:

- Python 3.10
- Required Python libraries (specified in requirements.txt), Please run the below command.
- pip install -r requirements.txt

### Installation

1. Clone this repository to your local machine:
  https://github.com/patllola/Machine-Learning-Model


