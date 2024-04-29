+++
title = 'Deep Learning Summary'
date = 2024-04-28T14:22:17+08:00
draft = false
+++


---

# Introduction to Deep Learning Projects

In this series of posts, I will share insights gained from working on four distinct projects in the field of deep learning. Each project presented unique challenges and opportunities for learning, covering various aspects of neural network architecture, training pipelines, and optimization techniques.

## Project 1: Training Neural Networks for MNIST Recognition

The first project focused on building a simple pipeline for training neural networks to recognize hand-written digits from the MNIST dataset. The pipeline implementation encompassed two neural network architectures, each equipped with functionalities to load data, train, and optimize the models.

### Neural Network Architectures

Two distinct neural network architectures were implemented from scratch for this project:
1. **Simple Softmax Regression**: Composed of a fully-connected layer followed by a ReLU activation.
2. **Two-Layer Multi-Layer Perceptron (MLP)**: Composed of two fully-connected layers with a Sigmoid activation in between. Note that the MLP model utilizes biases.

### Activation Functions

Two activation functions were used in this project:
1. **ReLU (Rectified Linear Unit)**: Used in the simple softmax regression.
2. **Sigmoid**: Used in the two-layer MLP.

### Training Processes

The training processes for the simple softmax regression and the two-layer MLP were implemented in this section.
An vanilla Stochastic Gradient Descent (SGD) optimizer was implemented.