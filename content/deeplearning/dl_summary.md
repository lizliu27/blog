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
In my experiment, adjustments were made to optimize the training process. The number of epochs was limited, the learning rate increased, and the batch size reduced. These changes improved accuracy and minimized loss, aligning with the expectation that more epochs enhance model learning and accuracy.
![train result](/blog/images/loss_curve.jpg)


## Project 2: Training Convolutional Neural Network (CNN) from scratch


1. **Module Implementation**:
   - I learned to build a two-layer network with fully connected layers and a sigmoid activation function.
   - I implemented a vanilla CNN with a convolutional layer, ReLU activation, max-pooling layer, and fully connected layer for classification.
   - I had the opportunity to design and build my own custom CNN model, adhering to the principles of network architecture design.

    CNN Model Summary

    **Model Structure:**
    - Input: 32x32x3 images
    - Convolutional Layers: Two sets of convolutions (each with two 3x3 kernels) followed by ReLU activation and max-pooling, gradually increasing output channels (32, 64, 128, 256).
    - Fully Connected Layers: Two FC layers with ReLU activation.

    **Hyperparameters:**
    - Batch Size: 64 (reduced for faster convergence)
    - Learning Rate: 0.03 (increased for faster convergence in limited epochs)
    - Regularization (reg): 0.0005
    - Epochs: 10 (limited)
    - Steps: [6, 8] (adjust learning rate during training)
    - Warmup: 0.0001 (minimal warmup)
    - Momentum: 0.9

    **Justification:**
    - Convolutional Layers: Efficiently capture hierarchical image features.
    - ReLU Activation: Introduces non-linearity for better learning.
    - Pooling Layers: Reduce dimensionality for faster computation and reduce overfitting.
    - Fully Connected Layers: Enable learning of complex decision boundaries.

    **Overall:**
    - Designed for efficient feature extraction from images.
    - Carefully chosen hyperparameters for faster convergence and prevention of overfitting within limited epochs.


    2. **Handling Unbalanced Datasets**:
    - I explored the challenges posed by unbalanced datasets, where samples of each class are not evenly distributed.
    - I experimented with the limitations of standard training strategies on unbalanced datasets using an unbalanced version of CIFAR-10.
    - I implemented and evaluated a solution to the imbalance problem using Class-Balanced Focal Loss, as proposed in the CVPR-19 paper.

    3. **Hands-on Experience**:
    - Through the implementation of various components of CNNs, I gained practical experience in neural network architecture design and implementation.
    - I learned about the importance of experimentation and testing in machine learning projects, especially when dealing with real-world datasets and challenges.

    Overall, this project provided me with valuable insights into CNN architecture, training strategies for handling unbalanced datasets, and the importance of implementing and evaluating novel techniques in machine learning research.


## Project 3: Implementing various techniques related to image analysis and manipulation using deep learning
Specifically convolutional neural networks (CNNs) and style transfer.
Here's a summary of the tasks and techniques covered:

### Class Model Visualizations:
- Synthesizing images to maximize classification scores for specific classes, providing insights into network focus during classification.
- Generating class-specific saliency maps to understand image areas influencing classification decisions.
- Creating fooling images by perturbing input images to mislead pretrained networks.

### Gradient Class Activation Mapping (GradCAM):
- Implementing guided backpropagation and GradCAM techniques to visualize areas of an image relevant to specific class labels.
- Utilizing Captum to implement GradCAM and visualize relevant image areas.

### Adversarial Examples:
- Generating fooling images by performing gradient ascent on input images to maximize classification for specific target classes.

## Summary of Saliency Maps and GradCAM

Saliency Maps provide visualization of image regions contributing most to the CNN's decision. They are determined by computing gradients with respect to input pixels, where high gradient magnitudes indicate strong impact on the model's output. Saliency maps are suitable for general interpretability and providing quick overviews of attention regions.

On the other hand, GradCAM (Gradient-weighted Class Activation Mapping) is designed for class-specific localization information. It highlights regions highly relevant to a specific class by combining information from multiple layers and inspecting gradients of output class scores with respect to intermediate layers. GradCAM is particularly valuable for understanding why the model made certain predictions, especially in object detection tasks. However, it requires access to model layers and is more involved to implement.

In practice, both methods have strengths and can be used in combination. Saliency maps provide quick insights into overall attention regions, while GradCAM offers detailed class-specific localization information for a deeper understanding of model decisions.

![gradcam](/blog/images/gradcam.jpg)

### Style Transfer:
- Implementing content loss, style loss, and total variation loss functions for style transfer using convolutional neural networks.
- Stringing together loss functions and update rules to perform style transfer from a style image to a content image.
- Applying style transfer to generate stylized images by combining the content of one image with the style of another.
![style_trans1](/blog/images/style_trans1.jpg)
![style_trans2](/blog/images/style_trans2.jpg)


The project provided hands-on experience with various techniques for interpreting and manipulating deep neural networks trained on image data. By implementing these techniques, participants gained insights into network behavior, learned methods for generating visually appealing outputs, and explored the nuances of image analysis in the context of deep learning.

Overall, Project 3 served as a comprehensive exploration of image analysis and manipulation techniques using deep learning, offering practical experience in implementing and understanding these methods.


## Project 4 Implementing sequence-to-sequence (Seq2Seq) models and transformers for natural language processing tasks.

A summary of the tasks and techniques covered:

### RNN and LSTM Implementation
- Implementing a vanilla `RNN` unit using PyTorch Linear layers and activations.
- Implementing an `LSTM` unit using PyTorch nn.Parameter and activations, following a set of equations.

### Seq2Seq Implementation
- Implementing Seq2Seq models with an `encoder` and `decoder`.

### Seq2Seq with Attention
- Implementing a simple form of attention, using cosine similarity, to evaluate its impact on model performance.
- Referencing research papers for deeper understanding of attention mechanisms.

### Transformers
- Implementing a one-layer `transformer encoder`.
- Tasks include embeddings, multi-head self-attention, element-wise feedforward layer, final layer, forward pass, and training.
- Training the transformer encoder architecture on the dataset with default hyperparameters.

### Full Transformer Implementation
- Implementing a full transformer model using PyTorch built-in modules.
- Training the model with hyper-parameter tuning and comparing results with other implemented models.


Project 4 provides hands-on experience with building and training various neural network architectures for natural language processing tasks. Participants learn about RNNs, LSTMs, attention mechanisms, and transformers, gaining insights into their functionalities and applications in NLP. Additionally, participants gain experience in model implementation, training, and evaluation, along with hyperparameter tuning and result analysis.

Overall, Project 4 serves as a comprehensive exploration of advanced NLP techniques, equipping participants with practical skills and knowledge in deep learning for sequence modeling and language translation tasks.
