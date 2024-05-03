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




