
+++
title = 'Saliency Maps, GradCAM and Style Transfer'
date = 2023-11-28T11:22:17+08:00
draft = false
+++


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
Style transfer is a technique that allows us to apply the style of one image to the content of another, resulting in a new image that combines the two. How to do this :) ? We do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep network, and then performing gradient
descent on the pixels of the image itself.
- Implementing content loss, style loss, and total variation loss functions for style transfer using convolutional neural networks.
- Stringing together loss functions and update rules to perform style transfer from a style image to a content image.
- Applying style transfer to generate stylized images by combining the content of one image with the style of another.
![style_trans1](/blog/images/style_trans1.jpg)
![style_trans2](/blog/images/style_trans2.jpg)


The project provided hands-on experience with various techniques for interpreting and manipulating deep neural networks trained on image data. By implementing these techniques, participants gained insights into network behavior, learned methods for generating visually appealing outputs, and explored the nuances of image analysis in the context of deep learning.

Overall, Project 3 served as a comprehensive exploration of image analysis and manipulation techniques using deep learning, offering practical experience in implementing and understanding these methods.
