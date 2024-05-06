
+++
title = ' Deep Learning in auto-diagnosis Electroencephalography '
date = 2023-12-22T14:22:17+08:00
draft = false
+++

This project was led by my awesome teammate zheng cheng. We together replicated a result an exsiting paper.

We tried to replicate the results and applied deep learning models to the auto-diagnosis of epilepsy via Electroencephalography (EEG) profiles.




We developed an architecture based on transformers, similar to those used for understanding sentiment in text. This architecture helped us extract important information from our data. We made several adjustments to this architecture and fine-tuned its settings. This process helped us understand the balance between capturing the complexity of the data (bias-variance trade-off) and the technical details like optimizer choices and regularization techniques in deep learning.

![architecture](/blog/images/transformer_achitecture.jpg)

Our project taught us three important lessons.

First, we learned how crucial it is to deal with situations where one class of data is much more common than others in a classification task, and also how different pieces of data relate to each other.

Second, we noticed something interesting: sometimes our model performed really well on some parts of the data but not so well on others. For instance, a particular type of neural network did exceptionally well on certain subsets of our data during testing, but not so well on others. This hinted that our deep learning model might have become too specialized during training, leading it to be biased in its classifications. Simply having low error rates on validation and test data doesn't always mean the model is perfect.

Lastly, we found that our combined transformer classifier could effectively tell the difference between epilepsy patients and healthy individuals. This success encouraged us to consider using this approach for more complex tasks in the future.

![learn curve](/blog/images/figure2.jpg)