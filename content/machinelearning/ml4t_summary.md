+++
title = 'Machine Learning for Trading Project3'
date = 2023-02-22T14:22:17+08:00
draft = false
+++


CART models are widely used in machine learning dealing with non-linear relationships and making predictions. To further understand which model is appropriate and the performance of models under different parameters, a series of experiments was performed. For this experiment, the main dataset was currency exchange data collected by Istanbul currency exchange. The 3 experiments were focused on overfitting, the use of bagging and the performance of classic decision tree vs random decision tree. The hypothesis is that as leaf size becomes smaller, the tree is divided into more branches and therefore leads to overfitting. Using bagging, the overfitting effect can be reduced. Compared to decision trees, random trees reduced the computation time without too much sacrifice in performance.


In this project, I implemented decision tree from scratch and tested the predicting power of this model. I then try to improve the performance of model based on different scenarios.

![train1](/blog/images/dtleaner.jpg)
![train2](/blog/images/bagleaner.jpg)

Three experiments were performed to analyse overfitting, the use of bagged learner and the performance of random tree compared to classic tree.
1.**Experiment 1**
Set up to test overfit vs leaf size. Leaf size of 1-50 was tested. The RMSE of different leaf sizes was plotted for in sample and out of sample test cases.

2.**Experiment 2**
A bagged learner was developed using 20 DT learners, with each learner using a different part of training data with replacement. The mean of the prediction for each learner was used as the final prediction. In

3.**Experiment 3**
Rather than trying to find the best feature to split, random tree learner uses a random feature to split.

The performance of random tree is examined based on train time and MAE.

![time](/blog/images/time_compare.jpg)