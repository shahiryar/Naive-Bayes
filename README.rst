.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/Naive-Bayes.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/Naive-Bayes
    .. image:: https://readthedocs.org/projects/Naive-Bayes/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://Naive-Bayes.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/Naive-Bayes/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/Naive-Bayes
    .. image:: https://img.shields.io/pypi/v/Naive-Bayes.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/Naive-Bayes/
    .. image:: https://img.shields.io/conda/vn/conda-forge/Naive-Bayes.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/Naive-Bayes
    .. image:: https://pepy.tech/badge/Naive-Bayes/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/Naive-Bayes
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/Naive-Bayes

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
Naive Bayes Algorithm
===========

Naive Bayes is a probabilistic algorithm used for classification tasks. It is based on Bayes' theorem, which describes the probability of an event occurring based on prior knowledge of conditions that might be related to the event. The algorithm is called "naive" because it makes a simplifying assumption that the features used for classification are independent of each other, which is often not the case in real-world scenarios.

This implementation uses the Gaussian distribution to model the probability density function of each class. The algorithm assumes that the features follow a normal distribution, and estimates the mean and variance for each class based on the training data. Given a new instance, the algorithm calculates the likelihood of each class using the estimated parameters and predicts the class with the highest likelihood.

==============
Implementation
==============

The `GaussianNB` class provides methods for training the model and making predictions. The class has the following attributes:

- `classes`: an array containing the unique class labels observed in the training data.
- `class_priors`: an array containing the prior probabilities of each class.
- `mean`: a matrix containing the mean of the feature values for each class.
- `variance`: a matrix containing the variance of the feature values for each class.

The class has the following methods:

   `fit(X, y)`

This method trains the Gaussian Naive Bayes model using the input data `X` and the corresponding target values `y`. The method estimates the prior probabilities, mean, and variance for each class based on the training data.

   `predict(X)`

This method predicts the class labels of the input data `X` using the fitted model. The method calculates the likelihood of each class for each instance in `X`, and predicts the class with the highest likelihood.

   `confusion_matrix(X_test, y_test)`

This method calculates the confusion matrix for the trained model given the test data and the corresponding true labels. The method returns an array containing the number of instances where the actual class was i and the predicted class was j.

   `accuracy(X_test, y_test, threshold=0.5)`

This method calculates the accuracy of the trained model on the test data. The method returns the number of correct predictions divided by the total number of predictions.

=====================
Uses and Applications
=====================

Naive Bayes is a simple yet effective algorithm that can be used for classification tasks. It is often used in text classification tasks such as spam filtering, sentiment analysis, and document classification. It has also been applied to other domains such as image recognition, medical diagnosis, and fraud detection. Naive Bayes is particularly useful in situations where the number of features is large compared to the number of instances, as it can handle high-dimensional data efficiently.

