import numpy as np
from scipy.stats import norm

class GaussianNB:
    """
    A Gaussian Naive Bayes classifier.

    Parameters
    ----------
    None

    Attributes
    ----------
    classes : ndarray of shape (n_classes,)
        The unique class labels observed in the training data.
    class_priors : ndarray of shape (n_classes,)
        The prior probabilities of each class.
    mean : ndarray of shape (n_classes, n_features)
        The mean of the feature values for each class.
    variance : ndarray of shape (n_classes, n_features)
        The variance of the feature values for each class.

    Methods
    -------
    fit(X, y)
        Fit the Gaussian Naive Bayes model to the training data.
    predict(X)
        Predict the class labels of the input data using the fitted model.
    """

    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.mean = None
        self.variance = None
        
    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training data.
        y : ndarray of shape (n_samples,)
            The target values.

        Returns
        -------
        self : GaussianNB
            The fitted model.
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.class_priors = np.zeros(n_classes)
        self.mean = np.zeros((n_classes, n_features))
        self.variance = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = X_c.shape[0] / X.shape[0]
            self.mean[i, :] = X_c.mean(axis=0)
            self.variance[i, :] = X_c.var(axis=0)
        
    def predict(self, X):
        """
        Predict the class labels of the input data using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        y_pred = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            posteriors = []
            
            for j, c in enumerate(self.classes):
                prior = np.log(self.class_priors[j])
                likelihood = np.sum(np.log(norm.pdf(x, self.mean[j, :], np.sqrt(self.variance[j, :]))))
                posterior = prior + likelihood
                posteriors.append(posterior)
                
            y_pred[i] = self.classes[np.argmax(posteriors)]
        
        return y_pred
