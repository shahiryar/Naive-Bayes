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
        posteriors = []
        for j, c in enumerate(self.classes):
            prior = np.log(self.class_priors[j])
            likelihood = np.sum(np.log(norm.pdf(X, self.mean[j, :], np.sqrt(self.variance[j, :]))), axis=1)
            posterior = prior + likelihood
            posteriors.append(posterior)
        posteriors = np.array(posteriors).T
        y_pred = self.classes[np.argmax(posteriors, axis=1)]
        return y_pred
    
    def confusion_matrix(X_test, y_test):
        """Compute the confusion matrix for the logistic regression model.
        Parameters:
        -----------
        X_test: array-like of shape (n_samples, n_features)
        Test data.

        y_test: array-like of shape (n_samples,)
        True labels for `X_test`.

        Returns:
        --------
        confusion_m: array-like of shape (n_classes, n_classes)
        Confusion matrix, where `n_classes` is the number of unique classes in `y_test`.
        The rows represent the actual classes and the columns represent the predicted classes.
        The (i, j) element of the matrix represents the number of instances where the actual class
        was i and the predicted class was j.
        """

        y_pred = self.predict(X_test)
        classes = np.unique(y_test)
        confusion_m = []
        for pred_c in classes:
            idx = np.where(y_pred == pred_c)
            _test_pred = y_test[idx]
            _row = []
            for actual_c in classes:
                _row.append(np.equal(_test_pred, actual_c).sum())
            confusion_m.append(_row)

        return np.array(confusion_m).T

