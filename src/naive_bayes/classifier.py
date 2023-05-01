#TODO: add Classifier for Discreet data
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
    def accuracy(X_test, y_test, threshold=0.5):
        """
        Calculates the accuracy of the logistic regression model on the test data.

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.

        y_test : array-like of shape (n_samples,)
            The true target values for the test input samples.

        threshold : float, optional (default=0.5)
            The threshold value to use for the predicted probabilities.
            All probabilities above this threshold are considered positive.

        Returns:
        --------
        float
            The accuracy of the logistic regression model on the test data.
            This is defined as the number of correct predictions divided by
            the total number of predictions.

        Raises:
        -------
        ValueError
            If X_test and y_test have incompatible shapes, or if y_test contains
            values other than 0 or 1.
        """
        y_pred = self.predict(X_test, threshold=threshold)
        return (np.equal(y_pred, y_test).sum()/len(y_test))
    def vizualize_results(X_test, y_test, method='confusion_matrix'):
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
        import matplotlib.pyplot as plt
        import seaborn as sns
        """Visualize the accuracy of the model using various metrics.

        Args:
            X_test (numpy.array): A numpy array containing the test input data.
            y_test (numpy.array): A numpy array containing the expected output values for the test data.
            method (str, optional): A string specifying the visualization method to use. 
                Default is 'confusion_matrix'. Possible values are 'confusion_matrix', 
                'roc_curve', and 'precision_recall_curve'.

        Returns:
            None: This function does not return anything, it only produces visualizations.

        Raises:
            ValueError: If an invalid method name is provided.

        Examples:
            >>> # Create and train a logistic regression model
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train)
            >>> 
            >>> # Visualize the accuracy of the model using a confusion matrix
            >>> model.vizualize_results(X_test, y_test, method='confusion_matrix')
        """
        if method == 'confusion_matrix':
            y_pred = self.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
            
        elif method == 'roc_curve':
            y_pred_prob = self.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.show()
            
        elif method == 'precision_recall_curve':
            y_pred_prob = self.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()
            
        else:
            print('Invalid method name.')

