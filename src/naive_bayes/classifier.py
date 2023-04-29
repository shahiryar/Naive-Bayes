
import numpy as np
from scipy.stats import norm

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.mean = None
        self.variance = None
        
    def fit(self, X, y):
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
