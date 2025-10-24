import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

import opns_pack.opns_np as op


class OPNsKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.distance = None
        self.X_train = None
        self.y_train = None
        self.classes_ = None  # Used when called by GridSearchCV

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def predict_single(self, x):
        # Calculate Euclidean distance between the new point and each point in the training set
        distances = self.X_train.distance(x)
        # Get indices of the k nearest points
        nearest = op.argsort(distances)[:self.n_neighbors]
        # Find the corresponding classifications based on the indices of the k nearest points
        topK_y = [self.y_train[i] for i in nearest]
        # Vote on the classifications of the k nearest points
        votes = Counter(topK_y)
        # Find the classification with the most votes
        y_pred = votes.most_common(1)[0][0]
        return y_pred

    def predict(self, X_test):
        # Predict each element in the test set individually
        return [self.predict_single(X_test[i]) for i in range(X_test.shape[0])]