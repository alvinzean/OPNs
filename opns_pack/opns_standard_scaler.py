import numpy as np
import opns_np as op


class OPNsStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = op.mean(X, axis=0)  # Compute the mean of each column, i.e., each feature's mean
        self.std = op.std(X, axis=0)  # Compute the standard deviation of each column

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

