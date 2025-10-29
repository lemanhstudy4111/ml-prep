import numpy as np
import pandas as pd


def normalize(np_arr):
    np_arr = np.array(np_arr)
    min_arr = np.min(np_arr)
    max_arr = np.max(np_arr)
    return (np_arr - min_arr) / (max_arr - min_arr)


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.0001, stop_crit=0.001):
        self.weights = None  # first weight is w0 intercept
        self.learning_rate = learning_rate
        self.stop_crit = stop_crit

    def fn(self, X):
        """
        function to get predicted score from model
        Args:
            X (_type_): 2D array of features
        return 1d model of len(X)
        """

    def compute_gradient(self, wb, X, y):
        n = len(X)
        sum_residual = np.sum((self.w0 + self.weights * X) - y) * wb
        return 1 / n * sum_residual

    def update_weight(self, gradients):
        self.weights -= self.learning_rate * gradients

    def compute_err(self, X, y):
        n = len(X)
        return 1 / (2 * n) * (np.sum(((self.w0 * X + self.w1) - y) ** 2))

    def fit(self, X, y):
        X = normalize(X)
        y = normalize(y)
        n = len(X)
        self.w0 = np.random.rand()
        self.weights = np.random.rand(n)
        gradients = None
        last_err = 0
        while True:
            gradients = self.compute_gradient(self.w0, X, y)
            self.update_weight(g0=g0, g1=g1)
            curr_err = self.compute_err(X, y) - last_err
            if curr_err < self.stop_crit:
                break
            last_err = curr_err
        return

    def predict(self, X):
        return self.w0 * X + self.w1
