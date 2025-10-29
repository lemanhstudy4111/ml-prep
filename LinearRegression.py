import numpy as np
import pandas as pd


def normalize(np_arr):
    np_arr = np.array(np_arr)
    min_arr = np.min(np_arr)
    max_arr = np.max(np_arr)
    return (np_arr - min_arr) / (max_arr - min_arr)


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.0001, stop_crit=0.001):
        self.gradient = None
        self.w0 = None
        self.w1 = None
        self.learning_rate = learning_rate
        self.stop_crit = stop_crit

    def compute_gradient(self, wb, X, y):
        n = len(X)
        sum_residual = np.sum((self.w0 * X + self.w1) - y) * wb
        return 1 / n * sum_residual

    def update_weight(self, g0, g1):
        self.w0 -= self.learning_rate * g0
        self.w1 -= self.learning_rate * g1

    def compute_err(self, X, y):
        n = len(X)
        return 1 / (2 * n) * (np.sum(((self.w0 * X + self.w1) - y) ** 2))

    def fit(self, X, y):
        X = normalize(X)
        y = normalize(y)
        self.w0 = np.random.rand()
        self.w1 = np.random.rand()
        g0 = 0
        g1 = 0
        last_err = 0
        while True:
            g0 = self.compute_gradient(self.w0, X, y)
            g1 = self.compute_gradient(self.w1, X, y)
            self.update_weight(g0=g0, g1=g1)
            curr_err = self.compute_err(X, y) - last_err
            if curr_err < self.stop_crit:
                break
            last_err = curr_err
        return

    def predict(self, X):
        return self.w0 * X + self.w1
