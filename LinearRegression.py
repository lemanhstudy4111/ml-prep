import numpy as np
import pandas as pd


def normalize(np_arr):
    arr_min = np.min(np_arr)
    arr_max = np.max(np_arr)
    return (np_arr - arr_min) / (arr_max - arr_min)


class LinearRegressionScratch:
    def __init__(self, a=0.001, epoch=1000, epsilon=0.1):
        self.a = a
        self.epoch = epoch
        self.epsilon = epsilon
        self.weights = None

    def get_predicted_values(self, weights, X):
        return weights[0] + np.dot(X, weights[1:])

    def compute_gradients(self, weights, X, y, n):
        """
        Compute gradient based on X and y. Hypothesis = w0 + w1x1 + w2x2 + ...
        Args:
            weights (_type_): 1d np array length m (features) + 1
            X (_type_): 2D np array for features
            y (_type_): 1D np array for labels
        """
        predicted_X = self.get_predicted_values(
            weights, X
        )  # 1d array of len=len(weights) - 1
        residual = predicted_X - y
        residual_features = np.multiply(X.T, residual)
        return (
            1 / n * residual_features
        )  # 1d array of gradient of each feature

    def update_weights(self, weights, gradients):
        new_weights = weights - self.a * gradients  # w0 = w0 - a*gradient_w0
        return new_weights

    def compute_err(self, n, weights, X):
        predicted_values = self.get_predicted_values(weights, X)
        return 1 / (2 * n) * np.sum(predicted_values**2)

    def fit(self, X, y):
        """
        Fit linear regression line based on gradient descent

        Args:
            X (_type_): 2D np array for features
            y (_type_): 1D np array for labels
        """
        n = len(X)
        X = normalize(X)
        y = normalize(y)
        weights = np.random.rand(len(X[0]) + 1)  # including intercept
        for _ in range(self.epoch):
            gradients = self.compute_gradients(weights, X, y, n)
            new_weights = self.update_weights(weights, gradients)
            old_err = self.compute_err(n, weights, X)
            new_err = self.compute_err(n, new_weights, X)
            if abs(new_err - old_err) < self.epsilon:
                break
            weights = new_weights
        self.weights = weights
        return

    def predict(self, X_test):
        return self.get_predicted_values(self.weights, X_test)
