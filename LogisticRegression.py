import numpy as np
import pandas as pd
from LinearRegression import normalize


class LogisitcRegressionScratch:
    def __init__(self, a=0.001, epsilon=0.1, epoch=1000):
        self.a = a
        self.epsilon = epsilon
        self.epoch = epoch
        self.weights = None

    def get_predicted_values(self, weights, X):
        return weights[0] + np.dot(X, weights[1:])

    def compute_gradient(self, X, y, weights, n):
        predicted = self.get_predicted_values(weights=weights, X=X)
        residual = predicted - y
        residual_feature = X.T * residual
        return 1 / n * np.sum(residual_feature)

    def compute_err(self, X, y, weights, n):
        predicted = self.get_predicted_values(weights=weights, X=X)
        err_each = -y * (np.log2(predicted)) - (1 - y)(np.log2(1 - predicted))
        return 1 / n * np.sum(err_each)

    def update_weights(self, weights, gradient):
        return weights - self.a * gradient

    def fit(self, X, y):
        n = len(X)
        X = normalize(X)
        y = normalize(y)
        weights = np.random.rand(len(X[0]) + 1)
        for _ in range(self.epoch):
            gradients = self.compute_gradient(X, y, weights=weights, n=n)
            new_weights = self.update_weights(weights, gradients)
            if abs(new_weights - weights) < self.epsilon:
                break
            weights = new_weights
        self.weights = weights
        return

    def predict(self, X_test):
        return self.get_predicted_values(self.weights, X_test)
