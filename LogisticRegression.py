import numpy as np
import pandas as pd
from LinearRegression import normalize


class LogisitcRegressionScratch:
    def __init__(self, a=0.001, epsilon=0.1, epoch=1000, fit_intercept=True):
        self.a = a
        self.epsilon = epsilon
        self.epoch = epoch
        self.fit_intercept = fit_intercept

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def get_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_predicted_values(self, weights, X):
        return np.dot(X, weights)

    def compute_gradient(self, X, y, predicted, n):
        residual = predicted - y
        residual_feature = X.T * residual
        return 1 / n * np.sum(residual_feature)

    def compute_err(self, y, sigmoid, n):
        err_each = -y * (np.log2(sigmoid)) - (1 - y)(np.log2(1 - sigmoid))
        return 1 / n * np.sum(err_each)

    def update_weights(self, weights, gradient):
        return weights - self.a * gradient

    def fit(self, X, y):
        n = len(X)
        X = normalize(X)
        y = normalize(y)
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.weights = np.random.rand(len(X[0]) + 1)
        for _ in range(self.epoch):
            predicted = self.get_predicted_values(self.weights, X)
            sigmoid = self.get_sigmoid(predicted)
            gradients = self.compute_gradient(y, sigmoid, n=n)
            new_weights = self.update_weights(self.weights, gradients)
            if abs(new_weights - self.weights) < self.epsilon:
                break
            self.weights = new_weights
        return

    def predict(self, X_test):
        return self.get_predicted_values(self.weights, X_test)
