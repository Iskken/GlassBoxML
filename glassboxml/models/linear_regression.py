import glassboxml.metrics.regression as regression_metrics
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.b = 0
        self.learning_rate = 0
        self.loss = 0
        self.epsilon = 1e-6
        self.losses = []
        pass

    def compute_b(self, X, y):
        b = np.mean(y) - self.w @ np.mean(X)
        return b

    def compute_weight(self, X,y, pred_y):
        # X is n by 1, y is n by 1, so (X - np.mean(X)).T @ (y - np.mean(y)) is 1 by 1,
        # and np.sum((X - np.mean(X))**2) is a scalar, so the result is a scalar
        w = (X - np.mean(X)).T @ (y - np.mean(y)) / np.sum((X - np.mean(X))**2)
        return w

    def predict(self, X):
        pred_y = X @ self.w + self.b
        return pred_y
    
    def fit_closed_form(self, X, y):
        av_x = np.mean(X)
        av_y = np.mean(y)

        numerator = (X - av_x).T @ (y - av_y)
        denominator = np.sum((X - av_x)**2)

        self.w = numerator / denominator
        self.b = av_y - self.w * av_x

    def fit_gradient_descent(self, X, y, epochs, learning_rate):
        # Initialize weights and bias
        self.w = np.zeros(len(X[0]))
        self.b = 0
        n_samples = len(X)

        for epoch in range(epochs):
            pred_y = self.predict(X)
            self.losses.append(regression_metrics.mse(y, pred_y))

            # Calculate gradient for weight
            dw = (-2) / n_samples * X.T @ (y - pred_y)

            # Calculate gradient for b
            db = (-2) / n_samples * np.sum(y - pred_y)

            if np.linalg.norm(dw) < self.epsilon and abs(db) < self.epsilon:
                print("Converged at epoch", epoch)
                break

            #update weight and b
            self.w = self.w - learning_rate*dw
            self.b = self.b - learning_rate*db

            if (epoch % 100 == 0):
                print("Updated weight: ", self.w)
                print("Updated b:", self.b, "\n")

    def plot(self, X, y):
        # Create smooth line range
        x_line = np.linspace(np.min(X), np.max(X), 100)
        y_line = self.w * x_line + self.b

        # Plot data points
        plt.scatter(X, y)

        # Plot regression line
        plt.plot(x_line, y_line)

        # Labels
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()