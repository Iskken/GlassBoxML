import glassboxml.losses.mse as mse
import numpy as np
import matplotlib.pyplot as plt
from glassboxml.losses.regularization import L1Regularization, L2Regularization
from glassboxml.optimizers.gradient_descent import GradientDescent

class LinearRegression:
    def __init__(self, regularization=None):
        self.regularization = regularization
        self.b = 0
        self.learning_rate = 0
        self.loss = 0
        self.epsilon = 1e-6
        self.losses = []
        pass

    def predict(self, X):
        pred_y = X @ self.w + self.b
        return pred_y

    def fit_closed_form(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Per-column mean, so this centers every feature independently
        # (np.mean(X) alone would collapse a multi-feature X to one scalar)
        av_x = np.mean(X, axis=0)
        av_y = np.mean(y)

        X_centered = X - av_x
        y_centered = y - av_y

        # Centering X and y removes the need for a bias column; lstsq solves
        # the normal equations without explicitly inverting X^T X, which is
        # both faster and numerically safer than np.linalg.inv
        self.w, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        self.b = av_y - self.w @ av_x

    def fit_gradient_descent(self, X, y, epochs, learning_rate):
        # Initialize weights and bias
        self.w = np.zeros(len(X[0]))
        self.b = 0
        n_samples = len(X)
        optimizer = GradientDescent(learning_rate)

        for epoch in range(epochs):
            pred_y = self.predict(X)
            # Include the regularization penalty so self.losses reflects the
            # actual objective being minimized, not just the MSE term
            loss = mse.mse_loss(y, pred_y)
            if self.regularization:
                loss += self.regularization.loss(self.w)
            self.losses.append(loss)

            # Calculate gradient for weight
            dw = (-2) / n_samples * X.T @ (y - pred_y) + (self.regularization.gradient(self.w) if self.regularization else 0)

            # Calculate gradient for b
            db = (-2) / n_samples * np.sum(y - pred_y)

            if np.linalg.norm(dw) < self.epsilon and abs(db) < self.epsilon:
                print("Converged at epoch", epoch)
                break

            params ={
                'w': self.w,
                'b': self.b
            }

            grads = {
                'w': dw,
                'b': db
            }

            optimizer.step(params, grads)

            #update weight and b
            self.w = params['w']
            self.b = params['b']

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