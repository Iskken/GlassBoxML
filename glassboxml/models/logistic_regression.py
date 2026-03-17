import numpy as np
from glassboxml.losses.regularization import L1Regularization, L2Regularization
from glassboxml.optimizers.gradient_descent import GradientDescent

class LogisticRegression:
    def __init__(self, regularization=None):
        self.regularization = regularization
        self.epsilon = 1e-6
        self.losses = []
        pass
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y, epochs, learning_rate):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
        optimizer = GradientDescent(learning_rate)

        for epoch in range(epochs):
            y_pred = self.predict_proba(X)

            dw = 1/n_samples * X.T @ (y_pred- y) + (self.regularization.gradient(self.w) if self.regularization else 0)
            db = 1/n_samples * np.sum(y_pred - y)

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

            loss = -np.mean(y * np.log(y_pred + self.epsilon) + (1 - y) * np.log(1 - y_pred + self.epsilon))
            self.losses.append(loss)

            self.w = params['w']
            self.b = params['b']