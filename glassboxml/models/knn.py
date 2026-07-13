import numpy as np
from glassboxml.distances.distances import calculate_euclidean

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in np.array(X)])

    def _predict_sample(self, x):
        distances = []
        for i in range(len(self.X_train)):
            distances.append((calculate_euclidean(self.X_train[i], x), i))

        k_closest_points = sorted(distances)[:self.k]
        if np.sum([1 if self.y_train[i] == 1 else 0 for _, i in k_closest_points]) > (self.k / 2):
            result = 1
        else:
            result = 0
        return result