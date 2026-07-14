import numpy as np
from collections import Counter

class _KNNBase:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in np.array(X)])

    def _k_nearest_labels(self, x):
        #Use vectorized calculations for calculating distances
        distances = np.linalg.norm(self.X_train - x, axis=1)

        #Use argpartition to obtain k closest neighbours
        partitioned_indices = np.argpartition(distances, self.k) #Put k closest elements at the front of the index list
        k_closest_indices = partitioned_indices[:self.k]

        return self.y_train[k_closest_indices]

    def _predict_sample(self, x):
        raise NotImplementedError

class KNNClassifier(_KNNBase):
    def _predict_sample(self, x):
        k_labels = self._k_nearest_labels(x)
        return Counter(k_labels).most_common(1)[0][0]

class KNNRegressor(_KNNBase):
    def _predict_sample(self, x):
        k_labels = self._k_nearest_labels(x)
        return np.mean(k_labels)