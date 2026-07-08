import numpy as np

class Leaf:
    def __init__(self, y):
        self.value = self._majority_class(y)

    def _majority_class(self, y):
        from collections import Counter
        return Counter(y).most_common(1)[0][0]
    

class Node:
    def __init__(self, best_feature, best_threshold, left_child, right_child):
        self.best_feature = best_feature
        self.best_threshold = best_threshold
        self.left_child = left_child
        self.right_child = right_child

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.root = self.build_tree(X,y, 0)
    
    def build_tree(self, X, y, depth):
        # if we either reach maximum depth or all of the elements left are of the same class, then return the leaf
        if depth >= self.max_depth or len(set(y))==1:
            return Leaf(y)
        
        best_feature, best_threshold = self.find_best_split(X, y)

        left_data, right_data, y_left, y_right = self.split(X, y, best_feature, best_threshold)

        left_child = self.build_tree(left_data, y_left, depth+1)
        right_child = self.build_tree(right_data, y_right, depth+1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def split(self, X, y, b_f, b_th):
        left_data = []
        right_data = []

        y_left = []
        y_right = []

        for i in range(len(X)):
            if X[i][b_f] <= b_th:
                left_data.append(X[i])
                y_left.append(y[i])
            else:
                right_data.append(X[i])
                y_right.append(y[i])

        return (
            np.array(left_data),
            np.array(right_data),
            np.array(y_left),
            np.array(y_right)
        )

    def calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_IG = 0
        best_feature = 0
        best_threshold = 0

        gini_parent = self.calculate_gini(y)

        for f in range(n_features):
            values = np.unique(X[:, f])
            thresholds = (values[:-1] + values[1:])/2
            for t in thresholds:
                left_idx = X[:,f] < t
                right_idx = X[:,f] > t

                if (len(y[left_idx]) == 0 or len(y[right_idx]) == 0):
                    continue

                gini_left = self.calculate_gini(y[left_idx])
                gini_right = self.calculate_gini(y[right_idx])

                curr_IG = gini_parent - (
                    (len(y[left_idx]) / n_samples) * gini_left +
                    (len(y[right_idx]) / n_samples) * gini_right
                )

                if curr_IG > best_IG:
                    best_IG = curr_IG
                    best_feature = f
                    best_threshold = t
        
        return best_feature, best_threshold
    
    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])

    def predict_sample(self, x, node):
        if isinstance(node, Leaf):
            return node.value

        if x[node.best_feature] <= node.best_threshold:
            return self.predict_sample(x, node.left_child)
        else:
            return self.predict_sample(x, node.right_child)