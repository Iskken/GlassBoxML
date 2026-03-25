import numpy as np

'''
Purpose:provide basic implementation of decision tree.
'''

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
    def __init__(self):
        self.depth = 0

    def build_tree(self, X, y, depth, max_depth):
        if depth >= max_depth or len(set(y)) == 1:
            return Leaf(y)

        best_feature, best_threshold = self.find_best_split(X, y)

        if best_feature is None:
            return Leaf(y)

        left_data, right_data, y_left, y_right = self.split(X, y, best_feature, best_threshold)

        left_child = self.build_tree(left_data, y_left, depth + 1, max_depth)
        right_child = self.build_tree(right_data, y_right, depth + 1, max_depth)

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

        return left_data, right_data, y_left, y_right

    def calculate_gini(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def find_best_split(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        gini_parent = self.calculate_gini(y)

        best_IG = 0
        best_feature = None
        best_threshold = None

        X = np.array(X)  # ensure numpy
        y = np.array(y)

        for f in range(n_features):
            values = np.unique(X[:, f])
            thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_idx = X[:, f] < t
                right_idx = X[:, f] >= t

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
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
    
    def predict_sample(self, x, node):
        # if leaf → return prediction
        if isinstance(node, Leaf):
            return node.value

        # otherwise follow the split
        if x[node.best_feature] <= node.best_threshold:
            return self.predict_sample(x, node.left_child)
        else:
            return self.predict_sample(x, node.right_child)