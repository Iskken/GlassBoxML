import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.stats import mode
from  glassboxml.models.decision_tree import DecisionTree
from glassboxml.metrics.classification import accuracy

def _fit_one_tree(train_X, train_y, X_oob, y_oob, max_features):
    # Module-level (not a method) so it can be pickled and sent to a
    # worker process - each tree only needs its own bootstrap sample and
    # OOB set, so this can run fully independently of the other trees
    model = DecisionTree(100, max_features)
    model.fit(train_X, train_y)
    tree_accuracy = accuracy(y_oob, model.predict(X_oob))
    return model, tree_accuracy

class RandomForest:
    def __init__(self, n, max_features=None):
        self.n = n
        self.max_features = max_features

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        if self.max_features == None:
            self.max_features = np.int16(np.floor(np.sqrt(len(X[0]))))

        #Bootstrapping
        rng = np.random.default_rng()
        # Create n samples for each tree where each sample is of size - len(X). 
        # Each point is selected with replacement.
        indice_samples = rng.choice(len(X), size = (self.n, len(X)), replace=True)
        
        # Create train dataset from sampled indices
        train_X = self.X_train[indice_samples]
        train_y = self.y_train[indice_samples]

        #Out-of-Bag(OOB) test computations
        orig_indices = np.arange(len(X))
        test_OOB = [np.setdiff1d(orig_indices, s) for s in indice_samples]

        # Training n trees on n samples - each tree is fit independently of
        # the others, so hand the n fits out across worker processes instead
        # of fitting them one at a time on a single core
        X_oob = [self.X_train[oob] for oob in test_OOB]
        y_oob = [self.y_train[oob] for oob in test_OOB]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                _fit_one_tree, train_X, train_y, X_oob, y_oob, repeat(self.max_features, self.n)
            ))

        self.models = [model for model, _ in results]
        # A tree's bootstrap sample can, by chance, cover every original index,
        # leaving its OOB set empty - accuracy() on an empty array is nan, and
        # a plain mean would let a single nan wipe out the whole average.
        # nanmean instead treats "no OOB samples for this tree" as no
        # information, and just excludes it from the estimate.
        self.accuracy = np.nanmean([tree_accuracy for _, tree_accuracy in results])

    def predict(self, X):
        # Each tree's own vectorized predict() returns every sample's
        # prediction in one call - stacking these gives a (n_trees, n_samples)
        # matrix where column j holds every tree's vote for sample j
        t_pred = np.array([t.predict(X) for t in self.models])

        # Majority vote per sample = the mode down each column (axis=0)
        y_pred, _ = mode(t_pred, axis=0)

        return y_pred