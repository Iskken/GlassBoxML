import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mode
from  glassboxml.models.decision_tree import DecisionTree
from glassboxml.metrics.classification import accuracy

def _fit_one_tree(train_X, train_y, X_oob, y_oob):
    # Module-level (not a method) so it can be pickled and sent to a
    # worker process - each tree only needs its own bootstrap sample and
    # OOB set, so this can run fully independently of the other trees
    model = DecisionTree(100)
    model.fit(train_X, train_y)
    tree_accuracy = accuracy(y_oob, model.predict(X_oob))
    return model, tree_accuracy

class RandomForest:
    def __init__(self, n):
        self.n = n

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        # #Bootstrapping
        # rng = np.random.default_rng()
        # indice_samples = np.array([rng.choice(len(X), size = len(X), replace=True) for i in range(self.n)])
        # train_X = [[self.X_train[i] for i in s] for s in indice_samples]
        # train_y = [[self.y_train[i] for i in s] for s in indice_samples]
        # # For each tree, the OOB indices are the training points that were
        # # never picked in that tree's bootstrap sample (set difference)
        # test_OOB = [np.setdiff1d(np.arange(len(X)), s) for s in indice_samples]

        # #Training N trees on N samples
        # accuracies = []
        # self.models = []
        # for i in range(self.n):
        #     model = DecisionTree(100)
        #     model.fit(train_X[i], train_y[i])
        #     y_pred = model.predict([self.X_train[j] for j in test_OOB[i]])
        #     y_true = [self.y_train[j] for j in test_OOB[i]]
        #     accuracies.append(accuracy(y_pred, y_true))
        #     self.models.append(model)
        
        # self.accuracy = np.mean(accuracies)


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
            results = list(executor.map(_fit_one_tree, train_X, train_y, X_oob, y_oob))

        self.models = [model for model, _ in results]
        self.accuracy = np.mean([tree_accuracy for _, tree_accuracy in results])




    def predict(self, X):
        # p_pred = []
        # for x in X:
        #     p = []
        #     for t in self.models:
        #         p.append(t.predict_sample(x, t.root))

        #     p_pred.append(p)
        
        # y_pred = []
        # for p in p_pred:
        #     from collections import Counter
        #     res = Counter(p).most_common(1)[0][0]
        #     y_pred.append(res)

        # return y_pred

        # Each tree's own vectorized predict() returns every sample's
        # prediction in one call - stacking these gives a (n_trees, n_samples)
        # matrix where column j holds every tree's vote for sample j
        t_pred = np.array([t.predict(X) for t in self.models])

        # Majority vote per sample = the mode down each column (axis=0)
        y_pred, _ = mode(t_pred, axis=0)

        return y_pred