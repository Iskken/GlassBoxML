import numpy as np
from glassboxml.models.decision_tree import DecisionTree, Leaf, Node

def test_tree_perfect_split():
    X = [[1], [2], [3], [10], [11], [12]]
    y = [0, 0, 0, 1, 1, 1]

    model = DecisionTree(max_depth=2)
    model.fit(X, y)

    preds = [model.predict_sample(x, model.root) for x in X]

    assert preds == y, f"Predictions {preds} do not match labels {y}"

def test_pure_node_returns_leaf():
    X = [[1], [2], [3]]
    y = [1, 1, 1]

    model = DecisionTree(max_depth=3)
    model.fit(X, y)

    assert isinstance(model.root, Leaf)
    assert model.root.value == 1

def test_max_depth_limits_growth():
    X = [[1], [2], [3], [10], [11], [12]]
    y = [0, 0, 0, 1, 1, 1]

    model = DecisionTree(max_depth=0)
    model.fit(X, y)

    assert isinstance(model.root, Leaf)

def test_gini_impurity():
    model = DecisionTree(max_depth=1)

    y = np.array([0, 0, 1, 1])
    gini = model.calculate_gini(y)

    assert np.isclose(gini, 0.5), f"Gini {gini} should be 0.5"

def test_predict_consistency():
    X = [[1], [2], [3], [10], [11], [12]]
    y = [0, 0, 0, 1, 1, 1]

    model = DecisionTree(max_depth=2)
    model.fit(X, y)

    preds1 = [model.predict_sample(x, model.root) for x in X]
    preds2 = [model.predict_sample(x, model.root) for x in X]

    assert preds1 == preds2