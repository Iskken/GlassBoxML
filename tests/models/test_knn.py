import numpy as np
from glassboxml.models.knn import KNNClassifier, KNNRegressor

# Test KNNClassifier
def test_multi_class_classification():
    X = np.array([[0.0], [0.1], [0.2],  [5.1], [5.2], [5.5],  [10.245], [11], [12.5]])
    y = np.array([0, 0, 0,  1, 1, 1,  2, 2, 2])

    model = KNNClassifier(k = 3)
    model.fit(X,y)

    y_pred = model.predict(X)

    assert np.array_equal(y, y_pred)

def test_KNNClassifier_k_equal_1():
    X = np.array([[0.0], [0.1],  [0.456]])
    y = np.array([0, 1, 2])
    X_test = np.array([[0.01], [0.245], [0.5555]])

    model = KNNClassifier(k = 1)
    model.fit(X,y)

    y_pred = model.predict(X_test)

    assert np.array_equal(y, y_pred)

def test_majority_vote_sanity():
    X = np.array([[0.3], [0.4], [0.5]
                  , [0.6], [0.7], [0.8], [0.9], [0.10]])
    y = np.array([0, 1, 1,
                  1, 1, 0, 0, 0])
    x_test = [0.51]
    
    model = KNNClassifier(k = 5)
    model.fit(X,y)

    y_pred = model.predict(x_test)

    assert y_pred == [1]

def test_equal_class_labels():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]

    x_test = [2.1]
    
    model = KNNClassifier(k = 4)
    model.fit(X,y)

    y_pred = model.predict(x_test)

    assert y_pred == [0]

def test_KNNClassifier_prediction_accuracy():
    from glassboxml.data.generators import generate_classification_dataset
    from sklearn.model_selection import train_test_split
    from glassboxml.metrics.classification import accuracy

    X, y, w_true, b_true = generate_classification_dataset([1.555, 0.245, 6.777, 0.123])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = KNNClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    assert accuracy(y_test, y_pred) >= 0.9

def test_k_equal_n_samples_collapses_to_global_majority():
    # 5 points clustered near 0 labeled class 0, 2 points clustered near 10 labeled class 1
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [10.0], [10.1]])
    y = np.array([0, 0, 0, 0, 0, 1, 1])

    query_near_class_1 = np.array([[10.05]])

    # With k=1, the nearest neighbour is a class-1 point, so k changes the prediction
    model_k1 = KNNClassifier(k = 1)
    model_k1.fit(X, y)
    assert model_k1.predict(query_near_class_1) == [1]

    # With k=n_samples, every training point is used regardless of query location,
    # so the prediction must collapse to the global majority class (0) everywhere,
    # even for a query sitting right next to the minority cluster
    model_k_all = KNNClassifier(k = len(X))
    model_k_all.fit(X, y)

    query_near_class_0 = np.array([[0.05]])
    assert model_k_all.predict(query_near_class_1) == [0]
    assert model_k_all.predict(query_near_class_0) == [0]

# Test KNNRegressor
def test_exact_mean():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.5, 0.5, 2.9, 3.4, 5.25])

    x_test = np.array([[1.6], [3.1], [4.1]])

    y_expected = np.array([np.mean([1.5,0.5,2.9]), np.mean([0.5, 2.9, 3.4]), np.mean([2.9, 3.4, 5.25])])

    model = KNNRegressor(k = 3)
    model.fit(X,y)
    y_pred = model.predict(x_test)

    assert np.array_equal(y_expected, y_pred)

def test_KNNRegressor_k_equal_1():
    X = np.array([[0.0], [0.1],  [0.456]])
    y = np.array([0, 1, 2])
    X_test = np.array([[0.01], [0.245], [0.5555]])

    model = KNNRegressor(k = 1)
    model.fit(X,y)

    y_pred = model.predict(X_test)

    assert np.array_equal(y, y_pred)