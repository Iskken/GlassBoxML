import numpy as np
import pytest
from glassboxml.models.random_forest import RandomForest
from glassboxml.data.generators import generate_classification_dataset
from sklearn.model_selection import train_test_split

def test_pipeline_works():
    X,y,_,_ = generate_classification_dataset(w_true=[1.5,0.1, 2.1], b_true=0.01, n_samples=1000)

    model = RandomForest(10)
    model.fit(X, y)

    assert model.accuracy >= 0.9

def test_output_shape_and_values():
    X, y, _, _ = generate_classification_dataset(w_true=[1.5,0.1, 2.1], b_true=0.01, n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = RandomForest(10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    assert y_pred.shape == (len(y_test),)

    assert np.isin(y_pred, y_train, assume_unique=False).all()

def test_all_trees_fit():
    X, y, _, _ = generate_classification_dataset(w_true=[1.5,0.1, 2.1], b_true=0.01, n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = RandomForest(10)
    model.fit(X_train, y_train)

    assert len(model.models) == 10

def test_default_max_features():
    X, y, _, _ = generate_classification_dataset(w_true=[1.5,0.1, 2.1], b_true=0.01, n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = RandomForest(10)
    model.fit(X_train, y_train)

    n_features = len(X[0])

    assert model.models[0].max_features == np.int16(np.floor(np.sqrt(n_features)))

def test_max_features_larger_orig_features():
    X, y, _, _ = generate_classification_dataset(w_true=[1.5,0.1, 2.1], b_true=0.01, n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = RandomForest(10, max_features=4)

    with pytest.raises(ValueError):
        model.fit(X_train, y_train)