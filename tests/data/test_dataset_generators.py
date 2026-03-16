import numpy as np
from glassboxml.data.generators import generate_regression_dataset

def test_dataset_shapes():
    w_true = np.array([1.0, 2.0, 3.0])

    X, y, w, b = generate_regression_dataset(
        w_true=w_true,
        n_samples=500
    )

    assert X.shape == (500, 3)
    assert y.shape == (500,)
    assert np.allclose(w, w_true)

def test_reproducibility():
    w_true = np.array([1.0, 2.0])

    X1, y1, _, _ = generate_regression_dataset(w_true, random_seed=42)
    X2, y2, _, _ = generate_regression_dataset(w_true, random_seed=42)

    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)

def test_no_noise_exact_relationship():
    w_true = np.array([2.0, -1.0])
    b_true = 0.5

    X, y, w, b = generate_regression_dataset(
        w_true=w_true,
        b_true=b_true,
        noise_std=0
    )

    y_expected = X @ w_true + b_true

    assert np.allclose(y, y_expected)

def test_noise_changes_targets():
    w_true = np.array([1.0, 1.0])

    X1, y1, _, _ = generate_regression_dataset(
        w_true=w_true,
        noise_std=0
    )

    X2, y2, _, _ = generate_regression_dataset(
        w_true=w_true,
        noise_std=1
    )

    assert not np.allclose(y1, y2)