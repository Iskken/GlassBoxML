import numpy as np
from glassboxml.data.generators import generate_classification_dataset
from glassboxml.models.logistic_regression import LogisticRegression

def test_logistic_regression_direction_similarity():
    # Generate synthetic data
    w_true = np.array([2.54321, 1.23456, 0.98765])
    b_true = 0.54321

    X, y, _, _ = generate_classification_dataset(
        w_true,
        b_true=b_true,
        n_samples=1000,
        noise_std=0.1
    )

    # Test direction similarity of learned weights to true weights
    model = LogisticRegression()
    model.fit(X, y, epochs=1000, learning_rate=0.01)
    cos_sim = np.dot(model.w, w_true) / (
        np.linalg.norm(model.w) * np.linalg.norm(w_true)
    )

    assert cos_sim > 0.99

def test_logistic_regression_prediction_accuracy():
    # Generate synthetic data
    w_true = np.array([2.54321, 1.23456, 0.98765])
    b_true = 0.54321

    X, y, _, _ = generate_classification_dataset(
        w_true,
        b_true=b_true,
        n_samples=1000,
        noise_std=0.1
    )

    # Test prediction accuracy of the model
    model = LogisticRegression()
    model.fit(X, y, epochs=1000, learning_rate=0.01)
    y_pred = model.predict(X)

    accuracy = np.mean(y_pred == y)
    assert accuracy > 0.95