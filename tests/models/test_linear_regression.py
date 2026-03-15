import numpy as np
from glassboxml.data.generators import generate_regression_dataset
from glassboxml.models.linear_regression import LinearRegression

def test_closed_form_recovers_weights():
    # Generate synthetic data
    w_true = np.array([2.54321])
    b_true = 0.54321

    X, y, _, _ = generate_regression_dataset(
        w_true,
        b_true=b_true,
        n_samples=1000,
        noise_std=0.3
    )

    # Test closed-form solution
    model_closed_form = LinearRegression()
    model_closed_form.fit_closed_form(X, y)
    assert np.allclose(model_closed_form.w, w_true, atol=0.1), f"Closed-form weights {model_closed_form.w} not close to true weights {w_true}"

def test_gradient_descent_recovers_weights():
    # Generate synthetic data
    w_true = np.array([2.54321, 1.23456, 0.98765])
    b_true = 0.54321

    X, y, _, _ = generate_regression_dataset(
        w_true,
        b_true=b_true,
        n_samples=1000,
        noise_std=0.3
    )

    # Test gradient descent solution
    model_gradient_descent = LinearRegression()
    model_gradient_descent.fit_gradient_descent(X, y, epochs=1000, learning_rate=0.01)
    assert np.allclose(model_gradient_descent.w, w_true, atol=0.1), f"Gradient descent weights {model_gradient_descent.w} not close to true weights {w_true}"
    assert np.isclose(model_gradient_descent.b, b_true, atol=0.1), f"Gradient descent bias {model_gradient_descent.b} not close to true bias {b_true}"