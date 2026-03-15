from glassboxml.data.generators import generate_regression_dataset
from glassboxml.models.linear_regression import LinearRegression
from glassboxml.losses.regularization import L1Regularization, L2Regularization
import numpy as np

def test_l1_loss():
    w = np.array([1.0, -2.0, 3.0])
    reg = L1Regularization(lambda_=0.5)

    loss = reg.loss(w)

    expected = 0.5 * (1 + 2 + 3)

    assert np.isclose(loss, expected)

def test_l1_gradient():
    w = np.array([1.0, -2.0, 3.0])
    reg = L1Regularization(lambda_=0.5)

    grad = reg.gradient(w)

    expected = 0.5 * np.sign(w)

    assert np.allclose(grad, expected)


def test_l2_loss():
    w = np.array([1.0, -2.0, 3.0])
    reg = L2Regularization(lambda_=0.5)

    loss = reg.loss(w)

    expected = 0.5 * (1**2 + (-2)**2 + 3**2)

    assert np.isclose(loss, expected)

def test_l2_gradient():
    w = np.array([1.0, -2.0, 3.0])
    reg = L2Regularization(lambda_=0.5)

    grad = reg.gradient(w)

    expected = 2 * 0.5 * w

    assert np.allclose(grad, expected)

def test_zero_lambda():
    w = np.array([1.0, -2.0, 3.0])

    l1 = L1Regularization(lambda_=0)
    l2 = L2Regularization(lambda_=0)

    assert l1.loss(w) == 0
    assert l2.loss(w) == 0

    assert np.allclose(l1.gradient(w), np.zeros_like(w))
    assert np.allclose(l2.gradient(w), np.zeros_like(w))