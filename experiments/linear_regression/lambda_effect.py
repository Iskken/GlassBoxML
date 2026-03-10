from glassboxml.data.generators import generate_regression_dataset
from glassboxml.models.linear_regression import LinearRegression
from glassboxml.losses.regularization import L1Regularization, L2Regularization

w_true = [2.54321, 1.23456, 0.98765]
b_true = 0.54321

X, y, w_true, b_true = generate_regression_dataset(
    w_true,
    b_true=b_true,
    n_samples=1000,
    noise_std=0.3
)
for l in [0, 0.01, 0.1, 1, 10]:
    print(f"Lambda: {l}")
    gradientDescentModel = LinearRegression(regularization=L2Regularization(lambda_=l))
    gradientDescentModel.fit_gradient_descent(X, y, epochs=1000, learning_rate=0.01)
    print(gradientDescentModel.w, gradientDescentModel.b)