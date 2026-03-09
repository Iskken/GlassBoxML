from glassboxml.data.generators import generate_regression_dataset
from glassboxml.models.linear_regression import LinearRegression

w_true = [2.5, -1.3, 0.7]
b_true = 0.5

X, y, w_true, b_true = generate_regression_dataset(
    w_true,
    b_true=b_true,
    n_samples=1000
)

model = LinearRegression()
model.fit_gradient_descent(X, y, 1000, 0.11)

print("X shape:", X.shape)
print("y shape:", y.shape)