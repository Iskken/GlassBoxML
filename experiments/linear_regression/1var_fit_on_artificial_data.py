from glassboxml.data.generators import generate_regression_dataset
from glassboxml.models.linear_regression import LinearRegression

w_true = [2.54321]
b_true = 0.5

X, y, w_true, b_true = generate_regression_dataset(
    w_true,
    b_true=b_true,
    n_samples=1000,
    noise_std=0.0
)


closedFormModel = LinearRegression()
closedFormModel.fit_closed_form(X, y)
closedFormModel.plot(X, y)
print(closedFormModel.w, closedFormModel.b)

gradientDescentModel = LinearRegression()
gradientDescentModel.fit_gradient_descent(X, y, epochs=1000, learning_rate=0.01)
gradientDescentModel.plot(X, y)
print(gradientDescentModel.w, gradientDescentModel.b)