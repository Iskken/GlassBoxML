from linear_regression import LinearRegression

# Example dataset
data = [(1,2), (2,3), (3,5), (4,4)]

model = LinearRegression()
model.fit_gradient_descent(data, 1000, 0.1)
model.plot(data)