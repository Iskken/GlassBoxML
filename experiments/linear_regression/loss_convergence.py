from glassboxml.models.linear_regression import LinearRegression
import matplotlib.pyplot as plt

# Example dataset
data = [(1,2), (2,3), (3,5), (4,4)]

model = LinearRegression()
model.fit_gradient_descent(data, 1000, 0.11)

plt.plot(model.losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()