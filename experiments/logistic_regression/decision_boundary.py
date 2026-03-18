import numpy as np
from glassboxml.data.generators import generate_classification_dataset
from glassboxml.models.logistic_regression import LogisticRegression
import matplotlib.pyplot as plt

w_true = np.array([-2.5])
b_true = 0.0

X, y, _, _ = generate_classification_dataset(
    w_true,
    b_true=b_true,
    n_samples=10,
    noise_std=0.1,
    random_seed=42
)

print(X)
print(y)

weights = []

model = LogisticRegression()
model.fit(X, y, epochs=1000, learning_rate=0.06)

print(model.w, model.b)

y_pred = model.predict(X)

x_vals = np.linspace(min(X), max(X), 100) #Creates 100 evenly spaced values between the minimum and maximum of X for plotting the decision boundary
z = model.w[0] * x_vals + model.b # Computes the linear combination of inputs and weights for the decision boundary
y_probs = 1 / (1 + np.exp(-z)) # Applies the sigmoid function to the linear combination to get predicted probabilities for the decision boundary
x_boundary = -model.b / model.w[0] # Calculates the x-value where the decision boundary occurs (where predicted probability is 0.5) by setting the linear combination to zero and solving for x

plt.figure()

plt.scatter(X, y, label="Data")
plt.plot(x_vals, y_probs, color='red', label='Sigmoid')

plt.axvline(x=x_boundary, color='green', linestyle='--', label='Boundary')

y_pred_probs = model.predict_proba(X)

plt.scatter(X, y_pred_probs, color='orange', label="Predicted probs")

plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.title("Logistic Regression Fit")

plt.show()