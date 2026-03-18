import numpy as np
from glassboxml.data.generators import generate_classification_dataset
from glassboxml.models.logistic_regression import LogisticRegression
import matplotlib.pyplot as plt

w_true = np.array([2.5])
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

plt.figure()

plt.plot(model.losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Convergence')

plt.grid(True)
plt.show()