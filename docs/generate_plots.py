"""Regenerate the README plots in docs/images/. Run with: python docs/generate_plots.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glassboxml.data.generators import (
    generate_classification_dataset,
    generate_regression_dataset,
)
from glassboxml.models.decision_tree import DecisionTree
from glassboxml.models.knn import KNNClassifier
from glassboxml.models.linear_regression import LinearRegression
from glassboxml.models.logistic_regression import LogisticRegression

OUT_DIR = "docs/images"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_linear_regression():
    X, y, w_true, b_true = generate_regression_dataset(
        w_true=[2.5], b_true=1.0, n_samples=200, noise_std=0.6,
    )
    model = LinearRegression()
    model.fit_gradient_descent(X, y, epochs=500, learning_rate=0.1)

    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 60)
    y_line = float(model.w[0]) * x_line + float(model.b)

    plt.figure(figsize=(6, 4.5))
    plt.scatter(X[:, 0], y, alpha=0.5, label="data")
    plt.plot(x_line, y_line, color="crimson", linewidth=2, label="fitted line")
    plt.title("Linear Regression (gradient descent)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/linear_regression.png", dpi=150)
    plt.close()


def plot_logistic_regression():
    X, y, w_true, b_true = generate_classification_dataset(
        w_true=[1.5, -2.0], b_true=0.5, n_samples=300, noise_std=0.5,
    )
    model = LogisticRegression()
    model.fit(X, y, epochs=1000, learning_rate=0.1)

    x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 60)
    x1_boundary = -(float(model.w[0]) * x0_range + float(model.b)) / float(model.w[1])

    plt.figure(figsize=(6, 4.5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.6, edgecolors="k", linewidths=0.3)
    plt.plot(x0_range, x1_boundary, color="black", linewidth=2, linestyle="--", label="decision boundary")
    plt.title("Logistic Regression")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/logistic_regression.png", dpi=150)
    plt.close()


def plot_decision_tree():
    X, y, w_true, b_true = generate_classification_dataset(
        w_true=[1.5, -2.0], b_true=0.5, n_samples=300, noise_std=0.5,
    )
    model = DecisionTree(max_depth=4)
    model.fit(X, y)

    x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 200),
        np.linspace(x1_min, x1_max, 200),
    )
    grid = np.column_stack([xx0.ravel(), xx1.ravel()])
    zz = model.predict(grid).reshape(xx0.shape)

    plt.figure(figsize=(6, 4.5))
    plt.contourf(xx0, xx1, zz, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.title(f"Decision Tree (max_depth={model.max_depth})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/decision_tree.png", dpi=150)
    plt.close()


def plot_knn():
    X, y, w_true, b_true = generate_classification_dataset(
        w_true=[1.5, -2.0], b_true=0.5, n_samples=300, noise_std=0.5,
    )
    model = KNNClassifier(k=5)
    model.fit(X, y)

    x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 200),
        np.linspace(x1_min, x1_max, 200),
    )
    grid = np.column_stack([xx0.ravel(), xx1.ravel()])
    zz = model.predict(grid).reshape(xx0.shape)

    plt.figure(figsize=(6, 4.5))
    plt.contourf(xx0, xx1, zz, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.title(f"K-Nearest Neighbors (k={model.k})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/knn.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_linear_regression()
    plot_logistic_regression()
    plot_decision_tree()
    plot_knn()
    print(f"Saved plots to {OUT_DIR}/")
