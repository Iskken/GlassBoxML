from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from glassboxml.data.generators import (
    generate_classification_dataset,
    generate_regression_dataset,
)
from glassboxml.metrics.regression import rmse as rmse_metric
from glassboxml.models.decision_tree import DecisionTree
from glassboxml.models.linear_regression import LinearRegression
from glassboxml.models.logistic_regression import LogisticRegression

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="GlassBoxML Demo")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class RunRequest(BaseModel):
    algorithm: str
    n_samples: int = 200
    noise_std: float = 0.5
    epochs: int = 500
    learning_rate: float = 0.1
    max_depth: int = 3


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


@app.post("/run")
def run(req: RunRequest):
    if req.algorithm == "linear_regression":
        return _linear_regression(req)
    if req.algorithm == "logistic_regression":
        return _logistic_regression(req)
    if req.algorithm == "decision_tree":
        return _decision_tree(req)
    return {"error": f"Unknown algorithm: {req.algorithm}"}


def _linear_regression(req: RunRequest):
    X, y, _, _ = generate_regression_dataset(
        w_true=[2.5], b_true=1.0,
        n_samples=req.n_samples, noise_std=req.noise_std,
    )
    model = LinearRegression()
    model.fit_gradient_descent(X, y, epochs=req.epochs, learning_rate=req.learning_rate)

    y_pred = model.predict(X)
    error = float(rmse_metric(y, y_pred))

    x_line = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), 60)
    y_line = float(model.w[0]) * x_line + float(model.b)

    return {
        "scatter": {"x": X[:, 0].tolist(), "y": y.tolist()},
        "line": {"x": x_line.tolist(), "y": y_line.tolist()},
        "metrics": {
            "rmse": round(error, 4),
            "learned w": round(float(model.w[0]), 4),
            "learned b": round(float(model.b), 4),
            "true w": 2.5,
            "true b": 1.0,
        },
    }


def _logistic_regression(req: RunRequest):
    X, y, _, _ = generate_classification_dataset(
        w_true=[1.5, -2.0], b_true=0.5,
        n_samples=req.n_samples, noise_std=req.noise_std,
    )
    model = LogisticRegression()
    model.fit(X, y, epochs=req.epochs, learning_rate=req.learning_rate)

    y_pred = model.predict(X)
    accuracy = float(np.mean(y_pred == y))

    # Decision boundary in 2D: w0*x0 + w1*x1 + b = 0  =>  x1 = -(w0*x0 + b) / w1
    x0_range = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), 60)
    if abs(float(model.w[1])) > 1e-10:
        x1_boundary = -(float(model.w[0]) * x0_range + float(model.b)) / float(model.w[1])
    else:
        x1_boundary = np.zeros_like(x0_range)

    return {
        "scatter": {
            "x1": X[:, 0].tolist(),
            "x2": X[:, 1].tolist(),
            "labels": y.tolist(),
        },
        "boundary": {"x1": x0_range.tolist(), "x2": x1_boundary.tolist()},
        "metrics": {
            "accuracy": f"{round(accuracy * 100, 1)}%",
            "learned w1": round(float(model.w[0]), 4),
            "learned w2": round(float(model.w[1]), 4),
            "learned b": round(float(model.b), 4),
        },
    }


def _decision_tree(req: RunRequest):
    X, y, _, _ = generate_classification_dataset(
        w_true=[1.5, -2.0], b_true=0.5,
        n_samples=req.n_samples, noise_std=req.noise_std,
    )
    model = DecisionTree(max_depth=req.max_depth)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = float(np.mean(y_pred == y))

    return {
        "scatter": {
            "x1": X[:, 0].tolist(),
            "x2": X[:, 1].tolist(),
            "labels": y.tolist(),
            "predicted": y_pred.tolist(),
        },
        "metrics": {
            "accuracy": f"{round(accuracy * 100, 1)}%",
            "max_depth": req.max_depth,
        },
    }
