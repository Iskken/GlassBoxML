# GlassBoxML - AI Coding Instructions

## Project Overview
GlassBoxML is a from-scratch implementation of classic machine learning algorithms with focus on interpretability and educational value. Components are organized by function (models, data, losses, metrics, optimizers, utils).

## Architecture & Data Flow

### Current Implementation
- **Models** (`glassboxml/models/`): Currently implements `LinearRegression` with two solving approaches
  - `fit_closed_form()`: Analytical solution using matrix math
  - `fit_gradient_descent()`: Iterative optimization with convergence checking
- **Data Module** (`glassboxml/data/generators.py`): Synthetic regression datasets with ground-truth weights, configurable noise, and seeds for reproducibility
- **Empty Stubs**: `losses/`, `metrics/`, `optimizers/` are scaffolded but not implemented

### Typical Workflow
1. Generate synthetic data via `generate_regression_dataset(w_true, b_true, n_samples, noise_std, random_seed)`
2. Instantiate model: `model = LinearRegression()`  
3. Fit via either closed-form or gradient descent
4. Use `model.plot(data)` and `model.losses` list for visualization

## Code Patterns & Conventions

### LinearRegression Class Structure
```python
class LinearRegression:
    def __init__(self):
        self.w = 0  # weights
        self.b = 0  # bias
        self.losses = []  # track convergence
        self.epsilon = 1e-6  # convergence threshold
```

- State tracked in instance variables; losses accumulated during training
- Gradient descent uses early stopping when gradients fall below `epsilon`
- Print debugging included in fit methods (epochs, weights) - keep for visibility

### Data Representation
- Datasets as tuples: `[(x1, y1), (x2, y2), ...]` in initial code
- Numpy arrays `(X, y)` format in generators: X shape `(n_samples, n_features)`, y shape `(n_samples,)`
- Always return `(X, y, w_true, b_true)` from generators for validation

### Testing Pattern
- Tests in `tests/test_*.py` import directly from `glassboxml` submodules
- Use concrete data for tests (small synthetic datasets)
- Example: `test_linear_regression.py` uses `[(1,2), (2,3), (3,5), (4,4)]`

## Project-Specific Approaches

### Why Closed-Form + Gradient Descent?
The LinearRegression class intentionally implements both methods to show analytical solutions vs optimization. When adding new models:
- Start with simpler closed-form if available
- Use gradient descent as the general solving approach
- Document why a model chooses one over the other

### Noise Generation
Always use `np.random.seed(random_seed)` at function start for reproducibility - critical for benchmarking.

### Matrix Operations
Use `@` operator for matrix multiplication (not `np.dot`): `y = X @ w_true + b_true`

## Development Workflows

### Running Tests
```bash
# From project root
python -m pytest tests/
# Or individual test
python -m pytest tests/test_linear_regression.py
```

### Using Experiments Notebook  
`experiments/1var_linear_regession.ipynb` is for interactive exploration of single-variable regression. Add more notebooks here for new model experiments before adding to main codebase.

### Debugging Model Convergence
- Check `model.losses` list after training - should be monotonic or plateau
- Verify `learning_rate` is appropriate (small enough to converge, large enough to progress)
- Use `fit_gradient_descent()` print statements to inspect weight updates every 20 epochs

## Integration Points & Dependencies

### External Libraries
- `numpy`: All numeric computation (matrix ops, random generators)
- `matplotlib.pyplot`: Visualization in model `plot()` methods
- No other major dependencies - intentional for educational clarity

### Module Structure
- Empty `__init__.py` files in submodules - import directly: `from glassboxml.models.linear_regression import LinearRegression`
- Not a packaged library yet (no setup.py) - designed for local import

## Key Files by Purpose
- [linear_regression.py](../glassboxml/models/linear_regression.py) - Primary model reference
- [generators.py](../glassboxml/data/generators.py) - Data generation patterns
- [test_linear_regression.py](../tests/test_linear_regression.py) - Testing patterns
- [1var_linear_regession.ipynb](../experiments/1var_linear_regession.ipynb) - Experimental workflow
