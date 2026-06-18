# Validation Framework

A lightweight machine learning validation framework built from scratch with Python, NumPy, and pandas.

The project implements the core ideas behind model validation and cross-validation without relying on scikit-learn internals. Its primary goal is to understand how validation pipelines work under the hood while practicing software architecture, testing, and object-oriented design.

This project is part of the larger **ML Engineering Lab** monorepository and integrates with the Experiment Tracker project for experiment logging and result comparison.

---

## Features

- Train/Test Split
- KFold cross-validation
- StratifiedKFold cross-validation
- Custom metrics
- Model-agnostic validation engine
- Optional feature scaling
- Experiment Tracker integration
- CLI leaderboard
- Automated tests with pytest

---

## Project Structure

```text
validation-framework/
├── pyproject.toml
├── README.md
├── src/
│   └── val_framework/
│       ├── __init__.py
│       ├── cli.py
│       ├── engine.py
│       ├── evaluators.py
│       ├── models.py
│       └── splits.py
└── tests/
    └── test_pipeline.py
```

---

## Architecture

The framework consists of four main components:

| Module | Responsibility |
|----------|----------|
| `splits.py` | Dataset splitting strategies |
| `models.py` | ML models implemented from scratch |
| `evaluators.py` | Metrics and preprocessing |
| `engine.py` | Validation orchestration |

The validation engine coordinates the entire workflow:

1. Split data into folds.
2. Train the model.
3. Generate predictions.
4. Calculate metrics.
5. Log results to the Experiment Tracker.
6. Display results through the CLI.

---

## Quick Start

### Create a dataset

```python
import numpy as np
import pandas as pd

np.random.seed(42)

X = np.random.rand(30, 2)
y = 2 * X[:, 0] + 3 * X[:, 1]

df = pd.DataFrame(
    X,
    columns=["feature_1", "feature_2"]
)

df["target"] = y
```

### Configure validation

```python
from val_framework.engine import ValidationEngine
from val_framework.models import MyLinearRegression
from val_framework.splits import custom_kfold
from val_framework.evaluators import (
    MyStandardScaler,
    r2_score_custom
)

model = MyLinearRegression(
    eta=0.01,
    n_iter=100
)

engine = ValidationEngine(
    model=model,
    splitter=lambda df: custom_kfold(
        df,
        k=3,
        random_state=42
    ),
    scaler=MyStandardScaler(),
    metrics={
        "R2_score": r2_score_custom
    }
)
```

### Run validation

```python
engine.run_validation(
    df=df,
    target_column="target",
    feature_columns=[
        "feature_1",
        "feature_2"
    ],
    run_name_prefix="linear_regression"
)
```

---

## Experiment Tracking

The framework can be connected to the Experiment Tracker project:

```text
MLOps/experiment-tracker/
```

Each validation run can automatically store:

- run name
- parameters
- metrics
- execution status
- timestamps

This enables reproducible experimentation and historical comparison of model performance.

---

## CLI Leaderboard

Show the best runs for a metric:

```bash
python -m val_framework.cli \
    --experiment "Pytest Val Framework" \
    --metric R2_score
```

Sort ascending for error metrics:

```bash
python -m val_framework.cli \
    --experiment "Pytest Val Framework" \
    --metric MAE \
    --asc
```

---

## Testing

Run all tests:

```bash
pytest tests/
```

Covered functionality:

- Train/Test Split
- KFold
- StratifiedKFold
- Custom metrics
- Tracker integration
- End-to-end validation pipeline

---

## What This Project Teaches

This project focuses on practical ML engineering skills:

- Object-oriented design
- Framework architecture
- Machine learning validation
- Numerical computing
- Testing
- Reproducibility
- Integration between independent systems

It also provides a deeper understanding of how tools like scikit-learn implement model evaluation internally.

---

## Future Improvements

- GroupKFold
- TimeSeriesSplit
- Additional metrics
- HTML reports
- Visualization tools
- Configuration-driven pipelines

---

## License

MIT License.