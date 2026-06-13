Python 3.12 | SQLite | Typer | Pytest

# Experiment Tracker

A lightweight experiment tracking system for machine learning projects built with Python and SQLite.

The project provides a simple API for logging experiments, hyperparameters, metrics, and execution status. It uses a context manager interface to automatically track successful and failed runs and includes a command-line interface for inspecting stored experiments.

## Features

* Experiment registration and management
* Run tracking with automatic status updates
* Hyperparameter logging
* Metric logging
* Persistent storage using SQLite
* Context manager API for safe execution tracking
* Command-line interface for experiment inspection
* Automated test suite with pytest

## Project Structure

```text
experiment-tracker/
├── pyproject.toml
├── README.md
├── src/
│   └── experiment_tracker/
│       ├── __init__.py
│       ├── database.py
│       ├── tracker.py
│       └── cli.py
└── tests/
    └── test_tracker.py
```

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd experiment-tracker

poetry install
```

Or using pip:

```bash
pip install -e .
```

## Quick Start

Create a tracker instance for an experiment:

```python
from experiment_tracker import MLTracker

tracker = MLTracker("Linear Regression Benchmark")
```

Track a training run:

```python
with tracker.start_run(
    run_name="baseline_model",
    source_type="custom"
) as run:

    run.log_param("learning_rate", 0.01)
    run.log_param("batch_size", 32)

    run.log_metric("accuracy", 0.91)
    run.log_metric("loss", 0.18)
```

After successful completion, the run status is automatically set to:

```text
FINISHED
```

If an exception occurs during execution:

```python
try:
    with tracker.start_run(
        run_name="failed_run",
        source_type="custom"
    ) as run:

        run.log_param("learning_rate", 0.01)

        raise RuntimeError("Training failed")

except RuntimeError:
    pass
```

The run status is automatically stored as:

```text
FAILED
```

## Data Model

The application stores data in four SQLite tables.

### experiments

Stores experiment metadata.

| Column     | Type    |
| ---------- | ------- |
| id         | INTEGER |
| name       | TEXT    |
| created_at | TEXT    |

### runs

Stores individual experiment executions.

| Column        | Type    |
| ------------- | ------- |
| id            | TEXT    |
| experiment_id | INTEGER |
| name          | TEXT    |
| status        | TEXT    |
| source_type   | TEXT    |
| start_time    | TEXT    |
| end_time      | TEXT    |

### params

Stores run hyperparameters.

| Column | Type    |
| ------ | ------- |
| id     | INTEGER |
| run_id | TEXT    |
| key    | TEXT    |
| value  | TEXT    |

### metrics

Stores run metrics.

| Column    | Type    |
| --------- | ------- |
| id        | INTEGER |
| run_id    | TEXT    |
| key       | TEXT    |
| value     | REAL    |
| timestamp | TEXT    |

## Command Line Interface

List all experiments:

```bash
python -m experiment_tracker.cli list-exp
```

Display experiment runs:

```bash
python -m experiment_tracker.cli show "Linear Regression Benchmark"
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Current test coverage includes:

* experiment creation
* experiment uniqueness validation
* successful run tracking
* failed run tracking
* logging protection outside active runs

## Design Decisions

### SQLite

SQLite was chosen because it requires no external infrastructure and is sufficient for local experiment tracking.

### Context Manager Interface

The tracker uses Python's context manager protocol (`with`) to ensure that run status is always updated, even when exceptions occur.

### Separation of Concerns

* `tracker.py` contains the user-facing API.
* `database.py` handles persistence and SQL operations.
* `cli.py` provides command-line access.
* `tests/` contains automated validation.

## Future Improvements

Possible extensions include:

* Artifact tracking
* Model versioning
* Dataset version tracking
* REST API
* PostgreSQL support
* Experiment comparison reports
* Visualization dashboard
* Integration with scikit-learn pipelines

## License

MIT License.
