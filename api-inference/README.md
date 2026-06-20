# API Inference

A lightweight FastAPI service for machine learning inference.

This project exposes a small REST API that validates incoming data with Pydantic, runs predictions, and returns structured results in JSON. It is designed as a practical backend-style ML project and can be used as a foundation for deploying a trained model behind an HTTP interface.

The current MVP keeps the inference logic simple and reproducible, while the project structure is ready for replacing the placeholder predictor with a persisted model loaded from `models/trained_model.pkl`.

---

## Purpose

This project is part of the larger **ML Engineering Lab** monorepository.

It focuses on the engineering side of model deployment:

- API design
- request validation
- inference workflow
- error handling
- Docker-based packaging
- automated testing

The goal is to build a clean and understandable inference service that can be extended into a production-ready microservice later.

---

## Features

- FastAPI application
- `/health` endpoint for service checks
- `/predict` endpoint for batch inference
- Pydantic validation for inputs and outputs
- Structured JSON responses
- Logging for incoming requests and errors
- Dockerfile for containerization
- docker-compose setup
- pytest-based API tests

---

## Project Structure

```text
api-inference/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── services.py
│   └── config.py
├── models/
│   └── trained_model.pkl
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Architecture

The service is split into small, focused modules:

| Module | Responsibility |
|----------|----------|
| `main.py` | FastAPI app, routes, error handling |
| `schemas.py` | Input and output validation |
| `services.py` | Inference logic |
| `config.py` | Paths and runtime settings |
| `tests/` | API validation tests |

The separation keeps the code easy to extend and makes the inference layer independent from the API layer.

---

## API Endpoints

### `GET /health`

Checks whether the service is running.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Accepts a batch of validated feature records and returns predictions.

Example request:

```json
{
  "data": [
    {
      "feature_1": 1.0,
      "feature_2": 2.0
    },
    {
      "feature_1": 0.0,
      "feature_2": -1.0
    }
  ]
}
```

Example response:

```json
{
  "prediction": [13.0, -5.0],
  "status": "SUCCESS"
}
```

If the payload is invalid, Pydantic returns a `422 Unprocessable Entity` response before the prediction code runs.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd api-inference

pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

---

## Quick Start

Run the server locally:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

Send a prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{
    "data": [
      {"feature_1": 1.0, "feature_2": 2.0}
    ]
  }'
```

---

## Running with Docker

Build the image:

```bash
docker build -t api-inference .
```

Run the container:

```bash
docker run -p 8000:8000 api-inference
```

Or start the service with Docker Compose:

```bash
docker compose up --build
```

---

## Testing

Run the test suite:

```bash
pytest
```

Current tests cover:

- health check endpoint
- successful inference request
- validation errors for incorrect input types

---

## Design Decisions

### FastAPI

FastAPI was chosen because it makes it easy to define HTTP endpoints, validate data, and generate API documentation automatically.

### Pydantic Schemas

Schemas protect the service from malformed input and keep the contract between client and server explicit.

### Small Service Modules

The code is intentionally split into small files so that request handling, validation, and inference logic remain separate.

### Docker Support

Containerization makes the project easy to run in a consistent environment.

---

## Future Improvements

Possible extensions include:

- loading a real persisted model from `models/trained_model.pkl`
- adding authentication
- adding request batching controls
- adding model versioning
- health checks for model readiness
- metrics and structured logging
- async inference support
- integration with the Experiment Tracker project

---

## Related Projects

This service is part of the **ML Engineering Lab** monorepository.

Related projects include:

- Experiment Tracker
- Validation Framework
- Dataset Analyzer
- Pipeline Framework
- Task Scheduler
- LRU Cache
- Matrix Library

Together, these projects cover the main engineering skills required for ML Engineering and MLOps.

---

## License

MIT License.
