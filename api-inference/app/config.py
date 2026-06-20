from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "trained_model.pkl"

APP_TITLE = os.getenv("APP_TITLE", "ML Model Inference API")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "A FastAPI service for machine learning inference"
)
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
