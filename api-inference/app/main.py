'''
Main application file (app/main.py)
Here we set up FastAPI, attach a standard /health check
(very important in production for Kubernetes/Docker) and our POST method for inference.
'''

from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictionResult
from app.services import model_service
from app.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-inference")

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

@app.get("/health", status_code=200)
def health_check():
    """
    Service health check.
    """
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResult) # Ответ этой функции должен соответствовать схеме PredictionResult: response_model=PredictionResult
def predict(request: PredictRequest):
    """
    Endpoint for batch model inference.
    """
    if not request.data:
        raise HTTPException(status_code=400, detail="Input data list is empty")
        
    logger.info(f"Received prediction request. Items: {len(request.data)}")
    
    try:
        preds = model_service.predict(request.data)
        return PredictionResult(prediction=preds)
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")