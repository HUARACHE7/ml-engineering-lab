'''
Data schema definitions (app/schemas.py)
The model won't receive a string instead of a number,
and missing fields will trigger a clear 422 Unprocessable Entity error
before the model code even runs.
'''

from pydantic import BaseModel, Field
from typing import List

class FeaturesInput(BaseModel):
    feature_1: float = Field(..., description="Value of the first feature (f1):", examples=[0.496])
    feature_2: float = Field(..., description="Value of the second feature (f2):", examples=[-0.138])

class PredictRequest(BaseModel):
    data: List[FeaturesInput]

class PredictionResult(BaseModel):
    prediction: List[float]
    status: str = "SUCCESS"