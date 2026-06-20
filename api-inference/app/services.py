'''
Inference service (app/services.py)
Class that loads the model once at server startup and
provides a method for prediction.
'''

from app.schemas import FeaturesInput
from typing import List

class ModelService:
    def __init__(self):
        # self.model = joblib.load("models/model.pkl")
        self.model = None 

    def predict(self, features_list: List[FeaturesInput]) -> List[float]:
        """
        Accepts validated Pydantic models, converts them to an array
        and runs them through model inference.
        """
        predictions = []
        for features in features_list:
            # Simulating our linear regression (w1=3, w2=5, b=0)
            pred = 3 * features.feature_1 + 5 * features.feature_2
            predictions.append(float(pred))
        return predictions

model_service = ModelService()
