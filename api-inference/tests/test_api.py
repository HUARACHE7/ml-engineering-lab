from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify that healthcheck returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_success():
    """Verify successful inference for valid input data."""
    payload = {
        "data": [
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 0.0, "feature_2": -1.0}
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    json_data = response.json()
    assert "prediction" in json_data
    assert json_data["status"] == "SUCCESS"
    assert len(json_data["prediction"]) == 2
    
    # 3 * f1 + 5 * f2
    # 3*(1) + 5*(2) = 13.0
    assert json_data["prediction"][0] == 13.0
    # 3*(0) + 5*(-1) = -5.0
    assert json_data["prediction"][1] == -5.0

def test_predict_validation_error():
    """Verify that Pydantic automatically rejects invalid data types."""
    payload = {
        "data": [
            {"feature_1": "str_insteadOf_num", "feature_2": 2.0}
        ]
    }
    response = client.post("/predict", json=payload)
    # Code 422 is the standard Unprocessable Entity that Pydantic returns on validation error
    assert response.status_code == 422
    