import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Adjust import path based on your project structure
# Assuming main.py is in src/api
from src.api.main import app, MLFLOW_MODEL_URI

# Use FastAPI's TestClient
client = TestClient(app)

# Mock the MLflow model loading for tests
@pytest.fixture(scope="module", autouse=True)
def mock_mlflow_model_loading():
    """
    Fixture to mock MLflow model loading.
    This prevents actual model loading during tests, speeding them up and
    making them independent of MLflow server availability.
    """
    with patch('mlflow.pyfunc.load_model') as mock_load_model:
        # Create a mock for the loaded model wrapper
        mock_model_wrapper = MagicMock()

        # Define the behavior of predict for the mock model
        # It should return a 2D array of probabilities (n_samples, 2)
        # where the second column is the probability of the positive class.
        mock_model_wrapper.predict.return_value = np.array([[0.1, 0.9]]) # Example: 90% risk

        mock_load_model.return_value = mock_model_wrapper
        yield # Yield control back to tests
        # Cleanup can go here if needed

def test_read_root():
    """
    Test the health check endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk Prediction API is running!"}

def test_predict_endpoint_success():
    """
    Test the /predict endpoint with valid data.
    Ensure your sample_features match the structure of your PredictionFeatures Pydantic model.
    """
    # Sample data matching your PredictionFeatures Pydantic model
    # YOU MUST REPLACE THESE WITH VALID NUMERIC VALUES FOR ALL YOUR FEATURES
    sample_features = {
        "Recency": 10.0,
        "Frequency": 5.0,
        "Monetary": 100.50,
        "TransactionCount": 20.0,
        "AvgDailyTransactions": 1.0,
        "TotalTransactionAmount": 500.0,
        "AvgTransactionAmount": 25.0,
        "MaxTransactionAmount": 100.0,
        "MinTransactionAmount": 5.0,
        "TotalDeclinedTransactions": 1.0,
        "AvgDeclinedTransactions": 0.05,
        "TotalSuccessfulTransactions": 19.0,
        "AvgSuccessfulTransactions": 0.95,
        "IsFraudulentTransaction": 0.0,
        "CurrencyCode_UGX": 1.0,
        "CountryCode_256": 1.0,
        "ProductCategory_airtime": 0.0,
        "ProductCategory_data_bundles": 1.0,
        "ProductCategory_financial_services": 0.0,
        "ProductCategory_movies": 0.0,
        "ProductCategory_other": 0.0,
        "ProductCategory_ticket": 0.0,
        "ProductCategory_transport": 0.0,
        "ProductCategory_tv": 0.0,
        "ProductCategory_utility_bill": 0.0,
        "ChannelId_ChannelId_1": 1.0,
        "ChannelId_ChannelId_2": 0.0,
        "ChannelId_ChannelId_3": 0.0,
        "ChannelId_ChannelId_5": 0.0,
        # Add all other features here matching PredictionFeatures
        "customer_id": "test_customer_123" # Include customer_id if it's in your Pydantic model
    }

    response = client.post("/predict", json=sample_features)

    assert response.status_code == 200
    response_json = response.json()
    assert "customer_id" in response_json
    assert "predicted_risk_probability" in response_json
    assert "risk_category" in response_json
    assert "model_version" in response_json
    assert response_json["customer_id"] == "test_customer_123"
    assert isinstance(response_json["predicted_risk_probability"], float)
    assert 0 <= response_json["predicted_risk_probability"] <= 1
    assert response_json["risk_category"] in ["Low Risk", "High Risk"]
    assert response_json["model_version"] == MLFLOW_MODEL_URI.split('/')[-1]


def test_predict_endpoint_invalid_input():
    """
    Test the /predict endpoint with invalid data (missing required fields).
    """
    invalid_features = {
        "Recency": 10.0,
        # Missing many required fields
        "customer_id": "invalid_customer"
    }
    response = client.post("/predict", json=invalid_features)
    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation error
    assert "detail" in response.json()
    assert "Input validation error" in response.json()["detail"][0]["msg"] # Check for validation error message