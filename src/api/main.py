import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import mlflow
from mlflow.pyfunc import PythonModel
import numpy as np

# Assuming pydantic_models.py is in the same directory
from .pydantic_models import PredictionFeatures, PredictionResponse

# --- MLflow Model Loading ---
# Define the MLflow model URI. This will depend on how you registered your model.
# Use the model name you used in the "registered_model_name" argument during registration.
# For example, if you registered it as "CreditRisk_HighRisk_Model"
MLFLOW_MODEL_URI = "models:/CreditRisk_HighRisk_Model/Production" # Or /Staging, or /latest, /1, /2 etc.

# Load the model globally when the FastAPI app starts
# This ensures the model is loaded only once, not on every request
try:
    print(f"Loading MLflow model from: {MLFLOW_MODEL_URI}")
    # Load the PythonModel wrapper
    loaded_model_wrapper = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
    print("MLflow model loaded successfully.")

    # Access the underlying sklearn model if it's stored within a pyfunc model
    # If your best_model was directly logged, it might be accessible via .unwrap_python_model()
    # or you might need to inspect the artifact path of the registered model.
    # For now, let's assume the pyfunc model's predict method handles everything.

except Exception as e:
    print(f"Error loading MLflow model: {e}")
    loaded_model_wrapper = None # Set to None if loading fails, handle in endpoint

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk of customers.",
    version="1.0.0"
)

# --- Utility Function for Preprocessing (Crucial!) ---
# This function replicates the exact preprocessing your pipeline does,
# but for a single input or a small batch from the API.
# This is simplified. In a real-world scenario, you might re-use
# your entire sklearn pipeline object if it was saved and loaded properly.
# However, if your pipeline involves custom transformers that aren't part of the
# logged MLflow model's `pyfunc` signature (which typically wraps the *final* sklearn model),
# you might need to explicitly load and apply the pipeline here *before* prediction.

# For this example, let's assume the `loaded_model_wrapper` (if it's a pyfunc model)
# handles any necessary transformation by expecting raw input, or that the features
# received by the API are already in the format expected by the *final* sklearn model.

# If your `loaded_model_wrapper` is truly just the final sklearn model,
# you'll need to re-implement the preprocessing here.
# For instance, if your model expects one-hot encoded columns, and the API
# receives raw categorical data, you need to apply the OHE logic.
# The best way is to save the *entire pipeline* as one MLflow model.

# For now, we'll assume PredictionFeatures directly maps to what the model expects (numeric).
# If it doesn't, this is where you'd re-apply parts of your src/feature_engineering.py logic.

# Example placeholder for a more complex preprocessing:
# def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
#     # Load your full pipeline or relevant parts (e.g., CustomEncoder, FeatureScaler)
#     # This requires saving the entire pipeline during training, not just the final model.
#     # For this example, we'll assume the input features are already processed
#     # as per the X_train used for model fitting.
#     return input_data

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    Health check endpoint.
    """
    return {"message": "Credit Risk Prediction API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(features: PredictionFeatures):
    """
    Predict the credit risk probability for a new customer.
    """
    if loaded_model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    try:
        # Convert Pydantic model to Pandas DataFrame
        # Each feature in PredictionFeatures must exactly match a column expected by the model
        input_df = pd.DataFrame([features.model_dump()])

        # --- IMPORTANT PREPROCESSING STEP HERE ---
        # If your model was trained on scaled data, you *must* scale the input_df here.
        # If it was trained on OHE data, and your API receives raw categories, you *must* OHE here.
        # The best practice is to save the *entire sklearn pipeline* as one MLflow model.
        # Then, loaded_model_wrapper.predict(input_df) would handle it.

        # For the current setup, we assume PredictionFeatures already has the final processed numeric features.
        # If your original ProviderId/ProductId were dropped, ensure they are NOT in PredictionFeatures.
        # If OHE columns were created (e.g., CurrencyCode_UGX), ensure PredictionFeatures has those.
        # The `loaded_model_wrapper` is a pyfunc model, it's designed to accept DataFrame and process it.
        # So, the column names and order in input_df MUST match the X_train used for training.

        # Get the feature names the model expects (e.g., from the signature or inspection)
        # This is a common point of failure: feature mismatch.
        # If you saved the *full pipeline* as an MLflow model, its predict method
        # would handle transformations. If you only saved the final classifier,
        # you need to apply transformations here.

        # A robust way is to load the *trained pipeline* not just the final classifier.
        # However, if your MLflow model logs the entire `full_pipeline` directly,
        # then `loaded_model_wrapper.predict` would handle the `input_df` as is.

        # For now, let's ensure the input_df has the correct column order and names
        # by re-indexing based on the training data columns (if known).
        # You would typically get this from a saved feature list or the model's signature.
        # Let's assume the `PredictionFeatures` order matches `X_train` order after preprocessing.

        # Make prediction
        # The `predict` method of the pyfunc model usually takes a pandas DataFrame.
        # It's crucial that `input_df` has the same columns and order as the training data.
        # If `loaded_model_wrapper` is just `LogisticRegression`, then you need:
        # `model_output_proba = loaded_model_wrapper.predict_proba(input_df)[:, 1]`
        model_output_proba = loaded_model_wrapper.predict(input_df)

        # mlflow.pyfunc.PythonModel.predict usually returns an array, check its structure.
        # If it's a binary classifier, it often returns probabilities of shape (n_samples, n_classes)
        # We need the probability of the positive class (index 1).
        if isinstance(model_output_proba, np.ndarray) and model_output_proba.ndim == 2:
            risk_probability = float(model_output_proba[:, 1][0])
        elif isinstance(model_output_proba, np.ndarray) and model_output_proba.ndim == 1:
            # If the model predict returns single values (e.g., for regression or direct probability)
            risk_probability = float(model_output_proba[0])
        else:
            # Fallback or error if prediction output format is unexpected
            raise ValueError(f"Unexpected model prediction output format: {type(model_output_proba)}, shape: {model_output_proba.shape if hasattr(model_output_proba, 'shape') else 'N/A'}")


        # Determine risk category
        if risk_probability >= 0.5: # Example threshold
            risk_category = "High Risk"
        else:
            risk_category = "Low Risk"

        # Get model version from MLflow (if needed)
        # This requires more complex MLflow API interaction if not available directly from loaded_model_wrapper
        model_version = MLFLOW_MODEL_URI.split('/')[-1] # Simple parse, not robust

        return PredictionResponse(
            customer_id=features.customer_id, # Assuming customer_id is part of PredictionFeatures
            predicted_risk_probability=risk_probability,
            risk_category=risk_category,
            model_version=model_version
        )

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Input validation error: {e.errors()}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Prediction error due to invalid data format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")