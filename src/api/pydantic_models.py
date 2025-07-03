from pydantic import BaseModel
from typing import List, Optional

# This model defines the structure of the input data for prediction.
# It should reflect the features that your *trained model expects*.
# You'll need to fill this out accurately based on your X_train columns.
# For simplicity, I'm adding a few example features. You MUST expand this
# to include ALL 34 features your model was trained on.

class PredictionFeatures(BaseModel):
    # Example features - REPLACE with your actual feature list and types!
    # Refer to X_train.columns after full_pipeline.fit_transform()
    Recency: float
    Frequency: float
    Monetary: float
    TransactionCount: float
    AvgDailyTransactions: float
    TotalTransactionAmount: float
    AvgTransactionAmount: float
    MaxTransactionAmount: float
    MinTransactionAmount: float
    TotalDeclinedTransactions: float
    AvgDeclinedTransactions: float
    TotalSuccessfulTransactions: float
    AvgSuccessfulTransactions: float
    IsFraudulentTransaction: float # Assuming this is a feature, not the target
    # Add all other numerical features
    # Example for OHE features (they should be 0.0 or 1.0 after processing)
    CurrencyCode_UGX: float
    CountryCode_256: float
    ProductCategory_airtime: float
    ProductCategory_data_bundles: float
    ProductCategory_financial_services: float
    ProductCategory_movies: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_transport: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ChannelId_ChannelId_1: float
    ChannelId_ChannelId_2: float
    ChannelId_ChannelId_3: float
    ChannelId_ChannelId_5: float
    # If there are other OHE columns, add them here

    # You might also have ProviderId_ and ProductId_ columns if you decided to OHE them
    # If they were dropped, don't include them here. If they were OHE, add them.
    # ProviderId_1: float
    # ProviderId_2: float
    # ...

    class Config:
        # This allows the Pydantic model to be created from ORM attributes
        # which can be useful if you're loading from a database, but also good for general config.
        from_attributes = True


class PredictionResponse(BaseModel):
    # This defines the structure of the response from your API
    # It should include the prediction (e.g., risk probability) and potentially other info.
    customer_id: str # You might want to pass the customer ID back
    predicted_risk_probability: float
    risk_category: str # e.g., "Low Risk", "High Risk"
    model_version: str # To indicate which model version served the prediction