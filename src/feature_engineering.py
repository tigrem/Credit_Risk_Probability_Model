# src/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
# We'll use xverse for WoE and IV later, but for now, let's set up the basic pipeline.
# from xverse.transformer import WOE
# from woe.feature_engineer import WoeEncoder


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to extract time-based features from 'TransactionStartTime'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure 'TransactionStartTime' column exists before converting
        if 'TransactionStartTime' not in X_copy.columns:
            raise ValueError("'TransactionStartTime' column not found for FeatureExtractor.")

        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        X_copy['TransactionHour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['TransactionDay'] = X_copy['TransactionStartTime'].dt.day
        X_copy['TransactionMonth'] = X_copy['TransactionStartTime'].dt.month
        X_copy['TransactionYear'] = X_copy['TransactionStartTime'].dt.year
        return X_copy.drop('TransactionStartTime', axis=1) # Drop original timestamp column


class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to calculate Recency, Frequency, and Monetary (RFM) features.
    Assumes 'TransactionStartTime', 'AccountId', and 'Amount' are available.
    """
    def __init__(self, snapshot_date=None):
        # snapshot_date: The date against which recency is calculated.
        # If None, it will be the maximum transaction date in the dataset.
        self.snapshot_date = snapshot_date
        self._fitted_snapshot_date = None

    def fit(self, X, y=None):
        if 'TransactionStartTime' not in X.columns or 'AccountId' not in X.columns or 'Amount' not in X.columns:
            raise ValueError("Required columns 'TransactionStartTime', 'AccountId', 'Amount' not found for RFM calculation during fit.")

        # Ensure 'TransactionStartTime' is datetime for snapshot_date calculation
        X_copy_for_fit = X.copy()
        X_copy_for_fit['TransactionStartTime'] = pd.to_datetime(X_copy_for_fit['TransactionStartTime'])

        if self.snapshot_date is None:
            # Set snapshot date to one day after the latest transaction date in the training data
            self._fitted_snapshot_date = X_copy_for_fit['TransactionStartTime'].max() + pd.Timedelta(days=1)
        else:
            self._fitted_snapshot_date = pd.to_datetime(self.snapshot_date)
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure columns exist before processing
        if 'TransactionStartTime' not in X_copy.columns or 'AccountId' not in X_copy.columns or 'Amount' not in X_copy.columns:
            raise ValueError("Required columns 'TransactionStartTime', 'AccountId', 'Amount' not found for RFM calculation during transform.")

        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')

        # Group by AccountId to calculate RFM
        # Handle potential empty groups or all NaNs for min/max
        rfm_df = X_copy.groupby('AccountId').agg(
            Recency=('TransactionStartTime', lambda date: (self._fitted_snapshot_date - date.max()).days if not date.empty and pd.notna(date.max()) else np.nan),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()

        # Fill NaNs from potential empty groups or all NaNs if necessary for Recency
        # For new accounts not seen in training, Recency will be NaN. This can be handled by imputation later.

        # Merge RFM features back to the original dataframe
        # This will add RFM columns to each transaction based on its AccountId
        # Using a left merge to keep all original transactions
        X_copy = pd.merge(X_copy, rfm_df, on='AccountId', how='left', suffixes=('', '_rfm'))
        return X_copy


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to create aggregate features per customer.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Ensure 'AccountId' exists and 'Amount' is numeric
        if 'AccountId' not in X_copy.columns:
            raise ValueError("'AccountId' column not found for AggregateFeatures.")
        if 'Amount' not in X_copy.columns:
            raise ValueError("'Amount' column not found for AggregateFeatures.")

        # Convert 'Amount' to numeric, coercing errors, ensuring it's ready for aggregation
        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')

        # Aggregate features
        # Filter out NaN 'Amount' values before aggregation for sum/mean/std, count will handle it
        agg_features = X_copy.groupby('AccountId').agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AverageTransactionAmount=('Amount', 'mean'),
            MinTransactionAmount=('Amount', 'min'),
            MaxTransactionAmount=('Amount', 'max'),
            TransactionCount=('TransactionId', 'count'), # Count of transactions, including those with NaN amount
            StdDevTransactionAmount=('Amount', 'std')
        ).reset_index()

        # Merge back to the original dataframe
        X_copy = pd.merge(X_copy, agg_features, on='AccountId', how='left')

        return X_copy


class CustomEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for encoding categorical variables.
    Handles both One-Hot Encoding and Label Encoding.
    """
    def __init__(self, method='onehot', columns=None):
        self.method = method
        self.columns = columns
        self.encoders = {}
        self.fitted_ohe_columns = [] # To store column names after one-hot encoding

    def fit(self, X, y=None):
        # Determine columns to encode if not explicitly provided
        if self.columns is None:
            self.columns = X.select_dtypes(include='object').columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                print(f"Warning: Column '{col}' not found in data during CustomEncoder fit. Skipping.")
                continue

            if self.method == 'label':
                le = LabelEncoder()
                # Fit on non-null unique values to avoid issues with NaN during fit
                # Convert to string to handle mixed types or potential non-string objects
                unique_values = X[col].dropna().astype(str).unique()
                if len(unique_values) > 0:
                    self.encoders[col] = le.fit(unique_values)
                else:
                    self.encoders[col] = None # No values to fit
            elif self.method == 'onehot':
                # For one-hot, `pd.get_dummies` will handle fit during transform by default.
                # We can record the expected columns here if needed, but it's often done during transform.
                # For robust OHE in a pipeline, consider sklearn's OneHotEncoder with handle_unknown='ignore'
                # but pd.get_dummies is simpler for basic use.
                pass
            else:
                raise ValueError(f"Method must be 'onehot' or 'label', got '{self.method}'")
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns is None:
            # If columns were not specified during init, determine them at transform time
            self.columns = X_copy.select_dtypes(include='object').columns.tolist()

        for col in self.columns:
            if col not in X_copy.columns:
                print(f"Warning: Column '{col}' not found in data during CustomEncoder transform. Skipping.")
                continue

            if self.method == 'onehot':
                # pandas get_dummies handles NaN values by default (does not create dummy for NaN unless dummy_na=True)
                X_copy = pd.get_dummies(X_copy, columns=[col], prefix=col, dummy_na=False)
            elif self.method == 'label':
                # Apply transform. Fill NaNs first if they exist in the column, otherwise LabelEncoder will error.
                # Use a mapping based on fitted encoder, and -1 for unseen/NaN values
                if col in self.encoders and self.encoders[col] is not None:
                    # Convert column to string to avoid type issues with map
                    X_copy[col] = X_copy[col].astype(str).map(
                        lambda s: self.encoders[col].transform([s])[0]
                        if s in self.encoders[col].classes_ else -1
                    )
                    # Convert to numeric type, can result in float if -1 is introduced and original was int
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                else:
                    # If encoder was not fitted (e.g., all NaNs in fit), fill with a placeholder
                    X_copy[col] = -1 # Or some other appropriate placeholder
            else:
                pass # Already handled in __init__
        return X_copy


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to handle missing values.
    """
    def __init__(self, strategy='mean', numerical_cols=None, categorical_cols=None):
        self.strategy = strategy
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.imputers = {} # Stores fitted imputers

    def fit(self, X, y=None):
        # Determine numerical and categorical columns if not explicitly provided
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include='object').columns.tolist()

        # Imputer for numerical columns
        numerical_cols_to_impute = [col for col in self.numerical_cols if col in X.columns and X[col].isnull().any()]
        if numerical_cols_to_impute:
            self.imputers['numerical'] = SimpleImputer(strategy=self.strategy)
            self.imputers['numerical'].fit(X[numerical_cols_to_impute])
            self._fitted_numerical_cols = numerical_cols_to_impute # Store which columns were fitted
        else:
            self._fitted_numerical_cols = []

        # Imputer for categorical columns (most frequent)
        categorical_cols_to_impute = [col for col in self.categorical_cols if col in X.columns and X[col].isnull().any()]
        if categorical_cols_to_impute:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            # For categorical, it's safer to impute object types directly
            self.imputers['categorical'].fit(X[categorical_cols_to_impute])
            self._fitted_categorical_cols = categorical_cols_to_impute # Store which columns were fitted
        else:
            self._fitted_categorical_cols = []

        return self

    def transform(self, X):
        X_copy = X.copy()

        # Apply numerical imputation
        if 'numerical' in self.imputers and self._fitted_numerical_cols:
            # Ensure columns exist in X_copy before transforming
            cols_to_transform = [col for col in self._fitted_numerical_cols if col in X_copy.columns]
            if cols_to_transform:
                X_copy[cols_to_transform] = self.imputers['numerical'].transform(X_copy[cols_to_transform])

        # Apply categorical imputation
        if 'categorical' in self.imputers and self._fitted_categorical_cols:
            # Ensure columns exist in X_copy before transforming
            cols_to_transform = [col for col in self._fitted_categorical_cols if col in X_copy.columns]
            if cols_to_transform:
                X_copy[cols_to_transform] = self.imputers['categorical'].transform(X_copy[cols_to_transform])

        return X_copy


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for scaling numerical features.
    """
    def __init__(self, method='standard', columns=None):
        self.method = method
        self.columns = columns
        self.scaler = None
        self._fitted_columns = []

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()

        # Filter out columns that are not in the current DataFrame or are constant
        cols_to_fit = []
        for col in self.columns:
            if col in X.columns and X[col].nunique() > 1: # Only scale non-constant numerical columns
                cols_to_fit.append(col)
            elif col in X.columns and X[col].nunique() <= 1:
                print(f"Warning: Column '{col}' is constant or has only one unique value. Skipping scaling for this column.")
            elif col not in X.columns:
                print(f"Warning: Column '{col}' not found in data during FeatureScaler fit. Skipping.")

        self._fitted_columns = cols_to_fit # Store columns that were actually fitted

        if self.method == 'standard':
            self.scaler = StandardScaler()
        # elif self.method == 'minmax':
        #     self.scaler = MinMaxScaler() # Need to import MinMaxScaler
        else:
            raise ValueError(f"Method must be 'standard', got '{self.method}'")

        if self._fitted_columns:
            self.scaler.fit(X[self._fitted_columns])
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self._fitted_columns and self.scaler:
            # Ensure columns still exist in X_copy before transforming
            cols_to_transform = [col for col in self._fitted_columns if col in X_copy.columns]
            if cols_to_transform:
                X_copy[cols_to_transform] = self.scaler.transform(X_copy[cols_to_transform])
        return X_copy


def create_feature_engineering_pipeline(numerical_imputation_strategy='mean', categorical_encoding_method='onehot', snapshot_date=None):
    """
    Creates a scikit-learn pipeline for feature engineering.
    """
    pipeline = Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('rfm_calculator', RFMCalculator(snapshot_date=snapshot_date)), # Added RFM calculation
        ('aggregate_features', AggregateFeatures()), # Aggregate features will now also see RFM
        ('missing_value_handler', MissingValueHandler(strategy=numerical_imputation_strategy)),
        ('categorical_encoder', CustomEncoder(method=categorical_encoding_method)),
        ('feature_scaler', FeatureScaler(method='standard'))
    ])
    return pipeline