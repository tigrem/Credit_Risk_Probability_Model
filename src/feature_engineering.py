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
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        X_copy['TransactionHour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['TransactionDay'] = X_copy['TransactionStartTime'].dt.day
        X_copy['TransactionMonth'] = X_copy['TransactionStartTime'].dt.month
        X_copy['TransactionYear'] = X_copy['TransactionStartTime'].dt.year
        return X_copy.drop('TransactionStartTime', axis=1) # Drop original timestamp column


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
            raise ValueError("'AccountId' column not found.")
        if 'Amount' not in X_copy.columns:
            raise ValueError("'Amount' column not found.")

        # Convert 'Amount' to numeric, coercing errors
        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')

        # Aggregate features
        agg_features = X_copy.groupby('AccountId').agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AverageTransactionAmount=('Amount', 'mean'),
            MinTransactionAmount=('Amount', 'min'),
            MaxTransactionAmount=('Amount', 'max'),
            TransactionCount=('TransactionId', 'count'),
            StdDevTransactionAmount=('Amount', 'std')
        ).reset_index()

        # Merge back to the original dataframe
        # We need to decide how to merge these. For a typical credit scoring model,
        # we'd usually have one row per customer/account.
        # For now, let's merge these back to the transaction level.
        # A more sophisticated approach might involve a different aggregation strategy
        # or creating a customer-level dataset.
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

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include='object').columns.tolist()

        for col in self.columns:
            if self.method == 'onehot':
                # Handle potential NaN values by converting them to a string representation
                # or by using handle_unknown='ignore' if we use OneHotEncoder directly.
                # For simplicity here, we'll let pandas get_dummies handle NaNs.
                pass
            elif self.method == 'label':
                le = LabelEncoder()
                # Fit on non-null unique values to avoid issues with NaN during fit
                self.encoders[col] = le.fit(X[col].astype(str).unique())
            else:
                raise ValueError("Method must be 'onehot' or 'label'")
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns is None:
            self.columns = X_copy.select_dtypes(include='object').columns.tolist()

        for col in self.columns:
            if col in X_copy.columns:
                if self.method == 'onehot':
                    # Use pandas get_dummies for simplicity and good handling of NaNs
                    X_copy = pd.get_dummies(X_copy, columns=[col], prefix=col, dummy_na=False)
                elif self.method == 'label':
                    # Apply transform, handling potential new categories with -1 or NaN
                    # Or, as a simpler approach, fill NaNs before encoding
                    X_copy[col] = X_copy[col].astype(str).map(lambda s: self.encoders[col].transform([s])[0] if s in self.encoders[col].classes_ else -1)
        return X_copy


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to handle missing values.
    """
    def __init__(self, strategy='mean', numerical_cols=None, categorical_cols=None):
        self.strategy = strategy
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.imputers = {}

    def fit(self, X, y=None):
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include='object').columns.tolist()

        # Imputer for numerical columns
        if self.numerical_cols:
            self.imputers['numerical'] = SimpleImputer(strategy=self.strategy)
            self.imputers['numerical'].fit(X[self.numerical_cols])

        # Imputer for categorical columns (most frequent)
        if self.categorical_cols:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'].fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.numerical_cols and 'numerical' in self.imputers:
            X_copy[self.numerical_cols] = self.imputers['numerical'].transform(X_copy[self.numerical_cols])
        if self.categorical_cols and 'categorical' in self.imputers:
            X_copy[self.categorical_cols] = self.imputers['categorical'].transform(X_copy[self.categorical_cols])
        return X_copy


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for scaling numerical features.
    """
    def __init__(self, method='standard', columns=None):
        self.method = method
        self.columns = columns
        self.scaler = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()

        if self.method == 'standard':
            self.scaler = StandardScaler()
        # elif self.method == 'minmax':
        #     self.scaler = MinMaxScaler() # Need to import MinMaxScaler
        else:
            raise ValueError("Method must be 'standard'")

        if self.columns:
            self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns is None:
            self.columns = X_copy.select_dtypes(include=np.number).columns.tolist()

        if self.columns and self.scaler:
            X_copy[self.columns] = self.scaler.transform(X_copy[self.columns])
        return X_copy


def create_feature_engineering_pipeline(numerical_imputation_strategy='mean', categorical_encoding_method='onehot'):
    """
    Creates a scikit-learn pipeline for feature engineering.
    """
    pipeline = Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('aggregate_features', AggregateFeatures()),
        ('missing_value_handler', MissingValueHandler(strategy=numerical_imputation_strategy)),
        ('categorical_encoder', CustomEncoder(method=categorical_encoding_method)),
        ('feature_scaler', FeatureScaler(method='standard'))
    ])
    return pipeline