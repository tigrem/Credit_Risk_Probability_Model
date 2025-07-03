# src/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


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
            # Removed the ValueError to allow pipeline to continue if column is dropped earlier
            # or if it's not relevant for all datasets.
            # Instead, we will print a warning and return the DataFrame without time features.
            print("Warning: 'TransactionStartTime' column not found for FeatureExtractor. Cannot extract time features.")
            X_copy['TransactionHour'] = np.nan
            X_copy['TransactionDay'] = np.nan
            X_copy['TransactionMonth'] = np.nan
            X_copy['TransactionYear'] = np.nan
            return X_copy.drop('TransactionStartTime', axis=1, errors='ignore')


        # Convert to datetime, coercing errors to NaT
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'], errors='coerce')

        # Only create features if TransactionStartTime is not entirely NaT
        if not X_copy['TransactionStartTime'].isnull().all():
            X_copy['TransactionHour'] = X_copy['TransactionStartTime'].dt.hour
            X_copy['TransactionDay'] = X_copy['TransactionStartTime'].dt.day
            X_copy['TransactionMonth'] = X_copy['TransactionStartTime'].dt.month
            X_copy['TransactionYear'] = X_copy['TransactionStartTime'].dt.year
        else:
            # If all are NaT, create columns with NaNs or a default value
            print("Warning: All 'TransactionStartTime' values are NaT. Creating NaN time features.")
            X_copy['TransactionHour'] = np.nan
            X_copy['TransactionDay'] = np.nan
            X_copy['TransactionMonth'] = np.nan
            X_copy['TransactionYear'] = np.nan

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
        # Allow fit to proceed even if some RFM columns are missing, but warn
        missing_rfm_cols = [col for col in ['TransactionStartTime', 'AccountId', 'Amount'] if col not in X.columns]
        if missing_rfm_cols:
            print(f"Warning: Missing RFM required columns for fit: {missing_rfm_cols}. RFM calculation might be incomplete.")

        X_copy_for_fit = X.copy()
        if 'TransactionStartTime' in X_copy_for_fit.columns:
            X_copy_for_fit['TransactionStartTime'] = pd.to_datetime(X_copy_for_fit['TransactionStartTime'], errors='coerce') # Coerce errors

            if self.snapshot_date is None:
                # Set snapshot date to one day after the latest *valid* transaction date in the training data
                max_date = X_copy_for_fit['TransactionStartTime'].max()
                if pd.isna(max_date):
                    # If no valid dates, pick a default reference date or raise an error
                    self._fitted_snapshot_date = pd.Timestamp.now(tz='UTC').normalize() + pd.Timedelta(days=1) # Use UTC for consistency
                    print("Warning: No valid 'TransactionStartTime' found. Using current UTC date + 1 day as snapshot_date for RFM fit.")
                else:
                    self._fitted_snapshot_date = max_date + pd.Timedelta(days=1)
            else:
                self._fitted_snapshot_date = pd.to_datetime(self.snapshot_date)
        else:
            # If TransactionStartTime is not present, set a default snapshot date
            self._fitted_snapshot_date = pd.Timestamp.now(tz='UTC').normalize() + pd.Timedelta(days=1)
            print("Warning: 'TransactionStartTime' column not found for RFMCalculator fit. Using current UTC date + 1 day as default snapshot_date.")
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure columns exist before processing, provide default if not
        required_cols = ['TransactionStartTime', 'AccountId', 'Amount']
        for col in required_cols:
            if col not in X_copy.columns:
                print(f"Warning: Required column '{col}' not found for RFM calculation during transform. RFM features will be NaN/zero.")
                X_copy[col] = np.nan # Add column with NaNs to prevent KeyError later

        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'], errors='coerce')
        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')

        # Filter out rows with NaT in TransactionStartTime or NaN in Amount before RFM calculation
        rfm_data = X_copy.dropna(subset=['TransactionStartTime', 'Amount', 'AccountId']) # Added AccountId to dropna

        # Check if rfm_data is empty after dropping NaNs
        if rfm_data.empty:
            print("Warning: No valid transactions found after dropping NaNs for RFM calculation. Returning original DataFrame with NaN RFM columns.")
            for col in ['Recency', 'Frequency', 'Monetary']:
                if col not in X_copy.columns: # Prevent re-adding if already added due to previous warning
                    X_copy[col] = np.nan
            return X_copy


        # Group by AccountId to calculate RFM
        rfm_df = rfm_data.groupby('AccountId').agg(
            last_transaction_date=('TransactionStartTime', 'max'),
            Frequency=('TransactionId', 'count'), # Assuming TransactionId is robust for counting
            Monetary=('Amount', 'sum')
        ).reset_index()

        # Calculate Recency separately after aggregation for clarity and robustness
        if self._fitted_snapshot_date is None:
            # This should ideally not happen if fit was called, but as a fallback
            snapshot = pd.Timestamp.now(tz='UTC').normalize() + pd.Timedelta(days=1) # Fallback to UTC now
            print("Warning: RFMCalculator transform called without a fitted snapshot_date. Using current UTC date + 1 day.")
        else:
            snapshot = self._fitted_snapshot_date

        rfm_df['Recency'] = (snapshot - rfm_df['last_transaction_date']).dt.days
        rfm_df = rfm_df.drop(columns=['last_transaction_date'])

        # Fill NaNs in RFM columns if any (e.g., if an AccountId appeared in X_copy but had no valid transactions for RFM)
        # These fills ensure the columns are always numeric even if some accounts have no data.
        # Recency fill is high, Frequency and Monetary are zero for non-transacting customers.
        max_recency_val = rfm_df['Recency'].max() + 1 if not rfm_df['Recency'].empty else 0
        rfm_df['Recency'] = rfm_df['Recency'].fillna(max_recency_val)
        rfm_df['Frequency'] = rfm_df['Frequency'].fillna(0)
        rfm_df['Monetary'] = rfm_df['Monetary'].fillna(0)


        # Merge RFM features back to the original dataframe (X_copy includes all original rows)
        X_copy = pd.merge(X_copy, rfm_df, on='AccountId', how='left')

        # Handle potential NaNs for accounts in X_copy that were not present in `rfm_data`
        # This will set RFM to a default for accounts that had no valid transactions at all.
        X_copy['Recency'] = X_copy['Recency'].fillna(max_recency_val)
        X_copy['Frequency'] = X_copy['Frequency'].fillna(0)
        X_copy['Monetary'] = X_copy['Monetary'].fillna(0)

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
            print("Warning: 'AccountId' column not found for AggregateFeatures. Skipping aggregation.")
            for col in ['TotalTransactionAmount', 'AverageTransactionAmount', 'MinTransactionAmount', 'MaxTransactionAmount', 'TransactionCount', 'StdDevTransactionAmount']:
                if col not in X_copy.columns:
                    X_copy[col] = np.nan # Ensure columns exist
            return X_copy
        if 'Amount' not in X_copy.columns:
            print("Warning: 'Amount' column not found for AggregateFeatures. Skipping aggregation.")
            for col in ['TotalTransactionAmount', 'AverageTransactionAmount', 'MinTransactionAmount', 'MaxTransactionAmount', 'TransactionCount', 'StdDevTransactionAmount']:
                if col not in X_copy.columns:
                    X_copy[col] = np.nan # Ensure columns exist
            return X_copy

        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')

        # Drop NaNs in 'Amount' and 'AccountId' for aggregation
        agg_data = X_copy.dropna(subset=['Amount', 'AccountId'])


        if agg_data.empty:
            print("Warning: No valid 'Amount' values after dropping NaNs for aggregation. Returning original DataFrame with NaN aggregate columns.")
            for col in ['TotalTransactionAmount', 'AverageTransactionAmount', 'MinTransactionAmount', 'MaxTransactionAmount', 'TransactionCount', 'StdDevTransactionAmount']:
                if col not in X_copy.columns:
                    X_copy[col] = np.nan
            return X_copy

        agg_features = agg_data.groupby('AccountId').agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AverageTransactionAmount=('Amount', 'mean'),
            MinTransactionAmount=('Amount', 'min'),
            MaxTransactionAmount=('Amount', 'max'),
            TransactionCount=('TransactionId', 'count'), # Use TransactionId for counting
            StdDevTransactionAmount=('Amount', 'std')
        ).reset_index()

        # Fill NaNs for StdDev (occurs if only one transaction for an account)
        agg_features['StdDevTransactionAmount'] = agg_features['StdDevTransactionAmount'].fillna(0)

        X_copy = pd.merge(X_copy, agg_features, on='AccountId', how='left')

        # Fill NaNs for accounts that had no valid transactions for aggregation
        X_copy['TotalTransactionAmount'] = X_copy['TotalTransactionAmount'].fillna(0)
        X_copy['AverageTransactionAmount'] = X_copy['AverageTransactionAmount'].fillna(0)
        X_copy['MinTransactionAmount'] = X_copy['MinTransactionAmount'].fillna(0)
        X_copy['MaxTransactionAmount'] = X_copy['MaxTransactionAmount'].fillna(0)
        X_copy['TransactionCount'] = X_copy['TransactionCount'].fillna(0)
        X_copy['StdDevTransactionAmount'] = X_copy['StdDevTransactionAmount'].fillna(0)

        return X_copy


class CustomEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for encoding categorical variables.
    Handles both One-Hot Encoding and Label Encoding.
    """
    def __init__(self, method='onehot', columns=None):
        self.method = method
        self.columns = columns # This will store the list of columns passed from create_feature_engineering_pipeline
        self.encoders = {}
        self.fitted_ohe_columns = [] # To store column names created by one-hot encoding

    def fit(self, X, y=None):
        # Determine columns to encode based on 'columns' parameter or infer from dtypes
        if self.columns is None:
            # If no columns specified, infer categorical columns
            self._cols_to_encode_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            # Filter the provided columns to ensure they exist in X
            self._cols_to_encode_ = [col for col in self.columns if col in X.columns]
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                print(f"Warning: Columns {missing_cols} specified for CustomEncoder fit were not found in the DataFrame. Skipping them.")

        for col in self._cols_to_encode_:
            if self.method == 'label':
                le = LabelEncoder()
                # Ensure values are strings before fitting LabelEncoder
                unique_values = X[col].dropna().astype(str).unique()
                if len(unique_values) > 0:
                    self.encoders[col] = le.fit(unique_values)
                else:
                    self.encoders[col] = None # No values to fit
            elif self.method == 'onehot':
                # For one-hot, `pd.get_dummies` handles fit during transform by creating columns based on categories.
                # We can store the unique categories for consistency in transform.
                unique_categories = X[col].dropna().astype(str).unique().tolist()
                self.encoders[col] = unique_categories # Store categories for consistency
            else:
                raise ValueError(f"Method must be 'onehot' or 'label', got '{self.method}'")
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Use the columns determined during fit
        if not hasattr(self, '_cols_to_encode_') or self._cols_to_encode_ is None:
            print("Warning: CustomEncoder was not fitted or '_cols_to_encode_' not set. Inferring columns from object/category dtypes.")
            self._cols_to_encode_ = X_copy.select_dtypes(include=['object', 'category']).columns.tolist()


        transformed_df = X_copy.copy()
        current_ohe_cols = [] # To track one-hot encoded columns in this transform call

        for col in self._cols_to_encode_:
            if col not in transformed_df.columns:
                print(f"Warning: Column '{col}' specified for CustomEncoder transform not found in data. Skipping.")
                continue

            # Ensure the column is treated as categorical (object or category dtype)
            if not pd.api.types.is_object_dtype(transformed_df[col]) and not pd.api.types.is_categorical_dtype(transformed_df[col]):
                # If it's numeric but meant to be categorical (e.g., CountryCode), convert to string.
                # This ensures get_dummies treats unique numbers as distinct categories.
                print(f"Converting column '{col}' to object dtype for encoding.")
                transformed_df[col] = transformed_df[col].astype(str)
            else:
                # Fill NaNs in categorical columns with a placeholder before encoding
                # This is crucial for consistent dummy creation.
                transformed_df[col] = transformed_df[col].fillna('__MISSING__')


            if self.method == 'onehot':
                # Get the dummy variables for the current column
                dummies = pd.get_dummies(transformed_df[col], prefix=col, dummy_na=False)

                # Ensure all columns that were seen during fit are present in transform
                # This handles cases where a category might be missing in a specific batch of data.
                if col in self.encoders: # self.encoders[col] stores unique_categories from fit
                    fitted_categories = self.encoders[col]
                    for cat in fitted_categories:
                        dummy_col_name = f"{col}_{cat}"
                        if dummy_col_name not in dummies.columns:
                            dummies[dummy_col_name] = 0 # Add missing category column as all zeros

                # Drop the original column and concatenate dummies
                transformed_df = transformed_df.drop(columns=[col])
                transformed_df = pd.concat([transformed_df, dummies], axis=1)
                current_ohe_cols.extend(dummies.columns.tolist())

            elif self.method == 'label':
                if col in self.encoders and self.encoders[col] is not None:
                    # Fill NaNs with a placeholder before mapping to avoid errors, then map
                    temp_col = transformed_df[col].astype(str)
                    # Use a lambda to handle unseen categories and map them to -1
                    transformed_df[col] = temp_col.apply(
                        lambda s: self.encoders[col].transform([s])[0]
                        if s in self.encoders[col].classes_ else -1
                    )
                    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
                else:
                    # If encoder was not fitted (e.g., all NaNs in fit, or no unique values), fill with -1
                    transformed_df[col] = -1
            else:
                pass

        # Update fitted_ohe_columns after all transformations
        if self.method == 'onehot':
            self.fitted_ohe_columns = list(set(self.fitted_ohe_columns) | set(current_ohe_cols)) # Union of previous and current

        return transformed_df


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to handle missing values.
    """
    def __init__(self, strategy='mean', numerical_cols=None, categorical_cols=None):
        self.strategy = strategy
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.imputers = {}
        self._fitted_numerical_cols = []
        self._fitted_categorical_cols = []

    def fit(self, X, y=None):
        # Identify numerical and categorical columns present in X
        current_numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        current_categorical_cols = X.select_dtypes(include='object').columns.tolist()

        # Determine which columns to actually impute based on provided lists or inference
        self._numerical_cols_to_use = (
            [col for col in self.numerical_cols if col in current_numerical_cols]
            if self.numerical_cols is not None
            else current_numerical_cols
        )
        self._categorical_cols_to_use = (
            [col for col in self.categorical_cols if col in current_categorical_cols]
            if self.categorical_cols is not None
            else current_categorical_cols
        )

        # Fit numerical imputer
        numerical_cols_to_impute = [col for col in self._numerical_cols_to_use if X[col].isnull().any()]
        if numerical_cols_to_impute:
            self.imputers['numerical'] = SimpleImputer(strategy=self.strategy)
            self.imputers['numerical'].fit(X[numerical_cols_to_impute])
            self._fitted_numerical_cols = numerical_cols_to_impute
        else:
            self._fitted_numerical_cols = []

        # Fit categorical imputer
        categorical_cols_to_impute = [col for col in self._categorical_cols_to_use if X[col].isnull().any()]
        if categorical_cols_to_impute:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent') # Always mode for categorical
            self.imputers['categorical'].fit(X[categorical_cols_to_impute])
            self._fitted_categorical_cols = categorical_cols_to_impute
        else:
            self._fitted_categorical_cols = []

        return self

    def transform(self, X):
        X_copy = X.copy()

        # Impute numerical columns
        if 'numerical' in self.imputers and self._fitted_numerical_cols:
            cols_to_transform = [col for col in self._fitted_numerical_cols if col in X_copy.columns]
            if cols_to_transform:
                # Ensure columns are numeric before imputation
                for col in cols_to_transform:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                # Perform imputation
                X_copy[cols_to_transform] = self.imputers['numerical'].transform(X_copy[cols_to_transform])

        # Impute categorical columns
        if 'categorical' in self.imputers and self._fitted_categorical_cols:
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
        current_numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        if self.columns is None:
            cols_to_consider = current_numerical_cols
        else:
            # Filter the provided columns to ensure they are numeric and exist
            cols_to_consider = [col for col in self.columns if col in current_numerical_cols and pd.api.types.is_numeric_dtype(X[col])]

        cols_to_fit = []
        for col in cols_to_consider:
            # Only fit scaler on columns that are not constant (have more than 1 unique value)
            if X[col].nunique() > 1:
                cols_to_fit.append(col)
            else:
                print(f"Warning: Column '{col}' is constant or has only one unique value. Skipping scaling for this column.")

        self._fitted_columns = cols_to_fit

        if self.method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Method must be 'standard', got '{self.method}'")

        if self._fitted_columns:
            # Ensure data is numeric before fitting the scaler
            X_temp = X[self._fitted_columns].apply(pd.to_numeric, errors='coerce')
            self.scaler.fit(X_temp)
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self._fitted_columns and self.scaler:
            # Filter columns to transform, ensuring they are present and numeric
            cols_to_transform = [col for col in self._fitted_columns if col in X_copy.columns and pd.api.types.is_numeric_dtype(X_copy[col])]
            if cols_to_transform:
                X_copy[cols_to_transform] = X_copy[cols_to_transform].apply(pd.to_numeric, errors='coerce')
                # Replace inf/-inf with NaN before scaling if any were introduced
                X_copy[cols_to_transform] = X_copy[cols_to_transform].replace([np.inf, -np.inf], np.nan)
                # Impute NaNs that might have been created by coerce or inf replacement before scaling
                for col in cols_to_transform:
                    if X_copy[col].isnull().any():
                        # Use the mean of the column for imputation (simple strategy for remaining NaNs)
                        col_mean = X_copy[col].mean()
                        X_copy[col] = X_copy[col].fillna(col_mean if not pd.isna(col_mean) else 0) # Fallback to 0 if mean is NaN too

                X_copy[cols_to_transform] = self.scaler.transform(X_copy[cols_to_transform])
        return X_copy


# --- Added functions and classes for column name cleaning ---

def clean_column_names(df):
    """
    Cleans column names by replacing spaces with underscores and converting to lowercase.
    """
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

# Transformer to integrate clean_column_names into a pipeline
class ColumnNamer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return clean_column_names(X.copy())


# --- Updated pipeline creation function ---

def create_feature_engineering_pipeline(numerical_imputation_strategy='mean', categorical_encoding_method='onehot', snapshot_date=None):
    """
    Creates a scikit-learn pipeline for feature engineering.
    """
    # Define columns for one-hot encoding explicitly.
    # EXCLUDE high-cardinality identifiers like TransactionId, AccountId, SubscriptionId, CustomerId.
    # Include genuinely categorical features with a manageable number of unique values.
    # Note: CountryCode and ProviderId might be numeric in source, but are treated as categories here.
    ohe_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

    pipeline = Pipeline([
        ('column_namer', ColumnNamer()), # Added: Optional step to clean column names early in the pipeline
        # Uncomment this line if you want to include it.
        ('rfm_calculator', RFMCalculator(snapshot_date=snapshot_date)),
        ('feature_extractor', FeatureExtractor()),
        ('aggregate_features', AggregateFeatures()),
        ('missing_value_handler', MissingValueHandler(strategy=numerical_imputation_strategy)),
        # Pass the explicit list of columns to encode to CustomEncoder
        ('categorical_encoder', CustomEncoder(method=categorical_encoding_method, columns=ohe_cols)),
        ('feature_scaler', FeatureScaler(method='standard'))
    ])
    return pipeline
def create_feature_engineering_pipeline(numerical_imputation_strategy='mean', categorical_encoding_method='onehot', snapshot_date=None):
    """
    Creates a scikit-learn pipeline for feature engineering.
    """
    # Define columns for one-hot encoding explicitly.
    # EXCLUDE high-cardinality identifiers like TransactionId, AccountId, SubscriptionId, CustomerId, ProductId, ProviderId.
    # Include genuinely categorical features with a manageable number of unique values.
    # Removed 'ProductId' and 'ProviderId' from ohe_cols
    ohe_cols = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId'] # <--- UPDATED THIS LINE

    pipeline = Pipeline([
        # ('column_namer', ColumnNamer()), # Uncomment this line if you want to include it.
        ('rfm_calculator', RFMCalculator(snapshot_date=snapshot_date)),
        ('feature_extractor', FeatureExtractor()),
        ('aggregate_features', AggregateFeatures()),
        ('missing_value_handler', MissingValueHandler(strategy=numerical_imputation_strategy)),
        ('categorical_encoder', CustomEncoder(method=categorical_encoding_method, columns=ohe_cols)),
        ('feature_scaler', FeatureScaler(method='standard'))
    ])
    return pipeline