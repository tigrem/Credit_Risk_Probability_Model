# tests/test_data_processing.py

import pandas as pd
import numpy as np
import pytest
import os
import sys

# Add the 'src' directory to the Python path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the function/class to be tested
# Make sure these imports match where you defined the functions/classes
from src.feature_engineering import clean_column_names, FeatureExtractor, RFMCalculator

# Test 1: Test clean_column_names function
def test_clean_column_names():
    """
    Tests if the clean_column_names function correctly cleans DataFrame column names.
    """
    data = {'Col One': [1, 2], 'Another Column': [3, 4], 'THIRD COLUMN': [5, 6]}
    df = pd.DataFrame(data)
    cleaned_df = clean_column_names(df.copy())
    expected_columns = ['col_one', 'another_column', 'third_column']
    assert list(cleaned_df.columns) == expected_columns, "Column names were not cleaned correctly."
    print("test_clean_column_names passed.")

# Test 2: Test FeatureExtractor's handling of TransactionStartTime
def test_feature_extractor_time_features():
    """
    Tests if FeatureExtractor correctly extracts time-based features and drops original column.
    """
    data = {
        'TransactionId': [1, 2],
        'TransactionStartTime': ['2023-01-15 10:30:00', '2024-03-20 22:15:00'],
        'Amount': [100, 200]
    }
    df = pd.DataFrame(data)
    extractor = FeatureExtractor()
    transformed_df = extractor.transform(df.copy())
    assert 'TransactionStartTime' not in transformed_df.columns, "Original 'TransactionStartTime' column was not dropped."
    expected_new_cols = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    for col in expected_new_cols:
        assert col in transformed_df.columns, f"'{col}' column was not created."
    assert transformed_df.loc[0, 'TransactionHour'] == 10
    assert transformed_df.loc[0, 'TransactionDay'] == 15
    assert transformed_df.loc[0, 'TransactionMonth'] == 1
    assert transformed_df.loc[0, 'TransactionYear'] == 2023
    print("test_feature_extractor_time_features passed.")

# Test 3: Test RFMCalculator handles missing data gracefully
def test_rfm_calculator_missing_data_handling():
    """
    Tests if RFMCalculator handles missing TransactionStartTime or Amount gracefully.
    """
    data = {
        'TransactionId': [1, 2, 3],
        'AccountId': [101, 101, 102],
        'TransactionStartTime': ['2023-01-01', '2023-01-05', np.nan],
        'Amount': [100, np.nan, 50]
    }
    df = pd.DataFrame(data)

    rfm_calc = RFMCalculator(snapshot_date='2023-01-10')
    rfm_calc.fit(df.copy())
    transformed_df = rfm_calc.transform(df.copy())

    assert 'Recency' in transformed_df.columns
    assert 'Frequency' in transformed_df.columns
    assert 'Monetary' in transformed_df.columns

    account_101_rfm = transformed_df[transformed_df['AccountId'] == 101].iloc[0]
    assert account_101_rfm['Recency'] == 5
    assert account_101_rfm['Frequency'] == 1
    assert account_101_rfm['Monetary'] == 100

    account_102_rfm = transformed_df[transformed_df['AccountId'] == 102].iloc[0]
    assert account_102_rfm['Frequency'] == 0
    assert account_102_rfm['Monetary'] == 0
    assert account_102_rfm['Recency'] == 6 # Based on 5+1

    print("test_rfm_calculator_missing_data_handling passed.")