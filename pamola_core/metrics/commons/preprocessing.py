"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Metric Preprocessing Utilities
Package:       pamola_core.metrics.commons.preprocessing
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides preprocessing utilities for transforming input datasets into numeric format
  suitable for distance-based metric calculations. Handles boolean, datetime, string,
  and categorical columns, ensuring compatibility for privacy, utility, and fidelity metrics.

Key Features:
  - Convert boolean columns to integers
  - Convert datetime columns to numeric timestamps
  - Ordinal encoding for string and categorical columns
  - Drop rows with missing values for robust metric computation

Design Principles:
  - Consistent and interpretable input formats for metrics
  - Ready for use in distance, nearest neighbor, and privacy metric calculations
  - Modular and reusable preprocessing logic

Dependencies:
  - pandas - DataFrame operations
  - scikit-learn - OrdinalEncoder for categorical encoding
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def prepare_data_for_distance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame for distance-based metric calculations.
    Converts boolean, datetime, string, and categorical columns to numeric.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset. Each row is a record, columns are features.

    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame with all features converted to numeric types.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    df_converted = df.copy()

    # 1. Convert boolean columns to integers
    bool_cols = df_converted.select_dtypes(include=["bool"]).columns.tolist()
    df_converted[bool_cols] = df_converted[bool_cols].astype(int)

    # 2. Convert datetime columns (including convertible object columns) to int64 (nanoseconds)
    datetime_cols = []
    for col in df_converted.columns:
        if pd.api.types.is_datetime64_any_dtype(df_converted[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_object_dtype(df_converted[col]):
            try:
                converted = pd.to_datetime(df_converted[col], errors="raise")
                df_converted[col] = converted
                datetime_cols.append(col)
            except Exception:
                continue

    for col in datetime_cols:
        df_converted[col] = df_converted[col].astype(
            "int64"
        )  # timestamp in nanoseconds

    # 3. Convert string and categorical columns to ordinal encoded numeric values
    string_cols = df_converted.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in string_cols:
        df_converted[col] = df_converted[col].astype(str)
        df_converted[col] = df_converted[col].str.strip()

    if string_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_converted[string_cols] = encoder.fit_transform(df_converted[string_cols])

    # 4. Drop rows with any NaN values
    df_converted = df_converted.dropna()

    numeric_cols = df_converted.select_dtypes(include=[np.number]).columns
    df_converted[numeric_cols] = df_converted[numeric_cols].fillna(df_converted[numeric_cols].median())

    return df_converted
