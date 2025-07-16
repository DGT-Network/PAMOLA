"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Dataset and Metric Validators
Package:       pamola_core.metrics.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides validation utilities for comparing datasets and ensuring compatibility
  between original and transformed data. Used across fidelity, utility, and
  privacy metric operations in PAMOLA.

Key Features:
  - Dataset compatibility validation (shape, columns, dtypes)
  - Metric input validation for required columns and types
  - Clear separation of validation result and exception-based validation
  - Customizable checks: column name match, dtype match

Design Principles:
  - Fail-fast validation for metric execution
  - Structured validation results with errors and warnings
  - Type-safe, testable, and integration-ready

Dependencies:
  - pandas - DataFrame operations
  - typing - Type hints
  - dataclasses - Structured validation result
"""

from typing import List, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class ValidationResult:
    success: bool
    errors: List[str]
    warnings: List[str]


def validate_dataset_compatibility(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    require_same_columns: bool = True,
    require_same_types: bool = True,
) -> ValidationResult:
    """
    Validate that two datasets can be compared.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataset.
    df2 : pd.DataFrame
        Second dataset.
    require_same_columns : bool
        Whether to enforce exact same columns in both datasets.
    require_same_types : bool
        Whether to enforce same column types.

    Returns
    -------
    ValidationResult
        Result of validation with success flag, errors, and warnings.
    """
    errors = []
    warnings = []

    # Check shape
    if df1.shape[0] != df2.shape[0]:
        warnings.append(
            f"Number of rows differ: df1={df1.shape[0]}, df2={df2.shape[0]}"
        )

    # Check columns
    if require_same_columns:
        missing_in_df1 = set(df2.columns) - set(df1.columns)
        missing_in_df2 = set(df1.columns) - set(df2.columns)
        if missing_in_df1:
            errors.append(f"Columns missing in df1: {missing_in_df1}")
        if missing_in_df2:
            errors.append(f"Columns missing in df2: {missing_in_df2}")
    else:
        common_columns = list(set(df1.columns) & set(df2.columns))
        if not common_columns:
            errors.append("No common columns between datasets.")

    # Check types
    if require_same_types:
        for col in df1.columns:
            if col in df2.columns:
                if df1[col].dtype != df2[col].dtype:
                    warnings.append(
                        f"Different types for column '{col}': df1={df1[col].dtype}, df2={df2[col].dtype}"
                    )

    return ValidationResult(
        success=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )


def validate_metric_inputs(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    columns: List[str],
    metric_type: str,
) -> None:
    """
    Validate inputs for specific metric type.

    Parameters
    ----------
    original : pd.DataFrame
        Original dataset.
    transformed : pd.DataFrame
        Transformed/anonymized dataset.
    columns : List[str]
        Columns to compare.
    metric_type : str
        Type of metric: "fidelity", "privacy", "utility", etc.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    if not isinstance(original, pd.DataFrame) or not isinstance(
        transformed, pd.DataFrame
    ):
        raise ValueError(
            "Both original and transformed inputs must be pandas DataFrames"
        )

    if not columns:
        raise ValueError("Column list cannot be empty")

    missing_in_original = [col for col in columns if col not in original.columns]
    missing_in_transformed = [col for col in columns if col not in transformed.columns]

    if missing_in_original:
        raise ValueError(f"Columns missing in original: {missing_in_original}")
    if missing_in_transformed:
        raise ValueError(f"Columns missing in transformed: {missing_in_transformed}")

    if metric_type not in ["fidelity", "privacy", "utility"]:
        raise ValueError(f"Unsupported metric_type: {metric_type}")
