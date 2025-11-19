"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Predicted Utility Scoring
Package:       pamola_core.metrics.commons.predicted_utility_scoring
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides functions for calculate predicted utility.

Key Features:
  - Provisional predicted utility scoring on sampled subsets (e.g., 10%)

Design Principles:
  - Simple functional interface for quick integration

Dependencies:
  - pandas - DataFrame operations
  - numpy - mathematical functions and vectorized operations
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


def calculate_predicted_utility(
    df: pd.DataFrame,
    require_fields: Optional[List[str]] = None,
    unique_fields: Optional[List[str]] = None,
    check_balance_fields: Optional[List[str]] = None,
    dtypes_dict: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Calculate predicted utility score of a dataset based on completeness,
    diversity, balance check, and schema richness.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame for calculating score.
    require_fields : List[str], optional
        Fields used to calculate completeness.
    unique_fields : List[str], optional
        Fields used to calculate diversity.
    check_balance_fields : List[str], optional
        Binary fields to check for balance (e.g., gender, yes/no flags).
    dtypes_dict : Dict[str, str], optional
        Optional external schema for data types.
    weights : Dict[str, float], optional
        Weighting of each component.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing component scores and overall predicted utility (0–100).
    """
    if df.empty:
        return {
            "completeness": 0,
            "diversity": 0,
            "balance_check": 0,
            "schema_richness": 0,
            "predicted_utility": 0,
        }

    # --- Default weights ---
    if weights is None:
        weights = {
            "completeness": 0.25,
            "diversity": 0.25,
            "balance_check": 0.25,
            "schema_richness": 0.25,
        }

    n_rows = len(df)

    # --- 1. Completeness ---
    completeness = 1.0
    if require_fields:
        existing_req = [c for c in require_fields if c in df.columns]
        if existing_req:
            non_null_count = df[existing_req].notnull().values.sum()
            total_cells = df[existing_req].size
            completeness = non_null_count / total_cells if total_cells > 0 else 0.0
        else:
            completeness = 0.0

    # --- 2. Diversity ---
    diversity = 1.0
    if unique_fields:
        existing_unique = [c for c in unique_fields if c in df.columns]
        if existing_unique:
            avg_unique = df[existing_unique].nunique().mean()
            diversity = avg_unique / n_rows if n_rows > 0 else 0.0
        else:
            diversity = 0.0

    # --- 3. Balance Check ---
    balance_check = 1.0
    if check_balance_fields:
        valid_balance_fields = [c for c in check_balance_fields if c in df.columns]
        if valid_balance_fields:
            balance_scores = []
            for col in valid_balance_fields:
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    dominant_ratio = value_counts.max() / n_rows
                    balance_scores.append(1 - dominant_ratio)
            balance_check = np.mean(balance_scores) if balance_scores else 0.0
        else:
            balance_check = 0.0

    # --- 4. Schema Richness ---
    if dtypes_dict:
        dtypes_series = pd.Series(dtypes_dict)
    else:
        dtypes_series = df.dtypes

    dtype_counts = dtypes_series.value_counts()
    schema_richness = (
        dtype_counts.count() / dtype_counts.sum() if dtype_counts.sum() > 0 else 0.0
    )

    # --- 5. Weighted Predicted Utility ---
    predicted_utility = (
        weights.get("completeness", 0.25) * completeness
        + weights.get("diversity", 0.25) * diversity
        + weights.get("balance_check", 0.25) * balance_check
        + weights.get("schema_richness", 0.25) * schema_richness
    )

    return {
        "completeness": round(completeness, 2),
        "diversity": round(diversity, 2),
        "balance_check": round(balance_check, 2),
        "schema_richness": round(schema_richness, 2),
        "predicted_utility": int(predicted_utility * 100),
    }
