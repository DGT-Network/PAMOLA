"""
PAMOLA.CORE - Descriptive Statistics Module
------------------------------------------------
Module:        Descriptive Statistics Analyzer
Package:       pamola_core.analysis
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides utilities to compute normalized descriptive statistics for pandas
  DataFrames. Normalizes pandas.DataFrame.describe() outputs to consistent
  dict structures suitable for downstream processing and visualization.
  Adds missing count and configurable extra statistics (median, mode), and
  handles numeric vs. non-numeric columns appropriately.

Key Features:
  - Normalizes descriptive outputs to consistent dicts
  - Adds configurable extra statistics (median, mode, unique)
  - Computes missing counts and preserves safe defaults
  - Handles numeric and non-numeric columns gracefully
  - Includes logging for diagnostics and production use

Dependencies:
  - pandas  - DataFrame operations
  - typing  - Type hints and validation
  - pamola_core.utils.logging - Module logging helper
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


def analyze_descriptive_stats(
    df: pd.DataFrame,
    field_names: Optional[List[str]] = None,
    describe_order: Optional[List[str]] = None,
    extra_statistics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze descriptive stats data frame.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame for calculate.
    field_names : list, optional
        Name of the fields to analyze.
    describe_order : list, optional
        Descriptive statistics.
    extra_statistics : list, optional
        Extra statistics.

    Returns:
    --------
    Dict[str, Any]
        Descriptive statistics of data frame.
    """
    if field_names is None:
        field_names = df.columns

    if describe_order is None:
        describe_order = ["count", "unique", "top", "freq", "mean", "std", "min", "max"]

    if extra_statistics is None:
        extra_statistics = ["unique", "median", "mode"]

    descriptive_statistics = df[field_names].describe(include="all").loc[describe_order]

    descriptive_statistics_transpose = descriptive_statistics.transpose()
    if "count" in describe_order:
        descriptive_statistics_transpose["missing"] = (
            len(df) - descriptive_statistics_transpose["count"]
        )

    descriptive_statistics_dict = {}

    for field_name, stats in descriptive_statistics_transpose.iterrows():
        field_stats = stats.dropna().to_dict()  # Remove NaN values

        # For numeric columns, manually convert stat
        if pd.api.types.is_numeric_dtype(df[field_name]):
            if "unique" in describe_order or "unique" in extra_statistics:
                field_stats["unique"] = df[field_name].nunique()

            if "median" in extra_statistics:
                field_stats["median"] = float(df[field_name].median())

            if "mode" in extra_statistics:
                field_stats["mode"] = float(df[field_name].mode().iloc[0])
        else:
            if "mode" in extra_statistics:
                field_stats["mode"] = df[field_name].mode().iloc[0]

        descriptive_statistics_dict[field_name] = field_stats

    return descriptive_statistics_dict
