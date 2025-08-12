"""
Custom Aggregation Function Registry for Aggregate Operations

This module defines and registers custom aggregation functions
that can be safely referenced by name from UI or config.
"""

from typing import Callable, Dict
import numpy as np
import pandas as pd

def count_above_200(series: pd.Series) -> int:
    """Count values greater than 200."""
    return (series > 200).sum()

def range_spread(series: pd.Series) -> float:
    """Calculate the range (max - min) of the series."""
    return series.max() - series.min()

def mode_value(series: pd.Series):
    """Return the mode (most frequent value) of the series."""
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else None

def nonzero_count(series: pd.Series) -> int:
    """Count nonzero values."""
    return (series != 0).sum()

def percentile_90(series: pd.Series) -> float:
    """Return the 90th percentile value."""
    return np.percentile(series.dropna(), 90) if len(series.dropna()) > 0 else None


# Standard aggregation functions mapping
STANDARD_AGGREGATIONS = {
    'count': pd.Series.count,
    'sum': pd.Series.sum,
    'mean': pd.Series.mean,
    'median': pd.Series.median,
    'min': pd.Series.min,
    'max': pd.Series.max,
    'std': pd.Series.std,
    'var': pd.Series.var,
    'first': lambda x: x.iloc[0] if not x.empty else None,
    'last': lambda x: x.iloc[-1] if not x.empty else None,
    'nunique': pd.Series.nunique
}

# Registry of allowed custom aggregation functions
CUSTOM_AGG_FUNCTIONS: Dict[str, Callable] = {
    "count_above_200": count_above_200,
    "range_spread": range_spread,
    "mode": mode_value,
    "nonzero_count": nonzero_count,
    "percentile_90": percentile_90,
    # Add more custom aggregations here as needed
}

def get_custom_aggregation_function(func_name: str) -> Callable:
    """
    Get a custom aggregation function by name.
    Raises KeyError if not found.
    """
    if func_name not in CUSTOM_AGG_FUNCTIONS:
        raise KeyError(f"Custom aggregation function '{func_name}' is not registered.")
    return CUSTOM_AGG_FUNCTIONS[func_name]