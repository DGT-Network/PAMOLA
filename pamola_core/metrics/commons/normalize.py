"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Metric Normalization Utilities
Package:       pamola_core.metrics.commons.normalize
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides normalization utilities for transforming metric values or distributions
  into a standard scale, typically [0, 1]. Supports both individual value normalization
  and full distribution normalization.

Key Features:
  - Min-max normalization to custom target ranges
  - Support for inverting metrics where lower is better
  - Normalize arrays for distribution-based comparisons
  - Pluggable normalization methods (minmax, zscore)

Design Principles:
  - Consistent and interpretable metric scales
  - Separation between value and distribution normalization
  - Ready for use in scoring, visualization, and evaluation steps

Dependencies:
  - numpy - Core numeric operations
  - typing - Type annotations for clarity
"""

from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize_metric_value(
    value: float,
    metric_range: Tuple[float, float],
    target_range: Tuple[float, float] = (0, 1),
    higher_is_better: bool = True,
) -> float:
    """
    Normalize a single metric value to a target range, usually [0, 1].

    Parameters
    ----------
    value : float
        The metric value to normalize.
    metric_range : Tuple[float, float]
        The expected min and max values of the metric.
    target_range : Tuple[float, float]
        Desired output range, default is (0, 1).
    higher_is_better : bool
        If False, the value will be inverted before scaling.

    Returns
    -------
    float
        Normalized metric value within the target range.
    """
    min_val, max_val = metric_range
    tgt_min, tgt_max = target_range

    if not higher_is_better:
        value = max_val - (value - min_val)  # invert

    if max_val == min_val:
        return tgt_min  # Avoid division by zero

    norm_value = (value - min_val) / (max_val - min_val)
    scaled_value = tgt_min + norm_value * (tgt_max - tgt_min)
    return float(scaled_value)


def normalize_array_np(values: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize an array using the specified method (manual implementation with NumPy).

    Parameters
    ----------
    values : np.ndarray
        1D array of numeric values to be normalized.
    method : str, optional, default="minmax"
        Normalization method to apply:
        - "minmax": scales values to the [0, 1] range.
        - "zscore": standardizes values to zero mean and unit variance.

    Returns
    -------
    np.ndarray
        Normalized array of the same shape as input.

    Raises
    ------
    ValueError
        If an unsupported normalization method is specified.
    """
    if method == "minmax":
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        if range_val == 0:
            return np.zeros_like(values)
        return (values - min_val) / range_val

    elif method == "zscore":
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - mean) / std

    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def normalize_array_sklearn(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize an array using scikit-learn's scalers or probability normalization.

    Parameters
    ----------
    data : np.ndarray
        1D array of numeric values to normalize.
    method : str or bool, default="zscore"
        Normalization method to apply:
        - "zscore" or True: standardizes values to zero mean and unit variance.
        - "minmax": scales values to the [0, 1] range.
        - "probability": normalizes values to sum to 1 (i.e., probability distribution).
        - "none" or False: returns the original array.

    Returns
    -------
    np.ndarray
        Normalized array of the same shape as input.

    Raises
    ------
    ValueError
        If an unsupported normalization method is specified.
    """
    if method in [False, "none"]:
        return data

    # Handle "zscore" or True
    if method in [True, "zscore"]:
        if np.std(data) == 0:
            return data  # Avoid division by zero
        scaler = StandardScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Handle "minmax"
    elif method == "minmax":
        if np.ptp(data) == 0:
            return data  # Avoid division by zero
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Handle "probability"
    elif method == "probability":
        total = np.sum(data)
        return data / total if total > 0 else data

    else:
        raise ValueError(f"Unknown normalization method: {method}")
