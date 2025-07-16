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


def normalize_distribution(values: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize a distribution (array of values) using specified method.

    Parameters
    ----------
    values : np.ndarray
        The array of values to normalize.
    method : str
        Normalization method: "minmax" or "zscore".

    Returns
    -------
    np.ndarray
        Normalized values.
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
