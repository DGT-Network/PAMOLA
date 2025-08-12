"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Metric Aggregation Utilities
Package:       pamola_core.metrics.commons.aggregation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides aggregation utilities for combining column-level or metric-level scores
  into dataset-level or composite quality scores. Useful for summarizing metric
  outputs across columns or metric types.

Key Features:
  - Aggregate column-wise metric results using mean or weighted average
  - Compute composite index scores across different metric types
  - Optional min-max normalization
  - Supports weighting for custom importance of fields/metrics

Design Principles:
  - General-purpose aggregation independent of metric type
  - Simple interfaces with pluggable methods
  - Ready for integration with reporting and visualization modules

Dependencies:
  - typing - Type hints for clarity and safety
  - numpy - Aggregation and normalization
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def aggregate_column_metrics(
    column_results: Dict[str, Dict[str, float]],
    method: str = "mean",
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Aggregate column-level metrics to a single dataset-level value.

    Parameters
    ----------
    column_results : Dict[str, Dict[str, float]]
        Dictionary of column names mapped to metric scores. Example:
        {
            "age": {"ks": 0.9},
            "income": {"ks": 0.8},
        }

    method : str
        Aggregation method: "mean", "weighted", or "sum".

    weights : Optional[Dict[str, float]]
        Weights per column. Only used if method == "weighted".

    Returns
    -------
    float
        Aggregated score for the dataset.
    """
    values = []

    for col, metrics in column_results.items():
        score = list(metrics.values())[0]  # Assumes 1 metric per column
        if method == "weighted":
            weight = weights.get(col, 1.0) if weights else 1.0
            values.append(score * weight)
        else:
            values.append(score)

    if not values:
        return 0.0

    if method == "sum":
        return float(np.sum(values))
    elif method == "weighted" and weights:
        total_weight = sum(weights.get(col, 1.0) for col in column_results)
        return float(np.sum(values) / total_weight) if total_weight > 0 else 0.0
    else:  # default to mean
        return float(np.mean(values))


def create_composite_score(
    metrics: Dict[str, float], weights: Dict[str, float], normalization: str = "minmax"
) -> float:
    """
    Create a composite score from multiple metric values using weights.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and their values. Example:
        {"fidelity": 0.85, "utility": 0.78, "privacy": 0.92}

    weights : Dict[str, float]
        Weights for each metric. Must match the keys in `metrics`.

    normalization : str
        Normalization method: "minmax" (default) or "none".

    Returns
    -------
    float
        Composite weighted score normalized to [0, 1] if applicable.
    """
    aligned_metrics = []
    aligned_weights = []

    for name, value in metrics.items():
        weight = weights.get(name, 1.0)
        aligned_metrics.append(value)
        aligned_weights.append(weight)

    if normalization == "minmax":
        min_val = min(aligned_metrics)
        max_val = max(aligned_metrics)
        range_val = max_val - min_val if max_val != min_val else 1.0
        aligned_metrics = [(v - min_val) / range_val for v in aligned_metrics]

    weighted_score = np.average(aligned_metrics, weights=aligned_weights)
    return float(weighted_score)


def create_value_dictionary(
    df: pd.DataFrame,
    key_fields: List[str],
    value_field: Optional[str] = None,
    aggregation: str = "sum",
) -> Dict[str, float]:
    """
    Create a dictionary with composite keys and aggregated values from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to group and aggregate.

    key_fields : List[str]
        List of column names to use for grouping. Each unique combination will form a composite key.

    value_field : Optional[str], default=None
        The column to aggregate. If None or aggregation is "count", the function will count occurrences per group.

    aggregation : str, default='sum'
        Aggregation method to apply to `value_field`.
        Supported options: 'sum', 'mean', 'min', 'max', 'count', 'first', 'last'.

    Returns
    -------
    Dict[str, float]
        A dictionary where:
        - Each key is a composite string from `key_fields` (e.g., "North_30_Yes").
        - Each value is the result of the aggregation applied to that group.
    """
    if value_field is not None and value_field not in df.columns:
        raise KeyError(f"Value field '{value_field}' not found in DataFrame.")

    if value_field is None or aggregation == "count":
        grouped = df.groupby(key_fields).size()
    else:
        grouped = df.groupby(key_fields)[value_field]

        agg_func_map = {
            "sum": grouped.sum,
            "mean": grouped.mean,
            "min": grouped.min,
            "max": grouped.max,
            "count": grouped.count,
            "first": grouped.first,
            "last": grouped.last,
        }

        if aggregation not in agg_func_map:
            raise ValueError(f"Unsupported aggregation: '{aggregation}'")

        grouped = agg_func_map[aggregation]()

    return {
        "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,))): float(v)
        for key, v in grouped.items()
    }
