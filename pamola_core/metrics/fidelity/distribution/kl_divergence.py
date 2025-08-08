"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fidelity Metric Operation - KLDivergence
Package:       pamola_core.metrics.fidelity
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
    This module implements the Kullback–Leibler (KL) divergence as a fidelity metric
    to measure how the probability distribution of a transformed dataset diverges
    from the original dataset.

    The implementation supports grouped aggregation over multiple fields and
    includes optional smoothing to avoid division by zero or undefined values
    in sparse categorical distributions.

Key Features:
    - Grouped KL divergence using key fields + aggregation
    - Smoothing with epsilon for robustness
    - KL value in both nats and bits
    - Fallback to column-wise 1D comparison
    - Seamless integration into PAMOLA.CORE metric pipelines
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from pamola_core.metrics.commons.aggregation import create_value_dictionary


class KLDivergence:
    """
    KL Divergence Metric Operation with support for grouped aggregation and smoothing.

    Parameters:
    ----------
    key_fields : Optional[List[str]]
        Fields to group the data before calculating distributions.
        If provided, KL is calculated over aggregated group distributions.

    value_field : Optional[str]
        The numeric field to aggregate when grouping is applied.

    aggregation : str
        Aggregation method applied to value_field within each group.
        Supported values: 'sum', 'mean', 'count', etc.
        Default is 'count'.

    epsilon : float
        Smoothing parameter added to all values to avoid zero-probability issues.
        Default is 0.01.
    """

    def __init__(
        self,
        key_fields: Optional[List[str]] = None,
        value_field: Optional[str] = None,
        aggregation: str = "count",
        epsilon: float = 0.01,
    ):
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation
        self.epsilon = epsilon

    def calculate_metric(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute KL divergence between the original and transformed datasets.

        Process:
        --------
        1. Create aggregated value dictionaries (if key_fields is provided)
        2. Apply smoothing using epsilon
        3. Normalize the dictionaries to probability distributions
        4. Compute KL divergence via scipy.stats.entropy or custom logic

        Returns:
        --------
        Dict[str, Any]
            - kl_divergence: float (in nats)
            - kl_divergence_bits: float (in bits)
            - interpretation: str
            - smoothing_applied: bool
        """
        if self.key_fields:
            # Aggregated (grouped) mode
            p_dict = create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            q_dict = create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )

            kl_value = self._calculate_kl_from_dicts(p_dict, q_dict)
        else:
            # Simple column-wise probability distribution
            col = list(set(original_df.columns) & set(transformed_df.columns))[0]
            p_vals, q_vals = self._prepare_distributions(
                original_df[col], transformed_df[col]
            )
            kl_value = stats.entropy(p_vals, q_vals)

        return {
            "kl_divergence": kl_value,
            "kl_divergence_bits": kl_value / np.log(2),
            "interpretation": self._interpret_kl(kl_value),
            "smoothing_applied": self.epsilon > 0,
        }

    def _calculate_kl_from_dicts(
        self, p_dict: Dict[str, float], q_dict: Dict[str, float]
    ) -> float:
        """
        Calculate KL divergence from two value dictionaries with epsilon smoothing.

        Parameters:
        ----------
        p_dict : Dict[str, float]
            Dictionary representing the original distribution.

        q_dict : Dict[str, float]
            Dictionary representing the transformed distribution.

        Returns:
        --------
        float
            KL divergence (in nats)
        """
        all_keys = set(p_dict.keys()) | set(q_dict.keys())
        smoothed_p, smoothed_q = {}, {}

        for key in all_keys:
            # Get values, apply smoothing if necessary
            p_val = max(p_dict.get(key, 0.0), self.epsilon)
            q_val = max(q_dict.get(key, 0.0), self.epsilon)

            smoothed_p[key] = p_val
            smoothed_q[key] = q_val

        total_p = sum(smoothed_p.values())
        total_q = sum(smoothed_q.values())

        kl = 0.0
        for key in all_keys:
            p = smoothed_p[key] / total_p
            q = smoothed_q[key] / total_q

            if p > 0:  # Avoid log(0)
                kl += p * np.log(p / q)

        return kl

    def _prepare_distributions(
        self, series1: pd.Series, series2: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert two series to aligned probability vectors with epsilon smoothing.

        Parameters:
        ----------
        series1 : pd.Series
            Original values.

        series2 : pd.Series
            Transformed values.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Normalized and smoothed probability vectors for KL calculation.
        """
        counts1 = series1.value_counts().to_dict()
        counts2 = series2.value_counts().to_dict()

        all_keys = set(counts1.keys()) | set(counts2.keys())
        smoothed_p, smoothed_q = [], []

        for key in all_keys:
            p_val = max(counts1.get(key, 0), self.epsilon)
            q_val = max(counts2.get(key, 0), self.epsilon)
            smoothed_p.append(p_val)
            smoothed_q.append(q_val)

        p_array = np.array(smoothed_p) / np.sum(smoothed_p)
        q_array = np.array(smoothed_q) / np.sum(smoothed_q)

        return p_array, q_array

    def _interpret_kl(self, kl_value: float) -> str:
        """
        Provide a qualitative interpretation of the KL divergence.

        Parameters:
        ----------
        kl_value : float
            The KL divergence value (in nats).

        Returns:
        --------
        str
            Interpretation of how different the distributions are.
        """
        if kl_value < 0.01:
            return "Distributions are almost identical"
        elif kl_value < 0.1:
            return "Minor difference between distributions"
        elif kl_value < 0.5:
            return "Moderate difference"
        else:
            return "Significant difference between distributions"
