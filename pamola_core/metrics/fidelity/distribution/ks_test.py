"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fidelity Metric Operation - KolmogorovSmirnovTest
Package:       pamola_core.metrics.fidelity
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
    This module implements the Kolmogorov-Smirnov (KS) test as a fidelity metric
    for comparing distributions between original and transformed datasets.
    It supports both standard univariate comparisons and extended grouped
    comparisons using aggregation logic over key fields.

    The KS test evaluates whether two samples are drawn from the same distribution
    by computing the maximum absolute difference between their empirical
    cumulative distribution functions (ECDFs). It is a non-parametric test
    and suitable for both continuous and discrete data.

Key Features:
    - Supports grouped KS testing with configurable aggregation (sum, mean, etc.)
    - Fallback to traditional KS test for ungrouped single-column comparison
    - Dictionary-based cumulative distribution computation for structured data
    - Approximate p-value calculation for grouped distributions
    - Human-readable interpretation of statistical significance
    - Seamless integration into the PAMOLA.CORE metric operation framework
    - Compatible with Dask-based pipelines for scalability

Usage:
    >>> ks_test = KolmogorovSmirnovTest(key_fields=['region'], value_field='income', aggregation='mean')
    >>> result = ks_test.calculate_metric(original_df, anonymized_df)
    >>> print(result)

Framework:
    This module is designed for use within the PAMOLA.CORE metric operation pipeline,
    adhering to its standardized lifecycle for execution, reporting, and integration
    with fidelity scoring, profiling layers, and transformation evaluation tools.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from pamola_core.metrics.commons.aggregation import create_value_dictionary


class KolmogorovSmirnovTest:
    """
    Extended Kolmogorov-Smirnov (KS) test supporting grouped aggregation.

    This implementation extends the traditional KS test to allow:
    - Grouped comparisons via key fields.
    - Aggregation on a target value field (sum, mean, etc.).

    Parameters:
    ----------
    key_fields : Optional[List[str]]
        List of columns used to group the dataset before comparing.
        Example: ['gender', 'region'].

    value_field : Optional[str]
        The name of the numeric column to aggregate and compare.

    aggregation : str
        Aggregation method to apply to value_field within each group.
        Supported values: 'sum', 'mean', 'min', 'max', 'count', 'first', 'last'.
        Default is 'sum'.
    """

    def __init__(
        self,
        key_fields: Optional[List[str]] = None,
        value_field: Optional[str] = None,
        aggregation: str = "sum",
    ):
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation

    def calculate_metric(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate the KS statistic between the original and transformed datasets.

        Returns:
        --------
        Dict[str, Any]
            A dictionary with:
            - ks_statistic: float
            - p_value: float
            - max_difference: float (D-statistic)
            - interpretation: str
        """
        if self.key_fields and self.value_field:
            # Use grouped + aggregated comparison
            orig_dist = create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            trans_dist = create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )

            ks_stat, p_value = self._calculate_ks_from_dicts(orig_dist, trans_dist)
        else:
            # Fallback to standard KS test (1D distributions)
            columns = list(set(original_df.columns) & set(transformed_df.columns))
            if not columns:
                raise ValueError("No common columns to compare.")

            col = columns[0]
            ks_stat, p_value = stats.ks_2samp(
                original_df[col].dropna(), transformed_df[col].dropna()
            )

        return {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "max_difference": ks_stat,
            "interpretation": self._interpret_ks(ks_stat, p_value),
        }

    def _calculate_ks_from_dicts(
        self, dict1: Dict[str, float], dict2: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculate the Kolmogorov–Smirnov (KS) statistic and approximate p-value
        between two distributions provided as dictionaries.

        Parameters:
        -----------
        dict1 : Dict[str, float]
            Dictionary of aggregated group values from the original dataset.
            Keys represent group identifiers, values represent aggregated metrics.

        dict2 : Dict[str, float]
            Dictionary of aggregated group values from the transformed dataset.

        Returns:
        --------
        Tuple[float, float]
            - max_diff (float): The maximum difference between cumulative distributions (D-statistic).
            - p_value (float): The approximate p-value indicating statistical significance.
        """

        all_keys = sorted(set(dict1) | set(dict2))
        total1, total2 = sum(dict1.values()), sum(dict2.values())

        cumulative1 = 0.0
        cumulative2 = 0.0
        max_diff = 0.0

        for key in all_keys:
            cumulative1 += dict1.get(key, 0.0) / total1 if total1 else 0.0
            cumulative2 += dict2.get(key, 0.0) / total2 if total2 else 0.0

            diff = abs(cumulative1 - cumulative2)
            max_diff = max(max_diff, diff)

        # Approximate p-value using the KS distribution
        n1 = len(dict1)
        n2 = len(dict2)
        en = np.sqrt(n1 * n2 / (n1 + n2)) if (n1 + n2) > 0 else 1e-8
        p_value = 2 * np.exp(-2 * en**2 * max_diff**2)

        return max_diff, p_value

    def _interpret_ks(self, p_value: float) -> str:
        """
        Provide an interpretation of the KS test result based on the p-value.

        Parameters:
        -----------
        p_value : float
            The p-value computed from the KS test, indicating the probability of
            observing the given D-statistic under the null hypothesis
            (i.e., that both samples come from the same distribution).

        Returns:
        --------
        str
            Human-readable interpretation:
            - "Significant difference (p < 0.01)" → very strong evidence of distribution shift
            - "Likely different distributions (p < 0.05)" → moderate evidence
            - "No significant difference" → distributions are likely similar
        """
        if p_value < 0.01:
            return "Significant difference (p < 0.01)"
        elif p_value < 0.05:
            return "Likely different distributions (p < 0.05)"
        else:
            return "No significant difference"
