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
    Enhanced Kolmogorov-Smirnov (KS) test implementation with improved confidence level
    handling and normalization options for comparing distributions between original
    and transformed datasets.

Key Enhancements:
    - Dynamic confidence level-based thresholds for statistical significance
    - Flexible normalization options (z-score, min-max, none)
    - Enhanced statistical testing with effect size calculation
    - Improved p-value interpretation with confidence intervals
    - Robust handling of edge cases and data validation
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from pamola_core.metrics.commons.aggregation import create_value_dictionary
from pamola_core.metrics.commons.normalize import normalize_array_sklearn
from pamola_core.metrics.commons.validation import validate_confidence_level


class KolmogorovSmirnovTest:
    """
    Enhanced Kolmogorov-Smirnov (KS) test with confidence level and normalization support.

    Parameters:
    ----------
    key_fields : Optional[List[str]]
        List of columns used to group the dataset before comparing.

    value_field : Optional[str]
        The name of the numeric column to aggregate and compare.

    aggregation : str
        Aggregation method to apply to value_field within each group.
        Supported values: 'sum', 'mean', 'min', 'max', 'count', 'first', 'last'.
        Default is 'sum'.

    confidence_level : float
        Confidence level for statistical significance testing (0.0 to 1.0).
        Default is 0.95 (95% confidence).

    normalize : bool or str
        Normalization method to apply before comparison:
        - True or 'zscore': Z-score normalization (standardization)
        - 'minmax': Min-max normalization to [0,1] range
        - False or 'none': No normalization
        Default is True.
    """

    def __init__(
        self,
        key_fields: Optional[List[str]] = None,
        value_field: Optional[str] = None,
        aggregation: str = "sum",
        confidence_level: float = 0.95,
        normalize: bool = True,
    ):
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation
        self.confidence_level = validate_confidence_level(confidence_level)
        self.normalize = normalize
        self.alpha = 1 - self.confidence_level  # Significance level

    def calculate_metric(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate the KS statistic between the original and transformed datasets.

        Returns:
        --------
        Dict[str, Any]
            Enhanced results dictionary with:
            - ks_statistic: float
            - p_value: float
            - max_difference: float (D-statistic)
            - interpretation: str
            - effect_size: str
            - confidence_interval: Tuple[float, float]
            - statistical_significance: bool
            - normalization_applied: str
        """
        if self.key_fields:
            # Use grouped + aggregated comparison
            orig_dist = create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            trans_dist = create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )

            ks_stat, p_value = self._calculate_ks_from_dicts(orig_dist, trans_dist)
            n_effective = max(len(orig_dist), len(trans_dist))
        else:
            # Fallback to standard KS test (1D distributions)
            columns = list(set(original_df.columns) & set(transformed_df.columns))
            if not columns:
                raise ValueError("No common columns to compare.")

            col = columns[0]
            orig_data = original_df[col].dropna().values
            trans_data = transformed_df[col].dropna().values

            # Apply normalization if requested
            if self.normalize:
                norm_method = "zscore" if self.normalize is True else self.normalize
                combined_data = np.concatenate([orig_data, trans_data])

                if norm_method == "zscore":
                    if np.std(combined_data) > 0:
                        orig_data = normalize_array_sklearn(orig_data, norm_method)
                        trans_data = normalize_array_sklearn(trans_data, norm_method)
                elif norm_method == "minmax":
                    if np.ptp(combined_data) > 0:
                        # Normalize both datasets using the same scale
                        combined_norm = normalize_array_sklearn(
                            combined_data, norm_method
                        )
                        orig_data = combined_norm[: len(orig_data)]
                        trans_data = combined_norm[len(orig_data) :]

            ks_stat, p_value = stats.ks_2samp(orig_data, trans_data)
            n_effective = min(len(orig_data), len(trans_data))

        # Calculate additional metrics
        effect_size = self._calculate_effect_size(ks_stat, n_effective)
        confidence_interval = self._calculate_confidence_interval(ks_stat, n_effective)
        statistical_significance = p_value < self.alpha

        return {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "max_difference": ks_stat,
            "interpretation": self._interpret_ks(p_value),
            "effect_size": effect_size,
            "confidence_interval": confidence_interval,
            "statistical_significance": statistical_significance,
            "normalization_applied": self._get_normalization_description(),
            "confidence_level": self.confidence_level,
            "alpha": self.alpha,
        }

    def _calculate_ks_from_dicts(
        self, dict1: Dict[str, float], dict2: Dict[str, float]
    ) -> Tuple[float, float]:
        """Enhanced KS calculation with normalization support for dictionaries."""
        all_keys = sorted(set(dict1) | set(dict2))

        # Extract values for normalization
        values1 = np.array([dict1.get(key, 0.0) for key in all_keys])
        values2 = np.array([dict2.get(key, 0.0) for key in all_keys])

        # Apply normalization if requested
        if self.normalize:
            norm_method = "zscore" if self.normalize is True else self.normalize
            combined_values = np.concatenate([values1, values2])

            if norm_method == "zscore" and np.std(combined_values) > 0:
                values1 = normalize_array_sklearn(values1, norm_method)
                values2 = normalize_array_sklearn(values2, norm_method)
            elif norm_method == "minmax" and np.ptp(combined_values) > 0:
                combined_norm = normalize_array_sklearn(combined_values, norm_method)
                values1 = combined_norm[: len(values1)]
                values2 = combined_norm[len(values1) :]

        # Calculate totals for probability normalization
        total1 = np.sum(np.abs(values1))
        total2 = np.sum(np.abs(values2))

        if total1 == 0 or total2 == 0:
            return 0.0, 1.0

        # Compute cumulative distributions
        cumulative1 = 0.0
        cumulative2 = 0.0
        max_diff = 0.0

        for i, key in enumerate(all_keys):
            cumulative1 += values1[i] / total1
            cumulative2 += values2[i] / total2
            diff = abs(cumulative1 - cumulative2)
            max_diff = max(max_diff, diff)

        # Enhanced p-value calculation
        n1 = len(dict1)
        n2 = len(dict2)
        en = np.sqrt(n1 * n2 / (n1 + n2)) if (n1 + n2) > 0 else 1e-8

        # Use more accurate p-value calculation
        c_alpha = np.sqrt(-0.5 * np.log(self.alpha / 2))
        critical_value = c_alpha / en

        if max_diff <= critical_value:
            # More accurate p-value for small differences
            p_value = 2 * np.exp(-2 * en**2 * max_diff**2)
        else:
            # Use asymptotic distribution for large differences
            p_value = 2 * np.sum(
                [
                    (-1) ** i * np.exp(-2 * i**2 * en**2 * max_diff**2)
                    for i in range(1, 100)
                ]
            )

        return max_diff, min(p_value, 1.0)

    def _calculate_effect_size(self, ks_stat: float, n: int) -> str:
        """
        Calculate effect size interpretation based on KS statistic.

        Parameters:
        ----------
        ks_stat : float
            KS statistic value
        n : int
            Effective sample size

        Returns:
        --------
        str
            Effect size interpretation
        """
        # Effect size thresholds adjusted for confidence level
        base_small = 0.1
        base_medium = 0.3
        base_large = 0.5

        # Adjust thresholds based on confidence level
        confidence_factor = 1 + (self.confidence_level - 0.95) * 0.2
        small_threshold = base_small * confidence_factor
        medium_threshold = base_medium * confidence_factor
        large_threshold = base_large * confidence_factor

        if ks_stat < small_threshold:
            return "negligible"
        elif ks_stat < medium_threshold:
            return "small"
        elif ks_stat < large_threshold:
            return "medium"
        else:
            return "large"

    def _calculate_confidence_interval(
        self, ks_stat: float, n: int
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the KS statistic.

        Parameters:
        ----------
        ks_stat : float
            KS statistic value
        n : int
            Effective sample size

        Returns:
        --------
        Tuple[float, float]
            Lower and upper bounds of confidence interval
        """
        # Standard error approximation
        se = np.sqrt(1.0 / (2 * n)) if n > 0 else 0.1

        # Critical value for the specified confidence level
        z_score = stats.norm.ppf(1 - self.alpha / 2)

        margin_of_error = z_score * se
        lower_bound = max(0.0, ks_stat - margin_of_error)
        upper_bound = min(1.0, ks_stat + margin_of_error)

        return (lower_bound, upper_bound)

    def _get_normalization_description(self) -> str:
        """Get description of normalization method applied."""
        if self.normalize is False or self.normalize == "none":
            return "none"
        elif self.normalize is True or self.normalize == "zscore":
            return "z-score standardization"
        elif self.normalize == "minmax":
            return "min-max normalization"
        else:
            return str(self.normalize)

    def _interpret_ks(self, p_value: float) -> str:
        """
        Enhanced interpretation of KS test results based on confidence level.

        Parameters:
        ----------
        p_value : float
            The p-value from the KS test

        Returns:
        --------
        str
            Detailed interpretation with confidence level context
        """
        alpha = self.alpha
        confidence_pct = int(self.confidence_level * 100)

        if p_value < alpha / 10:  # Very strong evidence
            return f"Very strong evidence of distributional difference (p < {alpha/10:.3f}, {confidence_pct}% confidence)"
        elif p_value < alpha / 2:  # Strong evidence
            return f"Strong evidence of distributional difference (p < {alpha/2:.3f}, {confidence_pct}% confidence)"
        elif p_value < alpha:  # Moderate evidence
            return f"Moderate evidence of distributional difference (p < {alpha:.3f}, {confidence_pct}% confidence)"
        else:  # Insufficient evidence
            return f"Insufficient evidence of distributional difference (p ≥ {alpha:.3f}, {confidence_pct}% confidence)"
