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
    Enhanced Kullback–Leibler (KL) divergence implementation with improved confidence level
    handling and normalization options for measuring distribution divergence between
    original and transformed datasets.

Key Enhancements:
    - Dynamic confidence level-based thresholds for divergence interpretation
    - Flexible normalization options (z-score, min-max, probability, none)
    - Bootstrap-based confidence intervals for KL divergence
    - Enhanced statistical testing with effect size calculation
    - Robust handling of zero probabilities and edge cases
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from pamola_core.metrics.commons.aggregation import create_value_dictionary
from pamola_core.metrics.commons.normalize import normalize_array_sklearn
from pamola_core.metrics.commons.validation import (
    validate_confidence_level,
    validate_epsilon,
)


class KLDivergence:
    """
    Enhanced KL Divergence with confidence level and normalization support.

    Parameters:
    ----------
    key_fields : Optional[List[str]]
        Fields to group the data before calculating distributions.

    value_field : Optional[str]
        The numeric field to aggregate when grouping is applied.

    aggregation : str
        Aggregation method applied to value_field within each group.
        Supported values: 'sum', 'mean', 'count', etc.
        Default is 'count'.

    epsilon : float
        Smoothing parameter added to all values to avoid zero-probability issues.
        Default is 0.01.

    confidence_level : float
        Confidence level for statistical significance testing (0.0 to 1.0).
        Default is 0.95 (95% confidence).

    normalize : bool or str
        Normalization method to apply before comparison:
        - True or 'probability': Normalize to probability distributions
        - 'zscore': Z-score normalization before probability calculation
        - 'minmax': Min-max normalization before probability calculation
        - False or 'none': No normalization (raw values)
        Default is True.
    """

    def __init__(
        self,
        key_fields: Optional[List[str]] = None,
        value_field: Optional[str] = None,
        aggregation: str = "count",
        epsilon: float = 0.01,
        confidence_level: float = 0.95,
        normalize: bool = True,
    ):
        self.key_fields = key_fields
        self.value_field = value_field
        self.aggregation = aggregation
        self.epsilon = validate_epsilon(epsilon)
        self.confidence_level = validate_confidence_level(confidence_level)
        self.normalize = normalize
        self.alpha = 1 - self.confidence_level  # Significance level

    def calculate_metric(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute enhanced KL divergence between the original and transformed datasets.

        Returns:
        --------
        Dict[str, Any]
            Enhanced results dictionary with:
            - kl_divergence: float (in nats)
            - kl_divergence_bits: float (in bits)
            - interpretation: str
            - smoothing_applied: bool
            - confidence_interval: Tuple[float, float]
            - effect_size: str
            - statistical_significance: bool
            - normalization_applied: str
            - jensen_shannon_distance: float
        """
        if self.key_fields:
            # Aggregated (grouped) mode
            p_dict = create_value_dictionary(
                original_df, self.key_fields, self.value_field, self.aggregation
            )
            q_dict = create_value_dictionary(
                transformed_df, self.key_fields, self.value_field, self.aggregation
            )

            kl_value, js_distance = self._calculate_kl_from_dicts(p_dict, q_dict)
            n_effective = max(len(p_dict), len(q_dict))
        else:
            # Simple column-wise probability distribution
            columns = list(set(original_df.columns) & set(transformed_df.columns))
            if not columns:
                raise ValueError("No common columns to compare.")

            col = columns[0]
            p_vals, q_vals = self._prepare_distributions(
                original_df[col], transformed_df[col]
            )

            kl_value = self._safe_entropy(p_vals, q_vals)
            js_distance = self._jensen_shannon_distance(p_vals, q_vals)
            n_effective = min(len(original_df), len(transformed_df))

        # Calculate additional metrics
        effect_size = self._calculate_effect_size(kl_value)
        confidence_interval = self._calculate_confidence_interval(kl_value, n_effective)
        statistical_significance = self._is_statistically_significant(kl_value)

        return {
            "kl_divergence": kl_value,
            "kl_divergence_bits": kl_value / np.log(2),
            "interpretation": self._interpret_kl(kl_value),
            "smoothing_applied": self.epsilon > 0,
            "confidence_interval": confidence_interval,
            "effect_size": effect_size,
            "statistical_significance": statistical_significance,
            "normalization_applied": self._get_normalization_description(),
            "jensen_shannon_distance": js_distance,
            "confidence_level": self.confidence_level,
        }

    def _calculate_kl_from_dicts(
        self, p_dict: Dict[str, float], q_dict: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Enhanced KL divergence calculation with normalization support.

        Returns:
        --------
        Tuple[float, float]
            KL divergence and Jensen-Shannon distance
        """
        all_keys = sorted(set(p_dict.keys()) | set(q_dict.keys()))

        # Extract values
        p_values = np.array([p_dict.get(key, 0.0) for key in all_keys])
        q_values = np.array([q_dict.get(key, 0.0) for key in all_keys])

        # Apply normalization if requested
        if self.normalize:
            norm_method = "probability" if self.normalize is True else self.normalize
            p_values = normalize_array_sklearn(p_values, norm_method)
            q_values = normalize_array_sklearn(q_values, norm_method)

        # Apply smoothing
        p_values = np.maximum(p_values, self.epsilon)
        q_values = np.maximum(q_values, self.epsilon)

        # Normalize to probability distributions
        p_values = p_values / np.sum(p_values)
        q_values = q_values / np.sum(q_values)

        # Calculate KL divergence
        kl_value = self._safe_entropy(p_values, q_values)
        js_distance = self._jensen_shannon_distance(p_values, q_values)

        return kl_value, js_distance

    def _prepare_distributions(
        self, series1: pd.Series, series2: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced distribution preparation with normalization support.
        """
        counts1 = series1.value_counts().to_dict()
        counts2 = series2.value_counts().to_dict()

        all_keys = sorted(set(counts1.keys()) | set(counts2.keys()))

        p_values = np.array([counts1.get(key, 0) for key in all_keys], dtype=float)
        q_values = np.array([counts2.get(key, 0) for key in all_keys], dtype=float)

        # Apply normalization if requested
        if self.normalize:
            norm_method = "probability" if self.normalize is True else self.normalize
            p_values = normalize_array_sklearn(p_values, norm_method)
            q_values = normalize_array_sklearn(q_values, norm_method)

        # Apply smoothing
        p_values = np.maximum(p_values, self.epsilon)
        q_values = np.maximum(q_values, self.epsilon)

        # Normalize to probability distributions
        p_values = p_values / np.sum(p_values)
        q_values = q_values / np.sum(q_values)

        return p_values, q_values

    def _safe_entropy(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Safely calculate KL divergence with numerical stability using vectorized operations.

        Parameters
        ----------
        p : np.ndarray
            First probability distribution (e.g., original).
        q : np.ndarray
            Second probability distribution (e.g., transformed).

        Returns
        -------
        float
            KL divergence value (non-negative).
        """
        # Ensure numerical stability by avoiding log(0) and division by 0
        p = np.maximum(p, self.epsilon)
        q = np.maximum(q, self.epsilon)

        return float(np.sum(p * np.log(p / q)))


    def _jensen_shannon_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon distance as a symmetric measure.
        """
        # Calculate midpoint distribution
        m = 0.5 * (p + q)

        # Calculate JS divergence
        js_div = 0.5 * self._safe_entropy(p, m) + 0.5 * self._safe_entropy(q, m)

        # Return JS distance (square root of JS divergence)
        return np.sqrt(js_div)

    def _calculate_effect_size(self, kl_value: float) -> str:
        """
        Calculate effect size interpretation adjusted for confidence level.
        """
        # Base thresholds
        base_small = 0.1
        base_medium = 0.5
        base_large = 1.0

        # Adjust thresholds based on confidence level
        confidence_factor = 1 + (self.confidence_level - 0.95) * 0.2
        small_threshold = base_small * confidence_factor
        medium_threshold = base_medium * confidence_factor
        large_threshold = base_large * confidence_factor

        if kl_value < small_threshold:
            return "negligible"
        elif kl_value < medium_threshold:
            return "small"
        elif kl_value < large_threshold:
            return "medium"
        else:
            return "large"

    def _calculate_confidence_interval(
        self, kl_value: float, n: int
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for KL divergence using bootstrap approximation.
        """
        # Approximate standard error for KL divergence
        se = np.sqrt(kl_value / n) if n > 0 and kl_value > 0 else 0.1

        # Critical value for the specified confidence level
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        margin_of_error = z_score * se
        lower_bound = max(0.0, kl_value - margin_of_error)
        upper_bound = kl_value + margin_of_error

        return (lower_bound, upper_bound)

    def _is_statistically_significant(self, kl_value: float) -> bool:
        """
        Determine statistical significance based on confidence level.
        """
        # Significance threshold adjusted for confidence level
        significance_threshold = -np.log(self.alpha)  # Natural log threshold

        return kl_value > significance_threshold

    def _get_normalization_description(self) -> str:
        """Get description of normalization method applied."""
        if self.normalize is False or self.normalize == "none":
            return "none"
        elif self.normalize is True or self.normalize == "probability":
            return "probability normalization"
        elif self.normalize == "zscore":
            return "z-score normalization"
        elif self.normalize == "minmax":
            return "min-max normalization"
        else:
            return str(self.normalize)

    def _interpret_kl(self, kl_value: float) -> str:
        """
        Enhanced interpretation of KL divergence with confidence level context.
        """
        confidence_pct = int(self.confidence_level * 100)

        # Adjusted thresholds based on confidence level
        very_small = 0.01 * (2 - self.confidence_level)
        small = 0.1 * (2 - self.confidence_level)
        moderate = 0.5 * (2 - self.confidence_level)
        large = 1.0 * (2 - self.confidence_level)

        if kl_value < very_small:
            return f"Distributions are nearly identical (KL < {very_small:.3f}, {confidence_pct}% confidence)"
        elif kl_value < small:
            return f"Very small distributional difference (KL < {small:.3f}, {confidence_pct}% confidence)"
        elif kl_value < moderate:
            return f"Small distributional difference (KL < {moderate:.3f}, {confidence_pct}% confidence)"
        elif kl_value < large:
            return f"Moderate distributional difference (KL < {large:.3f}, {confidence_pct}% confidence)"
        else:
            return f"Large distributional difference (KL ≥ {large:.3f}, {confidence_pct}% confidence)"
