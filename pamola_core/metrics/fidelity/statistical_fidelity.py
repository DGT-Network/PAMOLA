"""
PAMOLA.CORE - Statistical Fidelity Metrics
------------------------------------------
This module provides metrics for assessing the statistical fidelity
of anonymized or synthetic datasets compared to original data. These
metrics measure how well statistical properties and relationships
are preserved during privacy transformations.

Key features:
- Mean, variance, and correlation preservation metrics
- Distribution similarity measures
- Overall statistical fidelity assessment
- Support for numerical and categorical data
- Column-level and dataset-level fidelity analysis

These metrics help data custodians understand how well anonymized
data preserves the statistical utility of the original data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats

from pamola_core.metrics.base import FidelityMetric, round_metric_values

# Configure logging
logger = logging.getLogger(__name__)


class StatisticalFidelityMetric(FidelityMetric):
    """
    Assesses the statistical fidelity of anonymized data.

    This class measures how well various statistical properties
    of the original data are preserved in the anonymized data,
    including means, variances, correlations, and distributions.

    The fidelity values are expressed as percentages, with higher
    values indicating better preservation of statistical properties.
    """

    def __init__(self,
                 mean_weight: float = 0.4,
                 variance_weight: float = 0.3,
                 correlation_weight: float = 0.3):
        """
        Initialize the statistical fidelity metric.

        Parameters:
        -----------
        mean_weight : float, optional
            Weight for mean preservation in overall calculation (default: 0.4).
        variance_weight : float, optional
            Weight for variance preservation in overall calculation (default: 0.3).
        correlation_weight : float, optional
            Weight for correlation preservation in overall calculation (default: 0.3).
        """
        super().__init__(
            name="Statistical Fidelity",
            description="Measures preservation of statistical properties after anonymization"
        )
        self.mean_weight = mean_weight
        self.variance_weight = variance_weight
        self.correlation_weight = correlation_weight

        # Ensure weights sum to 1
        total_weight = mean_weight + variance_weight + correlation_weight
        if abs(total_weight - 1.0) > 1e-10:
            self.mean_weight /= total_weight
            self.variance_weight /= total_weight
            self.correlation_weight /= total_weight

    def calculate(self, original_data: pd.DataFrame,
                  anonymized_data: pd.DataFrame,
                  columns: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate statistical fidelity metrics.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset.
        columns : list[str], optional
            List of columns to evaluate. If None, all numeric columns will be used.
        **kwargs : dict
            Additional parameters for calculation.

        Returns:
        --------
        dict
            Dictionary with statistical fidelity metrics:
            - "mean_preservation": How well means are preserved
            - "variance_preservation": How well variances are preserved
            - "correlation_preservation": How well correlations are preserved
            - "overall_fidelity": Overall statistical fidelity score
            - "column_level_fidelity": Column-by-column fidelity metrics
        """
        logger.info("Calculating statistical fidelity metrics")

        try:
            # Identify numeric columns in both datasets
            if columns is None:
                numeric_columns = [col for col in original_data.select_dtypes(include=['number']).columns
                                   if col in anonymized_data.columns]
            else:
                numeric_columns = [col for col in columns if col in original_data.columns
                                   and col in anonymized_data.columns
                                   and np.issubdtype(original_data[col].dtype, np.number)
                                   and np.issubdtype(anonymized_data[col].dtype, np.number)]

            if not numeric_columns:
                logger.warning("No common numeric columns found for fidelity calculation")
                return {
                    "mean_preservation": 0,
                    "variance_preservation": 0,
                    "correlation_preservation": 0,
                    "overall_fidelity": 0,
                    "column_level_fidelity": {}
                }

            # Calculate column-level fidelity
            column_fidelity = {}

            # Calculate mean preservation
            orig_means = original_data[numeric_columns].mean()
            anon_means = anonymized_data[numeric_columns].mean()

            mean_preservations = {}
            for col in numeric_columns:
                if abs(orig_means[col]) > 1e-10:  # Avoid division by zero
                    preservation = 100 * (1 - abs(orig_means[col] - anon_means[col]) / abs(orig_means[col]))
                else:
                    preservation = 100 if abs(anon_means[col]) < 1e-10 else 0

                mean_preservations[col] = min(100, max(0, preservation))

                # Initialize column fidelity dict
                column_fidelity[col] = {"mean_preservation": mean_preservations[col]}

            # Overall mean preservation
            mean_preservation = np.mean(list(mean_preservations.values())) if mean_preservations else 0

            # Calculate variance preservation
            orig_vars = original_data[numeric_columns].var()
            anon_vars = anonymized_data[numeric_columns].var()

            var_preservations = {}
            for col in numeric_columns:
                if orig_vars[col] > 1e-10:  # Avoid division by zero
                    preservation = 100 * (1 - abs(orig_vars[col] - anon_vars[col]) / orig_vars[col])
                else:
                    preservation = 100 if anon_vars[col] < 1e-10 else 0

                var_preservations[col] = min(100, max(0, preservation))

                # Add to column fidelity dict
                column_fidelity[col]["variance_preservation"] = var_preservations[col]

            # Overall variance preservation
            variance_preservation = np.mean(list(var_preservations.values())) if var_preservations else 0

            # Calculate correlation preservation
            correlation_preservation = 100  # Default if only one column
            if len(numeric_columns) > 1:
                orig_corr = original_data[numeric_columns].corr().values
                anon_corr = anonymized_data[numeric_columns].corr().values

                # Get upper triangle indices (excluding diagonal)
                indices = np.triu_indices(len(numeric_columns), k=1)

                orig_corrs = orig_corr[indices]
                anon_corrs = anon_corr[indices]

                # Calculate average difference
                corr_diff = np.mean(np.abs(orig_corrs - anon_corrs))
                correlation_preservation = max(0, min(100, 100 * (1 - corr_diff)))

                # Add pairwise correlation preservation to column fidelity
                for i, col1 in enumerate(numeric_columns):
                    for j, col2 in enumerate(numeric_columns):
                        if i < j:  # Upper triangle only
                            pair_diff = abs(orig_corr[i, j] - anon_corr[i, j])
                            pair_preservation = max(0, min(100, 100 * (1 - pair_diff)))

                            # Add to both columns
                            if "correlation_with" not in column_fidelity[col1]:
                                column_fidelity[col1]["correlation_with"] = {}
                            if "correlation_with" not in column_fidelity[col2]:
                                column_fidelity[col2]["correlation_with"] = {}

                            column_fidelity[col1]["correlation_with"][col2] = pair_preservation
                            column_fidelity[col2]["correlation_with"][col1] = pair_preservation

            # Calculate overall fidelity (weighted average)
            overall_fidelity = (self.mean_weight * mean_preservation +
                                self.variance_weight * variance_preservation +
                                self.correlation_weight * correlation_preservation)

            # Calculate additional distribution similarity metrics if requested
            if kwargs.get("distribution_tests", False):
                distribution_similarity = self._calculate_distribution_similarity(
                    original_data, anonymized_data, numeric_columns
                )

                # Add to result
                for col, metrics in distribution_similarity.items():
                    if col in column_fidelity:
                        column_fidelity[col]["distribution_tests"] = metrics

            # Prepare result
            result = {
                "mean_preservation": mean_preservation,
                "variance_preservation": variance_preservation,
                "correlation_preservation": correlation_preservation,
                "overall_fidelity": overall_fidelity,
                "column_level_fidelity": column_fidelity,
                "weights": {
                    "mean_weight": self.mean_weight,
                    "variance_weight": self.variance_weight,
                    "correlation_weight": self.correlation_weight
                }
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Statistical fidelity analysis: Overall fidelity = {overall_fidelity:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Error during statistical fidelity calculation: {e}")
            raise

    def _calculate_distribution_similarity(self, original_data: pd.DataFrame,
                                           anonymized_data: pd.DataFrame,
                                           columns: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate distribution similarity metrics.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset.
        columns : list[str]
            List of columns to evaluate.

        Returns:
        --------
        dict
            Dictionary with distribution similarity metrics for each column.
        """
        result = {}

        for col in columns:
            # Extract non-null values
            orig_values = original_data[col].dropna().values
            anon_values = anonymized_data[col].dropna().values

            # Skip if too few values
            if len(orig_values) < 5 or len(anon_values) < 5:
                continue

            # Initialize metrics dictionary
            col_metrics = {}

            try:
                # Kolmogorov-Smirnov test for distribution similarity
                ks_stat, ks_pvalue = stats.ks_2samp(orig_values, anon_values)

                # Convert to similarity percentage (lower KS statistic = higher similarity)
                ks_similarity = max(0, min(100, 100 * (1 - ks_stat)))

                col_metrics["ks_similarity"] = ks_similarity
                col_metrics["ks_pvalue"] = ks_pvalue

                # Mann-Whitney U test for similarity of distributions
                try:
                    mw_stat, mw_pvalue = stats.mannwhitneyu(orig_values, anon_values, alternative='two-sided')
                    col_metrics["mw_pvalue"] = mw_pvalue
                except:
                    # Mann-Whitney can fail for some distributions
                    pass

                # Add percentiles comparison
                orig_percentiles = np.percentile(orig_values, [25, 50, 75])
                anon_percentiles = np.percentile(anon_values, [25, 50, 75])

                percentile_diffs = []
                for i, p in enumerate([25, 50, 75]):
                    if abs(orig_percentiles[i]) > 1e-10:
                        diff = abs(orig_percentiles[i] - anon_percentiles[i]) / abs(orig_percentiles[i])
                        percentile_diffs.append(diff)

                if percentile_diffs:
                    percentile_similarity = 100 * (1 - np.mean(percentile_diffs))
                    col_metrics["percentile_similarity"] = max(0, min(100, percentile_similarity))

                # Add result for this column
                result[col] = col_metrics

            except Exception as e:
                logger.warning(f"Error calculating distribution similarity for column {col}: {e}")

        return result

    def interpret(self, value: float) -> str:
        """
        Interpret a statistical fidelity value.

        Parameters:
        -----------
        value : float
            The statistical fidelity value (percentage).

        Returns:
        --------
        str
            Human-readable interpretation of the statistical fidelity.
        """
        if value < 50:
            return f"Statistical Fidelity: {value:.2f}% - Poor preservation of statistical properties"
        elif value < 70:
            return f"Statistical Fidelity: {value:.2f}% - Fair preservation of statistical properties"
        elif value < 85:
            return f"Statistical Fidelity: {value:.2f}% - Good preservation of statistical properties"
        elif value < 95:
            return f"Statistical Fidelity: {value:.2f}% - Very good preservation of statistical properties"
        else:
            return f"Statistical Fidelity: {value:.2f}% - Excellent preservation of statistical properties"


class DistributionFidelityMetric(FidelityMetric):
    """
    Specialized metric for measuring the fidelity of data distributions.

    This metric focuses on how well the distributions of individual columns
    are preserved in the anonymized data, using various statistical tests
    and distribution comparison methods.
    """

    def __init__(self):
        """
        Initialize the distribution fidelity metric.
        """
        super().__init__(
            name="Distribution Fidelity",
            description="Measures preservation of data distributions after anonymization"
        )

    def calculate(self, original_data: pd.DataFrame,
                  anonymized_data: pd.DataFrame,
                  columns: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate distribution fidelity metrics.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset.
        columns : list[str], optional
            List of columns to evaluate. If None, all numeric columns will be used.
        **kwargs : dict
            Additional parameters for calculation.

        Returns:
        --------
        dict
            Dictionary with distribution fidelity metrics:
            - "overall_distribution_fidelity": Average fidelity across all columns
            - "column_distribution_fidelity": Fidelity for each column
            - "best_preserved_column": Column with highest distribution fidelity
            - "worst_preserved_column": Column with lowest distribution fidelity
        """
        logger.info("Calculating distribution fidelity metrics")

        try:
            # Identify columns to analyze
            if columns is None:
                numeric_columns = [col for col in original_data.select_dtypes(include=['number']).columns
                                   if col in anonymized_data.columns]
            else:
                numeric_columns = [col for col in columns if col in original_data.columns
                                   and col in anonymized_data.columns]

            if not numeric_columns:
                logger.warning("No common columns found for distribution fidelity calculation")
                return {
                    "overall_distribution_fidelity": 0,
                    "column_distribution_fidelity": {},
                }

            # Calculate column-level distribution fidelity
            column_fidelity = {}

            for col in numeric_columns:
                # Skip columns with too few values
                if original_data[col].count() < 5 or anonymized_data[col].count() < 5:
                    continue

                # Get values (handle categorical vs numeric differently)
                if np.issubdtype(original_data[col].dtype, np.number):
                    # Numeric column: use continuous distribution comparison
                    orig_values = original_data[col].dropna().values
                    anon_values = anonymized_data[col].dropna().values

                    # KS test
                    ks_stat, ks_pvalue = stats.ks_2samp(orig_values, anon_values)
                    ks_similarity = max(0, min(100, 100 * (1 - ks_stat)))

                    # Percentile comparison
                    orig_percentiles = np.percentile(orig_values, [10, 25, 50, 75, 90])
                    anon_percentiles = np.percentile(anon_values, [10, 25, 50, 75, 90])

                    percentile_diffs = []
                    for i, p in enumerate([10, 25, 50, 75, 90]):
                        if abs(orig_percentiles[i]) > 1e-10:
                            diff = abs(orig_percentiles[i] - anon_percentiles[i]) / abs(orig_percentiles[i])
                            percentile_diffs.append(diff)

                    percentile_similarity = 100 * (1 - np.mean(percentile_diffs)) if percentile_diffs else 0
                    percentile_similarity = max(0, min(100, percentile_similarity))

                    # Overall distribution similarity (weighted average)
                    distribution_similarity = 0.6 * ks_similarity + 0.4 * percentile_similarity

                    column_fidelity[col] = {
                        "distribution_fidelity": distribution_similarity,
                        "ks_similarity": ks_similarity,
                        "percentile_similarity": percentile_similarity,
                    }
                else:
                    # Categorical column: compare frequency distributions
                    orig_counts = original_data[col].value_counts(normalize=True)
                    anon_counts = anonymized_data[col].value_counts(normalize=True)

                    # Combine all categories from both datasets
                    all_categories = set(orig_counts.index) | set(anon_counts.index)

                    # Calculate Jensen-Shannon divergence
                    orig_probs = np.array([orig_counts.get(cat, 0) for cat in all_categories])
                    anon_probs = np.array([anon_counts.get(cat, 0) for cat in all_categories])

                    # Normalize if needed
                    if sum(orig_probs) > 0:
                        orig_probs = orig_probs / sum(orig_probs)
                    if sum(anon_probs) > 0:
                        anon_probs = anon_probs / sum(anon_probs)

                    # Calculate JS divergence
                    try:
                        js_divergence = _jensen_shannon_divergence(orig_probs, anon_probs)
                        js_similarity = max(0, min(100, 100 * (1 - js_divergence)))
                    except:
                        js_similarity = 0

                    # Calculate chi-square test
                    try:
                        # Convert to counts for chi-square test
                        orig_counts_abs = original_data[col].value_counts()
                        anon_counts_abs = anonymized_data[col].value_counts()

                        # Get observed and expected frequencies
                        observed = []
                        expected = []

                        for cat in set(orig_counts_abs.index) | set(anon_counts_abs.index):
                            observed.append(anon_counts_abs.get(cat, 0))
                            expected.append(orig_counts_abs.get(cat, 0))

                        # Ensure expected has no zeros
                        expected = [max(1, e) for e in expected]

                        # Calculate chi-square statistic
                        chi2_stat = sum([(o - e) ** 2 / e for o, e in zip(observed, expected)])

                        # Convert to similarity percentage (lower chi2 = higher similarity)
                        chi2_similarity = max(0, min(100, 100 / (1 + chi2_stat)))
                    except:
                        chi2_similarity = 0

                    # Overall distribution similarity (weighted average)
                    distribution_similarity = 0.7 * js_similarity + 0.3 * chi2_similarity

                    column_fidelity[col] = {
                        "distribution_fidelity": distribution_similarity,
                        "js_similarity": js_similarity,
                        "chi2_similarity": chi2_similarity,
                    }

            # Calculate overall distribution fidelity
            overall_fidelity = np.mean([metrics["distribution_fidelity"]
                                        for metrics in column_fidelity.values()]) if column_fidelity else 0

            # Find best and worst preserved columns
            best_column = None
            worst_column = None

            if column_fidelity:
                best_column = max(column_fidelity.items(),
                                  key=lambda x: x[1]["distribution_fidelity"])[0]
                worst_column = min(column_fidelity.items(),
                                   key=lambda x: x[1]["distribution_fidelity"])[0]

            # Prepare result
            result = {
                "overall_distribution_fidelity": overall_fidelity,
                "column_distribution_fidelity": column_fidelity,
                "best_preserved_column": best_column,
                "worst_preserved_column": worst_column
            }

            # Round numeric values for readability
            result = round_metric_values(result)

            # Store the result
            self.last_result = result

            logger.info(f"Distribution fidelity analysis: Overall fidelity = {overall_fidelity:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Error during distribution fidelity calculation: {e}")
            raise


# Helper function for calculating Jensen-Shannon divergence
def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.

    Parameters:
    -----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.

    Returns:
    --------
    float
        Jensen-Shannon divergence value.
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate midpoint distribution
    m = 0.5 * (p + q)

    # Calculate KL divergences
    kl_p_m = np.sum(p * np.log2(p / m))
    kl_q_m = np.sum(q * np.log2(q / m))

    # Jensen-Shannon divergence
    js = 0.5 * (kl_p_m + kl_q_m)

    return js


# Convenience function for calculating all fidelity metrics
def calculate_fidelity_metrics(original_data: pd.DataFrame,
                               anonymized_data: pd.DataFrame,
                               **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Calculate multiple fidelity metrics for anonymized data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        The original dataset.
    anonymized_data : pd.DataFrame
        The anonymized dataset.
    **kwargs : dict
        Additional parameters for calculation.

    Returns:
    --------
    dict
        Dictionary with results from all fidelity metrics.
    """
    results = {}

    # Calculate statistical fidelity
    stat_fidelity = StatisticalFidelityMetric(
        mean_weight=kwargs.get('mean_weight', 0.4),
        variance_weight=kwargs.get('variance_weight', 0.3),
        correlation_weight=kwargs.get('correlation_weight', 0.3)
    )
    results["statistical_fidelity"] = stat_fidelity.calculate(
        original_data, anonymized_data, kwargs.get('columns'),
        distribution_tests=kwargs.get('distribution_tests', False)
    )

    # Calculate distribution fidelity
    dist_fidelity = DistributionFidelityMetric()
    results["distribution_fidelity"] = dist_fidelity.calculate(
        original_data, anonymized_data, kwargs.get('columns')
    )

    return results