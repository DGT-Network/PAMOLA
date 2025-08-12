"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Privacy Metric Operation - Uniqueness
Package:       pamola_core.metrics.privacy
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-07-17
License:       BSD 3-Clause

Description:
    This module implements the Uniqueness privacy metrics, including k-anonymity,
    l-diversity, and t-closeness, to assess uniqueness and re-identification risks
    in tabular datasets. These metrics help quantify privacy risk and guide data
    anonymization strategies for synthetic and real data.

Key Features:
    - K-anonymity, l-diversity, and t-closeness metrics
    - Support for multiple sensitive attributes and quasi-identifiers
    - Configurable thresholds and aggregation
    - Detailed violation statistics and privacy risk assessment
    - Seamless integration into PAMOLA.CORE metric pipelines
"""

from typing import Any, Dict, List
import pandas as pd

from scipy.stats import wasserstein_distance


class Uniqueness:
    """
    Uniqueness Privacy Metrics Implementation

    Implements core privacy metrics to assess uniqueness and re-identification risks:
    - K-anonymity: Basic privacy protection
    - L-diversity: Diversity of sensitive attributes (if enabled)
    - T-closeness: Distribution similarity between groups and population (if enabled)

    Parameters:
    - quasi_identifiers: List[str] - Columns used for identification
    - sensitive_attributes: List[str] or str - Sensitive columns to protect
    - k_values: List[int] - K values for k-anonymity analysis
    - l_diversity: bool - Whether to compute l-diversity
    - t_closeness: bool - Whether to compute t-closeness
    """

    def __init__(
        self,
        quasi_identifiers: List[str],
        sensitives: List[str],
        k_values: List[int] = [2, 3, 5, 10],
        l_diversity: bool = True,
        t_closeness: bool = True,
    ):
        self.quasi_identifiers = quasi_identifiers
        self.sensitives = sensitives
        self.k_values = k_values
        self.l_diversity = l_diversity
        self.t_closeness = t_closeness

    def calculate_metric(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate privacy metrics (k-anonymity, l-diversity, t-closeness) for the given DataFrame.

        Args:
            df (pd.DataFrame): Input tabular data to assess privacy risks.
        Returns:
            Dict[str, Any]: Dictionary containing k-anonymity, l-diversity, and t-closeness statistics.
        """
        # Group by QID
        grouped = df.groupby(self.quasi_identifiers, observed=True)

        # Check k-anonymity
        group_sizes = grouped.size()
        # Calculate identity disclosure rate
        num_unique_records = (group_sizes == 1).sum()
        total_records = len(df)
        identity_disclosure_rate = (
            num_unique_records / total_records if total_records > 0 else 0
        )

        k_anonymity_stats = {
            "identity_disclosure_rate": identity_disclosure_rate,
            "num_groups": len(group_sizes),
            "k_anonymity_stats": [],
        }

        for k_value in self.k_values:
            k_anonymity_stat = {
                "k_value": k_value,
                "num_violations": (group_sizes < k_value).sum(),
                "percent_violation": (
                    (group_sizes < k_value).sum() / len(group_sizes) * 100
                    if len(group_sizes) > 0
                    else 0
                ),
            }
            k_anonymity_stats["k_anonymity_stats"].append(k_anonymity_stat)

        # Calculate l-diversity
        l_diversity_stats = {}
        if self.l_diversity:
            l_diversity_stats = self._calculate_l_diversity(df)

        # Calculate t-closeness
        t_closeness_stats = {}
        if self.t_closeness:
            t_closeness_stats = self._calculate_t_closeness(df)

        # Compile results
        return {
            "k_anonymity": k_anonymity_stats,
            "l_diversity": l_diversity_stats,
            "t_closeness": t_closeness_stats,
        }

    def _calculate_l_diversity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate l-diversity statistics for each group defined by quasi-identifiers.

        Args:
            df (pd.DataFrame): Input tabular data.
        Returns:
            dict: l-diversity statistics including min, max, average, and distribution.
        """
        l_diversity_stats = {}

        grouped = df.groupby(self.quasi_identifiers, observed=True)
        diversity_counts = grouped.apply(
            lambda group: group[self.sensitives].drop_duplicates().shape[0]
        )

        l_diversity_stats = {
            "min_l_diversity": diversity_counts.min(),
            "max_l_diversity": diversity_counts.max(),
            "avg_l_diversity": diversity_counts.mean(),
            "l_diversity_distribution": diversity_counts.value_counts()
            .sort_index()
            .to_dict(),
        }

        return l_diversity_stats

    def _calculate_t_closeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate t-closeness scores for each group using Wasserstein distance.

        Args:
            df (pd.DataFrame): Input tabular data.
        Returns:
            dict: t-closeness statistics including min, max, average, and all scores.
        """
        t_closeness_stats = {}

        # Create a single tuple column for combinations of sensitive attributes
        df_combined = df.copy()
        df_combined["__sensitive_combo__"] = df_combined[self.sensitives].apply(
            tuple, axis=1
        )

        # Global distribution
        global_dist = df_combined["__sensitive_combo__"].value_counts(normalize=True)

        t_scores = []

        for _, group in df_combined.groupby(self.quasi_identifiers):
            group_dist = group["__sensitive_combo__"].value_counts(normalize=True)

            # Align both distributions
            aligned_index = global_dist.index.union(group_dist.index)
            aligned_global = global_dist.reindex(aligned_index, fill_value=0)
            aligned_group = group_dist.reindex(aligned_index, fill_value=0)

            # Compute Wasserstein distance (t-closeness)
            t_score = wasserstein_distance(
                range(len(aligned_index)),
                range(len(aligned_index)),
                u_weights=aligned_global.values,
                v_weights=aligned_group.values,
            )
            t_scores.append(t_score)

        t_closeness_stats = {
            "calculate_method": "wasserstein distance",
            "min_t_closeness": min(t_scores),
            "max_t_closeness": max(t_scores),
            "avg_t_closeness": sum(t_scores) / len(t_scores),
            "t_closeness_scores": t_scores,
        }

        return t_closeness_stats
