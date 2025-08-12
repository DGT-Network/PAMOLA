"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Privacy Process Metrics
Package:       pamola_core.anonymization.commons
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
 This module provides lightweight privacy metrics for monitoring and controlling
 the anonymization process in real-time. It focuses on fast, simple indicators
 that help track the progress and effectiveness of anonymization operations
 without heavy computational overhead.

Purpose:
 Serves as a real-time monitoring tool for anonymization operations, providing
 quick feedback on privacy levels, coverage, and basic risk indicators during
 the anonymization process. NOT intended for final quality assessment or
 detailed risk analysis.

Key Features:
 - Fast coverage and suppression metrics
 - Quick k-anonymity checks (minimum k only)
 - Simple generalization level indicators
 - Basic group size distribution
 - Threshold monitoring for process control
 - Lightweight risk indicators

Design Principles:
 - Performance: Each function completes in <100ms on 100K records
 - Simplicity: No complex statistics or ML algorithms
 - Focus: Process monitoring only, not final assessment
 - Independence: Minimal dependencies on other modules

Usage:
 Used by anonymization operations during batch processing to monitor progress,
 detect issues early, and make adaptive decisions during the anonymization
 process.

Dependencies:
 - numpy: For basic numerical operations
 - pandas: For data manipulation
 - logging: For process monitoring

Changelog:
 2.0.0 - Complete rewrite focusing on lightweight process metrics
 1.0.0 - Initial implementation (deprecated - too heavy)
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Constants for process monitoring
DEFAULT_K_THRESHOLD = 5
DEFAULT_SUPPRESSION_WARNING = 0.2  # Warn if >20% suppressed
DEFAULT_COVERAGE_TARGET = 0.95  # Target 95% coverage
EPSILON = 1e-10  # Small constant to avoid division by zero


def calculate_anonymization_coverage(
    original: pd.Series, anonymized: pd.Series
) -> Dict[str, float]:
    """
    Calculate the coverage of anonymization process.

    Fast metric to track what percentage of values were successfully anonymized
    vs. suppressed or left unchanged.

    Parameters:
    -----------
    original : pd.Series
        Original data before anonymization
    anonymized : pd.Series
        Data after anonymization

    Returns:
    --------
    Dict[str, float]
        Coverage metrics including:
        - total_coverage: Percentage of non-null anonymized values
        - changed_ratio: Percentage of values that were modified
        - suppressed_ratio: Percentage of values set to null
        - unchanged_ratio: Percentage of values left unchanged
    """
    try:
        total_count = len(original)
        if total_count == 0:
            return {
                "total_coverage": 0.0,
                "changed_ratio": 0.0,
                "suppressed_ratio": 0.0,
                "unchanged_ratio": 0.0,
            }

        # Count nulls
        original_nulls = original.isnull().sum()
        anonymized_nulls = anonymized.isnull().sum()

        # Calculate suppressed (new nulls)
        suppressed = max(0, anonymized_nulls - original_nulls)

        # Calculate changed values (excluding nulls)
        mask = ~(original.isnull() | anonymized.isnull())
        changed = (original[mask] != anonymized[mask]).sum()  # type: ignore

        # Calculate ratios
        non_null_original = total_count - original_nulls

        return {
            "total_coverage": float((total_count - anonymized_nulls) / total_count),
            "changed_ratio": (
                float(changed / non_null_original) if non_null_original > 0 else 0.0
            ),
            "suppressed_ratio": (
                float(suppressed / non_null_original) if non_null_original > 0 else 0.0
            ),
            "unchanged_ratio": (
                float((non_null_original - changed - suppressed) / non_null_original)
                if non_null_original > 0
                else 0.0
            ),
        }

    except Exception as e:
        logger.error(f"Error calculating coverage: {e}")
        return {
            "total_coverage": 0.0,
            "changed_ratio": 0.0,
            "suppressed_ratio": 0.0,
            "unchanged_ratio": 0.0,
            "error": str(e),
        }


def calculate_suppression_rate(
    series: pd.Series, original_nulls: Optional[int] = None
) -> float:
    """
    Calculate the suppression rate in anonymized data.

    Quick check for how many values were suppressed (set to null) during
    the anonymization process.

    Parameters:
    -----------
    series : pd.Series
        Anonymized data series
    original_nulls : Optional[int]
        Number of nulls in original data (if known)

    Returns:
    --------
    float
        Suppression rate [0.0, 1.0]
    """
    try:
        total_count = len(series)
        if total_count == 0:
            return 0.0

        current_nulls = series.isnull().sum()

        if original_nulls is not None:
            # Calculate new suppressions only
            new_suppressions = max(0, current_nulls - original_nulls)
            non_null_original = total_count - original_nulls
            return (
                float(new_suppressions / non_null_original)
                if non_null_original > 0
                else 0.0
            )
        else:
            # Return total null rate
            return float(current_nulls / total_count)

    except Exception as e:
        logger.error(f"Error calculating suppression rate: {e}")
        return 0.0


def get_group_size_distribution(
    df: pd.DataFrame, quasi_identifiers: List[str], max_groups: int = 100
) -> Dict[str, Any]:
    """
    Get quick distribution of group sizes for quasi-identifiers.

    Fast calculation of how records are distributed across equivalence classes.
    Limited to top groups for performance.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to analyze
    quasi_identifiers : List[str]
        List of quasi-identifier columns
    max_groups : int
        Maximum number of groups to analyze (for performance)

    Returns:
    --------
    Dict[str, Any]
        Distribution info including group sizes and counts
    """
    try:
        if not quasi_identifiers or not all(
            col in df.columns for col in quasi_identifiers
        ):
            return {"error": "Invalid quasi-identifiers"}

        # Get group sizes (limited for performance)
        group_sizes = df.groupby(quasi_identifiers).size()

        # If too many groups, sample them
        if len(group_sizes) > max_groups:
            group_sizes = group_sizes.nlargest(max_groups)

        # Get size distribution
        size_counts = Counter(group_sizes.values)

        return {
            "total_groups": len(group_sizes),
            "size_distribution": dict(size_counts),
            "min_size": int(group_sizes.min()) if len(group_sizes) > 0 else 0,
            "max_size": int(group_sizes.max()) if len(group_sizes) > 0 else 0,
            "mean_size": float(group_sizes.mean()) if len(group_sizes) > 0 else 0.0,
        }

    except Exception as e:
        logger.error(f"Error getting group distribution: {e}")
        return {"error": str(e)}


def calculate_min_group_size(
    df: pd.DataFrame, quasi_identifiers: List[str], sample_size: Optional[int] = 10000
) -> int:
    """
    Calculate minimum group size (k) for quasi-identifiers.

    Fast check of the minimum k-anonymity level. Uses sampling for large
    datasets to maintain performance.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to analyze
    quasi_identifiers : List[str]
        List of quasi-identifier columns
    sample_size : Optional[int]
        Sample size for large datasets (None = use all data)

    Returns:
    --------
    int
        Minimum group size (k value)
    """
    try:
        if not quasi_identifiers or not all(
            col in df.columns for col in quasi_identifiers
        ):
            return 0

        # Sample for performance if needed
        if sample_size and len(df) > sample_size:
            working_df = df.sample(n=sample_size, random_state=42)
        else:
            working_df = df

        # Quick group size calculation
        group_sizes = working_df.groupby(quasi_identifiers).size()

        return int(group_sizes.min()) if len(group_sizes) > 0 else 0

    except Exception as e:
        logger.error(f"Error calculating min group size: {e}")
        return 0


def calculate_vulnerable_records_ratio(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    k_threshold: int = DEFAULT_K_THRESHOLD,
    sample_size: Optional[int] = 10000,
) -> float:
    """
    Calculate ratio of vulnerable records (k < threshold).

    Quick assessment of how many records are in small groups that don't
    meet the k-anonymity threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to analyze
    quasi_identifiers : List[str]
        List of quasi-identifier columns
    k_threshold : int
        Minimum acceptable group size
    sample_size : Optional[int]
        Sample size for large datasets

    Returns:
    --------
    float
        Ratio of vulnerable records [0.0, 1.0]
    """
    try:
        if not quasi_identifiers or not all(
            col in df.columns for col in quasi_identifiers
        ):
            return 1.0  # All records vulnerable if no QIs

        # Sample for performance
        if sample_size and len(df) > sample_size:
            working_df = df.sample(n=sample_size, random_state=42)
        else:
            working_df = df

        total_records = len(working_df)
        if total_records == 0:
            return 0.0

        # Calculate group sizes
        group_sizes = working_df.groupby(quasi_identifiers).size()

        # Count vulnerable records
        vulnerable_groups = group_sizes[group_sizes < k_threshold]
        vulnerable_records = vulnerable_groups.sum()

        return float(vulnerable_records / total_records)

    except Exception as e:
        logger.error(f"Error calculating vulnerable ratio: {e}")
        return 1.0  # Assume all vulnerable on error


def calculate_generalization_level(
    original: pd.Series, generalized: pd.Series
) -> float:
    """
    Calculate the level of generalization applied.

    Simple metric showing how much the cardinality was reduced through
    generalization.

    Parameters:
    -----------
    original : pd.Series
        Original data
    generalized : pd.Series
        Generalized data

    Returns:
    --------
    float
        Generalization level [0.0, 1.0] where 1.0 = maximum generalization
    """
    try:
        original_unique = original.nunique()
        generalized_unique = generalized.nunique()

        if original_unique <= 1:
            return 0.0  # No generalization possible

        # Calculate reduction ratio
        reduction = (original_unique - generalized_unique) / (original_unique - 1)

        return float(max(0.0, min(1.0, reduction)))

    except Exception as e:
        logger.error(f"Error calculating generalization level: {e}")
        return 0.0


def calculate_value_reduction_ratio(
    original: pd.Series, anonymized: pd.Series
) -> float:
    """
    Calculate the ratio of unique value reduction.

    Quick metric for how much the diversity of values was reduced.

    Parameters:
    -----------
    original : pd.Series
        Original data
    anonymized : pd.Series
        Anonymized data

    Returns:
    --------
    float
        Value reduction ratio [0.0, 1.0]
    """
    try:
        original_unique = original.nunique()
        if original_unique == 0:
            return 0.0

        anonymized_unique = anonymized.nunique()
        reduction = (original_unique - anonymized_unique) / original_unique

        return float(max(0.0, min(1.0, reduction)))

    except Exception as e:
        logger.error(f"Error calculating value reduction: {e}")
        return 0.0


def calculate_uniqueness_score(series: pd.Series) -> float:
    """
    Calculate a simple uniqueness score for a field.

    Fast indicator of how unique/identifying a field might be.

    Parameters:
    -----------
    series : pd.Series
        Data to analyze

    Returns:
    --------
    float
        Uniqueness score [0.0, 1.0] where 1.0 = all values unique
    """
    try:
        non_null_count = series.count()
        if non_null_count == 0:
            return 0.0

        unique_count = series.nunique()
        return float(unique_count / non_null_count)

    except Exception as e:
        logger.error(f"Error calculating uniqueness: {e}")
        return 0.0


def calculate_simple_disclosure_risk(
    df: pd.DataFrame, quasi_identifiers: List[str]
) -> float:
    """
    Calculate a simple disclosure risk score.

    Basic risk indicator based on the proportion of unique combinations
    in quasi-identifiers.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to analyze
    quasi_identifiers : List[str]
        List of quasi-identifier columns

    Returns:
    --------
    float
        Simple risk score [0.0, 1.0] where 1.0 = maximum risk
    """
    try:
        if not quasi_identifiers or not all(
            col in df.columns for col in quasi_identifiers
        ):
            return 0.0

        total_records = len(df)
        if total_records == 0:
            return 0.0

        # Count unique combinations
        unique_combinations = df[quasi_identifiers].drop_duplicates().shape[0]

        # Simple risk = ratio of unique combinations to total records
        risk = unique_combinations / total_records

        return float(min(1.0, risk))

    except Exception as e:
        logger.error(f"Error calculating disclosure risk: {e}")
        return 1.0


def check_anonymization_thresholds(
    metrics: Dict[str, float], thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, bool]:
    """
    Check if anonymization metrics meet specified thresholds.

    Quick validation of process metrics against target thresholds.

    Parameters:
    -----------
    metrics : Dict[str, float]
        Current process metrics
    thresholds : Optional[Dict[str, float]]
        Target thresholds (uses defaults if not provided)

    Returns:
    --------
    Dict[str, bool]
        Pass/fail status for each threshold
    """
    if thresholds is None:
        thresholds = {
            "min_k": DEFAULT_K_THRESHOLD,
            "max_suppression": DEFAULT_SUPPRESSION_WARNING,
            "min_coverage": DEFAULT_COVERAGE_TARGET,
            "max_vulnerable_ratio": 0.05,
        }

    results = {}

    # Check each threshold
    if "min_k" in metrics and "min_k" in thresholds:
        results["k_anonymity_ok"] = metrics["min_k"] >= thresholds["min_k"]

    if "suppression_rate" in metrics and "max_suppression" in thresholds:
        results["suppression_ok"] = (
            metrics["suppression_rate"] <= thresholds["max_suppression"]
        )

    if "total_coverage" in metrics and "min_coverage" in thresholds:
        results["coverage_ok"] = metrics["total_coverage"] >= thresholds["min_coverage"]

    if "vulnerable_ratio" in metrics and "max_vulnerable_ratio" in thresholds:
        results["vulnerability_ok"] = (
            metrics["vulnerable_ratio"] <= thresholds["max_vulnerable_ratio"]
        )

    # Overall status
    results["all_thresholds_met"] = all(results.values()) if results else False

    return results


def get_process_summary(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a human-readable summary of process metrics.

    Quick summary for logging and monitoring dashboards.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Process metrics

    Returns:
    --------
    Dict[str, str]
        Summary messages
    """
    summary = {}

    # Coverage summary
    if "total_coverage" in metrics:
        coverage_pct = metrics["total_coverage"] * 100
        summary["coverage"] = f"{coverage_pct:.1f}% records successfully anonymized"

    # K-anonymity summary
    if "min_k" in metrics:
        min_k = metrics["min_k"]
        if min_k >= DEFAULT_K_THRESHOLD:
            summary["k_anonymity"] = f"K-anonymity satisfied (min k={min_k})"
        else:
            summary["k_anonymity"] = f"K-anonymity violated (min k={min_k})"

    # Suppression summary
    if "suppression_rate" in metrics:
        supp_pct = metrics["suppression_rate"] * 100
        if supp_pct > DEFAULT_SUPPRESSION_WARNING * 100:
            summary["suppression"] = f"High suppression rate: {supp_pct:.1f}%"
        else:
            summary["suppression"] = f"Suppression rate: {supp_pct:.1f}%"

    # Risk summary
    if "vulnerable_ratio" in metrics:
        vuln_pct = metrics["vulnerable_ratio"] * 100
        if vuln_pct > 5:
            summary["risk"] = f"{vuln_pct:.1f}% records at risk"
        else:
            summary["risk"] = f"Low risk: {vuln_pct:.1f}% vulnerable records"

    # Overall status
    threshold_check = check_anonymization_thresholds(metrics)
    if threshold_check.get("all_thresholds_met", False):
        summary["status"] = "All anonymization targets met"
    else:
        summary["status"] = "Some anonymization targets not met"

    return summary


def calculate_batch_metrics(
    original_batch: pd.DataFrame,
    anonymized_batch: pd.DataFrame,
    original_field_name: str,
    anonymized_field_name: str,
    quasi_identifiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate all process metrics for a batch.

    Convenience function to get all key metrics in one call during
    batch processing.

    Parameters:
    -----------
    original_batch : pd.DataFrame
        Original data batch
    anonymized_batch : pd.DataFrame
        Anonymized data batch
    original_field_name : str
        Name of the original field
    anonymized_field_name : str
        Name of the anonymized field
    quasi_identifiers : Optional[List[str]]
        List of quasi-identifier columns

    Returns:
    --------
    Dict[str, Any]
        Complete set of process metrics
    """
    metrics = {}

    try:
        # Coverage metrics
        coverage = calculate_anonymization_coverage(
            original_batch[original_field_name], anonymized_batch[anonymized_field_name]
        )
        metrics.update(coverage)

        # Suppression rate
        metrics["suppression_rate"] = calculate_suppression_rate(
            anonymized_batch[anonymized_field_name],
            original_batch[original_field_name].isnull().sum(),
        )

        # Generalization metrics
        metrics["generalization_level"] = calculate_generalization_level(
            original_batch[original_field_name], anonymized_batch[anonymized_field_name]
        )

        metrics["value_reduction_ratio"] = calculate_value_reduction_ratio(
            original_batch[original_field_name], anonymized_batch[anonymized_field_name]
        )

        # K-anonymity metrics if quasi-identifiers provided
        if quasi_identifiers:
            metrics["min_k"] = calculate_min_group_size(
                anonymized_batch, quasi_identifiers
            )

            metrics["vulnerable_ratio"] = calculate_vulnerable_records_ratio(
                anonymized_batch, quasi_identifiers
            )

            # Group distribution (limited info for performance)
            group_dist = get_group_size_distribution(
                anonymized_batch,
                quasi_identifiers,
                max_groups=50,  # Limit for performance
            )
            metrics["group_count"] = group_dist.get("total_groups", 0)
            metrics["mean_group_size"] = group_dist.get("mean_size", 0.0)

        # Basic risk indicators
        metrics["uniqueness_score"] = calculate_uniqueness_score(
            anonymized_batch[anonymized_field_name]
        )

        if quasi_identifiers:
            metrics["disclosure_risk"] = calculate_simple_disclosure_risk(
                anonymized_batch, quasi_identifiers
            )

        # Threshold check
        metrics["thresholds"] = check_anonymization_thresholds(metrics)

    except Exception as e:
        logger.error(f"Error calculating batch metrics: {e}")
        metrics["error"] = str(e)

    return metrics


# Module metadata
__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    # Coverage metrics
    "calculate_anonymization_coverage",
    "calculate_suppression_rate",
    # Group metrics
    "get_group_size_distribution",
    "calculate_min_group_size",
    "calculate_vulnerable_records_ratio",
    # Generalization metrics
    "calculate_generalization_level",
    "calculate_value_reduction_ratio",
    # Risk indicators
    "calculate_uniqueness_score",
    "calculate_simple_disclosure_risk",
    # Process control
    "check_anonymization_thresholds",
    "get_process_summary",
    "calculate_batch_metrics",
]
