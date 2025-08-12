"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Anonymization Metric Utilities
Package:       pamola_core.anonymization.commons
Version:       2.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2024
Revised:       2025
License:       BSD 3-Clause

Description:
 This module provides lightweight metric utilities specifically for anonymization
 operations within the PAMOLA.CORE framework. It focuses on process-oriented
 metrics for monitoring and guiding anonymization operations in real-time,
 particularly for generalization, masking, and suppression techniques.

Purpose:
 Serves as a metrics collection and calculation engine for anonymization
 operations, providing quick feedback on the anonymization process without
 heavy computational overhead. NOT intended for final quality assessment
 or detailed utility analysis.

Key Features:
 - Basic anonymization effectiveness metrics
 - Generalization-specific measurements
 - Categorical information loss metrics
 - Simple performance tracking
 - Lightweight visualization data preparation
 - Process-oriented metric persistence

Design Principles:
 - Speed: Fast calculations suitable for batch processing
 - Simplicity: Basic metrics only, no complex statistics
 - Focus: Process monitoring, not final assessment
 - Integration: Works with DataWriter and ProgressTracker

Usage:
 Used by anonymization operations to track progress, measure basic
 effectiveness, and provide feedback during the anonymization process.

Dependencies:
 - numpy: For basic numerical operations
 - pandas: For data manipulation
 - pamola_core.utils: For I/O and visualization support

Changelog:
 2.1.0 - Added categorical information loss and generalization height metrics
 2.0.0 - Streamlined for process metrics only
 1.0.0 - Initial implementation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

import numpy as np
import pandas as pd

# Import core utilities
from pamola_core.utils.io import write_json, ensure_directory
from pamola_core.utils.ops.op_data_writer import DataWriter

logger = logging.getLogger(__name__)


def calculate_anonymization_effectiveness(
    original_series: pd.Series, anonymized_series: pd.Series
) -> Dict[str, float]:
    """
    Calculate basic effectiveness metrics for anonymization.

    Quick metrics to assess how well the anonymization process worked
    without heavy statistical analysis.

    Parameters:
    -----------
    original_series : pd.Series
        Original data before anonymization
    anonymized_series : pd.Series
        Data after anonymization

    Returns:
    --------
    Dict[str, float]
        Basic effectiveness metrics
    """
    try:
        total_records = len(original_series)
        if total_records == 0:
            return {
                "total_records": 0,
                "effectiveness_ratio": 0.0,
                "null_increase": 0.0,
            }

        # Basic counts
        orig_nulls = original_series.isna().sum()
        anon_nulls = anonymized_series.isna().sum()
        orig_unique = original_series.nunique()
        anon_unique = anonymized_series.nunique()

        # Calculate effectiveness (reduction in uniqueness)
        effectiveness = 0.0
        if orig_unique > 1:
            effectiveness = (orig_unique - anon_unique) / (orig_unique - 1)
            effectiveness = max(0.0, min(1.0, effectiveness))

        return {
            "total_records": int(total_records),
            "original_unique": int(orig_unique),
            "anonymized_unique": int(anon_unique),
            "effectiveness_ratio": float(effectiveness),
            "null_increase": (
                float((anon_nulls - orig_nulls) / total_records)
                if total_records > 0
                else 0.0
            ),
        }

    except Exception as e:
        logger.error(f"Error calculating effectiveness: {e}")
        return {
            "total_records": len(original_series),
            "effectiveness_ratio": 0.0,
            "error": str(e),
        }


def calculate_generalization_metrics(
    original_series: pd.Series,
    anonymized_series: pd.Series,
    strategy: str,
    strategy_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate metrics specific to generalization strategies.

    Lightweight metrics focused on the generalization process itself.

    Parameters:
    -----------
    original_series : pd.Series
        Original data
    anonymized_series : pd.Series
        Generalized data
    strategy : str
        Generalization strategy used
    strategy_params : Dict[str, Any]
        Parameters of the strategy

    Returns:
    --------
    Dict[str, Any]
        Generalization process metrics
    """
    try:
        # Initialize metrics dictionary with proper typing
        metrics: Dict[str, Any] = {"strategy": strategy, "parameters": strategy_params}

        # Calculate basic effectiveness
        effectiveness = calculate_anonymization_effectiveness(
            original_series, anonymized_series
        )
        metrics["reduction_ratio"] = effectiveness["effectiveness_ratio"]

        # Strategy-specific lightweight metrics
        if strategy == "binning":
            bin_count = strategy_params.get("bin_count", 0)
            if bin_count > 0:
                non_null_count = anonymized_series.count()
                metrics["bin_count"] = int(bin_count)
                metrics["avg_bin_size"] = (
                    float(non_null_count / bin_count) if bin_count > 0 else 0.0
                )

                # Simple bin utilization
                actual_bins = anonymized_series.nunique()
                metrics["bin_utilization"] = (
                    float(actual_bins / bin_count) if bin_count > 0 else 0.0
                )

        elif strategy == "rounding":
            precision = strategy_params.get("precision", 0)
            metrics["precision_level"] = int(precision)

            # Simple precision impact
            if pd.api.types.is_numeric_dtype(original_series):
                orig_std = original_series.std()
                if orig_std > 0 and precision >= 0:
                    # Rough estimate of precision impact
                    precision_impact = min(1.0, (10**-precision) / orig_std)
                    metrics["precision_impact"] = float(precision_impact)

        elif strategy == "range":
            # Simple range-based metric
            range_limits = strategy_params.get("range_limits", [])

            if (
                range_limits
                and isinstance(range_limits[0], list)
                and len(range_limits[0]) == 2
            ):
                try:
                    # Compute the span for each [min, max] pair
                    spans = [float(upper - lower) for lower, upper in range_limits]
                    # Save individual spans
                    metrics["range_span_list"] = spans
                    # Compute average span
                    metrics["range_span_avg"] = sum(spans) / len(spans)
                    # Compute total span across all ranges
                    flat_values = [val for sublist in range_limits for val in sublist]
                    metrics["range_span_total"] = float(
                        max(flat_values) - min(flat_values)
                    )
                except Exception as e:
                    logger.error(f"Error calculating range spans: {e}")
                    return {
                        "strategy": strategy,
                        "error": str(e),
                    }
            else:
                logger.warning(f"Invalid or missing range_limits: {range_limits}")
                return {
                    "strategy": strategy,
                    "error": "Invalid range_limits format",
                }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating generalization metrics: {e}")
        return {"strategy": strategy, "error": str(e)}


def calculate_categorical_information_loss(
    original_series: pd.Series,
    anonymized_series: pd.Series,
    category_mapping: Optional[Dict[str, str]] = None,
    hierarchy_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Calculate information loss metrics for categorical generalization.

    This function measures how much information is lost when categorical
    values are generalized, providing quick feedback on the trade-off
    between privacy and utility.

    Parameters:
    -----------
    original_series : pd.Series
        Original categorical data
    anonymized_series : pd.Series
        Generalized categorical data
    category_mapping : Optional[Dict[str, str]]
        Mapping from original to generalized categories
    hierarchy_info : Optional[Dict[str, Any]]
        Information about hierarchy (levels, structure) if available

    Returns:
    --------
    Dict[str, float]
        Information loss metrics:
        - precision_loss: Loss of granularity (0-1)
        - entropy_loss: Normalized entropy reduction (0-1)
        - distribution_shift: Change in value distribution (0-1)
        - category_reduction_ratio: Ratio of unique values reduced
        - average_group_size: Average size of generalized groups
    """
    try:
        # Basic validation
        if len(original_series) == 0:
            return {
                "precision_loss": 0.0,
                "entropy_loss": 0.0,
                "distribution_shift": 0.0,
                "category_reduction_ratio": 0.0,
                "average_group_size": 0.0,
            }

        # Calculate unique value reduction
        orig_unique = original_series.nunique()
        anon_unique = anonymized_series.nunique()

        # Category reduction ratio
        category_reduction_ratio = 0.0
        if orig_unique > 0:
            category_reduction_ratio = (orig_unique - anon_unique) / orig_unique

        # Calculate precision loss (simple version)
        precision_loss = category_reduction_ratio  # Simplified measure

        # Calculate entropy loss
        def calculate_entropy(series: pd.Series) -> float:
            """Calculate Shannon entropy of a series."""
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            # Avoid log(0) by filtering out zero probabilities
            probabilities = probabilities[probabilities > 0]
            if len(probabilities) <= 1:
                return 0.0
            return -float((probabilities * np.log2(probabilities)).sum())

        orig_entropy = calculate_entropy(original_series)
        anon_entropy = calculate_entropy(anonymized_series)

        # Normalized entropy loss
        entropy_loss = 0.0
        if orig_entropy > 0:
            entropy_loss = (orig_entropy - anon_entropy) / orig_entropy
            entropy_loss = max(0.0, min(1.0, entropy_loss))

        # Calculate distribution shift (simplified)
        # Compare the distribution of frequencies
        orig_freq = original_series.value_counts(normalize=True).sort_index()
        anon_freq = anonymized_series.value_counts(normalize=True).sort_index()

        # For each generalized value, calculate how much the distribution changed
        distribution_shift = 0.0
        if len(anon_freq) > 0:
            # Simple approach: average change in frequency for generalized values
            freq_changes = []
            for gen_value in anon_freq.index:
                # Find all original values that map to this generalized value
                if category_mapping:
                    orig_values = [
                        k for k, v in category_mapping.items() if v == gen_value
                    ]
                    if orig_values:
                        orig_total_freq = sum(orig_freq.get(v, 0) for v in orig_values)
                        freq_change = abs(anon_freq[gen_value] - orig_total_freq)
                        freq_changes.append(freq_change)
                else:
                    # Without mapping, use simple frequency difference
                    freq_changes.append(anon_freq[gen_value])

            if freq_changes:
                # numpy → native float
                distribution_shift = float(np.mean(freq_changes))
                distribution_shift = min(1.0, distribution_shift)

        # Calculate average group size
        average_group_size = 0.0
        if anon_unique > 0:
            average_group_size = len(anonymized_series) / anon_unique

        # If hierarchy info is provided, adjust precision loss
        if hierarchy_info:
            # Adjust precision loss based on hierarchy depth
            max_depth = hierarchy_info.get("max_depth", 1)
            avg_depth_change = hierarchy_info.get("avg_depth_change", 1)
            if max_depth > 0:
                precision_loss = min(1.0, avg_depth_change / max_depth)

        return {
            "precision_loss": float(precision_loss),
            "entropy_loss": float(entropy_loss),
            "distribution_shift": float(distribution_shift),
            "category_reduction_ratio": float(category_reduction_ratio),
            "average_group_size": float(average_group_size),
        }

    except Exception as e:
        logger.error(f"Error calculating categorical information loss: {e}")
        return {
            "precision_loss": 0.0,
            "entropy_loss": 0.0,
            "distribution_shift": 0.0,
            "category_reduction_ratio": 0.0,
            "average_group_size": 0.0,
            "error": str(e),
        }


def calculate_generalization_height(
    original_values: Union[pd.Series, List[str]],
    generalized_values: Union[pd.Series, List[str]],
    hierarchy_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Calculate the generalization height in a hierarchy.

    This function measures how many levels up in the hierarchy the values
    have been generalized. It's a quick metric for understanding the degree
    of generalization applied.

    Parameters:
    -----------
    original_values : Union[pd.Series, List[str]]
        Original values before generalization
    generalized_values : Union[pd.Series, List[str]]
        Values after generalization
    hierarchy_dict : Optional[Dict[str, Dict[str, Any]]]
        Hierarchy information with structure:
        {
            "value": {
                "parent": "parent_value",
                "level": 0,  # Leaf level
                "root": "root_value"
            }
        }

    Returns:
    --------
    Dict[str, Any]
        Generalization height metrics:
        - min_height: Minimum generalization height
        - max_height: Maximum generalization height
        - avg_height: Average generalization height
        - height_distribution: Distribution of heights
        - uniform_generalization: Whether all values generalized equally
    """
    try:
        # Convert to lists if Series
        if isinstance(original_values, pd.Series):
            original_values = original_values.tolist()
        if isinstance(generalized_values, pd.Series):
            generalized_values = generalized_values.tolist()

        # Validate same length
        if len(original_values) != len(generalized_values):
            raise ValueError("Original and generalized values must have same length")

        heights = []

        # Calculate heights
        for orig, gen in zip(original_values, generalized_values):
            if pd.isna(orig) or pd.isna(gen):
                continue

            # If values are the same, height is 0
            if str(orig) == str(gen):
                heights.append(0)
                continue

            # If we have hierarchy information
            if hierarchy_dict and str(orig) in hierarchy_dict:
                # Calculate actual height using hierarchy
                height = 0
                current = str(orig)

                # Traverse up the hierarchy until we reach the generalized value
                while current != str(gen) and current in hierarchy_dict:
                    parent = hierarchy_dict[current].get("parent")
                    if not parent:
                        # Reached root without finding generalized value
                        # This means generalization is outside hierarchy
                        height = -1
                        break
                    height += 1
                    current = parent

                    # Safety check to prevent infinite loops
                    if height > 100:
                        logger.warning(
                            f"Potential infinite loop in hierarchy for {orig}"
                        )
                        height = -1
                        break

                if height >= 0:
                    heights.append(height)
                else:
                    # If we couldn't trace the path, estimate height
                    heights.append(1)  # Default to 1 level
            else:
                # Without hierarchy, we can only detect if generalization occurred
                # Assume height of 1 for any generalization
                heights.append(1)

        # Calculate statistics
        if not heights:
            return {
                "min_height": 0,
                "max_height": 0,
                "avg_height": 0.0,
                "height_distribution": {},
                "uniform_generalization": True,
            }

        # Height distribution
        height_counts = {}
        for h in heights:
            height_counts[h] = height_counts.get(h, 0) + 1

        # Check if generalization is uniform
        unique_heights = len(set(heights))
        uniform_generalization = unique_heights == 1

        return {
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "avg_height": float(np.mean(heights)),
            "height_distribution": height_counts,
            "uniform_generalization": bool(uniform_generalization),
            "total_generalized": len([h for h in heights if h > 0]),
            "total_unchanged": len([h for h in heights if h == 0]),
        }

    except Exception as e:
        logger.error(f"Error calculating generalization height: {e}")
        return {
            "min_height": 0,
            "max_height": 0,
            "avg_height": 0.0,
            "height_distribution": {},
            "uniform_generalization": False,
            "error": str(e),
        }


def calculate_masking_metrics(
    original_series: pd.Series, masked_series: pd.Series, mask_char: str = "*"
) -> Dict[str, float]:
    """
    Calculate metrics for masking operations.

    Simple metrics to track masking effectiveness.

    Parameters:
    -----------
    original_series : pd.Series
        Original data
    masked_series : pd.Series
        Masked data
    mask_char : str
        Character used for masking

    Returns:
    --------
    Dict[str, float]
        Masking process metrics
    """
    try:
        total_records = len(original_series)
        if total_records == 0:
            return {"masking_ratio": 0.0, "total_records": 0}

        # Count masked values
        masked_count = 0
        for orig, masked in zip(original_series, masked_series):
            if pd.notna(masked) and pd.notna(orig):
                # Check if value contains mask character
                if isinstance(masked, str) and mask_char in str(masked):
                    masked_count += 1

        return {
            "total_records": int(total_records),
            "masked_records": int(masked_count),
            "masking_ratio": float(masked_count / total_records),
            "null_count": int(masked_series.isna().sum()),
        }

    except Exception as e:
        logger.error(f"Error calculating masking metrics: {e}")
        return {"masking_ratio": 0.0, "error": str(e)}


def calculate_suppression_metrics(
    original_series: pd.Series, suppressed_series: pd.Series
) -> Dict[str, float]:
    """
    Calculate metrics for suppression operations.

    Track how many values were suppressed during anonymization.

    Parameters:
    -----------
    original_series : pd.Series
        Original data
    suppressed_series : pd.Series
        Data after suppression

    Returns:
    --------
    Dict[str, float]
        Suppression metrics
    """
    try:
        total_records = len(original_series)
        if total_records == 0:
            return {"suppression_ratio": 0.0, "total_records": 0}

        # Calculate suppressions
        orig_nulls = original_series.isna().sum()
        supp_nulls = suppressed_series.isna().sum()
        new_suppressions = max(0, supp_nulls - orig_nulls)

        # Also count values that were replaced with special markers
        special_markers = ["*", "SUPPRESSED", "REDACTED", "N/A"]
        marker_count = 0

        for val in suppressed_series:
            if pd.notna(val) and str(val).upper() in special_markers:
                marker_count += 1

        total_suppressions = new_suppressions + marker_count
        non_null_original = total_records - orig_nulls

        return {
            "total_records": int(total_records),
            "suppressed_count": int(total_suppressions),
            "suppression_ratio": (
                float(total_suppressions / non_null_original)
                if non_null_original > 0
                else 0.0
            ),
            "null_suppressions": int(new_suppressions),
            "marker_suppressions": int(marker_count),
        }

    except Exception as e:
        logger.error(f"Error calculating suppression metrics: {e}")
        return {"suppression_ratio": 0.0, "error": str(e)}


def calculate_process_performance(
    start_time: float,
    end_time: float,
    records_processed: int,
    batch_count: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate performance metrics for the anonymization process.

    Simple timing and throughput metrics.

    Parameters:
    -----------
    start_time : float
        Process start time
    end_time : float
        Process end time
    records_processed : int
        Number of records processed
    batch_count : Optional[int]
        Number of batches processed

    Returns:
    --------
    Dict[str, float]
        Performance metrics
    """
    try:
        duration = max(0.0, end_time - start_time)

        metrics = {
            "duration_seconds": float(duration),
            "records_processed": int(records_processed),
            "records_per_second": (
                float(records_processed / duration) if duration > 0 else 0.0
            ),
        }

        if batch_count:
            metrics["batch_count"] = int(batch_count)
            metrics["avg_batch_size"] = (
                float(records_processed / batch_count) if batch_count > 0 else 0.0
            )
            metrics["batches_per_second"] = (
                float(batch_count / duration) if duration > 0 else 0.0
            )

        return metrics

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {
            "duration_seconds": 0.0,
            "records_processed": records_processed,
            "error": str(e),
        }


def get_value_distribution_summary(
    series: pd.Series, max_categories: int = 10
) -> Dict[str, Any]:
    """
    Get a simple summary of value distribution.

    Lightweight distribution info for process monitoring.

    Parameters:
    -----------
    series : pd.Series
        Data to summarize
    max_categories : int
        Maximum categories to include

    Returns:
    --------
    Dict[str, Any]
        Distribution summary
    """
    try:
        total_count = len(series)
        null_count = series.isna().sum()
        unique_count = series.nunique()

        summary: Dict[str, Any] = {
            "total_count": int(total_count),
            "null_count": int(null_count),
            "unique_count": int(unique_count),
            "uniqueness_ratio": (
                float(unique_count / (total_count - null_count))
                if (total_count - null_count) > 0
                else 0.0
            ),
        }

        # Add top values for categorical data
        if unique_count <= max_categories * 2:
            value_counts = series.value_counts().head(max_categories)
            summary["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

        return summary

    except Exception as e:
        logger.error(f"Error getting distribution summary: {e}")
        return {"total_count": len(series), "error": str(e)}


def collect_operation_metrics(
    operation_type: str,
    original_data: pd.Series,
    processed_data: pd.Series,
    operation_params: Dict[str, Any],
    timing_info: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Collect all relevant metrics for an anonymization operation.

    Central function to gather process metrics based on operation type.

    Parameters:
    -----------
    operation_type : str
        Type of operation (generalization, masking, suppression, etc.)
    original_data : pd.Series
        Original data
    processed_data : pd.Series
        Processed data
    operation_params : Dict[str, Any]
        Operation parameters
    timing_info : Optional[Dict[str, float]]
        Timing information (start_time, end_time)

    Returns:
    --------
    Dict[str, Any]
        Collected metrics
    """
    # Initialize metrics with proper typing
    metrics: Dict[str, Any] = {
        "operation_type": operation_type,
        "timestamp": datetime.now().isoformat(),
        "field_info": {
            "original": get_value_distribution_summary(original_data),
            "processed": get_value_distribution_summary(processed_data),
        },
    }

    # Add operation-specific metrics
    if operation_type == "generalization":
        strategy = operation_params.get("strategy", "unknown")
        metrics["generalization"] = calculate_generalization_metrics(
            original_data, processed_data, strategy, operation_params
        )

        # Add categorical-specific metrics if applicable
        if strategy in ["merge_low_freq", "hierarchy", "frequency_based"]:
            metrics["categorical_info_loss"] = calculate_categorical_information_loss(
                original_data,
                processed_data,
                operation_params.get("category_mapping"),
                operation_params.get("hierarchy_info"),
            )

            # Add height metrics if hierarchy is used
            if strategy == "hierarchy" and operation_params.get("hierarchy_dict"):
                metrics["generalization_height"] = calculate_generalization_height(
                    original_data,
                    processed_data,
                    operation_params.get("hierarchy_dict"),
                )

    elif operation_type == "masking":
        mask_char = operation_params.get("mask_char", "*")
        metrics["masking"] = calculate_masking_metrics(
            original_data, processed_data, mask_char
        )

    elif operation_type == "suppression":
        metrics["suppression"] = calculate_suppression_metrics(
            original_data, processed_data
        )

    else:
        # Default effectiveness metrics
        metrics["effectiveness"] = calculate_anonymization_effectiveness(
            original_data, processed_data
        )

    # Add timing if available
    if timing_info and "start_time" in timing_info and "end_time" in timing_info:
        metrics["performance"] = calculate_process_performance(
            timing_info["start_time"],
            timing_info["end_time"],
            len(original_data),
            timing_info.get("batch_count"),
        )

        # Extract performance metrics
        performance_metrics = metrics["performance"]
        total_duration = performance_metrics.get("duration_seconds", 0)
        total_records = performance_metrics.get("records_processed", 0)
        # Calculate average records per second
        metrics["performance_summary"] = {
            "total_duration_seconds": total_duration,
            "total_records_processed": total_records,
            "avg_records_per_second": (
                total_records / total_duration if total_duration > 0 else 0
            ),
        }

    return metrics


def save_process_metrics(
    metrics: Dict[str, Any],
    task_dir: Path,
    operation_name: str,
    field_name: str,
    writer: Optional[DataWriter] = None,
) -> Optional[Path]:
    """
    Save process metrics to file.

    Lightweight metric persistence for process monitoring.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics to save
    task_dir : Path
        Task directory
    operation_name : str
        Operation name
    field_name : str
        Field name
    writer : Optional[DataWriter]
        DataWriter instance

    Returns:
    --------
    Optional[Path]
        Path to saved metrics or None
    """
    try:
        # Add metadata
        metrics["metadata"] = {
            "operation": operation_name,
            "field": field_name,
            "saved_at": datetime.now().isoformat(),
        }

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{field_name}_{operation_name}_process_metrics_{timestamp}.json"

        # Try using DataWriter first
        if writer:
            try:
                result = writer.write_metrics(
                    metrics=metrics,
                    name=f"{field_name}_{operation_name}_process",
                    timestamp_in_name=True,
                )
                logger.info(f"Process metrics saved via DataWriter: {result.path}")
                return Path(result.path)
            except Exception as e:
                logger.warning(f"DataWriter failed, using direct write: {e}")

        # Fallback to direct write
        ensure_directory(task_dir)
        file_path = task_dir / filename

        write_json(metrics, file_path, indent=2, ensure_ascii=False, convert_numpy=True)

        logger.info(f"Process metrics saved to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save process metrics: {e}")
        return None


def get_process_summary_message(metrics: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of process metrics.

    Quick summary for logging and monitoring.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Process metrics

    Returns:
    --------
    str
        Summary message
    """
    try:
        operation = metrics.get("operation_type", "anonymization")

        # Build summary parts
        parts = [f"{operation.capitalize()} process completed"]

        # Add effectiveness info
        if "effectiveness" in metrics:
            eff = metrics["effectiveness"]
            ratio = eff.get("effectiveness_ratio", 0) * 100
            parts.append(f"effectiveness: {ratio:.1f}%")

        # Add operation-specific info
        if "generalization" in metrics:
            gen = metrics["generalization"]
            if "bin_count" in gen:
                parts.append(f"bins: {gen['bin_count']}")
            if "reduction_ratio" in gen:
                ratio = gen["reduction_ratio"] * 100
                parts.append(f"reduction: {ratio:.1f}%")

        elif "masking" in metrics:
            mask = metrics["masking"]
            ratio = mask.get("masking_ratio", 0) * 100
            parts.append(f"masked: {ratio:.1f}%")

        elif "suppression" in metrics:
            supp = metrics["suppression"]
            ratio = supp.get("suppression_ratio", 0) * 100
            parts.append(f"suppressed: {ratio:.1f}%")

        # Add categorical info loss if present
        if "categorical_info_loss" in metrics:
            cat_loss = metrics["categorical_info_loss"]
            precision_loss = cat_loss.get("precision_loss", 0) * 100
            parts.append(f"precision loss: {precision_loss:.1f}%")

        # Add generalization height if present
        if "generalization_height" in metrics:
            height = metrics["generalization_height"]
            avg_height = height.get("avg_height", 0)
            parts.append(f"avg height: {avg_height:.1f}")

        # Add performance info
        if "performance" in metrics:
            perf = metrics["performance"]
            rps = perf.get("records_per_second", 0)
            parts.append(f"speed: {rps:.0f} rec/s")

        return " | ".join(parts)

    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return "Process completed"


# Module metadata
__version__ = "2.1.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    # Basic metrics
    "calculate_anonymization_effectiveness",
    "calculate_generalization_metrics",
    "calculate_masking_metrics",
    "calculate_suppression_metrics",
    # Categorical metrics
    "calculate_categorical_information_loss",
    "calculate_generalization_height",
    # Performance metrics
    "calculate_process_performance",
    # Distribution summaries
    "get_value_distribution_summary",
    # Collection and persistence
    "collect_operation_metrics",
    "save_process_metrics",
    "get_process_summary_message",
]
