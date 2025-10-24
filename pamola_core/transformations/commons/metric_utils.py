"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Transformation Metric Utilities
Description: Common metric utilities for transformation operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functions for collecting and calculating metrics
related to transformation operations. Supports dataset comparison,
field-level statistics, and transformation impact analysis.
"""

from datetime import datetime
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats

from pamola_core.utils.io import ensure_directory, write_json
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.progress import HierarchicalProgressTracker

# Set up logger
logger = logging.getLogger(__name__)


def calculate_dataset_comparison(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate metrics comparing two datasets.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame before transformation.
    transformed_df : pd.DataFrame
        The transformed DataFrame after processing.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing comparison metrics.
    """
    if original_df is None or transformed_df is None:
        raise ValueError("Both DataFrames must be provided")

    start_time = time.time()
    logger.info("Calculating dataset comparison metrics")

    try:
        result = {}

        # Row count comparison
        result["row_counts"] = _compare_row_counts(original_df, transformed_df)

        # Column count comparison
        (
            result["column_counts"],
            result["common_columns"],
            result["added_columns"],
            result["removed_columns"],
        ) = _compare_column_counts(original_df, transformed_df)

        # Value and null changes in common columns
        value_changes, null_changes = _compare_values_and_nulls(
            original_df, transformed_df
        )
        result["value_changes"] = value_changes
        result["null_changes"] = null_changes

        # Memory usage comparison
        result["memory_usage"] = _compare_memory_usage(original_df, transformed_df)

        elapsed_time = time.time() - start_time
        logger.info(f"Dataset comparison completed in {elapsed_time:.4f} seconds")

        return result

    except Exception as e:
        logger.error(f"Error during dataset comparison: {str(e)}")
        raise


def _compare_row_counts(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare the row counts between the original and transformed DataFrames.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame.
    transformed_df : pd.DataFrame
        The transformed DataFrame.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing row count differences and percentage change.
    """
    original_rows = len(original_df)
    transformed_rows = len(transformed_df)
    row_diff = transformed_rows - original_rows
    row_pct_change = (
        (row_diff / original_rows * 100) if original_rows > 0 else float("inf")
    )

    return {
        "original": original_rows,
        "transformed": transformed_rows,
        "difference": row_diff,
        "percent_change": row_pct_change,
    }


def _compare_column_counts(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    """
    Compare the column counts between the original and transformed DataFrames.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame.
    transformed_df : pd.DataFrame
        The transformed DataFrame.

    Returns:
    --------
    Tuple containing:
        - Dict with column count differences and percentage change.
        - List of common columns.
        - List of added columns.
        - List of removed columns.
    """
    original_cols = set(original_df.columns)
    transformed_cols = set(transformed_df.columns)
    common_cols = original_cols.intersection(transformed_cols)
    added_cols = transformed_cols - original_cols
    removed_cols = original_cols - transformed_cols

    column_count = {
        "original": len(original_cols),
        "transformed": len(transformed_cols),
        "difference": len(transformed_cols) - len(original_cols),
        "percent_change": (
            ((len(transformed_cols) - len(original_cols)) / len(original_cols) * 100)
            if len(original_cols) > 0
            else float("inf")
        ),
    }

    return column_count, list(common_cols), list(added_cols), list(removed_cols)


def _compare_values_and_nulls(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compare values and nulls in common columns between the original and transformed DataFrames.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame.
    transformed_df : pd.DataFrame
        The transformed DataFrame.

    Returns:
    --------
    Tuple containing:
        - Dictionary of value changes for common columns.
        - Dictionary of null changes for common columns.
    """
    value_changes = {}
    null_changes = {}

    common_cols = original_df.columns.intersection(transformed_df.columns)

    # Sample data to speed up comparison for large datasets
    sample_size = min(10000, len(original_df), len(transformed_df))
    original_sample = (
        original_df.sample(sample_size)
        if len(original_df) > sample_size
        else original_df
    )

    if set(original_sample.index).issubset(set(transformed_df.index)):
        transformed_sample = transformed_df.loc[original_sample.index]

        for col in common_cols:
            # Compare values in common columns
            if not pd.api.types.is_dtype_equal(
                original_sample[col].dtype, transformed_sample[col].dtype
            ):
                value_changes[col] = {
                    "changed": "N/A - incompatible dtypes",
                    "original_dtype": str(original_sample[col].dtype),
                    "transformed_dtype": str(transformed_sample[col].dtype),
                }
                continue

            changes = _count_value_changes(
                original_sample[col], transformed_sample[col]
            )
            value_changes[col] = changes

            # Count null changes
            null_changes[col] = _count_null_changes(
                original_sample[col], transformed_sample[col]
            )

    return value_changes, null_changes


def _count_value_changes(
    original_col: pd.Series, transformed_col: pd.Series
) -> Dict[str, Any]:
    """
    Count the value changes between the original and transformed columns.

    Parameters:
    -----------
    original_col : pd.Series
        The original column values.
    transformed_col : pd.Series
        The transformed column values.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the number of changes and the percentage change.
    """
    if pd.api.types.is_numeric_dtype(original_col):
        changes = (
            ~np.isclose(
                original_col.fillna(np.nan),
                transformed_col.fillna(np.nan),
                equal_nan=True,
            )
        ).sum()
    else:
        changes = (
            ~np.where(
                original_col.isna() & transformed_col.isna(),
                True,
                original_col == transformed_col
            )
        ).sum()

    return {
        "changed": int(changes),
        "percent_changed": (
            (changes / len(original_col) * 100) if len(original_col) > 0 else 0
        ),
    }


def _count_null_changes(
    original_col: pd.Series, transformed_col: pd.Series
) -> Dict[str, Any]:
    """
    Count the null value changes between the original and transformed columns.

    Parameters:
    -----------
    original_col : pd.Series
        The original column values.
    transformed_col : pd.Series
        The transformed column values.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the number of null changes and the percentage change.
    """
    original_nulls = original_col.isna().sum()
    transformed_nulls = transformed_col.isna().sum()

    return {
        "original_nulls": int(original_nulls),
        "transformed_nulls": int(transformed_nulls),
        "difference": int(transformed_nulls - original_nulls),
        "percent_change": (
            ((transformed_nulls - original_nulls) / len(original_col) * 100)
            if len(original_col) > 0
            else 0
        ),
    }


def _compare_memory_usage(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare the memory usage between the original and transformed DataFrames.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame.
    transformed_df : pd.DataFrame
        The transformed DataFrame.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing memory usage comparison.
    """
    original_memory = original_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    transformed_memory = transformed_df.memory_usage(deep=True).sum() / (
        1024 * 1024
    )  # MB

    return {
        "original_mb": round(original_memory, 2),
        "transformed_mb": round(transformed_memory, 2),
        "difference_mb": round(transformed_memory - original_memory, 2),
        "percent_change": (
            round((transformed_memory - original_memory) / original_memory * 100, 2)
            if original_memory > 0
            else float("inf")
        ),
    }


def calculate_field_statistics(
    df: pd.DataFrame, fields: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistical metrics for specified fields.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.
    fields : Optional[List[str]]
        List of field names to analyze. If None, all columns are analyzed.

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary mapping field names to statistics dictionaries containing:
        - basic_stats: Basic statistics (count, mean, std, min, 25%, 50%, 75%, max)
        - null_stats: Null value statistics (count, percentage)
        - unique_stats: Unique value statistics (count, percentage)
        - distribution: Distribution metrics (skewness, kurtosis)
        - top_values: Most frequent values and their counts (for non-numeric fields)
        - data_type: Field data type
    """
    if df is None:
        raise ValueError("DataFrame must be provided")

    if len(df) == 0:
        logger.warning("Empty DataFrame provided")
        return {}

    start_time = time.time()

    # Use all fields if none specified
    fields = _validate_fields(df, fields)

    logger.info(
        f"Calculating statistics for {len(fields)} fields in DataFrame with {len(df)} rows"
    )

    try:
        result = {}

        for field in fields:
            field_stats = {}
            series = df[field]

            # Store data type
            field_stats["data_type"] = str(series.dtype)

            # Calculate basic statistics
            field_stats["basic_stats"] = _calculate_basic_stats(series)

            # Calculate null statistics
            field_stats["null_stats"] = _calculate_null_stats(series, len(df))

            # Calculate unique value statistics
            field_stats["unique_stats"] = _calculate_unique_stats(series, len(df))

            # Get top values for non-numeric fields
            if not pd.api.types.is_numeric_dtype(series):
                field_stats["top_values"] = _calculate_top_values(series)

            # Add pattern detection for string fields
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
                series
            ):
                field_stats["pattern_detection"] = _detect_patterns(series)

            result[field] = field_stats

        elapsed_time = time.time() - start_time
        logger.info(
            f"Field statistics calculation completed in {elapsed_time:.4f} seconds"
        )

        return result

    except Exception as e:
        logger.error(f"Error calculating field statistics: {str(e)}")
        raise


def _validate_fields(df: pd.DataFrame, fields: Optional[List[str]]) -> List[str]:
    """
    Validate the field names provided for the analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.
    fields : Optional[List[str]]
        List of field names to analyze. If None, all columns are considered.

    Returns:
    --------
    List[str]
        List of valid field names from the DataFrame columns.
    """
    if fields is None:
        return list(df.columns)

    # Validate field names
    invalid_fields = [field for field in fields if field not in df.columns]
    if invalid_fields:
        logger.warning(f"Invalid fields specified: {invalid_fields}")
    return [field for field in fields if field in df.columns]


def _calculate_basic_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic statistics for a field (count, mean, std, min, etc.).

    Parameters:
    -----------
    series : pd.Series
        The data series to calculate statistics for.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing basic statistics for the field.
    """
    try:
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            return {
                "count": int(desc["count"]),
                "mean": None if np.isnan(desc["mean"]) else float(desc["mean"]),
                "std": None if np.isnan(desc["std"]) else float(desc["std"]),
                "min": None if np.isnan(desc["min"]) else float(desc["min"]),
                "25%": None if np.isnan(desc["25%"]) else float(desc["25%"]),
                "50%": None if np.isnan(desc["50%"]) else float(desc["50%"]),
                "75%": None if np.isnan(desc["75%"]) else float(desc["75%"]),
                "max": None if np.isnan(desc["max"]) else float(desc["max"]),
            }
        else:
            return {
                "count": int(len(series)),
                "length_mean": float(series.astype(str).str.len().mean()),
                "length_min": int(series.astype(str).str.len().min()),
                "length_max": int(series.astype(str).str.len().max()),
            }
    except Exception as e:
        logger.warning(f"Error calculating basic statistics: {str(e)}")
        return {"error": str(e)}


def _calculate_null_stats(series: pd.Series, df_length: int) -> Dict[str, Any]:
    """
    Calculate the statistics for null values in a field.

    Parameters:
    -----------
    series : pd.Series
        The data series to calculate null statistics for.
    df_length : int
        The total number of rows in the DataFrame.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing null value statistics.
    """
    null_count = series.isna().sum()
    return {
        "count": int(null_count),
        "percentage": float(null_count / df_length * 100) if df_length > 0 else 0,
    }


def _calculate_unique_stats(series: pd.Series, df_length: int) -> Dict[str, Any]:
    """
    Calculate statistics for unique values in a field.

    Parameters:
    -----------
    series : pd.Series
        The data series to calculate unique value statistics for.
    df_length : int
        The total number of rows in the DataFrame.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing unique value statistics.
    """
    try:
        unique_count = series.nunique()
        return {
            "count": int(unique_count),
            "percentage": float(unique_count / df_length * 100) if df_length > 0 else 0,
        }
    except Exception as e:
        logger.warning(f"Error calculating unique statistics: {str(e)}")
        return {"error": str(e)}


def _calculate_top_values(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate the top values for non-numeric fields or fields with few unique values.

    Parameters:
    -----------
    series : pd.Series
        The data series to calculate top values for.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the top values and their counts.
    """
    try:
        value_counts = series.value_counts().head(10).to_dict()
        return {str(k): int(v) for k, v in value_counts.items()}
    except Exception as e:
        logger.warning(f"Error calculating top values: {str(e)}")
        return {"error": str(e)}


def _detect_patterns(series: pd.Series) -> Dict[str, Any]:
    """
    Detect patterns in string fields, such as date/time, numeric strings, and email addresses.

    Parameters:
    -----------
    series : pd.Series
        The data series to detect patterns in.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing detected patterns in the data.
    """
    try:
        sample = (
            series.dropna().sample(min(1000, len(series.dropna())))
            if len(series.dropna()) > 1000
            else series.dropna()
        )

        # Detect potential date/time fields
        date_conversion_success = (
            pd.to_datetime(sample, errors="coerce").notna().mean() > 0.5
        )

        # Detect if field contains mostly numeric values as strings
        numeric_strings = sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.5

        # Detect if field contains mostly email addresses
        email_pattern = (
            sample.str.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").mean()
            > 0.5
        )

        return {
            "potential_date": bool(date_conversion_success),
            "numeric_strings": bool(numeric_strings),
            "email_pattern": bool(email_pattern),
        }
    except Exception as e:
        logger.warning(f"Error during pattern detection: {str(e)}")
        return {"error": str(e)}


def calculate_transformation_impact(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate metrics showing the impact of data transformation on the dataset.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame before transformation.
    transformed_df : pd.DataFrame
        The transformed DataFrame after processing.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the transformation impact metrics.
    """
    if original_df is None or transformed_df is None:
        raise ValueError("Both DataFrames must be provided")

    start_time = time.time()

    result = {}

    # Calculate data quality metrics
    result["data_quality"] = {
        "original": {
            "null_percentage": _calculate_null_percentage(original_df),
            "duplicate_percentage": _calculate_duplicate_percentage(original_df),
        },
        "transformed": {
            "null_percentage": _calculate_null_percentage(transformed_df),
            "duplicate_percentage": _calculate_duplicate_percentage(transformed_df),
        },
    }

    # Calculate data completeness metrics
    completeness_metrics = {
        "original": _calculate_completeness(original_df),
        "transformed": _calculate_completeness(transformed_df),
    }
    result["data_completeness"] = completeness_metrics

    # Calculate distribution changes for numeric fields
    numeric_cols = [
        col
        for col in original_df.columns
        if (
                col in transformed_df.columns
                and pd.api.types.is_numeric_dtype(original_df[col])
                and pd.api.types.is_numeric_dtype(transformed_df[col])
        )
    ]
    distribution_metrics = {}
    for col in numeric_cols:
        distribution_metrics[col] = _calculate_distribution_metrics(
            original_df[col], transformed_df[col]
        )
    result["data_distribution"] = distribution_metrics

    # Calculate correlation changes for numeric fields
    if len(numeric_cols) > 1:
        result["correlation_changes"] = _calculate_correlation_changes(
            original_df, transformed_df, numeric_cols
        )
    else:
        result["correlation_changes"] = {
            "warning": "Not enough numeric columns for correlation analysis"
        }

    # Calculate impact on individual fields
    common_cols = set(original_df.columns).intersection(set(transformed_df.columns))
    result["field_impact"] = _calculate_field_impact(
        original_df, transformed_df, common_cols
    )

    elapsed_time = time.time() - start_time
    result["elapsed_time"] = elapsed_time

    return result


def _calculate_null_percentage(df: pd.DataFrame) -> float:
    """
    Calculate the percentage of null values in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns:
    --------
    float
        The percentage of null values in the DataFrame.
    """
    return df.isna().mean().mean() * 100


def _calculate_duplicate_percentage(df: pd.DataFrame) -> float:
    """
    Calculate the percentage of duplicate rows in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns:
    --------
    float
        The percentage of duplicate rows in the DataFrame.
    """
    return df.duplicated().mean() * 100


def _calculate_completeness(df: pd.DataFrame) -> pd.Series:
    """
    Calculate completeness for each column (percentage of non-null values).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns:
    --------
    pd.Series
        A Series representing the completeness percentage for each column.
    """
    return (1 - df.isna().mean()) * 100


def _calculate_distribution_metrics(
    original_series: pd.Series, transformed_series: pd.Series
) -> Dict[str, Any]:
    """
    Calculate distribution metrics for a numeric field, comparing original vs. transformed data.

    Parameters:
    -----------
    original_series : pd.Series
        The original series to analyze.
    transformed_series : pd.Series
        The transformed series to compare with the original series.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the distribution metrics and results of statistical tests.
    """
    metrics = {}

    try:
        # Basic statistics
        original_mean, transformed_mean = float(original_series.mean()), float(
            transformed_series.mean()
        )
        original_std, transformed_std = float(original_series.std()), float(
            transformed_series.std()
        )

        # Skewness and Kurtosis
        original_skew, transformed_skew = float(original_series.skew()), float(
            transformed_series.skew()
        )
        original_kurt, transformed_kurt = float(original_series.kurt()), float(
            transformed_series.kurt()
        )

        # Kolmogorov-Smirnov Test for distribution difference
        ks_stat, ks_pvalue = None, None
        distribution_changed = None
        if len(original_series) > 5 and len(transformed_series) > 5:
            sample_size = min(1000, min(len(original_series), len(transformed_series)))
            original_sample = (
                original_series.sample(sample_size)
                if len(original_series) > sample_size
                else original_series
            )
            transformed_sample = (
                transformed_series.sample(sample_size)
                if len(transformed_series) > sample_size
                else transformed_series
            )
            ks_stat, ks_pvalue = stats.ks_2samp(original_sample, transformed_sample)
            distribution_changed = ks_pvalue < 0.05

        metrics = {
            "mean": {
                "original": original_mean,
                "transformed": transformed_mean,
                "difference": transformed_mean - original_mean,
            },
            "std": {
                "original": original_std,
                "transformed": transformed_std,
                "difference": transformed_std - original_std,
            },
            "skewness": {
                "original": original_skew,
                "transformed": transformed_skew,
                "difference": transformed_skew - original_skew,
            },
            "kurtosis": {
                "original": original_kurt,
                "transformed": transformed_kurt,
                "difference": transformed_kurt - original_kurt,
            },
            "distribution_test": {
                "ks_statistic": "NaN" if np.isnan(ks_stat) else ks_stat,
                "ks_pvalue": "NaN" if np.isnan(ks_pvalue) else ks_pvalue,
                "distribution_significantly_changed": distribution_changed,
            },
        }
    except Exception as e:
        metrics = {"error": str(e)}

    return metrics


def _calculate_correlation_changes(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame, numeric_cols: list
) -> Dict[str, Any]:
    """
    Calculate the changes in correlation between numeric columns of the original and transformed datasets.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame to compare.
    transformed_df : pd.DataFrame
        The transformed DataFrame to compare with the original.
    numeric_cols : list
        A list of numeric column names to compute correlations for.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the correlation changes for each pair of numeric columns.
    """
    correlation_changes = {}

    original_corr = original_df[numeric_cols].corr()
    transformed_corr = transformed_df[numeric_cols].corr()

    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1 :]:
            orig_corr = original_corr.loc[col1, col2]
            trans_corr = transformed_corr.loc[col1, col2]

            if not (np.isnan(orig_corr) or np.isnan(trans_corr)):
                correlation_changes[f"{col1}_{col2}"] = {
                    "original": float(orig_corr),
                    "transformed": float(trans_corr),
                    "difference": float(trans_corr - orig_corr),
                    "abs_difference": float(abs(trans_corr - orig_corr)),
                }

    return correlation_changes


def _calculate_field_impact(
    original_df: pd.DataFrame, transformed_df: pd.DataFrame, common_cols: set
) -> Dict[str, Any]:
    """
    Calculate the impact of transformation on individual fields such as null values, unique values, and data types.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original DataFrame.
    transformed_df : pd.DataFrame
        The transformed DataFrame.
    common_cols : set
        A set of common columns between the original and transformed DataFrames.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the impact metrics for each field.
    """
    field_impact = {}

    for col in common_cols:
        try:
            impact = {}
            original_nulls = original_df[col].isna().sum()
            transformed_nulls = transformed_df[col].isna().sum()
            impact["null_values"] = {
                "original": int(original_nulls),
                "transformed": int(transformed_nulls),
                "difference": int(transformed_nulls - original_nulls),
                "percent_change": (
                    float((transformed_nulls - original_nulls) / len(original_df) * 100)
                    if len(original_df) > 0
                    else 0
                ),
            }

            original_uniques = original_df[col].nunique()
            transformed_uniques = transformed_df[col].nunique()
            impact["unique_values"] = {
                "original": int(original_uniques),
                "transformed": int(transformed_uniques),
                "difference": int(transformed_uniques - original_uniques),
                "percent_change": (
                    float(
                        (transformed_uniques - original_uniques)
                        / original_uniques
                        * 100
                    )
                    if original_uniques > 0
                    else float("inf")
                ),
            }

            impact["data_type"] = {
                "original": str(original_df[col].dtype),
                "transformed": str(transformed_df[col].dtype),
                "changed": str(original_df[col].dtype)
                != str(transformed_df[col].dtype),
            }

            field_impact[col] = impact
        except Exception as e:
            field_impact[col] = {"error": str(e)}

    return field_impact


def calculate_performance_metrics(
    start_time: float, end_time: float, input_rows: int, output_rows: int
) -> Dict[str, Any]:
    """
    Calculate performance metrics for the operation.

    Parameters:
    -----------
    start_time : float
        Start time of the operation in seconds (from time.time()).
    end_time : float
        End time of the operation in seconds (from time.time()).
    input_rows : int
        Number of input rows processed.
    output_rows : int
        Number of output rows produced.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing performance metrics:
        - elapsed_time: Total time taken in seconds.
        - rows_per_second: Processing speed in rows per second.
        - throughput_ratio: Ratio of output rows to input rows.
        - timing_breakdown: Optional breakdown of timing by phases (if provided).
    """
    try:
        # Validate start and end times
        _validate_times(start_time, end_time)

        # Calculate elapsed time
        elapsed_time = _calculate_elapsed_time(start_time, end_time)

        result = {
            "elapsed_time": {
                "seconds": float(elapsed_time),
                "formatted": format_time_duration(elapsed_time),
            }
        }

        # Calculate throughput metrics
        result["rows_per_second"] = _calculate_rows_per_second(input_rows, elapsed_time)

        # Calculate throughput ratio
        result["throughput_ratio"] = _calculate_throughput_ratio(
            input_rows, output_rows
        )

        # Determine performance rating
        if elapsed_time > 0 and input_rows > 0:
            rows_per_second = _calculate_rows_per_second(input_rows, elapsed_time)
            result["performance_rating"] = _determine_performance_rating(
                rows_per_second
            )

        return result

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        raise


def _validate_times(start_time: float, end_time: float) -> None:
    """
    Validate the provided start and end times.

    Parameters:
    -----------
    start_time : float
        Start time of the operation in seconds.
    end_time : float
        End time of the operation in seconds.

    Raises:
    -------
    ValueError
        If start_time is None or greater than end_time.
    """
    if start_time is None or end_time is None:
        raise ValueError("Start and end times must be provided")

    if start_time > end_time:
        raise ValueError("Start time cannot be after end time")


def _calculate_elapsed_time(start_time: float, end_time: float) -> float:
    """
    Calculate the elapsed time between start and end times.

    Parameters:
    -----------
    start_time : float
        Start time of the operation in seconds.
    end_time : float
        End time of the operation in seconds.

    Returns:
    --------
    float
        The elapsed time in seconds.
    """
    return end_time - start_time


def _calculate_rows_per_second(input_rows: int, elapsed_time: float) -> float:
    """
    Calculate the processing speed in rows per second.

    Parameters:
    -----------
    input_rows : int
        Number of input rows processed.
    elapsed_time : float
        Total time taken for processing in seconds.

    Returns:
    --------
    float
        The processing speed in rows per second. Returns infinity if elapsed_time is zero.
    """
    return float(input_rows / elapsed_time) if elapsed_time > 0 else float("inf")


def _calculate_throughput_ratio(input_rows: int, output_rows: int) -> Dict[str, float]:
    """
    Calculate the throughput ratio and percent change between input and output rows.

    Parameters:
    -----------
    input_rows : int
        Number of input rows processed.
    output_rows : int
        Number of output rows produced.

    Returns:
    --------
    Dict[str, float]
        A dictionary containing:
        - ratio: The ratio of output rows to input rows.
        - percent_change: The percent change in rows from input to output.
    """
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "ratio": float(output_rows / input_rows) if input_rows > 0 else float("inf"),
        "percent_change": (
            float((output_rows - input_rows) / input_rows * 100)
            if input_rows > 0
            else float("inf")
        ),
    }


def _determine_performance_rating(rows_per_second: float) -> str:
    """
    Determine the performance rating based on rows processed per second.

    Parameters:
    -----------
    rows_per_second : float
        The processing speed in rows per second.

    Returns:
    --------
    str
        The performance rating: "Excellent", "Good", "Average", "Below Average", or "Poor".
    """
    if rows_per_second >= 100000:
        return "Excellent"
    elif rows_per_second >= 10000:
        return "Good"
    elif rows_per_second >= 1000:
        return "Average"
    elif rows_per_second >= 100:
        return "Below Average"
    else:
        return "Poor"


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in a human-readable format.

    Parameters:
    -----------
    seconds : float
        The time duration in seconds.

    Returns:
    --------
    str
        The formatted time duration as a string.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def save_metrics_json(
    metrics: Dict[str, Any],
    task_dir: Path,
    operation_name: str,
    field_name: str,
    writer: Optional[DataWriter] = None,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    use_encryption: bool = False,
) -> Path:
    """
    Save metrics to a JSON file using DataWriter if available, otherwise direct file write.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics to save
    task_dir : Path
        Task directory
    operation_name : str
        Name of the operation
    field_name : str
        Name of the field
    writer : Optional[DataWriter]
        DataWriter instance to use for saving
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for the operation
    use_encryption : bool, optional
        Whether to use encryption when saving the file (default is False)

    Returns:
    --------
    Path
        Path to the saved metrics file
    """
    # Update progress if provided
    if progress_tracker:
        progress_tracker.update(0, {"step": "Saving metrics"})

    # Add timestamp and metadata
    metrics_with_metadata = (
        metrics.copy()
    )  # Create a copy to avoid modifying the input dictionary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_with_metadata.update(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation_name,
            "field": field_name,
        }
    )

    # Create filename base with meaningful information
    base_filename = f"{field_name}_{operation_name}_metrics"

    # Create complete filename with timestamp
    filename = f"{base_filename}_{timestamp}.json"

    # Use DataWriter if available
    if writer:
        try:
            encryption_key = None
            if use_encryption:
                encryption_key = f"{field_name}_{timestamp}_key"

            # Use DataWriter to save metrics
            metrics_result = writer.write_metrics(
                metrics=metrics_with_metadata,
                name=base_filename,
                timestamp_in_name=True,
                encryption_key=encryption_key,
            )

            logger.info(f"Metrics saved using DataWriter to {metrics_result.path}")
            return Path(metrics_result.path)

        except Exception as e:
            logger.warning(f"Failed to save metrics using DataWriter: {e}")
            logger.info("Falling back to direct file write")

    # Fallback to direct file write
    try:
        # Ensure task_dir exists
        ensure_directory(task_dir)

        # Create file path
        file_path = task_dir / filename

        # Use the io module to write the JSON file
        json_path = write_json(
            metrics_with_metadata,
            file_path,
            encoding="utf-8",
            indent=2,
            ensure_ascii=False,
            convert_numpy=True,
            encryption_key=(
                None if not use_encryption else f"{field_name}_{timestamp}_key"
            ),
        )

        logger.info(f"Metrics saved to {json_path}")
        return json_path

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        # Return original path even if save failed
        return task_dir / filename
