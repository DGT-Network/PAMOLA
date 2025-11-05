"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Currency Field Profiler Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module defines operations for profiling currency fields in tabular datasets.
  It extracts detailed statistical summaries, handles locale-aware parsing of currency
  values, and produces artifacts to support subsequent anonymization, data quality
  evaluation, and semantic transformation.

Key Features:
  - Locale-aware parsing and profiling of currency fields
  - Descriptive statistics, outlier detection, and normality testing
  - Robust handling of locale-specific formatting and inconsistencies
  - Visualization generation for distributions and boxplots
  - Multi-currency detection and semantic notes
  - Efficient chunked, parallel, and Dask-based processing for large datasets
  - Integration with PAMOLA.CORE operation framework for standardized input/output
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Union

import dask
import dask.dataframe as dd

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from pamola_core.profiling.commons.currency_utils import (
    is_currency_field,
    parse_currency_field,
    analyze_currency_stats,
    detect_currency_from_sample,
    generate_currency_samples,
    create_empty_currency_stats,
)
from pamola_core.profiling.commons.numeric_utils import (
    calculate_percentiles,
    calculate_histogram,
)
from pamola_core.profiling.schemas.currency_schema import CurrencyOperationConfig
from pamola_core.utils.io import (
    write_dataframe_to_csv,
    write_json,
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_processing import get_dataframe_chunks
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import (
    OperationArtifact,
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import (
    create_histogram,
    create_boxplot,
    create_correlation_pair_plot,
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

# Configure logger
logger = logging.getLogger(__name__)


class CurrencyAnalyzer:
    """
    Analyzer for currency fields.

    This analyzer provides methods for handling, parsing, and analyzing
    currency fields in tabular datasets with support for locale-aware
    parsing and robust error handling.
    """

    def analyze(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        field_name: str,
        locale: str = "en_US",
        bins: int = 10,
        detect_outliers: bool = True,
        test_normality: bool = True,
        chunk_size: int = 10000,
        use_dask: bool = False,
        use_vectorization: bool = False,
        parallel_processes: int = 1,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a currency field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        locale : str
            Locale to use for parsing (default: 'en_US')
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        chunk_size : int
            Size of chunks for processing large datasets
        use_dask : bool
            Whether to use Dask for large datasets
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        task_logger : Optional[logging.Logger]
            Logger for tracking task progress and debugging.

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        if task_logger:
            global logger
            logger = task_logger

        logger.info(f"Analyzing currency field: {field_name}")

        # Validate input
        if field_name not in df.columns:
            logger.error(f"Field {field_name} not found in DataFrame")
            return {"error": f"Field {field_name} not found in DataFrame"}

        total_rows = (
            int(df.map_partitions(len).sum().compute())
            if isinstance(df, dd.DataFrame)
            else len(df)
        )

        # Check if this is actually a currency field
        # This is both a validation step and may determine parsing approach
        is_detected_currency = is_currency_field(field_name)

        if not is_detected_currency:
            logger.info(f"Field {field_name} does not appear to be a currency field. ")
            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Field validation",
                        "warning": "Not likely a currency field",
                    },
                )

        # Initialize result structure
        result = {
            "field_name": field_name,
            "total_rows": total_rows,
            "is_detected_currency": is_detected_currency,
            "locale_used": locale,
        }

        # Handle large datasets with chunking if needed
        is_large_df = total_rows > chunk_size

        if is_large_df is False:
            task_logger.warning("Small DataFrame! Process as usual")
            return self._analyze_small_dataset(
                df.compute() if isinstance(df, dd.DataFrame) else df,
                field_name,
                locale,
                bins,
                total_rows,
                detect_outliers,
                test_normality,
                progress_tracker,
                logger,
            )

        if use_dask and is_large_df:
            try:
                logger.info(f"Using Dask for large dataset with {total_rows} rows")
                return self._analyze_with_dask(
                    df,
                    field_name,
                    locale,
                    bins,
                    detect_outliers,
                    test_normality,
                    progress_tracker,
                    logger,
                )
            except ImportError:
                logger.warning(
                    "Dask requested but not available. Falling back to chunked processing."
                )
                if progress_tracker:
                    progress_tracker.update(
                        0,
                        {
                            "step": "Dask fallback",
                            "warning": "Dask not available, using chunks",
                        },
                    )

        # Process in parallel
        if use_vectorization and parallel_processes > 0:
            return self._analyze_with_parallel(
                df,
                field_name,
                locale,
                chunk_size,
                bins,
                detect_outliers,
                test_normality,
                parallel_processes,
                progress_tracker,
                logger,
            )
        # Process in chunks for large datasets
        elif is_large_df and not use_dask:
            return self._analyze_in_chunks(
                df,
                field_name,
                locale,
                bins,
                detect_outliers,
                test_normality,
                chunk_size,
                progress_tracker,
                logger,
            )

        # Standard processing for smaller datasets
        if progress_tracker:
            progress_tracker.update(
                1, {"step": "Parsing currency values", "rows": total_rows}
            )

        # Parse currency values
        try:
            normalized_values, currency_counts = parse_currency_field(
                df, field_name, locale
            )

            # Count valid, null, and invalid values
            if hasattr(normalized_values, "valid_flags"):
                valid_flags = normalized_values.valid_flags
                valid_count = sum(
                    1
                    for flag, val in zip(valid_flags, normalized_values)
                    if flag and pd.notna(val)
                )
            else:
                valid_count = normalized_values.notna().sum()
            null_count = normalized_values.isna().sum()
            invalid_count = total_rows - valid_count - null_count

            result.update(
                {
                    "valid_count": valid_count,
                    "null_count": null_count,
                    "invalid_count": invalid_count,
                    "null_percentage": (
                        (null_count / total_rows * 100) if total_rows > 0 else 0.0
                    ),
                    "invalid_percentage": (
                        (invalid_count / total_rows * 100) if total_rows > 0 else 0.0
                    ),
                    "currency_counts": currency_counts,
                    "multi_currency": len(currency_counts) > 1,
                }
            )

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Currency parsing complete",
                        "valid": valid_count,
                        "null": null_count,
                        "invalid": invalid_count,
                    },
                )

        except Exception as e:
            logger.error(
                f"Error parsing currency field {field_name}: {e}", exc_info=True
            )
            return {
                "error": f"Error parsing currency field: {str(e)}",
                "field_name": field_name,
                "total_rows": total_rows,
            }

        # Calculate statistics on valid values
        valid_values = normalized_values.dropna()

        if len(valid_values) == 0:
            stats = create_empty_currency_stats()
            result["stats"] = stats

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Statistics calculation",
                        "warning": "No valid values for statistics",
                    },
                )

            return result

        # Calculate currency statistics
        try:
            stats = analyze_currency_stats(valid_values, currency_counts)

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Basic statistics calculated",
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                    },
                )

            # Calculate percentiles
            stats["percentiles"] = calculate_percentiles(valid_values)

            # Calculate histogram data
            stats["histogram"] = calculate_histogram(valid_values, bins)

            # Detect outliers if requested
            if detect_outliers:
                from pamola_core.profiling.commons.numeric_utils import (
                    detect_outliers as detect_outliers_func,
                )

                outlier_results = detect_outliers_func(valid_values)
                stats["outliers"] = outlier_results

                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Outlier detection",
                            "outliers_found": outlier_results.get("count", 0),
                        },
                    )

            # Test normality if requested and we have enough data
            if test_normality and len(valid_values) >= 8:
                try:
                    from pamola_core.profiling.commons.numeric_utils import (
                        test_normality as test_normality_func,
                    )

                    normality_results = test_normality_func(valid_values)
                    stats["normality"] = normality_results

                    if progress_tracker:
                        progress_tracker.update(
                            1,
                            {
                                "step": "Normality testing",
                                "is_normal": normality_results.get("is_normal", False),
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"Error during normality testing for {field_name}: {e}"
                    )
                    stats["normality"] = {"error": str(e), "is_normal": False}
            else:
                stats["normality"] = {
                    "is_normal": False,
                    "message": (
                        "Insufficient data for normality testing"
                        if len(valid_values) < 8
                        else "Normality testing skipped"
                    ),
                }

            # Generate samples for JSON output
            stats["samples"] = generate_currency_samples(stats)

            # Currency-specific semantic analysis
            negative_count = stats.get("negative_count", 0)
            if negative_count > 0:
                stats["semantic_notes"] = stats.get("semantic_notes", []) + [
                    f"Field contains {negative_count} negative values, possibly representing debits or expenses."
                ]

            zero_count = stats.get("zero_count", 0)
            if zero_count > 0:
                stats["semantic_notes"] = stats.get("semantic_notes", []) + [
                    f"Field contains {zero_count} zero values, possibly representing unpaid/free items or placeholder values."
                ]

            # Check for suspiciously large values (potential data entry errors)
            if stats.get("max", 0) > stats.get("mean", 0) * 100:
                stats["semantic_notes"] = stats.get("semantic_notes", []) + [
                    "Field contains extremely large values that may be data entry errors."
                ]

            # Add to result
            result["stats"] = stats

            # Finalize progress
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Analysis complete", "field": field_name}
                )

            return result

        except Exception as e:
            logger.error(
                f"Error calculating statistics for {field_name}: {e}", exc_info=True
            )
            result["error"] = f"Error calculating statistics: {str(e)}"
            return result

    def _analyze_with_dask(
        self,
        ddf: dd.DataFrame,
        field_name: str,
        locale: str,
        bins: int,
        detect_outliers: bool,
        test_normality: bool,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a currency field using Dask for large datasets.

        Parameters:
        -----------
        df : dd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        locale : str
            Locale to use for parsing (default: 'en_US')
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        task_logger : Optional[logging.Logger]
            Logger for tracking task progress and debugging.

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        try:
            if task_logger:
                global logger
                logger = task_logger

            logger.info("Parallel Enabled")
            logger.info("Parallel Engine: Dask")

            # Basic counts that can be done with Dask
            total_rows = len(ddf)
            null_count = ddf[field_name].isna().sum().compute()

            # For currency parsing, we need custom logic that may not work well with Dask
            # Do a sample-based analysis first
            sample_size = min(1000, total_rows)
            sample_df = ddf.sample(n=sample_size) if total_rows > sample_size else ddf

            # Detect currency from sample
            detected_currency = detect_currency_from_sample(sample_df, field_name)

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {"step": "Sample analysis", "detected_currency": detected_currency},
                )

            # Define a function to normalize values that can be applied to Dask partitions
            def normalize_partition(partition):
                normalized, _ = parse_currency_field(partition, field_name, locale)
                return normalized

            # Apply to Dask DataFrame
            normalized_values = ddf.map_partitions(
                normalize_partition, meta=pd.Series(dtype="float64")
            )

            # Compute basic statistics
            (mean, median, std, min_val, max_val) = dask.compute(
                normalized_values.mean(),
                normalized_values.quantile(0.5),
                normalized_values.std(),
                normalized_values.min(),
                normalized_values.max(),
            )

            # Create results structure
            valid_count = total_rows - null_count
            result = {
                "field_name": field_name,
                "total_rows": total_rows,
                "valid_count": valid_count,
                "null_count": null_count,
                "null_percentage": (
                    (null_count / total_rows * 100) if total_rows > 0 else 0.0
                ),
                "valid_values": normalized_values.dropna(),
                "is_detected_currency": True,
                "locale_used": locale,
                "detected_currency": detected_currency,
                "stats": {
                    "min": float(min_val),
                    "max": float(max_val),
                    "mean": float(mean),
                    "median": float(median),
                    "std": float(std),
                    "valid_count": int(valid_count),
                    "multi_currency": False,  # Simplified assumption with Dask
                    "currency_distribution": (
                        {detected_currency: valid_count}
                        if detected_currency != "UNKNOWN"
                        else {}
                    ),
                },
            }

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Dask statistics calculated",
                        "min": result["stats"]["min"],
                        "max": result["stats"]["max"],
                    },
                )

            # For more advanced statistics, we'd need to bring data back to pandas
            # which may defeat the purpose of using Dask for large datasets
            # Consider implementing with Dask's built-in histogram, etc.

            # Add note about Dask usage
            result["note"] = (
                "Analysis performed using Dask for large dataset. Some detailed metrics may be unavailable."
            )

            return result

        except Exception as e:
            logger.error(f"Error in Dask analysis for {field_name}: {e}", exc_info=True)
            return {
                "error": f"Error in Dask analysis: {str(e)}",
                "field_name": field_name,
            }

    def _analyze_in_chunks(
        self,
        df: pd.DataFrame,
        field_name: str,
        locale: str,
        bins: int,
        detect_outliers: bool,
        test_normality: bool,
        chunk_size: int,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a currency field in chunks for large datasets.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        locale : str
            Locale to use for parsing (default: 'en_US')
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        chunk_size : int
            Size of chunks for processing large datasets
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        task_logger : Optional[logging.Logger]
            Logger for tracking task progress and debugging.

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        if task_logger:
            logger = task_logger

        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size

        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Chunked processing setup",
                    "total_chunks": total_chunks,
                    "chunk_size": chunk_size,
                },
            )

        # Initialize accumulators
        all_values = []
        all_currencies = {}
        null_count = 0
        invalid_count = 0

        # Process each chunk
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]

            # Parse chunk
            normalized_values, currency_counts = parse_currency_field(
                chunk, field_name, locale
            )

            # Count nulls and invalids
            chunk_valid_count = normalized_values.notna().sum()
            chunk_null_count = normalized_values.isna().sum()
            chunk_invalid_count = len(chunk) - chunk_valid_count - chunk_null_count

            # Update accumulators
            all_values.append(normalized_values.dropna())
            null_count += chunk_null_count
            invalid_count += chunk_invalid_count

            # Merge currency counts
            for currency, count in currency_counts.items():
                all_currencies[currency] = all_currencies.get(currency, 0) + count

            if progress_tracker:
                progress_tracker.update(
                    0,
                    {
                        "step": f"Processing chunk {i + 1}/{total_chunks}",
                        "valid": chunk_valid_count,
                        "null": chunk_null_count,
                        "invalid": chunk_invalid_count,
                    },
                )

        # Combine values from all chunks
        if all_values:
            combined_values = pd.concat(all_values)
            if isinstance(combined_values, pd.DataFrame):
                if combined_values.shape[1] == 1:
                    combined_values = combined_values.iloc[:, 0]
                else:
                    combined_values = combined_values.squeeze()
            if not isinstance(combined_values, pd.Series):
                combined_values = pd.Series([combined_values])
        else:
            combined_values = pd.Series(dtype="float64")

        valid_count = len(combined_values)

        # Create result structure
        result = {
            "field_name": field_name,
            "total_rows": total_rows,
            "valid_count": valid_count,
            "null_count": null_count,
            "invalid_count": invalid_count,
            "null_percentage": (
                (null_count / total_rows * 100) if total_rows > 0 else 0.0
            ),
            "valid_values": normalized_values.dropna(),
            "invalid_percentage": (
                (invalid_count / total_rows * 100) if total_rows > 0 else 0.0
            ),
            "currency_counts": all_currencies,
            "multi_currency": len(all_currencies) > 1,
            "is_detected_currency": is_currency_field(field_name),
            "locale_used": locale,
        }

        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Chunks processed",
                    "valid_total": valid_count,
                    "currencies_detected": len(all_currencies),
                },
            )

        # If we have no valid values, return early
        if valid_count == 0:
            result["stats"] = create_empty_currency_stats()
            return result

        # Calculate statistics on combined values
        stats = analyze_currency_stats(combined_values, all_currencies)

        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Statistics calculated",
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                },
            )

        # Add percentiles and histogram
        stats["percentiles"] = calculate_percentiles(combined_values)
        stats["histogram"] = calculate_histogram(combined_values, bins)

        # Detect outliers if requested and we have enough data
        if detect_outliers and len(combined_values) >= 5:
            from pamola_core.profiling.commons.numeric_utils import (
                detect_outliers as detect_outliers_func,
            )

            stats["outliers"] = detect_outliers_func(combined_values)
        else:
            stats["outliers"] = {
                "count": 0,
                "percentage": 0.0,
                "message": "Outlier detection skipped or insufficient data",
            }

        # Test normality if requested and we have enough data
        if test_normality and len(combined_values) >= 8:
            try:
                from pamola_core.profiling.commons.numeric_utils import (
                    test_normality as test_normality_func,
                )

                stats["normality"] = test_normality_func(combined_values)
            except Exception as e:
                logger.warning(f"Error in normality testing for {field_name}: {e}")
                stats["normality"] = {"is_normal": False, "error": str(e)}
        else:
            stats["normality"] = {
                "is_normal": False,
                "message": (
                    "Insufficient data for normality testing"
                    if len(combined_values) < 8
                    else "Normality testing skipped"
                ),
            }

        # Generate samples for JSON output
        stats["samples"] = generate_currency_samples(stats)

        # Add to result
        result["stats"] = stats
        result["note"] = "Analysis performed in chunks due to large dataset size"

        if progress_tracker:
            progress_tracker.update(
                1, {"step": "Chunked analysis complete", "field": field_name}
            )

        return result

    def _analyze_with_parallel(
        self,
        df: pd.DataFrame,
        field_name: str,
        locale: str,
        chunk_size: int,
        bins: int,
        detect_outliers: bool,
        test_normality: bool,
        parallel_processes: int = -1,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a currency field using parallel processing for large datasets.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        locale : str
            Locale to use for parsing
        chunk_size : int
            Size of chunks for processing
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        n_jobs : int
            Number of parallel jobs to run (-1 for all cores)
        parallel_processes : int, optional
            Number of parallel processes to use (default: None, uses all available cores)
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        try:
            if task_logger:
                global logger
                logger = task_logger

            logger.info(f"Starting parallel analysis for field: {field_name}")
            total_rows = len(df)
            chunks = list(get_dataframe_chunks(df, chunk_size=chunk_size))
            total_chunks = len(chunks)

            if progress_tracker:
                progress_tracker.total = total_chunks
                progress_tracker.update(
                    0,
                    {
                        "step": "Parallel processing setup",
                        "total_chunks": total_chunks,
                        "chunk_size": chunk_size,
                    },
                )

            logger.info(
                f"Processing {total_rows} rows in {total_chunks} chunks with {parallel_processes} workers"
            )
            start_time = time.time()

            def process_chunk(chunk_idx, chunk_df):
                """Process a single chunk of data"""
                logger.debug(
                    f"Processing chunk {chunk_idx+1}/{total_chunks} (rows {chunk_df.index[0]}-{chunk_df.index[-1]})"
                )
                normalized_values, currency_counts = parse_currency_field(
                    chunk_df, field_name, locale
                )
                valid_values = normalized_values.dropna()

                logger.debug(
                    f"Chunk {chunk_idx+1}: valid={len(valid_values)}, null={normalized_values.isna().sum()}, invalid={len(chunk_df) - len(valid_values) - normalized_values.isna().sum()}"
                )
                return {
                    "chunk_idx": chunk_idx,
                    "valid_values": valid_values,
                    "currency_counts": currency_counts,
                    "valid_count": len(valid_values),
                    "null_count": normalized_values.isna().sum(),
                    "invalid_count": len(chunk_df)
                    - len(valid_values)
                    - normalized_values.isna().sum(),
                    "chunk_size": len(chunk_df),
                }

            logger.info("Launching parallel processing of chunks...")
            processed_chunks = Parallel(n_jobs=parallel_processes)(
                delayed(process_chunk)(i, chunk) for i, chunk in enumerate(chunks)
            )
            logger.info("Parallel processing of chunks complete.")

            if progress_tracker:
                progress_tracker.update(
                    total_chunks,
                    {
                        "step": "Parallel processing complete",
                        "chunks_processed": len(processed_chunks),
                    },
                )

            all_values = []
            all_currencies = {}
            total_valid_count = 0
            total_null_count = 0
            total_invalid_count = 0

            for i, chunk in enumerate(processed_chunks):
                if chunk is None:
                    logger.warning(f"Chunk {i} is None. Skipping.")
                    continue
                if not chunk.get("valid_values", pd.Series(dtype="float64")).empty:
                    all_values.append(chunk["valid_values"])

                for currency, count in chunk.get("currency_counts", {}).items():
                    all_currencies[currency] = all_currencies.get(currency, 0) + count

                total_valid_count += chunk.get("valid_count", 0)
                total_null_count += chunk.get("null_count", 0)
                total_invalid_count += chunk.get("invalid_count", 0)

            logger.info(
                f"Aggregated results: valid={total_valid_count}, null={total_null_count}, invalid={total_invalid_count}, currencies={all_currencies}"
            )

            combined_values = (
                pd.concat(all_values, ignore_index=True)
                if all_values
                else pd.Series(dtype="float64")
            )
            if isinstance(combined_values, pd.DataFrame):
                combined_values = combined_values.squeeze()
            if not isinstance(combined_values, pd.Series):
                combined_values = pd.Series([combined_values])

            sample_df = df.sample(n=min(1000, total_rows)) if total_rows > 0 else df
            detected_currency = detect_currency_from_sample(sample_df, field_name)

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {"step": "Sample analysis", "detected_currency": detected_currency},
                )

            result = {
                "field_name": field_name,
                "total_rows": total_rows,
                "valid_count": total_valid_count,
                "null_count": total_null_count,
                "invalid_count": total_invalid_count,
                "null_percentage": (
                    (total_null_count / total_rows * 100) if total_rows else 0.0
                ),
                "valid_values": combined_values.dropna(),
                "invalid_percentage": (
                    (total_invalid_count / total_rows * 100) if total_rows else 0.0
                ),
                "currency_counts": all_currencies,
                "multi_currency": len(all_currencies) > 1,
                "is_detected_currency": is_currency_field(field_name),
                "locale_used": locale,
                "detected_currency": detected_currency,
            }

            if progress_tracker:
                progress_tracker.update(
                    0,
                    {
                        "step": "Aggregating results",
                        "valid_total": total_valid_count,
                        "currencies_detected": len(all_currencies),
                    },
                )

            if combined_values.empty:
                logger.warning(
                    "No valid currency values found after parallel processing."
                )
                result["stats"] = create_empty_currency_stats()
                result["note"] = (
                    "Analysis completed with parallel processing. No valid currency values found."
                )
                return result

            logger.info("Calculating statistics on combined values...")
            stats = analyze_currency_stats(combined_values, all_currencies)
            stats["percentiles"] = calculate_percentiles(combined_values)
            stats["histogram"] = calculate_histogram(combined_values, bins)

            # Outlier Detection
            if detect_outliers and len(combined_values) >= 5:
                try:
                    from pamola_core.profiling.commons.numeric_utils import (
                        detect_outliers as detect_outliers_func,
                    )

                    stats["outliers"] = detect_outliers_func(combined_values)
                    logger.info(
                        f"Outlier detection complete. Outliers found: {stats['outliers'].get('count', 0)}"
                    )
                    if progress_tracker:
                        progress_tracker.update(
                            0,
                            {
                                "step": "Outlier detection complete",
                                "outliers_found": stats["outliers"].get("count", 0),
                            },
                        )
                except Exception as e:
                    logger.warning(f"Error in outlier detection for {field_name}: {e}")
                    stats["outliers"] = {"count": 0, "percentage": 0.0, "error": str(e)}
            else:
                stats["outliers"] = {
                    "count": 0,
                    "percentage": 0.0,
                    "message": "Skipped or insufficient data",
                }

            # Normality Test
            if test_normality and len(combined_values) >= 8:
                try:
                    from pamola_core.profiling.commons.numeric_utils import (
                        test_normality as test_normality_func,
                    )

                    stats["normality"] = test_normality_func(combined_values)
                    logger.info(
                        f"Normality testing complete. Is normal: {stats['normality'].get('is_normal', False)}"
                    )
                    if progress_tracker:
                        progress_tracker.update(
                            0,
                            {
                                "step": "Normality testing complete",
                                "is_normal": stats["normality"].get("is_normal", False),
                            },
                        )
                except Exception as e:
                    logger.warning(f"Error in normality testing for {field_name}: {e}")
                    stats["normality"] = {"is_normal": False, "error": str(e)}
            else:
                stats["normality"] = {
                    "is_normal": False,
                    "message": (
                        "Insufficient data for normality testing"
                        if len(combined_values) < 8
                        else "Skipped"
                    ),
                }

            stats["samples"] = generate_currency_samples(stats)

            # Semantic Notes
            semantic_notes = []
            if stats.get("negative_count", 0) > 0:
                semantic_notes.append(
                    f"Field contains {stats['negative_count']} negative values."
                )
            if stats.get("zero_count", 0) > 0:
                semantic_notes.append(
                    f"Field contains {stats['zero_count']} zero values."
                )
            if stats.get("max", 0) > stats.get("mean", 0) * 100:
                semantic_notes.append(
                    "Field contains extremely large values that may be data entry errors."
                )
            if semantic_notes:
                stats["semantic_notes"] = semantic_notes

            processing_time = round(time.time() - start_time, 2)
            stats["processing_metadata"] = {
                "method": "parallel",
                "chunks_processed": total_chunks,
                "n_jobs": parallel_processes,
                "chunk_size": chunk_size,
                "processing_time_seconds": processing_time,
            }

            logger.info(f"Parallel analysis completed in {processing_time:.2f} seconds")
            result["stats"] = stats
            result["note"] = (
                f"Analysis performed using parallel processing with {total_chunks} chunks and {parallel_processes} workers."
            )
            result["processing_time"] = processing_time

            if progress_tracker:
                progress_tracker.update(
                    0,
                    {
                        "step": "Parallel analysis complete",
                        "field": field_name,
                        "processing_time": processing_time,
                    },
                )

            return result

        except Exception as e:
            logger.error(
                f"Error in parallel analysis for {field_name}: {e}", exc_info=True
            )
            return {
                "error": f"Error in parallel analysis: {str(e)}",
                "field_name": field_name,
                "processing_method": "parallel",
            }

    def _analyze_small_dataset(
        self,
        df: pd.DataFrame,
        field_name: str,
        locale: str,
        bins: int,
        total_rows: int,
        detect_outliers: bool,
        test_normality: bool,
        progress_tracker: Optional[HierarchicalProgressTracker],
        logger: logging.Logger,
    ) -> Dict[str, Any]:
        result = {
            "field_name": field_name,
            "total_rows": total_rows,
            "locale_used": locale,
        }

        # Step 1: Parse currency values
        if progress_tracker:
            progress_tracker.update(
                1, {"step": "Parsing currency values", "rows": total_rows}
            )

        try:
            normalized_values, currency_counts = parse_currency_field(
                df, field_name, locale
            )

            if hasattr(normalized_values, "valid_flags"):
                valid_flags = normalized_values.valid_flags
                valid_count = sum(
                    1
                    for flag, val in zip(valid_flags, normalized_values)
                    if flag and pd.notna(val)
                )
            else:
                valid_count = normalized_values.notna().sum()

            null_count = normalized_values.isna().sum()
            invalid_count = total_rows - valid_count - null_count

            result.update(
                {
                    "valid_count": valid_count,
                    "null_count": null_count,
                    "invalid_count": invalid_count,
                    "null_percentage": (
                        (null_count / total_rows * 100) if total_rows > 0 else 0.0
                    ),
                    "valid_values": normalized_values.dropna(),
                    "invalid_percentage": (
                        (invalid_count / total_rows * 100) if total_rows > 0 else 0.0
                    ),
                    "currency_counts": currency_counts,
                    "multi_currency": len(currency_counts) > 1,
                }
            )

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Currency parsing complete",
                        "valid": valid_count,
                        "null": null_count,
                        "invalid": invalid_count,
                    },
                )

        except Exception as e:
            logger.error(
                f"Error parsing currency field {field_name}: {e}", exc_info=True
            )
            return {
                "error": f"Error parsing currency field: {str(e)}",
                "field_name": field_name,
                "total_rows": total_rows,
            }

        valid_values = normalized_values.dropna()

        if len(valid_values) == 0:
            stats = create_empty_currency_stats()
            result["stats"] = stats

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Statistics calculation",
                        "warning": "No valid values for statistics",
                    },
                )

            return result

        try:
            stats = analyze_currency_stats(valid_values, currency_counts)

            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Basic statistics calculated",
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                    },
                )

            stats["percentiles"] = calculate_percentiles(valid_values)
            stats["histogram"] = calculate_histogram(valid_values, bins)

            if detect_outliers:
                from pamola_core.profiling.commons.numeric_utils import (
                    detect_outliers as detect_outliers_func,
                )

                stats["outliers"] = detect_outliers_func(valid_values)
                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Outlier detection",
                            "outliers_found": stats["outliers"].get("count", 0),
                        },
                    )

            if test_normality and len(valid_values) >= 8:
                from pamola_core.profiling.commons.numeric_utils import (
                    test_normality as test_normality_func,
                )

                stats["normality"] = test_normality_func(valid_values)
                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Normality testing",
                            "is_normal": stats["normality"].get("is_normal", False),
                        },
                    )
            else:
                stats["normality"] = {
                    "is_normal": False,
                    "message": (
                        "Insufficient data for normality testing"
                        if len(valid_values) < 8
                        else "Normality testing skipped"
                    ),
                }

            stats["samples"] = generate_currency_samples(stats)

            if stats.get("negative_count", 0) > 0:
                stats.setdefault("semantic_notes", []).append(
                    f"Field contains {stats['negative_count']} negative values, possibly representing debits or expenses."
                )
            if stats.get("zero_count", 0) > 0:
                stats.setdefault("semantic_notes", []).append(
                    f"Field contains {stats['zero_count']} zero values, possibly representing unpaid/free items or placeholder values."
                )
            if stats.get("max", 0) > stats.get("mean", 0) * 100:
                stats.setdefault("semantic_notes", []).append(
                    "Field contains extremely large values that may be data entry errors."
                )

            result["stats"] = stats

            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Analysis complete", "field": field_name}
                )

            return result

        except Exception as e:
            logger.error(
                f"Error calculating statistics for {field_name}: {e}", exc_info=True
            )
            result["error"] = f"Error calculating statistics: {str(e)}"
            return result


@register(version="1.0.0")
class CurrencyOperation(FieldOperation):
    """
    Operation for analyzing currency fields.

    This operation extends the FieldOperation base class and provides methods for
    executing currency field analysis, including visualization generation and result reporting.
    """

    def __init__(
        self,
        field_name: str,
        locale: str = "en_US",
        bins: int = 10,
        detect_outliers: bool = True,
        test_normality: bool = True,
        **kwargs,
    ):
        """
        Initialize a currency field operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to analyze
        locale : str
            Locale to use for parsing (default: 'en_US')
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        **kwargs
            Additional keyword arguments passed to FieldOperation.
        """
        # Description fallback
        kwargs.setdefault("description", f"Analysis of currency field '{field_name}'")

        # --- Build unified config ---
        config = CurrencyOperationConfig(
            field_name=field_name,
            locale=locale,
            bins=bins,
            detect_outliers=detect_outliers,
            test_normality=test_normality,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize base FieldOperation
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Analyzer binding
        self.analyzer = CurrencyAnalyzer()

        # Operation metadata
        self.operation_name = self.__class__.__name__

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the currency field analysis operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            if kwargs.get("logger"):
                self.logger = kwargs["logger"]

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            # Set up directories
            dirs = self._prepare_directories(task_dir)
            output_dir = dirs["output"]
            visualizations_dir = dirs["visualizations"]
            dictionaries_dir = dirs["dictionaries"]

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create the main result object with initial status
            result = OperationResult(status=OperationStatus.SUCCESS)

            # Save configuration
            self.save_config(task_dir)

            # Set up progress tracking
            # Preparation, Data Loading, Cache Check, Analysis, Visualizations, Finalization
            total_steps = 5 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            current_steps = 0

            # Step 1: Preparation
            if progress_tracker:
                progress_tracker.total = total_steps  # Define total steps for tracking
                progress_tracker.update(
                    current_steps,
                    {"step": "Preparation", "operation": self.operation_name},
                )

            # Step 2: Data Loading
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Data Loading"})

            # Load data
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                error_message = "Failed to load input data"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )
        except Exception as e:
            error_message = f"Error loading data: {str(e)}"
            self.logger.error(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message, exception=e
            )

        # Step 3: Check Cache (if enabled and not forced to recalculate)
        if self.use_cache and not self.force_recalculation:
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Checking Cache"})

            logger.info("Checking operation cache...")
            cache_result = self._check_cache(df, dataset_name, **kwargs)

            if cache_result:
                self.logger.info("Cache hit! Using cached results.")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(total_steps, {"step": "Complete (cached)"})

                # Report cache hit to reporter
                if reporter:
                    reporter.add_operation(
                        f"Date field analysis for '{self.field_name}' (from cache)",
                        details={"cached": True},
                    )
                return cache_result

        try:

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame",
                )

            # Add operation to reporter
            if reporter:
                reporter.add_operation(
                    f"Analyzing currency field: {self.field_name}",
                    details={
                        "field_name": self.field_name,
                        "locale": self.locale,
                        "bins": self.bins,
                        "detect_outliers": self.detect_outliers,
                        "test_normality": self.test_normality,
                        "operation_type": "currency_analysis",
                    },
                )

            # Step 4: Analysis
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "K-Anonymity Analysis"})

            # Execute the analyzer
            analysis_results = self.analyzer.analyze(
                df=df,
                field_name=self.field_name,
                locale=self.locale,
                bins=self.bins,
                detect_outliers=self.detect_outliers,
                test_normality=self.test_normality,
                chunk_size=self.chunk_size,
                use_dask=self.use_dask,
                use_vectorization=self.use_vectorization,
                parallel_processes=self.parallel_processes,
                progress_tracker=progress_tracker,
                task_logger=self.logger,
            )

            # Check for errors
            if "error" in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results["error"],
                )

            # Save analysis results to JSON
            stats_filename = f"{self.field_name}_stats.json"
            stats_path = output_dir / stats_filename

            encryption_mode = get_encryption_mode(analysis_results, **kwargs)
            write_json(
                analysis_results,
                stats_path,
                encryption_key=self.encryption_key,
                encryption_mode=encryption_mode,
            )
            result.add_artifact(
                "json",
                stats_path,
                f"{self.field_name} statistical analysis",
                category=Constants.Artifact_Category_Output,
            )

            # Add to reporter
            reporter.add_artifact(
                "json", str(stats_path), f"{self.field_name} currency analysis"
            )

            # Generate visualizations if requested
            if self.generate_visualization:

                # Step 5: Creating Visualizations
                if progress_tracker:
                    current_steps += 1
                    progress_tracker.update(
                        current_steps, {"step": "Creating Visualizations"}
                    )

                self._handle_visualizations(
                    analysis_results=analysis_results,
                    vis_dir=visualizations_dir,
                    timestamp=operation_timestamp,
                    result=result,
                    reporter=reporter,
                    vis_theme=self.visualization_theme,
                    vis_backend=self.visualization_backend,
                    vis_strict=self.visualization_strict,
                    vis_timeout=self.visualization_timeout,
                    progress_tracker=progress_tracker,
                    use_encryption=self.use_encryption,
                    encryption_key=self.encryption_key,
                )

            # Save sample records with original currency values
            self._save_sample_records(
                df,
                analysis_results,
                dictionaries_dir,
                result,
                reporter,
                self.encryption_key,
            )

            # Add metrics to the result
            self._add_metrics_to_result(analysis_results, result)

            # Step 6: Finalization
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(
                    current_steps, {"step": "Operation complete", "status": "success"}
                )

            # Add final operation status to reporter
            reporter.add_operation(
                f"Analysis of currency field {self.field_name} completed",
                details={
                    "valid_values": analysis_results.get("valid_count", 0),
                    "null_percentage": analysis_results.get("null_percentage", 0),
                    "multi_currency": analysis_results.get("multi_currency", False),
                    "currencies_detected": len(
                        analysis_results.get("currency_counts", {})
                    ),
                },
            )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        artifacts=result.artifacts,
                        original_df=df,
                        metrics=result.metrics,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            return result

        except Exception as e:
            self.logger.exception(
                f"Error in currency operation for {self.field_name}: {e}"
            )

            # Add error to reporter
            reporter.add_operation(
                f"Error analyzing currency field {self.field_name}",
                status="error",
                details={"error": str(e)},
            )

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing currency field {self.field_name}: {str(e)}",
                exception=e,
            )

    def _generate_visualizations(
        self,
        analysis_results: Dict[str, Any],
        vis_dir: Path,
        timestamp: Optional[str],
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        **kwargs,
    ):
        """
        Generate visualizations for the currency field analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        timestamp : Optional[str]
            Timestamp for naming visualization files
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors"""
        stats_dict = analysis_results.get("stats", {})

        # Handle both old and new key names for backward compatibility
        valid_values = analysis_results.get("valid_values")
        if valid_values is None:
            valid_values = analysis_results.get("valid_value")

        if valid_values is None:
            self.logger.warning(
                f"No valid currency values found in analysis results for {self.field_name}"
            )
            return

        if len(valid_values) == 0:
            self.logger.warning(
                f"No valid currency values for visualization in {self.field_name}"
            )
            return

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate distribution histogram
        if "histogram" in stats_dict:
            try:
                hist_filename = f"{self.field_name}_distribution_{timestamp}.png"
                hist_path = vis_dir / hist_filename

                # Create histogram using the visualization module
                min_value = stats_dict.get("min")
                max_value = stats_dict.get("max")

                title = f"Distribution of {self.field_name}"
                if min_value is not None and max_value is not None:
                    title += f" (min: {min_value:.2f}, max: {max_value:.2f})"

                # Add currency information if available
                currencies = analysis_results.get("currency_counts", {})
                if currencies:
                    main_currency = (
                        max(currencies.items(), key=lambda x: x[1])[0]
                        if currencies
                        else "Unknown"
                    )
                    if main_currency != "UNKNOWN":
                        title += f" ({main_currency})"

                # Create the histogram
                hist_result = create_histogram(
                    data=valid_values,
                    output_path=str(hist_path),
                    title=title,
                    x_label=f"{self.field_name} Value",
                    y_label="Frequency",
                    bins=self.bins,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )

                if not hist_result.startswith("Error"):
                    result.add_artifact(
                        "png",
                        hist_path,
                        f"{self.field_name} distribution histogram",
                        category=Constants.Artifact_Category_Visualization,
                    )
                    reporter.add_artifact(
                        "png",
                        str(hist_path),
                        f"{self.field_name} distribution histogram",
                    )
            except Exception as e:
                self.logger.warning(
                    f"Error creating histogram for {self.field_name}: {e}"
                )

        # Generate boxplot
        if len(valid_values) >= 5:
            try:
                boxplot_filename = f"{self.field_name}_boxplot_{timestamp}.png"
                boxplot_path = vis_dir / boxplot_filename

                # Create boxplot using the visualization module
                boxplot_result = create_boxplot(
                    data={self.field_name: valid_values.tolist()},
                    output_path=str(boxplot_path),
                    title=f"Boxplot of {self.field_name}",
                    y_label=self.field_name,
                    show_points=True,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )

                if not boxplot_result.startswith("Error"):
                    result.add_artifact(
                        "png",
                        boxplot_path,
                        f"{self.field_name} boxplot",
                        category=Constants.Artifact_Category_Visualization,
                    )
                    reporter.add_artifact(
                        "png", str(boxplot_path), f"{self.field_name} boxplot"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Error creating boxplot for {self.field_name}: {e}"
                )

        # Generate Q-Q plot for normality if requested
        if (
            self.test_normality
            and "normality" in stats_dict
            and len(valid_values) >= 10
        ):
            try:
                qq_filename = f"{self.field_name}_qq_plot_{timestamp}.png"
                qq_path = vis_dir / qq_filename

                # Generate synthetic normal data for comparison
                np.random.seed(42)  # For reproducibility
                normal_data = np.random.normal(
                    loc=valid_values.mean(),
                    scale=valid_values.std(),
                    size=len(valid_values),
                )
                normal_data.sort()

                # Sort the actual data for Q-Q plot
                sorted_data = valid_values.sort_values()

                # Create scatter plot comparing theoretical quantiles to actual data
                normality_info = stats_dict["normality"]
                is_normal = normality_info.get("is_normal", False)
                p_value = normality_info.get("shapiro", {}).get("p_value", None)

                title = f"Q-Q Plot for {self.field_name}"
                if p_value is not None:
                    title += f" (Shapiro p-value: {p_value:.4f})"

                # Create the Q-Q plot
                qq_result = create_correlation_pair_plot(
                    x_data=normal_data,
                    y_data=sorted_data,
                    output_path=str(qq_path),
                    title=title,
                    x_label="Theoretical Quantiles",
                    y_label="Sample Quantiles",
                    add_trendline=True,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )

                if not qq_result.startswith("Error"):
                    result.add_artifact(
                        "png",
                        qq_path,
                        f"{self.field_name} Q-Q plot (normality test)",
                        category=Constants.Artifact_Category_Visualization,
                    )
                    reporter.add_artifact(
                        "png", str(qq_path), f"{self.field_name} Q-Q plot"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Error creating Q-Q plot for {self.field_name}: {e}"
                )

    def _save_sample_records(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        analysis_results: Dict[str, Any],
        dict_dir: Path,
        result: OperationResult,
        reporter: Any,
        encryption_key: Optional[str] = None,
    ):
        """
        Save sample records with original currency values.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        dict_dir : Path
            Directory to save sample records
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        try:
            # Create a sample of records containing the field
            # Prioritize getting diverse examples including:
            # - Records with different currencies
            # - Records with high/low values
            # - Records with missing/invalid values

            # Get field values
            field_values = df[self.field_name]
            total_records = len(field_values)

            # Determine how many samples to take
            sample_size = min(100, total_records)
            if sample_size == 0:
                return

            # Create a sample index
            indices = []

            # Start with any outliers if available
            stats_dict = analysis_results.get("stats", {})
            outliers = stats_dict.get("outliers", {})

            if outliers and "indices" in outliers and outliers["indices"]:
                # Take up to 20% of the sample from outliers
                outlier_indices = outliers["indices"]
                outlier_sample_size = min(len(outlier_indices), int(sample_size * 0.2))
                if outlier_sample_size > 0:
                    indices.extend(
                        np.random.choice(
                            outlier_indices, size=outlier_sample_size, replace=False
                        )
                    )

            # Handle multi-currency examples if available
            if analysis_results.get("multi_currency", False):
                # Handle both old and new key names for backward compatibility
                valid_values_data = analysis_results.get("valid_values")
                if valid_values_data is None:
                    valid_values_data = analysis_results.get("valid_value")

                if valid_values_data is not None:
                    currencies = getattr(
                        valid_values_data, "currencies", [None] * len(valid_values_data)
                    )

                    # Get indices for each currency
                    for currency in set(currencies):
                        if currency:
                            currency_indices = [
                                i for i, c in enumerate(currencies) if c == currency
                            ]
                            if currency_indices:
                                # Take up to 10% of the sample from each currency
                                currency_sample_size = min(
                                    len(currency_indices), int(sample_size * 0.1)
                                )
                                if currency_sample_size > 0:
                                    indices.extend(
                                        np.random.choice(
                                            currency_indices,
                                            size=currency_sample_size,
                                            replace=False,
                                        )
                                    )

            # Add some null values if available
            null_indices = df[df[self.field_name].isna()].index.tolist()
            if null_indices:
                # Take up to 10% of the sample from null values
                null_sample_size = min(len(null_indices), int(sample_size * 0.1))
                if null_sample_size > 0:
                    indices.extend(
                        np.random.choice(
                            null_indices, size=null_sample_size, replace=False
                        )
                    )

            # Fill the rest with random samples
            if len(indices) < sample_size:
                remaining_indices = list(set(range(total_records)) - set(indices))
                remaining_sample_size = sample_size - len(indices)

                if remaining_indices and remaining_sample_size > 0:
                    indices.extend(
                        np.random.choice(
                            remaining_indices,
                            size=min(len(remaining_indices), remaining_sample_size),
                            replace=False,
                        )
                    )

            # Deduplicate and sort indices
            indices = sorted(set(indices))

            # Create sample DataFrame
            if df.index.name:
                id_field = df.index.name
            else:
                id_field = "index"

            sample_df = df.loc[indices, [self.field_name]].copy()
            sample_df = sample_df.reset_index()

            # Save to CSV
            sample_filename = f"{self.field_name}_sample.csv"
            sample_path = dict_dir / sample_filename

            write_dataframe_to_csv(
                sample_df, sample_path, encryption_key=encryption_key
            )

            # Add to result
            result.add_artifact(
                "csv",
                sample_path,
                f"{self.field_name} sample records",
                category=Constants.Artifact_Category_Dictionary,
            )
            reporter.add_artifact(
                "csv", str(sample_path), f"{self.field_name} sample records"
            )

        except Exception as e:
            self.logger.warning(
                f"Error saving sample records for {self.field_name}: {e}"
            )

    def _add_metrics_to_result(
        self, analysis_results: Dict[str, Any], result: OperationResult
    ):
        """
        Add metrics from the analysis results to the operation result.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        result : OperationResult
            Operation result to add metrics to
        """
        # Add basic metrics
        result.add_metric("total_rows", analysis_results.get("total_rows", 0))
        result.add_metric("valid_count", analysis_results.get("valid_count", 0))
        result.add_metric("null_count", analysis_results.get("null_count", 0))
        result.add_metric("invalid_count", analysis_results.get("invalid_count", 0))
        result.add_metric(
            "null_percentage", analysis_results.get("null_percentage", 0.0)
        )
        result.add_metric(
            "invalid_percentage", analysis_results.get("invalid_percentage", 0.0)
        )
        result.add_metric(
            "multi_currency", analysis_results.get("multi_currency", False)
        )
        result.add_metric(
            "currency_count", len(analysis_results.get("currency_counts", {}))
        )

        # Add statistics metrics
        stats_dict = analysis_results.get("stats", {})

        if stats_dict:
            # Add basic statistics
            result.add_nested_metric("statistics", "min", stats_dict.get("min"))
            result.add_nested_metric("statistics", "max", stats_dict.get("max"))
            result.add_nested_metric("statistics", "mean", stats_dict.get("mean"))
            result.add_nested_metric("statistics", "median", stats_dict.get("median"))
            result.add_nested_metric("statistics", "std", stats_dict.get("std"))

            # Add distribution statistics
            result.add_nested_metric(
                "statistics", "skewness", stats_dict.get("skewness")
            )
            result.add_nested_metric(
                "statistics", "kurtosis", stats_dict.get("kurtosis")
            )

            # Add zero and negative counts
            result.add_nested_metric(
                "statistics", "zero_count", stats_dict.get("zero_count", 0)
            )
            result.add_nested_metric(
                "statistics", "zero_percentage", stats_dict.get("zero_percentage", 0.0)
            )
            result.add_nested_metric(
                "statistics", "negative_count", stats_dict.get("negative_count", 0)
            )
            result.add_nested_metric(
                "statistics",
                "negative_percentage",
                stats_dict.get("negative_percentage", 0.0),
            )

            # Add outlier metrics
            outliers = stats_dict.get("outliers", {})
            if outliers:
                result.add_nested_metric("outliers", "count", outliers.get("count", 0))
                result.add_nested_metric(
                    "outliers", "percentage", outliers.get("percentage", 0.0)
                )
                result.add_nested_metric(
                    "outliers", "lower_bound", outliers.get("lower_bound")
                )
                result.add_nested_metric(
                    "outliers", "upper_bound", outliers.get("upper_bound")
                )

            # Add normality metrics
            normality = stats_dict.get("normality", {})
            if normality:
                result.add_nested_metric(
                    "normality", "is_normal", normality.get("is_normal", False)
                )
                shapiro = normality.get("shapiro", {})
                if shapiro:
                    result.add_nested_metric(
                        "normality", "shapiro_stat", shapiro.get("statistic")
                    )
                    result.add_nested_metric(
                        "normality", "shapiro_p_value", shapiro.get("p_value")
                    )

        # Add currency metrics
        currency_counts = analysis_results.get("currency_counts", {})
        if currency_counts:
            for currency, count in currency_counts.items():
                result.add_nested_metric("currencies", currency, count)

    def _check_cache(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        task_dir: Path,
        reporter: Any,
        **kwargs,
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame for the operation
        task_dir : Path
            Task directory
        reporter : Any
            The reporter to log artifacts to

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import OperationCache

            operation_cache_dir = OperationCache(cache_dir=task_dir / "cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache_dir.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        cached_result.add_metric(key, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        artifact_type = artifact.get("artifact_type", "")
                        artifact_path = artifact.get("path", "")
                        artifact_name = artifact.get("description", "")
                        artifact_category = artifact.get("category", "output")
                        cached_result.add_artifact(
                            artifact_type,
                            artifact_path,
                            artifact_name,
                            artifact_category,
                        )

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        artifacts: List[OperationArtifact],
        metrics: Dict[str, Any],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            Original input data
        metrics : dict
            The metrics of operation
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache or (not artifacts and not metrics):
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            artifacts_for_cache = [artifact.to_dict() for artifact in artifacts]

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "artifacts": artifacts_for_cache,
                "metrics": metrics,
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(self, df: Union[pd.DataFrame, dd.DataFrame]) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get operation parameters
        parameters = self._get_operation_parameters()

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(df)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.operation_name,
            parameters=parameters,
            data_hash=data_hash,
        )

    def _get_operation_parameters(self) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "locale": self.locale,
            "bins": self.bins,
            "detect_outliers": self.detect_outliers,
            "test_normality": self.test_normality,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "chunk_size": self.chunk_size,
            "visualization_theme": self.visualization_theme,
            "visualization_backend": self.visualization_backend,
            "visualization_strict": self.visualization_strict,
            "visualization_timeout": self.visualization_timeout,
            "use_cache": self.use_cache,
            "use_encryption": self.use_encryption,
            "encryption_mode": self.encryption_mode,
            "encryption_key": self.encryption_key,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

    def _generate_data_hash(self, df: Union[pd.DataFrame, dd.DataFrame]) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Hash string representing the data
        """
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _handle_visualizations(
        self,
        analysis_results: Dict[str, Any],
        vis_dir: Path,
        timestamp: Optional[str],
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        timestamp : Optional[str]
            Timestamp for naming visualization files
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        logger.info(
            f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s"
        )

        try:
            import threading
            import contextvars

            visualization_paths = {}
            visualization_error = None

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                logger.info(
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
                    except Exception as ctx_e:
                        logger.warning(f"[DIAG] Could not inspect context: {ctx_e}")

                    # Generate visualizations with visualization context parameters
                    logger.info(f"[DIAG] Calling _generate_visualizations...")
                    # Create child progress tracker for visualization if available
                    total_steps = 3  # prepare data, create viz, save
                    viz_progress = None
                    if progress_tracker and hasattr(progress_tracker, "create_subtask"):
                        try:
                            viz_progress = progress_tracker.create_subtask(
                                total=total_steps,
                                description="Generating visualizations",
                                unit="steps",
                            )
                        except Exception as e:
                            logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations
                    self._generate_visualizations(
                        analysis_results,
                        vis_dir,
                        timestamp,
                        result,
                        reporter,
                        vis_theme,
                        vis_backend,
                        vis_strict,
                        **kwargs,
                    )

                    # Close visualization progress tracker
                    if viz_progress:
                        try:
                            viz_progress.close()
                        except:
                            pass

                    elapsed = time.time() - start_time
                    logger.info(
                        f"[DIAG] Visualization completed in {elapsed:.2f}s, generated {len(visualization_paths)} files"
                    )
                except Exception as e:
                    elapsed = time.time() - start_time
                    visualization_error = e
                    logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
                    logger.error(f"[DIAG] Stack trace:", exc_info=True)

            # Copy context for the thread
            logger.info(f"[DIAG] Preparing to launch visualization thread...")
            ctx = contextvars.copy_context()

            # Create thread with context
            viz_thread = threading.Thread(
                target=ctx.run,
                args=(generate_viz_with_diagnostics,),
                name=f"VizThread-",
                daemon=False,  # Changed from True to ensure proper cleanup
            )

            logger.info(
                f"[DIAG] Starting visualization thread with timeout={vis_timeout}s"
            )
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = {}
            elif visualization_error:
                logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                logger.info(
                    f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}"
                )
        except Exception as e:
            logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = {}

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization",
                )

        return visualization_paths
