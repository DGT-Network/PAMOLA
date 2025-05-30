"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Anonymization Metric Utilities
Description: Common metric utilities for anonymization operations
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides functions for collecting and calculating metrics
related to anonymization operations, particularly for assessing the
effectiveness and utility of generalization techniques.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

# Import pamola_core utilities
from pamola_core.utils.io import write_json, ensure_directory
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import create_histogram, create_bar_plot

logger = logging.getLogger(__name__)


def calculate_basic_numeric_metrics(original_series: pd.Series,
                                    anonymized_series: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic metrics comparing original and anonymized numeric data.

    Parameters:
    -----------
    original_series : pd.Series
        The original numeric data
    anonymized_series : pd.Series
        The anonymized numeric data

    Returns:
    --------
    Dict[str, Any]
        Dictionary with basic metrics
    """
    try:
        # Filter out nulls for calculations
        orig_non_null = original_series.dropna()
        anon_non_null = anonymized_series.dropna()

        # Handle case where anonymized values might be strings (e.g., in binning)
        try:
            anon_numeric = pd.to_numeric(anon_non_null, errors='coerce')
            anon_is_numeric = not anon_numeric.isna().all()
        except Exception:
            anon_is_numeric = False
            anon_numeric = None  # Add a default definition

        # Create metrics dictionary with explicit typing
        metrics: Dict[str, Any] = {
            "total_records": len(original_series),
            "null_count_original": int(original_series.isna().sum()),
            "null_count_anonymized": int(anonymized_series.isna().sum()),
            "unique_values_original": int(orig_non_null.nunique()),
            "unique_values_anonymized": int(anon_non_null.nunique())
        }

        # Add additional metrics if anonymized data is still numeric
        if anon_is_numeric and anon_numeric is not None:
            try:
                # Compare statistical properties
                metrics.update({
                    "mean_original": float(orig_non_null.mean()),
                    "mean_anonymized": float(anon_numeric.mean()),
                    "std_original": float(orig_non_null.std()),
                    "std_anonymized": float(anon_numeric.std()),
                    "min_original": float(orig_non_null.min()),
                    "min_anonymized": float(anon_numeric.min()),
                    "max_original": float(orig_non_null.max()),
                    "max_anonymized": float(anon_numeric.max()),
                    "median_original": float(orig_non_null.median()),
                    "median_anonymized": float(anon_numeric.median())
                })

                # Calculate mean absolute difference if possible
                if orig_non_null.index.equals(anon_numeric.index):
                    metrics["mean_absolute_difference"] = float(
                        (orig_non_null - anon_numeric).abs().mean()
                    )
            except Exception as e:
                logger.warning(f"Could not compute all numeric comparison metrics: {e}")

        # Calculate generalization ratio (reduction in unique values)
        unique_values_original = metrics["unique_values_original"]
        if unique_values_original > 0:
            unique_values_anonymized = metrics["unique_values_anonymized"]
            metrics["generalization_ratio"] = 1.0 - (unique_values_anonymized / unique_values_original)
        else:
            metrics["generalization_ratio"] = 0.0

        return metrics
    except Exception as e:
        logger.error(f"Error calculating basic numeric metrics: {e}")
        # Return minimal metrics on error
        return {
            "total_records": len(original_series),
            "error": str(e)
        }


def calculate_generalization_metrics(original_series: pd.Series,
                                     anonymized_series: pd.Series,
                                     strategy: str,
                                     strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate generalization-specific metrics based on strategy.

    Parameters:
    -----------
    original_series : pd.Series
        The original data
    anonymized_series : pd.Series
        The anonymized data
    strategy : str
        The generalization strategy used
    strategy_params : Dict[str, Any]
        Parameters used for the strategy

    Returns:
    --------
    Dict[str, Any]
        Dictionary with generalization metrics
    """
    try:
        # Using regular dict instead of TypedDict
        metrics = {"generalization_strategy": strategy, "strategy_parameters": strategy_params}

        # Basic metrics

        # Strategy-specific metrics
        if strategy == "binning":
            bin_count = strategy_params.get("bin_count", 0)
            metrics["bin_count"] = bin_count
            metrics["average_bin_size"] = len(original_series.dropna()) / bin_count if bin_count > 0 else 0

            # Calculate distribution of values per bin
            if anonymized_series.dtype == 'category' or isinstance(anonymized_series.dtype, pd.CategoricalDtype):
                # Count records per bin
                bin_distribution = anonymized_series.value_counts().to_dict()
                # Filter out nulls
                bin_distribution = {k: v for k, v in bin_distribution.items() if pd.notna(k)}
                # Convert keys to strings for JSON serialization
                metrics["bin_distribution"] = {str(k): int(v) for k, v in bin_distribution.items()}

        elif strategy == "rounding":
            metrics["rounding_precision"] = strategy_params.get("precision", 0)

            # Estimate information loss
            orig_non_null = original_series.dropna()
            if len(orig_non_null) > 0:
                # Calculate number of significant digits lost
                orig_std = orig_non_null.std()
                precision = strategy_params.get("precision", 0)  # Get precision directly from strategy_params

                if orig_std > 0:
                    # Estimate information loss based on standard deviation and precision
                    if precision >= 0:
                        # For decimal places
                        metrics["estimated_information_loss"] = min(1.0, max(0.0,
                                                                             1.0 - (10.0 ** -precision / orig_std)
                                                                             ))
                    else:
                        # For rounding to 10s, 100s, etc.
                        metrics["estimated_information_loss"] = min(1.0, max(0.0,
                                                                             1.0 - (orig_std / (10.0 ** abs(precision)))
                                                                             ))

        elif strategy == "range":
            range_min, range_max = strategy_params.get("range_limits", (0, 0))
            metrics["range_min"] = range_min
            metrics["range_max"] = range_max
            metrics["range_size"] = range_max - range_min

            # Calculate distribution by range
            range_distribution = anonymized_series.value_counts().to_dict()
            # Convert keys to strings for JSON serialization
            metrics["range_distribution"] = {str(k): int(v) for k, v in range_distribution.items()}

        return metrics
    except Exception as e:
        logger.error(f"Error calculating generalization metrics: {e}")
        # Return minimal metrics on error
        return {
            "generalization_strategy": strategy,
            "error": str(e)
        }


def calculate_performance_metrics(start_time: float, end_time: float, records_processed: int) -> Dict[str, Any]:
    """
    Calculate performance metrics for the operation.

    Parameters:
    -----------
    start_time : float
        Start time of the operation (from time.time())
    end_time : float
        End time of the operation (from time.time())
    records_processed : int
        Number of records processed

    Returns:
    --------
    Dict[str, Any]
        Dictionary with performance metrics
    """
    try:
        execution_time = end_time - start_time if end_time and start_time else 0

        metrics = {
            "execution_time_seconds": execution_time,
            "records_processed": records_processed,
            "records_per_second": records_processed / execution_time if execution_time > 0 else 0
        }

        return metrics
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        # Return minimal metrics on error
        return {
            "execution_time_seconds": 0,
            "records_processed": records_processed,
            "error": str(e)
        }


def save_metrics_json(metrics: Dict[str, Any],
                      task_dir: Path,
                      operation_name: str,
                      field_name: str,
                      writer: Optional[DataWriter] = None,
                      progress_tracker: Optional[ProgressTracker] = None,
                      encrypt_output: bool = False) -> Path:
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
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for the operation
    encrypt_output : bool, optional
        Whether to encrypt output file (default: False)

    Returns:
    --------
    Path
        Path to the saved metrics file
    """
    # Update progress if provided
    if progress_tracker:
        progress_tracker.update(0, {"step": "Saving metrics"})

    # Add timestamp and metadata
    metrics_with_metadata = metrics.copy()  # Create a copy to avoid modifying the input dictionary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_with_metadata.update({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation_name,
        "field": field_name
    })

    # Create filename base with meaningful information
    base_filename = f"{field_name}_{operation_name}_metrics"

    # Create complete filename with timestamp
    filename = f"{base_filename}_{timestamp}.json"

    # Use DataWriter if available
    if writer:
        try:
            encryption_key = None
            if encrypt_output:
                encryption_key = f"{field_name}_{timestamp}_key"

            # Use DataWriter to save metrics
            metrics_result = writer.write_metrics(
                metrics=metrics_with_metadata,
                name=base_filename,
                timestamp_in_name=True,
                encryption_key=encryption_key
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
            encryption_key=None if not encrypt_output else f"{field_name}_{timestamp}_key"
        )

        logger.info(f"Metrics saved to {json_path}")
        return json_path

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        # Return original path even if save failed
        return task_dir / filename


def get_distribution_data(series: pd.Series, bins: int = 10) -> Dict[str, Any]:
    """
    Get distribution data for visualization.

    Parameters:
    -----------
    series : pd.Series
        Data series to analyze
    bins : int, optional
        Number of bins for histogram (default: 10)

    Returns:
    --------
    Dict[str, Any]
        Distribution data for visualization
    """
    try:
        # Handle different types of data
        if pd.api.types.is_numeric_dtype(series):
            # For numeric data, create histogram
            try:
                # Drop nulls for the histogram
                non_null_series = series.dropna()

                # Check if we have enough data for a histogram
                if len(non_null_series) < 2:
                    logger.warning("Not enough non-null data points for histogram, falling back to value counts")
                    return get_categorical_distribution(series)

                hist, bin_edges = np.histogram(non_null_series, bins=bins)

                # Convert to lists for JSON serialization
                hist_list = [int(x) for x in hist]
                bin_edges_list = [float(x) for x in bin_edges]

                # Calculate basic statistics for the data
                stats = {
                    "min": float(non_null_series.min()),
                    "max": float(non_null_series.max()),
                    "mean": float(non_null_series.mean()),
                    "std": float(non_null_series.std()),
                    "count": len(non_null_series),
                    "null_count": int(series.isnull().sum())
                }

                return {
                    "type": "histogram",
                    "counts": hist_list,
                    "bin_edges": bin_edges_list,
                    **stats
                }
            except Exception as e:
                # Fallback to value counts if histogram fails
                logger.warning(f"Could not create histogram, falling back to value counts: {e}")
                return get_categorical_distribution(series)
        else:
            # For categorical/string data, use value counts
            return get_categorical_distribution(series)

    except Exception as e:
        logger.error(f"Error getting distribution data: {e}")
        # Return minimal data on error
        return {
            "type": "error",
            "error": str(e),
            "count": len(series),
            "null_count": int(series.isnull().sum())
        }


def get_categorical_distribution(series: pd.Series, max_categories: int = 20) -> Dict[str, Any]:
    """
    Get distribution data for categorical or string data.

    Parameters:
    -----------
    series : pd.Series
        Data series to analyze
    max_categories : int, optional
        Maximum number of categories to include (default: 20)

    Returns:
    --------
    Dict[str, Any]
        Distribution data for visualization
    """
    try:
        # Get value counts
        value_counts = series.value_counts()

        # Limit to max_categories
        if len(value_counts) > max_categories:
            # Keep top categories and group others
            top_categories = value_counts.iloc[:max_categories - 1]
            other_count = int(value_counts.iloc[max_categories - 1:].sum())

            # Create new Series with "Other" category
            categories = top_categories.copy()
            categories["Other"] = other_count
        else:
            categories = value_counts

        # Convert to dict for JSON serialization
        categories_dict = {str(k): int(v) for k, v in categories.items()}

        return {
            "type": "categorical",
            "categories": categories_dict,
            "count": len(series),
            "unique_count": series.nunique(),
            "null_count": int(series.isnull().sum())
        }
    except Exception as e:
        logger.error(f"Error getting categorical distribution: {e}")
        # Return minimal data on error
        return {
            "type": "error",
            "error": str(e),
            "count": len(series),
            "null_count": int(series.isnull().sum())
        }


def create_distribution_visualization(original_data: pd.Series,
                                      anonymized_data: pd.Series,
                                      task_dir: Path,
                                      field_name: str,
                                      operation_name: str,
                                      writer: Optional[DataWriter] = None,
                                      progress_tracker: Optional[ProgressTracker] = None) -> Union[
    Dict[str, Path], None]:
    """
    Create distribution visualizations comparing original and anonymized data.

    Parameters:
    -----------
    original_data : pd.Series
        Original data before anonymization
    anonymized_data : pd.Series
        Anonymized data after processing
    task_dir : Path
        Task directory for saving visualizations
    field_name : str
        Name of the field being analyzed
    operation_name : str
        Name of the anonymization operation
    writer : Optional[DataWriter]
        DataWriter instance to use for saving
    progress_tracker : Optional[ProgressTracker]
        Progress tracker for monitoring the operation

    Returns:
    --------
    Union[Dict[str, Path], None]
        Dictionary mapping visualization types to file paths, or None if no visualizations were created
    """
    # Update progress if provided
    if progress_tracker:
        progress_tracker.update(0, {"step": "Creating visualizations"})

    visualization_paths: Dict[str, Path] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean data (drop nulls) for visualization
    original_clean = original_data.dropna()

    # Initialize variables before the try block
    anonymized_numeric = None
    has_numeric_data = False

    try:
        # Determine if data is numeric
        try:
            anonymized_numeric = pd.to_numeric(anonymized_data.dropna(), errors='coerce')
            has_numeric_data = not anonymized_numeric.isna().all()
        except Exception:
            has_numeric_data = False
            logger.debug("Could not convert anonymized data to numeric")

        # Update progress
        if progress_tracker:
            progress_tracker.update(0, {"step": "Processing data types"})

        # Determine the adaptive number of bins
        n_bins = min(20, max(5, int(np.sqrt(len(original_clean)))))

        # Create visualizations based on data type
        if has_numeric_data and pd.api.types.is_numeric_dtype(original_data):
            # For numeric data, create histogram
            anonymized_clean = anonymized_numeric.dropna()

            # Define histogram paths
            hist_filename = f"{field_name}_{operation_name}_histogram_{timestamp}.png"
            hist_path = task_dir / hist_filename

            # Create histogram
            create_histogram(
                data={
                    "Original": original_clean.tolist(),
                    "Anonymized": anonymized_clean.tolist()
                },
                output_path=str(hist_path),
                title=f"Distribution Comparison for {field_name}",
                x_label=field_name,
                y_label="Frequency",
                bins=n_bins,
                kde=True
            )

            # Register visualization with writer if available
            if writer:
                try:
                    writer.write_visualization(
                        figure=hist_path,  # Path to the saved image
                        name=f"{field_name}_histogram_comparison",
                        format="png"
                    )
                except Exception as e:
                    logger.warning(f"Error registering histogram with writer: {e}")

            visualization_paths["distribution_comparison"] = hist_path

        else:
            # For categorical data, create a bar plot
            max_categories = 10

            # Get value counts for original and anonymized data
            try:
                orig_counts = original_data.value_counts().head(max_categories)
                anon_counts = anonymized_data.value_counts().head(max_categories)

                # Define bar chart paths
                bar_filename = f"{field_name}_{operation_name}_categories_{timestamp}.png"
                bar_path = task_dir / bar_filename

                create_bar_plot(
                    data={
                        "Original": orig_counts.to_dict(),
                        "Anonymized": anon_counts.to_dict()
                    },
                    output_path=str(bar_path),
                    title=f"Category Comparison for {field_name}",
                    x_label="Category",
                    y_label="Count",
                    orientation="v",
                    sort_by="value",
                    max_items=max_categories
                )

                # Register visualization with writer if available
                if writer:
                    try:
                        writer.write_visualization(
                            figure=bar_path,  # Path to the saved image
                            name=f"{field_name}_category_comparison",
                            format="png"
                        )
                    except Exception as e:
                        logger.warning(f"Error registering bar chart with writer: {e}")

                visualization_paths["category_comparison"] = bar_path

            except Exception as e:
                logger.warning(f"Error creating bar plot: {e}")

        # Update progress
        if progress_tracker:
            progress_tracker.update(1, {"step": "Visualization complete"})

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        # Return empty dict if no visualizations were created

    finally:
        if progress_tracker:
            progress_tracker.update(0, {"step": "Visualization process completed"})

    return visualization_paths if visualization_paths else None


def generate_metrics_hash(metrics: Dict[str, Any]) -> str:
    """
    Generate a hash of the metrics for caching and comparison.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics to hash

    Returns:
    --------
    str
        Hash string of the metrics
    """
    try:
        # Filter out non-serializable values and timestamps
        filtered_metrics = {}
        for key, value in metrics.items():
            if key != "timestamp" and (isinstance(value, (int, float, str, bool)) or value is None):
                filtered_metrics[key] = value

        # Convert dictionary to JSON string and hash
        metrics_json = json.dumps(filtered_metrics, sort_keys=True)
        return hashlib.md5(metrics_json.encode()).hexdigest()

    except Exception as e:
        logger.error(f"Error generating metrics hash: {e}")
        # Return fallback hash
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()