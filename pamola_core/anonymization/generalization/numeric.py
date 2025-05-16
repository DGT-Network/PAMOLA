"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Numeric Generalization Operation
Description: Operation for generalizing numeric fields through binning, rounding, and range-based strategies
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides an operation for generalizing numeric fields to enhance privacy
while maintaining data utility. It implements various strategies:

1. Binning: Groups numeric values into discrete intervals (bins)
2. Rounding: Reduces precision by rounding to specified decimal places
3. Range-based: Maps values to custom ranges with special handling for outliers

Key features:
- Direct in-place DataFrame modification with both REPLACE and ENRICH modes
- Robust null value handling with configurable strategies (PRESERVE, EXCLUDE, ERROR)
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for distribution comparisons
- Chunked processing support for large datasets
- Graceful handling of already-processed non-numeric fields
- Memory-efficient operation with explicit cleanup for large datasets

Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.metric_utils import (
    calculate_basic_numeric_metrics,
    calculate_generalization_metrics,
    calculate_performance_metrics
)
from pamola_core.anonymization.commons.processing_utils import (
    numeric_generalization_binning,
    numeric_generalization_rounding,
    numeric_generalization_range,
    process_nulls,
    generate_output_field_name,
    process_in_chunks,
    process_dataframe_parallel
)
from pamola_core.anonymization.commons.validation_utils import (
    validate_numeric_field
)
from pamola_core.anonymization.commons.visualization_utils import (
    prepare_comparison_data,
    generate_visualization_filename,
    create_visualization_path,
    register_visualization_artifact,
    calculate_optimal_bins, sample_large_dataset
)
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import create_histogram, create_bar_plot
from pamola_core.utils.io import load_data_operation

# Configure module logger
logger = logging.getLogger(__name__)


class NumericGeneralizationConfig(OperationConfig):
    """Configuration for NumericGeneralizationOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "strategy": {"type": "string", "enum": ["binning", "rounding", "range"]},
            "bin_count": {"type": "integer", "minimum": 2},
            "precision": {"type": "integer"},
            "range_limits": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2
            },
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},
            "null_strategy": {"type": "string", "enum": ["PRESERVE", "EXCLUDE", "ERROR"]},
            "batch_size": {"type": "integer", "minimum": 1},
            "use_cache": {"type": "boolean"},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]}
        },
        "required": ["field_name", "strategy"],
        "allOf": [
            {
                "if": {"properties": {"strategy": {"const": "binning"}}},
                "then": {"required": ["bin_count"]}
            },
            {
                "if": {"properties": {"strategy": {"const": "rounding"}}},
                "then": {"required": ["precision"]}
            },
            {
                "if": {"properties": {"strategy": {"const": "range"}}},
                "then": {"required": ["range_limits"]}
            }
        ]
    }

@register(version="1.0.0")
class NumericGeneralizationOperation(AnonymizationOperation):
    """
    Operation for generalizing numeric data.

    This operation generalizes numeric fields using strategies like binning,
    rounding, or range-based generalization to reduce precision and improve
    anonymity while preserving analytical utility.
    """

    def __init__(self,
                 field_name: str,
                 strategy: str = "binning",  # "binning", "rounding", "range"
                 bin_count: int = 10,
                 precision: int = 0,  # For rounding strategy
                 range_limits: Optional[Tuple[float, float]] = None,  # For range strategy
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 column_prefix: str = "_",
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 description: str = ""):
        """
        Initialize numeric generalization operation.

        Parameters:
        -----------
        field_name : str
            Field to generalize
        strategy : str, optional
            Generalization strategy: "binning", "rounding", or "range" (default: "binning")
        bin_count : int, optional
            Number of bins for binning strategy (default: 10)
        precision : int, optional
            Decimal places to retain for rounding strategy (default: 0)
        range_limits : Tuple[float, float], optional
            (min, max) limits for range strategy (default: None)
        mode : str, optional
            "REPLACE" to modify the field in-place, or "ENRICH" to create a new field (default: "REPLACE")
        output_field_name : str, optional
            Name for the output field if mode is "ENRICH" (default: None)
        column_prefix : str, optional
            Prefix for new column if mode is "ENRICH" (default: "_")
        null_strategy : str, optional
            How to handle NULL values: "PRESERVE", "EXCLUDE", or "ERROR" (default: "PRESERVE")
        batch_size : int, optional
            Batch size for processing large datasets (default: 10000)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]], optional
            Encryption key for securing outputs (default: None)
        description : str, optional
            Operation description (default: "")
        """
        # Create configuration and validate parameters
        config = NumericGeneralizationConfig(
            field_name=field_name,
            strategy=strategy,
            bin_count=bin_count,
            precision=precision,
            range_limits=range_limits,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,
            null_strategy=null_strategy,
            batch_size=batch_size,
            use_cache=use_cache,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        # Use a default description if none provided
        if not description:
            description = f"Numeric generalization for field '{field_name}' using {strategy} strategy"

        # Initialize base class
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,
            null_strategy=null_strategy,
            batch_size=batch_size,
            use_cache=use_cache,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            description=description
        )

        # Store strategy parameters from validated config
        self.strategy = config.get("strategy").lower()
        self.bin_count = config.get("bin_count", 10)
        self.precision = config.get("precision", 0)
        self.range_limits = config.get("range_limits")
        self.version = "1.1.0"  # Semantic versioning

        # Temp storage for cleanup
        self._temp_data = None

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the operation with timing and error handling.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[ProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation including:
            - force_recalculation: bool - Skip cache check
            - parallel_processes: int - Number of parallel processes
            - encrypt_output: bool - Override encryption setting

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Initialize timing and result
            self.start_time = time.time()
            self.process_count = 0
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Create DataWriter for consistent file operations
            writer = DataWriter(task_dir=task_dir, logger=logger, progress_tracker=progress_tracker)

            # Save configuration to task directory
            self.save_config(task_dir)

            # Set up progress tracking
            total_steps = 5  # Data loading, validation, processing, metrics, visualization
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"step": "Starting numeric generalization", "field": self.field_name})

            dataset_name = kwargs.get('dataset_name', "main")
            # Check if we have a cached result using the operation_cache utility
            if self.use_cache and not kwargs.get("force_recalculation", False):
                # Generate cache key based on operation parameters
                cache_key = operation_cache.generate_cache_key(
                    operation_name=self.__class__.__name__,
                    parameters=self._get_cache_parameters(),
                    data_hash=self._generate_data_hash(load_data_operation(data_source, dataset_name)[self.field_name])
                )

                # Check for cached result
                cached_result = operation_cache.get_cache(
                    cache_key=cache_key,
                    operation_type=self.__class__.__name__
                )

                if cached_result:
                    logger.info(f"Using cached result for {self.field_name} generalization")

                    # Create result from cached data
                    cached_result_obj = OperationResult(status=OperationStatus.SUCCESS)

                    # Add metrics from cache
                    metrics = cached_result.get("metrics", {})
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            cached_result_obj.add_metric(key, value)

                    # Add cache info
                    cached_result_obj.add_metric("cached", True)
                    cached_result_obj.add_metric("cache_key", cache_key)

                    # Report to the reporter
                    if reporter:
                        reporter.add_operation(
                            f"Numeric generalization of {self.field_name} (cached)",
                            details={"strategy": self.strategy, "cached": True}
                        )

                    return cached_result_obj

            # Step 1: Load and validate data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Loading and validating data"})

            # Get DataFrame from data source        
            df = load_data_operation(data_source, dataset_name)

            # Validate the field exists
            if self.field_name not in df.columns:
                error_message = f"Field '{self.field_name}' not found in DataFrame"
                logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Get a copy of the original data for metrics calculation
            original_data = df[self.field_name].copy()

            # Step 2: Check if field is numeric
            if progress_tracker:
                progress_tracker.update(2, {"step": "Validating field type"})

            # We handle two cases:
            # 1. Field is numeric - proceed with validation and generalization
            # 2. Field is non-numeric - might be already processed, handle gracefully

            is_numeric = pd.api.types.is_numeric_dtype(df[self.field_name])

            if not is_numeric:
                logger.warning(f"Field '{self.field_name}' is not numeric, possibly already processed. "
                               f"Will copy to output field if in ENRICH mode.")

                # Determine output field
                output_field = generate_output_field_name(
                    self.field_name, self.mode, self.output_field_name, self.column_prefix
                )

                # If in ENRICH mode, copy values, if in REPLACE mode, leave as is
                if self.mode == "ENRICH":
                    df[output_field] = df[self.field_name]
                    logger.info(f"Copied non-numeric values to {output_field}")

                self.end_time = time.time()
                result.status = OperationStatus.SUCCESS

                # Add basic metrics about the field
                result.add_metric("field_name", self.field_name)
                result.add_metric("operation", "numeric_generalization")
                result.add_metric("strategy", self.strategy)
                result.add_metric("is_numeric", False)
                result.add_metric("execution_time", self.end_time - self.start_time)

                # Write output if needed
                output_result = writer.write_dataframe(
                    df=df,
                    name=f"{self.field_name}_generalized",
                    format="csv",
                    subdir="output",
                    timestamp_in_name=True,
                    encryption_key=self.encryption_key if self.use_encryption else None
                )

                # Add output artifact to result
                result.add_artifact(
                    artifact_type="csv",
                    path=output_result.path,
                    description=f"{self.field_name} generalized data",
                    category="output"
                )

                # Report to the reporter
                if reporter:
                    reporter.add_artifact(
                        "csv",
                        str(output_result.path),
                        f"{self.field_name} generalized data"
                    )

                return result

            # Now validate as numeric field
            try:
                validate_numeric_field(df, self.field_name, allow_null=(self.null_strategy != "ERROR"))
            except ValueError as e:
                if "null values" in str(e) and self.null_strategy == "ERROR":
                    return OperationResult(status=OperationStatus.ERROR, error_message=str(e))
                else:
                    raise

            # Step 3: Process the data
            if progress_tracker:
                progress_tracker.update(3, {"step": "Processing data"})

            # Determine output field based on mode
            output_field = generate_output_field_name(
                self.field_name, self.mode, self.output_field_name, self.column_prefix
            )

            # Process the data with the selected strategy
            try:
                if kwargs.get("parallel_processes", 1) > 1:
                    # Process in parallel
                    processed_df = process_dataframe_parallel(
                        df=df,
                        process_function=self.process_batch,
                        n_jobs=kwargs.get("parallel_processes", -1),
                        batch_size=self.batch_size,
                        progress_tracker=progress_tracker
                    )
                else:
                    # Process in chunks
                    processed_df = process_in_chunks(
                        df=df,
                        process_function=self.process_batch,
                        batch_size=self.batch_size,
                        progress_tracker=progress_tracker
                    )

                # Get the anonymized data for metrics calculation
                anonymized_data = processed_df[output_field]

            except Exception as e:
                logger.exception(f"Error processing data: {e}")
                return OperationResult(status=OperationStatus.ERROR, error_message=str(e))

            # Step 4: Calculate metrics
            if progress_tracker:
                progress_tracker.update(4, {"step": "Calculating metrics"})

            # Record end time after processing
            self.end_time = time.time()

            # Calculate metrics
            metrics = self._collect_metrics(original_data, anonymized_data)
            metrics.update(calculate_performance_metrics(self.start_time, self.end_time, self.process_count))

            # Generate standardized metrics filename
            metrics_filename = generate_visualization_filename(
                self.field_name,
                f"{self.__class__.__name__}_{self.strategy}",
                "metrics",
                extension="json"
            )

            # Write metrics file
            metrics_result = writer.write_metrics(
                metrics=metrics,
                name=metrics_filename.replace(".json", ""),  # writer appends .json
                timestamp_in_name=False,  # Already included in the filename
                encryption_key=self.encryption_key if self.use_encryption else None
            )

            # Add metrics to result
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    result.add_metric(key, value)

            # Add metrics artifact
            result.add_artifact(
                artifact_type="json",
                path=metrics_result.path,
                description=f"{self.field_name} generalization metrics",
                category="metrics"
            )

            # Report artifact
            if reporter:
                reporter.add_artifact(
                    "json",
                    str(metrics_result.path),
                    f"{self.field_name} generalization metrics"
                )

            # Step 5: Generate visualizations
            if progress_tracker:
                progress_tracker.update(5, {"step": "Generating visualizations"})

            # Generate visualizations
            visualization_paths = self._generate_visualizations(
                original_data, anonymized_data, task_dir, result, reporter
            )

            # Add visualization artifacts to result and reporter
            for viz_type, path in visualization_paths.items():
                result.add_artifact(
                    artifact_type="png",
                    path=path,
                    description=f"{self.field_name} {viz_type} visualization",
                    category="visualization"
                )

                if reporter:
                    reporter.add_artifact(
                        "png",
                        str(path),
                        f"{self.field_name} {viz_type} visualization"
                    )

            # Generate standardized output filename
            output_filename = generate_visualization_filename(
                self.field_name,
                f"{self.__class__.__name__}_{self.strategy}",
                "generalized",
                extension="csv"
            )

            # Save output data
            output_result = writer.write_dataframe(
                df=processed_df,
                name=output_filename.replace(".csv", ""),  # writer appends .csv
                format="csv",
                subdir="output",
                timestamp_in_name=False,  # Already included in the filename
                encryption_key=self.encryption_key if self.use_encryption else None
            )

            # Add output artifact to result
            result.add_artifact(
                artifact_type="csv",
                path=output_result.path,
                description=f"{self.field_name} generalized data",
                category="output"
            )

            # Report to the reporter
            if reporter:
                reporter.add_artifact(
                    "csv",
                    str(output_result.path),
                    f"{self.field_name} generalized data"
                )

            # Cache the result if enabled
            if self.use_cache:
                # Save to cache
                cache_data = {
                    "metrics": metrics,
                    "parameters": self._get_cache_parameters(),
                    "data_info": {
                        "original_length": len(original_data),
                        "anonymized_length": len(anonymized_data)
                    }
                }

                # Generate cache key
                cache_key = operation_cache.generate_cache_key(
                    operation_name=self.__class__.__name__,
                    parameters=self._get_cache_parameters(),
                    data_hash=self._generate_data_hash(original_data)
                )

                # Save to cache
                operation_cache.save_cache(
                    data=cache_data,
                    cache_key=cache_key,
                    operation_type=self.__class__.__name__,
                    metadata={"task_dir": str(task_dir)}
                )

            # Clean up memory
            self._cleanup_memory(processed_df, original_data, anonymized_data)

            # Set success status
            result.status = OperationStatus.SUCCESS

            # Report operation completion
            if reporter:
                reporter.add_operation(
                    f"Numeric generalization of {self.field_name} completed",
                    details={
                        "strategy": self.strategy,
                        "records_processed": self.process_count,
                        "execution_time": self.end_time - self.start_time,
                        "generalization_ratio": metrics.get("generalization_ratio", 0)
                    }
                )

            return result

        except Exception as e:
            # Handle unexpected errors
            error_message = f"Error in numeric generalization operation: {str(e)}"
            logger.exception(error_message)
            return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to generalize numeric values.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch with generalized values
        """
        # Check if the field exists
        if self.field_name not in batch.columns:
            raise ValueError(f"Field {self.field_name} not found in DataFrame")

        # Handle the case where the field might already be processed (non-numeric)
        field_data = batch[self.field_name]

        # Get the field value series - we'll need a copy for processing
        field_values = field_data.copy()

        # Determine output field based on mode
        output_field = generate_output_field_name(
            self.field_name, self.mode, self.output_field_name, self.column_prefix
        )

        # Check if field appears to be numeric
        if not pd.api.types.is_numeric_dtype(field_data):
            # If field isn't numeric, check if it's already been processed
            logger.warning(f"Field '{self.field_name}' is not a numeric type, "
                           f"possibly already processed. Skipping validation.")

            # If this is ENRICH mode, just copy the values to the output field
            batch[output_field] = field_values
            self.process_count += len(batch)
            return batch
        else:
            # Only validate if field appears to be numeric
            validate_numeric_field(batch, self.field_name, allow_null=(self.null_strategy != "ERROR"))

        # Handle null values according to strategy
        if self.null_strategy != "PRESERVE":
            field_values = process_nulls(field_values, self.null_strategy)

        # Initialize generalized_values
        generalized_values = None

        # Apply generalization based on strategy
        if self.strategy == "binning":
            generalized_values = numeric_generalization_binning(
                field_values, self.bin_count, handle_nulls=True
            )

        elif self.strategy == "rounding":
            generalized_values = numeric_generalization_rounding(
                field_values, self.precision, handle_nulls=True
            )

        elif self.strategy == "range":
            generalized_values = numeric_generalization_range(
                field_values, self.range_limits, handle_nulls=True
            )

        # Update the DataFrame directly
        if generalized_values is not None:
            # Add the generalized values to the result DataFrame
            batch[output_field] = generalized_values

            # If mode is REPLACE, also update the original field
            if self.mode == "REPLACE":
                batch[self.field_name] = generalized_values
        else:
            logger.warning(f"No generalized values produced for strategy {self.strategy}")
            # Fallback to original values
            batch[output_field] = field_values

            # If mode is REPLACE, ensure field is touched
            if self.mode == "REPLACE":
                batch[self.field_name] = field_values

        # Update process count
        self.process_count += len(batch)

        return batch  # Return the modified DataFrame

    def process_value(self, value, **params):
        """
        Process a single value using the appropriate generalization method.

        Note: This method has limitations for the 'binning' strategy as it requires
        knowing the overall data distribution. For accurate binning, use process_batch
        which processes an entire dataset with proper bin edges calculation.

        Parameters:
        -----------
        value : Any
            Value to process
        **params : dict
            Additional parameters for processing:
                - min_value: Minimum value in the dataset (for binning)
                - max_value: Maximum value in the dataset (for binning)

        Returns:
        --------
        Any
            Processed value
        """
        # Handle null value
        if pd.isna(value):
            if self.null_strategy == "PRESERVE":
                return np.nan
            elif self.null_strategy == "EXCLUDE":
                return np.nan
            elif self.null_strategy == "ERROR":
                raise ValueError("Null value encountered with null_strategy='ERROR'")

        # Convert to numeric if not already
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return value  # Return original if can't convert to numeric

        # Apply generalization based on strategy
        if self.strategy == "binning":
            # For single value binning, we need to know the overall data distribution
            # Which is not available here, so this is a simplified version
            bin_size = (params.get("max_value", 100) - params.get("min_value", 0)) / self.bin_count
            bin_index = int((numeric_value - params.get("min_value", 0)) / bin_size)
            bin_min = params.get("min_value", 0) + bin_index * bin_size
            bin_max = bin_min + bin_size
            return f"{bin_min:.2f}-{bin_max:.2f}"

        elif self.strategy == "rounding":
            if self.precision >= 0:
                # Round to decimal places
                return round(numeric_value, self.precision)
            else:
                # Round to nearest 10^|precision|
                factor = 10 ** abs(self.precision)
                return round(numeric_value / factor) * factor

        elif self.strategy == "range":
            min_val, max_val = self.range_limits
            if min_val <= numeric_value < max_val:
                return f"{min_val}-{max_val}"
            elif numeric_value < min_val:
                return f"<{min_val}"
            else:
                return f">={max_val}"

        # Fallback
        return value

    def _collect_metrics(self, original_data: pd.Series, anonymized_data: pd.Series) -> Dict[str, Any]:
        """
        Collect metrics for the operation.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing

        Returns:
        --------
        Dict[str, Any]
            Dictionary with metrics
        """
        # Basic numeric metrics
        metrics = calculate_basic_numeric_metrics(original_data, anonymized_data)

        # General metrics
        metrics.update({
            "field_name": self.field_name,
            "operation": "numeric_generalization",
            "strategy": self.strategy,
            "total_records": len(original_data),
            "null_count": int(original_data.isna().sum()),
            "mode": self.mode
        })

        # Add strategy parameters
        strategy_params = {}

        if self.strategy == "binning":
            strategy_params["bin_count"] = self.bin_count
        elif self.strategy == "rounding":
            strategy_params["precision"] = self.precision
        elif self.strategy == "range":
            strategy_params["range_limits"] = self.range_limits

        # Add generalization metrics
        metrics.update(calculate_generalization_metrics(
            original_data,
            anonymized_data,
            self.strategy,
            strategy_params
        ))

        # Add operation-specific metrics from the subclass
        metrics.update(self._collect_specific_metrics(original_data, anonymized_data))

        return metrics

    def _collect_specific_metrics(self, original_data: pd.Series, anonymized_data: pd.Series) -> Dict[str, Any]:
        """
        Collect strategy-specific metrics for numeric generalization.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before generalization
        anonymized_data : pd.Series
            Generalized data after processing

        Returns:
        --------
        Dict[str, Any]
            Dictionary with strategy-specific metrics
        """
        metrics = {}

        if self.strategy == "binning":
            # Add additional binning metrics
            metrics["average_records_per_bin"] = len(original_data) / self.bin_count if self.bin_count > 0 else 0

            # Calculate bin distribution if anonymized data is categorical
            if anonymized_data.dtype == 'category' or isinstance(anonymized_data.dtype, pd.CategoricalDtype):
                # Count records per bin
                bin_counts = anonymized_data.value_counts().to_dict()
                # Filter out nulls
                bin_counts = {k: v for k, v in bin_counts.items() if pd.notna(k)}

                # Calculate statistics on bin distribution
                if bin_counts:
                    bin_values = list(bin_counts.values())
                    metrics["min_bin_count"] = min(bin_values)
                    metrics["max_bin_count"] = max(bin_values)
                    metrics["avg_bin_count"] = sum(bin_values) / len(bin_values)
                    metrics["bin_count_std"] = np.std(bin_values) if len(bin_values) > 1 else 0

        elif self.strategy == "rounding":
            # Add additional rounding metrics
            if isinstance(self.precision, int):
                # Calculate order of magnitude of precision
                if self.precision >= 0:
                    metrics["precision_factor"] = 10 ** (-self.precision)
                else:
                    metrics["precision_factor"] = 10 ** abs(self.precision)

                # Estimate privacy level based on precision (0-1 scale)
                metrics["privacy_level"] = min(1.0, max(0.0, (self.precision + 6) / 12))

        elif self.strategy == "range":
            # Add additional range metrics
            if self.range_limits:
                min_val, max_val = self.range_limits
                metrics["range_size"] = max_val - min_val

                # Calculate coverage - percentage of values in the range
                if len(original_data) > 0:
                    # Create a boolean Series where True means the value is in the range
                    in_range = (original_data >= min_val) & (original_data < max_val)

                    # Calculate mean (proportion in range)
                    # Fix: Convert boolean Series to numeric values
                    metrics["range_coverage"] = float(np.mean(np.array(in_range, dtype=int)))
                else:
                    metrics["range_coverage"] = 0.0

        return metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        params = {
            "strategy": self.strategy,
            "version": self.version  # Include version for cache invalidation
        }

        # Add strategy-specific parameters
        if self.strategy == "binning":
            params["bin_count"] = self.bin_count
        elif self.strategy == "rounding":
            params["precision"] = self.precision
        elif self.strategy == "range":
            # Convert range_limits to strings for consistent serialization
            if self.range_limits:
                params["range_min"] = float(self.range_limits[0])
                params["range_max"] = float(self.range_limits[1])

        return params

    def _generate_visualizations(self,
                                 original_data: pd.Series,
                                 anonymized_data: pd.Series,
                                 task_dir: Path,
                                 result: OperationResult,
                                 reporter: Any) -> Dict[str, Path]:
        """
        Generate visualizations for the operation.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing
        task_dir : Path
            Task directory for saving visualizations
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter for sending artifacts

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        visualization_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Sample large datasets for visualization
            if len(original_data) > 10000:
                original_for_viz = sample_large_dataset(original_data, max_samples=10000)
                anonymized_for_viz = sample_large_dataset(anonymized_data, max_samples=10000)
            else:
                original_for_viz = original_data
                anonymized_for_viz = anonymized_data

            # Prepare data for visualization based on its type
            prepared_data, data_type = prepare_comparison_data(original_for_viz, anonymized_for_viz)

            # Generate visualization based on data type
            if data_type == "numeric" and pd.api.types.is_numeric_dtype(original_data):
                # Try to convert anonymized data to numeric if possible
                try:
                    # Generate standardized filename
                    hist_filename = generate_visualization_filename(
                        self.field_name,
                        f"{self.__class__.__name__}_{self.strategy}",
                        "histogram",
                        timestamp=timestamp
                    )

                    # Create full path for visualization
                    hist_path = create_visualization_path(task_dir, hist_filename)

                    # Determine appropriate bin count
                    n_bins = calculate_optimal_bins(original_for_viz, min_bins=5, max_bins=30)

                    # Create histogram using standard utility from pamola_core module
                    create_histogram(
                        data=prepared_data,
                        output_path=str(hist_path),
                        title=f"Distribution Comparison for {self.field_name}",
                        x_label=self.field_name,
                        y_label="Frequency",
                        bins=n_bins,
                        kde=True
                    )

                    # Register the visualization artifact
                    register_visualization_artifact(
                        result=result,
                        reporter=reporter,
                        path=hist_path,
                        field_name=self.field_name,
                        visualization_type="distribution",
                        description=f"Distribution comparison for {self.field_name} before and after generalization"
                    )

                    visualization_paths["distribution"] = hist_path
                except Exception as e:
                    logger.warning(f"Could not create numeric histogram visualization: {e}")

            # For categorical data (e.g., if binning produces categorical values)
            elif data_type == "categorical" or anonymized_data.dtype == 'category' or isinstance(anonymized_data.dtype,
                                                                                                 pd.CategoricalDtype):
                try:
                    # Generate standardized filename
                    bar_filename = generate_visualization_filename(
                        self.field_name,
                        f"{self.__class__.__name__}_{self.strategy}",
                        "categories",
                        timestamp=timestamp
                    )

                    # Create full path for visualization
                    bar_path = create_visualization_path(task_dir, bar_filename)

                    # Create bar plot using standard utility from pamola_core module
                    create_bar_plot(
                        data=prepared_data,
                        output_path=str(bar_path),
                        title=f"Category Comparison for {self.field_name}",
                        x_label="Category",
                        y_label="Count",
                        orientation="v",
                        sort_by="value",
                        max_items=10
                    )

                    # Register the visualization artifact
                    register_visualization_artifact(
                        result=result,
                        reporter=reporter,
                        path=bar_path,
                        field_name=self.field_name,
                        visualization_type="categories",
                        description=f"Category comparison for {self.field_name} before and after generalization"
                    )

                    visualization_paths["categories"] = bar_path
                except Exception as e:
                    logger.warning(f"Could not create categorical bar plot visualization: {e}")

        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")

        return visualization_paths

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare directories for artifacts following PAMOLA.CORE conventions.

        Parameters:
        -----------
        task_dir : Path
            Root task directory

        Returns:
        --------
        Dict[str, Path]
            Dictionary with prepared directories
        """
        directories = {}

        # Create standard directories following PAMOLA.CORE conventions
        directories["root"] = task_dir
        directories["output"] = task_dir / "output"
        directories["dictionaries"] = task_dir / "dictionaries"
        directories["logs"] = task_dir / "logs"
        directories["cache"] = task_dir / "cache"

        # Ensure all directories exist
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

    # Using visualization_utils functions instead of this method

    def _cleanup_memory(
            self,
            processed_df: Optional[pd.DataFrame] = None,
            original_data: Optional[pd.Series] = None,
            anonymized_data: Optional[pd.Series] = None
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        temporary attributes and forcing garbage collection.

        Parameters:
        -----------
        processed_df : pd.DataFrame, optional
            Processed DataFrame to clear from memory
        original_data : pd.Series, optional
            Original data to clear from memory
        anonymized_data : pd.Series, optional
            Anonymized data to clear from memory
        """
        # Clear argument references
        if processed_df is not None:
            del processed_df

        if original_data is not None:
            del original_data

        if anonymized_data is not None:
            del anonymized_data

        # Clear instance attribute references
        if hasattr(self, '_temp_data') and self._temp_data is not None:
            del self._temp_data
            self._temp_data = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith('_temp_'):
                delattr(self, attr_name)

        # Force garbage collection
        import gc
        gc.collect()


# Helper function to create the operation easily
def create_numeric_generalization_operation(field_name: str, **kwargs) -> NumericGeneralizationOperation:
    """
    Create a numeric generalization operation with default settings.

    Parameters:
    -----------
    field_name : str
        Field to generalize
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    NumericGeneralizationOperation
        Configured numeric generalization operation
    """
    return NumericGeneralizationOperation(field_name=field_name, **kwargs)
