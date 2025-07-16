"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Generalization Operation
Package:       pamola.core.operations.generalization
Version:       1.4.1+refactor.2025.05.22
Status:        stable
Author:        PAMOLA Core Team
Created:       2024
License:       BSD 3-Clause
Description:
   This module provides an operation for generalizing numeric fields to enhance privacy
   while maintaining data utility. It implements various strategies:
   1. Binning: Groups numeric values into discrete intervals (bins)
   2. Rounding: Reduces precision by rounding to specified decimal places
   3. Range-based: Maps values to custom ranges with special handling for outliers

Key Features:
   - Direct in-place DataFrame modification with both REPLACE and ENRICH modes
   - Robust null value handling with configurable strategies (PRESERVE, EXCLUDE, ERROR)
   - Comprehensive metrics collection for privacy impact assessment
   - Visualization generation for distribution comparisons
   - Chunked processing support for large datasets
   - Graceful handling of already-processed non-numeric fields
   - Memory-efficient operation with explicit cleanup for large datasets
   - Thread-safe visualization generation with context isolation
   - Support for multiple file formats (CSV, Parquet, Arrow)
   - Configurable visualization timeout
   - Complete cache support with artifact restoration

Framework:
   Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
   for input/output, progress tracking, and result reporting.

Change Log:
   v1.4.1 (2025-05-22):
       - Fixed operation registration to module level
       - Enhanced progress tracking with proper child trackers
       - Complete cache functionality with artifact restoration
       - Fixed duplicate timestamp in visualization filenames
       - Removed duplicate method definitions
       - Improved memory cleanup
"""

from functools import partial
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from pamola_core.common.constants import Constants

import numpy as np
import pandas as pd

from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.metric_utils import (
    calculate_basic_numeric_metrics,
    calculate_generalization_metrics,
    calculate_performance_metrics,
)
from pamola_core.anonymization.commons.processing_utils import (
    process_partition_static,
    numeric_generalization_binning,
    numeric_generalization_rounding,
    numeric_generalization_range,
    process_dataframe_dask,
    process_nulls,
    generate_output_field_name,
    process_in_chunks,
    process_dataframe_parallel,
)
from pamola_core.anonymization.commons.validation_utils import validate_numeric_field
from pamola_core.anonymization.commons.visualization_utils import (
    generate_visualization_filename,
    create_visualization_path,
    calculate_optimal_bins,
    sample_large_dataset,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker, ProgressTracker
from pamola_core.utils.visualization import create_histogram, create_bar_plot
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode


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
                "type": ["array", "null"],  # Allow null for non-range strategies
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ERROR"],
            },
            "chunk_size": {"type": "integer", "minimum": 1},
            "use_cache": {"type": "boolean"},
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": ["integer", "null"]},
            "use_vectorization": {"type": "boolean"},
            "parallel_processes": {"type": ["integer", "null"]},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]},
            # Visualization-related properties
            "visualization_theme": {"type": ["string", "null"]},
            "visualization_backend": {
                "type": ["string", "null"],
                "enum": ["plotly", "matplotlib", None],
            },
            "visualization_strict": {"type": "boolean"},
            "visualization_timeout": {"type": "integer", "minimum": 1, "default": 120},
            # Output format properties
            "output_format": {
                "type": "string",
                "enum": ["csv", "parquet", "json"],
                "default": "csv",
            },
            "encryption_mode": {"type": ["string", "null"]},
        },
        "required": ["field_name", "strategy"],
        "allOf": [
            {
                "if": {"properties": {"strategy": {"const": "binning"}}},
                "then": {"required": ["bin_count"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "rounding"}}},
                "then": {"required": ["precision"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "range"}}},
                "then": {
                    "required": ["range_limits"],
                    "properties": {
                        "range_limits": {
                            "type": "array",  # Must be array for range strategy
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        }
                    },
                },
            },
        ],
    }


@register(version="1.0.0")
class NumericGeneralizationOperation(AnonymizationOperation):
    """
    Operation for generalizing numeric data.

    This operation generalizes numeric fields using strategies like binning,
    rounding, or range-based generalization to reduce precision and improve
    anonymity while preserving analytical utility.
    """

    def __init__(
        self,
        field_name: str,
        strategy: str = "binning",  # "binning", "rounding", "range"
        bin_count: int = 10,
        precision: int = 0,  # For rounding strategy
        range_limits: Optional[Tuple[float, float]] = None,  # For range strategy
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
        null_strategy: str = "PRESERVE",
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        use_cache: bool = True,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = None,
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        output_format: str = "csv",
        description: str = "",
        encryption_mode: Optional[str] = None
    ):
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
        chunk_size : int, optional
            Chunk size for processing large datasets (default: 10000)
        use_dask : bool, optional
            Whether to use Dask for parallel processing (default: False)
        npartitions : Optional[int], optional
            Number of partitions to use with Dask (default: None)
        use_vectorization : bool, optional
            Whether to use vectorized operations (default: False)
        parallel_processes : Optional[int], optional
            Number of parallel processes to use (default: None)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]], optional
            Encryption key for securing outputs (default: None)
        visualization_theme : str, optional
            Theme to use for visualizations (default: None - uses system default)
        visualization_backend : str, optional
            Backend to use for visualizations: "plotly" or "matplotlib" (default: None - uses system default)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout in seconds for visualization generation (default: 120)
        output_format : str, optional
            Output file format: "csv", "parquet", or "arrow" (default: "csv")
        description : str, optional
            Operation description (default: "")
        """
        # Use a default description if none provided
        if not description:
            description = f"Numeric generalization for field '{field_name}' using {strategy} strategy"

        # Normalize range_limits if None
        range_limits = list(range_limits) if range_limits else [0.0, 100.0]

        # Build config parameters, excluding None values for optional fields
        config_params = {
            "field_name": field_name,
            "strategy": strategy,
            "bin_count": bin_count,
            "precision": precision,
            "range_limits": range_limits,
            "mode": mode,
            "output_field_name": output_field_name,
            "column_prefix": column_prefix,
            "null_strategy": null_strategy,
            "chunk_size": chunk_size,
            "use_dask": use_dask,
            "npartitions": npartitions,
            "use_vectorization": use_vectorization,
            "parallel_processes": parallel_processes,
            "use_cache": use_cache,
            "use_encryption": use_encryption,
            "encryption_key": encryption_key,
            "visualization_theme": visualization_theme,
            "visualization_backend": visualization_backend,
            "visualization_strict": visualization_strict,
            "visualization_timeout": visualization_timeout,
            "output_format": output_format,
            "encryption_mode": encryption_mode
        }

        # Create configuration and validate parameters
        config = NumericGeneralizationConfig(**config_params)

        # Initialize base class
        super().__init__(
            field_name=config.get("field_name"),
            mode=config.get("mode"),
            output_field_name=config.get("output_field_name"),
            column_prefix=config.get("column_prefix"),
            null_strategy=config.get("null_strategy"),
            chunk_size=config.get("chunk_size"),
            use_dask=config.get("use_dask"),
            npartitions=config.get("npartitions"),
            use_vectorization=config.get("use_vectorization"),
            parallel_processes=config.get("parallel_processes"),
            use_cache=config.get("use_cache"),
            use_encryption=config.get("use_encryption"),
            encryption_key=config.get("encryption_key"),
            visualization_backend=config.get("visualization_backend"),
            visualization_theme=config.get("visualization_theme"),
            visualization_strict=config.get("visualization_strict"),
            visualization_timeout=config.get("visualization_timeout"),
            output_format=config.get("output_format"),
            description=description,
            encryption_mode=config.get("encryption_mode")
        )

        # Assign instance properties from config
        for key, value in config_params.items():
            setattr(self, key, value)

        # Optionally store the config
        self.config = config

        # Set up performance tracking variables
        self.start_time = None
        self.end_time = None
        self.process_count = 0

        # Set up common variables
        self.force_recalculation = False  # Skip cache check
        self.generate_visualization = True  # Create visualizations
        self.encrypt_output = False  # Override encryption setting
        self.save_output = True  # Save processed data to output directory

        # Updated version for fixes
        self.version = "1.4.1"
        self.operation_name = self.__class__.__name__
        self.operation_cache = None

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
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
            - generate_visualization: bool - Create visualizations
            - encrypt_output: bool - Override encryption setting
            - save_output: bool - Save processed data to output directory
            - visualization_theme: str - Override theme for visualizations
            - visualization_backend: str - Override backend for visualizations
            - visualization_strict: bool - Override strict mode for visualizations
            - visualization_timeout: int - Override timeout for visualizations

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Initialize timing and result
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(
                f"Starting {self.operation_name} operation at {self.start_time}"
            )
            self.process_count = 0
            df = None
            result = OperationResult(status=OperationStatus.PENDING)

            self.logger.info(
                f"Starting execute for field '{self.field_name}' with strategy '{self.strategy}'"
            )

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Decompose kwargs and introduce variables for clarity
            self.encrypt_output = (
                kwargs.get("encrypt_output", False) or self.use_encryption
            )
            self.generate_visualization = kwargs.get("generate_visualization", True)
            self.save_output = kwargs.get("save_output", True)
            self.force_recalculation = kwargs.get("force_recalculation", False)
            dataset_name = kwargs.get("dataset_name", "main")

            # Extract visualization parameters
            self.visualization_theme = kwargs.get(
                "visualization_theme", self.visualization_theme
            )
            self.visualization_backend = kwargs.get(
                "visualization_backend", self.visualization_backend
            )
            self.visualization_strict = kwargs.get(
                "visualization_strict", self.visualization_strict
            )
            self.visualization_timeout = kwargs.get(
                "visualization_timeout", self.visualization_timeout
            )

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Load settings operation
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Data loading, 3. Validation, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
            TOTAL_MAIN_STEPS = 6 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            main_progress = progress_tracker
            current_steps = 0
            if main_progress:
                self.logger.info(
                    f"Setting up progress tracker with {TOTAL_MAIN_STEPS} main steps"
                )
                try:
                    main_progress.total = TOTAL_MAIN_STEPS
                    main_progress.update(
                        current_steps,
                        {
                            "step": "Starting numeric generalization",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            if self.use_cache and not self.force_recalculation:
                # Step 1: Check if we have a cached result
                if main_progress:
                    current_steps += 1
                    main_progress.update(
                        current_steps,
                        {"step": "Checking cache", "field": self.field_name},
                    )

                # Load data for cache check
                df = load_data_operation(data_source, dataset_name, **settings_operation)

                # Generate cache key based on operation parameters
                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df, reporter)

                if cache_result:
                    self.logger.info(
                        f"Using cached result for {self.field_name} generalization"
                    )

                    # Update progress
                    if main_progress:
                        main_progress.update(
                            current_steps,
                            {"step": "Complete (cached)", "field": self.field_name},
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Numeric generalization of {self.field_name} (cached)",
                            details={"cached": True},
                        )

                    return cache_result
                else:
                    self.logger.info(
                        "No cached result found, proceeding with operation"
                    )

            # Step 2: Data Loading
            self.logger.info("Step 2: Data Loading")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Data Loading"})

            ## Validate and get dataframe
            try:
                if df is None:
                    df = self._validate_and_get_dataframe(data_source, dataset_name, **settings_operation)
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Get a copy of the original data for metrics calculation
            original_data = df[self.field_name].copy(deep=True)
            self.logger.info(
                f"Original data: {len(original_data)} records, dtype: {original_data.dtype}"
            )

            # Step 3: Validation
            self.logger.info("Step 3: Validation")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Validation"})

            # Validate the field exists
            if self.field_name not in df.columns:
                error_message = f"Field '{self.field_name}' not found in DataFrame"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # We handle two cases:
            # 1. Field is numeric - proceed with validation and generalization
            # 2. Field is non-numeric - might be already processed, handle gracefully

            # Determine output field based on mode
            output_field = generate_output_field_name(
                df,
                self.field_name,
                self.mode,
                self.output_field_name,
                self.column_prefix,
            )

            # Update the output field name
            self.output_field_name = output_field

            self.logger.info(f"Output field: {output_field}, mode: {self.mode}")

            is_numeric = pd.api.types.is_numeric_dtype(df[self.field_name])
            self.logger.info(f"Field '{self.field_name}' is_numeric: {is_numeric}")

            if not is_numeric:
                self.logger.warning(
                    f"Field '{self.field_name}' is not numeric, possibly already processed. "
                    f"Will copy to output field if in ENRICH mode."
                )

                # If in ENRICH mode, copy values, if in REPLACE mode, leave as is
                if self.mode == "ENRICH":
                    df[output_field] = df[self.field_name]
                    self.logger.info(f"Copied non-numeric values to {output_field}")

                result.status = OperationStatus.SUCCESS

                # Add basic metrics about the field
                result.add_metric("field_name", self.field_name)
                result.add_metric("operation", "numeric_generalization")
                result.add_metric("strategy", self.strategy)
                result.add_metric("is_numeric", False)

                # Write output if needed
                self.logger.info(f"Writing output in format: {self.output_format}")
                encryption_mode = get_encryption_mode(df, **kwargs)
                output_result = writer.write_dataframe(
                    df=df,
                    name=f"{self.field_name}_generalized",
                    format=self.output_format,
                    subdir="output",
                    timestamp_in_name=True,
                    encryption_key=self.encryption_key if self.use_encryption else None,
                    encryption_mode=encryption_mode
                )

                # Add output artifact to result
                result.add_artifact(
                    artifact_type=self.output_format,
                    path=output_result.path,
                    description=f"{self.field_name} generalized data",
                    category=Constants.Artifact_Category_Output,
                )

                # Report to the reporter
                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} generalized data (cached)",
                        details={
                            "artifact_type": self.output_format,
                            "path": str(output_result.path),
                        },
                    )
                # Compute elapsed time
                self.end_time = time.time()
                self.logger.info(
                    f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
                )
                result.add_metric("execution_time", self.end_time - self.start_time)

                return result

            # Now validate as numeric field
            try:
                validate_numeric_field(
                    df, self.field_name, allow_null=(self.null_strategy != "ERROR")
                )
                self.logger.info("Numeric field validation passed")
            except ValueError as e:
                if "null values" in str(e) and self.null_strategy == "ERROR":
                    return OperationResult(
                        status=OperationStatus.ERROR, error_message=str(e)
                    )
                else:
                    raise

            # Step 4: Process the data
            self.logger.info("Step 4: Processing data")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Processing data"})

            # Process the data with the selected strategy
            try:
                self.logger.info(f"Processing with strategy: {self.strategy}")

                # Create child progress tracker for chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="Numeric generalization processing",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Process the dataframe
                processed_df = self._process_dataframe(df, data_tracker)

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass

                # Get the anonymized data for metrics calculation
                anonymized_data = processed_df[output_field]
                self.logger.info(
                    f"Processed data: {len(anonymized_data)} records, dtype: {anonymized_data.dtype}"
                )

                # Log sample of processed data
                if len(anonymized_data) > 0:
                    self.logger.debug(
                        f"Sample of processed data (first 5): {anonymized_data.head().tolist()}"
                    )

            except Exception as e:
                self.logger.exception(f"Error processing data: {e}")
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=str(e)
                )

            # Record end time after processing
            self.end_time = time.time()

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Step 5: Calculate metrics
            self.logger.info("Step 5: Calculating metrics")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Calculating metrics"})

            # Calculate metrics
            metrics = self._collect_metrics(original_data, anonymized_data)
            metrics.update(
                calculate_performance_metrics(
                    self.start_time, self.end_time, self.process_count
                )
            )
            self.logger.info(f"Collected {len(metrics)} metrics")

            # Generate standardized metrics filename with timestamp
            metrics_filename = generate_visualization_filename(
                self.field_name,
                f"{self.operation_name}_{self.strategy}",
                "metrics",
                timestamp=operation_timestamp,
                extension="json",
            )

            # Write metrics file
            self.logger.info("Writing metrics file")
            metrics_result = writer.write_metrics(
                metrics=metrics,
                name=metrics_filename.replace(".json", ""),  # writer appends .json
                timestamp_in_name=False,  # Already included in the filename
                encryption_key=(self.encryption_key if self.encrypt_output else None),
            )

            # Add metrics to result (ensure JSON serializable)
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    # Convert numpy types to Python native types
                    if hasattr(value, "item"):  # numpy scalar
                        value = value.item()
                    result.add_metric(key, value)
                elif isinstance(value, (np.integer, np.floating)):
                    result.add_metric(key, float(value))

            # Add metrics artifact
            result.add_artifact(
                artifact_type="json",
                path=metrics_result.path,
                description=f"{self.field_name} Numeric generalization metrics",
                category=Constants.Artifact_Category_Metrics,
            )

            # Report artifact
            if reporter:
                reporter.add_operation(
                    f"{self.field_name} Numeric generalization metrics",
                    details={
                        "artifact_type": "json",
                        "path": str(metrics_result.path),
                    },
                )

            # Step 6: Generate visualizations with context support and enhanced diagnostics
            self.logger.info("Step 6: Generating visualizations")
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Generating visualizations"}
                )

            # Generate visualizations if required
            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    kwargs_encryption = {
                        "use_encryption": self.encrypt_output,
                        "encryption_key": self.encryption_key,
                    }
                    visualization_paths = self._handle_visualizations(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        operation_timestamp=operation_timestamp,
                        **kwargs_encryption,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical
            else:
                self.logger.info(
                    "Skipping visualizations as generate_visualization is False or backend is not set"
                )

            # Step 7: Save output data
            self.logger.info("Step 7: Saving output data")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Saving output data"})

            # Save output data if required
            if self.save_output:
                try:
                    output_result_path = self._save_output_data(
                        result_df=processed_df,
                        is_encryption_required=self.encrypt_output,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        **kwargs,
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR, error_message=error_message
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        metrics=metrics,
                        visualization_paths=visualization_paths,
                        metrics_result_path=str(metrics_result.path),
                        output_result_path=output_result_path,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(processed_df, original_data, anonymized_data)

            # Report operation completion
            if reporter:
                reporter.add_operation(
                    f"Numeric generalization of {self.field_name} completed",
                    details={
                        "strategy": self.strategy,
                        "records_processed": self.process_count,
                        "execution_time": self.end_time - self.start_time,
                        "generalization_ratio": metrics.get("generalization_ratio", 0),
                    },
                )

            self.logger.info(
                f"Operation completed successfully for field '{self.field_name}'"
            )

            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle unexpected errors
            error_message = f"Error in numeric generalization operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message
            )

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single Pandas batch (partition) for numeric generalization.

        Parameters:
        -----------
        batch : pd.DataFrame
            Input DataFrame batch.

        Returns:
        --------
        pd.DataFrame
            Output batch with generalized values.
        """
        if self.field_name not in batch.columns:
            raise ValueError(f"Field '{self.field_name}' not found in DataFrame")

        field_data = batch[self.field_name]
        field_values = field_data.copy()

        # Skip non-numeric fields
        if not pd.api.types.is_numeric_dtype(field_data):
            self.logger.warning(
                f"Field '{self.field_name}' is not numeric. Skipping generalization."
            )
            batch[self.output_field_name] = field_values
            if self.mode == "REPLACE":
                batch[self.field_name] = field_values
            self.process_count += len(batch)
            return batch

        # Validate and handle nulls
        validate_numeric_field(
            batch, self.field_name, allow_null=(self.null_strategy != "ERROR")
        )
        if self.null_strategy != "PRESERVE":
            field_values = process_nulls(field_values, self.null_strategy)

        # Apply strategy (binning, rounding, etc.)
        generalized_values = self._apply_generalization_strategy(field_values)

        # Store results
        batch[self.output_field_name] = generalized_values
        if self.mode == "REPLACE":
            batch[self.field_name] = generalized_values

        self.process_count += len(batch)
        return batch

    def _apply_generalization_strategy(self, field_values: pd.Series) -> pd.Series:
        """
        Apply the selected generalization strategy to a pandas Series of numeric values.

        This method supports the following strategies:
        - "binning": groups values into discrete intervals (bins).
        - "rounding": rounds values to a specified decimal precision.
        - "range": maps values into a defined numeric range.

        If the strategy is unknown or unsupported, the original values are returned.

        Parameters:
        -----------
        field_values : pd.Series
            The numeric values to be generalized.

        Returns:
        --------
        pd.Series
            The generalized values, with nulls handled as needed.
        """
        if self.strategy == "binning":
            return numeric_generalization_binning(
                field_values, self.bin_count, handle_nulls=True
            )

        elif self.strategy == "rounding":
            return numeric_generalization_rounding(
                field_values, self.precision, handle_nulls=True
            )

        elif self.strategy == "range":
            range_limits = self._get_valid_range_limits()
            return numeric_generalization_range(
                field_values, range_limits, handle_nulls=True
            )

        self.logger.warning(
            f"No generalized values produced for unknown strategy '{self.strategy}'"
        )
        return field_values

    def _get_valid_range_limits(self) -> Tuple[float, float]:
        """
        Validate and retrieve the numeric range limits for the 'range' strategy.

        Ensures that self.range_limits is a tuple of two numeric values.
        If validation fails or range_limits is missing, returns a default range (0.0, 100.0).

        Returns:
        --------
        Tuple[float, float]
            A validated (min, max) range tuple to be used in range-based generalization.
        """
        if isinstance(self.range_limits, (list, tuple)) and len(self.range_limits) == 2:
            return tuple(self.range_limits)

        self.logger.error(
            f"Invalid or missing range_limits: {self.range_limits}. Using default (0.0, 100.0)."
        )
        return (0.0, 100.0)

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
            bin_size = (
                params.get("max_value", 100) - params.get("min_value", 0)
            ) / self.bin_count
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

    def _compute_global_bins(self, ddf: Any) -> Optional[np.ndarray]:
        """
        Compute consistent bin edges across all partitions for 'binning' strategy.

        Parameters:
        -----------
        ddf : dask.dataframe.DataFrame

        Returns:
        --------
        np.ndarray or None
            Bin edges to be reused across all partitions.
        """
        if self.strategy != "binning":
            return None

        min_val = ddf[self.field_name].min().compute()
        max_val = ddf[self.field_name].max().compute()

        return pd.cut(
            pd.Series([min_val, max_val]),
            bins=self.bin_count,
            retbins=True,
            duplicates="drop",
        )[1]

    def _build_meta(self, ddf: Any) -> pd.DataFrame:
        """
        Build Dask metadata DataFrame (empty) to ensure type safety.

        Parameters:
        -----------
        ddf : dask.dataframe.DataFrame

        Returns:
        --------
        pd.DataFrame
            Metadata DataFrame with correct types for output.
        """
        meta = ddf._meta.copy(deep=True)

        # Ensure output column exists and is of type string
        if self.output_field_name not in meta.columns:
            meta[self.output_field_name] = pd.Series(dtype=str)
        else:
            meta[self.output_field_name] = meta[self.output_field_name].astype(str)

        if self.mode == "REPLACE":
            meta[self.field_name] = meta[self.field_name].astype(str)

        return meta

    def process_with_dask(self, ddf: Any) -> pd.DataFrame:
        """
        Process the input Dask DataFrame using generalization strategy.

        Steps:
        - Precompute global bin edges if needed (for binning strategy)
        - Construct metadata to guide Dask partition processing
        - Apply generalization logic using safe, stateless static function

        Parameters:
        -----------
        ddf : dask.dataframe.DataFrame
            Input Dask DataFrame to generalize.

        Returns:
        --------
        pd.DataFrame
            Fully processed DataFrame (materialized in memory).
        """
        global_bins = self._compute_global_bins(ddf)
        meta = self._build_meta(ddf)

        # Prepare partial function with only stateless/static arguments
        process_fn = partial(
            process_partition_static,
            global_bins=global_bins,
            field_name=self.field_name,
            output_field_name=self.output_field_name,
            mode=self.mode,
            strategy=self.strategy,
            precision=getattr(self, "precision", None),
            bin_count=getattr(self, "bin_count", None),
            range_limits=getattr(self, "_get_valid_range_limits", lambda: None)(),
        )

        # Apply the function safely across Dask partitions
        processed_ddf = ddf.map_partitions(process_fn, meta=meta)
        return processed_ddf.compute()

    def _collect_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
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
        metrics.update(
            {
                "field_name": self.field_name,
                "operation": "numeric_generalization",
                "strategy": self.strategy,
                "total_records": len(original_data),
                "null_count": int(original_data.isna().sum()),
                "mode": self.mode,
            }
        )

        # Add strategy parameters
        strategy_params = {}

        if self.strategy == "binning":
            strategy_params["bin_count"] = self.bin_count
        elif self.strategy == "rounding":
            strategy_params["precision"] = self.precision
        elif self.strategy == "range":
            strategy_params["range_limits"] = self.range_limits

        # Add generalization metrics
        metrics.update(
            calculate_generalization_metrics(
                original_data, anonymized_data, self.strategy, strategy_params
            )
        )

        # Add operation-specific metrics from the subclass
        metrics.update(self._collect_specific_metrics(original_data, anonymized_data))

        return metrics

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
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
            metrics["average_records_per_bin"] = (
                len(original_data) / self.bin_count if self.bin_count > 0 else 0
            )

            # Calculate bin distribution if anonymized data is categorical
            if anonymized_data.dtype == "category" or isinstance(
                anonymized_data.dtype, pd.CategoricalDtype
            ):
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
                    metrics["bin_count_std"] = (
                        np.std(bin_values) if len(bin_values) > 1 else 0
                    )

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
                    metrics["range_coverage"] = float(
                        np.mean(np.array(in_range, dtype=int))
                    )
                else:
                    metrics["range_coverage"] = 0.0

        return metrics

    def _save_output_data(
        self,
        result_df: pd.DataFrame,
        is_encryption_required: bool,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Save the processed output data.

        Parameters:
        -----------
        result_df : pd.DataFrame
            The processed dataframe to save
        is_encryption_required : bool
            Whether to encrypt the output
        writer : DataWriter
            The writer to use for saving data
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        timestamp : Optional[str]
            Optional timestamp for the operation
        **kwargs : dict
            Additional parameters for the operation
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        custom_kwargs = self._get_custom_kwargs(result_df, **kwargs)
        # Generate standardized output filename with timestamp
        field_name_output = generate_visualization_filename(
            self.field_name,
            f"{self.operation_name}_{self.strategy}",
            "output",
            timestamp=timestamp,
        )

        # Use the DataWriter to save the DataFrame
        output_result = writer.write_dataframe(
            df=result_df,
            name=field_name_output,
            format=self.output_format,
            subdir="output",
            timestamp_in_name=False,
            encryption_key=self.encryption_key if is_encryption_required else None,
            **custom_kwargs,
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type=self.output_format,
            path=output_result.path,
            description=f"{self.field_name} generalized data",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_operation(
                f"{self.field_name} generalized data",
                details={
                    "artifact_type": self.output_format,
                    "path": str(output_result.path),
                },
            )
        return str(output_result.path)

    def _handle_visualizations(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations with thread-safe context support.

        Parameters:
        -----------
        original_data : pd.Series
            The original data before anonymization
        anonymized_data : pd.Series
            The anonymized data after processing
        task_dir : Path
            The task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        operation_timestamp : str, optional
            Timestamp for the operation (default: current time)
        **kwargs : dict
            Additional parameters for the operation
        """
        self.logger.info(
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

                self.logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                self.logger.info(
                    f"[DIAG] Field: {self.field_name}, Strategy: {self.strategy}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    self.logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        self.logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
                    except Exception as ctx_e:
                        self.logger.warning(
                            f"[DIAG] Could not inspect context: {ctx_e}"
                        )

                    # Generate visualizations with visualization context parameters
                    self.logger.info(f"[DIAG] Calling _generate_visualizations...")
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
                            self.logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations with context parameters
                    visualization_paths = self._generate_visualizations(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        task_dir=task_dir,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        progress_tracker=viz_progress,
                        timestamp=operation_timestamp,  # Pass the same timestamp
                        **kwargs,
                    )

                    # Close visualization progress tracker
                    if viz_progress:
                        try:
                            viz_progress.close()
                        except:
                            pass

                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"[DIAG] Visualization completed in {elapsed:.2f}s, generated {len(visualization_paths)} files"
                    )

                except Exception as e:
                    elapsed = time.time() - start_time
                    visualization_error = e
                    self.logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
                    self.logger.error(f"[DIAG] Stack trace:", exc_info=True)

            # Copy context for the thread
            self.logger.info(f"[DIAG] Preparing to launch visualization thread...")
            ctx = contextvars.copy_context()

            # Create thread with context
            viz_thread = threading.Thread(
                target=ctx.run,
                args=(generate_viz_with_diagnostics,),
                name=f"VizThread-{self.field_name}",
                daemon=False,  # Changed from True to ensure proper cleanup
            )

            self.logger.info(
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
                    self.logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                self.logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                self.logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = {}
            elif visualization_error:
                self.logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                self.logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                self.logger.info(
                    f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}"
                )

        except Exception as e:
            self.logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            self.logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = {}

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{self.field_name} {viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_operation(
                    f"{self.field_name} {viz_type} visualization",
                    details={"artifact_type": "png", "path": str(path)},
                )

        return visualization_paths

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        params = {
            "field_name": self.field_name,
            "strategy": self.strategy,
            "bin_count": self.bin_count,
            "precision": self.precision,
            "range_limits": self.range_limits,
            "mode": self.mode,
            "output_field_name": self.output_field_name,
            "column_prefix": self.column_prefix,
            "null_strategy": self.null_strategy,
            "chunk_size": self.chunk_size,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "use_cache": self.use_cache,
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
            "visualization_theme": self.visualization_theme,
            "visualization_backend": self.visualization_backend,
            "visualization_strict": self.visualization_strict,
            "visualization_timeout": self.visualization_timeout,
            "output_format": self.output_format,
            "force_recalculation": self.force_recalculation,
            "generate_visualization": self.generate_visualization,
            "encrypt_output": self.encrypt_output,
            "save_output": self.save_output,
        }

        return params

    def _generate_visualizations(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        task_dir: Path,
        theme: Optional[str] = None,
        backend: Optional[str] = None,
        strict: bool = False,
        progress_tracker: Optional[ProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Path]:
        """
        Generate visualizations for the operation with thread-safe context support.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing
        task_dir : Path
            Task directory for saving visualizations
        theme : str, optional
            Theme to use for visualizations
        backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        strict : bool, optional
            If True, raise exceptions for configuration errors
        progress_tracker : Optional[ProgressTracker]
            Progress tracker for visualization steps
        timestamp : str, optional
            Timestamp to use for file naming consistency
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        visualization_paths = {}

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if visualization should be skipped
        if backend is None:
            self.logger.info(
                f"Skipping visualization for {self.field_name} (backend=None)"
            )
            return visualization_paths

        self.logger.info(
            f"[VIZ] Starting visualization generation for {self.field_name} using {self.strategy} strategy"
        )
        self.logger.debug(f"[VIZ] Backend: {backend}, Theme: {theme}, Strict: {strict}")

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            # Sample large datasets for visualization
            if len(original_data) > 10000:
                self.logger.info(
                    f"[VIZ] Sampling large dataset: {len(original_data)} -> 10000 samples"
                )
                original_for_viz = sample_large_dataset(
                    original_data, max_samples=10000
                )
                anonymized_for_viz = sample_large_dataset(
                    anonymized_data, max_samples=10000
                )
            else:
                original_for_viz = original_data
                anonymized_for_viz = anonymized_data

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_for_viz)} samples"
            )
            self.logger.debug(
                f"[VIZ] Original data type: {original_for_viz.dtype}, Anonymized data type: {anonymized_for_viz.dtype}"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # Determine visualization type based on strategy and data type
            # For binning and range strategies, anonymized data will be categorical (strings)
            # For rounding strategy, data might remain numeric

            if self.strategy in ["binning", "range"]:
                self.logger.info(f"[VIZ] Using bar plot for {self.strategy} strategy")

                # Use bar plot for binning and range strategies
                try:
                    # Generate standardized filename
                    bar_filename = generate_visualization_filename(
                        self.field_name,
                        f"{self.operation_name}_{self.strategy}",
                        "distribution",
                        timestamp=timestamp,
                    )

                    # Create full path for visualization
                    bar_path = create_visualization_path(task_dir, bar_filename)
                    self.logger.info(f"[VIZ] Bar plot path: {bar_path}")

                    # Prepare data for bar plot - count occurrences of each category
                    value_counts = anonymized_for_viz.value_counts()
                    self.logger.info(
                        f"[VIZ] Value counts calculated: {len(value_counts)} unique categories"
                    )

                    # Log first few categories for debugging
                    if len(value_counts) > 0:
                        self.logger.debug(
                            f"[VIZ] Top 5 categories: {value_counts.head().to_dict()}"
                        )

                    # Step 3: Save visualization
                    if progress_tracker:
                        progress_tracker.update(3, {"step": "Saving visualization"})

                    # Create bar plot using standard utility with context support
                    self.logger.info(f"[VIZ] Calling create_bar_plot...")
                    call_start = time.time()

                    save_path = create_bar_plot(
                        data=value_counts.to_dict(),
                        output_path=str(bar_path),
                        title=f"{self.strategy.capitalize()} Distribution for {self.field_name}",
                        x_label="Category" if self.strategy == "binning" else "Range",
                        y_label="Count",
                        orientation="v",
                        sort_by="key",  # Sort by category name for better readability
                        max_items=20,
                        theme=theme,
                        backend=backend
                        or "plotly",  # Use plotly as default if backend is not None
                        strict=strict,
                        **kwargs,
                    )

                    call_time = time.time() - call_start
                    self.logger.info(
                        f"[VIZ] create_bar_plot returned after {call_time:.2f}s: {save_path}"
                    )

                    # Check if visualization was created successfully
                    if not save_path.startswith("Error"):
                        self.logger.info(
                            f"[VIZ] Bar plot created successfully: {save_path}"
                        )
                        visualization_paths["distribution"] = bar_path
                    else:
                        self.logger.error(
                            f"[VIZ] Failed to create bar plot: {save_path}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"[VIZ] Error creating bar plot visualization: {type(e).__name__}: {e}"
                    )
                    self.logger.error(f"[VIZ] Stack trace:", exc_info=True)

            elif self.strategy == "rounding":
                self.logger.info(
                    f"[VIZ] Checking data type for rounding strategy visualization"
                )

                # For rounding, check if data is still numeric
                if pd.api.types.is_numeric_dtype(anonymized_for_viz):
                    self.logger.info(
                        "[VIZ] Data is still numeric after rounding, using histogram"
                    )

                    # Use histogram for numeric data
                    try:
                        # Generate standardized filename
                        hist_filename = generate_visualization_filename(
                            self.field_name,
                            f"{self.operation_name}_{self.strategy}",
                            "histogram",
                            timestamp=timestamp,
                        )

                        # Create full path for visualization
                        hist_path = create_visualization_path(task_dir, hist_filename)
                        self.logger.info(f"[VIZ] Histogram path: {hist_path}")

                        # Prepare comparison data for histogram
                        comparison_data = {
                            "Original": original_for_viz.dropna().values,
                            "Rounded": anonymized_for_viz.dropna().values,
                        }
                        self.logger.debug(
                            f"[VIZ] Comparison data prepared: Original={len(comparison_data['Original'])}, Rounded={len(comparison_data['Rounded'])}"
                        )

                        # Determine appropriate bin count
                        n_bins = calculate_optimal_bins(
                            original_for_viz, min_bins=5, max_bins=30
                        )
                        self.logger.debug(f"[VIZ] Using {n_bins} bins for histogram")

                        # Step 3: Save visualization
                        if progress_tracker:
                            progress_tracker.update(3, {"step": "Saving visualization"})

                        # Create histogram using standard utility with context support
                        self.logger.info(f"[VIZ] Calling create_histogram...")
                        call_start = time.time()

                        save_path = create_histogram(
                            data=comparison_data,
                            output_path=str(hist_path),
                            title=f"Distribution Comparison for {self.field_name} (Rounding)",
                            x_label=self.field_name,
                            y_label="Frequency",
                            bins=n_bins,
                            kde=True,
                            theme=theme,
                            backend=backend or "plotly",
                            strict=strict,
                            **kwargs,
                        )

                        call_time = time.time() - call_start
                        self.logger.info(
                            f"[VIZ] create_histogram returned after {call_time:.2f}s: {save_path}"
                        )

                        # Check if visualization was created successfully
                        if not save_path.startswith("Error"):
                            self.logger.info(
                                f"[VIZ] Histogram created successfully: {save_path}"
                            )

                            visualization_paths["distribution"] = hist_path
                        else:
                            self.logger.error(
                                f"[VIZ] Failed to create histogram: {save_path}"
                            )

                    except Exception as e:
                        self.logger.error(
                            f"[VIZ] Error creating histogram visualization: {type(e).__name__}: {e}"
                        )
                        self.logger.error(f"[VIZ] Stack trace:", exc_info=True)
                else:
                    # If rounding produced non-numeric data, use bar plot
                    self.logger.info(
                        f"[VIZ] Rounding produced non-numeric data for {self.field_name}, using bar plot"
                    )
                    try:
                        # Generate standardized filename
                        bar_filename = generate_visualization_filename(
                            self.field_name,
                            f"{self.operation_name}_{self.strategy}",
                            "distribution",
                            timestamp=timestamp,
                        )

                        # Create full path for visualization
                        bar_path = create_visualization_path(task_dir, bar_filename)
                        self.logger.info(f"[VIZ] Bar plot path: {bar_path}")

                        # Prepare data for bar plot
                        value_counts = anonymized_for_viz.value_counts()
                        self.logger.debug(
                            f"[VIZ] Value counts calculated: {len(value_counts)} unique values"
                        )

                        # Step 3: Save visualization
                        if progress_tracker:
                            progress_tracker.update(3, {"step": "Saving visualization"})

                        # Create bar plot
                        self.logger.info(f"[VIZ] Calling create_bar_plot...")
                        call_start = time.time()

                        save_path = create_bar_plot(
                            data=value_counts.to_dict(),
                            output_path=str(bar_path),
                            title=f"Rounded Value Distribution for {self.field_name}",
                            x_label="Rounded Value",
                            y_label="Count",
                            orientation="v",
                            sort_by="key",
                            max_items=20,
                            theme=theme,
                            backend=backend or "plotly",
                            strict=strict,
                            **kwargs,
                        )

                        call_time = time.time() - call_start
                        self.logger.info(
                            f"[VIZ] create_bar_plot returned after {call_time:.2f}s: {save_path}"
                        )

                        # Check if visualization was created successfully
                        if not save_path.startswith("Error"):
                            self.logger.info(
                                f"[VIZ] Bar plot created successfully: {save_path}"
                            )

                            visualization_paths["distribution"] = bar_path
                        else:
                            self.logger.error(
                                f"[VIZ] Failed to create bar plot: {save_path}"
                            )

                    except Exception as e:
                        self.logger.error(
                            f"[VIZ] Error creating bar plot visualization: {type(e).__name__}: {e}"
                        )
                        self.logger.error(f"[VIZ] Stack trace:", exc_info=True)

        except Exception as e:
            self.logger.error(
                f"[VIZ] Error in visualization generation: {type(e).__name__}: {e}"
            )
            self.logger.debug(f"[VIZ] Stack trace:", exc_info=True)

        self.logger.info(
            f"[VIZ] Visualization generation completed. Created {len(visualization_paths)} visualizations"
        )
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

    def _cleanup_memory(
        self,
        processed_df: Optional[pd.DataFrame] = None,
        original_data: Optional[pd.Series] = None,
        anonymized_data: Optional[pd.Series] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        references and optionally calling garbage collection.

        Parameters:
        -----------
        processed_df : pd.DataFrame, optional
            Processed DataFrame to clear from memory
        original_data : pd.Series, optional
            Original data to clear from memory
        anonymized_data : pd.Series, optional
            Anonymized data to clear from memory
        """
        # Delete references
        if processed_df is not None:
            del processed_df
        if original_data is not None:
            del original_data
        if anonymized_data is not None:
            del anonymized_data

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Optional: Force garbage collection for large datasets
        # Uncomment if memory pressure is an issue
        # import gc
        # gc.collect()


# Helper function to create the operation easily
def create_numeric_generalization_operation(
    field_name: str, **kwargs
) -> NumericGeneralizationOperation:
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
