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

import logging
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
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker, ProgressTracker
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
                "enum": ["csv", "parquet", "arrow"],
                "default": "csv",
            },
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
        )

        # Assign instance properties from config
        for key, value in config_params.items():
            setattr(self, key, value)

        # Set up performance tracking variables
        self.start_time = None
        self.end_time = None
        self.process_count = 0

        # Updated version for fixes
        self.version = "1.4.1"
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
            self.process_count = 0
            df = None
            result = OperationResult(status=OperationStatus.PENDING)

            logger.info(
                f"Starting execute for field '{self.field_name}' with strategy '{self.strategy}'"
            )

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Save configuration to task directory
            self.save_config(task_dir)

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=logger, progress_tracker=progress_tracker
            )

            # Decompose kwargs and introduce variables for clarity
            is_encryption_required = (
                kwargs.get("encrypt_output", False) or self.use_encryption
            )
            generate_visualization = kwargs.get("generate_visualization", True)
            save_output = kwargs.get("save_output", True)
            force_recalculation = kwargs.get("force_recalculation", False)
            dataset_name = kwargs.get("dataset_name", "main")

            # Extract visualization parameters
            vis_theme = kwargs.get("visualization_theme", self.visualization_theme)
            vis_backend = kwargs.get(
                "visualization_backend", self.visualization_backend
            )
            vis_strict = kwargs.get("visualization_strict", self.visualization_strict)
            vis_timeout = kwargs.get(
                "visualization_timeout", self.visualization_timeout
            )

            logger.info(
                f"Visualization settings: theme={vis_theme}, backend={vis_backend}, strict={vis_strict}, timeout={vis_timeout}s"
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Data loading, 3. Validation, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
            TOTAL_MAIN_STEPS = 6 + (
                1 if self.use_cache and not force_recalculation else 0
            )
            main_progress = progress_tracker
            current_steps = 0
            if main_progress:
                logger.info(
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
                    logger.warning(f"Could not update progress tracker: {e}")

            if self.use_cache and not force_recalculation:
                # Step 1: Check if we have a cached result
                if main_progress:
                    current_steps += 1
                    main_progress.update(
                        current_steps,
                        {"step": "Checking cache", "field": self.field_name},
                    )

                # Load data for cache check
                df = load_data_operation(data_source, dataset_name)

                logger.info("Checking operation cache...")
                # Generate cache key based on operation parameters
                data_hash = self._generate_data_hash(df[self.field_name])
                parameters = self._get_cache_parameters()
                cache_key = operation_cache.generate_cache_key(
                    operation_name=self.operation_name,
                    parameters=parameters,
                    data_hash=data_hash,
                )

                # Check for cached result
                cached_result = operation_cache.get_cache(
                    cache_key=cache_key, operation_type=self.operation_name
                )

                if cached_result:
                    logger.info(
                        f"Using cached result for {self.field_name} generalization"
                    )

                    # Create complete result from cached data
                    cached_result_obj = OperationResult(status=OperationStatus.SUCCESS)

                    # Add metrics from cache
                    metrics = cached_result.get("metrics", {})
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            cached_result_obj.add_metric(key, value)

                    # Restore artifacts from cache
                    artifacts_restored = 0

                    # Add output artifact if file exists
                    output_file = cached_result.get("output_file")
                    if output_file:
                        output_path = Path(output_file)
                        if output_path.exists():
                            cached_result_obj.add_artifact(
                                artifact_type=self.output_format,
                                path=output_path,
                                description=f"{self.field_name} generalized data (cached)",
                                category=Constants.Artifact_Category_Output,
                            )
                            artifacts_restored += 1

                            # Also report to reporter
                            if reporter:
                                reporter.add_operation(
                                    f"{self.field_name} generalized data (cached)",
                                    details={
                                        "artifact_type": self.output_format,
                                        "path": str(output_path),
                                    },
                                )
                        else:
                            logger.warning(
                                f"Cached output file not found: {output_path}"
                            )

                    # Add metrics artifact if exists
                    metrics_file = cached_result.get("metrics_file")
                    if metrics_file:
                        metrics_path = Path(metrics_file)
                        if metrics_path.exists():
                            cached_result_obj.add_artifact(
                                artifact_type="json",
                                path=metrics_path,
                                description=f"{self.field_name} generalization metrics (cached)",
                                category=Constants.Artifact_Category_Metrics,
                            )
                            artifacts_restored += 1

                            if reporter:
                                reporter.add_operation(
                                    f"{self.field_name} generalization metrics (cached)",
                                    details={
                                        "artifact_type": "json",
                                        "path": str(metrics_path),
                                    },
                                )

                    # Add visualization artifacts
                    visualizations = cached_result.get("visualizations", {})
                    for viz_type, viz_path in visualizations.items():
                        path = Path(viz_path)
                        if path.exists():
                            cached_result_obj.add_artifact(
                                artifact_type="png",
                                path=path,
                                description=f"{self.field_name} {viz_type} visualization (cached)",
                                category=Constants.Artifact_Category_Visualization,
                            )
                            artifacts_restored += 1

                            if reporter:
                                reporter.add_operation(
                                    f"{self.field_name} {viz_type} visualization (cached)",
                                    details={
                                        "artifact_type": "png",
                                        "path": str(path),
                                    },
                                )

                    # Add cache info
                    cached_result_obj.add_metric("cached", True)
                    cached_result_obj.add_metric("cache_key", cache_key)
                    cached_result_obj.add_metric(
                        "cache_timestamp", cached_result.get("timestamp", "unknown")
                    )
                    cached_result_obj.add_metric(
                        "artifacts_restored", artifacts_restored
                    )

                    # Report operation
                    if reporter:
                        reporter.add_operation(
                            f"Numeric generalization of {self.field_name} (cached)",
                            details={
                                "strategy": self.strategy,
                                "cached": True,
                                "artifacts_restored": artifacts_restored,
                            },
                        )

                    logger.info(
                        f"Cache hit successful: restored {artifacts_restored} artifacts"
                    )
                    return cached_result_obj
                else:
                    logger.info("No cached result found, proceeding with operation")

            # Step 2: Data Loading
            logger.info("Step 2: Data Loading")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Data Loading"})

            # Get DataFrame from data source
            if df is None:
                df = load_data_operation(data_source, dataset_name)

            logger.info(f"Loaded DataFrame with shape: {df.shape}")

            # Get a copy of the original data for metrics calculation
            original_data = df[self.field_name].copy(deep=True)
            logger.info(
                f"Original data: {len(original_data)} records, dtype: {original_data.dtype}"
            )

            # Step 3: Validation
            logger.info("Step 3: Validation")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Validation"})

            # Validate the field exists
            if self.field_name not in df.columns:
                error_message = f"Field '{self.field_name}' not found in DataFrame"
                logger.error(error_message)
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

            logger.info(f"Output field: {output_field}, mode: {self.mode}")

            is_numeric = pd.api.types.is_numeric_dtype(df[self.field_name])
            logger.info(f"Field '{self.field_name}' is_numeric: {is_numeric}")

            if not is_numeric:
                logger.warning(
                    f"Field '{self.field_name}' is not numeric, possibly already processed. "
                    f"Will copy to output field if in ENRICH mode."
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
                logger.info(f"Writing output in format: {self.output_format}")
                output_result = writer.write_dataframe(
                    df=df,
                    name=f"{self.field_name}_generalized",
                    format=self.output_format,
                    subdir="output",
                    timestamp_in_name=True,
                    encryption_key=self.encryption_key if self.use_encryption else None,
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

                return result

            # Now validate as numeric field
            try:
                validate_numeric_field(
                    df, self.field_name, allow_null=(self.null_strategy != "ERROR")
                )
                logger.info("Numeric field validation passed")
            except ValueError as e:
                if "null values" in str(e) and self.null_strategy == "ERROR":
                    return OperationResult(
                        status=OperationStatus.ERROR, error_message=str(e)
                    )
                else:
                    raise

            # Step 4: Process the data
            logger.info("Step 4: Processing data")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Processing data"})

            # Process the data with the selected strategy
            try:
                logger.info(f"Processing with strategy: {self.strategy}")

                # Create child progress tracker for chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        total_chunks = (len(df) - 1) // self.chunk_size + 1
                        data_tracker = main_progress.create_subtask(
                            total=total_chunks,
                            description="Chunk processing",
                            unit="chunk",
                        )
                    except Exception as e:
                        logger.debug(f"Could not create child progress tracker: {e}")

                # For larger dataframes, check if we should use parallel processing
                if self.use_dask:
                    try:
                        logger.info(
                            f"Using dask processing with chunk size {self.chunk_size}"
                        )
                        if data_tracker:
                            data_tracker.total = 3  # Setup, Processing, Finalization
                            data_tracker.update(
                                0, {"step": "Setting up dask processing"}
                            )

                        processed_df = process_dataframe_dask(
                            df=df,
                            process_function=self.process_with_dask,
                            process_function_backup=self.process_batch,
                            chunk_size=self.chunk_size,
                            npartitions=self.npartitions,
                            progress_tracker=data_tracker,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in dask processing: {e}, falling back to chunk processing"
                        )
                elif self.use_vectorization:
                    try:
                        logger.info(
                            f"Using vectorized processing with chunk size {self.chunk_size}"
                        )
                        if data_tracker:
                            data_tracker.update(
                                0, {"step": "Setting up vectorized processing"}
                            )

                        processed_df = process_dataframe_parallel(
                            df=df,
                            process_function=self.process_batch,
                            n_jobs=self.parallel_processes
                            or 1,  # Use specified threads for vectorization
                            chunk_size=self.chunk_size,
                            progress_tracker=data_tracker,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in vectorized processing: {e}, falling back to chunk processing"
                        )
                else:
                    try:
                        # Regular chunk processing
                        logger.info(
                            f"Processing in chunks with chunk size {self.chunk_size}"
                        )
                        if data_tracker:
                            total_chunks = (
                                len(df) + self.chunk_size - 1
                            ) // self.chunk_size
                            data_tracker.update(
                                0,
                                {
                                    "step": "Processing in chunks",
                                    "total_chunks": total_chunks,
                                },
                            )

                        processed_df = process_in_chunks(
                            df=df,
                            process_function=self.process_batch,
                            chunk_size=self.chunk_size,
                            progress_tracker=data_tracker,
                        )
                    except Exception as e:
                        logger.warning(f"Error in chunk processing: {e}")

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass

                # Get the anonymized data for metrics calculation
                anonymized_data = processed_df[output_field]
                logger.info(
                    f"Processed data: {len(anonymized_data)} records, dtype: {anonymized_data.dtype}"
                )

                # Log sample of processed data
                if len(anonymized_data) > 0:
                    logger.debug(
                        f"Sample of processed data (first 5): {anonymized_data.head().tolist()}"
                    )

            except Exception as e:
                logger.exception(f"Error processing data: {e}")
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=str(e)
                )

            # Step 5: Calculate metrics
            logger.info("Step 5: Calculating metrics")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Calculating metrics"})

            # Record end time after processing
            self.end_time = time.time()

            # Calculate metrics
            metrics = self._collect_metrics(original_data, anonymized_data)
            metrics.update(
                calculate_performance_metrics(
                    self.start_time, self.end_time, self.process_count
                )
            )
            logger.info(f"Collected {len(metrics)} metrics")

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate standardized metrics filename with timestamp
            metrics_filename = generate_visualization_filename(
                self.field_name,
                f"{self.operation_name}_{self.strategy}",
                "metrics",
                extension="json",
                timestamp=operation_timestamp,
            )

            # Write metrics file
            logger.info("Writing metrics file")
            metrics_result = writer.write_metrics(
                metrics=metrics,
                name=metrics_filename.replace(".json", ""),  # writer appends .json
                timestamp_in_name=False,  # Already included in the filename
                encryption_key=(
                    self.encryption_key if is_encryption_required else None
                ),
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
                description=f"{self.field_name} generalization metrics",
                category=Constants.Artifact_Category_Metrics,
            )

            # Report artifact
            if reporter:
                reporter.add_operation(
                    f"{self.field_name} generalization metrics",
                    details={
                        "artifact_type": "json",
                        "path": str(metrics_result.path),
                    },
                )

            # Step 6: Generate visualizations with context support and enhanced diagnostics
            logger.info("Step 6: Generating visualizations")
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Generating visualizations"}
                )

            # Generate visualizations if required
            if generate_visualization and vis_backend is not None:
                try:
                    visualization_paths = self._handle_visualizations(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend,
                        vis_strict=vis_strict,
                        vis_timeout=vis_timeout,
                        operation_timestamp=operation_timestamp,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    logger.warning(error_message)
                    # Continue execution - visualization failure is not critical
            else:
                logger.info(
                    "Skipping visualizations as generate_visualization is False or backend is not set"
                )

            # Step 7: Save output data
            logger.info("Step 7: Saving output data")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Saving output data"})

            # Save output data if required
            if save_output:
                # Generate standardized output filename with timestamp
                output_filename = generate_visualization_filename(
                    self.field_name,
                    f"{self.operation_name}_{self.strategy}",
                    "output",
                    extension=self.output_format,
                    timestamp=operation_timestamp,
                )

                logger.info(f"Saving output data in format: {self.output_format}")
                output_result = writer.write_dataframe(
                    df=processed_df,
                    name=output_filename.replace(
                        f".{self.output_format}", ""
                    ),  # writer appends extension
                    format=self.output_format,
                    subdir="output",
                    timestamp_in_name=False,  # Already included in the filename
                    encryption_key=self.encryption_key if self.use_encryption else None,
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
                        f"{self.field_name} generalized data",
                        details={
                            "artifact_type": self.output_format,
                            "path": str(output_result.path),
                        },
                    )

            # Cache the result if enabled - include all artifacts
            if self.use_cache:
                logger.info("Saving result to cache with complete artifacts")

                # Prepare complete cache data with all artifacts
                cache_data = {
                    "metrics": {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in metrics.items()
                    },
                    "parameters": self._get_cache_parameters(),
                    "data_info": {
                        "original_length": len(original_data),
                        "anonymized_length": len(anonymized_data),
                    },
                    "output_file": str(output_result.path),  # Path to main output file
                    "metrics_file": str(metrics_result.path),  # Path to metrics file
                    "visualizations": {
                        k: str(v) for k, v in visualization_paths.items()
                    },  # Paths to visualizations
                }

                # Generate cache key
                cache_key = operation_cache.generate_cache_key(
                    operation_name=self.operation_name,
                    parameters=self._get_cache_parameters(),
                    data_hash=self._generate_data_hash(original_data),
                )

                # Save to cache
                operation_cache.save_cache(
                    data=cache_data,
                    cache_key=cache_key,
                    operation_type=self.operation_name,
                    metadata={"task_dir": str(task_dir)},
                )

            # Clean up memory AFTER all write operations are complete
            logger.info("Cleaning up memory after all file operations")
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
                        "generalization_ratio": metrics.get("generalization_ratio", 0),
                    },
                )

            logger.info(
                f"Operation completed successfully for field '{self.field_name}'"
            )
            return result

        except Exception as e:
            # Handle unexpected errors
            error_message = f"Error in numeric generalization operation: {str(e)}"
            logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message
            )

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
        if self.field_name not in batch.columns:
            raise ValueError(f"Field '{self.field_name}' not found in DataFrame")

        field_data = batch[self.field_name]
        field_values = field_data.copy()

        if not pd.api.types.is_numeric_dtype(field_data):
            logger.warning(
                f"Field '{self.field_name}' is not a numeric type, "
                f"possibly already processed. Skipping generalization."
            )
            batch[self.output_field_name] = field_values
            if self.mode == "REPLACE":
                batch[self.field_name] = field_values
            self.process_count += len(batch)
            return batch

        # Validate the field
        validate_numeric_field(
            batch, self.field_name, allow_null=(self.null_strategy != "ERROR")
        )

        # Handle null values if needed
        if self.null_strategy != "PRESERVE":
            field_values = process_nulls(field_values, self.null_strategy)

        # Dispatch to appropriate generalization strategy
        generalized_values = self._apply_generalization_strategy(field_values)

        # Assign generalized values
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

        logger.warning(
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

        logger.error(
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

    def process_with_dask(self, ddf: Any) -> pd.DataFrame:
        """
        Process a Dask DataFrame using the same logic as `process_batch`, with optional
        binning support and correct type propagation across partitions.

        Parameters:
        -----------
        ddf : dd.DataFrame
            Dask DataFrame to process.

        Returns:
        --------
        pd.DataFrame
            The fully processed DataFrame with generalized or transformed values.
        """

        def _compute_global_bins(ddf: Any) -> Optional[np.ndarray]:
            """
            Compute global bin edges if the strategy is 'binning'.

            Returns:
            --------
            Optional[np.ndarray]
                An array of bin edges, or None if binning is not applicable.
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

        def _build_meta(ddf: Any) -> pd.DataFrame:
            """
            Construct the metadata DataFrame used by Dask for partition inference.

            Returns:
            --------
            pd.DataFrame
                A meta DataFrame with correct types for output fields.
            """
            meta = ddf._meta.copy(deep=True)

            if self.output_field_name not in meta.columns:
                meta[self.output_field_name] = pd.Series(dtype=str)
            else:
                meta[self.output_field_name] = meta[self.output_field_name].astype(str)

            if self.mode == "REPLACE":
                meta[self.field_name] = meta[self.field_name].astype(str)

            return meta

        def _process_partition(partition: pd.DataFrame) -> pd.DataFrame:
            """
            Process a single Dask partition, applying binning or fallback to `process_batch`.

            Parameters:
            -----------
            partition : pd.DataFrame
                A single partition of the Dask DataFrame.

            Returns:
            --------
            pd.DataFrame
                The transformed partition.
            """
            if self.strategy == "binning" and global_bins is not None:
                partition[self.output_field_name] = pd.cut(
                    partition[self.field_name], bins=global_bins, include_lowest=True
                ).astype(str)

                if self.mode == "REPLACE":
                    partition[self.field_name] = partition[self.output_field_name]

                return partition

            return self.process_batch(partition)

        # ---- Main Dask workflow ----
        global_bins = _compute_global_bins(ddf)
        meta = _build_meta(ddf)
        processed_ddf = ddf.map_partitions(_process_partition, meta=meta)
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
        """
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
                    f"[DIAG] Field: {self.field_name}, Strategy: {self.strategy}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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
                name=f"VizThread-{self.field_name}",
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
            "strategy": self.strategy,
            "version": self.version,  # Include version for cache invalidation
        }

        # Add strategy-specific parameters
        if self.strategy == "binning":
            params["bin_count"] = self.bin_count
        elif self.strategy == "rounding":
            params["precision"] = self.precision
        elif self.strategy == "range":
            # Store range_limits as tuple if not None
            if self.range_limits is not None:
                params["range_limits"] = self.range_limits

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
            logger.info(f"Skipping visualization for {self.field_name} (backend=None)")
            return visualization_paths

        logger.info(
            f"[VIZ] Starting visualization generation for {self.field_name} using {self.strategy} strategy"
        )
        logger.debug(f"[VIZ] Backend: {backend}, Theme: {theme}, Strict: {strict}")

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            # Sample large datasets for visualization
            if len(original_data) > 10000:
                logger.info(
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

            logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_for_viz)} samples"
            )
            logger.debug(
                f"[VIZ] Original data type: {original_for_viz.dtype}, Anonymized data type: {anonymized_for_viz.dtype}"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # Determine visualization type based on strategy and data type
            # For binning and range strategies, anonymized data will be categorical (strings)
            # For rounding strategy, data might remain numeric

            if self.strategy in ["binning", "range"]:
                logger.info(f"[VIZ] Using bar plot for {self.strategy} strategy")

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
                    logger.info(f"[VIZ] Bar plot path: {bar_path}")

                    # Prepare data for bar plot - count occurrences of each category
                    value_counts = anonymized_for_viz.value_counts()
                    logger.info(
                        f"[VIZ] Value counts calculated: {len(value_counts)} unique categories"
                    )

                    # Log first few categories for debugging
                    if len(value_counts) > 0:
                        logger.debug(
                            f"[VIZ] Top 5 categories: {value_counts.head().to_dict()}"
                        )

                    # Step 3: Save visualization
                    if progress_tracker:
                        progress_tracker.update(3, {"step": "Saving visualization"})

                    # Create bar plot using standard utility with context support
                    logger.info(f"[VIZ] Calling create_bar_plot...")
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
                    )

                    call_time = time.time() - call_start
                    logger.info(
                        f"[VIZ] create_bar_plot returned after {call_time:.2f}s: {save_path}"
                    )

                    # Check if visualization was created successfully
                    if not save_path.startswith("Error"):
                        logger.info(f"[VIZ] Bar plot created successfully: {save_path}")
                        visualization_paths["distribution"] = bar_path
                    else:
                        logger.error(f"[VIZ] Failed to create bar plot: {save_path}")

                except Exception as e:
                    logger.error(
                        f"[VIZ] Error creating bar plot visualization: {type(e).__name__}: {e}"
                    )
                    logger.error(f"[VIZ] Stack trace:", exc_info=True)

            elif self.strategy == "rounding":
                logger.info(
                    f"[VIZ] Checking data type for rounding strategy visualization"
                )

                # For rounding, check if data is still numeric
                if pd.api.types.is_numeric_dtype(anonymized_for_viz):
                    logger.info(
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
                        logger.info(f"[VIZ] Histogram path: {hist_path}")

                        # Prepare comparison data for histogram
                        comparison_data = {
                            "Original": original_for_viz.dropna().values,
                            "Rounded": anonymized_for_viz.dropna().values,
                        }
                        logger.debug(
                            f"[VIZ] Comparison data prepared: Original={len(comparison_data['Original'])}, Rounded={len(comparison_data['Rounded'])}"
                        )

                        # Determine appropriate bin count
                        n_bins = calculate_optimal_bins(
                            original_for_viz, min_bins=5, max_bins=30
                        )
                        logger.debug(f"[VIZ] Using {n_bins} bins for histogram")

                        # Step 3: Save visualization
                        if progress_tracker:
                            progress_tracker.update(3, {"step": "Saving visualization"})

                        # Create histogram using standard utility with context support
                        logger.info(f"[VIZ] Calling create_histogram...")
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
                        )

                        call_time = time.time() - call_start
                        logger.info(
                            f"[VIZ] create_histogram returned after {call_time:.2f}s: {save_path}"
                        )

                        # Check if visualization was created successfully
                        if not save_path.startswith("Error"):
                            logger.info(
                                f"[VIZ] Histogram created successfully: {save_path}"
                            )

                            visualization_paths["distribution"] = hist_path
                        else:
                            logger.error(
                                f"[VIZ] Failed to create histogram: {save_path}"
                            )

                    except Exception as e:
                        logger.error(
                            f"[VIZ] Error creating histogram visualization: {type(e).__name__}: {e}"
                        )
                        logger.error(f"[VIZ] Stack trace:", exc_info=True)
                else:
                    # If rounding produced non-numeric data, use bar plot
                    logger.info(
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
                        logger.info(f"[VIZ] Bar plot path: {bar_path}")

                        # Prepare data for bar plot
                        value_counts = anonymized_for_viz.value_counts()
                        logger.debug(
                            f"[VIZ] Value counts calculated: {len(value_counts)} unique values"
                        )

                        # Step 3: Save visualization
                        if progress_tracker:
                            progress_tracker.update(3, {"step": "Saving visualization"})

                        # Create bar plot
                        logger.info(f"[VIZ] Calling create_bar_plot...")
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
                        )

                        call_time = time.time() - call_start
                        logger.info(
                            f"[VIZ] create_bar_plot returned after {call_time:.2f}s: {save_path}"
                        )

                        # Check if visualization was created successfully
                        if not save_path.startswith("Error"):
                            logger.info(
                                f"[VIZ] Bar plot created successfully: {save_path}"
                            )

                            visualization_paths["distribution"] = bar_path
                        else:
                            logger.error(
                                f"[VIZ] Failed to create bar plot: {save_path}"
                            )

                    except Exception as e:
                        logger.error(
                            f"[VIZ] Error creating bar plot visualization: {type(e).__name__}: {e}"
                        )
                        logger.error(f"[VIZ] Stack trace:", exc_info=True)

        except Exception as e:
            logger.error(
                f"[VIZ] Error in visualization generation: {type(e).__name__}: {e}"
            )
            logger.debug(f"[VIZ] Stack trace:", exc_info=True)

        logger.info(
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
