"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        DateTime Generalization Operation
Package:       pamola_core.anonymization.generalization
Version:       2.0.1
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
Updated:       2025-01-15
License:       BSD 3-Clause

Description:
This module provides an operation for generalizing datetime fields to enhance privacy
while maintaining temporal utility. It implements various strategies:
1. Rounding: Reduce precision to year, quarter, month, week, day, or hour
2. Binning: Group into time intervals (hour ranges, day ranges, custom periods)
3. Component-based: Keep only specific datetime components
4. Relative: Express as relative to reference date

Key Features:
- Multiple generalization strategies for temporal data
- Robust handling of various datetime formats and timezones
- Support for incomplete dates and null values
- Memory-efficient processing with vectorized operations
- Comprehensive temporal information loss metrics
- Visualization of temporal distributions
- Integration with k-anonymity risk assessment

Framework:
Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.

Changelog:
2.0.1 - 2025-01-15 - Critical bug fixes and improvements:
      - Improved timezone handling for naive datetimes
      - Added privacy level validation
      - Enhanced error handling with specific exceptions
      - Added constants for magic numbers
      - Fixed binning edge cases
2.0.0 - 2025-01-15 - Complete rewrite with improvements
1.1.0 - 2025-06-16 - Bug fixes and metric improvements
1.0.0 - 2025-06-15 - Initial implementation
"""

from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pandas as pd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.categorical_config import NullStrategy
from pamola_core.anonymization.commons.data_utils import process_nulls
from pamola_core.anonymization.commons.validation.exceptions import FieldValueError
from pamola_core.anonymization.commons.validation_utils import (
    BaseValidator,
    ValidationResult,
    ValidationError,
    validate_datetime_field,
)
import dask.dataframe as dd
from pamola_core.common.constants import Constants
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.helpers import filter_used_kwargs


# Constants to replace magic numbers
class DateTimeConstants:
    """Constants for datetime operations."""

    HOURS_PER_DAY = 24
    MONTHS_PER_QUARTER = 3
    MONTHS_PER_YEAR = 12
    BUSINESS_HOURS_START = 6
    BUSINESS_HOURS_MORNING_END = 12
    BUSINESS_HOURS_AFTERNOON_END = 18
    DEFAULT_SAMPLE_SIZE = 100
    DEFAULT_BATCH_SIZE = 10000
    DAYS_PER_YEAR = 365
    DAYS_PER_MONTH = 30
    DAYS_PER_WEEK = 7
    MIN_PRIVACY_REDUCTION = 0.3  # Minimum 30% reduction in unique values


# Custom exceptions for better error handling
class DateTimeParsingError(Exception):
    """Exception raised when datetime parsing fails."""

    pass


class DateTimeGeneralizationError(Exception):
    """Exception raised when generalization fails."""

    pass


class InsufficientPrivacyError(Exception):
    """Exception raised when generalization doesn't provide enough privacy."""

    pass


class DateTimeGeneralizationConfig(OperationConfig):
    """Configuration for DateTimeGeneralizationOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "strategy": {
                "type": "string",
                "enum": ["rounding", "binning", "component", "relative"],
            },
            # Rounding parameters
            "rounding_unit": {
                "type": "string",
                "enum": ["year", "quarter", "month", "week", "day", "hour"],
            },
            # Binning parameters
            "bin_type": {
                "type": "string",
                "enum": [
                    "hour_range",
                    "day_range",
                    "business_period",
                    "seasonal",
                    "custom",
                ],
            },
            "interval_size": {"type": "integer", "minimum": 1},
            "interval_unit": {
                "type": "string",
                "enum": ["hours", "days", "weeks", "months"],
            },
            "reference_date": {"type": ["string", "null"]},
            "custom_bins": {"type": ["array", "null"]},
            # Component parameters
            "keep_components": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": ["year", "month", "day", "hour", "minute", "weekday"],
                },
            },
            "strftime_output_format": {"type": ["string", "null"]},
            # Common parameters
            "timezone_handling": {
                "type": "string",
                "enum": ["preserve", "utc", "remove"],
                "default": "preserve",
            },
            "default_timezone": {"type": "string", "default": "UTC"},
            "input_formats": {"type": ["array", "null"], "items": {"type": "string"}},
            "min_privacy_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ERROR", "ANONYMIZE"],
            },
            "description": {"type": "string", "default": ""},
            "optimize_memory": {"type": "boolean"},
            "adaptive_chunk_size": {"type": "boolean"},
            "chunk_size": {"type": "integer", "minimum": 1},
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": ["integer", "null"], "minimum": 1},
            "dask_partition_size": {"type": ["string", "null"], "default": "100MB"},
            "use_vectorization": {"type": "boolean"},
            "parallel_processes": {"type": ["integer", "null"], "minimum": 1},
            "use_cache": {"type": "boolean"},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]},
            "encryption_mode": {
                "type": ["string", "null"],
                "enum": ["age", "simple", "none"],
                "default": "none",
            },
            "visualization_theme": {"type": ["string", "null"]},
            "visualization_backend": {
                "type": ["string", "null"],
                "enum": ["plotly", "matplotlib", None],
            },
            "visualization_strict": {"type": "boolean"},
            "visualization_timeout": {"type": "integer", "minimum": 1, "default": 120},
            "output_format": {
                "type": "string",
                "enum": ["csv", "parquet", "json"],
                "default": "csv",
            },
        },
        "required": ["field_name", "strategy"],
    }


@register(version="1.0.0")
class DateTimeGeneralizationOperation(AnonymizationOperation):
    """
    Operation for generalizing datetime data.

    This operation generalizes datetime fields using strategies like rounding,
    binning, or component-based generalization to reduce precision and improve
    anonymity while preserving temporal patterns.
    """

    def __init__(
        self,
        field_name: str,
        strategy: str = "rounding",
        # Rounding parameters
        rounding_unit: str = "day",
        # Binning parameters
        bin_type: str = "day_range",
        interval_size: int = 7,
        interval_unit: str = "days",
        reference_date: Optional[Union[str, datetime]] = None,
        custom_bins: Optional[List[Union[str, datetime]]] = None,
        # Component parameters
        keep_components: Optional[List[str]] = ["year", "month"],
        # Common parameters
        strftime_output_format: Optional[str] = None,
        timezone_handling: str = "preserve",
        default_timezone: str = "UTC",
        input_formats: Optional[List[str]] = None,
        min_privacy_threshold: float = DateTimeConstants.MIN_PRIVACY_REDUCTION,
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
        null_strategy: str = "PRESERVE",
        description: str = "",
        # Memory optimization
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        # Specific parameters
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        use_cache: bool = True,
        use_encryption: bool = False,
        encryption_mode: Optional[str] = "none",
        encryption_key: Optional[Union[str, Path]] = None,
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        output_format: str = "csv",
    ):
        """
        Initialize datetime generalization operation.

        Parameters:
        -----------
        field_name: str
            Name of the field to generalize
        strategy: str = "rounding",
            Generalization strategy: "rounding", "binning", "component", "relative"
        rounding_unit: str = "day",
            Unit for rounding: "year", "quarter", "month", "week", "day", "hour"
        bin_type: str = "day_range",
            Type of binning: "hour_range", "day_range", "business_period", "seasonal", "custom"
        interval_size: int = 7,
            Size of interval for binning
        interval_unit: str = "days",
            Unit for intervals: "hours", "days", "weeks", "months"
        reference_date: Optional[Union[str, datetime]] = None,
            Reference date for relative calculations
        custom_bins: Optional[List[Union[str, datetime]]] = None,
            Custom bin boundaries for "custom" bin_type
        keep_components: Optional[List[str]] = None,
            Components to keep for component strategy
        strftime_output_format: Optional[str] = None,
            Output format for strftime
        timezone_handling: str = "preserve",
            How to handle timezones: "preserve", "utc", "remove"
        default_timezone: str = "UTC",
            Default timezone for naive datetimes when converting to UTC
        input_formats: Optional[List[str]] = None,
            List of input formats to try when parsing
        min_privacy_threshold: float = DateTimeConstants.MIN_PRIVACY_REDUCTION,
            Minimum reduction in unique values (0-1) for privacy validation
        Other parameters follow base class convention
        """
        # Set default description if missing
        description = (
            description
            or f"DateTime generalization for '{field_name}' using {strategy}"
        )

        # Group parameters into a config dict
        config_params = dict(
            field_name=field_name,
            strategy=strategy,
            rounding_unit=rounding_unit,
            bin_type=bin_type,
            interval_size=interval_size,
            interval_unit=interval_unit,
            reference_date=reference_date,
            custom_bins=custom_bins,
            keep_components=keep_components,
            strftime_output_format=strftime_output_format,
            timezone_handling=timezone_handling,
            default_timezone=default_timezone,
            min_privacy_threshold=min_privacy_threshold,
            input_formats=input_formats,
            mode=mode,
            description=description,
            output_field_name=output_field_name,
            column_prefix=column_prefix,
            null_strategy=null_strategy,
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_cache=use_cache,
            use_encryption=use_encryption,
            encryption_mode=encryption_mode,
            encryption_key=encryption_key,
            visualization_theme=visualization_theme,
            visualization_backend=visualization_backend,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
            output_format=output_format,
        )

        # Create configuration
        config = DateTimeGeneralizationConfig(**config_params)

        # Initialize parent class
        super().__init__(
            **{
                k: config_params[k]
                for k in [
                    "field_name",
                    "mode",
                    "output_field_name",
                    "column_prefix",
                    "null_strategy",
                    "description",
                    "optimize_memory",
                    "adaptive_chunk_size",
                    "chunk_size",
                    "use_dask",
                    "npartitions",
                    "dask_partition_size",
                    "use_vectorization",
                    "parallel_processes",
                    "use_cache",
                    "use_encryption",
                    "encryption_mode",
                    "encryption_key",
                    "visualization_theme",
                    "visualization_backend",
                    "visualization_strict",
                    "visualization_timeout",
                    "output_format",
                ]
            }
        )

        # Set default input formats if missing
        config_params["input_formats"] = config_params.get("input_formats") or [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%m-%d-%Y",
        ]

        # Parse reference date and custom bins
        config_params["reference_date"] = self._parse_reference_date(reference_date)
        config_params["custom_bins"] = self._parse_custom_bins(custom_bins)

        # Save config attributes to self
        for k, v in config_params.items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

        self.config = config
        self.version = "3.0.2"
        self.operation_name = self.__class__.__name__
        self.operation_cache = None
        self.start_time = None
        self.end_time = None
        self.process_count = 0

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
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation including:
            - force_recalculation: bool - Skip cache check
            - generate_visualization: bool - Create visualizations
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
            self._prepare_directories(task_dir)

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
            # Main steps: 1. Cache check, 2. Data Loading & Validation, 3. Prepare output field, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
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
                            "step": "Starting date time generalization",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            if self.use_cache and not self.force_recalculation:
                try:
                    # Step 1: Check if we have a cached result
                    if main_progress:
                        current_steps += 1
                        main_progress.update(
                            current_steps,
                            {"step": "Checking cache", "field": self.field_name},
                        )

                    # Load data for cache check
                    df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
                    )

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
                                f"Date time generalization of {self.field_name} (cached)",
                                details={"cached": True},
                            )

                        return cache_result
                    else:
                        self.logger.info(
                            "No cached result found, proceeding with operation"
                        )
                except Exception as e:
                    error_message = f"Error checking cache: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Step 2: Data Loading & Validation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Data Loading", "field": self.field_name}
                )

            # Validate and get dataframe
            try:
                if df is None:
                    self.logger.info(f"Loading data for field '{self.field_name}'")
                    df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
                    )

                # Validate field is suitable for datetime operations
                is_valid = validate_datetime_field(
                    df,
                    self.field_name,
                    allow_null=(self.null_strategy != NullStrategy.ERROR.value),
                    logger_instance=self.logger,
                )

                if not is_valid:
                    raise FieldValueError(
                        self.field_name,
                        reason="Invalid datetime format",
                    )
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 3: Prepare output field
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Preparing output field", "field": self.field_name},
                )

            try:
                self.output_field_name = self._prepare_output_field(df)
                self.logger.info(f"Prepared output_field: '{self.output_field_name}'")
                self._report_operation_details(reporter, self.output_field_name)
            except Exception as e:
                error_message = f"Preparing output field error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 4: Processing
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", "field": self.field_name}
                )

            try:
                # Copy original data for processing
                original_data = df[self.field_name].copy(deep=True)

                # Create child progress tracker for Chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="Processing dataframe",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Process the filtered data
                processed_df = self._process_data_with_config(
                    df=df, progress_tracker=data_tracker
                )

                # Get the anonymized data
                anonymized_data = processed_df[self.output_field_name]

                # Validate privacy level
                if not self._validate_privacy_level(
                    original=original_data, generalized=anonymized_data
                ):
                    self.logger.warning(
                        f"Generalization doesn't provide sufficient privacy. "
                        f"Unique values reduced by less than {self.min_privacy_threshold * 100}%"
                    )

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Record end time after processing metrics
            self.end_time = time.time()

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Metrics Calculation", "field": self.field_name},
                )

            # Initialize metrics in scope
            metrics = {}

            try:
                metrics = self._collect_all_metrics(original_data, anonymized_data)

                # Generate metrics file name (in self.name existed field_name)
                metrics_file_name = (
                    f"{self.field_name}_{self.name}_metrics_{operation_timestamp}"
                )

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if self.use_encryption else None
                    ),
                )

                # Add metrics to result
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        result.add_metric(key, value)

                # Register metrics artifact
                result.add_artifact(
                    artifact_type="json",
                    path=metrics_result.path,
                    description=f"{self.field_name} anonymization metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report artifact
                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} anonymization metrics",
                        details={
                            "artifact_type": "json",
                            "path": str(metrics_result.path),
                        },
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Generating Visualizations", "field": self.field_name},
                )

            # Generate visualizations if required
            # Initialize visualization paths dictionary
            visualization_paths = {}
            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
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

            # Step 7: Save Output Data
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Save Output Data", "field": self.field_name},
                )

            # Save output data if required
            output_result_path = None
            if self.save_output:
                try:
                    safe_kwargs = filter_used_kwargs(kwargs, self._save_output_data)
                    output_result_path = self._save_output_data(
                        result_df=processed_df,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        use_encryption=self.use_encryption,
                        **safe_kwargs,
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=original_data,
                        anonymized_data=anonymized_data,
                        metrics=metrics,
                        task_dir=task_dir,
                        visualization_paths=visualization_paths,
                        metrics_result_path=str(metrics_result.path),
                        output_result_path=output_result_path,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=processed_df,
                original_data=original_data,
                anonymized_data=anonymized_data,
            )

            # Finalize timing
            self.end_time = time.time()

            # Report completion
            if reporter:
                reporter.add_operation(
                    f"Anonymization of {self.field_name} completed",
                    details={
                        "records_processed": self.process_count,
                        "execution_time": self.end_time - self.start_time,
                        "processed_df": len(processed_df),
                        "vulnerable_records_handled": metrics.get(
                            "vulnerable_records", 0
                        ),
                    },
                )

            # Set success status
            result.status = OperationStatus.SUCCESS
            result.execution_time = self.end_time - self.start_time
            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in date time generalization: {str(e)}")
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                exception=e,
            )

    def _parse_reference_date(
        self, ref_date: Optional[Union[str, datetime]]
    ) -> Optional[pd.Timestamp]:
        """Parse reference date from various formats."""
        if ref_date is None:
            return None
        if isinstance(ref_date, datetime):
            return pd.Timestamp(ref_date)
        try:
            return pd.to_datetime(ref_date)
        except (ValueError, TypeError) as e:
            raise DateTimeParsingError(
                f"Invalid reference date format: {ref_date}"
            ) from e

    def _parse_custom_bins(
        self, bins: Optional[List[Union[str, datetime]]]
    ) -> Optional[List[pd.Timestamp]]:
        """Parse custom bin boundaries."""
        if bins is None:
            return None
        parsed_bins = []
        for i, bin_value in enumerate(bins):
            if isinstance(bin_value, datetime):
                parsed_bins.append(pd.Timestamp(bin_value))
            else:
                try:
                    parsed_bins.append(pd.to_datetime(bin_value))
                except (ValueError, TypeError) as e:
                    raise DateTimeParsingError(
                        f"Invalid bin value at index {i}: {bin_value}"
                    ) from e
        return sorted(parsed_bins) if parsed_bins else None

    @classmethod
    def process_batch(cls, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of datetime data.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process
        **kwargs : Any
            Additional keyword arguments for processing

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with generalized datetimes
        """
        # Extract parameters from kwargs
        field_name = kwargs.get("field_name")
        output_field_name = kwargs.get("output_field_name", f"{field_name}_generalized")
        mode = kwargs.get("mode", "REPLACE")
        strategy = kwargs.get("strategy", "rounding")
        null_strategy = kwargs.get("null_strategy", "PRESERVE")

        # Get field data and convert to datetime if needed
        field_data = batch[field_name]

        if not pd.api.types.is_datetime64_any_dtype(field_data):
            field_data = cls._convert_to_datetime(field_data, **kwargs)

        # Handle timezone
        field_data = cls._handle_timezone(field_data, **kwargs)

        # Handle null values
        if null_strategy == "ANONYMIZE":
            field_values = process_nulls(field_data, null_strategy, cast(str, pd.NaT))
        else:
            field_values = process_nulls(field_data, null_strategy)

        # Apply generalization
        if strategy == "rounding":
            generalized = cls._apply_rounding(field_values, **kwargs)
        elif strategy == "binning":
            generalized = cls._apply_binning(field_values, **kwargs)
        elif strategy == "component":
            generalized = cls._apply_component(field_values, **kwargs)
        elif strategy == "relative":
            generalized = cls._apply_relative(field_values, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Update DataFrame
        if mode == "REPLACE":
            batch[field_name] = generalized
        else:  # ENRICH
            batch[output_field_name] = generalized

        return batch

    @classmethod
    def process_batch_dask(cls, ddf: dd.DataFrame, **kwargs) -> dd.DataFrame:
        """
        Process Dask DataFrame. Should be overridden by subclasses for optimal performance.

        Parameters:
        -----------
        ddf : dd.DataFrame
            Dask DataFrame to process
        kwargs : Any
            Additional keyword arguments for processing

        Returns:
        --------
        dd.DataFrame
            Processed Dask DataFrame
        """

        # Default implementation: process each partition with process_batch
        def process_partition(partition):
            return cls.process_batch(partition.copy(deep=True), **kwargs)

        return ddf.map_partitions(process_partition)

    @classmethod
    def process_value(cls, value, **kwargs):
        """
        Process a single datetime value.

        Parameters:
        -----------
        value : Any
            DateTime value to process
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Any
            Processed datetime value
        """
        # Extract null_strategy from kwargs
        field_name = kwargs.get("field_name")
        strategy = kwargs.get("strategy", "rounding")
        null_strategy = kwargs.get("null_strategy", "PRESERVE")

        # Handle null
        if pd.isna(value):
            if null_strategy == "PRESERVE":
                return value
            elif null_strategy == "ERROR":
                raise ValueError(f"Null value found in {field_name}")
            elif null_strategy == "ANONYMIZE":
                return pd.NaT
            return pd.NaT

        # Convert to pandas Timestamp if needed
        if not isinstance(value, pd.Timestamp):
            value = pd.Timestamp(value)

        # Apply strategy
        if strategy == "rounding":
            return cls._round_single_value(value, **kwargs)
        elif strategy == "binning":
            return cls._bin_single_value(value, **kwargs)
        elif strategy == "component":
            return cls._component_single_value(value, **kwargs)
        elif strategy == "relative":
            return cls._relative_single_value(value, **kwargs)
        else:
            return value

    @staticmethod
    def _convert_to_datetime(series: pd.Series, **kwargs) -> pd.Series:
        """Convert series to datetime trying multiple formats."""
        # First try pandas default parsing
        try:
            return pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError):
            pass
        input_formats = kwargs.get("input_formats", [])
        # Try each input format
        for fmt in input_formats:
            try:
                return pd.to_datetime(series, format=fmt, errors="coerce")
            except (ValueError, TypeError):
                continue

        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def _handle_timezone(series: pd.Series, **kwargs) -> pd.Series:
        """Handle timezone according to strategy."""
        if not pd.api.types.is_datetime64_any_dtype(series):
            return series

        # Extract timezone handling and default timezone from kwargs
        timezone_handling = kwargs.get("timezone_handling", "preserve")
        default_timezone = kwargs.get("default_timezone", "UTC")

        if timezone_handling == "utc":
            try:
                # Check if timezone aware
                if hasattr(series.dt, "tz") and series.dt.tz is not None:
                    return series.dt.tz_convert("UTC")
                else:
                    # For naive datetimes, localize to default timezone first, then convert to UTC
                    return series.dt.tz_localize(default_timezone).dt.tz_convert("UTC")
            except (AttributeError, TypeError) as e:
                return series
        elif timezone_handling == "remove":
            try:
                if hasattr(series.dt, "tz") and series.dt.tz is not None:
                    return series.dt.tz_localize(None)
                else:
                    return series
            except (AttributeError, TypeError):
                return series
        else:  # preserve
            return series

    @staticmethod
    def _apply_rounding(series: pd.Series, **kwargs) -> pd.Series:
        """Apply rounding generalization using vectorized operations."""
        result = series.copy()
        non_null_mask = ~series.isna()

        # Extract rounding unit from kwargs
        rounding_unit = kwargs.get("rounding_unit", "year")
        strftime_output_format = kwargs.get("strftime_output_format", None)

        if not non_null_mask.any():
            return result

        # Store rounded timestamps first
        rounded_values = series.copy()

        # Use vectorized operations for better performance
        if rounding_unit == "year":
            rounded_values[non_null_mask] = (
                series[non_null_mask].dt.to_period("Y").dt.to_timestamp()
            )
        elif rounding_unit == "quarter":
            rounded_values[non_null_mask] = (
                series[non_null_mask].dt.to_period("Q").dt.to_timestamp()
            )
        elif rounding_unit == "month":
            rounded_values[non_null_mask] = (
                series[non_null_mask].dt.to_period("M").dt.to_timestamp()
            )
        elif rounding_unit == "week":
            rounded_values[non_null_mask] = (
                series[non_null_mask].dt.to_period("W").dt.to_timestamp()
            )
        elif rounding_unit == "day":
            rounded_values[non_null_mask] = series[non_null_mask].dt.floor("D")
        elif rounding_unit == "hour":
            rounded_values[non_null_mask] = series[non_null_mask].dt.floor("h")

        # Apply custom format if specified
        if strftime_output_format and non_null_mask.any():
            result = result.astype("object")  # Convert to object to store strings
            # Use the rounded values for formatting, not the original series
            result[non_null_mask] = rounded_values[non_null_mask].dt.strftime(
                strftime_output_format
            )
        else:
            result = rounded_values

        return result

    @staticmethod
    def _apply_binning(series: pd.Series, **kwargs) -> pd.Series:
        """Apply binning generalization."""
        result = pd.Series(index=series.index, dtype="object")
        non_null_mask = ~series.isna()

        # Extract bin type and interval size from kwargs
        bin_type = kwargs.get("bin_type", "day_range")
        interval_size = kwargs.get("interval_size", 7)

        if not non_null_mask.any():
            return result

        non_null_values = series[non_null_mask]

        if bin_type == "hour_range":
            # Hour ranges from start of day
            hours = non_null_values.dt.hour
            bin_edges = list(
                range(0, DateTimeConstants.HOURS_PER_DAY + 1, interval_size)
            )
            # Ensure last edge covers 24 hours
            if bin_edges[-1] != DateTimeConstants.HOURS_PER_DAY:
                bin_edges.append(DateTimeConstants.HOURS_PER_DAY)

            labels = [
                f"{i:02d}:00-{min(i + interval_size, DateTimeConstants.HOURS_PER_DAY):02d}:00"
                for i in bin_edges[:-1]
            ]
            result[non_null_mask] = pd.cut(
                hours, bins=bin_edges, labels=labels, include_lowest=True
            )

        elif bin_type == "day_range":
            # Day ranges from reference date
            ref_date = kwargs.get("reference_date") or non_null_values.min()

            # Convert reference date to datetime if it's a string
            if isinstance(ref_date, str):
                ref_date = pd.to_datetime(ref_date)

            days_diff = (non_null_values - ref_date).dt.days

            # Create bins that cover all values
            min_days = int(days_diff.min())
            max_days = int(days_diff.max()) + 1

            # Ensure bins cover full range
            bin_start = min_days - (min_days % interval_size)
            bin_end = max_days + (interval_size - max_days % interval_size)
            bins = list(range(bin_start, bin_end + 1, interval_size))

            labels = [f"Day {i}-{i + interval_size - 1}" for i in bins[:-1]]
            result[non_null_mask] = pd.cut(
                days_diff, bins=bins, labels=labels, include_lowest=True
            )

        elif bin_type == "business_period":
            # Business hours
            hours = non_null_values.dt.hour
            conditions = [
                (hours >= DateTimeConstants.BUSINESS_HOURS_START)
                & (hours < DateTimeConstants.BUSINESS_HOURS_MORNING_END),
                (hours >= DateTimeConstants.BUSINESS_HOURS_MORNING_END)
                & (hours < DateTimeConstants.BUSINESS_HOURS_AFTERNOON_END),
                (hours >= DateTimeConstants.BUSINESS_HOURS_AFTERNOON_END)
                | (hours < DateTimeConstants.BUSINESS_HOURS_START),
            ]
            choices = ["Morning", "Afternoon", "Night"]
            result[non_null_mask] = np.select(conditions, choices, default="Unknown")

        elif bin_type == "seasonal":
            # Seasons
            months = non_null_values.dt.month
            conditions = [
                months.isin([12, 1, 2]),
                months.isin([3, 4, 5]),
                months.isin([6, 7, 8]),
                months.isin([9, 10, 11]),
            ]
            choices = ["Winter", "Spring", "Summer", "Fall"]
            result[non_null_mask] = np.select(conditions, choices, default="Unknown")

        elif bin_type == "custom":
            custom_bins = kwargs.get("custom_bins")
            if not custom_bins or len(custom_bins) < 2:
                raise DateTimeGeneralizationError(
                    "Custom bins must contain at least 2 datetime boundaries."
                )

            try:
                # Convert to datetime and sort
                bins = sorted(pd.to_datetime(custom_bins))
            except Exception as e:
                raise DateTimeGeneralizationError(
                    f"Invalid datetime format in custom_bins: {custom_bins}"
                ) from e

            # Generate human-readable labels from bins
            labels = [
                f"{bins[i].date()} to {bins[i + 1].date()}"
                for i in range(len(bins) - 1)
            ]

            result[non_null_mask] = pd.cut(
                non_null_values, bins=bins, labels=labels, include_lowest=True
            )

        return result

    @staticmethod
    def _apply_component(series: pd.Series, **kwargs) -> pd.Series:
        """Apply component-based generalization."""
        result = pd.Series(index=series.index, dtype="object")
        non_null_mask = ~series.isna()

        # Extract keep_components from kwargs
        keep_components = kwargs.get("keep_components", [])
        if not non_null_mask.any():
            return result

        # Build format string based on components
        format_parts = []
        if "year" in keep_components:
            format_parts.append("%Y")
        if "month" in keep_components:
            format_parts.append("%m")
        if "day" in keep_components:
            format_parts.append("%d")
        if "hour" in keep_components:
            format_parts.append("%H")
        if "minute" in keep_components:
            format_parts.append("%M")
        if "weekday" in keep_components:
            format_parts.append("%A")

        format_string = "-".join(format_parts)
        result[non_null_mask] = series[non_null_mask].dt.strftime(format_string)

        return result

    @staticmethod
    def _apply_relative(series: pd.Series, **kwargs) -> pd.Series:
        """Apply relative date generalization."""
        result = pd.Series(index=series.index, dtype="object")
        non_null_mask = ~series.isna()
        # Extract reference_date from kwargs
        reference_date = kwargs.get("reference_date", None)
        if not non_null_mask.any():
            return result

        # Use reference date or current date
        ref_date = reference_date or pd.Timestamp.now()

        # Calculate differences
        diff_days = (series[non_null_mask] - ref_date).dt.days

        # Categorize
        conditions = [
            diff_days < -DateTimeConstants.DAYS_PER_YEAR,
            (diff_days >= -DateTimeConstants.DAYS_PER_YEAR)
            & (diff_days < -DateTimeConstants.DAYS_PER_MONTH),
            (diff_days >= -DateTimeConstants.DAYS_PER_MONTH)
            & (diff_days < -DateTimeConstants.DAYS_PER_WEEK),
            (diff_days >= -DateTimeConstants.DAYS_PER_WEEK) & (diff_days < 0),
            diff_days == 0,
            (diff_days > 0) & (diff_days <= DateTimeConstants.DAYS_PER_WEEK),
            (diff_days > DateTimeConstants.DAYS_PER_WEEK)
            & (diff_days <= DateTimeConstants.DAYS_PER_MONTH),
            (diff_days > DateTimeConstants.DAYS_PER_MONTH)
            & (diff_days <= DateTimeConstants.DAYS_PER_YEAR),
            diff_days > DateTimeConstants.DAYS_PER_YEAR,
        ]
        choices = [
            "More than a year ago",
            "Months ago",
            "Weeks ago",
            "Days ago",
            "Same day",
            "Days ahead",
            "Weeks ahead",
            "Months ahead",
            "More than a year ahead",
        ]

        result[non_null_mask] = np.select(conditions, choices, default="Unknown")

        return result

    @staticmethod
    def _round_single_value(value: pd.Timestamp, **kwargs) -> Union[pd.Timestamp, str]:
        """Round a single timestamp value."""
        # Extract rounding_unit and strftime_output_format from kwargs
        rounding_unit = kwargs.get("rounding_unit", "year")
        strftime_output_format = kwargs.get("strftime_output_format", None)

        if rounding_unit == "year":
            rounded = value.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
        elif rounding_unit == "quarter":
            quarter = (value.month - 1) // DateTimeConstants.MONTHS_PER_QUARTER + 1
            rounded = value.replace(
                month=quarter * DateTimeConstants.MONTHS_PER_QUARTER - 2,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        elif rounding_unit == "month":
            rounded = value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif rounding_unit == "week":
            rounded = value - timedelta(days=value.weekday())
            rounded = rounded.replace(hour=0, minute=0, second=0, microsecond=0)
        elif rounding_unit == "day":
            rounded = value.replace(hour=0, minute=0, second=0, microsecond=0)
        elif rounding_unit == "hour":
            rounded = value.replace(minute=0, second=0, microsecond=0)
        else:
            rounded = value

        if strftime_output_format:
            return rounded.strftime(strftime_output_format)
        return rounded

    @staticmethod
    def _bin_single_value(value: pd.Timestamp, **kwargs) -> str:
        """Bin a single timestamp value."""
        # Extract bin_type and interval_size from kwargs
        bin_type = kwargs.get("bin_type", "day_range")
        interval_size = kwargs.get("interval_size", 7)

        if bin_type == "hour_range":
            hour = value.hour
            bin_start = (hour // interval_size) * interval_size
            bin_end = min(bin_start + interval_size, DateTimeConstants.HOURS_PER_DAY)
            return f"{bin_start:02d}:00-{bin_end:02d}:00"
        elif bin_type == "business_period":
            hour = value.hour
            if (
                DateTimeConstants.BUSINESS_HOURS_START
                <= hour
                < DateTimeConstants.BUSINESS_HOURS_MORNING_END
            ):
                return "Morning"
            elif (
                DateTimeConstants.BUSINESS_HOURS_MORNING_END
                <= hour
                < DateTimeConstants.BUSINESS_HOURS_AFTERNOON_END
            ):
                return "Afternoon"
            else:
                return "Night"
        elif bin_type == "seasonal":
            month = value.month
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"
        else:
            return str(value)

    @staticmethod
    def _component_single_value(value: pd.Timestamp, **kwargs) -> str:
        """Extract components from a single timestamp value."""
        # Extract keep_components and strftime_output_format from kwargs
        keep_components = kwargs.get("keep_components", [])
        strftime_output_format = kwargs.get("strftime_output_format", None)
        parts = []
        if "year" in keep_components:
            parts.append(str(value.year))
        if "month" in keep_components:
            parts.append(f"{value.month:02d}")
        if "day" in keep_components:
            parts.append(f"{value.day:02d}")
        if "hour" in keep_components:
            parts.append(f"{value.hour:02d}")
        if "minute" in keep_components:
            parts.append(f"{value.minute:02d}")
        if "weekday" in keep_components:
            parts.append(value.strftime("%A"))

        if strftime_output_format:
            return value.strftime(strftime_output_format)
        return "-".join(parts)

    @staticmethod
    def _relative_single_value(value: pd.Timestamp, **kwargs) -> str:
        """Convert single timestamp to relative description."""
        # Extract reference_date from kwargs
        ref_date = kwargs.get("reference_date", pd.Timestamp.now())
        diff_days = (value - ref_date).days

        if diff_days < -DateTimeConstants.DAYS_PER_YEAR:
            return "More than a year ago"
        elif (
            -DateTimeConstants.DAYS_PER_YEAR
            <= diff_days
            < -DateTimeConstants.DAYS_PER_MONTH
        ):
            return "Months ago"
        elif (
            -DateTimeConstants.DAYS_PER_MONTH
            <= diff_days
            < -DateTimeConstants.DAYS_PER_WEEK
        ):
            return "Weeks ago"
        elif -DateTimeConstants.DAYS_PER_WEEK <= diff_days < 0:
            return "Days ago"
        elif diff_days == 0:
            return "Same day"
        elif 0 < diff_days <= DateTimeConstants.DAYS_PER_WEEK:
            return "Days ahead"
        elif (
            DateTimeConstants.DAYS_PER_WEEK
            < diff_days
            <= DateTimeConstants.DAYS_PER_MONTH
        ):
            return "Weeks ahead"
        elif (
            DateTimeConstants.DAYS_PER_MONTH
            < diff_days
            <= DateTimeConstants.DAYS_PER_YEAR
        ):
            return "Months ahead"
        else:
            return "More than a year ahead"

    def _validate_date_range(self, series: pd.Series) -> Tuple[int, int]:
        """
        Validate dates are within pandas datetime range.

        Returns:
        --------
        Tuple[int, int]
            (out_of_range_count, total_count)
        """
        min_date = pd.Timestamp.min
        max_date = pd.Timestamp.max

        out_of_range = ((series < min_date) | (series > max_date)).sum()
        total = len(series.dropna())

        if out_of_range > 0:
            self.logger.warning(
                f"{out_of_range}/{total} dates out of valid pandas range "
                f"[{min_date}, {max_date}]"
            )

        return out_of_range, total

    def _validate_privacy_level(
        self, original: pd.Series, generalized: pd.Series
    ) -> bool:
        """
        Validate that generalization provides sufficient privacy.

        Parameters:
        -----------
        original : pd.Series
            Original datetime series
        generalized : pd.Series
            Generalized datetime series

        Returns:
        --------
        bool
            True if privacy level is sufficient, False otherwise
        """
        original_unique = original.nunique()
        generalized_unique = generalized.nunique()

        if original_unique == 0:
            return True  # No data to protect

        # Calculate reduction ratio
        reduction_ratio = 1 - (generalized_unique / original_unique)

        self.logger.info(
            f"Privacy validation: {original_unique} unique values reduced to "
            f"{generalized_unique} ({reduction_ratio:.2%} reduction)"
        )

        return reduction_ratio >= self.min_privacy_threshold

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """Collect datetime-specific metrics."""
        metrics = {}

        # Calculate temporal granularity loss
        if pd.api.types.is_datetime64_any_dtype(original_data):
            # Validate date ranges
            out_of_range, total = self._validate_date_range(original_data)
            if out_of_range > 0:
                metrics["dates_out_of_range"] = int(out_of_range)
                metrics["out_of_range_ratio"] = (
                    float(out_of_range / total) if total > 0 else 0.0
                )

            # Convert anonymized back to datetime if possible for comparison
            try:
                if self.strategy == "rounding" and self.rounding_unit in [
                    "day",
                    "hour",
                ]:
                    if pd.api.types.is_datetime64_any_dtype(anonymized_data):
                        anon_dt = anonymized_data
                    else:
                        anon_dt = pd.to_datetime(anonymized_data, errors="coerce")

                    # Calculate average time difference
                    valid_mask = ~(original_data.isna() | anon_dt.isna())
                    if valid_mask.any():
                        time_diffs = abs(
                            original_data[valid_mask] - anon_dt[valid_mask]
                        )
                        avg_loss_hours = time_diffs.dt.total_seconds().mean() / 3600
                        metrics["avg_temporal_loss_hours"] = float(avg_loss_hours)

            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not calculate temporal loss: {e}")

        # Date range coverage
        orig_range = original_data.dropna()
        if len(orig_range) > 0 and pd.api.types.is_datetime64_any_dtype(orig_range):
            date_span = (orig_range.max() - orig_range.min()).days
            metrics["original_date_span_days"] = int(date_span)

        # Unique temporal patterns
        metrics["unique_patterns_before"] = int(original_data.nunique())
        metrics["unique_patterns_after"] = int(anonymized_data.nunique())

        # Privacy metrics
        if metrics["unique_patterns_before"] > 0:
            metrics["privacy_reduction_ratio"] = float(
                1 - metrics["unique_patterns_after"] / metrics["unique_patterns_before"]
            )

        # Strategy-specific metrics
        metrics["datetime_strategy"] = self.strategy
        if self.strategy == "rounding":
            metrics["rounding_unit"] = self.rounding_unit
        elif self.strategy == "binning":
            metrics["bin_type"] = self.bin_type
            metrics["interval_size"] = self.interval_size
            if self.bin_type == "custom" and self.custom_bins:
                metrics["custom_bin_count"] = len(self.custom_bins) - 1
        elif self.strategy == "component":
            metrics["kept_components"] = self.keep_components

        return metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for date time generalization
        """
        params = dict(
            field_name=self.field_name,
            strategy=self.strategy,
            rounding_unit=self.rounding_unit,
            bin_type=self.bin_type,
            interval_size=self.interval_size,
            interval_unit=self.interval_unit,
            reference_date=self.reference_date,
            custom_bins=self.custom_bins,
            keep_components=self.keep_components,
            strftime_output_format=self.strftime_output_format,
            timezone_handling=self.timezone_handling,
            default_timezone=self.default_timezone,
            input_formats=self.input_formats,
            min_privacy_threshold=self.min_privacy_threshold,
            mode=self.mode,
            output_field_name=self.output_field_name,
            column_prefix=self.column_prefix,
            optimize_memory=self.optimize_memory,
            adaptive_chunk_size=self.adaptive_chunk_size,
            chunk_size=self.chunk_size,
            use_dask=self.use_dask,
            npartitions=self.npartitions,
            dask_partition_size=self.dask_partition_size,
            use_vectorization=self.use_vectorization,
            parallel_processes=self.parallel_processes,
            use_cache=self.use_cache,
            use_encryption=self.use_encryption,
            encryption_mode=self.encryption_mode,
            encryption_key=self.encryption_key,
            visualization_theme=self.visualization_theme,
            visualization_backend=self.visualization_backend,
            visualization_strict=self.visualization_strict,
            visualization_timeout=self.visualization_timeout,
            output_format=self.output_format,
            force_recalculation=self.force_recalculation,
            generate_visualization=self.generate_visualization,
            save_output=self.save_output,
        )

        return params


def create_datetime_generalization_operation(
    field_name: str, **kwargs
) -> DateTimeGeneralizationOperation:
    """
    Create a datetime generalization operation.

    Parameters:
    -----------
    field_name : str
        Field to generalize
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    DateTimeGeneralizationOperation
        Configured operation instance
    """
    return DateTimeGeneralizationOperation(field_name=field_name, **kwargs)
