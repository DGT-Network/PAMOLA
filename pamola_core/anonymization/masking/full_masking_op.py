"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Full Masking Operation
Package:       pamola_core.anonymization.masking
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
Updated:       2025-07-25
License:       BSD 3-Clause

Description:
This module implements the FullMaskingOperation class that provides complete field value masking
by replacing entire values with masking characters while optionally preserving format and structure.

Key Features:
- Full value masking: replace all characters in a field with a configurable mask character
- Optional format preservation using regex patterns
- Support for numeric, string, and date fields with flexible output types
- Conditional masking and k-anonymity integration
- Memory and performance optimizations (chunking, Dask, vectorization)
- Metrics and visualization support for privacy assessment
- Configurable output, encryption, and reporting

Framework:
Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting. Utilizes masking_patterns.py
for centralized pattern management.
"""

import time
from pathlib import Path
import random
import re
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Import base class
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Import commons utilities
from pamola_core.anonymization.commons.masking_patterns import MaskingPatterns
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
)
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_suppression_rate,
    get_process_summary,
)
from pamola_core.anonymization.commons.text_processing_utils import normalize_text
from pamola_core.anonymization.commons.visualization_utils import (
    create_category_distribution_comparison,
    create_comparison_visualization,
    sample_large_dataset,
)
from pamola_core.common.constants import Constants
from pamola_core.io.base import DataWriter
from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_data_writer import DataWriter

# Default values
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS = 20


class FullMaskingConfig(OperationConfig):
    """Configuration for FullMaskingOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "mask_char": {"type": "string", "default": "*"},
            "preserve_length": {"type": "boolean", "default": True},
            "fixed_length": {"type": ["integer", "null"], "minimum": 0},
            "random_mask": {"type": "boolean", "default": False},
            "mask_char_pool": {"type": ["string", "null"]},
            "preserve_format": {"type": "boolean", "default": False},
            "format_patterns": {
                "type": ["object", "null"],
            },
            "numeric_output": {
                "type": "string",
                "enum": ["string", "numeric", "preserve"],
                "default": "string",
            },
            "date_format": {"type": ["string", "null"]},
            "condition_field": {"type": ["string", "null"]},
            "condition_values": {"type": ["array", "null"]},
            "condition_operator": {"type": "string"},
            "ka_risk_field": {"type": ["string", "null"]},
            "risk_threshold": {"type": "number"},
            "vulnerable_record_strategy": {"type": "string"},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"],
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
        "required": ["field_name", "mask_char"],
    }


@register(version="1.0.0")
class FullMaskingOperation(AnonymizationOperation):
    """
    Full masking operation that replaces entire field values with masking characters.

    This operation provides complete obfuscation of sensitive data by replacing all
    characters in a field with a configurable masking character while optionally
    preserving data format and structure.
    """

    def __init__(
        self,
        field_name: str,
        # ==== Output & Replacement ====
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "masked_",
        null_strategy: str = "PRESERVE",
        description: str = "",
        # ==== Masking configuration ====
        mask_char: str = "*",
        preserve_length: bool = True,
        fixed_length: Optional[int] = None,
        random_mask: bool = False,
        mask_char_pool: Optional[str] = None,
        # Format handling
        preserve_format: bool = False,
        format_patterns: Optional[Dict[str, str]] = None,
        # Type-specific handling
        numeric_output: str = "string",  # string, numeric, preserve
        date_format: Optional[str] = None,
        # Conditional masking
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: str = "in",
        # K-anonymity integration
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        vulnerable_record_strategy: str = "mask",
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
        Initialize full masking operation.

        Parameters:
        -----------
        field_name: str
            Name of the field to mask
        mode: str
            Operation mode, e.g. 'REPLACE' or 'ENRICH' (default: 'REPLACE')
        output_field_name: Optional[str]
            Name of the output field (default: None)
        column_prefix: str
            Prefix for output column if enriching (default: 'masked_')
        null_strategy: str
            Strategy for handling null values (default: 'PRESERVE')
        description: str
            Description of the operation (default: '')
        mask_char: str
            Character used for masking (default: '*')
        preserve_length: bool
            Whether to preserve the original length of the field (default: True)
        fixed_length: Optional[int]
            Fixed length for masking (if None, uses original length)
        random_mask: bool
            Whether to use random characters from a pool for masking (default: False)
        mask_char_pool: Optional[str]
            Pool of characters to use for random masking (default: None, uses alphanumeric + symbols if random_mask is True)
        preserve_format: bool
            Whether to preserve the format of the original field (default: False)
        format_patterns: Dict[str, str]
            Custom regex patterns for format preservation (default: {}, uses common patterns)
        numeric_output: str
            Output type for numeric fields ('string', 'numeric', or 'preserve') (default: 'string')
        date_format: Optional[str]
            Format for date fields (default: None, uses ISO format)
        condition_field: Optional[str]
            Field to apply conditional masking (default: None)
        condition_values: Optional[List]
            Values for conditional masking (default: None)
        condition_operator: str
            Operator for conditional masking (default: 'in')
        ka_risk_field: Optional[str]
            Field for k-anonymity risk calculation (default: None)
        risk_threshold: float
            Risk threshold for k-anonymity (default: 5.0)
        vulnerable_record_strategy: str
            Strategy for vulnerable records (default: 'mask')
        optimize_memory: bool
            Whether to optimize memory usage (default: True)
        adaptive_chunk_size: bool
            Whether to use adaptive chunk size (default: True)
        chunk_size: int
            Size of data chunks for processing (default: 10000)
        use_dask: bool
            Whether to use Dask for distributed processing (default: False)
        npartitions: Optional[int]
            Number of Dask partitions (default: None)
        dask_partition_size: Optional[str]
            Dask partition size (default: None)
        use_vectorization: bool
            Whether to use vectorized operations (default: False)
        parallel_processes: Optional[int]
            Number of parallel processes (default: None)
        use_cache: bool
            Whether to use cache for results (default: True)
        use_encryption: bool
            Whether to encrypt output files (default: False)
        encryption_mode: Optional[str]
            Encryption mode (default: 'none')
        encryption_key: Optional[Union[str, Path]]
            Encryption key or path (default: None)
        visualization_theme: Optional[str]
            Theme for visualizations (default: None)
        visualization_backend: Optional[str]
            Backend for visualizations (default: 'plotly')
        visualization_strict: bool
            Whether to use strict mode for visualizations (default: False)
        visualization_timeout: int
            Timeout for visualizations in seconds (default: 120)
        output_format: str
            Output file format (default: 'csv')
        Other parameters follow base class convention
        """

        # Set default description if missing
        description = (
            description
            or f"Full masking operation for '{field_name}' with mask character '{mask_char}'"
        )

        # Group parameters into a config dict
        config_params = dict(
            field_name=field_name,
            mask_char=mask_char,
            preserve_length=preserve_length,
            fixed_length=fixed_length,
            random_mask=random_mask,
            mask_char_pool=mask_char_pool,
            preserve_format=preserve_format,
            format_patterns=format_patterns or {},
            numeric_output=numeric_output,
            date_format=date_format,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,
            null_strategy=null_strategy,
            description=description,
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
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

        # Create config object (you can keep this if needed for validation)
        config = FullMaskingConfig(**config_params)

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
                    "condition_field",
                    "condition_values",
                    "condition_operator",
                    "ka_risk_field",
                    "risk_threshold",
                    "vulnerable_record_strategy",
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

        # Save config attributes to self
        for k, v in config_params.items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

        if not self.format_patterns:
            self._setup_format_patterns()

        self.process_kwargs["format_patterns"] = self.format_patterns

        # Setup random mask pool if needed
        if self.random_mask and not self.mask_char_pool:
            self.mask_char_pool = string.ascii_letters + string.digits + "!@#$%^&*"

        self.config = config
        self.version = "4.0.0"
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

            # Initialize result object
            result = OperationResult(status=OperationStatus.PENDING)

            self.logger.info(
                f"Starting execute for field '{self.field_name}' with strategy '{self.null_strategy}'"
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
                            "step": "Starting full masking operation",
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
                                f"Full masking operation of {self.field_name} (cached)",
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
                # Validate configuration early
                self._validate_configuration()

                if df is None:
                    self.logger.info(f"Loading data for field '{self.field_name}'")
                    df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
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

                # Apply conditional filtering
                self.filter_mask, filtered_df = self._apply_conditional_filtering(df)

                # Handle vulnerable records if k-anonymity is enabled
                if self.ka_risk_field and self.ka_risk_field in df.columns:
                    filtered_df = self._handle_vulnerable_records(
                        filtered_df, self.output_field_name
                    )

                # Process the filtered data
                processed_df = self._process_data_with_config(
                    df=filtered_df,
                    progress_tracker=data_tracker,
                )

                # Get the anonymized data
                anonymized_data = processed_df[self.output_field_name]

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
                metrics = self._collect_comprehensive_metrics(
                    original_data, anonymized_data, processed_df
                )

                # Generate metrics file name (in self.name existed field_name)
                metrics_file_name = f"{self.field_name}_{self.name}_{self.null_strategy}_metrics_{operation_timestamp}"

                # Save metrics using write
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        str(self.encryption_key)
                        if self.use_encryption and self.encryption_key
                        else None
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
                # Log summary
                summary = get_process_summary(metrics.get("privacy_metrics", {}))
                for key, message in summary.items():
                    self.logger.info(f"{key}: {message}")

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
                    # Save the processed DataFrame
                    safe_kwargs = filter_used_kwargs(
                        kwargs, FullMaskingOperation._save_output_data
                    )
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
                    f"Full masking operation of {self.field_name} completed",
                    details={
                        "records_processed": self.process_count,
                        "execution_time": self.end_time - self.start_time,
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
            self.logger.error(f"Error in Full masking operation: {str(e)}")
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                exception=e,
            )

    def _setup_format_patterns(self):
        """
        Setup default format patterns for preservation.

        This method initializes the format_patterns attribute with default regex patterns
        if not already provided. It merges user-defined patterns with defaults.
        """
        try:
            # Get default patterns as Dict[str, str] (regex strings)
            default_patterns = MaskingPatterns.get_default_patterns()
            # Merge with custom patterns
            for pattern_name, pattern in default_patterns.items():
                if pattern_name not in self.format_patterns:
                    self.format_patterns[pattern_name] = pattern.regex
        except Exception as e:
            self.logger.warning(f"Error setting up format patterns: {e}")
            if not hasattr(self, "format_patterns") or self.format_patterns is None:
                self.format_patterns = {}

    def _validate_configuration(self) -> None:
        """
        Validate masking operation configuration.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """

        # Validate mask character
        if not self.mask_char or len(self.mask_char) != 1:
            raise ValueError("mask_char must be a single character")

        # Validate fixed length
        if self.fixed_length is not None and self.fixed_length < 0:
            raise ValueError(
                "fixed_length must be non-negative"
            )  # Validate numeric output option
        if self.numeric_output not in ["string", "numeric", "preserve"]:
            raise ValueError(
                "numeric_output must be 'string', 'numeric', or 'preserve'"
            )

        # Validate risk threshold
        if self.risk_threshold < 0:
            raise ValueError("risk_threshold must be non-negative")

        # Validate format patterns
        if self.format_patterns is not None:
            if not isinstance(self.format_patterns, dict):
                raise ValueError("format_patterns must be a dictionary")

            for pattern_name, pattern_regex in self.format_patterns.items():
                if not isinstance(pattern_name, str):
                    raise ValueError(
                        f"format_patterns keys must be strings, got {type(pattern_name)}"
                    )

                if not isinstance(pattern_regex, str):
                    raise ValueError(
                        f"format_patterns values must be regex strings, got {type(pattern_regex)} for pattern '{pattern_name}'"
                    )

                # Validate regex pattern
                try:
                    re.compile(pattern_regex)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern for '{pattern_name}': {e}")

    @classmethod
    def _mask_with_format(cls, value: str, **kwargs) -> str:
        """
        Mask the value while preserving structural format using regex patterns.

        If no format pattern matches, fallback to basic masking.

        Args:
            value (str): The value to mask.
            **kwargs: Additional masking parameters.

        Returns:
            str: Masked value with format preserved if possible.
        """

        mask_char = kwargs.get("mask_char", "*")
        format_patterns = kwargs.get("format_patterns", {})

        if pd.isna(value) or value is None:
            return value

        for name, regex in format_patterns.items():
            # Handle both PatternConfig and raw regex string
            match = re.match(regex, value)
            if not match:
                continue
            groups = match.groups()
            masked_groups = []
            for i, g in enumerate(groups, start=1):
                masked_groups.append(mask_char * len(g))

            return cls._reconstruct_format(masked_groups, match, value)

        return cls._mask_value(value, **kwargs)

    @staticmethod
    def _reconstruct_format(groups: List[str], match: re.Match, value: str) -> str:
        """
        Reconstruct string from masked/unmasked groups, preserving original separators.

        Args:
            groups (List[str]): List of masked/unmasked group values.
            match (re.Match): Regex match object.
            value (str): Original value to reconstruct within.

        Returns:
            str: Fully reconstructed value with masked content.
        """
        start, end = match.span()
        prefix = value[:start]
        suffix = value[end:]
        # Build reconstructed string from match span
        group_starts = [match.start(i + 1) - start for i in range(len(groups))]
        group_ends = [match.end(i + 1) - start for i in range(len(groups))]
        reconstructed = ""
        last_idx = 0
        for gs, ge, group in zip(group_starts, group_ends, groups):
            if gs > last_idx:
                reconstructed += value[start + last_idx : start + gs]
            reconstructed += group
            last_idx = ge
        if last_idx < end - start:
            reconstructed += value[start + last_idx : end]
        return prefix + reconstructed + suffix

    @staticmethod
    def _mask_to_numeric(mask: str, value: str, **kwargs) -> Union[int, float]:
        """
        Convert a masked string into a numeric representation, preserving
        the format (int, float, scientific notation) of the original value.

        Args:
            mask (str): The masked string (e.g. '******' or '@9#K$%').
            value (str): The original value as a string (e.g. '1234.56', '1.2e+04').
            **kwargs: Additional masking parameters.

        Returns:
            int or float: Masked numeric representation.
        """

        # Extract parameters from kwargs
        random_mask = kwargs.get("random_mask", False)
        mask_char = kwargs.get("mask_char", "*")

        def is_scientific(s: str) -> bool:
            return bool(re.search(r"^[+-]?[\d.]+[eE][+-]?\d+$", s))

        original_str = str(value).strip()
        is_float = "." in original_str or is_scientific(original_str)

        # Determine digit string based on masking strategy
        if random_mask:
            digit_str = "".join(random.choices(string.digits, k=len(mask)))
        else:
            digit_char = mask_char if mask_char.isdigit() else "9"
            digit_str = digit_char * len(mask)

        # --- Handle scientific notation
        if is_scientific(original_str):
            match = re.match(r"([+-]?[\d.]+)[eE]([+-]?\d+)", original_str)
            if match:
                base = match.group(1)
                exponent = match.group(2)

                # Build a fake base with one decimal digit
                if len(digit_str) >= 2:
                    base_str = f"{digit_str[0]}.{digit_str[1:]}"
                else:
                    base_str = digit_str

                try:
                    return float(f"{base_str}e{exponent}")
                except ValueError:
                    return 0.0

        # --- Handle float
        if is_float and "." in original_str:
            int_len = original_str.find(".")
            frac_len = len(original_str) - int_len - 1

            # Pad if not enough digits
            if len(digit_str) < int_len + frac_len:
                digit_str = digit_str.ljust(int_len + frac_len, "0")

            int_part = digit_str[:int_len]
            frac_part = digit_str[int_len : int_len + frac_len]
            numeric = f"{int_part}.{frac_part}"

            try:
                return float(numeric)
            except ValueError:
                return 0.0

        # --- Default: integer output
        try:
            return int(digit_str)
        except ValueError:
            return 0

    @classmethod
    def process_batch(cls, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of data with full masking.

        Args:
            batch (pd.DataFrame): Input DataFrame batch.
            **kwargs: Masking configuration parameters.

        Returns:
            pd.DataFrame: DataFrame with masked values.
        """

        # Extract parameters from kwargs
        field_name = kwargs.get("field_name", "field")
        mode = kwargs.get("mode")
        output_field_name = kwargs.get("output_field_name", "masked")
        date_format = kwargs.get("date_format", None)
        numeric_output = kwargs.get("numeric_output", "string")
        preserve_format = kwargs.get("preserve_format", False)

        # Check if the field exists
        if field_name not in batch.columns:
            raise ValueError(f"Field {field_name} not found in DataFrame")

        result = batch.copy()

        # Determine output column
        output_col = field_name if mode == "REPLACE" else output_field_name

        # If output_col is new, initialize it
        if output_col != field_name and output_col not in result.columns:
            result[output_col] = result[field_name]

        result[output_col] = result[output_col].astype(str)

        if date_format:
            # Convert datetime to string with specified format
            try:
                result[output_col] = pd.to_datetime(result[field_name], errors="coerce")
                result[output_col] = result[output_col].dt.strftime(date_format)
            except Exception:
                cls.logger.warning(
                    f"Failed to convert {field_name} to datetime with format {date_format}. "
                    "Falling back to string masking."
                )
                pass

        if numeric_output != "preserve" or not is_numeric_dtype(result[field_name]):
            if preserve_format and cls._is_string_field(batch[field_name]):
                # Format-aware masking
                result[output_col] = result[output_col].apply(
                    lambda x: cls._mask_with_format(x, **kwargs)
                )
            else:
                # Standard masking
                result[output_col] = result[output_col].apply(
                    lambda x: cls._mask_value(x, **kwargs)
                )

        return result

    @staticmethod
    def _is_string_field(series: pd.Series) -> bool:
        """
        Check if a field contains string data.

        Args:
            series (pd.Series): Input data series to check.

        Returns:
            bool: True if the field contains string data.
        """

        # Check if dtype is string or object
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # For object dtype, check if values are actually strings
            if pd.api.types.is_object_dtype(series):
                # Sample some values to check
                sample = series.dropna().head(100)
                if len(sample) > 0:
                    return all(isinstance(x, str) for x in sample)
            else:
                return True
        return False

    @staticmethod
    def _can_vectorize(**kwargs) -> bool:
        """
        Check if masking can be vectorized for performance.

        Args:
            **kwargs: Masking configuration parameters.

        Returns:
            bool: True if vectorization is possible.
        """

        # Extract parameters from kwargs
        random_mask = kwargs.get("random_mask", False)
        date_format = kwargs.get("date_format", None)
        numeric_output = kwargs.get("numeric_output", "string")
        preserve_format = kwargs.get("preserve_format", False)

        # Simple cases that can be vectorized
        return (
            not preserve_format
            and not random_mask
            and numeric_output == "string"
            and not date_format
        )

    @staticmethod
    def _vectorized_mask(series: pd.Series, **kwargs) -> pd.Series:
        """
        Apply masking using vectorized operations.

        Args:
            series (pd.Series): Series to mask.
            **kwargs: Masking configuration parameters.

        Returns:
            pd.Series: Masked series.
        """

        # Extract parameters from kwargs
        fixed_length = kwargs.get("fixed_length", None)
        preserve_length = kwargs.get("preserve_length", False)
        mask_char = kwargs.get("mask_char", "*")

        if fixed_length:
            # Fixed length masking
            return pd.Series(
                [mask_char * fixed_length] * len(series), index=series.index
            )
        elif preserve_length:
            # Preserve length masking
            return series.astype(str).str.len().apply(lambda x: mask_char * x)
        else:
            # Default length
            return pd.Series([mask_char * 8] * len(series), index=series.index)

    @classmethod
    def _mask_value(cls, value: Any, **kwargs) -> Union[str, float, int, None]:
        """
        Apply full masking to a single value.

        Args:
            value: Value to mask.
            **kwargs: Masking configuration parameters.

        Returns:
            Masked value as string, number, or None.
        """

        # Extract parameters from kwargs
        null_strategy = kwargs.get("null_strategy", "PRESERVE")
        preserve_length = kwargs.get("preserve_length", False)
        mask_char = kwargs.get("mask_char", "*")
        fixed_length = kwargs.get("fixed_length", None)
        random_mask = kwargs.get("random_mask", False)
        mask_char_pool = kwargs.get("mask_char_pool", None)
        numeric_output = kwargs.get("numeric_output", "string")

        # Handle null values
        if pd.isna(value) and null_strategy == "PRESERVE":
            return value

        # Normalize and convert to string
        str_value = normalize_text(str(value))

        # Determine mask length
        if fixed_length:
            mask_length = fixed_length
        elif preserve_length:
            mask_length = len(str_value)
        else:
            mask_length = 8  # Default fixed length

        # Generate mask
        if random_mask:
            if mask_char_pool:
                masked = "".join(
                    random.choice(mask_char_pool) for _ in range(mask_length)
                )
            else:
                # Default pool of safe characters
                pool = string.ascii_letters + string.digits + "!@#$%^&*"
                masked = "".join(random.choice(pool) for _ in range(mask_length))
        else:
            masked = mask_char * mask_length

        # Handle type-specific output for numeric values
        try:
            is_numeric = isinstance(value, (int, float)) or (
                isinstance(value, str) and value.replace(".", "", 1).isdigit()
            )

            if is_numeric:
                if numeric_output == "numeric":
                    return cls._mask_to_numeric(masked, str_value)
                elif numeric_output == "preserve":
                    return value
        except:
            pass  # If any error in numeric detection, continue with string masking

        return masked

    @classmethod
    def process_batch_dask(cls, ddf: dd.DataFrame, **kwargs) -> dd.DataFrame:
        """
        Process Dask DataFrame with full masking.

        Args:
            ddf (dd.DataFrame): Input Dask DataFrame.
            **kwargs: Masking configuration parameters.

        Returns:
            dd.DataFrame: Processed Dask DataFrame.
        """

        # Define masking function for map_partitions
        def mask_partition(partition):
            return cls.process_batch(partition.copy(deep=True), **kwargs)

        # Apply masking to each partition
        result = ddf.map_partitions(mask_partition, meta=ddf._meta)

        return result

    def get_operation_summary(self) -> Dict[str, Any]:
        """
        Get summary of the masking operation.

        Returns:
            dict: Dictionary with operation summary.
        """

        masking_summary = {
            "field_name": self.field_name,
            "mask_character": self.mask_char,
            "preserve_length": self.preserve_length,
            "fixed_length": self.fixed_length,
            "random_mask": self.random_mask,
            "mask_char_pool": self.mask_char_pool,
            "preserve_format": self.preserve_format,
            "format_patterns": self.format_patterns,
            "numeric_output": self.numeric_output,
            "date_format": self.date_format,
            "format_patterns": self.format_patterns,
        }

        return masking_summary

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
            dict: Strategy-specific parameters for full masking operation.
        """

        params = dict(
            field_name=self.field_name,
            mask_char=self.mask_char,
            preserve_length=self.preserve_length,
            fixed_length=self.fixed_length,
            random_mask=self.random_mask,
            mask_char_pool=self.mask_char_pool,
            preserve_format=self.preserve_format,
            format_patterns=self.format_patterns,
            numeric_output=self.numeric_output,
            date_format=self.date_format,
            mode=self.mode,
            output_field_name=self.output_field_name,
            column_prefix=self.column_prefix,
            null_strategy=self.null_strategy,
            description=self.description,
            condition_field=self.condition_field,
            condition_values=self.condition_values,
            condition_operator=self.condition_operator,
            ka_risk_field=self.ka_risk_field,
            risk_threshold=self.risk_threshold,
            vulnerable_record_strategy=self.vulnerable_record_strategy,
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

    def _collect_comprehensive_metrics(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        processed_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Collect comprehensive metrics for the full masking operation.

        Args:
            original_data (pd.Series): Original data series.
            anonymized_data (pd.Series): Anonymized data series.
            processed_df (pd.DataFrame): The processed DataFrame.

        Returns:
            dict: Dictionary of metrics.
        """

        metrics = {}

        field_series = processed_df.get(self.output_field_name)
        operation_params = self.get_operation_summary()

        # Core operation metrics
        operation_metrics = collect_operation_metrics(
            operation_type="masking",
            original_data=original_data,
            processed_data=anonymized_data,
            operation_params=operation_params,
        )

        # Custom or strategy-specific metrics
        metrics = self._collect_specific_metrics(original_data, anonymized_data)

        # Suppression rate
        null_count = original_data.isna().sum()
        metrics["suppression_rate"] = (
            calculate_suppression_rate(field_series, null_count)
            if field_series is not None
            else 0.0
        )

        # Strategy distribution
        strategy_counts = (
            field_series.value_counts().to_dict() if field_series is not None else {}
        )

        metrics["masking_strategy_distribution"] = {
            strategy: strategy_counts.get(strategy, 0)
            for strategy in ["fixed", "pattern", "random", "words"]
        }

        # Merge core + extended metrics
        metrics.update(operation_metrics)

        return metrics

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect full masking specific metrics.

        Args:
            original_data (pd.Series): Original data series.
            anonymized_data (pd.Series): Anonymized data series.

        Returns:
            dict: Dictionary of specific metrics.
        """

        metrics = {}

        # Initialize full masking stats
        masked_count = 0
        format_preserved_count = 0
        conditional_mask_count = 0
        risk_based_mask_count = 0

        # Filter out NaN for pairwise comparison
        original = original_data.dropna()
        anonymized = anonymized_data.dropna()

        for idx, (orig, anon) in enumerate(zip(original, anonymized)):
            orig_str, anon_str = str(orig), str(anon)

            if orig_str == anon_str:
                continue  # Skip if no masking was applied

            masked_count += 1
            orig_len = len(orig_str)

            if orig_len == 0:
                continue

            # Format-preserved: check if preserve_format is True and mask differs only in masked chars
            if (
                self.preserve_format
                and hasattr(self, "format_patterns")
                and self.format_patterns
            ):
                for pattern_config in self.format_patterns.values():
                    if hasattr(pattern_config, "regex") and re.match(
                        pattern_config.regex, orig_str
                    ):
                        format_preserved_count += 1
                        break

            # Conditional mask: check if condition_field and index in filter_mask
            if (
                self.condition_field
                and hasattr(self, "filter_mask")
                and idx in getattr(self, "filter_mask", set())
            ):
                conditional_mask_count += 1

            # Risk-based mask: check if ka_risk_field and index in vulnerable indices
            if (
                self.ka_risk_field
                and hasattr(self, "vulnerable_indices")
                and idx in getattr(self, "vulnerable_indices", set())
            ):
                risk_based_mask_count += 1

        total_records = len(original_data)
        metrics.update(
            {
                "values_masked": masked_count,
                "masking_rate": (masked_count / total_records if total_records else 0),
                "format_preserved_count": format_preserved_count,
                "conditional_mask_count": conditional_mask_count,
                "risk_based_mask_count": risk_based_mask_count,
            }
        )

        return metrics

    def _generate_visualizations(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate visualizations using the core visualization utilities with thread-safe context support.

        This is a base implementation that provides a basic distribution comparison.
        Subclasses should override to provide operation-specific visualizations.

        Args:
            original_data (pd.Series): Original data before anonymization.
            anonymized_data (pd.Series): Anonymized data after processing.
            task_dir (Path): Task directory for saving visualizations.
            vis_theme (str, optional): Theme to use for visualizations.
            vis_backend (str, optional): Backend to use: "plotly" or "matplotlib".
            vis_strict (bool, optional): If True, raise exceptions for configuration errors.
            progress_tracker (Optional[HierarchicalProgressTracker]): Progress tracker for the operation.
            timestamp (str, optional): Timestamp for artifact naming.
            **kwargs: Additional parameters for the operation.

        Returns:
            dict: Dictionary with visualization types and paths.
        """

        visualization_paths = {}
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if visualization should be skipped
        if vis_backend is None:
            self.logger.info(
                f"Skipping visualization for {self.field_name} (backend=None)"
            )
            return visualization_paths

        self.logger.info(
            f"[VIZ] Starting visualization generation for {self.field_name} using {self.null_strategy} strategy"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            # Sample large datasets for visualization
            if len(original_data) > DEFAULT_SAMPLE_SIZE:
                self.logger.info(
                    f"[VIZ] Sampling large dataset: {len(original_data)} -> {DEFAULT_SAMPLE_SIZE} samples"
                )
                original_for_viz = sample_large_dataset(
                    original_data, max_samples=DEFAULT_SAMPLE_SIZE
                )
                anonymized_for_viz = sample_large_dataset(
                    anonymized_data, max_samples=DEFAULT_SAMPLE_SIZE
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

            # 1. Category distribution comparison
            dist_path = create_category_distribution_comparison(
                original_data=original_for_viz,
                anonymized_data=anonymized_for_viz,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=self.operation_name,
                max_categories=DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS,
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )
            if dist_path:
                visualization_paths["distribution"] = dist_path
                self.logger.info(
                    f"Created distribution visualization: {dist_path.name}"
                )

            # 2. General comparison (order 5 as per SRS)
            comp_path = create_comparison_visualization(
                original_data=original_for_viz,
                anonymized_data=anonymized_for_viz,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=self.operation_name,
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )
            if comp_path:
                visualization_paths["comparison"] = comp_path
                self.logger.info(f"Created comparison visualization: {comp_path.name}")

            self.logger.info(
                f"Generated {len(visualization_paths)} visualizations successfully"
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to generate visualizations: {e}", exc_info=True
            )

        return visualization_paths
