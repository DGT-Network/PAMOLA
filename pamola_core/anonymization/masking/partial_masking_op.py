"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking Operation
Package:       pamola_core.anonymization.masking
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
Updated:       2025-07-22
License:       BSD 3-Clause

Description:
This module implements the PartialMaskingOperation class for selective masking
of string fields while preserving specified portions of data. It supports multiple
masking strategies, including position-based, pattern-based, random percentage,
word boundary preserving, and preset-based masking.

Key Features:
- Position-based masking: preserve prefix/suffix or specific positions
- Pattern-based masking: use regex or predefined patterns for masking
- Random percentage masking: mask a configurable percentage of characters
- Word boundary preserving: mask words while maintaining structure
- Preset-based masking: leverage reusable masking presets for common data types
- Multi-field consistency: ensure consistent masking across related fields
- Format and separator preservation options
- Integration with k-anonymity and conditional masking
- Metrics and visualization support for privacy assessment

Framework:
Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting. Utilizes masking_patterns.py
and masking_presets.py for centralized pattern and preset management.
"""

from datetime import datetime
from pathlib import Path
import re
import random
import time
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import dask.dataframe as dd
import numpy as np
from pamola_core.anonymization.commons.validation.exceptions import (
    FieldTypeError,
    InvalidDataFormatError,
)
from pamola_core.common.constants import Constants
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
)
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_registry import register
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.text_processing_utils import (
    normalize_text,
)
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_suppression_rate,
    get_process_summary,
)
from pamola_core.anonymization.commons.visualization_utils import (
    create_category_distribution_comparison,
    create_comparison_visualization,
    sample_large_dataset,
)

# Import pattern library (assumed to exist)
from pamola_core.anonymization.commons.masking_patterns import (
    MaskingPatterns,
    apply_pattern_mask,
    generate_mask,
    generate_mask_char,
    is_separator,
    preserve_pattern_mask,
)
from pamola_core.anonymization.commons.masking_presets import (
    MaskingPresetManager,
    MaskingType,
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker

# Default values
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_TOP_CATEGORIES_FOR_ANALYSIS = 20


class PartialMaskingConfig(OperationConfig):
    """Configuration for PartialMaskingOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            "mask_char": {"type": "string", "default": "*"},
            "unmasked_prefix": {"type": "integer", "minimum": 0, "default": 0},
            "unmasked_suffix": {"type": "integer", "minimum": 0, "default": 0},
            "unmasked_positions": {
                "type": ["array", "null"],
                "items": {"type": "integer", "minimum": 0},
            },
            "preset_type": {"type": ["string", "null"]},
            "preset_name": {"type": ["string", "null"]},
            "pattern_type": {
                "type": ["string", "null"],
            },
            "mask_pattern": {"type": ["string", "null"]},
            "preserve_pattern": {"type": ["string", "null"]},
            "preserve_separators": {"type": "boolean", "default": True},
            "mask_percentage": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 100,
            },
            "mask_strategy": {
                "type": "string",
                "enum": [
                    MaskStrategyEnum.FIXED.value,
                    MaskStrategyEnum.PATTERN.value,
                    MaskStrategyEnum.RANDOM.value,
                    MaskStrategyEnum.WORDS.value,
                ],
                "default": "fixed",
            },
            "consistency_fields": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "case_sensitive": {"type": "boolean", "default": True},
            "preserve_word_boundaries": {"type": "boolean", "default": False},
            "random_mask": {"type": "boolean", "default": False},
            "mask_char_pool": {"type": ["string", "null"]},
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
        "required": ["field_name"],
    }


@register(version="1.0.0")
class PartialMaskingOperation(AnonymizationOperation):
    """
    Partial masking operation that selectively masks portions of data
    while preserving specified parts for utility.

    Supports position-based and pattern-based masking strategies.
    """

    def __init__(
        self,
        # ==== Required ====
        field_name: str,
        # ==== Output & Replacement ====
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "masked_",
        null_strategy: str = "PRESERVE",
        description: str = "",
        # ==== Masking Basics ====
        mask_char: str = "*",
        mask_strategy: str = MaskStrategyEnum.FIXED.value,  # fixed, pattern, random, words
        mask_percentage: Optional[float] = None,  # Random % to mask
        # ==== Position-based Masking ====
        unmasked_prefix: int = 0,
        unmasked_suffix: int = 0,
        unmasked_positions: Optional[List[int]] = None,
        # ==== Pattern-based Masking ====
        pattern_type: Optional[str] = None,
        mask_pattern: Optional[str] = None,
        preserve_pattern: Optional[str] = None,
        # ==== Format & Word Preservation ====
        preserve_separators: bool = True,
        preserve_word_boundaries: bool = False,
        # ==== Advanced Masking Behavior ====
        case_sensitive: bool = True,
        random_mask: bool = False,
        mask_char_pool: Optional[str] = None,
        # ==== Preset / Templates ====
        preset_type: Optional[str] = None,
        preset_name: Optional[str] = None,
        # ==== Multi-field Consistency ====
        consistency_fields: Optional[List[str]] = None,
        # ==== Conditional Masking ====
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: str = "in",
        # ==== K-anonymity Integration ====
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        vulnerable_record_strategy: str = "mask",  # or "full_mask"
        # ==== System Settings ====
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        use_cache: bool = True,
        # ==== Security ====
        use_encryption: bool = False,
        encryption_mode: Optional[str] = "none",
        encryption_key: Optional[Union[str, Path]] = None,
        # ==== Visualization ====
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        output_format: str = "csv",
    ):
        """
        Initialize Partial masking operation.

        Parameters:
        -----------
        field_name: str
            Name of the field to mask
        mask_char: str
            Character used for masking (default: '*')
        unmasked_prefix: int
            Number of characters to leave unmasked at the start
        unmasked_suffix: int
            Number of characters to leave unmasked at the end
        unmasked_positions: Optional[List[int]]
            Specific positions to leave unmasked (default: None)
        pattern_type: Optional[str]
            Type of masking pattern to use (e.g., 'email', 'phone')
        mask_pattern: Optional[str]
            Custom regex pattern for masking
        preserve_pattern: Optional[str]
            Regex pattern to preserve (mask everything else)
        preserve_separators: bool
            Whether to preserve separators (e.g., '-', '_', '.')
        mask_percentage: Optional[float]
            Percentage of characters to mask randomly (0-100)
        mask_strategy: str
            Strategy for masking ('fixed', 'pattern', 'random', 'words')
        consistency_fields: Optional[List[str]]
            Additional fields to mask consistently with the main field
        case_sensitive: bool
            Whether masking is case-sensitive (default: True)
        preserve_word_boundaries: bool
            Whether to preserve word boundaries during masking
        random_mask: bool
            Whether to use random characters from a pool for masking
        mask_char_pool: Optional[str]
            Pool of characters to use for random masking (default: alphanumeric)
        Other parameters follow base class convention
        """
        # Set default description if missing
        description = (
            description
            or f"Partial masking operation for '{field_name}' with mask character '{mask_char}'"
        )

        # Group parameters into a config dict
        config_params = dict(
            field_name=field_name,
            mask_char=mask_char,
            preset_type=preset_type,
            preset_name=preset_name,
            unmasked_prefix=unmasked_prefix,
            unmasked_suffix=unmasked_suffix,
            unmasked_positions=unmasked_positions,
            pattern_type=pattern_type,
            mask_pattern=mask_pattern,
            preserve_pattern=preserve_pattern,
            preserve_separators=preserve_separators,
            mask_percentage=mask_percentage,
            mask_strategy=mask_strategy,
            consistency_fields=consistency_fields or [],
            case_sensitive=case_sensitive,
            preserve_word_boundaries=preserve_word_boundaries,
            random_mask=random_mask,
            mask_char_pool=mask_char_pool,
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
        config = PartialMaskingConfig(**config_params)

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

        if mask_strategy == MaskStrategyEnum.PATTERN.value and pattern_type:
            self._pattern_config = MaskingPatterns.get_pattern(pattern_type)
            self.process_kwargs["pattern_config"] = self._pattern_config

        if preset_type and preset_name:
            self.process_kwargs["manager"] = MaskingPresetManager()

        # Initialize additional attributes
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
                            "step": "Starting Partial masking operation",
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
                                f"Partial masking operation of {self.field_name} (cached)",
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
                        kwargs, PartialMaskingOperation._save_output_data
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
                    f"Partial masking operation of {self.field_name} completed",
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
            self.logger.error(f"Error in Partial masking operation: {str(e)}")
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                exception=e,
            )

    def _validate_configuration(self) -> None:
        """Validate partial masking configuration."""

        # --- Mask character ---
        if not self.mask_char or len(self.mask_char) != 1:
            raise ValueError("mask_char must be a single character")

        # --- Prefix/suffix: validate non-negative numeric ---
        if self.unmasked_prefix is not None:
            if not isinstance(self.unmasked_prefix, int) or self.unmasked_prefix < 0:
                raise FieldTypeError(
                    field_name="unmasked_prefix",
                    expected_type="non-negative int",
                    actual_type=type(self.unmasked_prefix).__name__,
                )

        if self.unmasked_suffix is not None:
            if not isinstance(self.unmasked_suffix, int) or self.unmasked_suffix < 0:
                raise FieldTypeError(
                    field_name="unmasked_suffix",
                    expected_type="non-negative int",
                    actual_type=type(self.unmasked_suffix).__name__,
                )

        if self.mask_pattern:
            self._validate_pattern(self.mask_pattern, "mask_pattern")

        if self.preserve_pattern:
            self._validate_pattern(self.preserve_pattern, "preserve_pattern")

        # --- Pattern type ---
        if (
            self.mask_strategy == MaskStrategyEnum.PATTERN.value
            and self.pattern_type
            and not self._pattern_config
        ):
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")

        # --- Masking strategy ---
        valid_strategies = [
            MaskStrategyEnum.FIXED.value,
            MaskStrategyEnum.PATTERN.value,
            MaskStrategyEnum.RANDOM.value,
            MaskStrategyEnum.WORDS.value,
        ]
        if self.mask_strategy not in valid_strategies:
            raise ValueError(f"mask_strategy must be one of {valid_strategies}")

        # --- Consistency fields ---
        if self.consistency_fields is not None and not isinstance(
            self.consistency_fields, list
        ):
            raise FieldTypeError(
                field_name="consistency_fields",
                expected_type="list",
                actual_type=type(self.consistency_fields).__name__,
            )

        self.logger.info("Partial masking configuration validation passed")

    def _validate_pattern(self, pattern_value: str, field_name: str):
        """
        Validate that a given string is a valid regular expression pattern.

        Parameters
        ----------
        pattern_value : str
            The regex pattern to validate.

        field_name : str
            The name of the field associated with the pattern (used in error messages).

        Raises
        ------
        InvalidDataFormatError
            If the pattern is not a valid regex.
        """
        try:
            re.compile(pattern_value)
        except re.error:
            raise InvalidDataFormatError(
                field_name=field_name,
                data_type="pattern",
                format_description="Valid regex pattern required",
                sample_invalid=[pattern_value],
            )

    @classmethod
    def process_batch(cls, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of data to partial mask values.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch with partial masked values
        """
        # Extract parameters from kwargs
        field_name = kwargs.get("field_name", "field")
        output_field_name = kwargs.get("output_field_name", "masked")
        mode = kwargs.get("mode")
        consistency_fields = kwargs.get("consistency_fields")

        if field_name not in batch.columns:
            raise ValueError(f"Field '{field_name}' not found in DataFrame.")

        # Create consistency map if needed
        consistency_map = None
        if consistency_fields:
            consistency_map = cls._create_consistency_map(batch, **kwargs)

        # --- Apply masking to main field ---
        def get_masked_value(val):
            if pd.isna(val):
                return val
            val_str = str(val).strip()
            if consistency_map:
                return consistency_map.get(val_str, val)
            return cls._apply_partial_mask(val, **kwargs)

        if mode == "REPLACE":
            batch[field_name] = batch[field_name].apply(get_masked_value)
        else:
            batch[output_field_name] = batch[field_name].apply(get_masked_value)

        # --- Apply masking to consistency fields ---
        if consistency_fields:
            for field in consistency_fields:
                if field in batch.columns and field != field_name:
                    masked_column_name = f"masked_{field}"
                    batch[masked_column_name] = batch[field].apply(
                        lambda val: (
                            consistency_map.get(str(val).strip(), val)
                            if pd.notna(val)
                            else val
                        )
                    )

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
    def _apply_partial_mask(cls, value: Any, **kwargs) -> Union[str, Any]:
        """
        Apply partial masking to a given value based on the configured strategy or preset.

        Parameters
        ----------
        value : Any
            The value to be masked. Typically a string, but may be other types
            (e.g., numeric, None) which will be converted to string or returned unmodified.
        kwargs : Any
            Additional keyword arguments for processing

        Returns
        -------
        Union[str, Any]
            The masked string if a masking strategy is applied, or the original value
            if no masking is applicable (e.g., value is NaN or None).
        """
        # Extract parameters from kwargs
        preset_type = kwargs.get("preset_type", None)
        preset_name = kwargs.get("preset_name", None)
        case_sensitive = kwargs.get("case_sensitive", False)
        mask_strategy = kwargs.get("mask_strategy", MaskStrategyEnum.FIXED.value)

        if pd.isna(value) or value is None:
            return value

        str_value = str(value)

        # Try preset masking
        if preset_type and preset_name:
            masked = cls._apply_preset_masking(str_value, **kwargs)
            if masked is not None:
                return masked

        # Normalize if case-insensitive
        normalized_value = (
            normalize_text(str_value) if not case_sensitive else str_value
        )

        # Mapping strategy to function
        strategy_map = {
            MaskStrategyEnum.FIXED.value: cls._position_based_mask,
            MaskStrategyEnum.PATTERN.value: cls._pattern_based_mask,
            MaskStrategyEnum.RANDOM.value: cls._random_percentage_mask,
            MaskStrategyEnum.WORDS.value: cls._word_based_mask,
        }

        mask_func = strategy_map.get(mask_strategy)
        if mask_func:
            return mask_func(normalized_value, **kwargs)

        return str_value

    @staticmethod
    def _apply_preset_masking(value: str, **kwargs) -> Optional[str]:
        """
        Apply masking using a predefined preset configuration.

        Parameters
        ----------
        value : str
            The string value to be masked using a preset.
        kwargs : Any
            Additional keyword arguments for processing

        Returns
        -------
        Optional[str]
            Masked string if preset is applied successfully; otherwise, None.
        """
        try:
            # Extract parameters from kwargs
            preset_type = kwargs.get("preset_type", None)
            preset_name = kwargs.get("preset_name", None)
            random_mask = kwargs.get("random_mask", False)
            if preset_type and preset_name:
                manager = kwargs.get("manager", MaskingPresetManager())
                mtype = MaskingType(preset_type.lower())
                return manager.apply_masking(
                    value, mtype, preset_name.upper(), random_mask=random_mask
                )
            return value
        except (ValueError, KeyError) as e:
            return None

    @staticmethod
    def _random_percentage_mask(value: str, **kwargs) -> str:
        """
        Randomly mask a percentage of characters in the input string.

        Returns
        -------
        str
            The string with a random subset of characters replaced by mask_char or random characters.
        kwargs : Any
            Additional keyword arguments for processing

        """
        # Extract parameters from kwargs
        mask_char = kwargs.get("mask_char", "*")
        random_mask = kwargs.get("random_mask", False)
        mask_char_pool = kwargs.get("mask_char_pool", None)
        mask_percentage = kwargs.get("mask_percentage", None)

        if not value or not mask_percentage:
            return value

        value_len = len(value)
        num_to_mask = max(0, min(int(value_len * mask_percentage), value_len))

        if num_to_mask == 0:
            return value

        positions_to_mask = set(random.sample(range(value_len), num_to_mask))

        masked_chars = []
        for i, ch in enumerate(value):
            if i in positions_to_mask:
                # Generate 1-char mask using the internal logic
                masked_char = generate_mask_char(mask_char, random_mask, mask_char_pool)
                masked_chars.append(masked_char)
            else:
                masked_chars.append(ch)

        return "".join(masked_chars)

    @classmethod
    def _word_based_mask(cls, value: str, **kwargs) -> str:
        """
        Apply masking while preserving word boundaries.

        If `preserve_word_boundaries` is False, falls back to `_apply_partial_mask`.
        For each word:
            - If length <= 3: mask all characters.
            - If length > 3: apply position-based partial mask.

        Parameters
        ----------
        value : str
            The input string to be masked.
        kwargs : Any
            Additional keyword arguments for processing

        Returns
        -------
        str
            The masked string with word boundaries preserved.
        """
        # Extract parameters from kwargs
        mask_char = kwargs.get("mask_char", "*")
        preserve_word_boundaries = kwargs.get("preserve_word_boundaries", False)

        if not preserve_word_boundaries:
            return cls._position_based_mask(value, **kwargs)

        # Mask each word individually
        def _mask_word(word: str) -> str:
            if len(word) <= 3:
                return mask_char * len(word)
            return cls._position_based_mask(word, **kwargs)

        words = value.split()
        masked_words = [_mask_word(word) for word in words]
        return " ".join(masked_words)

    @staticmethod
    def _pattern_based_mask(value: str, **kwargs) -> str:
        """
        Apply pattern-based masking to a string value.

        Supports 3 modes:
        1. If `pattern_type` and `pattern_config` are set: use predefined masking patterns.
        2. If `mask_pattern` is set: apply regex-based masking.
        3. If `preserve_pattern` is set: preserve matching parts, mask others.

        Parameters
        ----------
        value : str
            Input string to be masked.
        kwargs : Any
            Additional keyword arguments for processing

        Returns
        -------
        str
            Masked string based on the matching pattern.
            Returns original string if no pattern is configured.
        """
        # Extract parameters from kwargs
        mask_char = kwargs.get("mask_char", "*")
        random_mask = kwargs.get("random_mask", False)
        mask_char_pool = kwargs.get("mask_char_pool", None)
        mask_pattern = kwargs.get("mask_pattern", None)
        preserve_pattern = kwargs.get("preserve_pattern", None)
        preserve_separators = kwargs.get("preserve_separators", True)
        pattern_type = kwargs.get("pattern_type", None)
        pattern_config = kwargs.get("pattern_config", None)

        if pattern_type and pattern_config:
            return apply_pattern_mask(value, pattern_config, mask_char)

        if mask_pattern:
            return re.sub(
                mask_pattern,
                lambda m: generate_mask(
                    mask_char, random_mask, mask_char_pool, len(m.group())
                ),
                value,
            )

        if preserve_pattern:
            return preserve_pattern_mask(
                value,
                mask_char,
                random_mask,
                mask_char_pool,
                preserve_pattern,
                preserve_separators,
            )

        return value

    @staticmethod
    def _position_based_mask(value: str, **kwargs) -> str:
        """
        Apply masking to a string based on character positions.

        Supports two modes:
        1. Explicit `unmasked_positions`: Preserves specified character positions.
        2. Prefix/Suffix masking: Preserves the beginning and/or end of the string.

        If `preserve_separators` is True, common separators (e.g., '-', '_', '.', etc.)
        are excluded from masking.

        Parameters
        ----------
        value : str
            The input string to mask.
        kwargs : Any
            Additional keyword arguments for processing

        Returns
        -------
        str
            The masked string with selected positions preserved.
        """
        # Extract parameters from kwargs
        mask_char = kwargs.get("mask_char", "*")
        random_mask = kwargs.get("random_mask", False)
        mask_char_pool = kwargs.get("mask_char_pool", None)
        unmasked_positions = kwargs.get("unmasked_positions", None)
        preserve_separators = kwargs.get("preserve_separators", True)
        unmasked_prefix = kwargs.get("unmasked_prefix", 0)
        unmasked_suffix = kwargs.get("unmasked_suffix", 0)
        value_len = len(value)

        # Strategy 1: Use explicit unmasked positions
        if unmasked_positions:
            masked = list(
                generate_mask(mask_char, random_mask, mask_char_pool, value_len)
            )
            for pos in unmasked_positions:
                if 0 <= pos < value_len:
                    masked[pos] = value[pos]

            if preserve_separators:
                for i, ch in enumerate(value):
                    if is_separator(ch):
                        masked[i] = ch

            return "".join(masked)

        # Strategy 2: Use prefix/suffix preservation
        if unmasked_prefix + unmasked_suffix >= value_len:
            return value  # nothing to mask

        result = list(value)
        mask_start = unmasked_prefix
        mask_end = value_len - unmasked_suffix

        for i in range(mask_start, mask_end):
            if preserve_separators and is_separator(result[i]):
                continue
            result[i] = generate_mask_char(mask_char, random_mask, mask_char_pool)

        return "".join(result)

    @classmethod
    def _create_consistency_map(cls, batch: pd.DataFrame, **kwargs) -> Dict[str, str]:
        """
        Create a map from original values to their consistently masked versions
        across `self.field_name` and `self.consistency_fields`.

        Parameters
        ----------
        batch : pd.DataFrame
            The batch of data containing fields to process.
        kwargs : dict
            Additional keyword arguments for processing

        Returns
        -------
        Dict[str, str]
            A dictionary mapping original string values to masked values.
        """
        # Extract parameters from kwargs
        field_name = kwargs.get("field_name", "field")
        random_mask = kwargs.get("random_mask", False)
        consistency_fields = kwargs.get("consistency_fields", [])
        # Step 1: Collect all unique values across specified fields
        all_values = {
            str(val).strip()
            for field in [field_name] + consistency_fields
            if field in batch.columns
            for val in batch[field].dropna().unique()
        }

        # Step 2: Create consistent masked mapping
        consistency_map = {}
        used_masks = set()
        MAX_RETRIES = 10

        for value in all_values:
            retries = 0
            masked = cls._apply_partial_mask(value, **kwargs)

            if random_mask:
                while masked in used_masks and retries < MAX_RETRIES:
                    retries += 1
                    masked = cls._apply_partial_mask(value, **kwargs)

                if retries >= MAX_RETRIES:
                    masked = f"MASKED_{hash(value) % 100000}"  # simple fallback or use original value

            consistency_map[value] = masked
            used_masks.add(masked)

        return consistency_map

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """Collect partial masking specific metrics."""
        metrics = {}

        # Initialize partial masking stats
        visibility_scores = []
        masked_count = 0

        # Filter out NaN for pairwise comparison
        original = original_data.dropna()
        anonymized = anonymized_data.dropna()

        for orig, anon in zip(original, anonymized):
            orig_str, anon_str = str(orig), str(anon)

            if orig_str == anon_str:
                continue  # Skip if no masking was applied

            masked_count += 1
            orig_len = len(orig_str)

            if orig_len == 0:
                continue

            visible_chars = sum(
                1
                for i, c in enumerate(anon_str)
                if i < orig_len and c != self.mask_char
            )
            visibility_scores.append(visible_chars / orig_len)

        total_records = len(original_data)
        metrics.update(
            {
                "partial_mask_rate": (
                    masked_count / total_records if total_records else 0
                ),
                "average_visibility": (
                    np.mean(visibility_scores) if visibility_scores else 0
                ),
                "values_masked": masked_count,
                "mask_strategy": self.mask_strategy,
                "pattern_type": self.pattern_type,
                "consistency_fields_count": (
                    len(self.consistency_fields) if self.consistency_fields else 0
                ),
                "preserve_separators": self.preserve_separators,
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
            Reporter object for tracking artifacts
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
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

    def get_operation_summary(self) -> Dict[str, Any]:
        """
        Get summary of the masking operation.

        Returns:
            Dictionary with operation summary
        """
        masking_summary = {
            "field_name": self.field_name,
            "mask_strategy": self.mask_strategy,
            "mask_char": self.mask_char,
            "pattern_type": self.pattern_type,
            "unmasked_prefix": self.unmasked_prefix,
            "unmasked_suffix": self.unmasked_suffix,
            "consistency_fields": self.consistency_fields,
        }
        return masking_summary

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for partial masking operation
        """
        params = dict(
            field_name=self.field_name,
            mode=self.mode,
            output_field_name=self.output_field_name,
            column_prefix=self.column_prefix,
            null_strategy=self.null_strategy,
            description=self.description,
            mask_char=self.mask_char,
            mask_strategy=self.mask_strategy,  # fixed, pattern, random, words
            mask_percentage=self.mask_percentage,  # Random % to mask
            unmasked_prefix=self.unmasked_prefix,
            unmasked_suffix=self.unmasked_suffix,
            unmasked_positions=self.unmasked_positions,
            pattern_type=self.pattern_type,
            mask_pattern=self.mask_pattern,
            preserve_pattern=self.preserve_pattern,
            preserve_separators=self.preserve_separators,
            preserve_word_boundaries=self.preserve_word_boundaries,
            case_sensitive=self.case_sensitive,
            random_mask=self.random_mask,
            mask_char_pool=self.mask_char_pool,
            preset_type=self.preset_type,
            preset_name=self.preset_name,
            consistency_fields=self.consistency_fields,
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
        Collect comprehensive metrics for the partial masking operation.

        Args:
            original_data: Original data series
            anonymized_data: Anonymized data series
            processed_df: The processed DataFrame

        Returns:
            Dictionary of metrics
        """
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

        # Consistency fields count
        metrics["consistency_fields_processed"] = (
            (len(self.consistency_fields) if self.consistency_fields else 0),
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
