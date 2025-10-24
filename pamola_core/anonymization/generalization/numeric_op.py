"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Generalization Operation
Package:       pamola_core.anonymization.generalization
Version:       3.0.2
Status:        stable
Author:        PAMOLA Core Team
Created:       2024
Revised:       2025
License:       BSD 3-Clause

Description:
This module provides an operation for generalizing numeric fields to enhance privacy
while maintaining data utility. It implements various strategies:
1. Binning: Groups numeric values into discrete intervals (bins)
2. Rounding: Reduces precision by rounding to specified decimal places
3. Range-based: Maps values to custom ranges with special handling for outliers

Key Features:
- Direct in-place DataFrame modification with both REPLACE and ENRICH modes
- Robust null value handling with configurable strategies (PRESERVE, EXCLUDE, ERROR, ANONYMIZE)
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for distribution comparisons
- Chunked processing support for large datasets with progress tracking
- Risk-based processing using k-anonymity scores from profiling
- Memory-efficient operation with explicit cleanup for large datasets
- Thread-safe visualization generation with context isolation
- Support for multiple file formats (CSV, Parquet, Arrow)
- Integration with DataWriter for standardized file operations
- Privacy-aware metrics collection and reporting
- Adaptive processing based on data characteristics

Framework:
Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.

Change Log:
v3.0.2 (2025-01-20):
   - Fixed available_memory calculation with proper psutil integration
   - Optimized _apply_range() using pandas.cut for multiple ranges
   - Enhanced progress tracking with detailed phase information
   - Added configuration validation method
   - Added memory cleanup for large chunkes
v3.0.1 (2025-01-20):
   - Fixed all DataWriter integration issues
   - Corrected get_memory_usage() function calls
   - Fixed vulnerable_mask type checking for .sum()
   - Fixed OperationStatus.PARTIAL_SUCCESS reference
   - Improved error handling and type safety
v3.0.0 (2025-01-20):
   - Complete refactoring based on architecture review feedback
   - Delegated all metrics to commons utilities
   - Fixed risk-based processing to avoid double processing
   - Unified output field handling throughout
   - Added support for multiple ranges and quantile binning
   - Improved memory management and adaptive chunk sizing
   - Full delegation of I/O operations to DataWriter
   - Added comprehensive privacy metrics reporting
   - Cleaned up unused imports and code
v2.0.0 (2025-01-15):
   - Complete refactoring following architecture review
   - Full integration with DataWriter for all file operations
   - Added privacy-aware metrics from privacy_metric_utils
   - Implemented risk-based processing with vulnerable record handling
   - Added progress tracking for all operations
   - Fixed all import issues and unused dependencies
   - Enhanced visualization with proper artifact registration
   - Improved memory management with framework utilities
v1.4.0 (2024-12-16):
   - Major refactoring for new commons architecture
   - Integration with enhanced base_anonymization_op.py
   - Added support for conditional processing
v1.3.0 (2024-11-20):
   - Added chunked processing for large datasets
   - Enhanced null value handling strategies
v1.2.0 (2024-10-15):
   - Added ENRICH mode support
   - Enhanced visualization generation
v1.1.0 (2024-09-10):
   - Added caching support
   - Performance optimizations
v1.0.0 (2024-08-01):
   - Initial implementation

TODO:
- Add differential privacy noise calibration
- Support for custom binning edges from external configuration
- GPU acceleration for large numeric datasets
- Streaming processing for continuous data
- Integration with privacy budget tracking
- Machine learning-based adaptive binning strategies
"""

from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.categorical_config import NullStrategy
from pamola_core.anonymization.commons.metric_utils import (
    calculate_anonymization_effectiveness,
    collect_operation_metrics,
)
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_batch_metrics,
    get_process_summary,
    calculate_simple_disclosure_risk,
    calculate_suppression_rate,
)
from pamola_core.anonymization.commons.validation.exceptions import FieldValueError
from pamola_core.anonymization.commons.validation_utils import (
    validate_numeric_field,
    validate_bin_count,
    validate_precision,
    validate_range_limits,
)
from pamola_core.anonymization.commons.visualization_utils import (
    create_metric_visualization,
    create_comparison_visualization,
    create_metrics_overview_visualization,
    sample_large_dataset,
)
from pamola_core.anonymization.schemas.numeric_op_config import NumericGeneralizationConfig
from pamola_core.common.constants import Constants
from pamola_core.common.helpers.data_helper import DataHelper
from pamola_core.utils.io import load_settings_operation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.helpers import filter_used_kwargs

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
        strategy: str = "binning",
        bin_count: int = 10,
        binning_method: str = "equal_width",
        precision: int = 0,
        range_limits: Optional[List[Tuple[float, float]]] = None,
        quasi_identifiers: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize numeric generalization operation.

        Parameters
        ----------
        field_name : str
            Name of the numeric field to generalize.
        strategy : str, optional
            Generalization strategy: 'binning', 'rounding', or 'range'.
        bin_count : int, optional
            Number of bins for 'binning' (default=10).
        binning_method : str, optional
            Binning method: 'equal_width', 'equal_frequency', or 'custom'.
        precision : int, optional
            Decimal precision for 'rounding' (default=0).
        range_limits : list of tuple, optional
            Range limits for 'range' strategy.
        quasi_identifiers : list of str, optional
            Optional QI list for risk evaluation.
        **kwargs
            Additional keyword arguments passed to AnonymizationOperation.
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Numeric generalization for '{field_name}' using {strategy} strategy",
        )

        # Normalize range limits (custom helper for schema validation)
        range_limits = self.convert_range_limits_for_schema(range_limits)

        # Build config object (if used for schema/validation)
        config = NumericGeneralizationConfig(
            field_name=field_name,
            strategy=strategy,
            bin_count=bin_count,
            binning_method=binning_method,
            precision=precision,
            range_limits=range_limits,
            quasi_identifiers=quasi_identifiers,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize base AnonymizationOperation
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

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
        Execute the numeric generalization operation.

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
            # Initialize timing and result
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(
                f"Starting {self.operation_name} operation at {self.start_time}"
            )

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

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

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
                            "step": "Starting numeric generalization",
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
                                f"Numeric generalization of {self.field_name} (cached)",
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

                # Validate field is suitable for numeric operations
                validation_result = validate_numeric_field(
                    df,
                    self.field_name,
                    allow_null=(self.null_strategy != NullStrategy.ERROR.value),
                    logger_instance=self.logger,
                )

                if not validation_result.is_valid:
                    raise FieldValueError(
                        self.field_name,
                        reason="Invalid numeric format",
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
                # Normalize integer dtype if required
                df[self.field_name] = DataHelper.normalize_int_dtype_vectorized(
                    df[self.field_name], safe_mode=False
                )

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

                # Process the filtered data only if not empty
                if not filtered_df.empty:
                    processed_df = self._process_data_with_config(
                        df=filtered_df,
                        progress_tracker=data_tracker,
                    )
                else:
                    self.logger.warning(
                        "Filtered DataFrame is empty. Skipping _process_data_with_config."
                    )
                    processed_df = df.copy(deep=True)
                    processed_df[self.output_field_name] = original_data

                # Handle vulnerable records if k-anonymity is enabled
                if self.ka_risk_field and self.ka_risk_field in df.columns:
                    processed_df = self._handle_vulnerable_records(
                        processed_df, self.output_field_name
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
                metrics_file_name = f"{self.field_name}_anonymization_numeric_{self.strategy}_metrics_{operation_timestamp}"

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
                        operation_metrics=metrics,
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
                        kwargs, NumericGeneralizationOperation._save_output_data
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
                    f"Numeric generalization of {self.field_name} completed",
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
            self.logger.error(f"Error in numeric generalization: {str(e)}")
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                exception=e,
            )

    @classmethod
    def process_batch(cls, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of data to generalize numeric values.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch with generalized values
        """
        # Extract parameters from kwargs
        field_name = kwargs.get("field_name")
        output_field_name = kwargs.get("output_field_name")
        mode = kwargs.get("mode")
        strategy = kwargs.get("strategy")

        # Check if the field exists
        if field_name not in batch.columns:
            raise ValueError(f"Field {field_name} not found in DataFrame")

        # Handle the case where the field might already be processed (non-numeric)
        field_data = batch[field_name]

        # Check if field appears to be numeric
        if not pd.api.types.is_numeric_dtype(field_data):
            # If this is ENRICH mode, just copy the values
            if mode == "ENRICH":
                batch[output_field_name] = field_data

            return batch

        # Apply generalization based on strategy
        if strategy == "binning":
            generalized_values = cls._apply_binning(field_data, **kwargs)
        elif strategy == "rounding":
            generalized_values = cls._apply_rounding(field_data, **kwargs)
        elif strategy == "range":
            generalized_values = cls._apply_range(field_data, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Update the DataFrame
        if mode == "REPLACE":
            batch[field_name] = generalized_values
        else:  # ENRICH
            batch[output_field_name] = generalized_values

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

    def _validate_configuration(self):
        """Validate operation configuration before execution."""
        # Strategy-specific validation
        if self.strategy == "binning":
            if self.bin_count < 2:
                raise ValueError("bin_count must be at least 2")
            if self.binning_method not in [
                "equal_width",
                "equal_frequency",
                "quantile",
            ]:
                raise ValueError(f"Unknown binning method: {self.binning_method}")
        elif self.strategy == "rounding":
            if not isinstance(self.precision, int):
                raise ValueError("precision must be an integer")
        elif self.strategy == "range":
            if not self.range_limits:
                raise ValueError("range_limits must be provided for range strategy")
            for i, (min_val, max_val) in enumerate(self.range_limits):
                if min_val >= max_val:
                    raise ValueError(
                        f"Invalid range at index {i}: min ({min_val}) >= max ({max_val})"
                    )

        # Output format validation
        if self.output_format not in ["csv", "parquet", "arrow"]:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        # Condition operator validation
        if self.condition_operator not in ["in", "not_in", "gt", "lt", "eq", "range"]:
            raise ValueError(f"Unknown condition operator: {self.condition_operator}")

    @staticmethod
    def _apply_binning(series: pd.Series, **kwargs) -> pd.Series:
        """
        Apply binning with specific parameters.

        Parameters:
        -----------
        series : pd.Series
            Series to generalize
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        pd.Series
            Binned series
        """
        # Extract parameters from kwargs
        bin_count = kwargs.get("bin_count")
        binning_method = kwargs.get("binning_method")

        # Validate bin count
        validate_bin_count(bin_count)

        # Prepare non-null values
        non_null_mask = ~series.isna()
        non_null_values = series[non_null_mask]

        if len(non_null_values) == 0:
            return series

        # Check if original series is integer
        is_int = pd.api.types.is_integer_dtype(series) or (
            pd.api.types.is_float_dtype(series) and (series.dropna() % 1 == 0).all()
        )

        # Initialize bins
        bins = None

        # Generate bin edges
        if binning_method == "equal_width":
            min_val = non_null_values.min()
            max_val = non_null_values.max()

            if min_val == max_val:
                # Single-value edge case
                label = f"{int(min_val)}" if is_int else f"{min_val:.2f}"
                result = series.copy().astype("object")
                result[non_null_mask] = label
                return result

            bins = np.linspace(min_val, max_val, bin_count + 1)
            bins[-1] += 0.001

        elif binning_method in ["equal_frequency", "quantile"]:
            quantiles = np.linspace(0, 1, bin_count + 1)
            if binning_method == "quantile" and bin_count == 4:
                quantiles = [0, 0.25, 0.5, 0.75, 1.0]
            bins = non_null_values.quantile(quantiles).values
            bins = np.unique(bins)

        else:
            raise ValueError(f"Unknown binning method: {binning_method}")

        # Fallback if binning failed
        if bins is None or len(bins) <= 1:
            return series

        # Format labels depending on type
        if is_int:
            labels = [
                f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
            ]
        else:
            labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

        # Apply pd.cut
        result = series.copy().astype("object")
        binned_values = pd.cut(
            non_null_values,
            bins=bins,
            labels=labels,
            include_lowest=True,
            ordered=False,
        )
        result[non_null_mask] = binned_values

        return result

    @staticmethod
    def _apply_rounding(series: pd.Series, **kwargs) -> pd.Series:
        """
        Apply rounding with specific precision.

        Parameters:
        -----------
        series : pd.Series
            Series to round
        kwargs : dict
            Decimal places (positive) or power of 10 (negative)

        Returns:
        --------
        pd.Series
            Rounded series
        """
        # Extract parameters from kwargs
        precision = kwargs.get("precision")

        # Validate precision
        validate_precision(precision)

        # Check if original series is integer
        is_int = pd.api.types.is_integer_dtype(series) or (
            pd.api.types.is_float_dtype(series) and (series.dropna() % 1 == 0).all()
        )

        if precision >= 0:
            # Round to decimal places
            result = series.round(precision)
            return result.astype("Int64") if is_int and precision == 0 else result
        else:
            # Round to nearest 10^|precision|
            factor = 10 ** abs(precision)
            result = (series / factor).round() * factor
            return result.astype("Int64") if is_int else result

    @staticmethod
    def _apply_range(series: pd.Series, **kwargs) -> pd.Series:
        """
        Apply range-based generalization to a series.

        Parameters:
        -----------
        series : pd.Series
            Series to generalize
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        pd.Series
            Range-generalized series
        """
        # Extract parameters from kwargs
        range_limits = kwargs.get("range_limits")

        # Validate range limits
        validate_range_limits(range_limits)

        # Handle empty series or no range limits
        if len(series) == 0 or not range_limits:
            return series

        # Check if original series is integer
        is_int = pd.api.types.is_integer_dtype(series) or (
            pd.api.types.is_float_dtype(series) and (series.dropna() % 1 == 0).all()
        )

        def fmt(val):
            return f"{int(val)}" if is_int else f"{val:.2f}"

        # Multiple ranges
        if len(range_limits) > 1:
            # Create bins from range limits
            bins = []
            labels = []

            # Lower bound
            min_val = range_limits[0][0]
            bins.append(float("-inf"))
            labels.append(f"<{fmt(min_val)}")

            # Main ranges
            for i, (range_min, range_max) in enumerate(range_limits):
                bins.append(range_min)
                if i < len(range_limits) - 1:
                    # Not the last range
                    labels.append(f"{fmt(range_min)}-{fmt(range_max)}")
                else:
                    # Last range includes upper bound
                    bins.append(range_max)
                    labels.append(f"{fmt(range_min)}-{fmt(range_max)}")
                    labels.append(f">={fmt(range_max)}")

            # Ensure final +inf
            if bins[-1] != float("inf"):
                bins.append(float("inf"))

            # Deduplicate bins
            seen = set()
            unique_bins = []
            for b in bins:
                if b not in seen:
                    seen.add(b)
                    unique_bins.append(b)
            bins = unique_bins

            # Apply pandas.cut for efficient categorization
            result = pd.cut(
                series,
                bins=bins,
                labels=labels[: len(bins) - 1],
                include_lowest=True,
            )
            return result.astype("object")

        else:
            # Single range fallback
            min_val, max_val = range_limits[0]
            result = pd.Series(index=series.index, dtype="object")

            # Values below minimum
            result[series < min_val] = f"<{fmt(min_val)}"

            # Values within range
            result[(series >= min_val) & (series < max_val)] = (
                f"{fmt(min_val)}-{fmt(max_val)}"
            )

            # Values above maximum
            result[series >= max_val] = f">={fmt(max_val)}"

            # Handle nulls
            result[series.isna()] = np.nan

            return result

    def _handle_visualizations(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        operation_metrics: Optional[Dict[str, Any]] = None,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        operation_metrics : Optional[Dict[str, Any]]
            Optional operation metrics for visualization
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
                    f"[DIAG] Field: {self.field_name}, Strategy: {self.null_strategy}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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
                        operation_metrics=operation_metrics,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend or "plotly",
                        vis_strict=vis_strict,
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

    def _generate_visualizations(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        task_dir: Path,
        operation_metrics: Optional[Dict[str, Any]] = None,
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
        operation_metrics : Optional[Dict[str, Any]]
            Optional operation metrics for visualization
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

            # 1. Comparison visualization
            comparison_path = create_comparison_visualization(
                original_data=original_for_viz,
                anonymized_data=anonymized_for_viz,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=f"anonymization_numeric_{self.strategy}",
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if comparison_path:
                visualization_paths["comparison"] = comparison_path

            # 2. Group distribution visualization if k-anonymity metrics available
            if (
                "privacy_metrics" in operation_metrics
                and "group_count" in operation_metrics["privacy_metrics"]
            ):
                metric_path = create_metric_visualization(
                    metric_name="group_distribution",
                    metric_data=operation_metrics["privacy_metrics"]["group_count"],
                    task_dir=viz_dir,
                    field_name=self.field_name,
                    operation_name=f"{self.name}_numeric_{self.strategy}",
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )

                if metric_path:
                    visualization_paths["group_distribution"] = metric_path

            # 3. Metrics heatmap/visualization
            if operation_metrics:
                metrics_viz_path = create_metrics_overview_visualization(
                    metrics=operation_metrics,
                    task_dir=viz_dir,
                    field_name=self.field_name,
                    operation_name=f"{self.name}_numeric_{self.strategy}",
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )
                if metrics_viz_path:
                    visualization_paths.update(metrics_viz_path)
                    for path in metrics_viz_path:
                        self.logger.info(
                            f"Created metrics visualization: {Path(path).name}"
                        )

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")

        return visualization_paths

    def _collect_comprehensive_metrics(
        self,
        original_series: pd.Series,
        processed_series: pd.Series,
        full_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Collect all metrics using commons utilities.

        Parameters:
        -----------
        original_series : pd.Series
            Original data
        processed_series : pd.Series
            Processed data
        full_df : pd.DataFrame
            Full dataframe for k-anonymity metrics

        Returns:
        --------
        Dict[str, Any]
            Comprehensive metrics
        """
        # Prepare timing information
        timing_info = {
            "start_time": self.start_time,
            "end_time": self.end_time or time.time(),
            "batch_count": getattr(self, "process_count", 0) // self.chunk_size,
        }

        # Prepare operation parameters
        operation_params: Dict[str, Any] = {
            "strategy": self.strategy,
            **self.process_kwargs,
        }

        operation_metrics = collect_operation_metrics(
            operation_type="generalization",
            original_data=original_series,
            processed_data=processed_series,
            operation_params=operation_params,
            timing_info=timing_info,
        )

        # Add effectiveness metrics
        effectiveness = calculate_anonymization_effectiveness(
            original_series, processed_series
        )
        operation_metrics["effectiveness"] = effectiveness

        # Add privacy metrics if quasi-identifiers are available
        if self.quasi_identifiers and all(
            qi in full_df.columns for qi in self.quasi_identifiers
        ):
            privacy_metrics = calculate_batch_metrics(
                original_batch=full_df[[self.field_name] + self.quasi_identifiers],
                anonymized_batch=full_df[
                    [
                        (
                            self.output_field_name
                            if self.mode == "ENRICH"
                            else self.field_name
                        )
                    ]
                    + self.quasi_identifiers
                ],
                original_field_name=self.field_name,
                anonymized_field_name=(
                    self.output_field_name if self.mode == "ENRICH" else self.field_name
                ),
                quasi_identifiers=self.quasi_identifiers,
            )
            operation_metrics["privacy_metrics"] = privacy_metrics

            # Calculate privacy metrics overview
            operation_metrics["privacy_metric_overview"] = {
                "min_k_overall": privacy_metrics.get("min_k", 0),
                "avg_suppression_rate": round(
                    privacy_metrics.get("suppression_rate", 0.0), 4
                ),
                "avg_coverage": round(privacy_metrics.get("total_coverage", 0.0), 4),
                "avg_generalization_level": round(
                    privacy_metrics.get("generalization_level", 0.0), 4
                ),
            }

            # Add additional privacy indicators
            privacy_metrics["disclosure_risk"] = calculate_simple_disclosure_risk(
                full_df, self.quasi_identifiers
            )
            privacy_metrics["suppression_rate"] = calculate_suppression_rate(
                processed_series, original_series.isna().sum()
            )

        # Add processing summary
        operation_metrics["processing_summary"] = {
            "total_records": len(full_df),
            "processed_records": len(full_df) - len(processed_series),
            "vulnerable_records": len(processed_series),
        }

        return operation_metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for numeric generalization
        """
        params = dict(
            field_name=self.field_name,
            strategy=self.strategy,
            bin_count=self.bin_count,
            binning_method=self.binning_method,
            precision=self.precision,
            range_limits=self.range_limits,
            mode=self.mode,
            output_field_name=self.output_field_name,
            column_prefix=self.column_prefix,
            quasi_identifiers=self.quasi_identifiers,
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

    def convert_range_limits_for_schema(
        self, range_limits: Optional[List[Tuple[float, float]]]
    ) -> Optional[List[List[float]]]:
        """
        Convert List[Tuple[float, float]] to List[List[float]] for JSON schema compatibility.
        """
        if range_limits is None:
            return None
        return [list(t) for t in range_limits]


# Factory function
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
