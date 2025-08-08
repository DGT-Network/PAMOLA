"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Base Anonymization Operation
Package:       pamola_core.anonymization
Version:       3.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause
Description:
  This module provides the base class for all anonymization operations,
  defining common functionality, interface, and behavior with enhanced
  support for conditional processing, profiling integration, memory-efficient
  operations, and Dask-based distributed processing for large datasets.

Key Features:
  - Standardized operation lifecycle with validation, execution, and result handling
  - Support for both in-place (REPLACE) and new field creation (ENRICH) modes
  - Configurable null value handling strategies (PRESERVE, EXCLUDE, ERROR)
  - Memory-efficient chunk-based processing for large datasets
  - Comprehensive metrics collection and visualization generation
  - Robust caching mechanism for operation results
  - Progress tracking and reporting throughout the operation
  - Secure output generation with optional encryption
  - Conditional processing based on field values and risk scores
  - Integration with k-anonymity profiling results
  - Enhanced memory management with automatic optimization
  - Dask integration for distributed processing of large datasets

Framework:
  Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
  for input/output, progress tracking, and result reporting.

Changelog:
  v3.0.0 (2025-01-24):
      - Added full Dask support with automatic engine switching
      - Added pandas/Dask DataFrame conversion utilities
      - Added engine parameter (pandas/dask/auto)
      - Added dask_partition_size parameter for partition control
      - Implemented process_batch_dask() base method
      - Enhanced DataWriter integration for Dask DataFrames
      - Added distributed processing metrics collection
  v2.0.0 (2025-12-16):
      - Integrated with pamola_core/utils/ops/data_processing.py for memory optimization
      - Integrated with pamola_core/utils/ops/field_utils.py for field management
      - Added conditional processing support with complex conditions
      - Added k-anonymity risk-based processing
      - Enhanced memory management with automatic dtype optimization
      - Improved progress tracking with hierarchical trackers
      - Standardized artifact generation through framework utilities
"""

import logging
import time
import dask.dataframe as dd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import anonymization-specific utilities
from pamola_core.anonymization.commons.data_utils import handle_vulnerable_records
from pamola_core.anonymization.commons.metric_utils import (
    calculate_anonymization_effectiveness,
)
from pamola_core.anonymization.commons.processing_utils import (
    process_dataframe_using_chunk,
    process_dataframe_using_dask,
    process_dataframe_using_joblib,
)
from pamola_core.anonymization.commons.visualization_utils import (
    create_metric_visualization,
    create_comparison_visualization,
    sample_large_dataset,
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io import (
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
from pamola_core.utils.ops.op_base import BaseOperation

# Import framework utilities
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_processing import (
    optimize_dataframe_dtypes,
    get_memory_usage,
    force_garbage_collection,
    process_null_values as process_nulls_framework,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_field_utils import (
    apply_condition_operator,
    create_field_mask,
    create_multi_field_mask,
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode


class AnonymizationOperation(BaseOperation):
    """
    Base class for all anonymization operations with Dask support.

    This class provides common functionality for all anonymization operations,
    including data source handling, result processing, metric generation,
    integration with profiling results for risk-based processing, and
    automatic switching to Dask for large dataset processing.
    """

    def __init__(
        self,
        field_name: str,
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
        null_strategy: str = "PRESERVE",
        description: str = "",
        # Conditional processing parameters
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: str = "in",
        # Multi-field conditions
        multi_conditions: Optional[List[Dict[str, Any]]] = None,
        condition_logic: str = "AND",
        # K-anonymity integration
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        vulnerable_record_strategy: str = "suppress",
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
        encryption_mode: Optional[str] = None,
        encryption_key: Optional[Union[str, Path]] = None,
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        output_format: str = "csv",
    ):
        """
        Initialize the anonymization operation with Dask support.

        Parameters:
        -----------
        field_name : str
            Field to anonymize
        mode : str, optional
            "REPLACE" to modify the field in-place, or "ENRICH" to create a new field
        output_field_name : str, optional
            Name for the output field if mode is "ENRICH"
        column_prefix : str, optional
            Prefix for new column if mode is "ENRICH"
        null_strategy : str, optional
            How to handle NULL values: "PRESERVE", "EXCLUDE", or "ERROR"
        description : str, optional
            Operation description
        condition_field : str, optional
            Field to use for conditional processing
        condition_values : List, optional
            Values to match for conditional processing
        condition_operator : str, optional
            Operator for condition matching: "in", "not_in", "gt", "lt", "eq", "range"
        multi_conditions : List[Dict], optional
            List of conditions for complex filtering
        condition_logic : str, optional
            Logic to combine conditions: "AND" or "OR"
        ka_risk_field : str, optional
            Field containing k-anonymity risk scores
        risk_threshold : float, optional
            Threshold for identifying vulnerable records (k < threshold)
        vulnerable_record_strategy : str, optional
            Strategy for handling vulnerable records
        optimize_memory : bool, optional
            Whether to optimize DataFrame memory usage
        adaptive_chunk_size : bool, optional
            Whether to adjust chunk size based on available memory
        chunk_size : int, optional
            Size of chunks for processing (default: 10000)
        use_dask : bool, optional
            Whether to use Dask for parallel processing (default: False)
        npartitions : Optional[int], optional
            Number of partitions to use with Dask (default: None)
        dask_partition_size : Optional[str], optional
            Size of Dask partitions (default: None, auto-determined)
        use_vectorization : bool, optional
            Whether to use vectorized operations (default: False)
        parallel_processes : Optional[int], optional
            Number of parallel processes to use (default: None)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        encryption_mode : Optional[str], optional
            The encryption mode to use (default: None)
        visualization_theme : Optional[str], optional
            Theme for visualizations (default: None)
        visualization_backend : Optional[str], optional
            Backend for visualizations ("plotly" or "matplotlib", default: None)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout for visualization generation in seconds (default: 120)
        output_format : str, optional
            Format for output files ("csv", "parquet", or "json", default: "csv")
        """
        # Use a default description if none provided
        if not description:
            description = f"Anonymization operation for field '{field_name}'"

        # Initialize base class
        super().__init__(
            name=f"{field_name}_anonymization",
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
        )

        # Store basic parameters
        self.field_name = field_name
        self.mode = mode.upper()
        self.output_field_name = output_field_name
        self.column_prefix = column_prefix
        self.null_strategy = null_strategy.upper()
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.dask_partition_size = dask_partition_size
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.encryption_mode = encryption_mode
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout
        self.output_format = output_format
        self.version = getattr(
            self, "version", "3.0.0"
        )  # Updated version for visualization context support

        # Conditional processing parameters
        self.condition_field = condition_field
        self.condition_values = condition_values
        self.condition_operator = condition_operator
        self.multi_conditions = multi_conditions
        self.condition_logic = condition_logic

        # K-anonymity parameters
        self.ka_risk_field = ka_risk_field
        self.risk_threshold = risk_threshold
        self.vulnerable_record_strategy = vulnerable_record_strategy

        # Memory optimization parameters
        self.optimize_memory = optimize_memory
        self.adaptive_chunk_size = adaptive_chunk_size
        self.original_chunk_size = chunk_size

        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.process_count = 0

        # Set up common variables
        self.force_recalculation = False  # Skip cache check
        self.generate_visualization = True  # Create visualizations
        self.save_output = True  # Save processed data to output directory
        self.process_kwargs = {}

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the anonymization operation with enhanced features including Dask support.

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
            Additional parameters including profiling_results

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Start timing
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(f"Starting {self.name} operation at {self.start_time}")
            self.process_count = 0
            df = None

            # Initialize result object
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Create writer for consistent output handling
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Save operation configuration
            self.save_config(task_dir)

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
                            "step": f"Starting {self.name}",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                try:
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
                                f"Anonymization of {self.field_name} (from cache)",
                                details={"cached": True},
                            )

                        return cache_result
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
                mask, filtered_df = self._apply_conditional_filtering(df)

                # Process the filtered data
                is_use_batch_dask = hasattr(self, "process_batch_dask")
                processed_df = self._process_data_with_config(
                    df=filtered_df,
                    is_use_batch_dask=is_use_batch_dask,
                    progress_tracker=data_tracker,
                )

                # Apply the processed data back to the original DataFrame
                if self.mode == "ENRICH":
                    df[self.output_field_name] = (
                        original_data  # Initialize with original
                    )

                df.loc[mask, self.output_field_name] = processed_df.loc[
                    mask, self.output_field_name
                ]

                # Handle vulnerable records if k-anonymity is enabled
                if self.ka_risk_field and self.ka_risk_field in df.columns:
                    df = self._handle_vulnerable_records(df, self.output_field_name)

                # Get the anonymized data
                anonymized_data = (
                    df[self.output_field_name]
                    if self.mode == "ENRICH"
                    else df[self.field_name]
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
                metrics = self._collect_all_metrics(
                    original_data, anonymized_data, mask
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
                        result_df=df,
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
                processed_df=df,
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
                        "records_filtered": len(filtered_df),
                        "vulnerable_records_handled": metrics.get(
                            "vulnerable_records", 0
                        ),
                    },
                )

            # Set success status
            result.status = OperationStatus.SUCCESS
            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )
            return result

        except Exception as e:
            error_message = f"Error in anonymization operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message,
                exception=e,
            )

    def _validate_and_get_dataframe(
        self, data_source: DataSource, dataset_name: str, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Validate data source and retrieve the main dataframe.

        Parameters:
        -----------
        data_source : DataSource
            The data source to validate
        dataset_name : str
            The name of the dataset to retrieve
        **kwargs : Any
            Additional keyword arguments to pass to the data loading function

        Returns:
        --------
        pd.DataFrame
            The validated dataframe

        Raises:
        -------
        ValueError
            If no valid dataframe is found or the field is missing
        """
        # Get DataFrame from the data source
        df = load_data_operation(data_source, dataset_name, **kwargs)
        if df is None:
            error_message = f"Failed to load input data!"
            self.logger.error(error_message)
            raise ValueError(error_message)

        if self.field_name not in df.columns:
            error_message = f"Field {self.field_name} not found in DataFrame"
            self.logger.error(error_message)
            raise ValueError(error_message)

        df = self._optimize_data(df)

        return df

    def _optimize_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Load data and optimize memory usage.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to optimize

        Returns:
        --------
        Optional[pd.DataFrame]
            Loaded and optimized DataFrame or None if error
        """
        # Optimize memory if enabled
        if self.optimize_memory and len(df) > 10000:
            self.logger.info("Optimizing DataFrame memory usage")
            initial_memory = get_memory_usage(df)

            df, optimization_info = optimize_dataframe_dtypes(
                df,
                categorical_threshold=0.5,
                downcast_integers=True,
                downcast_floats=True,
            )

            self.logger.info(
                f"Memory optimization: {initial_memory['total_mb']:.2f}MB -> "
                f"{optimization_info['memory_after_mb']:.2f}MB "
                f"(saved {optimization_info['memory_saved_percent']:.1f}%)"
            )

            # Adjust chunk size if adaptive
            if self.adaptive_chunk_size:
                self._adjust_chunk_size(df)

        return df

    def _adjust_chunk_size(self, df: pd.DataFrame) -> None:
        """
        Adjust chunk size based on available memory.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        """
        memory_info = get_memory_usage(df)
        row_memory = memory_info["per_row_bytes"]

        # Target to use at most 100MB per chunk
        target_memory_mb = 100
        target_rows = int((target_memory_mb * 1024 * 1024) / row_memory)

        # Adjust chunk size within reasonable bounds
        self.chunk_size = max(1000, min(target_rows, 100000))

        # Ensure chunk size doesn't exceed data size
        self.chunk_size = min(self.chunk_size, len(df))

        if self.chunk_size != self.original_chunk_size:
            self.logger.info(
                f"Adjusted chunk size from {self.original_chunk_size} "
                f"to {self.chunk_size} based on memory usage"
            )

    def _prepare_output_field(self, df: pd.DataFrame) -> str:
        """
        Validate and generate the output field name.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to check field names against

        Returns:
        --------
        str
            The validated output field name
        """
        # Determine output field name based on mode
        if self.mode == "REPLACE":
            return self.field_name
        else:  # ENRICH mode
            if self.output_field_name:
                output_field = self.output_field_name
            else:
                output_field = f"{self.column_prefix}{self.field_name}"

            # Check if output field already exists in DataFrame
            if output_field in df.columns:
                self.logger.warning(
                    f"Output field '{output_field}' already exists and will be overwritten"
                )
        return output_field

    def _report_operation_details(self, reporter: Any, output_field: str) -> None:
        """
        Report details of the operation to the reporter.

        Parameters:
        -----------
        reporter : Any
            The reporter to log details to
        output_field : str
            The name of the output field
        """
        if reporter:
            reporter.add_operation(
                f"Anonymizing field: {self.field_name}",
                details={
                    "field_name": self.field_name,
                    "output_field": output_field,
                    "mode": self.mode,
                    "null_strategy": self.null_strategy,
                    "operation_type": self.__class__.__name__,
                },
            )

    def _apply_conditional_filtering(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Apply conditional filtering based on conditions and profiling results.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to filter

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            (mask, filtered_dataframe)
        """
        # Start with all records
        mask = pd.Series(True, index=df.index)

        # Apply simple condition if specified
        if self.condition_field and self.condition_values:
            field_mask = create_field_mask(
                df,
                self.field_name,
                self.condition_field,
                self.condition_values,
                self.condition_operator,
            )
            mask = mask & field_mask
            self.logger.info(
                f"Applied condition on '{self.condition_field}': "
                f"{mask.sum()} records match"
            )

        # Apply multi-field conditions if specified
        if self.multi_conditions:
            multi_mask = create_multi_field_mask(
                df, self.multi_conditions, self.condition_logic
            )
            mask = mask & multi_mask
            self.logger.info(
                f"Applied {len(self.multi_conditions)} conditions: "
                f"{mask.sum()} records match"
            )

        # Apply k-anonymity filtering if specified
        if self.ka_risk_field and self.ka_risk_field in df.columns:
            risk_mask = df[self.ka_risk_field] < self.risk_threshold
            mask = mask & risk_mask
            self.logger.info(
                f"Applied k-anonymity filter (k < {self.risk_threshold}): "
                f"{mask.sum()} vulnerable records"
            )

        # Create filtered DataFrame
        filtered_df = df[mask].copy()

        return mask, filtered_df

    def _process_data_with_config(
        self,
        df: pd.DataFrame,
        is_use_batch_dask: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
    ) -> pd.DataFrame:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        is_use_batch_dask : bool
            Whether to use Dask for batch processing
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        pd.DataFrame
            The processed dataframe
        """
        # Check if dataframe is empty
        if len(df) == 0:
            self.logger.warning("Empty DataFrame provided, returning as is")
            return df

        # Handle null values based on strategy
        if self.null_strategy != "PRESERVE":
            df[self.field_name] = process_nulls_framework(
                df[self.field_name], strategy=self.null_strategy.lower()
            )

        processed_df = None
        flag_processed = False
        self.logger.info("Process with config")

        # For larger dataframes, check if we should use parallel processing
        if not flag_processed and self.use_dask:
            try:
                self.logger.info("Parallel Enabled")
                self.logger.info("Parallel Engine: Dask")
                self.logger.info(f"Parallel Workers: {self.npartitions}")
                self.logger.info(
                    f"Using dask processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Setting up dask processing"})

                self.logger.info("Process using Dask")

                # Process with Dask - delegate to subclass method
                if is_use_batch_dask:
                    processed_df, flag_processed = process_dataframe_using_dask(
                        df=df,
                        process_function=self.process_batch_dask,
                        is_use_batch_dask=is_use_batch_dask,
                        progress_tracker=progress_tracker,
                        task_logger=self.logger,
                        **self.process_kwargs,
                    )
                else:
                    processed_df, flag_processed = process_dataframe_using_dask(
                        df=df,
                        process_function=self.process_batch,
                        is_use_batch_dask=is_use_batch_dask,
                        progress_tracker=progress_tracker,
                        task_logger=self.logger,
                        **self.process_kwargs,
                    )

                if flag_processed:
                    self.logger.info("Completed using Dask")

            except Exception as e:
                self.logger.warning(
                    f"Error in dask processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and self.use_vectorization:
            try:
                self.logger.info("Parallel Enabled")
                self.logger.info("Parallel Engine: Joblib")
                self.logger.info(f"Parallel Workers: {self.parallel_processes}")
                self.logger.info(
                    f"Using vectorized processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(
                        0, {"step": "Setting up vectorized processing"}
                    )

                self.logger.info("Process using Joblib")

                processed_df, flag_processed = process_dataframe_using_joblib(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **self.process_kwargs,
                )

                if flag_processed:
                    self.logger.info("Completed using Joblib")

            except Exception as e:
                self.logger.warning(
                    f"Error in vectorized processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and self.chunk_size > 1:
            try:
                # Regular chunk processing
                self.logger.info(
                    f"Processing in chunks with chunk size {self.chunk_size}"
                )
                total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
                self.logger.info(f"Total chunks to process: {total_chunks}")
                if progress_tracker:
                    progress_tracker.update(
                        0,
                        {
                            "step": "Processing in chunks",
                            "total_chunks": total_chunks,
                        },
                    )
                self.logger.info("Process using chunk")
                processed_df, flag_processed = process_dataframe_using_chunk(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **self.process_kwargs,
                )
                if flag_processed:
                    self.logger.info("Completed using chunk")
            except Exception as e:
                self.logger.warning(f"Error in chunk processing: {e}")

        if not flag_processed:
            self.logger.info("Fallback process as usual")
            processed_df = self.process_batch(df, **self.process_kwargs)
            flag_processed = True

        # Update process count
        self.process_count += len(df)
        return processed_df

    def _handle_vulnerable_records(
        self,
        df: pd.DataFrame,
        output_field: str,
    ) -> pd.DataFrame:
        """
        Handle records identified as vulnerable by k-anonymity analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with processed data
        output_field : str
            Output field name

        Returns:
        --------
        pd.DataFrame
            DataFrame with vulnerable records handled
        """
        # Identify vulnerable records
        vulnerability_mask = df[self.ka_risk_field] < self.risk_threshold
        vulnerable_count = vulnerability_mask.sum()

        if vulnerable_count > 0:
            self.logger.info(
                f"Handling {vulnerable_count} vulnerable records "
                f"with strategy: {self.vulnerable_record_strategy}"
            )

            # Apply strategy using commons utility
            df = handle_vulnerable_records(
                df, output_field, vulnerability_mask, self.vulnerable_record_strategy
            )

        return df

    def _collect_all_metrics(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        mask: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Collect comprehensive metrics.

        Parameters:
        -----------
        original_data : pd.Series
            Original data
        anonymized_data : pd.Series
            Anonymized data
        mask : pd.Series, optional
            Processing mask

        Returns:
        --------
        Dict[str, Any]
            Collected metrics
        """
        # Basic anonymization effectiveness metrics
        metrics: Dict[str, Any] = calculate_anonymization_effectiveness(
            original_data, anonymized_data
        )

        # Add performance metrics
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            metrics.update(
                {
                    "duration_seconds": round(duration, 2),
                    "records_processed": self.process_count,
                    "records_per_second": (
                        round(self.process_count / duration, 2) if duration > 0 else 0
                    ),
                    "chunk_count": (
                        int((self.process_count - 1) / self.chunk_size + 1)
                        if self.process_count > 0
                        else 0
                    ),
                }
            )

        # Add operation-specific metrics
        metrics.update(self._collect_specific_metrics(original_data, anonymized_data))
        # Calculate effective mask
        effective_mask = mask.sum() if mask is not None else len(anonymized_data)
        # Add filtering metrics
        metrics.update(
            {
                "total_records": len(original_data),
                "processed_records": effective_mask,
                "filtered_records": len(original_data) - effective_mask,
                "processing_rate": (
                    (effective_mask / len(original_data)) * 100
                    if len(original_data) > 0
                    else 0
                ),
            }
        )

        # Add memory metrics if optimization was performed
        if self.optimize_memory:
            metrics["chunk_size_used"] = self.chunk_size
            metrics["adaptive_chunk_size"] = self.adaptive_chunk_size

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

            # Create comparison visualization
            comparison_path = create_comparison_visualization(
                original_data=original_for_viz,
                anonymized_data=anonymized_for_viz,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=self.__class__.__name__,
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if comparison_path:
                visualization_paths["comparison"] = comparison_path

            # Create distribution visualization for anonymized data
            metric_data = anonymized_for_viz.value_counts().to_dict()
            dist_path = create_metric_visualization(
                metric_name="distribution",
                metric_data=metric_data,
                task_dir=viz_dir,
                field_name=self.field_name,
                operation_name=self.__class__.__name__,
                timestamp=timestamp,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if dist_path:
                visualization_paths["distribution"] = dist_path

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")

        return visualization_paths

    def _save_output_data(
        self,
        result_df: pd.DataFrame,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        timestamp: Optional[str] = None,
        use_encryption: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """
        Save the processed output data.

        Parameters:
        -----------
        result_df : pd.DataFrame
            The processed dataframe to save
        use_encryption : bool
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

        # Generate standardized output filename with timestamp
        operation_name = self.__class__.__name__
        field_name_output = f"{self.field_name}_{operation_name}_output_{timestamp}"

        # Use the DataWriter to save the DataFrame
        safe_kwargs = filter_used_kwargs(kwargs, writer.write_dataframe)
        safe_kwargs["encryption_mode"] = get_encryption_mode(result_df, **kwargs)
        output_result = writer.write_dataframe(
            df=result_df,
            name=field_name_output,
            format=self.output_format,
            subdir="output",
            timestamp_in_name=False,
            encryption_key=self.encryption_key if use_encryption else None,
            overwrite=True,
            **safe_kwargs,
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type=self.output_format,
            path=output_result.path,
            description=f"{self.field_name} anonymized data",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_operation(
                f"{self.field_name} anonymized data",
                details={
                    "artifact_type": self.output_format,
                    "path": str(output_result.path),
                },
            )
        return str(output_result.path)

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

        # Clear process kwargs
        if hasattr(self, "process_kwargs"):
            self.process_kwargs = {}

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Force garbage collection
        force_garbage_collection()

    def _should_process_record(self, record: pd.Series) -> bool:
        """
        Determine if a record should be processed based on conditions.

        Parameters:
        -----------
        record : pd.Series
            Record to check

        Returns:
        --------
        bool
            True if record should be processed
        """
        # No conditions means process all records
        if not self.condition_field:
            return True

        # Check simple condition
        if self.condition_field in record.index:
            value = record[self.condition_field]
            mask = apply_condition_operator(
                pd.Series([value]), self.condition_values, self.condition_operator
            )
            return mask.iloc[0]

        return True

    @classmethod
    def process_batch(cls, batch: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process a batch of data. Must be implemented by subclasses.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch
        """
        raise NotImplementedError("Subclasses must implement process_batch method")

    @classmethod
    def process_batch_dask(cls, ddf: dd.DataFrame, **kwargs) -> dd.DataFrame:
        """
        Process Dask DataFrame. Should be overridden by subclasses for optimal performance.

        Parameters:
        -----------
        ddf : dd.DataFrame
            Dask DataFrame to process
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        dd.DataFrame
            Processed Dask DataFrame
        """

        # Default implementation: process each partition with process_batch
        def process_partition(partition):
            return cls.process_batch(partition, **kwargs)

        return ddf.map_partitions(process_partition)

    @classmethod
    def process_value(cls, value, **kwargs):
        """
        Process a single value. Must be implemented by subclasses.

        Parameters:
        -----------
        value : Any
            Value to process
        kwargs : dict
            Additional parameters

        Returns:
        --------
        Any
            Processed value
        """
        raise NotImplementedError("Subclasses must implement process_value method")

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect operation-specific metrics. Should be overridden by subclasses.

        Parameters:
        -----------
        original_data : pd.Series
            Original data
        anonymized_data : pd.Series
            Anonymized data

        Returns:
        --------
        Dict[str, Any]
            Operation-specific metrics
        """
        return {}

    def _check_cache(
        self, df: pd.DataFrame, reporter: Any
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for this operation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame for the operation
        reporter : Any
            Reporter object for tracking progress and artifacts

        Returns
        -------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            if self.field_name not in df.columns:
                self.logger.warning(
                    f"Field '{self.field_name}' not found in DataFrame."
                )
                return None

            cache_key = self._generate_cache_key(df[self.field_name])
            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(
                f"Using cached result for {self.field_name} generalization"
            )

            result = OperationResult(status=OperationStatus.SUCCESS)
            # Restore cached data
            self._add_cached_metrics(result, cached_result)
            artifacts_restored = self._restore_cached_artifacts(
                result, cached_result, reporter
            )

            # Add cache metadata
            result.add_metric("cached", True)
            result.add_metric("cache_key", cache_key)
            result.add_metric(
                "cache_timestamp", cached_result.get("timestamp", "unknown")
            )
            result.add_metric("artifacts_restored", artifacts_restored)

            if reporter:
                reporter.add_operation(
                    f"Generalization of {self.field_name} (cached)",
                    details={
                        "null_strategy": self.null_strategy,
                        "cached": True,
                        "artifacts_restored": artifacts_restored,
                    },
                )

            self.logger.info(
                f"Cache hit successful: restored {artifacts_restored} artifacts"
            )
            return result

        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _add_cached_metrics(self, result: OperationResult, cached: dict):
        """
        Add cached scalar metrics (int, float, str, bool) to the OperationResult.

        Parameters
        ----------
        result : OperationResult
            The result object to update.
        cached : dict
            Cached result dictionary from cache manager.
        """
        for key, value in cached.get("metrics", {}).items():
            if isinstance(value, (int, float, str, bool)):
                result.add_metric(key, value)

    def _restore_cached_artifacts(
        self, result: OperationResult, cached: dict, reporter: Optional[Any]
    ) -> int:
        """
        Restore artifacts (output, metrics, visualizations) from cached result if files exist.

        Parameters
        ----------
        result : OperationResult
            OperationResult object to update with restored artifacts.
        cached : dict
            Cached result dictionary from cache manager.
        reporter : Optional[Any]
            Optional reporter object for tracking operation-level artifacts.

        Returns
        -------
        int
            Number of artifacts successfully restored.
        """
        artifacts_restored = 0

        def restore_file_artifact(
            path: Union[str, Path], artifact_type: str, desc_suffix: str, category: str
        ):
            """
            Restore a single artifact from a file path if it exists.

            Parameters
            ----------
            path : Union[str, Path]
                Path to the artifact file.
            artifact_type : str
                Type of the artifact (e.g., 'json', 'csv', 'png').
            desc_suffix : str
                Description suffix (e.g., 'visualization', 'metrics').
            category : str
                Artifact category (e.g., output, metrics, visualization).
            """
            nonlocal artifacts_restored
            if not path:
                return

            artifact_path = Path(path)
            if artifact_path.exists():
                result.add_artifact(
                    artifact_type=artifact_type,
                    path=artifact_path,
                    description=f"{self.field_name} {desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} {desc_suffix} (cached)",
                        details={
                            "artifact_type": artifact_type,
                            "path": str(artifact_path),
                        },
                    )
            else:
                self.logger.warning(f"Cached file not found: {artifact_path}")

        # Restore main output and metrics and mapping artifacts
        restore_file_artifact(
            cached.get("output_file"),
            self.output_format,
            "generalized data",
            Constants.Artifact_Category_Output,
        )
        restore_file_artifact(
            cached.get("metrics_file"),
            "json",
            "generalization metrics",
            Constants.Artifact_Category_Metrics,
        )
        restore_file_artifact(
            cached.get("mapping_file"),
            "json",
            "generalized mapping",
            Constants.Artifact_Category_Mapping,
        )

        # Restore visualizations
        for viz_type, path_str in cached.get("visualizations", {}).items():
            restore_file_artifact(
                path_str,
                "png",
                f"{viz_type} visualization",
                Constants.Artifact_Category_Visualization,
            )

        return artifacts_restored

    def _save_to_cache(
        self,
        original_data: pd.Series,
        anonymized_data: pd.Series,
        metrics: Dict[str, Any],
        task_dir: Path,
        visualization_paths: Dict[str, Path] = {},
        metrics_result_path: Optional[str] = None,
        output_result_path: Optional[str] = None,
        mapping_result_path: Optional[str] = None,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_data : pd.Series
            Original input data
        anonymized_data : pd.Series
            Anonymized output data
        metrics : Dict[str, Any]
            Metrics collected during the operation
        visualization_paths : Dict[str, Path]
            Paths to generated visualizations
        task_dir : Path
            Task directory
        metrics_result_path : Optional[str]
            Path to the metrics result file
            If not provided, a default path will be used.
        output_result_path : Optional[str]
            Path to the output result file
            If not provided, a default path will be used.
        mapping_result_path : Optional[str]
            Path to the mapping result file
            If not provided, a default path will be used.

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(original_data)

            # Prepare metadata for cache
            operation_params = self._get_basic_parameters()
            operation_params.update(self._get_cache_parameters())
            self.logger.debug(f"Operation parameters for cache: {operation_params}")

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in metrics.items()
                },
                "parameters": operation_params,
                "data_info": {
                    "original_length": len(original_data),
                    "anonymized_length": len(anonymized_data),
                    "original_null_count": int(original_data.isna().sum()),
                    "anonymized_null_count": int(anonymized_data.isna().sum()),
                },
                "output_file": output_result_path,  # Path to main output file
                "metrics_file": metrics_result_path,  # Path to metrics file
                "mapping_file": mapping_result_path,  # Path to mapping file if applicable
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")

            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(
                    f"Successfully saved {self.field_name} anonymization results to cache"
                )
            else:
                self.logger.warning(
                    f"Failed to save {self.field_name} anonymization results to cache"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(self, data: pd.Series) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        data : pd.Series
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        # Get basic operation parameters
        parameters = self._get_basic_parameters()

        # Add operation-specific parameters through method that subclasses can override
        parameters.update(self._get_cache_parameters())

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(data)

        # Use the operation_cache utility to generate a consistent cache key
        return self.operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash,
        )

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generate a hash representing the key characteristics of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data for the operation

        Returns
        -------
        str
            Hash string representing the data
        """
        import hashlib
        import json

        try:
            # Generate summary statistics for all columns (numeric and non-numeric)
            characteristics = df.describe(include="all")

            # Convert to JSON string with consistent formatting (ISO for dates)
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback: use length and column data types
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _get_basic_parameters(self) -> Dict[str, str]:
        """Get the basic parameters for the cache key generation."""
        return {
            "name": self.name,
            "null_strategy": self.null_strategy,
            "description": self.description,
            "version": self.version,
        }

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        This method should be overridden by subclasses to provide
        operation-specific parameters for caching.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Base implementation returns minimal parameters
        return {}
