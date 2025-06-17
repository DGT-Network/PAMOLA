"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Base Anonymization Operation
Package:       pamola.core.operations.base
Version:       1.1.0+refactor.2025.08.21
Status:        stable
Author:        PAMOLA Core Team
Created:       2024
License:       BSD 3-Clause
Description:
    This module provides the base class for all anonymization operations,
    defining common functionality, interface, and behavior with thread-safe
    visualization support.

Key Features:
    - Standardized operation lifecycle with validation, execution, and result handling
    - Support for both in-place (REPLACE) and new field creation (ENRICH) modes
    - Configurable null value handling strategies (PRESERVE, EXCLUDE, ERROR)
    - Memory-efficient chunk-based processing for large datasets
    - Comprehensive metrics collection and visualization generation
    - Robust caching mechanism for operation results
    - Progress tracking and reporting throughout the operation
    - Secure output generation with optional encryption
    - Thread-safe visualization generation with context isolation

Framework:
    Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
    for input/output, progress tracking, and result reporting.

TODO:
    - Add support for distributed processing using Dask for very large datasets
    - Implement delta metrics calculation for cached vs. fresh execution
    - Support custom visualization options beyond default charts
    - Add configuration option for metrics verbosity levels
    - Implement optimization for batch sizes based on data characteristics
    - Support resumable operations for very large datasets
    - Add integration with privacy model validation
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from pamola_core.anonymization.commons.processing_utils import (
    process_dataframe_dask,
    process_dataframe_parallel,
    process_in_chunks,
)
from pamola_core.anonymization.commons.visualization_utils import sample_large_dataset
from pamola_core.utils.io import load_data_operation
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import create_histogram, create_bar_plot
from pamola_core.common.constants import Constants

logger = logging.getLogger(__name__)


class AnonymizationOperation(BaseOperation):
    """
    Base class for all anonymization operations.

    This class provides common functionality for all anonymization operations,
    including data source handling, result processing, and metric generation.
    """

    def __init__(
        self,
        field_name: str,
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
        null_strategy: str = "PRESERVE",
        description: str = "",
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
    ):
        """
        Initialize the anonymization operation.

        Parameters:
        -----------
        field_name : str
            Field to anonymize
        mode : str, optional
            "REPLACE" to modify the field in-place, or "ENRICH" to create a new field (default: "REPLACE")
        output_field_name : str, optional
            Name for the output field if mode is "ENRICH" (default: None)
        column_prefix : str, optional
            Prefix for new column if mode is "ENRICH" (default: "_")
        null_strategy : str, optional
            How to handle NULL values: "PRESERVE", "EXCLUDE", or "ERROR" (default: "PRESERVE")
        description : str, optional
            Operation description (default: "")
        chunk_size : int, optional
            Size of chunks for processing (default: 10000)
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
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        visualization_theme : Optional[str], optional
            Theme for visualizations (default: None)
        visualization_backend : Optional[str], optional
            Backend for visualizations ("plotly" or "matplotlib", default: None)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout for visualization generation in seconds (default: 120)
        output_format : str, optional
            Format for output files ("csv", "parquet", or "arrow", default: "csv")
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
        )

        # Store parameters
        self.field_name = field_name
        self.mode = mode.upper()  # Ensure uppercase
        self.output_field_name = output_field_name
        self.column_prefix = column_prefix
        self.null_strategy = null_strategy.upper()  # Ensure uppercase
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        self.use_cache = use_cache
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout
        self.output_format = output_format
        self.version = getattr(
            self, "version", "1.1.0"
        )  # Updated version for visualization context support

        # Set up performance tracking variables
        self.start_time = None
        self.end_time = None
        self.process_count = 0

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
        Execute the anonymization operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for the operation (default: None)
        **kwargs : dict
            Additional parameters for the operation
            - force_recalculation: bool - Force operation even if cached results exist
            - encrypt_output: bool - Override encryption setting for outputs
            - generate_visualization: bool - Create visualizations
            - save_output: bool - Save processed data to output directory
            - visualization_theme: str - Theme for visualizations
            - visualization_backend: str - Backend for visualizations ("plotly" or "matplotlib")
            - visualization_strict: bool - If True, raise exceptions for visualization config errors
            - visualization_timeout: int - Timeout for visualization generation in seconds (default: 120)

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Start timing
        self.start_time = time.time()
        self.process_count = 0
        df = None

        # Initialize result object
        result = OperationResult(status=OperationStatus.PENDING)

        # Save operation configuration
        self.save_config(task_dir)

        # Create writer for consistent output handling
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
        vis_backend = kwargs.get("visualization_backend", self.visualization_backend)
        vis_strict = kwargs.get("visualization_strict", self.visualization_strict)
        vis_timeout = kwargs.get("visualization_timeout", self.visualization_timeout)

        # Set up progress tracking with proper steps
        # Main steps: 1. Cache check, 2. Data loading, 3. Validation, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
        TOTAL_MAIN_STEPS = 6 + (1 if self.use_cache and not force_recalculation else 0)
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

        try:
            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not force_recalculation:
                if main_progress:
                    current_steps += 1
                    main_progress.update(
                        current_steps,
                        {"step": "Checking cache", "field": self.field_name},
                    )
                # Load data for cache check
                df = load_data_operation(data_source, dataset_name)

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df, reporter)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

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

            # Step 2: Data Loading
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Data Loading", "field": self.field_name}
                )

            # Validate and get dataframe
            try:
                if df is None:
                    df = self._validate_and_get_dataframe(data_source, dataset_name)
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 3: Validation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Validation", "field": self.field_name}
                )

            try:
                output_field = self._prepare_output_field(df)
                self._report_operation_details(reporter, output_field)
                original_data = df[self.field_name].copy(deep=True)
            except Exception as e:
                error_message = f"Validation error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 4: Processing
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", "field": self.field_name}
                )

            try:
                # Create child progress tracker for Chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        total_chunks = (len(df) - 1) // self.chunk_size + 1
                        data_tracker = main_progress.create_subtask(
                            total=total_chunks,
                            description="Chunk processing",
                            unit="Chunk",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                result_df = self._process_dataframe(
                    df=df, progress_tracker=data_tracker
                )

                anonymized_data = result_df[output_field]
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
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Collect final metrics before using them
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
                metrics = self._calculate_all_metrics(original_data, anonymized_data)

                # Generate metrics file name (in self.name existed field_name)
                metrics_file_name = f"{self.field_name}_{self.__class__.__name__}_metrics_{operation_timestamp}"

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if is_encryption_required else None
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
            if save_output:
                try:
                    output_result_path = self._save_output_data(
                        result_df=result_df,
                        is_encryption_required=is_encryption_required,
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

            # Report completion
            if reporter:
                # Create the details dictionary with checks for all values
                details = {
                    "records_processed": self.process_count,
                    "execution_time": (
                        self.end_time - self.start_time
                        if self.end_time and self.start_time
                        else None
                    ),
                }

                # Only add generalization_ratio if metrics exists and has this key
                if metrics and isinstance(metrics, dict):
                    generalization_ratio = metrics.get("generalization_ratio")
                    if generalization_ratio is not None:
                        details["generalization_ratio"] = generalization_ratio

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Anonymization of {self.field_name} completed", details=details
                )

            ## Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=result_df,
                original_data=original_data,
                anonymized_data=anonymized_data,
            )

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error in anonymization operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message
            )

    def _validate_and_get_dataframe(
        self, data_source: DataSource, dataset_name: str
    ) -> pd.DataFrame:
        """
        Validate data source and retrieve the main dataframe.

        Parameters:
        -----------
        data_source : DataSource
            The data source to validate

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
        df = load_data_operation(data_source, dataset_name)
        if df is None:
            error_message = f"Failed to load input data!"
            self.logger.error(error_message)
            raise ValueError(error_message)

        if self.field_name not in df.columns:
            error_message = f"Field {self.field_name} not found in DataFrame"
            self.logger.error(error_message)
            raise ValueError(error_message)

        return df

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

        # Update the output field name
        self.output_field_name = output_field

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

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> pd.DataFrame:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        use_dask : bool
            Whether to use Dask for distributed processing
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

        # For larger dataframes, check if we should use parallel processing
        if self.use_dask:
            try:
                self.logger.info(
                    f"Using dask processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Setting up dask processing"})

                return process_dataframe_dask(
                    df=df,
                    process_function=self.process_with_dask,
                    process_function_backup=self.process_batch,
                    chunk_size=self.chunk_size,
                    npartitions=self.npartitions,
                    progress_tracker=progress_tracker,
                )
            except Exception as e:
                self.logger.warning(
                    f"Error in dask processing: {e}, falling back to chunk processing"
                )
        elif self.use_vectorization:
            try:
                self.logger.info(
                    f"Using vectorized processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(
                        0, {"step": "Setting up vectorized processing"}
                    )

                return process_dataframe_parallel(
                    df=df,
                    process_function=self.process_batch,
                    n_jobs=self.parallel_processes
                    or 1,  # Use specified threads for vectorization
                    chunk_size=self.chunk_size,
                    progress_tracker=progress_tracker,
                )
            except Exception as e:
                self.logger.warning(
                    f"Error in vectorized processing: {e}, falling back to chunk processing"
                )

        # Regular chunk processing
        self.logger.info(f"Processing in chunks with chunk size {self.chunk_size}")
        if progress_tracker:
            total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
            progress_tracker.update(
                0, {"step": "Processing in chunks", "total_chunks": total_chunks}
            )

        return process_in_chunks(
            df=df,
            process_function=self.process_batch,
            chunk_size=self.chunk_size,
            progress_tracker=progress_tracker,
        )

    def _calculate_all_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for the operation.

        Parameters:
        -----------
        original_data : pd.Series
            The original data before anonymization
        anonymized_data : pd.Series
            The anonymized data after processing

        Returns:
        --------
        Dict[str, Any]
            A dictionary of calculated metrics
        """
        # Calculate basic metrics
        metrics = self._collect_metrics(original_data, anonymized_data)

        # Add performance metrics
        metrics.update(
            {
                "execution_time_seconds": (
                    self.end_time - self.start_time
                    if self.end_time and self.start_time
                    else 0
                ),
                "records_processed": self.process_count,
                "records_per_second": (
                    self.process_count / (self.end_time - self.start_time)
                    if self.end_time
                    and self.start_time
                    and (self.end_time - self.start_time) > 0
                    else 0
                ),
            }
        )

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
                        vis_backend=vis_backend,
                        vis_strict=vis_strict,
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
        encrypt_output : bool
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
        field_name_output = (
            f"{self.field_name}_{self.__class__.__name__}_output_{timestamp}"
        )

        # Use the DataWriter to save the DataFrame
        output_result = writer.write_dataframe(
            df=result_df,
            name=field_name_output,
            format=self.output_format,
            subdir="output",
            timestamp_in_name=False,
            encryption_key=self.encryption_key if is_encryption_required else None,
            **kwargs,
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

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Optional: Force garbage collection for large datasets
        # Uncomment if memory pressure is an issue
        # import gc
        # gc.collect()

    def _generate_data_hash(self, data: pd.Series) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters:
        -----------
        data : pd.Series
            Input data for the operation

        Returns:
        --------
        str
            Hash string representing the data
        """
        import hashlib
        import json
        from typing import Any

        try:
            # Create dictionary of key data characteristics with proper typing
            # Use Dict[str, Any] to allow mixed value types (int, float, dict)
            characteristics: dict[str, Any] = {
                "length": len(data),
                "null_count": int(data.isna().sum()),
                "unique_count": int(data.nunique()),
            }

            # Add data type-specific characteristics
            if pd.api.types.is_numeric_dtype(data):
                # For numeric data, add statistical measures
                non_null = data.dropna()
                if len(non_null) > 0:
                    characteristics.update(
                        {
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "mean": float(non_null.mean()),
                            "median": float(non_null.median()),
                            "std": float(non_null.std()),
                        }
                    )
            elif isinstance(
                data.dtype, pd.CategoricalDtype
            ) or pd.api.types.is_object_dtype(data):
                # For categorical data, sample most common values
                top_values = data.value_counts().head(10)
                # Convert to dictionary with string keys and int values for JSON serialization
                top_values_dict: dict[str, int] = {}
                for k, v in top_values.items():
                    top_values_dict[str(k)] = int(v)
                characteristics["top_values"] = top_values_dict

            # Convert to JSON string and hash
            json_str = json.dumps(characteristics, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")
            # Fallback to a simple hash of the data length and type
            return hashlib.md5(f"{len(data)}_{str(data.dtype)}".encode()).hexdigest()

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
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get basic operation parameters
        parameters = {
            "field_name": self.field_name,
            "mode": self.mode,
            "null_strategy": self.null_strategy,
            "version": self.version,
        }

        # Add operation-specific parameters through method that subclasses can override
        parameters.update(self._get_cache_parameters())

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(data)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash,
        )

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
            from pamola_core.utils.ops.op_cache import operation_cache

            if self.field_name not in df.columns:
                self.logger.warning(
                    f"Field '{self.field_name}' not found in DataFrame."
                )
                return None

            cache_key = self._generate_cache_key(df[self.field_name])
            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = operation_cache.get_cache(
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

        # Restore main output and metrics artifacts
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
        visualization_paths: Dict[str, Path],
        task_dir: Path,
        metrics_result_path: Optional[str] = "",
        output_result_path: Optional[str] = "",
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

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_data)

            # Prepare metadata for cache
            operation_params = self._get_cache_parameters()
            operation_params.update(
                {
                    "field_name": self.field_name,
                    "mode": self.mode,
                    "null_strategy": self.null_strategy,
                    "operation_class": self.__class__.__name__,
                    "version": self.version,
                }
            )
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
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")

            success = operation_cache.save_cache(
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

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch
        """
        # This method should be implemented by subclasses
        field_data = batch[self.field_name]
        batch[self.output_field_name] = field_data
        return batch
        # raise NotImplementedError("Subclasses must implement process_batch method")

    def process_value(self, value, **params):
        """
        Process a single value.

        Parameters:
        -----------
        value : Any
            Value to process
        **params : dict
            Additional parameters for processing

        Returns:
        --------
        Any
            Processed value
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement process_value method")

    def process_with_dask(
        self, df: Any, npartitions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process a DataFrame with Dask.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        chunk_size : int, optional
            Number of rows to process in each chunk (default: 10000)
        npartitions : int, optional
            Number of partitions to use for Dask (default: None)

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with Dask
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement process_with_dask method")

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
        # Basic metrics that apply to all anonymization operations
        metrics: Dict[str, Any] = {
            "operation_type": self.__class__.__name__,
            "field_name": self.field_name,
            "mode": self.mode,
            "null_strategy": self.null_strategy,
            "total_records": len(original_data),
            "null_count": int(original_data.isna().sum()),
            "unique_values_before": int(original_data.nunique()),
            "unique_values_after": int(anonymized_data.nunique()),
        }

        # If the field is numeric, add numeric metrics
        if pd.api.types.is_numeric_dtype(original_data):
            try:
                # Filter out nulls for calculations
                orig_non_null = original_data.dropna()
                anon_non_null = anonymized_data.dropna()

                # Handle case where anonymized values might be strings (e.g., in binning)
                try:
                    anon_numeric = pd.to_numeric(anon_non_null, errors="coerce")
                    anon_is_numeric = not anon_numeric.isna().all()
                except Exception:
                    anon_is_numeric = False
                    anon_numeric = None  # Add a default definition

                # Add additional metrics if anonymized data is still numeric
                if anon_is_numeric and anon_numeric is not None:
                    try:
                        # Compare statistical properties
                        metrics.update(
                            {
                                "mean_original": float(orig_non_null.mean()),
                                "mean_anonymized": float(anon_numeric.mean()),
                                "std_original": float(orig_non_null.std()),
                                "std_anonymized": float(anon_numeric.std()),
                                "min_original": float(orig_non_null.min()),
                                "min_anonymized": float(anon_numeric.min()),
                                "max_original": float(orig_non_null.max()),
                                "max_anonymized": float(anon_numeric.max()),
                                "median_original": float(orig_non_null.median()),
                                "median_anonymized": float(anon_numeric.median()),
                            }
                        )

                        # Calculate mean absolute difference if possible
                        if orig_non_null.index.equals(anon_numeric.index):
                            metrics["mean_absolute_difference"] = float(
                                (orig_non_null - anon_numeric).abs().mean()
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not compute all numeric comparison metrics: {e}"
                        )
            except Exception as e:
                self.logger.warning(f"Error calculating numeric metrics: {e}")

        # Calculate generalization ratio (reduction in unique values)
        unique_values_before = metrics.get("unique_values_before", 0)
        if unique_values_before > 0:
            unique_values_after = metrics.get("unique_values_after", 0)
            metrics["generalization_ratio"] = 1.0 - (
                unique_values_after / unique_values_before
            )
        else:
            metrics["generalization_ratio"] = 0.0

        # Add operation-specific metrics (to be implemented by subclasses)
        specific_metrics = self._collect_specific_metrics(
            original_data, anonymized_data
        )
        if specific_metrics:
            metrics.update(specific_metrics)

        return metrics

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect operation-specific metrics.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing

        Returns:
        --------
        Dict[str, Any]
            Dictionary with operation-specific metrics
        """
        # This method should be overridden by subclasses to add specific metrics
        return {}

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

            # Determine if data is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(original_for_viz)
            # Create a basic distribution comparison
            if is_numeric:
                # Clean data (drop nulls) for visualization
                original_clean = original_for_viz.dropna()

                # Try to convert anonymized data to numeric if possible
                try:
                    anonymized_numeric = pd.to_numeric(
                        anonymized_for_viz.dropna(), errors="coerce"
                    )
                    has_numeric_data = not anonymized_numeric.isna().all()
                except:
                    has_numeric_data = False
                    anonymized_numeric = None

                if has_numeric_data and anonymized_numeric is not None:
                    self.logger.info(
                        "[VIZ] Data is still numeric after rounding, using histogram"
                    )
                    # Determining the adaptive number of bins
                    n_bins = min(20, max(5, int(np.sqrt(len(original_clean)))))

                    self.logger.debug(f"[VIZ] Using {n_bins} bins for histogram")

                    # Create histogram comparison
                    hist_filename = f"{self.field_name}_{self.__class__.__name__}_histogram_{timestamp}.png"
                    hist_path = viz_dir / hist_filename

                    # Prepare data for histogram - use Any to avoid typing conflicts
                    hist_data: Any = {
                        "Original": original_clean.tolist(),
                        "Anonymized": anonymized_numeric.dropna().tolist(),
                    }

                    self.logger.debug(
                        f"[VIZ] Comparison data prepared: Original={len(hist_data['Original'])}, Anonymized={len(hist_data['Anonymized'])}"
                    )
                    # Step 3: Save visualization
                    if progress_tracker:
                        progress_tracker.update(1, {"step": "Saving visualization"})

                    # Create histogram using standard utility with context support
                    self.logger.info(f"[VIZ] Calling create_histogram...")
                    call_start = time.time()

                    # Create histogram comparison with context support
                    save_path = create_histogram(
                        data=hist_data,
                        output_path=str(hist_path),
                        title=f"Distribution Comparison for {self.field_name}",
                        x_label=self.field_name,
                        y_label="Frequency",
                        bins=n_bins,
                        kde=True,
                        theme=vis_theme,
                        backend=vis_backend or "plotly",
                        strict=vis_strict,
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
                        visualization_paths["distribution_comparison"] = save_path
                    else:
                        self.logger.error(
                            f"[VIZ] Failed to create histogram: {hist_path}"
                        )
            else:
                self.logger.info("[VIZ] Data is categorical, using bar chart")
                # For categorical data, create a bar chart
                # Get value distribution (top N categories)
                max_categories = 10
                orig_counts = original_data.value_counts().head(max_categories)
                anon_counts = anonymized_data.value_counts().head(max_categories)

                # Create bar chart
                bar_filename = f"{self.field_name}_{self.__class__.__name__}_categories_{timestamp}.png"
                bar_path = viz_dir / bar_filename

                # Prepare data for bar plot - use Any to avoid typing conflicts
                bar_data: Any = {
                    "Original": orig_counts.to_dict(),
                    "Anonymized": anon_counts.to_dict(),
                }

                self.logger.debug(
                    f"[VIZ] Bar data prepared: Original={len(bar_data['Original'])}, Anonymized={len(bar_data['Anonymized'])}"
                )

                # Step 3: Save visualization
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Saving visualization"})

                # Create bar plot using standard utility with context support
                self.logger.info(f"[VIZ] Calling create_bar_plot...")
                call_start = time.time()

                # Create bar plot with context support
                save_path = create_bar_plot(
                    data=bar_data,
                    output_path=str(bar_path),
                    title=f"Category Comparison for {self.field_name}",
                    x_label="Category",
                    y_label="Count",
                    orientation="v",
                    sort_by="value",
                    max_items=max_categories,
                    theme=vis_theme,
                    backend=vis_backend or "plotly",
                    strict=vis_strict,
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
                    visualization_paths["category_comparison"] = save_path
                else:
                    self.logger.error(f"[VIZ] Failed to create bar plot: {bar_path}")

        except Exception as e:
            self.logger.error(
                f"[VIZ] Error creating visualization: {type(e).__name__}: {e}"
            )
            self.logger.error(f"[VIZ] Stack trace:", exc_info=True)

        return visualization_paths

    def _is_visualizable(self, data: pd.Series) -> bool:
        """
        Check if the data can be visualized.

        Parameters:
        -----------
        data : pd.Series
            Data to check

        Returns:
        --------
        bool
            Whether the data can be visualized
        """
        # Check if series is empty
        if len(data) == 0:
            return False

        # Check if all values are null
        if data.isnull().all():
            return False

        # If data is numeric or categorical, it can be visualized
        column_dtype = data.dtype
        is_categorical = isinstance(column_dtype, pd.CategoricalDtype)
        is_string = pd.api.types.is_string_dtype(data)
        is_object = pd.api.types.is_object_dtype(data)
        is_numeric = pd.api.types.is_numeric_dtype(data)

        if is_numeric or is_categorical or is_string or is_object:
            return True

        return False

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
