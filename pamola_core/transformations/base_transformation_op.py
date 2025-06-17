"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Base Transformation Operation
Description: Base class for all data transformation operations in PAMOLA Core.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides the foundational class for all transformation operations,
defining common interfaces, behaviors, and utilities for data transformation tasks.

Key features:
- Standardized operation lifecycle: validation, execution, result handling
- Support for in-place (REPLACE) and enrichment (ENRICH) transformation modes
- Configurable batch processing for large datasets
- Optional caching, Dask-based distributed processing, and output encryption
- Integrated progress tracking and artifact reporting
- Flexible handling of output field naming and operation metadata
- Hooks for metrics calculation and visualization generation
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import hashlib
import json
import gc
from pamola_core.common.constants import Constants
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.transformations.commons.processing_utils import (
    process_dataframe_with_config,
)
from pamola_core.transformations.commons.visualization_utils import (
    generate_dataset_overview_vis,
    generate_data_distribution_comparison_vis,
    generate_field_count_comparison_vis,
    generate_record_count_comparison_vis,
    sample_large_dataset,
)

logger = logging.getLogger(__name__)


class TransformationOperation(BaseOperation):
    """Base class for all transformation operations.

    This class provides common functionality for all transformation operations,
    including data source handling, result processing, and metric generation.
    """

    def __init__(
        self,
        field_name: str = "",
        name: str = "transformation_operation",
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        column_prefix: str = "_",
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
        """Initialize the transformation operation.

        Parameters:
        -----------
        field_name : str
            Field name to transform (default: "")
        name : str
            Name of the operation (default: "transformation_operation")
        mode : str
            "REPLACE" to modify the field in-place, or "ENRICH" to create a new field (default: "REPLACE")
        output_field_name : str, optional
            Name for the output field if mode is "ENRICH" (default: None)
        column_prefix : str, optional
            Prefix for new column if mode is "ENRICH" (default: "_")
        description : str, optional
            Operation description (default: "")
        chunk_size : int, optional
            Chunk size for processing large datasets (default: 10000)
        use_dask : bool, optional
            Whether to use Dask for distributed processing (default: False)
        npartitions : int, optional
            Number of partitions for Dask processing (default: None)
        use_vectorization : bool, optional
            Whether to use vectorized operations for performance (default: False)
        parallel_processes : int, optional
            Number of parallel processes to use (default: None, uses all available cores)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        visualization_theme : str, optional
            Theme for visualizations (default: None, uses PAMOLA default)
        visualization_backend : str, optional
            Backend for visualizations (default: None, uses PAMOLA default)
        visualization_strict : bool, optional
            Whether to enforce strict visualization rules (default: False)
        visualization_timeout : int, optional
            Timeout for visualization generation in seconds (default: 120)
        output_format : str
            Output format: "csv" or "parquet" or "json".
        """
        # Use a default description if none provided
        if not description:
            description = f"Transformation operation"

        self.field_label = field_name if field_name else "all_fields"
        self.name = f"{self.field_label}_{name}" if field_name else name
        # Initialize base class
        super().__init__(
            name=f"{self.name}_transformation",
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
        )

        # Store parameters
        self.field_name = field_name
        self.mode = mode.upper()  # Ensure uppercase
        self.output_field_name = output_field_name
        self.column_prefix = column_prefix
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
        self.version = getattr(self, "version", "1.0.0")  # Default version

        if self.use_encryption and not self.encryption_key:
            raise ValueError(
                "Encryption key must be provided when use_encryption is True"
            )

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
        Execute the transformation operation.

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

        # Create a field List for display purposes
        field_info = (
            {"field": self.field_name}
            if self.field_name
            else {"field_label": self.field_label}
        )

        # Save operation configuration
        self.save_config(task_dir)

        # Create writer for consistent output handling
        writer = DataWriter(
            task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
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
                        **field_info,
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
                        {"step": "Checking cache", **field_info},
                    )

                # Load settings and data
                settings_operation = load_settings_operation(
                    data_source, dataset_name, **kwargs
                )
                df = load_data_operation(
                    data_source, dataset_name, **settings_operation
                )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df, reporter)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if main_progress:
                        main_progress.update(
                            current_steps,
                            {"step": "Complete (cached)", **field_info},
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Transformation of {self.field_label} (from cache)",
                            details={"cached": True},
                        )

                    return cache_result

            # Step 2: Data Loading
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Data Loading", **field_info}
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
                    current_steps, {"step": "Validation", **field_info}
                )

            try:
                output_fields = self._prepare_output_fields(df)
                self._report_operation_details(reporter, output_fields)

                original_data = (
                    df[[self.field_name]].copy(deep=True)
                    if self.field_name
                    else df.copy(deep=True)
                )

            except Exception as e:
                error_message = (
                    f"Validation error in '{self.__class__.__name__}': {str(e)}"
                )
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 4: Processing
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", **field_info}
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
                transformed_data = result_df[output_fields]

            except Exception as e:
                error_message = (
                    f"Processing error in '{self.__class__.__name__}': {str(e)}"
                )
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Metrics Calculation", **field_info}
                )

            # Record end time for metrics calculation timing
            self.end_time = time.time()

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initialize metrics dictionary
            metrics = {}

            try:
                # Calculate all relevant metrics based on original and transformed data
                metrics = self._calculate_all_metrics(original_data, transformed_data)

                # Generate metrics file name (in self.name existed field_name)
                metrics_file_name = f"{self.field_label}_{self.__class__.__name__}_metrics_{operation_timestamp}"

                # Write metrics to persistent storage/artifact repository
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if is_encryption_required else None
                    ),
                )

                # Add simple metrics (int, float, str, bool) to the result object
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        result.add_metric(key, value)

                # Register the metrics artifact for tracking and visualization
                result.add_artifact(
                    artifact_type="json",
                    path=metrics_result.path,
                    description=f"{field_info} transformation metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report the metrics artifact to the reporter if available
                if reporter:
                    reporter.add_operation(
                        f"{field_info} transformation metrics",
                        details={"type": "json", "path": str(metrics_result.path)},
                    )

            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution since metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Generating Visualizations", **field_info}
                )

            # Generate visualizations if required
            # Initialize visualization paths dictionary
            visualization_paths = {}
            if generate_visualization and vis_backend is not None:
                try:
                    visualization_paths = self._handle_visualizations(
                        original_data=original_data,
                        transformed_data=transformed_data,
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
                    {"step": "Save Output Data", **field_info},
                )
            # Save output data if required
            if save_output:
                try:
                    output_result_path = self._save_output_data(
                        result_df=result_df,
                        task_dir=task_dir,
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
                        transformed_data=transformed_data,
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
                    f"Transformation of {field_info} completed",
                    details=details,
                )

            ## Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                result_df=result_df,
                original_data=original_data,
                transformed_data=transformed_data,
            )

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error in transformation operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message
            )

    def _validate_and_get_dataframe(
        self, data_source: DataSource, dataset_name: str
    ) -> pd.DataFrame:
        """
        Validate data source and retrieve the main dataframe.

        Parameters
        ----------
        data_source : DataSource
            The data source to validate and load data from.
        dataset_name : str
            The name of the dataset to load.

        Returns
        -------
        pd.DataFrame
            The validated dataframe.

        Raises
        ------
        ValueError
            If no valid dataframe is found or the specified field is missing.
        """
        # Attempt to get the main DataFrame from the data source
        df = load_data_operation(data_source, dataset_name)

        if df is None:
            error_msg = (
                f"Data source '{data_source}' does not contain a valid DataFrame"
            )
            raise ValueError(error_msg)

        if self.field_name:
            if self.field_name not in df.columns:
                error_msg = f"Field '{self.field_name}' not found in DataFrame columns"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug(
                    f"Field '{self.field_name}' found in DataFrame columns"
                )

        return df

    def _prepare_output_fields(self, df: pd.DataFrame) -> List[str]:
        """
        Prepare output field names based on whether `field_name` is specified.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame whose columns are used to determine output fields.

        Returns
        -------
        List[str]
            List of output field names:
            - If `field_name` is empty, returns all columns (for processing all).
            - If `mode` is "REPLACE", returns a list containing the `field_name`.
            - Otherwise (mode "ENRICH"), returns a list with the output field name,
            either specified or generated by prefixing `field_name`.
        """
        if not self.field_name:
            self.logger.debug(
                "No field_name provided; returning all DataFrame columns as output fields."
            )
            return df.columns.tolist()

        if self.mode == "REPLACE":
            self.logger.debug(
                f"Mode REPLACE: output fields set to ['{self.field_name}']."
            )
            return [self.field_name]

        output_field = (
            self.output_field_name or f"{self.column_prefix}{self.field_name}"
        )
        if output_field in df.columns:
            self.logger.warning(
                f"Output field '{output_field}' already exists in DataFrame and will be overwritten."
            )

        self.logger.debug(f"Mode ENRICH: output field set to ['{output_field}'].")
        return [output_field]

    def _report_operation_details(
        self, reporter: Optional[Any], output_fields: List[str]
    ) -> None:
        """
        Report details of the operation to the reporter.

        Parameters
        ----------
        reporter : Optional[Any]
            The reporter to log details to. If None, reporting is skipped.
        output_fields : List[str]
            The list of output field names.
        """
        if not reporter:
            self.logger.debug("No reporter provided, skipping reporting.")
            return

        operation_desc = (
            f"Transformation field: {self.field_name}"
            if self.field_name
            else "Transformation on all applicable fields"
        )

        reporter.add_operation(
            operation_desc,
            details={
                "field_name": self.field_name or "N/A",
                "output_fields": (
                    output_fields[0]
                    if len(output_fields) == 1
                    else ", ".join(output_fields)
                ),
                "mode": self.mode,
                "operation_type": self.__class__.__name__,
            },
        )

    def _process_dataframe(
        self, df: pd.DataFrame, progress_tracker: Optional[HierarchicalProgressTracker]
    ) -> pd.DataFrame:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
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

        processed_df = process_dataframe_with_config(
            df=df,
            process_function=self.process_batch,
            chunk_size=self.chunk_size,
            use_dask=self.use_dask,
            npartitions=self.npartitions,
            meta=None,
            use_vectorization=self.use_vectorization,
            parallel_processes=self.parallel_processes,
            progress_tracker=progress_tracker,
        )

        return processed_df

    def _calculate_all_metrics(
        self,
        original_data: Union[pd.Series, pd.DataFrame],
        transformed_data: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for the transformation operation.

        Supports both single-field (Series) and multi-field (DataFrame) inputs.

        Parameters
        ----------
        original_data : Union[pd.Series, pd.DataFrame]
            The original data before transformation.
        transformed_data : Union[pd.Series, pd.DataFrame]
            The transformed data after processing.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing calculated metrics.
        """
        # Calculate basic metrics (e.g., accuracy, difference, etc.)
        metrics = self._collect_metrics(original_data, transformed_data)

        # Calculate execution time in seconds, safely handling missing timestamps
        execution_time = (
            self.end_time - self.start_time
            if self.end_time is not None and self.start_time is not None
            else 0
        )

        # Calculate processing rate (records per second), avoid division by zero
        records_per_second = (
            self.process_count / execution_time if execution_time > 0 else 0
        )

        # Update metrics dictionary with performance metrics
        metrics.update(
            {
                "execution_time_seconds": execution_time,
                "records_processed": self.process_count,
                "records_per_second": records_per_second,
                "processing_date": datetime.now(),
            }
        )

        return metrics

    def _handle_visualizations(
        self,
        original_data: pd.Series,
        transformed_data: pd.Series,
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
            The original data before transformation
        transformed_data : pd.Series
            The transformed data after processing
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
                        transformed_data=transformed_data,
                        task_dir=task_dir,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend or "plotly",
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

    def _generate_visualizations(
        self,
        original_data: Union[pd.Series, pd.DataFrame],
        transformed_data: Union[pd.Series, pd.DataFrame],
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Generate visualizations and metadata summaries using all core visualization utilities.

        Parameters
        ----------
        original_data : pd.Series or pd.DataFrame
            Original data before transformation.
        transformed_data : pd.Series or pd.DataFrame
            Transformed data after processing.
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
        operation_name = self.__class__.__name__
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
                transformed_for_viz = sample_large_dataset(
                    transformed_data, max_samples=10000
                )
            else:
                original_for_viz = original_data
                transformed_for_viz = transformed_data

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_for_viz)} samples"
            )

            # Log dtype info depending on whether data is Series or DataFrame
            if isinstance(original_for_viz, pd.Series) and isinstance(
                transformed_for_viz, pd.Series
            ):
                self.logger.debug(
                    f"[VIZ] Original data type: {original_for_viz.dtype}, "
                    f"transformed data type: {transformed_for_viz.dtype}"
                )

            elif isinstance(original_for_viz, pd.DataFrame) and isinstance(
                transformed_for_viz, pd.DataFrame
            ):
                original_cols = list(original_for_viz.dtypes.to_dict().items())
                transformed_cols = list(transformed_for_viz.dtypes.to_dict().items())
                self.logger.debug(
                    f"[VIZ] Original DataFrame columns: {original_cols}, "
                    f"transformed DataFrame columns: {transformed_cols}"
                )

            # Convert original and transformed Series to DataFrames (if Series)
            original_df: pd.DataFrame = (
                original_for_viz.to_frame()
                if isinstance(original_for_viz, pd.Series)
                else original_for_viz
            )
            transformed_df: pd.DataFrame = (
                transformed_for_viz.to_frame()
                if isinstance(transformed_for_viz, pd.Series)
                else transformed_for_viz
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # Generate visualizations for each column (supports both Series and DataFrame)
            if original_data is not None and transformed_data is not None:
                for column in original_df.columns:
                    # Handle ENRICH mode column renaming
                    transformed_column = column
                    if self.mode == "ENRICH":
                        transformed_column = (
                            self.output_field_name or f"{self.column_prefix}{column}"
                        )
                        self.logger.debug(
                            f"Mode ENRICH: output field set to ['{transformed_column}']."
                        )

                    # Generate field count comparison visualization

                    visualization_paths.update(
                        generate_field_count_comparison_vis(
                            original_df=original_df[[column]],
                            transformed_df=transformed_df[[transformed_column]],
                            field_label=column,
                            operation_name=operation_name,
                            task_dir=viz_dir,
                            timestamp=timestamp,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            visualization_paths=visualization_paths,
                        )
                    )

                    # Record count comparison for each column
                    transformed_dfs: Dict[str, pd.DataFrame] = {
                        column: transformed_df[[transformed_column]]
                    }
                    visualization_paths.update(
                        generate_record_count_comparison_vis(
                            original_df=original_df[[column]],
                            transformed_dfs=transformed_dfs,
                            field_label=column,
                            operation_name=operation_name,
                            task_dir=viz_dir,
                            timestamp=timestamp,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            visualization_paths=visualization_paths,
                        )
                    )

                    # Data distribution comparison for each column
                    visualization_paths.update(
                        generate_data_distribution_comparison_vis(
                            original_df=original_df[column],
                            transformed_data=transformed_df[transformed_column],
                            field_label=column,
                            operation_name=operation_name,
                            task_dir=viz_dir,
                            timestamp=timestamp,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            visualization_paths=visualization_paths,
                        )
                    )

                # Dataset overview visualizations for original
                if original_df is not None:
                    visualization_paths.update(
                        generate_dataset_overview_vis(
                            df=original_df,
                            operation_name=operation_name,
                            dataset_type="original",
                            field_label=self.field_label,
                            task_dir=viz_dir,
                            timestamp=timestamp,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            visualization_paths=visualization_paths,
                        )
                    )

                # Dataset overview visualizations for transformed
                if transformed_df is not None:
                    visualization_paths.update(
                        generate_dataset_overview_vis(
                            df=transformed_df,
                            operation_name=operation_name,
                            dataset_type="transformed",
                            field_label=self.field_label,
                            task_dir=viz_dir,
                            timestamp=timestamp,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            visualization_paths=visualization_paths,
                        )
                    )

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")

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
            Optional timestamp for the output file
        **kwargs : dict
            Additional parameters for the operation
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        # Generate standardized output filename with timestamp
        field_name_output = (
            f"{self.field_label}_{self.__class__.__name__}_output_{timestamp}"
        )

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
            description=f"{self.name} transformed data",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_artifact(
                self.output_format,
                str(output_result.path),
                f"{self.name} transformed data",
            )

        return str(output_result.path)

    def _generate_cache_key(self, data: Union[pd.Series, pd.DataFrame]) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        # Get basic operation parameters
        parameters = self._get_basic_parameters()

        # Add operation-specific parameters (could be overridden by subclasses)
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

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to check in the cache
        reporter : Any
            Reporter to log cache hits/misses

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            cache_key_df = df if not self.field_name else df.get(self.field_name)
            if cache_key_df is None:
                self.logger.warning(
                    f"Field '{self.field_name}' not found in DataFrame columns."
                )
                return None

            cache_key = self._generate_cache_key(cache_key_df)
            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(
                f"Using cached result for {self.field_label} transformation"
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
                    f"Transformation of {self.field_label} (cached)",
                    details={
                        "field_label": self.field_label,
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
                    description=f"{self.field_label} {desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{self.field_label} {desc_suffix} (cached)",
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
            "transformed data",
            Constants.Artifact_Category_Output,
        )
        restore_file_artifact(
            cached.get("metrics_file"),
            "json",
            "transformation metrics",
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
        original_data: Union[pd.Series, pd.DataFrame],
        transformed_data: Union[pd.Series, pd.DataFrame],
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
        original_data : pd.Series or pd.DataFrame
            Original input data
        transformed_data : pd.Series or pd.DataFrame
            Transformed output data
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

        # If no metrics are provided, initialize as an empty dictionary
        if metrics is None:
            metrics = {}

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(original_data)

            # Prepare metadata for cache
            operation_params = self._get_cache_parameters()
            operation_params.update(
                {
                    "field_name": self.field_label,
                    "mode": self.mode,
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
                    "transformed_length": len(transformed_data),
                    "original_null_count": int(original_data.isna().sum()),
                    "transformed_null_count": int(transformed_data.isna().sum()),
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
                    f"Successfully saved {self.field_name} transformation results to cache"
                )
            else:
                self.logger.warning(
                    f"Failed to save {self.field_name} transformation results to cache"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _cleanup_memory(
        self,
        result_df: Optional[Union[pd.Series, pd.DataFrame]] = None,
        original_data: Optional[Union[pd.Series, pd.DataFrame]] = None,
        transformed_data: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        references and optionally calling garbage collection.

        Parameters:
        -----------
        result_df : pd.DataFrame, optional
            Processed DataFrame to clear from memory
        original_data: Union[pd.Series, pd.DataFrame], optional
            Original data to clear from memory
        transformed_data: Union[pd.Series, pd.DataFrame], optional
            Transformed data to clear from memory
        """
        # Delete references
        if result_df is not None:
            del result_df
        if original_data is not None:
            del original_data
        if transformed_data is not None:
            del transformed_data

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Optional: Force garbage collection for large datasets
        # Uncomment if memory pressure is an issue
        # import gc
        # gc.collect()

    def _generate_data_hash(self, data: Union[pd.Series, pd.DataFrame]) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame]
            Input data for the operation. Can be a pandas Series or DataFrame.

        Returns
        -------
        str
            Hash string representing the data's key characteristics.
        """
        try:
            if isinstance(data, pd.Series):
                characteristics = self._series_characteristics(data)
            elif isinstance(data, pd.DataFrame):
                characteristics = self._df_characteristics(data)
            else:
                raise TypeError("Input must be a pandas Series or DataFrame")

            json_str = json.dumps(characteristics, sort_keys=True)
            return hashlib.md5(json_str.encode("utf-8")).hexdigest()

        except Exception as e:
            logger.warning(f"Error generating data hash: {e}")
            fallback_str = f"{len(data)}_{type(data)}"
            return hashlib.md5(fallback_str.encode("utf-8")).hexdigest()

    def _series_characteristics(self, s: pd.Series) -> dict:
        """
        Extract key characteristics from a pandas Series for hashing.

        Parameters
        ----------
        s : pd.Series
            The pandas Series to extract characteristics from.

        Returns
        -------
        dict
            Dictionary of characteristics (length, null count, unique count, dtype, and summary stats).
        """
        char = {
            "length": len(s),
            "null_count": int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "dtype": str(s.dtype),
        }
        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if not non_null.empty:
                char.update(
                    {
                        "min": float(non_null.min()),
                        "max": float(non_null.max()),
                        "mean": float(non_null.mean()),
                        "median": float(non_null.median()),
                        "std": float(non_null.std()),
                    }
                )
        elif pd.api.types.is_object_dtype(s) or isinstance(
            s.dtype, pd.CategoricalDtype
        ):
            top_values = s.value_counts().head(10)
            char["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        return char

    def _df_characteristics(self, df: pd.DataFrame) -> dict:
        """
        Extract key characteristics from a pandas DataFrame for hashing.

        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame to extract characteristics from.

        Returns
        -------
        dict
            Dictionary of characteristics (shape, null count, unique count, dtypes, columns, and numeric summary).
        """
        char = {
            "shape": df.shape,
            "null_count": int(df.isna().sum().sum()),
            "unique_count": int(df.nunique().sum()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "columns": list(df.columns),
        }
        # Numeric summary
        numeric_cols = df.select_dtypes(include="number")
        if not numeric_cols.empty:
            char["numeric_summary"] = {
                col: {
                    "min": float(numeric_cols[col].min()),
                    "max": float(numeric_cols[col].max()),
                    "mean": float(numeric_cols[col].mean()),
                    "median": float(numeric_cols[col].median()),
                    "std": float(numeric_cols[col].std()),
                }
                for col in numeric_cols.columns
            }
        # Object/Categorical summary
        object_cols = [
            col
            for col in df.columns
            if pd.api.types.is_object_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ]
        if object_cols:
            char["object_summary"] = {
                col: {
                    "top_values": {
                        str(k): int(v)
                        for k, v in df[col].value_counts().head(10).items()
                    }
                }
                for col in object_cols
            }
        return char

    def _get_basic_parameters(self) -> Dict[str, str]:
        """Get the basic parameters for the cache key generation."""
        return {
            "name": self.name,
            "description": self.description,
            "encryption_key": self.encryption_key,
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
        raise NotImplementedError("Subclasses must implement process_batch method")

    def _collect_metrics(
        self,
        original_data: Union[pd.Series, pd.DataFrame],
        transformed_data: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Collect universal and transformation-specific metrics.

        Parameters
        ----------
        original_data : pd.Series or pd.DataFrame
            Original data before transformation.
        transformed_data : pd.Series or pd.DataFrame
            Transformed data after processing.

        Returns
        -------
        Dict[str, Any]
            Dictionary with collected metrics.
        """
        # Ensure input is DataFrame for consistent metric calculation
        original_df = (
            original_data.to_frame()
            if isinstance(original_data, pd.Series)
            else original_data
        )
        transformed_df = (
            transformed_data.to_frame()
            if isinstance(transformed_data, pd.Series)
            else transformed_data
        )

        total_input_records = len(original_df)
        total_output_records = len(transformed_df)
        total_input_fields = original_df.shape[1]
        total_output_fields = transformed_df.shape[1]

        # Universal metrics
        metrics: Dict[str, Any] = {
            "total_input_records": total_input_records,
            "total_output_records": total_output_records,
            "total_input_fields": total_input_fields,
            "total_output_fields": total_output_fields,
            "transformation_type": self.__class__.__name__,
            "execution_time_seconds": None,
            "records_per_second": None,
            "records_processed": None,
            "processing_date": None,
            "operation_type": self.__class__.__name__,
            "field_name": self.field_label,
            "mode": self.mode,
            "null_count": int(original_df.isna().sum().sum()),
            "unique_values_before": int(original_df.nunique().sum()),
            "unique_values_after": int(transformed_df.nunique().sum()),
        }

        # Add transformation-specific metrics
        specific_metrics = self._collect_specific_metrics(
            original_data, transformed_data
        )
        if specific_metrics:
            metrics.update(specific_metrics)

        return metrics

    def _collect_specific_metrics(
        self,
        original_data: Union[pd.Series, pd.DataFrame],
        transformed_data: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Collect operation-specific metrics.

        Parameters
        ----------
        original_data : pd.Series or pd.DataFrame
            Original data before transformation.
        transformed_data : pd.Series or pd.DataFrame
            Transformed data after processing.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with operation-specific metrics
        """
        # This method should be overridden by subclasses to add specific metrics
        return {}

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
        directories["visualizations"] = task_dir / "visualizations"
        directories["logs"] = task_dir / "logs"
        directories["cache"] = task_dir / "cache"

        # Ensure all directories exist
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories
