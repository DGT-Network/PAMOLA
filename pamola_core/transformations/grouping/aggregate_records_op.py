"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Aggregate Records Operation
Description: Operation for grouping and aggregating records with flexible aggregation functions
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides an operation for grouping and aggregating records
using standard and custom aggregation functions, while maintaining data utility.

Key features:
- Supports all standard aggregation functions: count, sum, mean, median, min, max, std, var, first, last, nunique
- Allows custom aggregation functions per field
- Efficient processing with both pandas and Dask for large datasets
- Robust null value handling
- Comprehensive metrics collection for aggregation impact assessment
- Visualization generation for group and aggregation results
- Chunked processing support for large datasets
- Memory-efficient operation with explicit cleanup

Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pamola_core.common.helpers.custom_aggregations_helper import (
    CUSTOM_AGG_FUNCTIONS,
    STANDARD_AGGREGATIONS,
)
from pamola_core.transformations.commons.processing_utils import (
    aggregate_dataframe,
)
from pamola_core.transformations.commons.visualization_utils import (
    sample_large_dataset,
)
from pamola_core.transformations.commons.aggregation_utils import (
    generate_aggregation_comparison_vis,
    generate_group_size_distribution_vis,
    generate_record_count_per_group_vis,
)
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.transformations.schemas.aggregate_records_op_core_schema import (
    AggregateRecordsOperationConfig,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.common.constants import Constants

# Configure module logger
logger = logging.getLogger(__name__)


@register(version="1.0.0")
class AggregateRecordsOperation(TransformationOperation):
    """Operation to aggregate records based on group by fields."""

    def __init__(
        self,
        name: str = "aggregate_records_operation",
        group_by_fields: List[str] = None,
        aggregations: Dict[str, List[str]] = None,
        custom_aggregations: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        """
        Initialize the AggregateRecordsOperation.

        Parameters
        ----------
        name : str, optional
            Operation name (default: "aggregate_records_operation").
        group_by_fields : list of str
            Fields to group by for aggregation.
        aggregations : dict
            Mapping of field names to list of aggregation functions.
        custom_aggregations : dict, optional
            Custom aggregation functions.
        **kwargs : dict
            Additional keyword arguments for TransformationOperation.
        """
        # Ensure default metadata
        kwargs.setdefault("name", name)
        kwargs.setdefault(
            "description",
            f"Aggregate records by fields {group_by_fields} "
            f"using {len(aggregations)} aggregation(s).",
        )

        # --- Build config object ---
        config = AggregateRecordsOperationConfig(
            group_by_fields=group_by_fields,
            aggregations=aggregations,
            custom_aggregations=custom_aggregations,
            **kwargs,
        )

        # Inject config into kwargs
        kwargs["config"] = config

        # --- Initialize TransformationOperation ---
        super().__init__(
            **kwargs,
        )

        # --- Apply config attributes to self ---
        for key, value in config.to_dict().items():
            setattr(self, key, value)

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
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Initialize timing and result
            self.start_time = time.time()

            # Config logger task for operation
            self.logger = kwargs.get("logger", self.logger)

            self.logger.info(
                f"Starting {self.operation_name} operation at {self.start_time}"
            )

            df = None
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=directories["cache"],
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Validation, 2. Cache check, 3. Data loading, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
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
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS,
                        current_steps,
                        {
                            "step": "Starting aggregation data",
                        },
                        main_progress,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # Step 1: Validation progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Validation", main_progress
                )

            # Validate input parameters
            self._validate_input_params(
                self.group_by_fields, self.aggregations, self.custom_aggregations
            )

            # Step 2: Loading progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Data Loading", main_progress
                )

            # Loading datasets
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source",
                )

            # Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                # Step 3: Check if we have a cached result
                if main_progress:
                    current_steps += 1
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS, current_steps, "Checking cache", main_progress
                    )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df=df, reporter=reporter)
                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if main_progress:
                        self._update_progress_tracker(
                            TOTAL_MAIN_STEPS,
                            current_steps,
                            "Complete (cached)",
                            main_progress,
                        )

                    return cache_result

            # Step 4: Processing progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Processing data", main_progress
                )

            try:
                self.logger.info(
                    f"Processing with group_by_fields: {self.group_by_fields}"
                )

                # Create child progress tracker for chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="Aggregation processing",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Process the data with the selected strategy
                self.process_count = len(df)
                processed_df = aggregate_dataframe(
                    df,
                    group_by_fields=self.group_by_fields,
                    aggregations=self.aggregations,
                    custom_aggregations=self.custom_aggregations,
                    chunk_size=self.chunk_size,
                    use_dask=self.use_dask,
                    npartitions=self.npartitions,
                    progress_tracker=data_tracker,
                    task_logger=self.logger,
                )

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass

                self.logger.info(
                    f"Processed data: {len(processed_df)} records, dtype: {processed_df.dtypes}"
                )

                # Log sample of processed data
                if len(processed_df) > 0:
                    self.logger.debug(
                        f"Sample of processed data (first 5 rows): {processed_df.head(5).to_dict(orient='records')}"
                    )
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS,
                    current_steps,
                    "Metrics Calculation",
                    main_progress,
                )

            # Record end time after processing
            self.end_time = time.time()

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initialize metrics in scope
            metrics = {}

            try:
                metrics = self._collect_metrics(df=df, processed_df=processed_df)

                # Generate metrics file name
                metrics_file_name = f"{self.name}_metrics_{operation_timestamp}"

                # Write metrics to persistent storage/artifact repository
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if self.use_encryption else None
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
                    description=f"Aggregation on {self.group_by_fields} — datasets transformation metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report the metrics artifact to the reporter if available
                if reporter:
                    reporter.add_operation(
                        f"Aggregation on {self.group_by_fields} — datasets transformation metrics",
                        details={"type": "json", "path": str(metrics_result.path)},
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS,
                    current_steps,
                    "Generating Visualizations",
                    main_progress,
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
                        original_df=df,
                        processed_df=processed_df,
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
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Save Output Data", main_progress
                )

            # Save output data if required
            if self.save_output:
                try:
                    output_result_path = self._save_output_data(
                        result_df=processed_df,
                        task_dir=task_dir,
                        is_encryption_required=self.use_encryption,
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
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=df,
                        transformed_data=processed_df,
                        metrics=metrics,
                        visualization_paths=visualization_paths,
                        metrics_result_path=str(metrics_result.path),
                        output_result_path=output_result_path,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Cleanup memory
            self._cleanup_memory(processed_df, df)

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

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Transformation of {self.name} completed",
                    details=details,
                )

            self.logger.info(
                f"Processing completed {self.operation_name} operation in {self.end_time - self.start_time:.2f} seconds"
            )

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error in transformation operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message, exception=e
            )

    def process_batch(self, batch_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Deprecated: No longer used for merging. All merging is now performed in a single step.
        """
        self.logger.warning(
            "process_batch is deprecated and not used for merging. All merging is performed in a single step."
        )
        return batch_df

    def _process_value(self, value: Any, **params) -> Any:
        """
        Deprecated in batch mode. Not used in ML imputation anymore.
        """
        self.logger.warning(
            "_process_value is deprecated and not used in ML-based batch processing."
        )
        return value

    def _collect_metrics(
        self,
        df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict:
        """
        Collect transformation and merge-related metrics for reporting and analysis.

        Parameters
        ----------
        df : pd.DataFrame
            The original DataFrame used in the transformation (e.g., the batch).
        processed_df : pd.DataFrame
            The resulting DataFrame after processing (e.g., after merge/join operation).

        Returns
        -------
        dict
            Dictionary containing metrics such as:
            - total_input_records
            - total_output_records
            - total_input_fields
            - total_output_fields
            - execution_time_seconds
            - records_per_second
            - transformation_type
            and other aggregate-specific metrics.
        """
        execution_time = (
            self.end_time - self.start_time
            if self.end_time and self.start_time
            else None
        )
        total_input_records = len(df)
        total_output_records = len(processed_df)
        total_input_fields = len(df.columns)
        total_output_fields = len(processed_df.columns)
        records_per_second = (
            total_output_records / execution_time
            if execution_time and execution_time > 0
            else None
        )
        transformation_type = self.operation_name

        metrics = {
            "total_input_records": total_input_records,
            "total_output_records": total_output_records,
            "total_input_fields": total_input_fields,
            "total_output_fields": total_output_fields,
            "execution_time_seconds": (
                round(execution_time, 4) if execution_time else None
            ),
            "records_per_second": (
                round(records_per_second, 2) if records_per_second else None
            ),
            "transformation_type": transformation_type,
        }

        # Merge-specific metrics
        metrics.update(self._collect_aggregate_metrics(df, processed_df))
        return metrics

    def _collect_aggregate_metrics(
        self,
        df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict:
        """
        Collect aggregate-specific metrics for reporting and analysis.

        Parameters
        ----------
        df : pd.DataFrame
            The original DataFrame used in the transformation.
        processed_df : pd.DataFrame
            The resulting DataFrame after aggregation.

        Returns
        -------
        dict
            Dictionary containing aggregate-specific metrics:
            - num_groups
            - group_size_min
            - group_size_max
            - group_size_mean
            - group_size_median
            - reduction_ratio
            - aggregated_field_stats (dict)
        """
        metrics = {}

        # Number of groups
        num_groups = len(processed_df)
        metrics["num_groups"] = num_groups

        # Group sizes (number of records per group)
        if self.group_by_fields and all(f in df.columns for f in self.group_by_fields):
            group_sizes = df.groupby(self.group_by_fields).size()
            metrics["group_size_min"] = int(group_sizes.min())
            metrics["group_size_max"] = int(group_sizes.max())
            metrics["group_size_mean"] = float(group_sizes.mean())
            metrics["group_size_median"] = float(group_sizes.median())
        else:
            metrics["group_size_min"] = None
            metrics["group_size_max"] = None
            metrics["group_size_mean"] = None
            metrics["group_size_median"] = None

        # Reduction ratio (output rows / input rows)
        reduction_ratio = len(processed_df) / len(df) if len(df) > 0 else None
        metrics["reduction_ratio"] = (
            round(reduction_ratio, 4) if reduction_ratio is not None else None
        )

        # Statistical properties of aggregated fields
        aggregated_field_stats = {}
        agg_fields = self.aggregations.keys() if self.aggregations else []
        for col in processed_df.columns:
            # Only consider columns that are results of aggregation
            # Exclude group_by fields
            if col in self.group_by_fields:
                continue
            series = processed_df[col]
            if pd.api.types.is_numeric_dtype(series):
                aggregated_field_stats[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                }
        metrics["aggregated_field_stats"] = aggregated_field_stats

        return metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for cache key generation
        """
        params = {
            "group_by_fields": self.group_by_fields,
            "aggregations": self.aggregations,
            "custom_aggregations": self.custom_aggregations,
        }

        return params

    def _handle_visualizations(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate and save visualizations with thread-safe context support.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original DataFrame before transformation
        processed_df : pd.DataFrame
            The transformed DataFrame after processing
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
        **kwargs: Any
            Additional keyword arguments for visualization functions.
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
                    f"[DIAG] Field: {self.group_by_fields}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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
                        original_df=original_df,
                        processed_df=processed_df,
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
                name=f"VizThread-{self.name}",
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
                description=f"{self.name} {viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_operation(
                    f"{self.name} {viz_type} visualization",
                    details={"artifact_type": "png", "path": str(path)},
                )

        return visualization_paths

    def _generate_visualizations(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Generate required visualizations for the aggregation operation using visualization utilities.

        Parameters
        ----------
        original_df : pd.DataFrame
            The original DataFrame used in the aggregation operation.
        processed_df : pd.DataFrame
            The processed DataFrame used in the aggregation operation.
        task_dir : Path
            The base directory where all task-related outputs (including visualizations) will be saved.
        vis_theme : Optional[str]
            The theme to use for visualizations.
        vis_backend : Optional[str]
            The backend to use for rendering visualizations.
        vis_strict : bool
            Whether to enforce strict visualization rules.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Tracker for monitoring progress of the visualization generation.
        timestamp : Optional[str]
            Timestamp to include in visualization filenames.
        **kwargs : Any
            Additional keyword arguments for visualization functions.

        Returns
        -------
        """
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        visualization_paths = {}

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if visualization should be skipped
        if vis_backend is None:
            self.logger.info(
                f"Skipping visualization for {self.group_by_fields} (backend=None)"
            )
            return visualization_paths

        self.logger.info(
            f"[VIZ] Starting visualization generation for {self.group_by_fields}"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            # Sample large datasets for visualization
            if len(original_df) > 10000:
                self.logger.info(
                    f"[VIZ] Sampling large dataset: {len(original_df)} -> 10000 samples"
                )
                original_viz = sample_large_dataset(original_df, max_samples=10000)
                aggrega_viz = sample_large_dataset(processed_df, max_samples=10000)
            else:
                original_viz = original_df
                aggrega_viz = processed_df

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_viz)} samples"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # Create visualizations
            # 1. Bar chart: record count per group
            visualization_paths.update(
                generate_record_count_per_group_vis(
                    agg_df=aggrega_viz,
                    group_by_fields=self.group_by_fields,
                    field_label="record_count",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )
            )

            # 2. Aggregation comparison across groups (one bar chart per agg field)
            agg_fields = aggrega_viz.columns.to_list()
            visualization_paths.update(
                generate_aggregation_comparison_vis(
                    agg_df=aggrega_viz,
                    group_by_fields=self.group_by_fields,
                    agg_fields=agg_fields,
                    field_label="aggregation_comparison",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )
            )

            # 3. Distribution of group sizes (histogram)
            visualization_paths.update(
                generate_group_size_distribution_vis(
                    agg_df=aggrega_viz,
                    group_by_fields=self.group_by_fields,
                    field_label="group_size_distribution",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )
            )

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")

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
        field_name_output = f"{self.name}_output_{timestamp}"

        custom_kwargs = self._get_custom_kwargs(result_df, **kwargs)
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

    def _cleanup_memory(
        self,
        processed_df: Optional[pd.DataFrame] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        temporary attributes and forcing garbage collection.

        Parameters:
        -----------
        processed_df : pd.DataFrame, optional
            Processed DataFrame to clear from memory
        df : pd.DataFrame, optional
            Original DataFrame to clear from memory
        """
        # Clear argument references
        if processed_df is not None:
            try:
                del processed_df
            except Exception:
                pass

        if df is not None:
            try:
                del df
            except Exception:
                pass

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Remove any temporary attributes starting with _temp_
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                setattr(self, attr_name, None)

        # Optional: Force garbage collection for large datasets
        # Uncomment if memory pressure is an issue
        # import gc
        # gc.collect()

    def _validate_input_params(
        self,
        group_by_fields: List[str],
        aggregations: Dict[str, List[str]] = None,
        custom_aggregations: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """
        Validates inputs for a dataset relationship operation.
        """
        if not group_by_fields:
            raise ValueError("At least one group_by_field must be specified")

        # Combine allowed aggregation functions
        allowed_aggs = set(STANDARD_AGGREGATIONS.keys()) | set(
            CUSTOM_AGG_FUNCTIONS.keys()
        )

        # Validate aggregation functions
        if aggregations is not None:
            if not isinstance(aggregations, dict):
                raise ValueError(
                    "aggregations must be a dictionary mapping field names to functions"
                )
            for field, agg_funcs in aggregations.items():
                for agg_func in agg_funcs:
                    if agg_func not in allowed_aggs:
                        raise ValueError(
                            f"Unsupported aggregation function: {agg_func} for field '{field}'. "
                            f"Allowed: {sorted(allowed_aggs)}"
                        )

        # Validate custom_aggregations: must be dict[str, Callable]
        if custom_aggregations is not None:
            if not isinstance(custom_aggregations, dict):
                raise ValueError(
                    "custom_aggregations must be a dictionary mapping field names to functions"
                )
            for field, custom_agg_funcs in custom_aggregations.items():
                for cus_agg_func in custom_agg_funcs:
                    if cus_agg_func not in allowed_aggs:
                        raise ValueError(
                            f"Unsupported custom aggregation function: {cus_agg_func} for field '{field}'. "
                            f"Allowed: {sorted(allowed_aggs)}"
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
            # Create cache key from entire dataframe or from group_by_fields if present
            if self.group_by_fields and not all(
                field in df.columns for field in self.group_by_fields
            ):
                missing = [f for f in self.group_by_fields if f not in df.columns]
                self.logger.warning(f"Missing fields for cache key: {missing}")
                return None

            cache_df = df[self.group_by_fields] if self.group_by_fields else df
            cache_key = self._generate_cache_key(cache_df)

            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_result = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(
                f"Using cached result for {self.group_by_fields} transformation"
            )

            result = OperationResult(status=OperationStatus.SUCCESS)
            self._add_cached_metrics(result, cached_result)
            artifacts_restored = self._restore_cached_artifacts(
                result, cached_result, reporter
            )

            result.add_metric("cached", True)
            result.add_metric("cache_key", cache_key)
            result.add_metric(
                "cache_timestamp", cached_result.get("timestamp", "unknown")
            )
            result.add_metric("artifacts_restored", artifacts_restored)

            # Report cache hit to reporter
            if reporter:
                reporter.add_operation(
                    f"Aggregate records transformation of {self.group_by_fields} (cached)",
                    details={
                        "cached": True,
                        "group_by_fields": self.group_by_fields,
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
                    description=f"{self.name} {desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{self.name} {desc_suffix} (cached)",
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

        if metrics is None:
            metrics = {}

        try:
            # Generate cache key (same logic as _check_cache)
            if isinstance(original_data, pd.DataFrame) and self.group_by_fields:
                if all(
                    field in original_data.columns for field in self.group_by_fields
                ):
                    cache_key = self._generate_cache_key(
                        original_data[self.group_by_fields]
                    )
                else:
                    cache_key = self._generate_cache_key(original_data)
            else:
                cache_key = self._generate_cache_key(original_data)

            # Prepare metadata for cache
            operation_params = self._get_operation_parameters()

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
                    "original_null_count": int(
                        original_data.isna().sum().sum()
                        if isinstance(original_data, pd.DataFrame)
                        else original_data.isna().sum()
                    ),
                    "transformed_null_count": int(
                        transformed_data.isna().sum().sum()
                        if isinstance(transformed_data, pd.DataFrame)
                        else transformed_data.isna().sum()
                    ),
                },
                "output_file": output_result_path,  # Path to main output file
                "metrics_file": metrics_result_path,  # Path to metrics file
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(
                    f"Successfully saved {self.name} operation results to cache"
                )
            else:
                self.logger.warning(
                    f"Failed to save {self.name} operation results to cache"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _update_progress_tracker(
        self,
        TOTAL_MAIN_STEPS: int,
        n: int,
        step_name: str,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> None:
        """
        Helper to update progress tracker for the step.
        """
        if progress_tracker:
            progress_tracker.total = TOTAL_MAIN_STEPS  # Ensure total steps is set
            progress_tracker.update(
                n,
                {
                    "step": step_name,
                    "operation": f"{self.name}",
                    "group_by_fields": f"{self.group_by_fields}",
                    "aggregations": f"{self.aggregations}",
                    "custom_aggregations": f"{self.custom_aggregations}",
                },
            )


# Helper function to create the operation easily
def create_aggregate_records_operation(**kwargs) -> AggregateRecordsOperation:
    """
    Create an aggregate records operation with default settings.

    Parameters:
    -----------
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    AggregateRecordsOperation
        Configured aggregate records operation
    """
    return AggregateRecordsOperation(**kwargs)
