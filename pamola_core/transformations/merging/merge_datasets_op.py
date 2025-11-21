"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Merge Datasets Operation
Description: Operation for merging datasets with various strategies
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides an operation for merging datasets with various strategies
while maintaining data utility. It implements various strategies:

1. Binning: Groups numeric values into discrete intervals (bins)
2. Rounding: Reduces precision by rounding to specified decimal places
3. Range-based: Maps values to custom ranges with special handling for outliers

Key features:
- Direct in-place DataFrame modification with both REPLACE and ENRICH modes
- Robust null value handling with configurable strategies (PRESERVE, EXCLUDE, ERROR)
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for distribution comparisons
- Chunked processing support for large datasets
- Graceful handling of already-processed non-numeric fields
- Memory-efficient operation with explicit cleanup for large datasets

Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pamola_core.common.enum.relationship_type import RelationshipType
from pamola_core.transformations.commons.processing_utils import (
    merge_dataframes,
)
from pamola_core.transformations.commons.visualization_utils import (
    sample_large_dataset,
)
from pamola_core.transformations.commons.merging_utils import (
    generate_dataset_size_comparison_vis,
    generate_field_overlap_vis,
    generate_join_type_distribution_vis,
    generate_record_overlap_vis,
)
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.transformations.schemas.merge_datasets_op_core_schema import (
    MergeDatasetsOperationConfig,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.common.constants import Constants


@register(version="1.0.0")
class MergeDatasetsOperation(TransformationOperation):
    """Operation to merge two datasets based on key fields."""

    def __init__(
        self,
        name: str = "merge_datasets_operation",
        left_dataset_name: str = "main",
        right_dataset_name: Optional[str] = None,
        right_dataset_path: Optional[Path] = None,
        left_key: Optional[str] = None,
        right_key: Optional[str] = None,
        join_type: str = "left",
        relationship_type: str = "auto",
        suffixes: Tuple[str, str] = ("_x", "_y"),
        **kwargs,
    ):
        """
        Initialize the merge datasets operation.

        Parameters:
        -----------
        name : str
            Name of the operation (default: "merge_datasets_operation")
        left_dataset_name : str
            Name of the left (main) dataset.
        right_dataset_name : str, optional
            Name of the right (lookup) dataset.
        right_dataset_path : Path, optional
            Path to external right-side dataset file.
        left_key : str
            Key field in the left dataset for join.
        right_key : str, optional
            Key field in the right dataset for join. Defaults to left_key if not set.
        join_type : str
            Join strategy: "inner", "left", "right", or "outer".
        suffixes : Tuple[str, str]
            Column suffixes for overlapping columns.
        relationship_type : str, optional
            Type of relationship between datasets ('auto', 'one-to-one', 'one-to-many').
            When 'auto', the function will detect the relationship type based on key uniqueness.
        **kwargs : dict
            Additional keyword arguments for TransformationOperation.
        """
        # Normalize and defaults
        right_key = right_key or left_key
        right_dataset_path = str(right_dataset_path) if right_dataset_path else None
        suffixes = list(suffixes or ("_x", "_y"))

        # Ensure default metadata
        kwargs.setdefault("name", name)
        kwargs.setdefault(
            "description",
            f"Merge '{left_dataset_name}' and '{right_dataset_name}' using '{left_key}'",
        )

        # --- Build config object ---
        config = MergeDatasetsOperationConfig(
            left_dataset_name=left_dataset_name,
            right_dataset_name=right_dataset_name,
            right_dataset_path=right_dataset_path,
            left_key=left_key,
            right_key=right_key,
            join_type=join_type,
            relationship_type=relationship_type,
            suffixes=suffixes,
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
        self.right_df = None

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

            left_df = None
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=directories["cache"],
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=directories["output"],
                logger=self.logger,
                progress_tracker=progress_tracker,
            )

            # Save configuration to task directory
            self.save_config(task_dir)

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
                            "step": "Starting merge datasets",
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
                relationship_type=self.relationship_type,
                left_key=self.left_key,
                left_dataset_name=self.left_dataset_name,
                right_dataset_name=self.right_dataset_name,
                right_dataset_path=self.right_dataset_path,
            )

            # Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                # Step 2: Check if we have a cached result
                if main_progress:
                    current_steps += 1
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS, current_steps, "Checking cache", main_progress
                    )

                # Load left dataset for check cache
                self.logger.info("Load left dataset for check cache...")
                left_df = self._get_dataset(
                    data_source, self.left_dataset_name, **kwargs
                )
                self.logger.info(
                    f"Left dataset '{self.left_dataset_name}' loaded with {len(left_df)} records."
                )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df=left_df, reporter=reporter)
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

            # Step 3: Loading progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Data Loading", main_progress
                )

            # Loading left dataset
            if left_df is None:
                left_df = self._get_dataset(
                    data_source, self.left_dataset_name, **kwargs
                )

            # Loading right dataset
            self.right_df = (
                self._get_dataset(data_source, self.right_dataset_name, **kwargs)
                if self.right_dataset_name is not None
                else self._get_dataset(data_source, self.right_dataset_path, **kwargs)
            )

            # Detect relationship type if set to 'auto'
            if self.relationship_type == RelationshipType.AUTO.value:
                self.relationship_type = self._detect_relationship_type_auto(
                    left_df=left_df,
                    right_df=self.right_df,
                    left_key=self.left_key,
                    right_key=self.right_key,
                )

            # Validate relationship on all left_df and right_df
            self._validate_relationship(left_df, self.right_df)

            # Step 4: Processing progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Processing data", main_progress
                )

            try:
                self.logger.info(f"Processing with left_key: {self.left_key}")

                # Create child progress tracker for chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="Merging processing",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Process the data with the selected keys
                self.process_count = len(left_df)
                processed_df = merge_dataframes(
                    left_df=left_df,
                    right_df=self.right_df,
                    left_key=self.left_key,
                    right_key=self.right_key,
                    join_type=self.join_type,
                    suffixes=self.suffixes,
                    chunk_size=self.chunk_size,
                    use_dask=self.use_dask,
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
                metrics = self._collect_metrics(
                    left_df=left_df, right_df=self.right_df, processed_df=processed_df
                )

                # Generate metrics file name (in self.name existed left_key)
                metrics_file_name = f"{self.left_key}_{self.operation_name}_metrics_{operation_timestamp}"

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
                    description=f"Merging on {self.left_key} ↔ {self.right_key} — datasets transformation metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report the metrics artifact to the reporter if available
                if reporter:
                    reporter.add_operation(
                        f"Merging on {self.left_key} ↔ {self.right_key} — datasets transformation metrics",
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
                        left_df=left_df,
                        right_df=self.right_df,
                        merged_df=processed_df,
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
                        left_df=left_df,
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
            self._cleanup_memory(processed_df, left_df, self.right_df)

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

    def _validate_relationship(
        self, left_df: pd.DataFrame, right_df: pd.DataFrame
    ) -> None:
        """Validate the relationship type between left and right data."""
        left_unique = left_df[self.left_key].is_unique
        right_unique = right_df[self.right_key].is_unique

        if self.relationship_type == RelationshipType.ONE_TO_ONE.value:
            if not left_unique or not right_unique:
                raise ValueError(
                    f"Expected one-to-one relationship, but got left_unique={left_unique}, right_unique={right_unique}."
                )
        elif self.relationship_type == RelationshipType.ONE_TO_MANY.value:
            if not right_unique:
                raise ValueError(
                    f"Expected one-to-many relationship, but right key '{self.right_key}' is not unique."
                )
            if not left_unique:
                self.logger.warning(
                    f"[Relationship Warning] Left key '{self.left_key}' is not unique (one-to-many). Possible data duplication."
                )
        else:
            raise ValueError(
                f"Unsupported relationship_type: '{self.relationship_type}'. Expected 'one-to-one' or 'one-to-many'."
            )

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
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict:
        """
        Collect transformation and merge-related metrics for reporting and analysis.

        Parameters
        ----------
        left_df : pd.DataFrame
            The original left DataFrame used in the transformation (e.g., the batch).
        right_df : pd.DataFrame
            The original right DataFrame used for joining with the left_df.
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
            and other merge-specific metrics.
        """
        execution_time = (
            self.end_time - self.start_time
            if self.end_time and self.start_time
            else None
        )
        total_input_records = len(left_df)
        total_output_records = len(processed_df)
        total_input_fields = len(left_df.columns) + len(right_df.columns)
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
        metrics.update(self._collect_merge_metrics(left_df, right_df, processed_df))
        return metrics

    def _collect_merge_metrics(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict:

        key_columns = self._get_processed_key_columns(left_df, right_df, processed_df)
        if not key_columns:
            return {}

        left_key_col, right_key_col = key_columns

        # Matching records: records with non-null values in both key columns
        matched = processed_df[
            processed_df[left_key_col].notnull() & processed_df[right_key_col].notnull()
        ]
        num_matched = len(matched)

        # Only in left: left_df keys not in right_df
        only_left = set(left_df[self.left_key]) - set(right_df[self.right_key])
        num_only_left = len(only_left)

        # Only in right: right_df keys not in left_df
        only_right = set(right_df[self.right_key]) - set(left_df[self.left_key])
        num_only_right = len(only_right)

        # Match percentage
        total_keys = len(
            set(left_df[self.left_key]).union(set(right_df[self.right_key]))
        )
        match_pct = num_matched / total_keys if total_keys > 0 else 0

        # Duplicate keys
        dup_left = left_df[self.left_key].duplicated().sum()
        dup_right = right_df[self.right_key].duplicated().sum()

        # Number of fields before and after
        fields_before = len(left_df.columns) + len(right_df.columns)
        fields_after = len(processed_df.columns)

        return {
            "num_matched_records": num_matched,
            "num_only_in_left": num_only_left,
            "num_only_in_right": num_only_right,
            "match_percentage": round(match_pct * 100, 2),
            "num_duplicate_keys_left": int(dup_left),
            "num_duplicate_keys_right": int(dup_right),
            "num_fields_before": fields_before,
            "num_fields_after": fields_after,
        }

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters for cache key generation
        """
        params = {
            "left_dataset_name": self.left_dataset_name,
            "right_dataset_name": self.right_dataset_name,
            "right_dataset_path": self.right_dataset_path,
            "left_key": self.left_key,
            "right_key": self.right_key,
            "join_type": self.join_type,
            "relationship_type": self.relationship_type,
            "suffixes": self.suffixes,
        }

        return params

    def _handle_visualizations(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        merged_df: pd.DataFrame,
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
                    f"[DIAG] Field: {self.left_key}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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
                        left_df=left_df,
                        right_df=right_df,
                        merged_df=merged_df,
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
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Generate required visualizations for the merge operation using visualization utilities.

        Parameters
        ----------
        left_df : pd.DataFrame
            The left input DataFrame used in the merge operation.
        right_df : pd.DataFrame
            The right input DataFrame used in the merge operation.
        merged_df : pd.DataFrame
            The resulting DataFrame after performing the merge.
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
        dict
            A dictionary mapping visualization types to their corresponding file paths. Example:
            {
                "record_overlap_venn": "path/to/venn.png",
                "dataset_size_comparison_bar_chart": "path/to/bar.png",
                "field_overlap_venn": "path/to/field_venn.png",
                "join_type_distribution_pie_chart": "path/to/pie.png"
            }
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
                f"Skipping visualization for {self.left_key} (backend=None)"
            )
            return visualization_paths

        self.logger.info(f"[VIZ] Starting visualization generation for {self.left_key}")
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            # Sample large datasets for visualization
            if len(left_df) > 10000:
                self.logger.info(
                    f"[VIZ] Sampling large dataset: {len(left_df)} -> 10000 samples"
                )
                left_viz = sample_large_dataset(left_df, max_samples=10000)
                right_viz = sample_large_dataset(right_df, max_samples=10000)
                merged_viz = sample_large_dataset(merged_df, max_samples=10000)
            else:
                left_viz = left_df
                right_viz = right_df
                merged_viz = merged_df

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(left_viz)} samples"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # 1. Record overlap (Venn or bar chart)
            visualization_paths.update(
                generate_record_overlap_vis(
                    left_viz,
                    right_viz,
                    self.left_key,
                    self.right_key,
                    field_label="record_overlap",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    visualization_paths=visualization_paths,
                    **kwargs,
                )
            )

            # 2. Dataset size comparison (bar chart)
            visualization_paths.update(
                generate_dataset_size_comparison_vis(
                    left_viz,
                    right_viz,
                    merged_viz,
                    field_label="dataset_size",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    visualization_paths=visualization_paths,
                    **kwargs,
                )
            )

            # 3. Field overlap (Venn)
            visualization_paths.update(
                generate_field_overlap_vis(
                    left_viz,
                    right_viz,
                    field_label="field_overlap",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    visualization_paths=visualization_paths,
                    **kwargs,
                )
            )

            # 4. Join type distribution (pie chart)
            key_columns = self._get_processed_key_columns(
                left_viz, right_viz, merged_viz
            )
            if not key_columns:
                return {}

            left_key_col, right_key_col = key_columns
            visualization_paths.update(
                generate_join_type_distribution_vis(
                    merged_viz,
                    left_key_col,
                    right_key_col,
                    self.join_type,
                    field_label="join_type",
                    operation_name=self.name,
                    task_dir=viz_dir,
                    timestamp=timestamp,
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    visualization_paths=visualization_paths,
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

        custom_kwargs = self._get_custom_kwargs(result_df, **kwargs)

        # Generate standardized output filename with timestamp
        field_name_output = f"{self.left_key}_{self.operation_name}_output_{timestamp}"

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
        left_df: Optional[pd.DataFrame] = None,
        right_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        temporary attributes and forcing garbage collection.

        Parameters:
        -----------
        processed_df : pd.DataFrame, optional
            Processed DataFrame to clear from memory
        left_df : pd.DataFrame, optional
            Left DataFrame to clear from memory
        right_df : pd.DataFrame, optional
            Right DataFrame to clear from memory
        """
        # Clear argument references
        if processed_df is not None:
            try:
                del processed_df
            except Exception:
                pass

        if left_df is not None:
            try:
                del left_df
            except Exception:
                pass

        if right_df is not None:
            try:
                del right_df
            except Exception:
                pass

        # Clear right DataFrame
        if hasattr(self, "right_df"):
            self.right_df = None

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

    def _get_dataset(
        self, source: Any, dataset_name_or_path: Optional[str], **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves a DataFrame from a given data source using a dataset name or path.

        Parameters:
        -----------
        source : Any
            The data source. Can be:
            - A dictionary-like pandas DataFrame with named datasets
            - A DataSource object
            - A file path (string or Path)
        dataset_name_or_path : str or None
            The name or path of the dataset to retrieve.
        Returns:
        --------
        pd.DataFrame or None
            The loaded DataFrame, or None if the dataset_name_or_path is None.
        """
        if dataset_name_or_path is None:
            return None
        if isinstance(source, pd.DataFrame):
            return source.get(dataset_name_or_path)

        # Load settings operation
        settings_operation = load_settings_operation(
            source, dataset_name_or_path, **kwargs
        )
        return load_data_operation(source, dataset_name_or_path, **settings_operation)

    def _validate_input_params(
        self,
        relationship_type: str,
        left_key: Optional[str],
        left_dataset_name: Optional[str],
        right_dataset_name: Optional[str],
        right_dataset_path: Optional[Path],
    ) -> None:
        """
        Validates inputs for a dataset relationship operation.

        Parameters:
        -----------
        relationship_type : str
            Type of relationship to validate. Must be one of: "auto", "one-to-one", "one-to-many".
        left_key : str or None
            The key column on the left dataset. Must not be None.
        left_dataset_name : Optional[str]
            Optional name of the left dataset. Required if `left_dataset_name` is not provided.
        right_dataset_name : str or None
            Optional name of the right dataset. Required if `right_dataset_path` is not provided.
        right_dataset_path : Optional[Path]
            Optional path to the right dataset. Required if `right_dataset_name` is not provided.
        Raises:
        -------
        ValueError
            If any input is invalid.
        """
        valid_relationship_types = [
            RelationshipType.AUTO.value,
            RelationshipType.ONE_TO_ONE.value,
            RelationshipType.ONE_TO_MANY.value,
        ]
        if relationship_type not in valid_relationship_types:
            raise ValueError(
                f"Invalid relationship_type. Must be one of {valid_relationship_types}"
            )

        if left_key is None:
            raise ValueError("left_key parameter is required")

        if left_dataset_name is None:
            raise ValueError("left_dataset_name parameter is required")

        if right_dataset_name is None and right_dataset_path is None:
            raise ValueError(
                "Either right_dataset_name or right_dataset_path must be provided"
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
            if self.left_key and self.left_key not in df.columns:
                self.logger.warning(
                    f"Field '{self.left_key}' not found in DataFrame columns."
                )
                return None

            target_df = df[self.left_key] if self.left_key else df
            cache_key = self._generate_cache_key(target_df)

            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(f"Using cached result for {self.left_key} transformation")

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
                    f"Merge datasets transformation of {self.left_key} (cached)",
                    details={
                        "left_key": self.left_key,
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
                    description=f"{self.left_key} {desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{self.left_key} {desc_suffix} (cached)",
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
        left_df: Union[pd.Series, pd.DataFrame],
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
        left_df : pd.Series or pd.DataFrame
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
            if isinstance(left_df, pd.DataFrame) and self.left_key:
                if self.left_key in left_df.columns:
                    cache_key = self._generate_cache_key(left_df[self.left_key])
                else:
                    cache_key = self._generate_cache_key(left_df)
            else:
                cache_key = self._generate_cache_key(left_df)

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
                    "original_length": len(left_df),
                    "transformed_length": len(transformed_data),
                    "original_null_count": int(
                        left_df.isna().sum().sum()
                        if isinstance(left_df, pd.DataFrame)
                        else left_df.isna().sum()
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

    def _detect_relationship_type_auto(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_key: str,
        right_key: str,
    ) -> str:
        """
        Automatically detect the relationship type between two datasets based on key uniqueness.

        Only supports:
            - one-to-one
            - one-to-many

        Parameters:
        -----------
        left_df : pd.DataFrame
            The left (main) dataset.
        right_df : pd.DataFrame
            The right (lookup) dataset.
        left_key : str
            The join key in the left dataset.
        right_key : str
            The join key in the right dataset.

        Returns:
        --------
        str
            The detected relationship type.

        Raises:
        -------
        ValueError
            If the relationship is not one-to-one or one-to-many.
        """
        left_key_is_unique = left_df[left_key].is_unique
        right_key_is_unique = right_df[right_key].is_unique

        if left_key_is_unique and right_key_is_unique:
            detected_relationship = RelationshipType.ONE_TO_ONE.value
        elif right_key_is_unique:

            detected_relationship = RelationshipType.ONE_TO_MANY.value
        else:
            raise ValueError(
                "Only 'one-to-one' and 'one-to-many' relationships are supported. "
                "Detected unsupported relationship (many-to-one or many-to-many)."
            )

        self.logger.info(f"Auto-detected relationship type: {detected_relationship}")
        return detected_relationship

    def _get_processed_key_columns(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> Optional[Tuple[str, str]]:
        """
        Determine the key column names in processed_df after merge, considering suffixes.

        Returns
        -------
        (left_key_col, right_key_col) if both exist in processed_df, else None.
        Logs warning if missing.
        """
        left_key_col = (
            f"{self.left_key}{self.suffixes[0]}"
            if self.left_key in right_df.columns
            else self.left_key
        )
        right_key_col = (
            f"{self.right_key}{self.suffixes[1]}"
            if self.right_key in left_df.columns
            else self.right_key
        )

        missing_keys = []
        if left_key_col not in processed_df.columns:
            missing_keys.append(f"processed_df['{left_key_col}']")
        if right_key_col not in processed_df.columns:
            missing_keys.append(f"processed_df['{right_key_col}']")
        if missing_keys:
            self.logger.warning(
                f"Cannot find key columns in processed_df: {', '.join(missing_keys)}"
            )
            return None
        return left_key_col, right_key_col

    def _update_progress_tracker(
        self,
        total_steps: int,
        n: int,
        step_name: str,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> None:
        """
        Helper to update progress tracker for the step.
        """
        if progress_tracker:
            progress_tracker.total = total_steps  # Ensure total steps is set
            progress_tracker.update(
                n,
                {
                    "step": step_name,
                    "operation": f"{self.name}",
                    "left_key": f"{self.left_key}",
                    "right_key": f"{self.right_key}",
                },
            )


# Helper function to create the operation easily
def create_merge_datasets_operation(**kwargs) -> MergeDatasetsOperation:
    """
    Create a merge datasets operation with default settings.

    Parameters:
    -----------
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    MergeDatasetsOperation
        Configured merge datasets operation
    """
    return MergeDatasetsOperation(**kwargs)
