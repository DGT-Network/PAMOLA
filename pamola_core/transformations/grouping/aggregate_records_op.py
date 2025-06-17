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
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.common.constants import Constants

# Configure module logger
logger = logging.getLogger(__name__)


class AggregateRecordsOperationConfig(OperationConfig):
    """Configuration for AggregateRecordsOperation."""

    schema = {
        "type": "object",
        "properties": {
            "group_by_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
            "aggregations": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "custom_aggregations": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "output_format": {"type": "string", "str": ["csv", "json", "parquet"]},
            "use_cache": {"type": "boolean"},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]},
            "use_dask": {"type": "boolean"},
        },
        "required": ["group_by_fields", "aggregations"],
    }


@register(version="1.0.0")
class AggregateRecordsOperation(TransformationOperation):
    """
    Operation to aggregate records based on group by fields.

    This operation supports various join types and allows configuration of
    suffixes, batch size, output format, caching, and encryption.
    """

    def __init__(
        self,
        name: str = "aggregate_records_operation",
        description: str = "Group and aggregate records",
        group_by_fields: List[str] = None,
        aggregations: Dict[str, List[str]] = None,
        custom_aggregations: Optional[Dict[str, Callable]] = None,
        output_format: str = "csv",
        use_cache: bool = True,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        use_dask: bool = False,
    ):
        """
        Initialize the group and aggregate records operation.

        Parameters:
        -----------
        name : str
            Name of the operation (default: "aggregate_records_operation")
        group_by_fields : List[str]
            Fields to group by for aggregation.
        aggregations : Dict[str, List[str]]
            Aggregation functions to apply to each field.
        custom_aggregations : Optional[Dict[str, Callable]]
            Custom aggregation functions.
        output_format : str
            Output format: "csv" or "parquet" or "json".
        use_cache : bool
            Enable or disable operation caching.
        use_encryption : bool
            Enable output encryption.
        encryption_key : str or Path, optional
            Key used for encryption.
        use_dask : bool
            Whether to use Dask for distributed computation.
        """

        # Assign parameters to instance variables
        self.name = name
        self.description = description
        self.group_by_fields = group_by_fields
        self.aggregations = aggregations
        self.custom_aggregations = custom_aggregations
        self.output_format = output_format
        self.use_cache = use_cache
        self.use_encryption = use_encryption
        self.output_format = output_format
        self.use_cache = use_cache
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.use_dask = use_dask

        # Call base class constructor
        super().__init__(
            name=self.name,
            use_cache=self.use_cache,
            use_dask=self.use_dask,
            use_encryption=self.use_encryption,
            encryption_key=self.encryption_key,
            description=self.description,
            output_format=self.output_format,
        )

        # Build and validate configuration
        config = AggregateRecordsOperationConfig(
            group_by_fields=self.group_by_fields,
            aggregations=self.aggregations or {},
            custom_aggregations=self.custom_aggregations or {},
            output_format=self.output_format,
            use_cache=self.use_cache,
            use_encryption=self.use_encryption,
            encryption_key=self.encryption_key,
            use_dask=self.use_dask,
        )
        self.config = config  # Optionally store the config

        # Use default description if not provided
        if not self.description:
            self.description = f"Aggregating records by {self.group_by_fields}."

        # Temporary storage for cleanup
        self._temp_data = None
        self.operation_cache = None

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
            - encrypt_output: bool - Override encryption setting

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Initialize timing and result
            self.start_time = time.time()
            self.process_count = 0
            result = OperationResult(status=OperationStatus.PENDING)
            encryption_key = kwargs.get('encryption_key', None)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=logger, progress_tracker=progress_tracker
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            # Set up progress tracking
            # Preparation, Validation, Data Loading, Processing, Metrics, Finalization
            force_recalculation = kwargs.get("force_recalculation", False)
            total_steps = 6 + (1 if self.use_cache and not force_recalculation else 0)
            current_steps = 0
            # Step 1: Preparation progress tracker
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Preparation", progress_tracker
                )

            # Decompose kwargs and introduce variables for clarity
            is_encryption_required = (
                kwargs.get("encrypt_output", False) or self.use_encryption
            )
            use_dask = kwargs.get("use_dask", self.use_dask)
            generate_visualization = kwargs.get("generate_visualization", True)
            include_timestamp_in_filenames = kwargs.get("include_timestamp", True)
            save_output = kwargs.get("save_output", True)
            dataset_name = kwargs.get("dataset_name", "main")

            # Step 2: Validation progress tracker
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Validation", progress_tracker
                )

            # Validate input parameters
            self._validate_input_params(
                self.group_by_fields, self.aggregations, self.custom_aggregations
            )

            # Step 3: Loading progress tracker
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Data Loading", progress_tracker
                )

            # Loading datasets
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source",
                )

            # Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not force_recalculation:
                if progress_tracker:
                    current_steps += 1
                    self._update_progress_tracker(
                        total_steps, current_steps, "Checking cache", progress_tracker
                    )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df)
                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        self._update_progress_tracker(
                            total_steps,
                            current_steps,
                            "Complete (cached)",
                            progress_tracker,
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Transformation of {self.name} (from cache)",
                            details={
                                "cached": True,
                                "group_by": self.group_by_fields,
                                "aggregations": self.aggregations,
                                "custom_aggregations": self.custom_aggregations,
                            },
                        )

                    return cache_result

            # Step 4: Processing progress tracker
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Processing", progress_tracker
                )

            try:
                # Process the data with the selected strategy
                self.process_count = len(df)
                processed_df = aggregate_dataframe(
                    df,
                    group_by_fields=self.group_by_fields,
                    aggregations=self.aggregations,
                    custom_aggregations=self.custom_aggregations,
                    use_dask=use_dask,
                )
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 5: Metrics Calculation
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Metrics Calculation", progress_tracker
                )

            # Collect final metrics before using them
            self.end_time = time.time()

            # Initialize metrics in scope
            metrics = {}

            try:
                metrics = self._collect_metrics(df=df, processed_df=processed_df)

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=f"{self.name}_metrics",
                    timestamp_in_name=True,
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
                    description=f"{self.name} metrics",
                    category=Constants.Artifact_Category_Metrics
                )

                # Report artifact
                if reporter:
                    reporter.add_artifact(
                        "json",
                        str(metrics_result.path),
                        f"{self.name} metrics",
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Finalization (Visualizations and Output Data)
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Finalization", progress_tracker
                )

            # Generate visualizations if required
            if generate_visualization:
                try:
                    kwargs_encryption = {
                        "use_encryption": kwargs.get('use_encryption', False),
                        "encryption_key": encryption_key
                    }
                    visualization_paths = self._generate_visualizations(
                        df=df,
                        processed_df=processed_df,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        **kwargs_encryption
                    )

                    # Register visualization artifacts and report
                    for viz_type, path in visualization_paths.items():
                        result.add_artifact(
                            artifact_type="png",
                            path=path,
                            description=f"{self.name} {viz_type} visualization",
                            category=Constants.Artifact_Category_Visualization
                        )

                        if reporter:
                            reporter.add_operation(
                                f"{self.name} {viz_type} visualization",
                                details={"type": "png", "path": str(path)},
                            )

                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical

            # Save output data if required
            if save_output:
                try:
                    self._save_output_data(
                        result_df=processed_df,
                        task_dir=task_dir,
                        include_timestamp_in_filenames=include_timestamp_in_filenames,
                        is_encryption_required=is_encryption_required,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=progress_tracker,
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
                        original_data=df,
                        transformed_data=processed_df,
                        task_dir=task_dir,
                        metrics=metrics,
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

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Transformation of {self.name} completed",
                    details=details,
                )

            # Cleanup memory
            self._cleanup_memory(processed_df, df)

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

    def process_batch(self, batch_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Deprecated: No longer used for merging. All merging is now performed in a single step.
        """
        logger.warning(
            "process_batch is deprecated and not used for merging. All merging is performed in a single step."
        )
        return batch_df

    def _process_value(self, value: Any, **params) -> Any:
        """
        Deprecated in batch mode. Not used in ML imputation anymore.
        """
        logger.warning(
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
        transformation_type = self.__class__.__name__

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
            "version": self.version,  # Include version for cache invalidation
        }

        return params

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        processed_df: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        reporter: Any = None,
        **kwargs
    ) -> dict:
        """
        Generate required visualizations for the merge operation using visualization utilities.

        Parameters
        ----------
        df : pd.DataFrame
            The left input DataFrame used in the merge operation.
        processed_df : pd.DataFrame
            The right input DataFrame used in the merge operation.
        task_dir : Path
            The base directory where all task-related outputs (including visualizations) will be saved.
        result : OperationResult
            An object representing the result of the merge operation, potentially containing metadata and statistics.
        reporter : Any, optional
            Optional reporter instance used for logging, tracking, or debugging the visualization process.

        Returns
        -------
        """
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_paths = {}

        try:
            # Sample large datasets for visualization
            if len(df) > 10000:
                df_viz = sample_large_dataset(df, max_samples=10000)
                aggrega_viz = sample_large_dataset(processed_df, max_samples=10000)
            else:
                df_viz = df
                aggrega_viz = processed_df

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
                    **kwargs
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
                    **kwargs
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
                    **kwargs
                )
            )

        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")

        return visualization_paths

    def _save_output_data(
        self,
        result_df: pd.DataFrame,
        task_dir: Path,
        include_timestamp_in_filenames: bool,
        is_encryption_required: bool,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        **kwargs,
    ) -> None:
        """
        Save the processed output data.

        Parameters:
        -----------
        result_df : pd.DataFrame
            The processed dataframe to save
        task_dir : Path
            Task directory for saving output
        include_timestamp_in_filenames : bool
            Whether to include a timestamp in the filename
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
        **kwargs : dict
            Additional parameters for the operation
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        custom_kwargs = {k: v for k, v in kwargs.items() if k != 'encryption_key'}
        output_result = writer.write_dataframe(
            df=result_df,
            name=f"{self.name}_transformed",
            format=self.output_format,
            subdir="output",
            timestamp_in_name=include_timestamp_in_filenames,
            encryption_key=self.encryption_key if is_encryption_required else None,
            **custom_kwargs,
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type=self.output_format,
            path=output_result.path,
            description=f"{self.name} transformed data",
            category=Constants.Artifact_Category_Output
        )

        # Report to reporter
        if reporter:
            reporter.add_artifact(
                self.output_format,
                str(output_result.path),
                f"{self.name} transformed data",
            )

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
        import gc

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

        # Clear instance attribute references
        if hasattr(self, "_temp_data"):
            self._temp_data = None

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Remove any temporary attributes starting with _temp_
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                setattr(self, attr_name, None)

        # Force garbage collection
        gc.collect()

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

    def _check_cache(self, df: pd.DataFrame) -> Optional[OperationResult]:
        """
        Check if a cached result exists for this operation.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to check in the cache

        Returns:
        --------
        Optional[OperationResult]
                Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            if not self.group_by_fields:
                # Use entire DataFrame or selected columns for cache key
                cache_key = self._generate_cache_key(df)
            else:
                for field in self.group_by_fields:
                    if field not in df.columns:
                        logger.warning(
                            f"Field '{field}' not found in DataFrame columns."
                        )
                        return None
                cache_key = self._generate_cache_key(df[self.group_by_fields])

            logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if cached_data:
                logger.info(
                    f"Cache hit for {self.group_by_fields} transformation operation"
                )

                result = OperationResult(status=OperationStatus.SUCCESS)

                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                result.add_metric("cached", True)
                result.add_metric("cache_key", cache_key)
                result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )

                return result

            logger.debug(f"No cache found for key: {cache_key}")
            return None

        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
        self,
        original_data: Union[pd.Series, pd.DataFrame],
        transformed_data: Union[pd.Series, pd.DataFrame],
        task_dir: Path,
        metrics: Dict[str, Any] = None,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_data : pd.Series or pd.DataFrame
            Original input data
        transformed_data : pd.Series or pd.DataFrame
            Transformed output data
        task_dir : Path
            Task directory
        metrics : Dict[str, Any], optional
            Metrics dictionary (default is None)

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
            operation_params = self._get_cache_parameters()
            operation_params.update(
                {
                    "operation_class": self.__class__.__name__,
                    "version": self.version,
                    "group_by_fields": self.group_by_fields,
                    "aggregations": self.aggregations,
                    "custom_aggregations": self.custom_aggregations,
                }
            )

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
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
            }

            # Save to cache
            logger.debug(f"Saving to cache with key: {cache_key}")
            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                logger.info(
                    f"Successfully saved {self.name} operation results to cache"
                )
            else:
                logger.warning(f"Failed to save {self.name} operation results to cache")

            return success

        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            return False

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
