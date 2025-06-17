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

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
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


class MergeDatasetsOperationConfig(OperationConfig):
    """Configuration for MergeDatasetsOperation."""

    schema = {
        "type": "object",
        "properties": {
            "left_dataset_name": {"type": "string"},
            "right_dataset_name": {"type": ["string", "null"]},
            "right_dataset_path": {"type": ["string", "null"]},
            "left_key": {"type": "string"},
            "right_key": {"type": ["string", "null"]},
            "join_type": {
                "type": "string",
                "str": ["inner", "left", "right", "outer"],
            },
            "suffixes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
            "relationship_type": {
                "type": "string",
                "str": ["auto", "one-to-one", "one-to-many"],
            },
            "output_format": {"type": "string", "str": ["csv", "json", "parquet"]},
            "use_cache": {"type": "boolean"},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]},
            "use_dask": {"type": "boolean"},
        },
        "required": [
            "left_dataset_name",
            "left_key",
            "right_dataset_name",
            "right_key",
            "join_type",
            "relationship_type",
        ],
    }


@register(version="1.0.0")
class MergeDatasetsOperation(TransformationOperation):
    """
    Operation to merge two datasets based on key fields.

    This operation supports various join types and allows configuration of
    suffixes, batch size, output format, caching, and encryption.
    """

    def __init__(
        self,
        name: str = "merge_datasets_operation",
        description: str = "Merge datasets by key field",
        left_dataset_name: str = "main",
        right_dataset_name: str = None,
        right_dataset_path: Optional[Path] = None,
        left_key: str = None,
        right_key: Optional[str] = None,
        join_type: str = "left",
        relationship_type: str = "auto",
        suffixes: Tuple[str, str] = ("_x", "_y"),
        output_format: str = "csv",
        use_cache: bool = True,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        use_dask: bool = False,
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
        relationship_type : str, optional
            Type of relationship between datasets ('auto', 'one-to-one', 'one-to-many').
            When 'auto', the function will detect the relationship type based on key uniqueness.
        """

        # Assign parameters to instance variables
        self.name = name
        self.description = description
        self.left_dataset_name = left_dataset_name
        self.right_dataset_name = right_dataset_name
        self.right_dataset_path = right_dataset_path
        self.left_key = left_key
        self.right_key = right_key or left_key  # Default fallback
        self.join_type = join_type
        self.suffixes = suffixes
        self.output_format = output_format
        self.use_cache = use_cache
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.use_dask = use_dask
        self.relationship_type = relationship_type

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
        config = MergeDatasetsOperationConfig(
            left_dataset_name=self.left_dataset_name,
            right_dataset_name=self.right_dataset_name,
            right_dataset_path=(
                str(self.right_dataset_path) if self.right_dataset_path else None
            ),
            left_key=self.left_key,
            right_key=self.right_key,
            join_type=self.join_type,
            suffixes=list(self.suffixes) if self.suffixes else ["_x", "_y"],
            relationship_type=self.relationship_type,
            output_format=self.output_format,
            use_cache=self.use_cache,
            use_encryption=self.use_encryption,
            encryption_key=self.encryption_key,
            use_dask=self.use_dask,
        )
        self.config = config  # Optionally store the config

        # Use default description if not provided
        if not self.description:
            self.description = f"Merging '{self.left_dataset_name}' with '{self.right_dataset_name}' using keys."

        # Temporary storage for cleanup
        self._temp_data = None
        self.right_df = None
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

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=logger, progress_tracker=progress_tracker
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            encryption_key = kwargs.get('encryption_key', None)
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

            # Step 2: Validation progress tracker
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Validation", progress_tracker
                )

            # Validate input parameters
            self._validate_input_params(
                relationship_type=self.relationship_type,
                left_key=self.left_key,
                left_dataset_name=self.left_dataset_name,
                right_dataset_name=self.right_dataset_name,
                right_dataset_path=self.right_dataset_path,
            )

            # Step 3: Loading progress tracker
            if progress_tracker:
                current_steps += 1
                self._update_progress_tracker(
                    total_steps, current_steps, "Data Loading", progress_tracker
                )

            # Loading datasets
            left_df = self._get_dataset(data_source, self.left_dataset_name, **kwargs)
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

            # Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not force_recalculation:
                if progress_tracker:
                    current_steps += 1
                    self._update_progress_tracker(
                        total_steps, current_steps, "Checking cache", progress_tracker
                    )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df=left_df)
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
                                "left_key": f"{self.left_key}",
                                "right_key": f"{self.right_key}",
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
                self.process_count = len(left_df)
                processed_df = merge_dataframes(
                    left_df=left_df,
                    right_df=self.right_df,
                    left_key=self.left_key,
                    right_key=self.right_key,
                    join_type=self.join_type,
                    suffixes=self.suffixes,
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
                metrics = self._collect_metrics(
                    left_df=left_df, right_df=self.right_df, processed_df=processed_df
                )

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
                    category=Constants.Artifact_Category_Metrics,
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
                        left_df,
                        self.right_df,
                        processed_df,
                        task_dir,
                        result,
                        reporter,
                        **kwargs_encryption
                    )

                    # Register visualization artifacts and report
                    for viz_type, path in visualization_paths.items():
                        result.add_artifact(
                            artifact_type="png",
                            path=path,
                            description=f"{self.name} {viz_type} visualization",
                            category=Constants.Artifact_Category_Visualization,
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
                        original_data=left_df,
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
            self._cleanup_memory(processed_df, left_df, self.right_df)

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
        elif left_key_is_unique and not right_key_is_unique:
            detected_relationship = RelationshipType.ONE_TO_MANY.value
        else:
            raise ValueError(
                "Only 'one-to-one' and 'one-to-many' relationships are supported. "
                "Detected unsupported relationship (many-to-one or many-to-many)."
            )

        logger.info(f"Auto-detected relationship type: {detected_relationship}")
        return detected_relationship

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
                logger.warning(
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
        logger.warning(
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
            "version": self.version,  # Include version for cache invalidation
        }

        return params

    def _generate_visualizations(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        reporter: Any = None,
        **kwargs
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
        result : OperationResult
            An object representing the result of the merge operation, potentially containing metadata and statistics.
        reporter : Any, optional
            Optional reporter instance used for logging, tracking, or debugging the visualization process.

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_paths = {}

        try:
            # Sample large datasets for visualization
            if len(left_df) > 10000:
                left_viz = sample_large_dataset(left_df, max_samples=10000)
                right_viz = sample_large_dataset(right_df, max_samples=10000)
                merged_viz = sample_large_dataset(merged_df, max_samples=10000)
            else:
                left_viz = left_df
                right_viz = right_df
                merged_viz = merged_df

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
                    **kwargs
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
                    **kwargs
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
                    **kwargs
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
            category=Constants.Artifact_Category_Output,
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
        directories["metrics"] = task_dir / "metrics"

        # Ensure all directories exist
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

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
        import gc

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

        # Clear instance attribute references
        if hasattr(self, "_temp_data"):
            self._temp_data = None

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

        # Force garbage collection
        gc.collect()

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
        
        settings_operation = load_settings_operation(source, dataset_name_or_path, **kwargs)
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
            if not self.left_key:
                # Use entire DataFrame or selected columns for cache key
                cache_key = self._generate_cache_key(df)
            else:
                if self.left_key not in df.columns:
                    logger.warning(
                        f"Field '{self.left_key}' not found in DataFrame columns."
                    )
                    return None
                cache_key = self._generate_cache_key(df[self.left_key])

            logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if cached_data:
                logger.info(f"Cache hit for {self.left_key} transformation operation")

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
            if isinstance(original_data, pd.DataFrame) and self.left_key:
                if self.left_key in original_data.columns:
                    cache_key = self._generate_cache_key(original_data[self.left_key])
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
                    "left_key": self.left_key,
                    "right_key": self.right_key,
                    "join_type": self.join_type,
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
        elif left_key_is_unique and not right_key_is_unique:
            detected_relationship = RelationshipType.ONE_TO_MANY.value
        else:
            raise ValueError(
                "Only 'one-to-one' and 'one-to-many' relationships are supported. "
                "Detected unsupported relationship (many-to-one or many-to-many)."
            )

        logger.info(f"Auto-detected relationship type: {detected_relationship}")
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
            logger.warning(
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
