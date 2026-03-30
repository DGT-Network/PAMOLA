"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module: Split By ID Values Operation
Description: Operation for splitting datasets by ID values or partitioning strategies
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides an operation for splitting datasets into multiple subsets
based on ID values or automatic partitioning strategies, while maintaining data utility.
It implements various strategies:

1. Value Groups: Explicitly split by user-defined groups of ID values
2. Equal Size: Partition data into equal-sized subsets
3. Random: Randomly assign records to partitions
4. Modulo: Partition based on hash(id) % number_of_partitions

Key features:
- Direct in-place DataFrame splitting with flexible strategies
- Robust null and invalid value handling
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for subset distributions
- Chunked and parallel processing support for large datasets (Dask, Joblib)
- Graceful handling of unmatched or "other" ID values
- Memory-efficient operation with explicit cleanup and caching

Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.
"""

from datetime import datetime
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import numpy as np
import pandas as pd

from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.exceptions import (
    FieldNotFoundError,
    InvalidParameterError,
    InvalidStrategyError,
)

from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.transformations.commons.enum import PartitionMethod
from pamola_core.utils.io import (
    ensure_directory,
    load_settings_operation,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_registry import register
from pamola_core.common.constants import Constants
from pamola_core.utils.visualization import (
    create_bar_plot,
    create_pie_chart,
    create_heatmap,
)
import dask.dataframe as dd
from joblib import Parallel, delayed
from pamola_core.transformations.schemas.split_by_id_values_op_core_schema import (
    SplitByIDValuesOperationConfig,
)


@register(version="1.0.0")
class SplitByIDValuesOperation(TransformationOperation):
    """Operation for splitting a dataset by ID values or automatic partitioning."""

    def __init__(
        self,
        name: str = "split_by_id_values_operation",
        id_field: Optional[str] = None,
        value_groups: Optional[Dict[str, List[Any]]] = None,
        number_of_partitions: int = 1,
        partition_method: str = PartitionMethod.EQUAL_SIZE.value,
        **kwargs,
    ):
        """
        Initialize the SplitByIDValuesOperation.

        Parameters
        ----------
        name : str, optional
            Name of the operation (default: "split_by_id_values_operation")
        id_field : str, optional
            Field used for identifying records uniquely.
        value_groups : dict[str, list], optional
            Explicit groups of ID values to split data.
        number_of_partitions : int, optional
            Number of partitions for automatic splitting if no value_groups provided.
        partition_method : str, optional
            Method for splitting when using automatic partitioning.
        **kwargs : dict
            Additional keyword arguments for TransformationOperation.
        """
        # --- Ensure metadata defaults ---
        kwargs.setdefault("name", name)
        kwargs.setdefault("description", "Split dataset by ID values or partitioning")

        # --- Build config object ---
        config = SplitByIDValuesOperationConfig(
            id_field=id_field,
            value_groups=value_groups,
            number_of_partitions=number_of_partitions,
            partition_method=partition_method,
            **kwargs,
        )

        # --- Inject config into kwargs for the base class ---
        kwargs["config"] = config

        # --- Initialize TransformationOperation ---
        super().__init__(
            **kwargs,
        )

        # --- Apply config attributes to instance ---
        for key, value in config.to_dict().items():
            setattr(self, key, value)

        # --- Operation metadata ---
        self.operation_name = self.__class__.__name__
        self._original_df = None

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ):
        """
        Execute the operation with timing and error handling.

        Parameters
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

        Returns
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Initialize timing and result
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(
                f"Starting: {self.operation_name} operation at {self.start_time}"
            )

            result = OperationResult(status=OperationStatus.PENDING)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare directories for artifacts
            dirs = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=dirs["cache"],
            )

            # Initialize error handler
            self.error_handler = ErrorHandler(
                logger=self.logger,
                operation_name=self.operation_name,
            )

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Save configuration
            self.save_config(task_dir)

            # Start operation
            self.logger.info(f"Operation: {self.operation_name}, Start operation")
            if progress_tracker:
                progress_tracker.total = self._compute_total_steps()
                progress_tracker.update(
                    1,
                    {
                        "step": "Start operation - Preparation",
                        "operation": self.operation_name,
                    },
                )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": "Preparation",
                        "message": "Preparation successfully",
                        "directories": {k: str(v) for k, v in dirs.items()},
                    },
                )

            # Load data and validate input parameters
            try:
                self.logger.info(
                    f"Operation: {self.operation_name}, Load data and validate input parameters"
                )
                step = "Load data and validate input parameters"
                if progress_tracker:
                    progress_tracker.update(
                        1, {"step": step, "operation": self.operation_name}
                    )

                # Load settings operation
                settings_operation = load_settings_operation(
                    data_source, dataset_name, **kwargs
                )
                self.logger.info(
                    f"Operation: {self.operation_name}, Load data and validate input parameters"
                )
                df = self._validate_and_get_dataframe(
                    data_source, dataset_name, **settings_operation
                )
                self._validate_input_parameters(df)
            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.DATA_LOAD_FAILED,
                    context={"dataset": dataset_name, "operation": self.operation_name},
                    message_kwargs={"source": dataset_name, "reason": str(e)},
                )

            # Handle cache if required
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Checking Cache"})

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df, reporter)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(1, {"step": "Complete (cached)"})

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            "Split by values (from cache)",
                            details={"cached": True},
                        )
                    return cache_result

            # Process data
            self.logger.info(f"Operation: {self.operation_name}, Process data")
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Process data", "operation": self.operation_name}
                )

            try:
                processed_df = self._process_data(df, **kwargs)
                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Process data",
                            "message": "Process data successfully",
                            "num_subsets": len(processed_df),
                        },
                    )
            except Exception as e:
                return self.error_handler.handle_error(
                    error=e,
                    error_code=ErrorCode.PROCESSING_FAILED,
                    context={"step": "processing", "operation": self.operation_name},
                    message_kwargs={
                        "field_name": self.field_label,
                        "operation": self.operation_name,
                        "reason": str(e),
                    },
                )

            self.end_time = time.time()
            if self.start_time and self.end_time:
                self.execution_time = self.end_time - self.start_time

            # Collect metric
            self.logger.info(f"Operation: {self.operation_name}, Collect metric")
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Collect metric", "operation": self.operation_name}
                )

            try:
                metrics = self._collect_metrics(df, processed_df)

                self._save_metrics(
                    metrics=metrics,
                    writer=writer,
                    result=result,
                    reporter=reporter,
                    progress_tracker=progress_tracker,
                    operation_timestamp=operation_timestamp,
                )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.error(error_message)
                # Continue execution - metrics failure is not critical

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": "Collect metric",
                        "message": "Collect metric successfully",
                        "summary": {
                            "input_dataset": metrics.get("input_dataset"),
                            "input_records": metrics.get("total_input_records"),
                            "input_fields": metrics.get("total_input_fields"),
                            "output_records": metrics.get("total_output_records"),
                            "output_fields": metrics.get("total_output_fields"),
                            "id_field": metrics.get("id_field"),
                            "splits": metrics.get("number_of_splits"),
                            "execution_time_seconds": metrics.get(
                                "execution_time_seconds"
                            ),
                        },
                    },
                )

            # Save output if required
            if self.save_output:
                self.logger.info(f"Operation: {self.operation_name}, Save output")
                if progress_tracker:
                    progress_tracker.update(
                        1, {"step": "Save output", "operation": self.operation_name}
                    )

                try:
                    self._save_multiple_output_data(
                        result_subsets=processed_df,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=progress_tracker,
                        timestamp=operation_timestamp,
                        **kwargs,
                    )
                except Exception as e:
                    return self.error_handler.handle_error(
                        error=e,
                        error_code=ErrorCode.ARTIFACT_WRITE_FAILED,
                        context={"step": "save_output", "field": self.field_label},
                        message_kwargs={
                            "path": str(task_dir / "output"),
                            "reason": str(e),
                        },
                    )

                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Save output",
                            "message": "Save output successfully",
                            "files_saved": len(result.artifacts),
                        },
                    )

            # Generate visualizations if required
            if self.generate_visualization:
                self.logger.info(
                    f"Operation: {self.operation_name}, Generate visualizations"
                )
                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Generate visualizations",
                            "operation": self.operation_name,
                        },
                    )

                try:
                    self._handle_visualizations(
                        input_data=df,
                        output_data=processed_df,
                        task_dir=task_dir,
                        result=result,
                        operation_timestamp=operation_timestamp,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.error(error_message)
                    # Continue execution - visualization failure is not critical

                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Generate visualizations",
                            "message": "Generate visualizations successfully",
                            "num_images": len(
                                [
                                    a
                                    for a in result.artifacts
                                    if a.artifact_type == "png"
                                ]
                            ),
                        },
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=self._original_df,
                        transformed_data=processed_df,
                        result=result,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                result_df=processed_df,
                original_data=df,
                transformed_data=None,
            )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": "Return result",
                        "message": "Operation completed successfully",
                    },
                )

            # Set success status
            result.status = OperationStatus.SUCCESS
            result.execution_time = self.execution_time
            self.logger.info(
                f"Processing completed {self.operation_name} operation in {self.execution_time:.2f} seconds"
            )
            return result

        except Exception as e:
            self.logger.exception(f"Error in {self.operation_name}: {str(e)}")
            return self.error_handler.handle_error(
                error=e,
                error_code=ErrorCode.PROCESSING_FAILED,
                context={"operation": self.operation_name, "field": self.field_label},
                message_kwargs={
                    "field_name": self.field_label,
                    "operation": self.operation_name,
                    "reason": str(e),
                },
            )

    def _process_data(
        self, df: pd.DataFrame, **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Main entry point to split a DataFrame into subsets based on ID values or partition strategy.

        The method automatically chooses an execution strategy depending on:
        - use_dask: for distributed processing with Dask.
        - use_vectorization: for parallel batch processing with Joblib.
        - Fallback to standard Pandas processing otherwise.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be split.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping partition names to DataFrame subsets.
        """

        # Dask should only be used with partition methods like MODULO or RANDOM.
        # Other methods (e.g., EQUAL_SIZE or value_groups) are better handled with Pandas or Joblib
        # Dask is not efficient for index-based slicing or group-based filtering
        if (
            self.use_dask
            and self.npartitions > 1
            and self.partition_method
            in {PartitionMethod.MODULO.value, PartitionMethod.RANDOM.value}
        ):
            return self._process_with_dask(df)

        # Joblib should only be used when value_groups are defined
        # It is not suitable for partition methods like MODULO, RANDOM, or EQUAL_SIZE
        # Since those can be handled more efficiently with Pandas or Dask
        elif (
            self.use_vectorization and self.parallel_processes > 0 and self.value_groups
        ):
            return self._process_with_joblib(df)

        else:
            return self._process_with_pandas(df)

    def _process_with_dask(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the DataFrame into partitions using Dask for scalable processing.

        Supported partition methods:
        - MODULO: partition based on hash(id) % number_of_partitions
        - RANDOM: random distribution into partitions

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to be partitioned.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary where keys are partition names (e.g., "partition_0") and values are DataFrame subsets.
        """
        subsets = {}
        ddf = dd.from_pandas(df, npartitions=self.npartitions)

        if self.partition_method == PartitionMethod.MODULO.value:

            def apply_modulo(part):
                part["_partition"] = part[self.id_field].apply(
                    lambda x: hash(x) % self.number_of_partitions
                )
                return part

            ddf = ddf.map_partitions(apply_modulo)

        elif self.partition_method == PartitionMethod.RANDOM.value:

            def apply_random_partition(part):
                np.random.seed(42)
                part["_partition"] = np.random.choice(
                    self.number_of_partitions, size=len(part)
                )
                return part

            ddf = ddf.map_partitions(apply_random_partition)

        for i in range(self.number_of_partitions):
            part_df = ddf[ddf["_partition"] == i].drop(columns=["_partition"]).compute()
            subsets[f"partition_{i}"] = part_df

        return subsets

    def _process_with_joblib(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the DataFrame based on explicit value_groups using Joblib for parallel filtering.

        Each group is defined by a list of values from the id_field.
        Any unmatched rows will be grouped under "others".

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to be split.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping group names to DataFrame subsets.
        """

        def filter_group(group_name, values):
            mask = df[self.id_field].isin(values)
            return group_name, df[mask].copy(deep=True)

        results = Parallel(n_jobs=self.parallel_processes)(
            delayed(filter_group)(group_name, values)
            for group_name, values in self.value_groups.items()
        )
        subsets = dict(results)

        # Handle "others"
        all_values = [v for vals in self.value_groups.values() for v in vals]
        others_mask = ~df[self.id_field].isin(all_values)
        if others_mask.any():
            subsets["others"] = df[others_mask].copy(deep=True)

        return subsets

    def _process_with_pandas(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the DataFrame using pure Pandas, based on value_groups or partition method.

        - If value_groups is provided: splits by defined groups and an optional "others" group.
        - If number_of_partitions is set: supports EQUAL_SIZE, RANDOM, or MODULO strategies.
        - If none of the above: returns full DataFrame under "all_data".

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to be split.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of split DataFrames by group or partition.
        """
        subsets = {}

        if self.value_groups:
            for group_name, values in self.value_groups.items():
                mask = df[self.id_field].isin(values)
                subsets[group_name] = df[mask].copy(deep=True)

            all_values = [v for vals in self.value_groups.values() for v in vals]
            others_mask = ~df[self.id_field].isin(all_values)
            if others_mask.any():
                subsets["others"] = df[others_mask].copy(deep=True)

        elif self.number_of_partitions > 0:
            if self.partition_method == PartitionMethod.EQUAL_SIZE.value:
                sorted_df = df.sort_values(by=self.id_field)
                partition_sizes = np.full(
                    self.number_of_partitions,
                    len(sorted_df) // self.number_of_partitions,
                )
                partition_sizes[: len(sorted_df) % self.number_of_partitions] += 1

                start_idx = 0
                for i, size in enumerate(partition_sizes):
                    end_idx = start_idx + size
                    subsets[f"partition_{i}"] = sorted_df.iloc[start_idx:end_idx].copy(
                        deep=True
                    )
                    start_idx = end_idx

            elif self.partition_method == PartitionMethod.RANDOM.value:
                np.random.seed(42)
                partitions = np.random.choice(self.number_of_partitions, size=len(df))
                for i in range(self.number_of_partitions):
                    subsets[f"partition_{i}"] = df[partitions == i].copy(deep=True)

            elif self.partition_method == PartitionMethod.MODULO.value:
                partitions = df[self.id_field].apply(
                    lambda x: hash(x) % self.number_of_partitions
                )
                for i in range(self.number_of_partitions):
                    subsets[f"partition_{i}"] = df[partitions == i].copy(deep=True)

        else:
            subsets["all_data"] = df.copy(deep=True)

        return subsets

    def _collect_metrics(
        self,
        input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Dict[str, Any]:
        """
        Collect operation-specific metrics for SplitByIDValuesOperation and return in structured format.

        Metrics include input/output size, number of splits, and summary info
        about each split (record count and included ID values).
        """
        if not isinstance(output_data, dict):
            self.logger.warning(
                "Transformed data is not in expected dictionary format."
            )
            return {}

        input_rows = input_data.shape[0] if isinstance(input_data, pd.DataFrame) else 0
        input_fields = (
            input_data.shape[1] if isinstance(input_data, pd.DataFrame) else 0
        )

        total_output_records = sum(len(df) for df in output_data.values())
        total_output_fields = sum(len(df.columns) for df in output_data.values())

        split_info = {
            name: {
                "record_count": len(df),
                "included_records": df[self.id_field].dropna().unique().tolist(),
            }
            for name, df in output_data.items()
        }

        return {
            "operation_type": self.operation_name,
            "total_input_records": input_rows,
            "total_input_fields": input_fields,
            "total_output_records": total_output_records,
            "total_output_fields": total_output_fields,
            "id_field": self.id_field,
            "number_of_splits": len(output_data),
            "split_info": split_info,
            "execution_time_seconds": round(self.execution_time, 2),
            "processing_date": datetime.now().isoformat(),
        }

    def _save_multiple_output_data(
        self,
        result_subsets: dict[str, pd.DataFrame],
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        timestamp: Optional[str] = None,
        **kwargs,
    ):
        """
        Save the processed subsets to output files and record artifact paths.

        Parameters
        ----------
        result_subsets : dict[str, pd.DataFrame]
            Dictionary mapping dataset names to DataFrames.
        task_dir : Path
            Root task directory for saving outputs.
        result : OperationResult
            Result object to append artifact file paths.
        operation_timestamp : str
            Timestamp string for naming output files.
        """

        for dataset_name, df in result_subsets.items():
            file_name = f"{self.operation_name}_{dataset_name}_output_{timestamp}"

            try:
                output_path = self._save_output_data(
                    result_df=df,
                    writer=writer,
                    result=result,
                    reporter=reporter,
                    progress_tracker=progress_tracker,
                    timestamp=timestamp,
                    file_name=file_name,
                    **kwargs,
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to save {dataset_name} to {output_path}: {e}"
                )

    def _generate_visualizations(
        self,
        input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        task_dir: Path,
        result: OperationResult,
        operation_timestamp: str,
    ) -> None:

        if not isinstance(output_data, dict) or not output_data:
            self.logger.warning(
                "Skipping visualization: output_data is not a non-empty dictionary of DataFrames."
            )
            return

        vis_dir = task_dir / "visualizations"
        ensure_directory(vis_dir)

        operation_name = self.operation_name.lower()

        kwargs_visualization = {
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
            "backend": self.visualization_backend,
            "theme": self.visualization_theme,
            "strict": self.visualization_strict,
        }

        # Prepare bar data for both bar and pie charts
        bar_data = {k: len(df) for k, df in output_data.items()}

        # 1. Bar chart: Record count per subset
        try:
            bar_path = (
                vis_dir
                / f"{operation_name}_record_count_bar_chart_{operation_timestamp}.png"
            )
            bar_result = create_bar_plot(
                data=bar_data,
                output_path=bar_path,
                title="Record Count per Subset",
                orientation="v",
                x_label="Subset",
                y_label="Number of Records",
                **kwargs_visualization,
            )
            if not bar_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=bar_result,
                    description="Bar chart showing record count per subset",
                    category=Constants.Artifact_Category_Visualization,
                )
        except Exception as e:
            self.logger.error(f"Failed to create record count bar chart: {e}")

        # 2. Pie chart: Distribution across subsets
        try:
            pie_path = (
                vis_dir
                / f"{operation_name}_record_distribution_pie_chart_{operation_timestamp}.png"
            )
            pie_result = create_pie_chart(
                data=bar_data,
                output_path=pie_path,
                title="Record Distribution Across Subsets",
                show_percentages=True,
                show_values=True,
                **kwargs_visualization,
            )
            if not pie_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=pie_result,
                    description="Pie chart showing record distribution across subsets",
                    category=Constants.Artifact_Category_Visualization,
                )
        except Exception as e:
            self.logger.error(f"Failed to create record distribution pie chart: {e}")

        # 3. Heatmap: ID value distribution across subsets
        try:
            id_values_per_subset = {
                subset_name: df[self.id_field].value_counts()
                for subset_name, df in output_data.items()
            }
            combined_df = pd.DataFrame(id_values_per_subset).fillna(0).sort_index()

            id_dist_path = (
                vis_dir
                / f"{operation_name}_id_value_distribution_{operation_timestamp}.png"
            )
            heatmap_result = create_heatmap(
                data=combined_df.T,
                output_path=id_dist_path,
                title="Distribution of ID Values Across Subsets",
                x_label="ID Value",
                y_label="Subset",
                annotate=True,
                annotation_format=".0f",
                **kwargs_visualization,
            )
            if not heatmap_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=heatmap_result,
                    description="Distribution of ID values across subsets (heatmap)",
                    category=Constants.Artifact_Category_Visualization,
                )
        except Exception as e:
            self.logger.error(f"Failed to create ID value distribution heatmap: {e}")

    def _handle_visualizations(
        self,
        input_data: pd.DataFrame,
        output_data: Dict[str, pd.DataFrame],
        task_dir: Path,
        result: OperationResult,
        operation_timestamp: str,
    ) -> None:

        import threading
        import contextvars

        self.logger.info(
            "[VIZ] Preparing to generate visualizations in a separate thread"
        )

        viz_error = None

        def run_visualizations():
            nonlocal viz_error
            try:
                self._generate_visualizations(
                    input_data=input_data,
                    output_data=output_data,
                    task_dir=task_dir,
                    result=result,
                    operation_timestamp=operation_timestamp,
                )
            except Exception as e:
                viz_error = e
                self.logger.error(
                    f"[VIZ] Visualization error: {type(e).__name__}: {e}", exc_info=True
                )

        try:
            ctx = contextvars.copy_context()
            thread = threading.Thread(
                target=ctx.run,
                args=(run_visualizations,),
                name=f"VizThread-{self.name}",
                daemon=True,
            )

            thread.start()
            thread.join(timeout=self.visualization_timeout)

            if thread.is_alive():
                self.logger.warning(
                    f"[VIZ] Visualization thread timed out after {self.visualization_timeout}s"
                )
            elif viz_error:
                self.logger.warning(f"[VIZ] Visualization thread failed: {viz_error}")
            else:
                self.logger.info("[VIZ] Visualization thread completed successfully")

        except Exception as e:
            self.logger.error(
                f"[VIZ] Error setting up visualization thread: {e}", exc_info=True
            )

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for SplitByIDValuesOperation using external kwargs.

        Parameters
        ----------
        **kwargs : dict
            Dictionary of parameters that may override instance attributes.

        Returns
        -------
        Dict[str, Any]
            Dictionary of relevant parameters to identify the operation configuration.
        """
        return {
            "id_field": self.id_field,
            "value_groups": self.value_groups,
            "number_of_partitions": self.number_of_partitions,
            "partition_method": self.partition_method,
        }

    def _validate_input_parameters(self, df: pd.DataFrame) -> bool:
        all_columns = set(df.columns)

        # Check if the ID field exists
        if not self.id_field:
            raise InvalidParameterError(
                param_name="id_field",
                param_value=self.id_field,
                reason="ID field must be specified",
            )

        if self.id_field not in all_columns:
            raise FieldNotFoundError(
                field_name=self.id_field,
                available_fields=list(all_columns),
            )

        # Validate value_groups or number_of_partitions
        if self.value_groups:
            id_values_in_df = set(df[self.id_field].unique())
            for group_name, values in self.value_groups.items():
                missing = [v for v in values if v not in id_values_in_df]
                if missing:
                    self.logger.warning(
                        f"Group '{group_name}' contains ID values not present in the DataFrame: {missing}"
                    )
        else:
            if self.number_of_partitions <= 0:
                raise InvalidParameterError(
                    param_name="number_of_partitions",
                    param_value=self.number_of_partitions,
                    reason=(
                        "Either 'value_groups' must be specified or "
                        "'number_of_partitions' must be a positive integer."
                    ),
                )

            if self.partition_method not in PartitionMethod._value2member_map_:
                raise InvalidStrategyError(
                    strategy=self.partition_method,
                    valid_strategies=[m.value for m in PartitionMethod],
                    operation_type="partition_method",
                )

        return True

    def _compute_total_steps(self) -> int:
        steps = 0

        steps += 1  # Step 1: Preparation
        steps += 1  # Step 2: Load data and validate input

        if self.use_cache and not self.force_recalculation:
            steps += 1  # Step 3: Try to load from cache

        steps += 1  # Step 4: Process data
        steps += 1  # Step 5: Collect metrics

        if self.save_output:
            steps += 1  # Step 6: Save output

        if self.generate_visualization:
            steps += 1  # Step 7: Generate visualizations

        if self.use_cache:
            steps += 1  # Step 8: Save cache

        return steps
