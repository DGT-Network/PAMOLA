"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Split Fields Operation
Description: Operation for splitting datasets into multiple groups of fields
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides an operation for splitting a dataset into multiple subsets,
each containing a specific group of fields (columns), while maintaining data utility.
It implements various strategies:

1. Field Groups: Explicitly split by user-defined groups of fields
2. ID Field Inclusion: Optionally include an identifier field in each subset

Key features:
- Direct in-place DataFrame splitting by column groups
- Robust validation of field existence and group definitions
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for field distribution and schema comparison
- Memory-efficient operation with explicit cleanup and caching
- Graceful handling of missing or invalid field specifications

Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, progress tracking, and result reporting.
"""

import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
import pandas as pd
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.io import (
    load_data_operation,
    ensure_directory,
    load_settings_operation,
    write_json,
    write_dataframe_to_csv,
)
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
    OperationArtifact,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.common.constants import Constants
from pamola_core.utils.visualization import create_bar_plot, plot_field_subset_network
from pamola_core.utils.io_helpers import crypto_utils, directory_utils
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
import matplotlib

# Set the backend to 'Agg' to avoid GUI issues
matplotlib.use("Agg")
import hashlib
from pamola_core.utils.ops.op_registry import register


class OutputFormat(Enum):
    CSV = "csv"
    JSON = "json"


class SplitFieldsOperationConfig(OperationConfig):
    """Configuration for SplitFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "id_field": {"type": ["string", "null"]},
                    "field_groups": {
                        "type": ["object", "null"],
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "include_id_field": {"type": "boolean", "default": True},
                },
            },
        ],
    }


@register(version="1.0.0")
class SplitFieldsOperation(TransformationOperation):
    """Operation for splitting a dataset into multiple groups of fields."""

    def __init__(
        self,
        name: str = "split_fields_operation",
        id_field: Optional[str] = None,
        field_groups: Optional[Dict[str, List[str]]] = None,
        include_id_field: bool = True,
        **kwargs,
    ):
        """
        Initialize the SplitFieldsOperation.

        Parameters
        ----------
        name : str, optional
            The name of the operation (default is "split_fields_operation").
        id_field : str, optional
            The field used to uniquely identify records in the dataset.
        field_groups : dict[str, list[str]], optional
            A dictionary mapping group names to lists of field names to be grouped together.
        include_id_field : bool, optional
            Whether to include the `id_field` in each output group (default is True).
        **kwargs : dict
            Additional keyword arguments for TransformationOperation.
        """
        # --- Ensure metadata defaults ---
        kwargs.setdefault("name", name)
        kwargs.setdefault("description", "Split dataset by fields")

        # --- Build config object ---
        config = SplitFieldsOperationConfig(
            id_field=id_field,
            field_groups=field_groups,
            include_id_field=include_id_field,
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

        # Metadata
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

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

            dirs = self._prepare_directories(task_dir)

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
            self.logger.info(
                f"Operation: {self.operation_name}, Load data and validate input parameters"
            )
            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Load data and validate input parameters",
                        "operation": self.operation_name,
                    },
                )

            df, is_valid = self._load_data_and_validate_input_parameters(
                data_source, **kwargs
            )

            if is_valid:
                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Load data and validate input parameters",
                            "message": "Load data and validate input parameters successfully",
                            "shape": df.shape,
                        },
                    )
            else:
                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Load data and validate input parameters",
                            "message": "Load data and validate input parameters failed",
                        },
                    )
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message="Load data and validate input parameters failed",
                    )

            # Handle cache if required
            if self.use_cache and not self.force_recalculation:
                self.logger.info(
                    f"Operation: {self.operation_name}, Load result from cache"
                )
                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Load result from cache",
                            "operation": self.operation_name,
                        },
                    )

                try:
                    # _get_cache now returns OperationResult or None
                    cached_result = self._get_cache(df.copy(), **kwargs)
                except Exception as e:
                    error_message = f"Check cache error: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

                if cached_result is not None and isinstance(
                    cached_result, OperationResult
                ):
                    if reporter:
                        reporter.add_operation(
                            f"Operation {self.operation_name}",
                            status="info",
                            details={
                                "step": "Load result from cache",
                                "message": "Load result from cache successfully",
                            },
                        )
                    return cached_result
                else:
                    self.logger.info(
                        f"Operation: {self.operation_name}, Load result from cache failed — proceeding with execution."
                    )
                    if reporter:
                        reporter.add_operation(
                            f"Operation {self.operation_name}",
                            status="info",
                            details={
                                "step": "Load result from cache",
                                "message": "Load result from cache failed - proceeding with execution",
                            },
                        )

            # Process data
            self.logger.info(f"Operation: {self.operation_name}, Process data")
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Process data", "operation": self.operation_name}
                )

            try:
                transformed_df = self._process_data(df, **kwargs)
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": "Process data",
                        "message": "Process data successfully",
                        "num_subsets": len(transformed_df),
                    },
                )

            result = OperationResult(
                status=OperationStatus.SUCCESS,
                artifacts=[],
                metrics={},
                error_message=None,
                execution_time=0,
                error_trace=None,
            )

            self.end_time = time.time()
            result.execution_time = self.end_time - self.start_time

            # Collect metric
            self.logger.info(f"Operation: {self.operation_name}, Collect metric")
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Collect metric", "operation": self.operation_name}
                )

            try:
                metrics = self._collect_metrics(df, transformed_df)
                result.metrics = metrics
                self._save_metrics(
                    metrics, task_dir, result, operation_timestamp, **kwargs
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
                    self._save_output(
                        transformed_df, task_dir, result, operation_timestamp, **kwargs
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
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
                        output_data=transformed_df,
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

            # Save cache if required
            if self.use_cache:
                self.logger.info(f"Operation: {self.operation_name}, Save cache")
                if progress_tracker:
                    progress_tracker.update(
                        1, {"step": "Save cache", "operation": self.operation_name}
                    )

                try:
                    self._save_cache(task_dir, result, **kwargs)
                except Exception as e:
                    error_message = f"Failed to cache results: {str(e)}"
                    self.logger.error(error_message)
                    # Continue execution - cache failure is not critical

                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Save cache",
                            "message": "Save cache successfully",
                        },
                    )

            # Operation completed successfully
            self.logger.info(
                f"Operation: {self.operation_name}, Operation completed successfully."
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

            return result

        except Exception as e:
            self.logger.error(f"Operation: {self.operation_name}, error occurred: {e}")
            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="error",
                    details={
                        "step": "Exception",
                        "message": "Operation failed due to an exception",
                        "error": str(e),
                    },
                )

            return OperationResult(
                status=OperationStatus.ERROR, error_message=str(e), exception=e
            )

    def _process_data(
        self, df: pd.DataFrame, **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Process data according to operation-specific logic.
        Split the dataset into multiple subsets based on field_groups.
        Each subset contains its specified columns, and optionally the ID field.

        Dask and Joblib are not used here because we are only splitting the DataFrame by column groups,
        which is a lightweight operation. Pandas handles column selection efficiently even with millions of rows,
        so adding parallelism would introduce unnecessary overhead without performance benefits.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset to process.
        **kwargs : dict
            Additional parameters (not used here directly).

        Returns:
        --------
        Dict[str, pd.DataFrame]
            A dictionary where keys are group names and values are corresponding dataframes.
        """

        result_subsets = {}

        for group_name, fields in self.field_groups.items():
            selected_columns = fields.copy()

            # Add ID field if required and not already present
            if (
                self.include_id_field
                and self.id_field
                and self.id_field not in selected_columns
            ):
                selected_columns.insert(0, self.id_field)

            # Subset the DataFrame
            subset_df = df[selected_columns].copy()
            result_subsets[group_name] = subset_df

        return result_subsets

    def _collect_metrics(
        self,
        input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Dict[str, Any]:
        """
        Collect operation-specific metrics for SplitFieldsOperation and return in structured format.
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
            name: {"field_count": len(df.columns), "included_fields": list(df.columns)}
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
            "execution_time_seconds": round(self.end_time - self.start_time, 2),
            "processing_date": datetime.now().isoformat(),
        }

    def _save_metrics(
        self,
        metrics: Dict[str, Any],
        task_dir: Path,
        result: OperationResult,
        operation_timestamp: str,
        **kwargs,
    ) -> Path:
        """
        Save the structured metrics dictionary to a JSON file in the task directory.
        """
        metrics_dir = task_dir / "metrics"
        ensure_directory(metrics_dir)

        operation_name = self.operation_name.lower()
        metrics_filename = f"{operation_name}_metrics_{operation_timestamp}.json"
        metrics_path = metrics_dir / metrics_filename

        try:

            # Save metrics to file
            encryption_key = self.encryption_key if self.use_encryption else None
            write_json(metrics, metrics_path, encryption_key=encryption_key)

            result.add_artifact(
                artifact_type="json",
                path=metrics_path,
                description=f"Metrics for {operation_name} saved at {operation_timestamp}",
                category=Constants.Artifact_Category_Metrics,
            )

            self.logger.info(f"Structured metrics saved successfully to {metrics_path}")
            return metrics_path

        except Exception as e:
            self.logger.error(f"Error saving structured metrics to {metrics_path}: {e}")
            raise

    def _save_output(
        self,
        result_subsets: dict[str, pd.DataFrame],
        task_dir: Path,
        result: OperationResult,
        operation_timestamp: str,
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
        """
        output_dir = task_dir / "output"
        ensure_directory(output_dir)

        for dataset_name, df in result_subsets.items():
            filename = (
                f"{dataset_name}_output_{operation_timestamp}.{self.output_format}"
            )
            output_path = output_dir / filename

            try:
                encryption_mode = get_encryption_mode(df, **kwargs)
                if self.output_format == OutputFormat.CSV.value:
                    write_dataframe_to_csv(
                        df=df,
                        file_path=output_path,
                        encryption_key=self.encryption_key,
                        use_encryption=self.use_encryption,
                        encryption_mode=encryption_mode,
                    )
                elif self.output_format == OutputFormat.JSON.value:
                    if self.encryption_key:
                        file_path = Path(output_path)
                        temp_dir = file_path.parent / "temp"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        temp_destination_path = temp_dir / f"decrypted_{file_path.name}"

                        df.to_json(temp_destination_path, orient="records", lines=True)

                        crypto_utils.encrypt_file(
                            source_path=temp_destination_path,
                            destination_path=output_path,
                            key=self.encryption_key,
                            mode=encryption_mode,
                        )
                        directory_utils.safe_remove_temp_file(temp_destination_path)
                    else:
                        df.to_json(temp_destination_path, orient="records", lines=True)
                else:
                    self.logger.warning(
                        f"Unsupported output format: {self.output_format}"
                    )
                    continue

                self.logger.info(f"Saved output: {output_path}")
                result.add_artifact(
                    artifact_type=self.output_format,
                    path=output_path,
                    description=f"Output for {dataset_name} saved at {operation_timestamp}",
                    category=Constants.Artifact_Category_Output,
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
        """Generate visualizations specific to this operation using standard visualization module."""
        if not isinstance(output_data, dict) or not output_data:
            self.logger.warning(
                "Skipping visualization: output_data is not a non-empty dictionary of DataFrames."
            )
            return

        vis_dir = task_dir / "visualizations"
        ensure_directory(vis_dir)

        operation_name = self.operation_name

        kwargs_visualization = {
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
            "backend": self.visualization_backend,
            "theme": self.visualization_theme,
            "strict": self.visualization_strict,
        }

        # 1. Bar chart: number of fields per subset
        try:
            subset_field_counts = {k: len(df.columns) for k, df in output_data.items()}
            bar_chart_path = (
                vis_dir
                / f"{operation_name}_viz_fields_per_subset_{operation_timestamp}.png"
            )
            bar_result = create_bar_plot(
                data=subset_field_counts,
                output_path=bar_chart_path,
                title="Number of Fields per Subset",
                orientation="v",
                x_label="Subset",
                y_label="Number of Fields",
                sort_by="key",
                **kwargs_visualization,
            )
            if not bar_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=bar_result,
                    description="Bar chart showing number of fields in each subset",
                    category=Constants.Artifact_Category_Visualization,
                )
        except Exception as e:
            self.logger.error(f"Failed to create fields-per-subset bar chart: {e}")

        # 2. Network diagram: field distribution across subsets
        try:
            network_path = (
                vis_dir
                / f"{operation_name}_viz_field_subset_network_{operation_timestamp}.png"
            )
            network_result = plot_field_subset_network(
                output_data=output_data,
                output_path=network_path,
                title="Field Distribution Across Subsets (Network Diagram)",
                **kwargs_visualization,
            )
            if not network_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=network_result,
                    description="Network diagram showing field distribution across subsets (Plotly)",
                    category=Constants.Artifact_Category_Visualization,
                )
        except Exception as e:
            self.logger.error(f"Failed to create field-subset network diagram: {e}")

        # 3. Bar chart: schema comparison (optional)
        if isinstance(input_data, pd.DataFrame):
            try:
                schema_data = {"Original": len(input_data.columns)}
                schema_data.update(
                    {k: len(df.columns) for k, df in output_data.items()}
                )

                schema_path = (
                    vis_dir
                    / f"{operation_name}_viz_schema_comparison_{operation_timestamp}.png"
                )
                schema_result = create_bar_plot(
                    data=schema_data,
                    output_path=schema_path,
                    title="Schema Comparison: Original vs. Split Datasets",
                    orientation="v",
                    x_label="Dataset",
                    y_label="Number of Fields",
                    sort_by="key",
                    **kwargs_visualization,
                )
                if not schema_result.startswith("Error"):
                    result.add_artifact(
                        artifact_type="png",
                        path=schema_result,
                        description="Schema visualization of original vs. split datasets",
                        category=Constants.Artifact_Category_Visualization,
                    )
            except Exception as e:
                self.logger.error(f"Failed to create schema comparison chart: {e}")

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
            f"[VIZ] Preparing to generate visualizations in a separate thread"
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
                self.logger.info(f"[VIZ] Visualization thread completed successfully")

        except Exception as e:
            self.logger.error(
                f"[VIZ] Error setting up visualization thread: {e}", exc_info=True
            )

    def _save_cache(self, task_dir: Path, result: OperationResult, **kwargs) -> None:
        """
        Save the operation result to cache.

        Parameters
        ----------
        task_dir : Path
            Root directory for the task.
        result : OperationResult
            The result object to be cached.
        """
        try:
            result_data = {
                "status": (
                    result.status.name
                    if isinstance(result.status, OperationStatus)
                    else str(result.status)
                ),
                "metrics": result.metrics,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "error_trace": result.error_trace,
                "artifacts": [artifact.to_dict() for artifact in result.artifacts],
            }

            cache_data = {
                "result": result_data,
                "parameters": self._get_cache_parameters(**kwargs),
            }

            cache_key = operation_cache.generate_cache_key(
                operation_name=self.operation_name,
                parameters=self._get_cache_parameters(**kwargs),
                data_hash=self._generate_data_hash(self._original_df.copy()),
            )

            operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            self.logger.info(f"Saved result to cache with key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _get_cache(self, df: pd.DataFrame, **kwargs) -> Optional[OperationResult]:
        """
        Retrieve cached result if available and valid.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame used to generate the cache key.

        Returns
        -------
        Optional[OperationResult]
            The cached OperationResult if available, otherwise None.
        """
        try:
            cache_key = operation_cache.generate_cache_key(
                operation_name=self.operation_name,
                parameters=self._get_cache_parameters(**kwargs),
                data_hash=self._generate_data_hash(df),
            )

            cached = operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            result_data = cached.get("result")
            if not isinstance(result_data, dict):
                return None

            # Parse enum safely
            status_str = result_data.get("status", OperationStatus.ERROR.name)
            status = (
                OperationStatus[status_str]
                if isinstance(status_str, str)
                and status_str in OperationStatus.__members__
                else OperationStatus.ERROR
            )

            # Rebuild artifacts
            artifacts = []
            for art_dict in result_data.get("artifacts", []):
                if isinstance(art_dict, dict):
                    try:
                        artifacts.append(
                            OperationArtifact(
                                artifact_type=art_dict.get("type"),
                                path=art_dict.get("path"),
                                description=art_dict.get("description", ""),
                                category=art_dict.get("category", "output"),
                                tags=art_dict.get("tags", []),
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to deserialize artifact: {e}")

            return OperationResult(
                status=status,
                artifacts=artifacts,
                metrics=result_data.get("metrics", {}),
                error_message=result_data.get("error_message"),
                execution_time=result_data.get("execution_time"),
                error_trace=result_data.get("error_trace"),
            )

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _get_cache_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Get operation-specific parameters for SplitFieldsOperation using external kwargs.

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
            "operation": self.operation_name,
            "version": self.version,
            "id_field": self.id_field,
            "field_groups": self.field_groups,
            "include_id_field": self.include_id_field,
            "output_format": self.output_format,
            "save_output": self.save_output,
            "use_cache": self.use_cache,
            "force_recalculation": self.force_recalculation,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "visualization_backend": self.visualization_backend,
            "visualization_theme": self.visualization_theme,
            "visualization_strict": self.visualization_strict,
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
        }

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Generate a hash that represents key characteristics of the input DataFrame.

        The hash is based on structure and summary statistics to detect changes
        for caching purposes.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to generate a representative hash from.

        Returns
        -------
        str
            A hash string representing the structure and key properties of the data.
        """
        try:
            characteristics = {
                "columns": list(data.columns),
                "shape": data.shape,
                "summary": {},
            }

            for col in data.columns:
                col_data = data[col]
                col_info = {
                    "dtype": str(col_data.dtype),
                    "null_count": int(col_data.isna().sum()),
                    "unique_count": int(col_data.nunique()),
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    non_null = col_data.dropna()
                    if not non_null.empty:
                        col_info.update(
                            {
                                "min": float(non_null.min()),
                                "max": float(non_null.max()),
                                "mean": float(non_null.mean()),
                                "median": float(non_null.median()),
                                "std": float(non_null.std()),
                            }
                        )
                elif pd.api.types.is_object_dtype(col_data) or isinstance(
                    col_data.dtype, pd.CategoricalDtype
                ):
                    top_values = col_data.value_counts(dropna=True).head(5)
                    col_info["top_values"] = {
                        str(k): int(v) for k, v in top_values.items()
                    }

                characteristics["summary"][col] = col_info

            json_str = json.dumps(characteristics, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")
            fallback = f"{data.shape}_{list(data.dtypes)}"
            return hashlib.md5(fallback.encode()).hexdigest()

    def _validate_input_parameters(self, df: pd.DataFrame) -> bool:
        """
        Validate that all specified fields in field_groups exist in the DataFrame.
        Optionally check if the ID field exists.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset to validate.

        Returns:
        --------
        bool
            True if all fields are valid; False otherwise.
        """
        all_columns = set(df.columns)

        # Ensure field_groups is not empty
        if not self.field_groups:
            self.logger.error("field_groups must not be empty.")
            return False

        # Check that every field in each group exists in the DataFrame
        for group_name, fields in self.field_groups.items():
            for field in fields:
                if field not in all_columns:
                    self.logger.error(
                        f"Field '{field}' in group '{group_name}' not found in DataFrame."
                    )
                    return False

        # If ID field is to be included, check that it exists in the DataFrame
        if self.include_id_field and self.id_field:
            if self.id_field not in all_columns:
                self.logger.error(f"ID field '{self.id_field}' not found in DataFrame.")
                return False

        # All validations passed
        return True

    def _load_data_and_validate_input_parameters(
        self, data_source: DataSource, **kwargs
    ) -> Tuple[Optional[pd.DataFrame], bool]:

        dataset_name = kwargs.get("dataset_name", "main")
        settings_operation = load_settings_operation(
            data_source, dataset_name, **kwargs
        )
        df = load_data_operation(data_source, dataset_name, **settings_operation)

        if df is None or df.empty:
            self.logger.error("Error data frame is None or empty")
            return None, False

        self._original_df = df.copy(deep=True)

        return df, self._validate_input_parameters(df)

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
