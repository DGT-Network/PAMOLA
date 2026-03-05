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

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import pandas as pd
from pamola_core.transformations.schemas.split_fields_op_core_schema import (
    SplitFieldsOperationConfig,
)
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.exceptions import FieldNotFoundError, MissingParameterError
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.io import (
    ensure_directory,
    load_settings_operation,
    write_json,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.common.constants import Constants
from pamola_core.utils.visualization import create_bar_plot, plot_field_subset_network
from pamola_core.utils.ops.op_registry import register


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
                            f"Split fields (from cache)",
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
            subset_df = df[selected_columns].copy(deep=True)
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

    def _get_cache_parameters(self) -> Dict[str, Any]:
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
            "id_field": self.id_field,
            "field_groups": self.field_groups,
            "include_id_field": self.include_id_field,
        }

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
            raise MissingParameterError(
                param_name="field_groups",
                operation=self.operation_name or "split_fields",
            )

        # Check that every field in each group exists in the DataFrame
        for group_name, fields in self.field_groups.items():
            for field in fields:
                if field not in all_columns:
                    raise FieldNotFoundError(
                        field_name=field,
                        available_fields=list(all_columns),
                    )

        # If ID field is to be included, check that it exists in the DataFrame
        if self.include_id_field and self.id_field:
            if self.id_field not in all_columns:
                raise FieldNotFoundError(
                    field_name=self.id_field,
                    available_fields=list(all_columns),
                )

        # All validations passed
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
