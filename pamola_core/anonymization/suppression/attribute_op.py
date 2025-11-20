"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Attribute Suppression Operation Module

License: BSD 3-Clause
Copyright (c) 2025 PAMOLA Development Team

Version: 1.4.0
Last Updated: 2025-06-15

Description:
    This module implements the AttributeSuppressionOperation class for removing
    entire columns (attributes) from datasets as part of the PAMOLA anonymization
    framework. This operation provides a simple yet effective method for protecting
    privacy by completely removing sensitive attributes.

Key Features:
    - Remove single or multiple columns from datasets
    - Save metadata about suppressed columns (dtype, null counts, unique values)
    - Generate suppression metrics and visualizations
    - Full integration with PAMOLA base framework
    - Support for batch processing (though typically processes entire DataFrame)
    - Dask support for distributed processing

Changelog:
    1.4.0 (2025-06-15):
        - Fixed all signature errors and type mismatches
        - Fixed FieldNotFoundError usage to match actual exception signature
        - Fixed visualization path return types
        - Fixed DataWriter result handling
        - Fixed artifact registration for all visualizations
        - Improved type annotations and error handling
"""

from datetime import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons import calculate_anonymization_effectiveness
from pamola_core.anonymization.commons.validation import (
    check_multiple_fields_exist,
    FieldNotFoundError,
)
from pamola_core.anonymization.commons.visualization_utils import create_bar_plot
from pamola_core.anonymization.schemas.attribute_op_core_schema import AttributeSuppressionConfig
from pamola_core.common.constants import Constants
from pamola_core.utils.io import (
    load_settings_operation,
    load_data_operation,
    ensure_directory,
    write_dataframe_to_csv,
    write_json,
)
from pamola_core.utils.io_helpers import crypto_utils, directory_utils
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_field_utils import (
    create_field_mask,
    create_multi_field_mask,
)
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
    OperationArtifact,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_registry import register

import dask.dataframe as dd

# Type alias for suppressed schema structure
SuppressedSchema = Dict[str, Dict[str, Any]]


@register(version="1.0.0")
class AttributeSuppressionOperation(AnonymizationOperation):
    """
    Operation for removing one or more columns (attributes) from datasets.

    Implements REQ-ATTR-001 through REQ-ATTR-005 from the
    PAMOLA.CORE Suppression Operations Sub-Specification.
    """

    def __init__(
        self,
        field_name: str,
        additional_fields: Optional[List[str]] = None,
        suppression_mode: str = "REMOVE",
        save_suppressed_schema: bool = True,
        **kwargs,
    ):
        """
        Initialize the Attribute Suppression Operation.

        This operation removes one or more columns (attributes) from the dataset,
        optionally based on conditional logic or k-anonymity risk thresholds.

        Parameters
        ----------
        field_name : str
            Name of the primary field (column) to suppress. Required.
        additional_fields : list[str], optional
            Additional fields to suppress alongside `field_name`.
        suppression_mode : str, optional
            Operation suppression_mode. Must be "REMOVE" for attribute suppression. Default = "REMOVE".
        save_suppressed_schema : bool, optional
            Whether to save metadata about suppressed columns. Default = True.
        **kwargs
            Additional keyword arguments passed to AnonymizationOperation.
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Attribute suppresstion for '{field_name}' using {suppression_mode} suppression mode",
        )

        # --- Validate suppression_mode ---
        if suppression_mode != "REMOVE":
            raise ValueError(
                f"AttributeSuppressionOperation only supports suppression_mode='REMOVE', got '{suppression_mode}'"
            )

        # --- Build config object ---
        config = AttributeSuppressionConfig(
            field_name=field_name,
            additional_fields=additional_fields or [],
            suppression_mode=suppression_mode,
            save_suppressed_schema=save_suppressed_schema,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # --- Initialize base operation ---
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # --- Save config attributes ---
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # --- Internal state ---
        self._suppressed_schema: SuppressedSchema = {}
        self._suppression_count = 0
        self._original_column_count = 0

        # --- Metadata ---
        self.operation_name = self.__class__.__name__
        self._original_df = None

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any = None,
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
            # Initialize operation
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initialize result object
            result = OperationResult(
                status=OperationStatus.PENDING,
                artifacts=[],
                metrics={},
                error_message=None,
                execution_time=0,
                error_trace=None,
            )

            # Start operation - Preparation
            self.logger.info(
                f"Operation: {self.operation_name}, Start operation - Preparation"
            )

            # Handle preparation
            self._handle_preparation(
                task_dir=task_dir,
                progress_tracker=progress_tracker,
                reporter=reporter,
                **kwargs,
            )

            # Load data and validate input parameters
            self.logger.info(
                f"Operation: {self.operation_name}, Load data and validate input parameters"
            )
            df, is_valid = self._load_data_and_validate_input_parameters(
                data_source,
                progress_tracker=progress_tracker,
                reporter=reporter,
                **kwargs,
            )

            # Handle cache if required
            if self.use_cache and not self.force_recalculation:
                try:
                    self.logger.info(
                        f"Operation: {self.operation_name}, Load result from cache"
                    )
                    cached_result = self._get_cache(
                        df.copy(), progress_tracker=progress_tracker, reporter=reporter
                    )
                    if cached_result is not None and isinstance(
                        cached_result, OperationResult
                    ):
                        return cached_result
                except Exception as e:
                    error_message = f"Error checking cache: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            try:
                # Process data
                self.logger.info(f"Operation: {self.operation_name}, Process data")
                mask, output_data = self._process_data(
                    df, progress_tracker=progress_tracker, reporter=reporter
                )
            except Exception as e:
                error_message = f"Error processing data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Record end time after processing metrics
            self.end_time = time.time()

            try:
                # Handle metric
                self.logger.info(f"Operation: {self.operation_name}, Collect metric")
                self._handle_metrics(
                    input_data=df,
                    output_data=output_data,
                    mask=mask,
                    result=result,
                    task_dir=task_dir,
                    progress_tracker=progress_tracker,
                    reporter=reporter,
                    operation_timestamp=operation_timestamp,
                )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Save output if required
            if self.save_output:
                try:
                    self.logger.info(f"Operation: {self.operation_name}, Save output")
                    self._save_output(
                        output_data=output_data,
                        task_dir=task_dir,
                        result=result,
                        progress_tracker=progress_tracker,
                        reporter=reporter,
                        operation_timestamp=operation_timestamp,
                    )
                except Exception as e:
                    error_message = f"Error saving output: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Generate visualizations if required
            if self.generate_visualization:
                try:
                    self.logger.info(
                        f"Operation: {self.operation_name}, Generate visualizations"
                    )
                    self._handle_visualizations(
                        input_data=df,
                        output_data=output_data,
                        task_dir=task_dir,
                        result=result,
                        progress_tracker=progress_tracker,
                        reporter=reporter,
                        operation_timestamp=operation_timestamp,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical

            # Save cache if required
            if self.use_cache:
                try:
                    self.logger.info(f"Operation: {self.operation_name}, Save cache")
                    self._save_cache(
                        task_dir,
                        result,
                        progress_tracker=progress_tracker,
                        reporter=reporter,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Finalize timing
            self.end_time = time.time()

            # Set success status
            result.status = OperationStatus.SUCCESS
            result.execution_time = self.end_time - self.start_time
            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
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
                status=OperationStatus.ERROR,
                error_message=str(e),
                exception=e,
            )

    def _process_batch_dask(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Process Dask DataFrame for distributed computing.

        Implements REQ-ANON-009 for Dask support. Attribute suppression is
        particularly simple for Dask as it's just column removal.

        Args:
            ddf: Dask DataFrame

        Returns:
            Dask DataFrame with columns removed

        Raises:
            FieldNotFoundError: If specified columns don't exist
        """
        # Store original column count for metrics (cheap operation)
        self._original_column_count = len(ddf.columns)

        # Collect all fields to suppress
        fields_to_drop = [self.field_name]
        if self.additional_fields:
            fields_to_drop.extend(self.additional_fields)

        # Remove duplicates
        fields_to_drop = list(dict.fromkeys(fields_to_drop))

        # Validate columns exist (cheap operation - no compute required)
        missing_columns = [col for col in fields_to_drop if col not in ddf.columns]
        if missing_columns:
            # FieldNotFoundError expects (field_name, existing_fields) signature
            raise FieldNotFoundError(
                missing_columns[0],  # First missing field
                list(ddf.columns),  # All existing columns
            )

        # Update suppression count for metrics
        self._suppression_count = len(fields_to_drop)

        self.logger.info(f"Dropping {len(fields_to_drop)} columns from Dask DataFrame")

        # Note: We cannot collect metadata for Dask without computing
        # This is a known limitation when using distributed processing
        if self.save_suppressed_schema:
            self.logger.warning(
                "Column metadata collection is not available for Dask DataFrames. "
                "Schema will only contain column names."
            )
            self._suppressed_schema = {col: {"dask": True} for col in fields_to_drop}

        return ddf.drop(columns=fields_to_drop)

    def _handle_preparation(
        self,
        task_dir: Path,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Handle preparation step at the beginning of the execute method.

        This includes:
        - Logging the start of the operation.
        - Setting up progress tracker total steps.
        - Preparing directories.
        - Sending preparation status to reporter.

        Parameters
        ----------
        task_dir : Path
            Root path for the task's working directory.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker for updating progress status.
        reporter : Optional[Any]
            Optional reporter for external operation updates.
        **kwargs : dict
            Additional keyword arguments passed to compute steps or directory creation.

        Returns
        -------
        Dict[str, Path]
            Dictionary of prepared directory paths.
        """
        step = "Preparation"

        # Setup total progress steps
        if progress_tracker:
            progress_tracker.total = self._compute_total_steps()
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        # Prepare necessary directories
        dirs = self._prepare_directories(task_dir)

        # Initialize operation cache
        self.operation_cache = OperationCache(
            cache_dir=task_dir / "cache",
        )

        # Save configuration to task directory
        self.save_config(task_dir)

        # Report preparation success
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

        return dirs

    def _build_suppression_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Build boolean mask based on condition fields, multi-conditions, and risk filtering.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to filter.

        Returns
        -------
        pd.Series
            Boolean mask indicating rows that match all specified conditions.
        """
        mask = pd.Series(True, index=df.index)

        if self.condition_field and self.condition_values:
            field_mask = create_field_mask(
                df,
                self.field_name,
                self.condition_field,
                self.condition_values,
                self.condition_operator,
            )
            mask &= field_mask
            self.logger.info(
                f"Applied condition on '{self.condition_field}': {mask.sum()} records match"
            )

        if self.multi_conditions:
            multi_mask = create_multi_field_mask(
                df, self.multi_conditions, self.condition_logic
            )
            mask &= multi_mask
            self.logger.info(
                f"Applied {len(self.multi_conditions)} conditions: {mask.sum()} records match"
            )

        if self.ka_risk_field and self.ka_risk_field in df.columns:
            risk_mask = df[self.ka_risk_field] < self.risk_threshold
            mask &= risk_mask
            self.logger.info(
                f"Applied k-anonymity filter (k < {self.risk_threshold}): {mask.sum()} vulnerable records"
            )

        return mask

    def _process_data(
        self,
        input_data: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Apply conditional filtering and suppress specified columns from the dataset.

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            - mask: Boolean mask of rows that match suppression conditions.
            - result: Final DataFrame after suppressing selected columns.
        """
        step = "Process data"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            mask = self._build_suppression_mask(input_data)
            filtered_df = input_data[mask].copy()
            self._original_column_count = len(filtered_df.columns)

            fields_to_drop = [self.field_name] + (self.additional_fields or [])
            unique_fields = list(dict.fromkeys(fields_to_drop))

            if len(unique_fields) != len(fields_to_drop):
                duplicates = [f for f in fields_to_drop if fields_to_drop.count(f) > 1]
                self.logger.warning(
                    f"Duplicate fields in suppression list will be ignored: {set(duplicates)}"
                )

            missing_check = check_multiple_fields_exist(filtered_df, unique_fields)
            if not missing_check[0]:
                missing_fields = missing_check[1]
                raise FieldNotFoundError(
                    ", ".join(missing_fields), list(filtered_df.columns)
                )

            if self.save_suppressed_schema:
                self._collect_suppressed_metadata(filtered_df, unique_fields)

            result = filtered_df.drop(columns=unique_fields)
            result.reset_index(drop=True, inplace=True)

            self._suppression_count = len(unique_fields)
            self.logger.info(
                f"Suppressed {self._suppression_count} columns. Remaining columns: {len(result.columns)}"
            )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Process data successfully",
                        "columns_suppressed": self._suppression_count,
                        "remaining_columns": len(result.columns),
                    },
                )

            return mask, result

        except Exception as e:
            self.logger.error(
                f"{self.operation_name} - {step} failed: {e}", exc_info=True
            )
            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="error",
                    details={
                        "step": step,
                        "message": "Process data failed",
                        "error": str(e),
                    },
                )
            raise

    def _collect_suppressed_metadata(
        self, df: pd.DataFrame, columns: List[str]
    ) -> None:
        """
        Collect metadata about columns to be suppressed.

        Args:
            df: Source DataFrame
            columns: List of column names to collect metadata for
        """
        self.logger.debug("Collecting metadata for suppressed columns")

        for col in columns:
            try:
                ser = df[col]
                meta: Dict[str, Any] = {
                    "dtype": str(ser.dtype),
                    "null_count": int(ser.isna().sum()),
                    "non_null_count": int(ser.notna().sum()),
                    "unique_count": int(ser.nunique(dropna=True)),
                    "memory_usage": int(ser.memory_usage(deep=True)),
                }

                # Add basic statistics for numeric columns
                if pd.api.types.is_numeric_dtype(ser):
                    numeric_ser = pd.to_numeric(ser, errors="coerce").dropna()
                    if not numeric_ser.empty:
                        # Convert numpy scalars to Python types for JSON serialization
                        meta.update(
                            {
                                "min": float(numeric_ser.min()),
                                "max": float(numeric_ser.max()),
                                "mean": float(numeric_ser.mean()),
                                "std": float(numeric_ser.std(ddof=0)),
                            }
                        )

                self._suppressed_schema[col] = meta

            except Exception as e:
                self.logger.warning(
                    f"Failed to collect metadata for column '{col}': {e}"
                )
                self._suppressed_schema[col] = {"error": str(e)}

    def _collect_specific_metrics(
        self, original_data: Optional[pd.Series], anonymized_data: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """
        Collect attribute suppression specific metrics.

        Note: For attribute suppression, we don't have series-level data since
        we're removing entire columns. This method provides operation-level metrics.

        Args:
            original_data: Not used for attribute suppression
            anonymized_data: Not used for attribute suppression

        Returns:
            Dictionary of suppression metrics
        """
        metrics: Dict[str, Any] = {
            "operation_type": "attribute_suppression",
            "columns_suppressed": self._suppression_count,
            "suppressed_column_names": list(self._suppressed_schema.keys()),
        }

        # Calculate data width reduction
        if self._original_column_count > 0:
            metrics["data_width_reduction"] = (
                self._suppression_count / self._original_column_count * 100
            )
        else:
            metrics["data_width_reduction"] = 0.0

        # Note: We don't include the full schema in metrics to keep them lightweight
        # The schema is saved as a separate artifact if requested

        return metrics

    def _collect_metrics(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        mask: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Collect all relevant metrics after performing attribute or record suppression.

        This function may raise exceptions if data structures are malformed.
        Caller is responsible for exception handling.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing effectiveness, performance, filtering, and suppression metrics.
        """
        metrics: Dict[str, Any] = {}

        try:
            # Effectiveness metrics
            shared_columns = list(set(input_data.columns) & set(output_data.columns))
            for col in shared_columns:
                col_effectiveness = calculate_anonymization_effectiveness(
                    input_data[col], output_data[col]
                )
                metrics[f"{col}_effectiveness_ratio"] = col_effectiveness.get(
                    "effectiveness_ratio", 0.0
                )
                metrics[f"{col}_original_unique"] = col_effectiveness.get(
                    "original_unique", 0
                )
                metrics[f"{col}_anonymized_unique"] = col_effectiveness.get(
                    "anonymized_unique", 0
                )
                metrics[f"{col}_null_increase"] = col_effectiveness.get(
                    "null_increase", 0.0
                )

            # Timing
            if self.start_time and self.end_time:
                duration = self.end_time - self.start_time
                metrics.update(
                    {
                        "duration_seconds": round(duration, 2),
                        "records_processed": self.process_count,
                        "records_per_second": (
                            round(self.process_count / duration, 2)
                            if duration > 0
                            else 0
                        ),
                        "chunk_count": (self.process_count + self.chunk_size - 1)
                        // self.chunk_size,
                    }
                )

            if mask is not None and isinstance(mask, pd.Series):
                total = len(mask)
                processed = mask.sum()
                metrics.update(
                    {
                        "total_records": total,
                        "processed_records": processed,
                        "filtered_records": total - processed,
                        "processing_rate": (
                            (processed / total) * 100 if total > 0 else 0
                        ),
                    }
                )

            if self.optimize_memory:
                metrics["chunk_size_used"] = self.chunk_size
                metrics["adaptive_chunk_size"] = self.adaptive_chunk_size

            # Suppression-specific metrics
            specific_metrics = self._collect_specific_metrics(
                original_data=None, anonymized_data=None
            )
            metrics.update(specific_metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error in _collect_metrics: {e}", exc_info=True)
            raise

    def _save_metrics(
        self, metrics: Dict[str, Any], task_dir: Path, result: OperationResult, operation_timestamp: Optional[str] = None
    ) -> None:
        """
        Save the collected metrics as a JSON file and register it as an artifact.

        Raises:
            Exception: If saving the metrics file fails.
        """
        try:
            metrics_dir = task_dir / "metrics"
            ensure_directory(metrics_dir)

            operation_name = self.operation_name.lower()
            metrics_filename = f"{operation_name}_metrics_{operation_timestamp}.json"
            metrics_path = metrics_dir / metrics_filename

            write_json(metrics, metrics_path, encryption_key=self.encryption_key)

            result.add_artifact(
                artifact_type="json",
                path=metrics_path,
                description=f"Metrics for {self.operation_name}",
                category=Constants.Artifact_Category_Metrics,
            )

            self.logger.info(f"Structured metrics saved to {metrics_path}")

            # Save suppressed schema if requested
            if self.save_suppressed_schema and self._suppressed_schema:
                try:
                    schema_filename = f"{operation_name}_suppressed_columns_schema_{operation_timestamp}.json"
                    schema_path = metrics_dir / schema_filename

                    # Write the schema using write_json method
                    write_json(
                        self._suppressed_schema,
                        schema_path,
                        encryption_key=self.encryption_key,
                    )

                    self.logger.info(
                        f"Saved suppressed column schema to: {schema_path}"
                    )

                    # Register the schema file as an artifact
                    result.add_artifact(
                        artifact_type="json",
                        path=schema_path,
                        description=f"Metadata about suppressed columns including data types and statistics",
                        category=Constants.Artifact_Category_Metrics,
                    )

                except Exception as e:
                    self.logger.error(f"Failed to save suppressed schema: {e}")
                    # Don't fail the operation if schema saving fails
                    result.add_metric("suppressed_schema_error", str(e))

        except Exception as e:
            self.logger.error(
                f"Failed to save metrics file to {metrics_path}: {e}", exc_info=True
            )
            raise

    def _handle_metrics(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        mask: Optional[pd.Series],
        result: OperationResult,
        task_dir: Path,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
        operation_timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle the collection and saving of metrics for the operation.

        This includes:
        - Calculating effectiveness, performance, filtering, and suppression metrics.
        - Saving the metrics as a JSON artifact into the task directory.
        - Attaching the metrics to the result object and logging/reporting progress.

        Parameters
        ----------
        input_data : pd.DataFrame
            Original input dataset before suppression.
        output_data : pd.DataFrame
            Resulting dataset after suppression.
        mask : Optional[pd.Series]
            Boolean mask indicating processed records.
        result : OperationResult
            Result object to attach metrics and artifacts.
        task_dir : Path
            Directory where metrics will be saved.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker to update.
        reporter : Optional[Any]
            Reporter to log detailed status.
        operation_timestamp : Optional[str]
            Timestamp string to use for naming metric files.

        Returns
        -------
        Optional[Dict[str, Any]]
            The collected metrics dictionary, or None if an error occurred.
        """
        step = "Collect metrics"
        try:
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": step, "operation": self.operation_name}
                )

            metrics = self._collect_metrics(input_data, output_data, mask)
            result.metrics = metrics
            self._save_metrics(metrics, task_dir, result, operation_timestamp)

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": "Collect metric",
                        "message": "Metrics collected and saved successfully",
                        "summary": {
                            "records_processed": metrics.get("records_processed"),
                            "total_records": metrics.get("total_records"),
                            "processed_records": metrics.get("processed_records"),
                            "filtered_records": metrics.get("filtered_records"),
                            "processing_rate": metrics.get("processing_rate"),
                            "columns_suppressed": metrics.get("columns_suppressed"),
                            "data_width_reduction": metrics.get("data_width_reduction"),
                            "duration_seconds": metrics.get("duration_seconds"),
                            "records_per_second": metrics.get("records_per_second"),
                        },
                    },
                )
            return metrics

        except Exception as e:
            self.logger.error(
                f"Failed to handle metrics in {self.operation_name}: {e}", exc_info=True
            )
            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="error",
                    details={"step": "Collect metric", "message": str(e)},
                )
            return None

    def _save_output(
        self,
        output_data: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
        operation_timestamp: Optional[str] = None,
    ):
        """
        Save the processed output DataFrame to disk and register it as an artifact.

        This method handles saving the data in the configured format (e.g., CSV, JSON),
        encrypting it if needed, and logging progress and errors. It also reports the
        result to any registered reporter and adds the output file as an artifact.

        Parameters
        ----------
        output_data : pd.DataFrame
            Processed DataFrame to save.
        task_dir : Path
            Path to the task directory.
        result : OperationResult
            Operation result object to register output artifacts.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker instance to update progress.
        reporter : Optional[Any]
            Reporter object to log progress and status.
        operation_timestamp : Optional[str]
            Timestamp string to use for naming output files.
        """
        step = "Save output"
        operation_name = self.operation_name.lower()

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        output_dir = task_dir / "output"
        ensure_directory(output_dir)

        use_encryption = self.use_encryption
        encryption_key = self.encryption_key
        kwargs_encryption = {
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
            "encryption_mode": self.encryption_mode,
        }
        encryption_mode = get_encryption_mode(output_data, **kwargs_encryption)
        filename = f"{operation_name}_{self.field_name}_output_{operation_timestamp}.{self.output_format}"
        output_path = output_dir / filename

        try:
            if self.output_format == "csv":
                write_dataframe_to_csv(
                    df=output_data,
                    file_path=output_path,
                    encryption_key=encryption_key,
                    use_encryption=use_encryption,
                    encryption_mode=encryption_mode,
                )

            elif self.output_format == "json":
                if encryption_key:
                    temp_dir = output_path.parent / "temp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / f"decrypted_{output_path.name}"

                    output_data.to_json(temp_path, orient="records", lines=True)
                    crypto_utils.encrypt_file(
                        source_path=temp_path,
                        destination_path=output_path,
                        key=encryption_key,
                        mode=encryption_mode,
                    )
                    directory_utils.safe_remove_temp_file(temp_path)
                else:
                    output_data.to_json(output_path, orient="records", lines=True)

            else:
                warning_msg = f"Unsupported output format: {self.output_format}"
                self.logger.warning(warning_msg)
                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="warning",
                        details={"step": step, "message": warning_msg},
                    )
                return

            self.logger.info(f"Saved output: {output_path}")
            result.add_artifact(
                artifact_type=self.output_format,
                path=output_path,
                description=f"Save output successfully",
                category=Constants.Artifact_Category_Output,
            )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Save output successfully",
                        "file": str(output_path),
                    },
                )

        except Exception as e:
            error_msg = f"Failed to save processed output to {output_path}: {e}"
            self.logger.error(error_msg, exc_info=True)

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="error",
                    details={
                        "step": step,
                        "message": "Save output failed",
                        "error": str(e),
                    },
                )

    def _generate_visualizations(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        operation_timestamp: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Generate visualizations showing columns before/after suppression and data type distribution.

        Args:
            input_data: Original DataFrame
            output_data: Processed DataFrame
            task_dir: Directory to save visualization
            result: OperationResult to register visualization artifacts
            operation_timestamp: Timestamp string for naming files

        Returns:
            Path to saved visualization or None
        """

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

        if not self._suppression_count:
            self.logger.info("No columns were suppressed. Skipping visualization.")
            return None

        try:
            # Prepare data for main visualization
            categories = ["Before", "After"]
            values = [len(input_data.columns), len(output_data.columns)]

            viz_data = pd.DataFrame({"Status": categories, "Column Count": values})
            bar_file_name = f"{operation_name}_attribute_suppression_summary_{operation_timestamp}.png"
            bar_path = vis_dir / bar_file_name
            bar_result = create_bar_plot(
                data=viz_data.set_index("Status")["Column Count"].to_dict(),
                output_path=bar_path,
                title=f"Attribute Suppression: {self._suppression_count} Columns Removed",
                x_label="Status",
                y_label="Column Count",
                orientation="v",
                **kwargs_visualization,
            )

            if not bar_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=bar_result,
                    description="Bar chart showing number of columns before and after suppression",
                    category=Constants.Artifact_Category_Visualization,
                )
            else:
                self.logger.error(f"Failed to create main visualization: {bar_result}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to generate main suppression visualization: {e}")
            return None

        if self._suppressed_schema and len(self._suppressed_schema) > 0:
            try:
                dtypes_count: Dict[str, int] = {}
                for col_info in self._suppressed_schema.values():
                    if (
                        isinstance(col_info, dict)
                        and "error" not in col_info
                        and "dask" not in col_info
                    ):
                        dtype = col_info.get("dtype", "unknown")
                        if "int" in dtype:
                            dtype = "integer"
                        elif "float" in dtype:
                            dtype = "float"
                        elif "object" in dtype:
                            dtype = "string/object"
                        elif "datetime" in dtype:
                            dtype = "datetime"
                        elif "bool" in dtype:
                            dtype = "boolean"

                        dtypes_count[dtype] = dtypes_count.get(dtype, 0) + 1

                if dtypes_count:
                    dtype_bar_file_name = f"{operation_name}_suppressed_columns_dtype_distribution_{operation_timestamp}.png"
                    dtype_bar_path = vis_dir / dtype_bar_file_name
                    dtype_bar_result = create_bar_plot(
                        data=dtypes_count,
                        output_path=dtype_bar_path,
                        title="Suppressed Columns by Data Type",
                        x_label="Data Type",
                        y_label="Count",
                        orientation="v",
                        **kwargs_visualization,
                    )

                    if not dtype_bar_result.startswith("Error"):
                        result.add_artifact(
                            artifact_type="png",
                            path=dtype_bar_result,
                            description="Bar chart showing data type distribution of suppressed columns",
                            category=Constants.Artifact_Category_Visualization,
                        )
                    else:
                        self.logger.error(
                            f"Failed to create dtype distribution visualization: {dtype_bar_result}"
                        )

            except Exception as e:
                self.logger.warning(
                    f"Failed to generate data type distribution visualization: {e}"
                )

        self.logger.info(
            f"Successfully generated suppression visualizations in: {vis_dir}"
        )
        return Path(bar_result)

    def _handle_visualizations(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
        operation_timestamp: Optional[str] = None,
    ) -> None:
        """
        Handle generation of visualizations in a separate thread, with logging and timeout.

        Parameters
        ----------
        input_data : pd.DataFrame
            The original input DataFrame.
        output_data : pd.DataFrame
            The processed/suppressed DataFrame.
        task_dir : Path
            The directory where visualizations will be saved.
        result : OperationResult
            The result object to store visualization artifacts.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker to report progress.
        reporter : Optional[Any]
            Optional reporter to log operation status.
        operation_timestamp : Optional[str]
            Timestamp string for naming files.
        """
        import threading
        import contextvars

        step = "Generate visualizations"
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
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": step, "operation": self.operation_name}
                )

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
                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="warning",
                        details={
                            "step": "Generate visualizations",
                            "message": f"Visualization thread timed out after {self.visualization_timeout}s",
                        },
                    )
            elif viz_error:
                self.logger.warning(f"[VIZ] Visualization thread failed: {viz_error}")
                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="warning",
                        details={
                            "step": "Generate visualizations",
                            "message": f"Visualization thread failed: {viz_error}",
                        },
                    )
            else:
                self.logger.info(f"[VIZ] Visualization thread completed successfully")
                if reporter:
                    num_images = len(
                        [a for a in result.artifacts if a.artifact_type == "png"]
                    )
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "step": "Generate visualizations",
                            "message": "Visualization completed successfully",
                            "num_images": num_images,
                        },
                    )

        except Exception as e:
            self.logger.error(
                f"[VIZ] Error setting up visualization thread: {e}", exc_info=True
            )
            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="error",
                    details={
                        "step": "Generate visualizations",
                        "message": f"Exception during visualization setup: {e}",
                    },
                )

    def _save_cache(
        self,
        task_dir: Path,
        result: OperationResult,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> None:
        """
        Save the operation result to cache and update progress or reporter if available.

        Parameters
        ----------
        task_dir : Path
            Root directory for the task.
        result : OperationResult
            The result object to be cached.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker to update UI or logs.
        reporter : Optional[Any]
            Reporter object to log external updates.
        **kwargs : dict
            Additional keyword arguments, used to compute the cache key.
        """
        step = "Save cache"

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
                "parameters": self._get_cache_parameters(),
            }

            cache_key = self.operation_cache.generate_cache_key(
                operation_name=self.operation_name,
                parameters=self._get_cache_parameters(),
                data_hash=self._generate_data_hash(self._original_df.copy()),
            )

            self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            self.logger.info(f"Saved result to cache with key: {cache_key}")

            if progress_tracker:
                progress_tracker.update(
                    1, {"step": step, "operation": self.operation_name}
                )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": "Save cache",
                        "message": "Save cache successfully",
                    },
                )

        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}", exc_info=True)

    def _get_cache(
        self,
        df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> Optional[OperationResult]:
        """
        Retrieve cached result if available and valid.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame used to generate the cache key.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker to log cache step.
        reporter : Optional[Any]
            Operation reporter for logging.
        **kwargs : dict
            Additional parameters for cache key generation.

        Returns
        -------
        Optional[OperationResult]
            The cached OperationResult if available, otherwise None.
        """
        step = "Load result from cache"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            cache_key = self.operation_cache.generate_cache_key(
                operation_name=self.operation_name,
                parameters=self._get_cache_parameters(),
                data_hash=self._generate_data_hash(df),
            )

            cached = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            result_data = cached.get("result")
            if not isinstance(result_data, dict):
                raise ValueError("Cached result is not a valid dictionary")

            status_str = result_data.get("status", OperationStatus.ERROR.name)
            status = (
                OperationStatus[status_str]
                if status_str in OperationStatus.__members__
                else OperationStatus.ERROR
            )

            artifacts = []
            for art_dict in result_data.get("artifacts", []):
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

            result = OperationResult(
                status=status,
                artifacts=artifacts,
                metrics=result_data.get("metrics", {}),
                error_message=result_data.get("error_message"),
                execution_time=result_data.get("execution_time"),
                error_trace=result_data.get("error_trace"),
            )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Loaded result from cache successfully",
                    },
                )

            return result

        except Exception as e:
            self.logger.info(f"{self.operation_name} - {step} failed: {e}")

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Load result from cache failed - proceeding with execution",
                        "error": str(e),
                    },
                )

            return None

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
            "operation": self.operation_name,
            "version": self.version,
            "field_name": self.field_name,
            "additional_fields": self.additional_fields,
            "suppression_mode": self.suppression_mode,
            "save_suppressed_schema": self.save_suppressed_schema,
            "condition_field": self.condition_field,
            "condition_values": self.condition_values,
            "condition_operator": self.condition_operator,
            "condition_logic": self.condition_logic,
            "multi_conditions": self.multi_conditions,
            "ka_risk_field": self.ka_risk_field,
            "risk_threshold": self.risk_threshold,
            "optimize_memory": self.optimize_memory,
            "adaptive_chunk_size": self.adaptive_chunk_size,
            "generate_visualization": self.generate_visualization,
            "output_format": self.output_format,
            "save_output": self.save_output,
            "use_cache": self.use_cache,
            "force_recalculation": self.force_recalculation,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "dask_partition_size": self.dask_partition_size,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "chunk_size": self.chunk_size,
            "visualization_backend": self.visualization_backend,
            "visualization_theme": self.visualization_theme,
            "visualization_strict": self.visualization_strict,
            "use_encryption": self.use_encryption,
            "encryption_key": str(self.encryption_key) if self.encryption_key else None,
            "encryption_mode": self.encryption_mode,
        }

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
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
                "columns": list(df.columns),
                "shape": df.shape,
                "summary": {},
            }

            for col in df.columns:
                col_data = df[col]
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
            fallback = f"{df.shape}_{list(df.dtypes)}"
            return hashlib.md5(fallback.encode()).hexdigest()

    def _validate_input_parameters(self, df: pd.DataFrame) -> bool:
        missing = [
            f for f in [self.field_name] + self.additional_fields if f not in df.columns
        ]
        if missing:
            self.logger.error(f"Missing fields in DataFrame: {missing}")
            return False
        return True

    def _load_data_and_validate_input_parameters(
        self,
        data_source: DataSource,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        """
        Load input data and validate the required fields.

        This method handles reporting and progress tracking internally.

        Parameters
        ----------
        data_source : DataSource
            Source of input data.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker to update step status.
        reporter : Optional[Any]
            Reporter for audit or status reporting.
        **kwargs : dict
            Additional parameters like dataset_name, fields, etc.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], bool]
            Loaded DataFrame (or None), and validation success flag.
        """
        step = "Load data and validate input parameters"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        dataset_name = kwargs.get("dataset_name", "main")
        settings_operation = load_settings_operation(
            data_source, dataset_name, **kwargs
        )
        df = load_data_operation(data_source, dataset_name, **settings_operation)

        if df is None or df.empty:
            self.logger.error("Error: loaded DataFrame is None or empty")

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="error",
                    details={
                        "step": step,
                        "message": "Data load failed or empty DataFrame",
                    },
                )
            return None, False

        self._original_df = df.copy(deep=True)

        is_valid = self._validate_input_parameters(df)

        if reporter:
            reporter.add_operation(
                f"Operation {self.operation_name}",
                status="info" if is_valid else "warning",
                details={
                    "step": step,
                    "message": (
                        "Validation succeeded" if is_valid else "Validation failed"
                    ),
                    "shape": df.shape if is_valid else None,
                },
            )

        df = self._optimize_data(df)

        return df, is_valid

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

    def __repr__(self) -> str:
        """String representation of the operation."""
        fields = [self.field_name] + self.additional_fields
        return (
            f"AttributeSuppressionOperation("
            f"fields={fields}, "
            f"save_schema={self.save_suppressed_schema})"
        )
