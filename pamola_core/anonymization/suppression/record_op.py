"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Record Suppression Operation Module

License: BSD 3-Clause
Copyright (c) 2025 PAMOLA Development Team

Version: 1.1.0
Last Updated: 2025-06-15

Description:
    This module implements the RecordSuppressionOperation class for removing
    entire rows (records) from datasets as part of the PAMOLA anonymization
    framework. This operation provides flexible methods for protecting privacy
    by removing records based on various conditions.

Key Features:
    - Remove records based on multiple condition types (null, value, range, risk, custom)
    - Support for multi-field conditions with AND/OR logic
    - Save suppressed records for audit purposes with memory management
    - Generate suppression metrics and visualizations
    - Full integration with PAMOLA base framework
    - Batch processing support with progress tracking
    - Dask support for distributed processing

Changelog:
    1.1.0 (2025-06-15):
        - Fixed multi-field condition implementation
        - Added proper _collect_specific_metrics() override
        - Implemented _process_batch_dask() for Dask support
        - Improved memory management for suppressed records
        - Enhanced security for sensitive data logging
        - Added comprehensive metrics collection
    1.0.0 (2025-06-15):
        - Initial implementation based on REQ-REC-001 through REQ-REC-005
"""

from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons import calculate_anonymization_effectiveness
from pamola_core.anonymization.commons.validation_utils import (
    check_field_exists,
    FieldNotFoundError,
    validate_numeric_field,
)
from pamola_core.anonymization.commons.visualization_utils import create_bar_plot
from pamola_core.anonymization.schemas.record_op_core_schema import (
    RecordSuppressionConfig,
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io import (
    load_settings_operation,
    ensure_directory,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_field_utils import create_multi_field_mask
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_registry import register

import dask.dataframe as dd
from functools import partial


@register(version="1.0.0")
class RecordSuppressionOperation(AnonymizationOperation):
    """
    Record Suppression Operation for removing entire rows from datasets.

    Implements REQ-REC-001 through REQ-REC-005 from the PAMOLA.CORE
    Suppression Operations Sub-Specification.
    """

    def __init__(
        self,
        field_name: str,
        suppression_mode: str = "REMOVE",
        suppression_condition: str = "null",
        suppression_values: Optional[List[Any]] = None,
        suppression_range: Optional[Tuple[Any, Any]] = None,
        save_suppressed_records: bool = False,
        suppression_reason_field: str = "_suppression_reason",
        **kwargs,
    ):
        """
        Initialize the Record Suppression Operation.

        Removes entire rows from a dataset based on configurable suppression
        rules: nulls, exact matches, ranges, k-anonymity risk, or complex conditions.

        Parameters
        ----------
        field_name : str
            Primary column to evaluate for suppression.
        suppression_mode : str, optional
            Operation suppression_mode. Must be "REMOVE" for record suppression. Default = "REMOVE".
        suppression_condition : str, optional
            Suppression condition type. One of:
            ["null", "value", "range", "risk", "custom"]. Defaults to "null".
        suppression_values : List[Any], optional
            Values to match if suppression_condition = "value".
        suppression_range : Tuple[Any, Any], optional
            Range bounds if suppression_condition = "range".
        save_suppressed_records : bool, optional
            Save removed records to a separate artifact. Defaults to False.
        suppression_reason_field : str, optional
            Field name for suppression reason in output. Defaults to "_suppression_reason".
        **kwargs
            Additional keyword arguments passed to AnonymizationOperation.
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Record suppresstion for '{field_name}' using {suppression_mode} suppression mode",
        )

        # Validate mode
        if suppression_mode != "REMOVE":
            raise ValueError(
                f"RecordSuppressionOperation only supports mode='REMOVE', got '{suppression_mode}'"
            )
        # Validate suppression_condition
        valid_conditions = ["null", "value", "range", "risk", "custom"]
        if suppression_condition not in valid_conditions:
            raise ValueError(
                f"Invalid suppression_condition '{suppression_condition}'. "
                f"Must be one of: {valid_conditions}"
            )

        # Condition-specific validation
        if suppression_condition == "value" and not suppression_values:
            raise ValueError("suppression_values required for 'value' condition")
        if suppression_condition == "range" and not suppression_range:
            raise ValueError("suppression_range required for 'range' condition")

        # --- Build config object ---
        config = RecordSuppressionConfig(
            field_name=field_name,
            suppression_mode=suppression_mode,
            suppression_condition=suppression_condition,
            suppression_values=suppression_values,
            suppression_range=suppression_range,
            save_suppressed_records=save_suppressed_records,
            suppression_reason_field=suppression_reason_field,
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
        self._original_record_count = 0
        self._suppressed_records_count = 0
        self._suppression_reasons: Dict[str, int] = {}
        self._batch_number = 0
        self._writer: Optional[DataWriter] = None
        self._task_dir: Optional[Path] = None

        # --- Metadata ---
        self.operation_name = self.__class__.__name__

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

            try:
                step = "Load data and validate input parameters"
                if progress_tracker:
                    progress_tracker.update(
                        1, {"step": step, "operation": self.operation_name}
                    )
                # Validate configuration early
                dataset_name = kwargs.get("dataset_name", "main")
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
                is_valid = self._validate_input_parameters(df)
                if not is_valid:
                    raise ValueError("Missing fields in DataFrame.")
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Store original data for caching
            original_data = df[self.field_name].copy(deep=True)

            # Handle cache if required
            if self.use_cache and not self.force_recalculation:
                try:
                    if progress_tracker:
                        progress_tracker.update(
                            1,
                            {
                                "step": "Load result from cache",
                                "operation": self.operation_name,
                            },
                        )

                    self.logger.info(
                        f"Operation: {self.operation_name}, Load result from cache"
                    )

                    cached_result = self._check_cache(df, reporter)

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

                metrics = self._collect_metrics(df, output_data, mask)

                file_name = f"{self.operation_name}_metrics_{operation_timestamp}"

                self._save_metrics(
                    metrics=metrics,
                    writer=self._writer,
                    result=result,
                    reporter=reporter,
                    progress_tracker=progress_tracker,
                    operation_timestamp=operation_timestamp,
                    file_name=file_name,
                )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Save output if required
            if self.save_output:
                try:
                    self.logger.info(f"Operation: {self.operation_name}, Save output")
                    file_name = f"{self.operation_name}_{self.field_name}_output_{operation_timestamp}"
                    self._save_output_data(
                        result_df=output_data,
                        writer=self._writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=progress_tracker,
                        timestamp=operation_timestamp,
                        file_name=file_name,
                        **kwargs,
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
                    self._save_to_cache(
                        original_data=original_data,
                        anonymized_data=output_data,
                        result=result,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=output_data,
                original_data=original_data,
                anonymized_data=None,
            )

            # Finalize timing
            self.end_time = time.time()

            # Set success status
            result.status = OperationStatus.SUCCESS
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
            cache_dir=dirs["cache"],
        )

        # Save configuration to task directory
        self.save_config(task_dir)

        self._task_dir = task_dir

        # Create writer for consistent output handling
        self._writer = DataWriter(
            task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
        )

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

    def _process_data(
        self,
        input_data: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Select the appropriate processing method (Pandas, Dask, or Joblib) based on configuration.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be processed.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker instance.
        reporter : Optional[Any]
            Reporting/logging object.

        Returns
        -------
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            The processed DataFrame, or a dictionary of DataFrames if applicable.
        """
        if self.use_dask and self.npartitions > 1:
            self.logger.info("Parallel Enabled")
            self.logger.info("Parallel Engine: Dask")
            self.logger.info(f"Parallel Workers: {self.npartitions}")
            return self._process_with_dask(
                input_data, progress_tracker=progress_tracker, reporter=reporter
            )

        elif self.use_vectorization and self.parallel_processes > 1:
            self.logger.info("Parallel Enabled")
            self.logger.info("Parallel Engine: Joblib")
            self.logger.info(f"Parallel Workers: {self.parallel_processes}")
            self.logger.info(
                f"Using vectorized processing with chunk size {self.chunk_size}"
            )
            return self._process_with_joblib(
                input_data, progress_tracker=progress_tracker, reporter=reporter
            )

        else:
            self.logger.info("Parallel Disabled")
            self.logger.info(
                f"Using Pandas processing with chunk size {self.chunk_size}"
            )
            return self._process_with_pandas(
                input_data, progress_tracker=progress_tracker, reporter=reporter
            )

    def _process_with_pandas(
        self,
        input_data: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Process the entire DataFrame in chunks using pandas.

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            - mask: A global boolean mask indicating suppressed records.
            - output_data: The resulting DataFrame after suppression.
        """
        step = "Process data with pandas"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            total_rows = len(input_data)
            chunk_size = max(1, min(self.chunk_size or total_rows, total_rows))
            processed_chunks = []
            suppressed_chunks = []
            global_mask = pd.Series(False, index=input_data.index)

            self._original_record_count = total_rows
            self._batch_number = 0
            self._suppressed_records_count = 0
            self._suppression_reasons = {}

            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                batch = input_data.iloc[start:end]

                # Build suppression mask for the batch
                mask = self._build_suppression_mask(batch)
                suppressed_count = mask.sum()

                if suppressed_count > 0:
                    self._suppressed_records_count += suppressed_count
                    self._suppression_reasons[self.suppression_condition] = (
                        self._suppression_reasons.get(self.suppression_condition, 0)
                        + suppressed_count
                    )

                    if self.save_suppressed_records:
                        suppressed_chunks.append(batch[mask].copy(deep=True))

                # Mark suppressed positions in the global mask
                global_mask.iloc[start:end] = mask.values

                # Keep non-suppressed rows
                result_batch = batch[~mask].copy(deep=True)
                processed_chunks.append(result_batch)
                self._batch_number += 1

                self.logger.debug(
                    f"Chunk {self._batch_number}: Suppressed {suppressed_count}/{len(batch)} records"
                )

            output_data = pd.concat(processed_chunks, axis=0).reset_index(drop=True)

            if self.save_suppressed_records and suppressed_chunks:
                suppressed_df = pd.concat(suppressed_chunks, axis=0).reset_index(
                    drop=True
                )
                if not suppressed_df.empty:
                    self._save_suppressed_records(
                        suppressed_df, record_num=self._suppressed_records_count
                    )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Processed with pandas",
                        "chunks": self._batch_number,
                        "total_rows": total_rows,
                        "rows_suppressed": self._suppressed_records_count,
                        "final_rows": len(output_data),
                    },
                )

            return global_mask, output_data

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

    def _process_with_dask(
        self,
        input_data: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Process the entire DataFrame using Dask for parallel batch suppression.

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            - mask: Boolean Series indicating rows to be suppressed
            - output_data: DataFrame after removing suppressed rows
        """
        step = "Process data with Dask"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(input_data, npartitions=self.npartitions)

            # Dask-friendly masking function
            mask_func = partial(
                build_suppression_mask_for_dask,
                suppression_condition=self.suppression_condition,
                field_name=self.field_name,
                suppression_values=self.suppression_values,
                suppression_range=self.suppression_range,
                ka_risk_field=self.ka_risk_field,
                risk_threshold=self.risk_threshold,
                multi_conditions=self.multi_conditions,
                condition_logic=self.condition_logic,
            )

            # Apply map_partitions with externalized masking function
            dask_mask_ddf = ddf.map_partitions(mask_func, meta=pd.Series(dtype=bool))
            mask = dask_mask_ddf.compute()

            # Apply mask to get result
            output_data = input_data[~mask].copy(deep=True)

            # Track suppression count
            self._original_record_count = len(input_data)
            self._suppressed_records_count = mask.sum()
            self._suppression_reasons[self.suppression_condition] = (
                self._suppression_reasons.get(self.suppression_condition, 0)
                + self._suppressed_records_count
            )

            if self.save_suppressed_records and self._suppressed_records_count > 0:
                suppressed_df = input_data[mask].copy(deep=True)
                if not suppressed_df.empty:
                    self._save_suppressed_records(
                        suppressed_df, record_num=self._suppressed_records_count
                    )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": f"Processed with Dask using {self.npartitions} partitions",
                        "total_rows": len(input_data),
                        "rows_suppressed": self._suppressed_records_count,
                        "final_rows": len(output_data),
                    },
                )

            return mask, output_data.reset_index(drop=True)

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
                        "message": "Dask processing failed",
                        "error": str(e),
                    },
                )
            raise

    def _process_with_joblib(
        self,
        input_data: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        from joblib import Parallel, delayed

        step = "Process data with Joblib"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            total_rows = len(input_data)
            chunk_size = max(1, min(self.chunk_size or total_rows, total_rows))
            chunks = [
                input_data.iloc[i : i + chunk_size]
                for i in range(0, total_rows, chunk_size)
            ]

            suppression_config = dict(
                suppression_condition=self.suppression_condition,
                field_name=self.field_name,
                suppression_values=self.suppression_values,
                suppression_range=self.suppression_range,
                ka_risk_field=self.ka_risk_field,
                risk_threshold=self.risk_threshold,
                multi_conditions=self.multi_conditions,
                condition_logic=self.condition_logic,
            )

            results = Parallel(n_jobs=self.parallel_processes)(
                delayed(process_batch_for_suppression)(
                    chunk, suppression_config, self.save_suppressed_records
                )
                for chunk in chunks
            )

            output_chunks, all_masks, suppressed_chunks = zip(*results)

            result_df = pd.concat(output_chunks, axis=0).reset_index(drop=True)
            final_mask = (
                pd.concat(all_masks, axis=0).astype(bool).reset_index(drop=True)
            )

            # Total suppressed records
            self._original_record_count = len(input_data)
            self._suppressed_records_count = final_mask.sum()
            self._suppression_reasons[self.suppression_condition] = (
                self._suppressed_records_count
            )

            # Save suppressed records once
            if self.save_suppressed_records and suppressed_chunks:
                suppressed_df = pd.concat(suppressed_chunks, axis=0).reset_index(
                    drop=True
                )
                if not suppressed_df.empty:
                    self._save_suppressed_records(
                        suppressed_df, record_num=self._suppressed_records_count
                    )

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": f"Processed with Joblib using {self.parallel_processes} processes",
                        "chunks": len(chunks),
                        "rows_suppressed": self._suppressed_records_count,
                        "final_rows": len(result_df),
                    },
                )

            return final_mask, result_df

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
                        "message": "Joblib processing failed",
                        "error": str(e),
                    },
                )
            raise

    def _build_suppression_mask(self, batch: pd.DataFrame) -> pd.Series:
        """
        Build boolean mask for records to suppress.

        Args:
            batch: DataFrame batch

        Returns:
            Boolean Series where True indicates record should be suppressed
        """
        if self.suppression_condition == "null":
            # Validate field exists
            if not check_field_exists(batch, self.field_name):
                raise FieldNotFoundError(self.field_name, list(batch.columns))
            mask = batch[self.field_name].isna()

        elif self.suppression_condition == "value":
            if not check_field_exists(batch, self.field_name):
                raise FieldNotFoundError(self.field_name, list(batch.columns))
            mask = batch[self.field_name].isin(self.suppression_values)

        elif self.suppression_condition == "range":
            if not check_field_exists(batch, self.field_name):
                raise FieldNotFoundError(self.field_name, list(batch.columns))

            # Validate numeric field for range comparison
            if not validate_numeric_field(batch, self.field_name, allow_null=True):
                raise ValueError(
                    f"Field '{self.field_name}' must be numeric for range condition"
                )

            min_val, max_val = self.suppression_range
            mask = batch[self.field_name].between(min_val, max_val, inclusive="both")

        elif self.suppression_condition == "risk":
            # Use ka_risk_field instead of field_name for risk assessment
            if not check_field_exists(batch, self.ka_risk_field):
                raise FieldNotFoundError(self.ka_risk_field, list(batch.columns))
            mask = batch[self.ka_risk_field] < self.risk_threshold

        elif self.suppression_condition == "custom":
            # Use framework utility for multi-field conditions
            mask = create_multi_field_mask(
                batch, self.multi_conditions, self.condition_logic
            )

        else:
            # Should not reach here due to validation, but just in case
            mask = pd.Series(False, index=batch.index)

        return mask

    def _save_suppressed_records(
        self, suppressed_df: pd.DataFrame, record_num: int
    ) -> None:
        """
        Save suppressed records to disk immediately to manage memory.

        Args:
            suppressed_df: DataFrame of suppressed records
            record_num: Record number for file naming (count of suppressed records or identifier)
        """
        if suppressed_df.empty:
            return

        try:
            suppressed_df = suppressed_df.copy(deep=True)
            self.suppression_reason_field = (
                self.suppression_reason_field or "ReasonForRemoval"
            )
            suppressed_df[self.suppression_reason_field] = (
                self._get_suppression_reason()
            )

            if self._writer and self._task_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    f"{self.field_name}_suppressed_records_{record_num:04d}_{timestamp}"
                )
                suppressed_record_result = self._writer.write_dataframe(
                    df=suppressed_df,
                    name=filename,
                    format="csv",
                    subdir="output/suppressed_records",
                    timestamp_in_name=False,
                    encryption_key=self.encryption_key if self.use_encryption else None,
                )
                self.logger.debug(
                    f"Saved {len(suppressed_df)} suppressed records (record_num={record_num}) "
                    f"to file {suppressed_record_result.path}"
                )
            else:
                self.logger.warning(
                    "DataWriter not available, suppressed records not saved"
                )

        except Exception as e:
            self.logger.error(f"Failed to save suppressed records: {e}", exc_info=True)

    def _get_suppression_reason(self) -> str:
        """
        Get human-readable suppression reason.

        Returns:
            String describing the suppression reason
        """
        if self.suppression_condition == "null":
            return f"Null value in field '{self.field_name}'"
        elif self.suppression_condition == "value":
            return f"Value in '{self.field_name}' matched suppression list"
        elif self.suppression_condition == "range":
            return f"Value in '{self.field_name}' within range {self.suppression_range}"
        elif self.suppression_condition == "risk":
            return f"K-anonymity risk below threshold ({self.risk_threshold})"
        elif self.suppression_condition == "custom":
            return f"Multi-field condition ({self.condition_logic})"
        else:
            return "Unknown reason"

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect record suppression specific metrics.

        Note: For record suppression, we work with accumulated counts,
        not series data.

        Args:
            original_data: Not used for record suppression
            anonymized_data: Not used for record suppression

        Returns:
            Dictionary of suppression metrics
        """
        suppression_rate = 0.0
        if self._original_record_count > 0:
            suppression_rate = (
                self._suppressed_records_count / self._original_record_count
            ) * 100

        remaining_records = self._original_record_count - self._suppressed_records_count

        metrics = {
            "operation_type": "record_suppression",
            "suppression_condition": self.suppression_condition,
            "records_suppressed": self._suppressed_records_count,
            "suppression_rate": suppression_rate,
            "remaining_records": remaining_records,
            "suppression_by_condition": dict(self._suppression_reasons),
        }

        # Add condition-specific metrics
        if self.suppression_condition == "value":
            metrics["suppression_values_count"] = len(self.suppression_values)
        elif self.suppression_condition == "range":
            metrics["suppression_range"] = str(self.suppression_range)
        elif self.suppression_condition == "risk":
            metrics["risk_threshold"] = self.risk_threshold
            metrics["ka_risk_field"] = self.ka_risk_field
        elif self.suppression_condition == "custom":
            metrics["multi_conditions_count"] = len(self.multi_conditions)
            metrics["condition_logic"] = self.condition_logic

        # Log final metrics
        self.logger.info(
            f"Record suppression completed: {self._suppressed_records_count}/{self._original_record_count} "
            f"records suppressed ({suppression_rate:.1f}%)"
        )

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

            # Record suppression specific metrics
            specific_metrics = self._collect_specific_metrics(
                original_data=(
                    input_data[self.field_name]
                    if self.field_name in input_data.columns
                    else pd.Series([])
                ),
                anonymized_data=(
                    output_data[self.field_name]
                    if self.field_name in output_data.columns
                    else pd.Series([])
                ),
            )
            metrics.update(specific_metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error in _collect_metrics: {e}", exc_info=True)
            raise

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
            operation_timestamp: Timestamp string for filenames

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

        try:
            # Calculate actual counts
            remaining_count = (
                self._original_record_count - self._suppressed_records_count
            )

            # Prepare data for visualization
            categories = ["Before", "After"]
            values = [self._original_record_count, remaining_count]

            # Create DataFrame for visualization
            viz_data = pd.DataFrame({"Status": categories, "Record Count": values})

            # Create bar chart
            bar_file_name = (
                f"{operation_name}_record_suppression_summary_{operation_timestamp}.png"
            )
            bar_path = vis_dir / bar_file_name
            bar_result = create_bar_plot(
                data=viz_data.set_index("Status")["Record Count"].to_dict(),
                output_path=bar_path,
                title=f"Record Suppression: {self._suppressed_records_count} Records Removed",
                x_label="Status",
                y_label="Record Count",
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

        # Create suppression breakdown visualization if multiple reasons
        if len(self._suppression_reasons) > 1:
            try:
                breakdown_bar_file_name = f"{operation_name}_suppression_reasons_breakdown_{operation_timestamp}.png"
                breakdown_bar_path = vis_dir / breakdown_bar_file_name
                breakdown_bar_result = create_bar_plot(
                    data=self._suppression_reasons,
                    output_path=breakdown_bar_path,
                    title="Records Suppressed by Condition",
                    x_label="Suppression Condition",
                    y_label="Count",
                    orientation="v",
                    **kwargs_visualization,
                )

                if not breakdown_bar_result.startswith("Error"):
                    result.add_artifact(
                        artifact_type="png",
                        path=breakdown_bar_result,
                        description="Bar chart showing data type distribution of suppressed columns",
                        category=Constants.Artifact_Category_Visualization,
                    )
                else:
                    self.logger.error(
                        f"Failed to create dtype distribution visualization: {breakdown_bar_result}"
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
            Timestamp string for filenames (not used here but kept for consistency).
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
            "save_suppressed_records": self.save_suppressed_records,
            "suppression_condition": self.suppression_condition,
            "suppression_values": self.suppression_values,
            "suppression_range": self.suppression_range,
            "suppression_reason_field": self.suppression_reason_field,
        }

    def _validate_input_parameters(self, df: pd.DataFrame) -> bool:
        """
        Validate required input fields exist in the DataFrame based on suppression condition.

        Returns
        -------
        bool
            True if validation passes, False otherwise.
        """
        required_fields = {self.field_name}

        # Add fields based on condition
        if self.suppression_condition == "risk" and self.ka_risk_field:
            required_fields.add(self.ka_risk_field)

        if self.suppression_condition == "custom" and self.multi_conditions:
            for cond in self.multi_conditions:
                if "field" in cond:
                    required_fields.add(cond["field"])

        # Validate existence
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            self.logger.error(f"Missing required fields in DataFrame: {missing}")
            return False

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

    def __repr__(self) -> str:
        """String representation of the operation."""
        return (
            f"RecordSuppressionOperation("
            f"field='{self.field_name}', "
            f"condition='{self.suppression_condition}', "
            f"save_records={self.save_suppressed_records})"
        )


def build_suppression_mask_for_dask(
    batch: pd.DataFrame,
    suppression_condition: str,
    field_name: str = None,
    suppression_values: Optional[List[Any]] = None,
    suppression_range: Optional[Tuple[float, float]] = None,
    ka_risk_field: Optional[str] = None,
    risk_threshold: Optional[float] = None,
    multi_conditions: Optional[List[Dict]] = None,
    condition_logic: Optional[str] = None,
) -> pd.Series:
    if suppression_condition == "null":
        if not check_field_exists(batch, field_name):
            raise FieldNotFoundError(field_name, list(batch.columns))
        return batch[field_name].isna()

    elif suppression_condition == "value":
        if not check_field_exists(batch, field_name):
            raise FieldNotFoundError(field_name, list(batch.columns))
        return batch[field_name].isin(suppression_values)

    elif suppression_condition == "range":
        if not check_field_exists(batch, field_name):
            raise FieldNotFoundError(field_name, list(batch.columns))
        if not validate_numeric_field(batch, field_name, allow_null=True):
            raise ValueError(
                f"Field '{field_name}' must be numeric for range condition"
            )
        return batch[field_name].between(
            suppression_range[0], suppression_range[1], inclusive="both"
        )

    elif suppression_condition == "risk":
        if not check_field_exists(batch, ka_risk_field):
            raise FieldNotFoundError(ka_risk_field, list(batch.columns))
        return batch[ka_risk_field] < risk_threshold

    elif suppression_condition == "custom":
        return create_multi_field_mask(batch, multi_conditions, condition_logic)

    return pd.Series(False, index=batch.index)


def build_suppression_mask_for_joblib(
    batch: pd.DataFrame,
    suppression_condition: str,
    field_name: Optional[str],
    suppression_values: Optional[List[Any]],
    suppression_range: Optional[Tuple[float, float]],
    ka_risk_field: Optional[str],
    risk_threshold: Optional[float],
    multi_conditions: Optional[List[Dict[str, Any]]],
    condition_logic: Optional[str],
) -> pd.Series:
    if suppression_condition == "null":
        mask = batch[field_name].isna()
    elif suppression_condition == "value":
        mask = batch[field_name].isin(suppression_values)
    elif suppression_condition == "range":
        mask = batch[field_name].between(*suppression_range, inclusive="both")
    elif suppression_condition == "risk":
        mask = batch[ka_risk_field] < risk_threshold
    elif suppression_condition == "custom":
        mask = create_multi_field_mask(batch, multi_conditions, condition_logic)
    else:
        mask = pd.Series(False, index=batch.index)

    return mask


def process_batch_for_suppression(
    batch: pd.DataFrame,
    suppression_config: Dict[str, Any],
    save_suppressed_records: bool,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    mask = build_suppression_mask_for_joblib(batch, **suppression_config)
    suppressed = batch[mask] if save_suppressed_records else None
    result = batch[~mask].copy(deep=True)
    return result, mask.astype(bool), suppressed
