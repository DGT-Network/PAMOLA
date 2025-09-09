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
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons import calculate_anonymization_effectiveness
from pamola_core.anonymization.commons.validation_utils import (
    check_field_exists,
    FieldNotFoundError,
    validate_numeric_field,
)
from pamola_core.anonymization.commons.visualization_utils import create_bar_plot
from pamola_core.common.constants import Constants
from pamola_core.utils.io import (
    load_settings_operation,
    load_data_operation,
    get_timestamped_filename,
    ensure_directory,
    write_dataframe_to_csv,
    write_json,
)
from pamola_core.utils.io_helpers import crypto_utils, directory_utils
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_field_utils import create_multi_field_mask
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
    OperationArtifact,
)
from pamola_core.utils.progress import ProgressTracker, HierarchicalProgressTracker
from pamola_core.utils.ops.op_registry import register_operation

import dask.dataframe as dd
from functools import partial


class RecordSuppressionOperation(AnonymizationOperation):
    """
    Record Suppression Operation for removing entire rows from datasets.

    This operation implements REQ-REC-001 through REQ-REC-005 from the
    PAMOLA.CORE Suppression Operations Sub-Specification.

    Attributes:
        field_name (str): Primary field to check for suppression
        suppression_condition (str): Type of suppression condition
        suppression_values (List[Any]): Values to match for value condition
        suppression_range (Tuple[Any, Any]): Range bounds for range condition
        save_suppressed_records (bool): Whether to save removed records
        suppression_reason_field (str): Field name for suppression reason
        multi_conditions (List[Dict], optional): List of conditions for complex filtering
        condition_logic (str): Logic for combining conditions (AND/OR)
        _original_record_count (int): Number of records before suppression
        _suppressed_records_count (int): Number of records suppressed
        _suppression_reasons (Dict[str, int]): Breakdown by suppression reason
        _batch_number (int): Current batch number for file writing
    """

    def __init__(
        self,
        field_name: str,
        mode: str = "REMOVE",
        save_suppressed_records: bool = False,
        suppression_condition: str = "null",
        suppression_values: Optional[List[Any]] = None,
        suppression_range: Optional[Tuple[Any, Any]] = None,
        suppression_reason_field: str = "_suppression_reason",
        condition_logic: str = "OR",
        multi_conditions: Optional[List[Dict[str, Any]]] = None,
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        output_format: str = "csv",
        save_output: bool = True,
        generate_visualization: bool = True,
        use_cache: bool = True,
        force_recalculation: bool = False,
        use_dask: bool = False,
        npartitions: int = 1,
        dask_partition_size: Optional[str] = None,
        use_vectorization: bool = False,
        parallel_processes: int = 1,
        chunk_size: int = 10000,
        visualization_backend: Optional[str] = "plotly",
        visualization_theme: Optional[str] = None,
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        encryption_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Record Suppression Operation.

        This operation removes entire records (rows) from a dataset based on configurable suppression conditions,
        such as null values, exact value matches, value ranges, risk scores, or user-defined rules.

        Parameters
        ----------
        field_name : str
            Primary column to evaluate for suppression.
        mode : str, optional
            Operation mode, must be "REMOVE". Defaults to "REMOVE".
        save_suppressed_records : bool, optional
            Whether to save the removed records into a separate artifact. Defaults to False.
        suppression_condition : str, optional
            Suppression condition type. One of: ["null", "value", "range", "risk", "custom"]. Defaults to "null".
                - "null": Suppress rows where field is null.
                - "value": Suppress rows where field matches values in `suppression_values`.
                - "range": Suppress rows where field is within `suppression_range`.
                - "risk": Suppress rows where `ka_risk_field` < `risk_threshold`.
                - "custom": Use `multi_conditions` for complex rules.
        suppression_values : List[Any], optional
            Required if `suppression_condition` is "value". List of values to match for suppression.
        suppression_range : Tuple[Any, Any], optional
            Required if `suppression_condition` is "range". A tuple (min, max) representing the value range.
        suppression_reason_field : str, optional
            Field name to annotate suppression reason in output, if enabled. Defaults to "_suppression_reason".
        condition_logic : str, optional
            Logic to combine multiple conditions: "AND" or "OR". Only applies when using `multi_conditions`. Defaults to "OR".
        multi_conditions : List[Dict[str, Any]], optional
            Required if `suppression_condition` is "custom". List of structured rules for advanced suppression logic.
        ka_risk_field : str, optional
            Required if `suppression_condition` is "risk". Field name containing k-anonymity risk scores.
        risk_threshold : float, optional
            Suppress records if k < threshold when using "risk" suppression. Defaults to 5.0.

        output_format : str, optional
            Output format: "csv", "parquet", "json", etc. Defaults to "csv".
        save_output : bool, optional
            Whether to persist the output dataset to disk. Defaults to True.
        generate_visualization : bool, optional
            Whether to generate visualizations summarizing the suppression results. Defaults to True.
        use_cache : bool, optional
            Enable internal caching of intermediate and final results. Defaults to True.
        force_recalculation : bool, optional
            Ignore cache and recompute even if cached results exist. Defaults to False.

        use_dask : bool, optional
            Use Dask for out-of-core or parallel processing of large datasets. Defaults to False.
        npartitions : int, optional
            Number of partitions to split the data into when using Dask. Defaults to 1.
        dask_partition_size : str, optional
            Dask-specific memory-based partition size, e.g., "100MB".
        use_vectorization : bool, optional
            Use joblib-based multiprocessing instead of Dask. Defaults to False.
        parallel_processes : int, optional
            Number of parallel processes to use with joblib. Only effective if `use_vectorization` is True.
        chunk_size : int, optional
            Number of records to process per chunk. Used for memory optimization. Defaults to 10000.

        visualization_backend : str, optional
            Backend for visualizations. One of: ["matplotlib", "plotly", "seaborn", ...].
        visualization_theme : str, optional
            Visual style/theme, e.g., "light" or "dark".
        visualization_strict : bool, optional
            Raise exceptions on visualization errors if True. Defaults to False.
        visualization_timeout : int, optional
            Timeout (in seconds) for visualization thread to complete. Defaults to 120.

        use_encryption : bool, optional
            Whether to enable encryption for saved output. Defaults to False.
        encryption_key : str or Path, optional
            Encryption key string or path to key file (if `use_encryption` is True).
        encryption_mode : str, optional
            Encryption mode (e.g., "AES"). Required if `use_encryption` is True.

        **kwargs : dict
            Additional keyword arguments passed to the base operation class.

        Raises
        ------
        ValueError
            - If `mode` is not "REMOVE".
            - If `suppression_condition` is not one of the supported types.
            - If required parameters are missing depending on `suppression_condition`:
                * "value" requires `suppression_values`
                * "range" requires `suppression_range`
                * "risk" requires `ka_risk_field`
                * "custom" requires `multi_conditions`
        """

        # Validate mode
        if mode != "REMOVE":
            raise ValueError(
                f"RecordSuppressionOperation only supports mode='REMOVE', got '{mode}'"
            )

        # Validate suppression condition
        valid_conditions = ["null", "value", "range", "risk", "custom"]
        if suppression_condition not in valid_conditions:
            raise ValueError(
                f"Invalid suppression_condition '{suppression_condition}'. "
                f"Must be one of: {valid_conditions}"
            )

        # Validate condition-specific parameters
        if suppression_condition == "value" and not suppression_values:
            raise ValueError("suppression_values required for 'value' condition")
        if suppression_condition == "range" and not suppression_range:
            raise ValueError("suppression_range required for 'range' condition")
        if suppression_condition == "risk" and not ka_risk_field:
            raise ValueError("ka_risk_field required for 'risk' condition")
        if suppression_condition == "custom" and not multi_conditions:
            raise ValueError("multi_conditions required for 'custom' condition")

        # Initialize parent class
        super().__init__(
            field_name=field_name,
            mode=mode,
            multi_conditions=multi_conditions,
            condition_logic=condition_logic,
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            chunk_size=chunk_size,
            output_format=output_format,
            use_cache=use_cache,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            visualization_backend=visualization_backend,
            visualization_theme=visualization_theme,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
            **kwargs,
        )

        # Store operation-specific parameters
        self.suppression_condition = suppression_condition
        self.suppression_values = suppression_values or []
        self.suppression_range = suppression_range
        self.save_suppressed_records = save_suppressed_records
        self.suppression_reason_field = suppression_reason_field
        self.multi_conditions = multi_conditions or []
        self.condition_logic = condition_logic.upper()

        self.save_output = save_output
        self.generate_visualization = generate_visualization
        self.force_recalculation = force_recalculation

        # Initialize internal state
        self._original_record_count = 0
        self._suppressed_records_count = 0
        self._suppression_reasons: Dict[str, int] = {}
        self._batch_number = 0
        self._writer: Optional[DataWriter] = None
        self._task_dir: Optional[Path] = None

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any = None,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the record suppression operation.

        This method performs the full pipeline for removing rows from the dataset
        based on a combination of configured suppression conditions such as:
        null checks, value matches, range thresholds, k-anonymity risks, or
        multi-field conditions.

        Steps include:
        - Preparing output directories and environment.
        - Loading and validating the input dataset.
        - Checking and returning cached results if available and permitted.
        - Processing data in batches or partitions to suppress records.
        - Collecting and saving suppression metrics.
        - Saving the final output dataset and visualizations.
        - Caching results if enabled.

        This method overrides the base `execute` method to customize logic specific
        to record-level suppression, including tracking suppressed record counts
        and optionally saving removed rows.

        Parameters
        ----------
        data_source : DataSource
            Source of input data for the suppression operation.
        task_dir : Path
            Directory where output files, visualizations, and metrics will be saved.
        reporter : Any, optional
            Reporter object for logging and reporting operation progress and results.
        progress_tracker : Optional[HierarchicalProgressTracker], optional
            Tracker to update real-time progress feedback.
        **kwargs : dict
            Additional parameters that may influence suppression behavior (e.g. logger, settings).

        Returns
        -------
        OperationResult
            Structured result object including status, metrics, artifacts, and optional errors.
        """
        try:
            # Initialize operation
            self.start_time = time.time()
            operation_name = self.__class__.__name__
            self.logger = kwargs.get("logger", self.logger)

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
                f"Operation: {operation_name}, Start operation - Preparation"
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
                f"Operation: {operation_name}, Load data and validate input parameters"
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
                        f"Operation: {operation_name}, Load result from cache"
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
                self.logger.info(f"Operation: {operation_name}, Process data")
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
                self.logger.info(f"Operation: {operation_name}, Collect metric")
                self._handle_metrics(
                    input_data=df,
                    output_data=output_data,
                    mask=mask,
                    result=result,
                    task_dir=task_dir,
                    progress_tracker=progress_tracker,
                    reporter=reporter,
                )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Save output if required
            if self.save_output:
                try:
                    self.logger.info(f"Operation: {operation_name}, Save output")
                    self._save_output(
                        output_data=output_data,
                        task_dir=task_dir,
                        result=result,
                        progress_tracker=progress_tracker,
                        reporter=reporter,
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
                        f"Operation: {operation_name}, Generate visualizations"
                    )
                    self._handle_visualizations(
                        input_data=df,
                        output_data=output_data,
                        task_dir=task_dir,
                        result=result,
                        progress_tracker=progress_tracker,
                        reporter=reporter,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical

            # Save cache if required
            if self.use_cache:
                try:
                    self.logger.info(f"Operation: {operation_name}, Save cache")
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
            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )

            return result

        except Exception as e:
            self.logger.error(f"Operation: {operation_name}, error occurred: {e}")
            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
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
        operation_name = self.__class__.__name__
        step = "Preparation"

        # Setup total progress steps
        if progress_tracker:
            progress_tracker.total = self._compute_total_steps(**kwargs)
            progress_tracker.update(1, {"step": step, "operation": operation_name})

        # Prepare necessary directories
        dirs = self._prepare_directories(task_dir)

        # Initialize operation cache
        self.operation_cache = OperationCache(
            cache_dir=task_dir / "cache",
        )

        self._task_dir = task_dir

        # Create writer for consistent output handling
        self._writer = DataWriter(
            task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
        )

        # Report preparation success
        if reporter:
            reporter.add_operation(
                f"Operation {operation_name}",
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
        operation_name = self.__class__.__name__
        step = "Process data with pandas"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": operation_name})

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
                        suppressed_chunks.append(batch[mask].copy())

                # Mark suppressed positions in the global mask
                global_mask.iloc[start:end] = mask.values

                # Keep non-suppressed rows
                result_batch = batch[~mask].copy()
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
                    f"Operation {operation_name}",
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
            self.logger.error(f"{operation_name} - {step} failed: {e}", exc_info=True)
            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
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
        operation_name = self.__class__.__name__
        step = "Process data with Dask"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": operation_name})

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
            output_data = input_data[~mask].copy()

            # Track suppression count
            self._original_record_count = len(input_data)
            self._suppressed_records_count = mask.sum()
            self._suppression_reasons[self.suppression_condition] = (
                self._suppression_reasons.get(self.suppression_condition, 0)
                + self._suppressed_records_count
            )

            if self.save_suppressed_records and self._suppressed_records_count > 0:
                suppressed_df = input_data[mask].copy()
                if not suppressed_df.empty:
                    self._save_suppressed_records(
                        suppressed_df, record_num=self._suppressed_records_count
                    )

            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
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
            self.logger.error(f"{operation_name} - {step} failed: {e}", exc_info=True)
            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
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

        operation_name = self.__class__.__name__
        step = "Process data with Joblib"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": operation_name})

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
                    f"Operation {operation_name}",
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
            self.logger.error(f"{operation_name} - {step} failed: {e}", exc_info=True)
            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
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
            suppressed_df = suppressed_df.copy()
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

    def _save_metrics(
        self, metrics: Dict[str, Any], task_dir: Path, result: OperationResult
    ) -> None:
        """
        Save the collected metrics as a JSON file and register it as an artifact.

        Raises:
            Exception: If saving the metrics file fails.
        """
        try:
            metrics_dir = task_dir / "metrics"
            ensure_directory(metrics_dir)

            operation_name = self.__class__.__name__
            filename = get_timestamped_filename(
                f"{operation_name}__metrics", "json", True
            )
            metrics_path = metrics_dir / filename

            write_json(metrics, metrics_path, encryption_key=self.encryption_key)

            result.add_artifact(
                artifact_type="json",
                path=metrics_path,
                description=f"Metrics for {operation_name}",
                category=Constants.Artifact_Category_Metrics,
            )

            self.logger.info(f"Structured metrics saved to {metrics_path}")

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
    ) -> Optional[Dict[str, Any]]:
        """
        Handle the collection and saving of metrics for record suppression.

        Parameters
        ----------
        (Same as before)

        Returns
        -------
        Optional[Dict[str, Any]]
        """
        operation_name = self.__class__.__name__
        step = "Collect metrics"
        try:
            if progress_tracker:
                progress_tracker.update(1, {"step": step, "operation": operation_name})

            metrics = self._collect_metrics(input_data, output_data, mask)
            result.metrics = metrics
            self._save_metrics(metrics, task_dir, result)

            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Metrics collected and saved successfully",
                        "summary": {
                            "operation_type": metrics.get("operation_type"),
                            "records_suppressed": metrics.get("records_suppressed"),
                            "suppression_rate": round(
                                metrics.get("suppression_rate", 0), 2
                            ),
                            "remaining_records": metrics.get("remaining_records"),
                            "records_processed": metrics.get("records_processed"),
                            "duration_seconds": metrics.get("duration_seconds"),
                            "records_per_second": metrics.get("records_per_second"),
                        },
                    },
                )

            return metrics

        except Exception as e:
            self.logger.error(
                f"Failed to handle metrics in {operation_name}: {e}", exc_info=True
            )
            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
                    status="error",
                    details={"step": step, "message": str(e)},
                )
            return None

    def _save_output(
        self,
        output_data: pd.DataFrame,
        task_dir: Path,
        result: OperationResult,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        reporter: Optional[Any] = None,
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
        **kwargs : dict
            Additional keyword arguments, including encryption settings.
        """
        operation_name = self.__class__.__name__
        step = "Save output"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": operation_name})

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

        filename = get_timestamped_filename(
            f"{operation_name}_{self.field_name}", self.output_format, True
        )
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
                        f"Operation {operation_name}",
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
                    f"Operation {operation_name}",
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
                    f"Operation {operation_name}",
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
    ) -> Optional[Path]:
        """
        Generate visualizations showing columns before/after suppression and data type distribution.

        Args:
            input_data: Original DataFrame
            output_data: Processed DataFrame
            task_dir: Directory to save visualization

        Returns:
            Path to saved visualization or None
        """

        vis_dir = task_dir / "visualizations"
        ensure_directory(vis_dir)
        operation_name = self.__class__.__name__

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
            bar_path = vis_dir / get_timestamped_filename(
                f"{operation_name}_record_suppression_summary", "png", True
            )
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
                breakdown_bar_path = vis_dir / get_timestamped_filename(
                    f"{operation_name}_suppression_reasons_breakdown", "png", True
                )
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
        """
        import threading
        import contextvars

        operation_name = self.__class__.__name__
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
                )
            except Exception as e:
                viz_error = e
                self.logger.error(
                    f"[VIZ] Visualization error: {type(e).__name__}: {e}", exc_info=True
                )

        try:
            if progress_tracker:
                progress_tracker.update(1, {"step": step, "operation": operation_name})

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
                        f"Operation {operation_name}",
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
                        f"Operation {operation_name}",
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
                        f"Operation {operation_name}",
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
                    f"Operation {operation_name}",
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
        operation_name = self.__class__.__name__

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
                operation_name=operation_name,
                parameters=self._get_cache_parameters(),
                data_hash=self._generate_data_hash(self._original_df.copy()),
            )

            self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            self.logger.info(f"Saved result to cache with key: {cache_key}")

            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Save cache", "operation": operation_name}
                )

            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
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
        operation_name = self.__class__.__name__
        step_name = "Load result from cache"

        if progress_tracker:
            progress_tracker.update(1, {"step": step_name, "operation": operation_name})

        try:
            cache_key = self.operation_cache.generate_cache_key(
                operation_name=operation_name,
                parameters=self._get_cache_parameters(),
                data_hash=self._generate_data_hash(df),
            )

            cached = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=operation_name
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
                    f"Operation {operation_name}",
                    status="info",
                    details={
                        "step": step_name,
                        "message": "Loaded result from cache successfully",
                    },
                )

            return result

        except Exception as e:
            self.logger.info(f"{operation_name} - {step_name} failed: {e}")

            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
                    status="info",
                    details={
                        "step": step_name,
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
            "operation": self.__class__.__name__,
            "version": self.version,
            "field_name": self.field_name,
            "mode": self.mode,
            "save_suppressed_records": self.save_suppressed_records,
            "suppression_condition": self.suppression_condition,
            "suppression_values": self.suppression_values,
            "suppression_range": self.suppression_range,
            "suppression_reason_field": self.suppression_reason_field,
            "condition_logic": self.condition_logic,
            "multi_conditions": self.multi_conditions,
            "ka_risk_field": self.multi_conditions,
            "risk_threshold": self.risk_threshold,
            "optimize_memory": self.optimize_memory,
            "adaptive_chunk_size": self.adaptive_chunk_size,
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

    def _set_input_parameters(self, **kwargs):
        """
        Set common configurable operation parameters from keyword arguments.
        """

        self.field_name = kwargs.get("field_name", getattr(self, "field_name", None))
        self.mode = kwargs.get("mode", getattr(self, "mode", "REMOVE"))
        self.save_suppressed_records = kwargs.get(
            "save_suppressed_records", getattr(self, "save_suppressed_records", True)
        )
        self.suppression_condition = kwargs.get(
            "suppression_condition", getattr(self, "suppression_condition", "null")
        )
        self.suppression_values = kwargs.get(
            "suppression_values", getattr(self, "suppression_values", None)
        )
        self.suppression_range = kwargs.get(
            "suppression_range", getattr(self, "suppression_range", None)
        )
        self.suppression_reason_field = kwargs.get(
            "suppression_reason_field", getattr(self, "suppression_reason_field", None)
        )
        self.condition_logic = kwargs.get(
            "condition_logic", getattr(self, "condition_logic", None)
        )
        self.multi_conditions = kwargs.get(
            "multi_conditions", getattr(self, "multi_conditions", None)
        )
        self.ka_risk_field = kwargs.get(
            "ka_risk_field", getattr(self, "ka_risk_field", None)
        )
        self.risk_threshold = kwargs.get(
            "risk_threshold", getattr(self, "risk_threshold", None)
        )
        self.optimize_memory = kwargs.get(
            "optimize_memory", getattr(self, "optimize_memory", True)
        )
        self.adaptive_chunk_size = kwargs.get(
            "adaptive_chunk_size", getattr(self, "adaptive_chunk_size", True)
        )

        self.generate_visualization = kwargs.get(
            "generate_visualization", getattr(self, "generate_visualization", True)
        )
        self.save_output = kwargs.get("save_output", getattr(self, "save_output", True))
        self.output_format = kwargs.get(
            "output_format", getattr(self, "output_format", "csv")
        )

        self.use_cache = kwargs.get("use_cache", getattr(self, "use_cache", True))
        self.force_recalculation = kwargs.get(
            "force_recalculation", getattr(self, "force_recalculation", False)
        )

        self.use_dask = kwargs.get("use_dask", getattr(self, "use_dask", False))
        self.npartitions = kwargs.get("npartitions", getattr(self, "npartitions", 1))
        self.dask_partition_size = kwargs.get(
            "dask_partition_size", getattr(self, "dask_partition_size", None)
        )

        self.use_vectorization = kwargs.get(
            "use_vectorization", getattr(self, "use_vectorization", False)
        )
        self.parallel_processes = kwargs.get(
            "parallel_processes", getattr(self, "parallel_processes", 1)
        )

        self.chunk_size = kwargs.get("chunk_size", getattr(self, "chunk_size", 10000))

        self.visualization_backend = kwargs.get(
            "visualization_backend", getattr(self, "visualization_backend", None)
        )
        self.visualization_theme = kwargs.get(
            "visualization_theme", getattr(self, "visualization_theme", None)
        )
        self.visualization_strict = kwargs.get(
            "visualization_strict", getattr(self, "visualization_strict", False)
        )
        self.visualization_timeout = kwargs.get(
            "visualization_timeout", getattr(self, "visualization_timeout", 120)
        )

        self.use_encryption = kwargs.get(
            "use_encryption", getattr(self, "use_encryption", False)
        )
        self.encryption_key = (
            kwargs.get("encryption_key", getattr(self, "encryption_key", None))
            if self.use_encryption
            else None
        )

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
        operation_name = self.__class__.__name__
        step_label = "Load data and validate input parameters"

        self._set_input_parameters(**kwargs)

        if progress_tracker:
            progress_tracker.update(
                1, {"step": step_label, "operation": operation_name}
            )

        dataset_name = kwargs.get("dataset_name", "main")
        settings_operation = load_settings_operation(
            data_source, dataset_name, **kwargs
        )
        df = load_data_operation(data_source, dataset_name, **settings_operation)

        if df is None or df.empty:
            self.logger.error("Error: loaded DataFrame is None or empty")

            if reporter:
                reporter.add_operation(
                    f"Operation {operation_name}",
                    status="error",
                    details={
                        "step": step_label,
                        "message": "Data load failed or empty DataFrame",
                    },
                )
            return None, False

        self._input_dataset = dataset_name
        self._original_df = df.copy(deep=True)

        is_valid = self._validate_input_parameters(df)

        if reporter:
            reporter.add_operation(
                f"Operation {operation_name}",
                status="info" if is_valid else "warning",
                details={
                    "step": step_label,
                    "message": (
                        "Validation succeeded" if is_valid else "Validation failed"
                    ),
                    "shape": df.shape if is_valid else None,
                },
            )

        df = self._optimize_data(df)

        return df, is_valid

    def _compute_total_steps(self, **kwargs) -> int:
        use_cache = kwargs.get("use_cache", self.use_cache)
        force_recalculation = kwargs.get(
            "force_recalculation", self.force_recalculation
        )
        save_output = kwargs.get("save_output", self.save_output)
        generate_visualization = kwargs.get(
            "generate_visualization", self.generate_visualization
        )

        steps = 0

        steps += 1  # Step 1: Preparation
        steps += 1  # Step 2: Load data and validate input

        if use_cache and not force_recalculation:
            steps += 1  # Step 3: Try to load from cache

        steps += 1  # Step 4: Process data
        steps += 1  # Step 5: Collect metrics

        if save_output:
            steps += 1  # Step 6: Save output

        if generate_visualization:
            steps += 1  # Step 7: Generate visualizations

        if use_cache:
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
    result = batch[~mask].copy()
    return result, mask.astype(bool), suppressed


# Register the operation with the framework
register_operation(RecordSuppressionOperation)
