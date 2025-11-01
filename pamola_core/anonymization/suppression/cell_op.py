"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Cell Suppression Operation Module

License: BSD 3-Clause
Copyright (c) 2025 PAMOLA Development Team

Version: 1.2.0
Last Updated: 2025-06-15

Description:
    This module implements the CellSuppressionOperation class for replacing
    individual cell values in datasets as part of the PAMOLA anonymization
    framework. Unlike attribute or record suppression, this operation provides
    fine-grained control over which specific values to suppress and how to
    replace them.

Key Features:
    - Replace individual cell values with NULL, mean, median, mode, or constant values
    - Support for conditional suppression based on various criteria
    - Group-based replacement strategies (group mean/mode)
    - Outlier and rare value detection for automatic suppression
    - Memory-efficient batch processing with statistics caching
    - Full integration with PAMOLA base framework
    - Dask support for distributed processing

Design Notes:
    - Numeric validation is enforced for mean-based strategies
    - Group statistics are computed per batch to manage memory
    - Outlier detection uses IQR and Z-score methods
    - Rare value detection based on frequency thresholds
    - Supports both global and group-based replacement strategies
    - Preserves original data types after replacement when possible

Changelog:
    1.2.0 (2025-06-15):
        - Fixed create_field_mask argument order and type issues
        - Fixed outlier detection type checker warnings
        - Fixed metrics type compatibility issues
        - Added explicit type annotations for detailed_stats
        - Added median suppression strategy
        - Improved constant value type validation
        - Changed to OrderedDict for proper FIFO cache management
        - Enhanced dtype preservation logic
        - Simplified outlier method validation
    1.1.0 (2025-06-15):
        - Fixed critical bug in create_field_mask argument order
        - Fixed cell counting to properly count cells instead of rows
        - Added thread-safe counters for Dask processing
        - Optimized group mean/mode calculations
        - Added validate_numeric_field usage for proper validation
        - Added data type preservation after replacement
        - Added memory management for large group statistics
        - Enhanced metrics with suppression_by_strategy
        - Improved efficiency of group-based calculations
    1.0.0 (2025-06-15):
        - Initial implementation based on REQ-CELL-001 through REQ-CELL-006
        - Implements all suppression strategies from specification
        - Includes outlier and rare value detection
        - Full Dask support for large-scale processing
"""

from datetime import datetime
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.metric_utils import (
    calculate_suppression_metrics,
    calculate_anonymization_effectiveness,
)
from pamola_core.anonymization.commons.validation.exceptions import FieldTypeError
from pamola_core.anonymization.commons.validation_utils import (
    check_field_exists,
    FieldNotFoundError,
    validate_numeric_field,
)
from pamola_core.anonymization.commons.visualization_utils import (
    create_histogram,
    create_comparison_visualization,
)
from pamola_core.anonymization.schemas.cell_op_config import CellSuppressionConfig
from pamola_core.common.constants import Constants
from pamola_core.utils.io import (
    load_settings_operation,
    load_data_operation,
    ensure_directory,
    write_json,
    write_dataframe_to_csv,
)
from pamola_core.utils.io_helpers import crypto_utils, directory_utils
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_field_utils import create_field_mask
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
    OperationArtifact,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_registry import register

import dask.dataframe as dd

# Constants for memory management
MAX_GROUP_STATISTICS_SIZE = 10000


@register(version="1.0.0")
class CellSuppressionOperation(AnonymizationOperation):
    """
    Operation for suppressing or replacing individual cell values based on rules,
    conditional logic, or statistical criteria.

    Implements REQ-CELL-001 through REQ-CELL-006 from the
    PAMOLA.CORE Suppression Operations Sub-Specification.
    """

    def __init__(
        self,
        field_name: str,
        suppression_strategy: str = "null",
        suppression_value: Optional[Any] = None,
        group_by_field: Optional[str] = None,
        min_group_size: int = 5,
        suppress_if: Optional[str] = None,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        rare_threshold: int = 10,
        **kwargs,
    ):
        """
        Initialize the Cell Suppression Operation.

        This operation replaces or enriches cell values based on suppression strategy,
        statistical thresholds, or conditional logic.

        Parameters
        ----------
        field_name : str
            The column containing cells to suppress.
        suppression_strategy : str, optional
            Suppression strategy. Supported:
            {"null", "mean", "median", "mode", "constant", "group_mean", "group_mode"}.
            Defaults to "null".
        suppression_value : Any, optional
            Replacement value when using "constant" strategy.
        group_by_field : str, optional
            Column for group-based suppression (required if using "group_mean" or "group_mode").
        min_group_size : int, optional
            Minimum group size for valid group-level suppression. Defaults to 5.
        suppress_if : str, optional
            Automatic suppression trigger. One of {"outlier", "rare", "null"}.
        outlier_method : str, optional
            Outlier detection method if `suppress_if="outlier"`. {"iqr", "zscore"}.
        outlier_threshold : float, optional
            Threshold for outlier detection. Defaults to 1.5.
        rare_threshold : int, optional
            Frequency threshold for rare value detection. Defaults to 10.
        **kwargs
            Additional keyword arguments passed to AnonymizationOperation.
        """
        # Description fallback
        kwargs.setdefault(
            "description",
            f"Cell suppresstion for '{field_name}' using {suppression_strategy} strategy",
        )
        # Validate suppression strategy
        valid_strategies = [
            "null",
            "mean",
            "median",
            "mode",
            "constant",
            "group_mean",
            "group_mode",
        ]
        if suppression_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid suppression_strategy '{suppression_strategy}'. "
                f"Must be one of: {valid_strategies}"
            )

        # Validate strategy-specific requirements
        if suppression_strategy == "constant" and suppression_value is None:
            raise ValueError("suppression_value required for 'constant' strategy")

        if suppression_strategy in ["group_mean", "group_mode"] and not group_by_field:
            raise ValueError(
                f"group_by_field required for '{suppression_strategy}' strategy"
            )

        # Validate suppress_if parameter
        if suppress_if and suppress_if not in ["outlier", "rare", "null"]:
            raise ValueError(
                f"Invalid suppress_if '{suppress_if}'. "
                "Must be one of: outlier, rare, null"
            )

        # --- Build config object for schema-based validation ---
        config = CellSuppressionConfig(
            field_name=field_name,
            suppression_strategy=suppression_strategy,
            suppression_value=suppression_value,
            group_by_field=group_by_field,
            min_group_size=min_group_size,
            suppress_if=suppress_if,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            rare_threshold=rare_threshold,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize base AnonymizationOperation
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        # Initialize internal state
        self._cells_suppressed = 0
        self._total_cells_processed = 0
        self._non_null_cells_processed = 0
        self._group_statistics: OrderedDict[Any, Dict[str, Any]] = OrderedDict()
        self._global_statistics: Dict[str, Any] = {}
        self._suppression_by_reason: Dict[str, int] = {}
        self._suppression_by_strategy: Dict[str, int] = {}

        # Thread safety for Dask
        self._counter_lock = Lock()

        # Operation metadata
        self.operation_name = self.__class__.__name__
        self._original_df = None

    def _cache_group_statistics(self, group_val: Any, stats: Dict[str, Any]) -> None:
        """
        Cache group statistics with memory management.

        Args:
            group_val: Group identifier
            stats: Statistics to cache
        """
        # Check memory limit
        if len(self._group_statistics) >= MAX_GROUP_STATISTICS_SIZE:
            self.logger.warning(
                f"Group statistics cache full ({MAX_GROUP_STATISTICS_SIZE} groups), "
                "clearing oldest entries"
            )
            # Remove first 10% of entries (FIFO with OrderedDict)
            num_to_remove = MAX_GROUP_STATISTICS_SIZE // 10
            for _ in range(num_to_remove):
                self._group_statistics.popitem(last=False)  # Remove oldest

        self._group_statistics[group_val] = stats

    def _process_batch_dask(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Process Dask DataFrame for distributed computing.

        Args:
            ddf: Dask DataFrame

        Returns:
            Dask DataFrame with cells suppressed

        Raises:
            ImportError: If Dask is not available
        """
        # For group-based strategies, compute global statistics first
        if self.suppression_strategy in ["group_mean", "group_mode"]:
            self.logger.warning(
                "Group-based strategies with Dask compute statistics per partition. "
                "For consistent results across partitions, consider using pandas mode."
            )

            # Pre-compute global statistics if possible
            if self.suppression_strategy == "group_mean":
                try:
                    # Compute global mean for fallback
                    global_mean = ddf[self.field_name].mean().compute()
                    self._global_statistics["global_mean_dask"] = float(global_mean)
                except Exception as e:
                    self.logger.warning(f"Could not pre-compute global mean: {e}")

        # Store original counters for aggregation
        original_cells_suppressed = self._cells_suppressed
        original_total_processed = self._total_cells_processed
        original_non_null_processed = self._non_null_cells_processed

        # Define partition processing function
        def process_partition(partition):
            # Process partition and return both result and metrics
            result = self.process_batch(partition)

            # Return result with metrics as metadata
            metrics = {
                "cells_suppressed": self._cells_suppressed - original_cells_suppressed,
                "total_processed": self._total_cells_processed
                - original_total_processed,
                "non_null_processed": self._non_null_cells_processed
                - original_non_null_processed,
            }

            # Reset counters for next partition
            self._cells_suppressed = original_cells_suppressed
            self._total_cells_processed = original_total_processed
            self._non_null_cells_processed = original_non_null_processed

            # Store metrics in result attributes for aggregation
            result._partition_metrics = metrics
            return result

        # Apply to all partitions
        result_ddf = ddf.map_partitions(process_partition, meta=ddf._meta)

        # Store metadata about partitions
        self._dask_partitions_used = ddf.npartitions

        # Note: Actual metric aggregation happens after compute() in execute()

        return result_ddf

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
            # Initialize operation metadata
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
            return self._process_with_dask(
                input_data, progress_tracker=progress_tracker, reporter=reporter
            )

        elif self.use_vectorization and self.parallel_processes > 1:
            return self._process_with_joblib(
                input_data, progress_tracker=progress_tracker, reporter=reporter
            )

        else:
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

        Parameters
        ----------
        input_data : pd.DataFrame
            The full input dataset to process.

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            - mask: A global boolean mask indicating which cells were suppressed.
            - output_data: The resulting DataFrame after cell suppression.
        """
        step = "Process data with pandas"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            total_rows = len(input_data)
            chunk_size = max(1, min(self.chunk_size or total_rows, total_rows))
            processed_chunks = []
            global_mask = pd.Series(False, index=input_data.index)

            self._batch_number = 0
            self._cells_suppressed = 0
            self._total_cells_processed = 0
            self._non_null_cells_processed = 0
            self._suppression_by_strategy = {}

            suppression_counter = self._suppression_by_strategy

            # Step 1: Build global suppression mask
            global_suppression_mask = build_suppression_mask(
                df=input_data,
                field_name=self.field_name,
                suppress_if=self.suppress_if,
                condition_field=self.condition_field,
                condition_values=self.condition_values,
                condition_operator=self.condition_operator,
                rare_threshold=self.rare_threshold,
                outlier_method=self.outlier_method,
                outlier_threshold=self.outlier_threshold,
                suppression_counter=suppression_counter,
            )

            # Step 2: Process each chunk
            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                batch = input_data.iloc[start:end].copy()
                batch_mask = global_suppression_mask.iloc[start:end]

                working_field = (
                    self.output_field_name
                    if self.mode == "ENRICH" and self.output_field_name
                    else self.field_name
                )

                if self.mode == "ENRICH" and self.output_field_name:
                    batch[self.output_field_name] = batch[self.field_name]

                original = batch[working_field].copy()

                batch = apply_suppression_strategy(
                    batch=batch,
                    batch_mask=batch_mask,
                    strategy=self.suppression_strategy,
                    field_name=working_field,
                    suppression_value=self.suppression_value,
                    group_by_field=self.group_by_field,
                    min_group_size=self.min_group_size,
                )

                diff_mask = (original != batch[working_field]) & original.notna()
                global_mask.iloc[start:end] = diff_mask.values

                with self._counter_lock:
                    self._cells_suppressed += diff_mask.sum()
                    self._total_cells_processed += len(batch)
                    self._non_null_cells_processed += original.notna().sum()
                    self._suppression_by_strategy[self.suppression_strategy] = (
                        self._suppression_by_strategy.get(self.suppression_strategy, 0)
                        + diff_mask.sum()
                    )

                processed_chunks.append(batch)
                self._batch_number += 1

                self.logger.debug(
                    f"Chunk {self._batch_number}: Suppressed {diff_mask.sum()}/{len(batch)} cells"
                )

            output_data = pd.concat(processed_chunks, axis=0).reset_index(drop=True)

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": "Processed with pandas",
                        "chunks": self._batch_number,
                        "total_rows": total_rows,
                        "cells_suppressed": self._cells_suppressed,
                        "non_null_cells": self._non_null_cells_processed,
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
        import dask.dataframe as dd
        from functools import partial

        step = "Process data with Dask"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            suppression_counter = {}

            global_suppression_mask = build_suppression_mask(
                df=input_data,
                field_name=self.field_name,
                suppress_if=self.suppress_if,
                condition_field=self.condition_field,
                condition_values=self.condition_values,
                condition_operator=self.condition_operator,
                rare_threshold=self.rare_threshold,
                outlier_method=self.outlier_method,
                outlier_threshold=self.outlier_threshold,
                suppression_counter=suppression_counter,
            )

            input_data = input_data.copy()
            input_data["_suppression_mask_"] = global_suppression_mask

            ddf = dd.from_pandas(input_data, npartitions=self.npartitions or 1)

            meta_columns = input_data.columns.tolist()
            meta_dtypes = input_data.dtypes.to_dict()

            if self.mode == "ENRICH":
                self.output_field_name = (
                    self.output_field_name or self.field_name + "_suppressed"
                )
                if self.output_field_name not in meta_columns:
                    meta_columns.append(self.output_field_name)
                    meta_dtypes[self.output_field_name] = input_data[
                        self.field_name
                    ].dtype

            # Ensure suppression mask column exists in meta
            if "_suppression_mask_" not in meta_columns:
                meta_columns.append("_suppression_mask_")
                meta_dtypes["_suppression_mask_"] = bool

            meta = pd.DataFrame(
                {
                    col: pd.Series(dtype=meta_dtypes.get(col, "object"))
                    for col in meta_columns
                }
            )

            # Use functools.partial to freeze arguments
            partition_func = partial(
                suppression_partition_dask,
                field_name=self.field_name,
                mode=self.mode,
                output_field_name=self.output_field_name,
                suppression_strategy=self.suppression_strategy,
                suppression_value=self.suppression_value,
                group_by_field=self.group_by_field,
                min_group_size=self.min_group_size,
            )

            processed_ddf = ddf.map_partitions(partition_func, meta=meta)
            result_df = processed_ddf.compute()

            global_mask = (
                result_df.pop("_suppression_mask_")
                .fillna(False)
                .astype(bool)
                .reset_index(drop=True)
            )
            output_df = result_df.reset_index(drop=True)

            self._cells_suppressed = global_mask.sum()
            self._non_null_cells_processed = input_data[self.field_name].notna().sum()
            self._total_cells_processed = len(input_data)
            self._suppression_by_strategy = suppression_counter

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": f"Processed with Dask using {self.npartitions} partitions",
                        "total_rows": len(input_data),
                        "cells_suppressed": self._cells_suppressed,
                        "non_null_cells": self._non_null_cells_processed,
                        "final_rows": len(output_df),
                    },
                )

            return global_mask, output_df

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
        """
        Process the entire DataFrame in parallel using Joblib.
        """
        from joblib import Parallel, delayed

        step = "Process data with Joblib"

        if progress_tracker:
            progress_tracker.update(1, {"step": step, "operation": self.operation_name})

        try:
            total_rows = len(input_data)
            chunk_size = max(1, min(self.chunk_size or total_rows, total_rows))
            chunks = [
                input_data.iloc[i : i + chunk_size].copy()
                for i in range(0, total_rows, chunk_size)
            ]

            suppression_counter = {}

            # Build global suppression mask once
            global_suppression_mask = build_suppression_mask(
                df=input_data,
                field_name=self.field_name,
                suppress_if=self.suppress_if,
                condition_field=self.condition_field,
                condition_values=self.condition_values,
                condition_operator=self.condition_operator,
                rare_threshold=self.rare_threshold,
                outlier_method=self.outlier_method,
                outlier_threshold=self.outlier_threshold,
                suppression_counter=suppression_counter,
            )

            batch_masks = [
                global_suppression_mask.iloc[i : i + chunk_size]
                for i in range(0, total_rows, chunk_size)
            ]

            # Run in parallel with all arguments passed explicitly
            results = Parallel(n_jobs=self.parallel_processes)(
                delayed(suppression_partition_joblib)(
                    chunk,
                    mask,
                    self.field_name,
                    self.output_field_name,
                    self.mode,
                    self.suppression_strategy,
                    self.suppression_value,
                    self.group_by_field,
                    self.min_group_size,
                )
                for chunk, mask in zip(chunks, batch_masks)
            )

            processed_batches, diff_masks = zip(*results)
            output_df = pd.concat(processed_batches, axis=0).reset_index(drop=True)
            global_mask = (
                pd.concat(diff_masks, axis=0).reset_index(drop=True).astype(bool)
            )

            # Stats
            self._original_record_count = total_rows
            self._cells_suppressed = global_mask.sum()
            self._non_null_cells_processed = input_data[self.field_name].notna().sum()
            self._total_cells_processed = total_rows
            self._batch_number = len(chunks)
            self._suppression_by_strategy = suppression_counter

            if reporter:
                reporter.add_operation(
                    f"Operation {self.operation_name}",
                    status="info",
                    details={
                        "step": step,
                        "message": f"Processed with Joblib using {self.parallel_processes} processes",
                        "chunks": self._batch_number,
                        "total_rows": total_rows,
                        "cells_suppressed": self._cells_suppressed,
                        "non_null_cells": self._non_null_cells_processed,
                        "final_rows": len(output_df),
                    },
                )

            return global_mask, output_df

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

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect cell suppression specific metrics.

        Args:
            original_data: Original data series
            anonymized_data: Anonymized data series

        Returns:
            Dictionary of suppression metrics
        """
        # Calculate suppression rate based on non-null cells
        suppression_rate = 0.0
        if self._non_null_cells_processed > 0:
            suppression_rate = (
                self._cells_suppressed / self._non_null_cells_processed
            ) * 100

        metrics = {
            "operation_type": "cell_suppression",
            "suppression_strategy": self.suppression_strategy,
            "cells_suppressed": self._cells_suppressed,
            "suppression_rate": suppression_rate,
            "total_cells_processed": self._total_cells_processed,
            "non_null_cells_processed": self._non_null_cells_processed,
            "suppression_by_reason": dict(self._suppression_by_reason),
            "suppression_by_strategy": dict(self._suppression_by_strategy),
        }

        # Add strategy-specific metrics
        if self.suppress_if == "outlier":
            metrics["outlier_method"] = self.outlier_method
            metrics["outlier_threshold"] = self.outlier_threshold
            if "outlier" in self._suppression_by_reason:
                metrics["outliers_detected"] = self._suppression_by_reason["outlier"]

        elif self.suppress_if == "rare":
            metrics["rare_threshold"] = self.rare_threshold
            if "rare" in self._suppression_by_reason:
                metrics["rare_values_detected"] = self._suppression_by_reason["rare"]

        # Add group statistics summary if applicable
        if self.suppression_strategy in ["group_mean", "group_mode"]:
            metrics["group_count"] = len(self._group_statistics)
            metrics["group_by_field"] = self.group_by_field
            metrics["min_group_size"] = self.min_group_size

            # Add sample of group statistics (not all to avoid huge metrics)
            if self._group_statistics:
                sample_size = min(10, len(self._group_statistics))
                # Simplify to single-level dict for type compatibility
                sample_groups = {}
                for i, (k, v) in enumerate(list(self._group_statistics.items())):
                    if i >= sample_size:
                        break
                    # Extract key metric based on strategy
                    if self.suppression_strategy == "group_mean":
                        sample_groups[str(k)] = v.get("mean", 0.0)
                    else:  # group_mode
                        sample_groups[str(k)] = str(v.get("mode", "N/A"))

                metrics["group_statistics_sample"] = sample_groups

        # Add global statistics if calculated
        if self._global_statistics:
            metrics["global_statistics"] = self._global_statistics.copy()

        # Use framework utility for additional metrics
        suppression_metrics = calculate_suppression_metrics(
            original_data, anonymized_data
        )
        metrics.update(suppression_metrics)

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
            filename = f"{operation_name}_metrics_{operation_timestamp}.json"
            metrics_path = metrics_dir / filename

            write_json(metrics, metrics_path, encryption_key=self.encryption_key)

            result.add_artifact(
                artifact_type="json",
                path=metrics_path,
                description=f"Metrics for {self.operation_name}",
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
        operation_timestamp: Optional[str] = None,
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
                        "step": step,
                        "message": "Metrics collected and saved successfully",
                        "summary": {
                            "operation_type": metrics.get("operation_type"),
                            "cells_suppressed": metrics.get("cells_suppressed"),
                            "suppression_rate": round(
                                metrics.get("suppression_rate", 0), 2
                            ),
                            "non_null_cells_processed": metrics.get(
                                "non_null_cells_processed"
                            ),
                            "total_cells_processed": metrics.get(
                                "total_cells_processed"
                            ),
                            "records_processed": metrics.get("records_processed"),
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
            Timestamp string for file naming.
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
        Generate visualization showing distribution changes.

        Args:
            input_data: Original DataFrame
            output_data: Processed DataFrame
            task_dir: Directory to save visualization
            result: OperationResult to register artifacts
            operation_timestamp: Timestamp string for file naming

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
            # Extract the relevant field
            if self.mode == "ENRICH" and self.output_field_name:
                original_series = input_data[self.field_name]
                processed_series = output_data[self.output_field_name]
            else:
                original_series = input_data[self.field_name]
                processed_series = output_data[self.field_name]

            # Create comparison visualization
            viz_path = create_comparison_visualization(
                original_series,
                processed_series,
                vis_dir,
                self.field_name,
                operation_name,
                timestamp=operation_timestamp,
                **kwargs_visualization,
            )

            if not str(viz_path).startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=viz_path,
                    description="Generated comparison visualization",
                    category=Constants.Artifact_Category_Visualization,
                )
            else:
                self.logger.error(
                    f"Failed to create comparison visualization: {viz_path}"
                )
                return None

            # Create histogram if numeric data
            if pd.api.types.is_numeric_dtype(original_series):
                try:
                    distribution_file_name = f"{operation_name}_suppressed_distribution_{operation_timestamp}.png"
                    file_path = vis_dir / distribution_file_name
                    hist_path = create_histogram(
                        data=processed_series.dropna().tolist(),
                        output_path=file_path,
                        title=f"{self.field_name} Distribution After Cell Suppression",
                        x_label=self.field_name,
                        bins=30,
                        **kwargs_visualization,
                    )

                    if not hist_path.startswith("Error"):
                        result.add_artifact(
                            artifact_type="png",
                            path=viz_path,
                            description="Generated histogram visualization",
                            category=Constants.Artifact_Category_Visualization,
                        )
                    else:
                        self.logger.error(
                            f"Failed to create histogram visualization: {viz_path}"
                        )
                        return None

                except Exception as e:
                    self.logger.warning(f"Failed to generate visualization: {e}")

            return viz_path

        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            return None

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
            Timestamp string for file naming.
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
        Get cache-relevant parameters for CellSuppressionOperation to uniquely
        identify this configuration and support caching.

        Returns:
            Dict[str, Any]: Dictionary of relevant parameters for cache identity.
        """
        return {
            "operation": self.operation_name,
            "version": self.version,
            "field_name": self.field_name,
            "mode": self.mode,
            "output_field_name": self.output_field_name,
            "save_suppressed_schema": getattr(self, "save_suppressed_schema", True),
            # Suppression-specific configuration
            "suppression_strategy": self.suppression_strategy,
            "suppression_value": self.suppression_value,
            "group_by_field": self.group_by_field,
            "min_group_size": self.min_group_size,
            "suppress_if": self.suppress_if,
            "outlier_method": self.outlier_method,
            "outlier_threshold": self.outlier_threshold,
            "rare_threshold": self.rare_threshold,
            # Conditional logic
            "condition_field": self.condition_field,
            "condition_values": self.condition_values,
            "condition_operator": self.condition_operator,
            "ka_risk_field": getattr(self, "ka_risk_field", None),
            "risk_threshold": getattr(self, "risk_threshold", 5.0),
            # Execution and system parameters
            "optimize_memory": self.optimize_memory,
            "adaptive_chunk_size": self.adaptive_chunk_size,
            "generate_visualization": self.generate_visualization,
            "save_output": self.save_output,
            "output_format": self.output_format,
            "use_cache": self.use_cache,
            "force_recalculation": self.force_recalculation,
            # Parallelization and performance
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "dask_partition_size": self.dask_partition_size,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "chunk_size": self.chunk_size,
            # Visualization
            "visualization_backend": self.visualization_backend,
            "visualization_theme": self.visualization_theme,
            "visualization_strict": self.visualization_strict,
            # Encryption
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
        """
        Validate that required input parameters and fields exist in the DataFrame.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        # Validate presence of field_name
        if self.field_name not in df.columns:
            self.logger.error(f"Missing required field: '{self.field_name}'")
            return False

        # Validate condition_field
        if self.condition_field and self.condition_field not in df.columns:
            self.logger.error(
                f"Condition field '{self.condition_field}' not found in DataFrame."
            )
            return False

        # Validate group_by_field for group-based strategies
        if self.suppression_strategy in ["group_mean", "group_mode"]:
            if not self.group_by_field:
                self.logger.error(
                    f"group_by_field is required for suppression strategy '{self.suppression_strategy}'"
                )
                return False

            if isinstance(self.group_by_field, str):
                group_fields = [self.group_by_field]
            elif isinstance(self.group_by_field, list):
                group_fields = self.group_by_field
            else:
                self.logger.error("group_by_field must be a string or list of strings")
                return False

            missing = [col for col in group_fields if col not in df.columns]
            if missing:
                self.logger.error(
                    f"group_by_field columns missing in DataFrame: {missing}"
                )
                return False

        # Validate suppression_value for constant strategy
        if self.suppression_strategy == "constant" and self.suppression_value is None:
            self.logger.error("suppression_value is required for strategy 'constant'")
            return False

        # Validate suppress_if logic
        if self.suppress_if and self.suppress_if not in {"outlier", "rare", "null"}:
            self.logger.error(f"Invalid suppress_if value: {self.suppress_if}")
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
        return (
            f"CellSuppressionOperation("
            f"field='{self.field_name}', "
            f"strategy='{self.suppression_strategy}', "
            f"suppress_if='{self.suppress_if}')"
        )


def suppression_partition_dask(
    batch: pd.DataFrame,
    field_name: str,
    mode: str,
    output_field_name: Optional[str],
    suppression_strategy: str,
    suppression_value: Any,
    group_by_field: Optional[List[str]],
    min_group_size: int,
) -> pd.DataFrame:

    working_field = (
        output_field_name if mode == "ENRICH" and output_field_name else field_name
    )

    if mode == "ENRICH" and working_field:
        batch[working_field] = batch[field_name]

    original = batch[working_field].copy()

    result = apply_suppression_strategy(
        batch=batch,
        batch_mask=batch["_suppression_mask_"],
        strategy=suppression_strategy,
        field_name=working_field,
        suppression_value=suppression_value,
        group_by_field=group_by_field,
        min_group_size=min_group_size,
    )

    result["_suppression_mask_"] = (
        original != result[working_field]
    ) & original.notna()
    return result


def suppression_partition_joblib(
    batch: pd.DataFrame,
    batch_mask: pd.Series,
    field_name: str,
    output_field_name: Optional[str],
    mode: str,
    suppression_strategy: str,
    suppression_value: Any,
    group_by_field: Optional[str],
    min_group_size: Optional[int],
) -> Tuple[pd.DataFrame, pd.Series]:

    working_field = (
        output_field_name if mode == "ENRICH" and output_field_name else field_name
    )

    if mode == "ENRICH" and working_field:
        batch[working_field] = batch[field_name]

    original = batch[working_field].copy()

    batch = apply_suppression_strategy(
        batch=batch,
        batch_mask=batch_mask,
        strategy=suppression_strategy,
        field_name=working_field,
        suppression_value=suppression_value,
        group_by_field=group_by_field,
        min_group_size=min_group_size,
    )

    diff_mask = (original != batch[working_field]) & original.notna()

    return batch, diff_mask


def build_suppression_mask(
    df: pd.DataFrame,
    field_name: str,
    suppress_if: Optional[str] = None,
    condition_field: Optional[str] = None,
    condition_values: Optional[List[Any]] = None,
    condition_operator: Optional[str] = None,
    rare_threshold: int = 2,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    suppression_counter: Optional[Dict[str, int]] = None,
) -> pd.Series:
    """
    Build boolean mask indicating which cells to suppress based on various criteria.

    Parameters:
        df: DataFrame to process
        field_name: Field to apply suppression logic
        suppress_if: Built-in condition ('outlier', 'rare', 'null', or None)
        condition_field: Field used for conditional suppression
        condition_values: Values used for matching in conditional suppression
        condition_operator: Operator used in conditional suppression
        rare_threshold: Frequency threshold for determining rare values
        outlier_method: Method for outlier detection ('iqr' or 'zscore')
        outlier_threshold: Threshold multiplier for outlier detection
        suppression_counter: Optional dictionary to track suppression reasons

    Returns:
        Boolean Series indicating which cells should be suppressed
    """
    mask = pd.Series(False, index=df.index)

    if suppress_if == "outlier":
        mask = detect_outliers(
            series=df[field_name], method=outlier_method, threshold=outlier_threshold
        )
        if suppression_counter is not None:
            suppression_counter["outlier"] = (
                suppression_counter.get("outlier", 0) + mask.sum()
            )

    elif suppress_if == "rare":
        value_counts = df[field_name].value_counts()
        rare_values = value_counts[value_counts < rare_threshold].index
        mask = df[field_name].isin(rare_values)
        if suppression_counter is not None:
            suppression_counter["rare"] = (
                suppression_counter.get("rare", 0) + mask.sum()
            )

    elif suppress_if == "null":
        mask = df[field_name].isna()
        if suppression_counter is not None:
            suppression_counter["null"] = (
                suppression_counter.get("null", 0) + mask.sum()
            )

    elif condition_field and condition_values:
        if not check_field_exists(df, condition_field):
            raise FieldNotFoundError(condition_field, list(df.columns))

        mask = create_field_mask(
            df=df,
            field_name=field_name,
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
        )
        if suppression_counter is not None:
            suppression_counter["conditional"] = (
                suppression_counter.get("conditional", 0) + mask.sum()
            )

    return mask


def detect_outliers(
    series: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a numeric Series using IQR or Z-score method.

    Parameters:
        series: The input Series to analyze
        method: Method for outlier detection ("iqr" or "zscore")
        threshold: Threshold multiplier (IQR or Z-score)

    Returns:
        Boolean Series where True indicates an outlier
    """
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)

    numeric_series = pd.to_numeric(series, errors="coerce")
    non_null_series = numeric_series.dropna()

    if len(non_null_series) < 3:
        return pd.Series(False, index=series.index)

    if method == "iqr":
        Q1 = non_null_series.quantile(0.25)
        Q3 = non_null_series.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            return pd.Series(False, index=series.index)

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        return ((numeric_series < lower_bound) | (numeric_series > upper_bound)).fillna(
            False
        )

    elif method == "zscore":
        mean = non_null_series.mean()
        std = non_null_series.std()

        if std == 0:
            return pd.Series(False, index=series.index)

        z_scores = np.abs((numeric_series - mean) / std)
        return (z_scores > threshold).fillna(False)

    return pd.Series(False, index=series.index)


def apply_group_mean(
    df: pd.DataFrame,
    field_name: str,
    suppress_mask: pd.Series,
    group_by_field: Union[str, List[str]],
    min_group_size: int,
    global_mean: Optional[float] = None,
) -> pd.Series:
    """
    Apply group-based mean suppression, falling back to global mean if needed.

    Parameters:
        df : DataFrame
        field_name : Column to suppress
        suppress_mask : Boolean Series indicating which cells to suppress
        group_by_field : Single column or list of columns to group by
        min_group_size : Minimum number of records in a group to use group mean
        global_mean : Value to use if group size is too small

    Returns:
        Series with suppressed values
    """
    result_series = df[field_name].copy()
    result_series = result_series.astype("float64")
    group_stats = df.groupby(group_by_field)[field_name].agg(["mean", "count"])

    for group_val, row in group_stats.iterrows():
        if isinstance(group_by_field, str):
            group_mask = df[group_by_field] == group_val
        else:
            # Ensure group_val is tuple if multiple group keys
            if not isinstance(group_val, tuple):
                group_val = (group_val,)
            group_mask = pd.Series(True, index=df.index)
            for col, val in zip(group_by_field, group_val):
                group_mask &= df[col] == val

        group_mask &= suppress_mask

        if group_mask.any():
            if row["count"] >= min_group_size:
                result_series.loc[group_mask] = row["mean"]
            elif pd.notna(global_mean):
                result_series.loc[group_mask] = global_mean

    return result_series


def apply_group_mode(
    df: pd.DataFrame,
    field_name: str,
    suppress_mask: pd.Series,
    group_by_field: Union[str, List[str]],
    min_group_size: int,
    global_mode: Optional[Any] = None,
) -> pd.Series:
    """
    Apply group-based mode suppression, falling back to global mode if needed.

    Parameters:
        df: Input DataFrame
        field_name: Column to modify
        suppress_mask: Boolean Series indicating which rows to suppress
        group_by_field: One or more columns to group by
        min_group_size: Minimum group size to apply group mode
        global_mode: Precomputed global mode for fallback

    Returns:
        Series with updated suppressed values
    """
    result_series = df[field_name].copy()
    grouped = df.groupby(group_by_field)[field_name]

    for group_val, group_series in grouped:
        # Build a boolean mask for rows belonging to the current group and marked for suppression
        if isinstance(group_by_field, str):
            group_mask = (df[group_by_field] == group_val) & suppress_mask
        else:
            # If grouping by multiple columns, group_val will be a tuple
            match = pd.Series([True] * len(df), index=df.index)
            for col, val in zip(group_by_field, group_val):
                match &= df[col] == val
            group_mask = match & suppress_mask

        if not group_mask.any():
            continue

        if len(group_series) >= min_group_size:
            mode_vals = group_series.mode()
            if not mode_vals.empty:
                result_series.loc[group_mask] = mode_vals.iloc[0]
        elif global_mode is not None:
            result_series.loc[group_mask] = global_mode

    return result_series


def apply_suppression_strategy(
    batch: pd.DataFrame,
    batch_mask: pd.Series,
    strategy: str,
    field_name: str,
    suppression_value: Any = None,
    group_by_field: Optional[List[str]] = None,
    min_group_size: int = 5,
) -> pd.DataFrame:
    working_field = field_name
    original_dtype = batch[working_field].dtype

    # --- Strategy cases ---
    if strategy == "null":
        if pd.api.types.is_numeric_dtype(original_dtype):
            batch.loc[batch_mask, working_field] = np.nan
        else:
            batch.loc[batch_mask, working_field] = None

    elif strategy == "mean":
        _require_numeric(batch, working_field)
        mean_val = pd.to_numeric(batch[working_field], errors="coerce").mean()
        if pd.notna(mean_val):
            batch.loc[batch_mask, working_field] = mean_val

    elif strategy == "median":
        _require_numeric(batch, working_field)
        median_val = pd.to_numeric(batch[working_field], errors="coerce").median()
        if pd.notna(median_val):
            batch.loc[batch_mask, working_field] = median_val

    elif strategy == "mode":
        mode_result = batch[working_field].mode()
        if not mode_result.empty:
            mode_val = mode_result.iloc[0]
            batch.loc[batch_mask, working_field] = mode_val

    elif strategy == "constant":
        try:
            suppression_value_casted = np.array([suppression_value]).astype(
                original_dtype
            )[0]
        except Exception:
            suppression_value_casted = suppression_value
        batch.loc[batch_mask, working_field] = suppression_value_casted

    elif strategy == "group_mean":
        _require_numeric(batch, working_field)
        if not group_by_field:
            raise ValueError("group_mean strategy requires group_by_field")
        global_mean = pd.to_numeric(batch[working_field], errors="coerce").mean()
        batch[working_field] = apply_group_mean(
            df=batch,
            field_name=working_field,
            suppress_mask=batch_mask,
            group_by_field=group_by_field,
            min_group_size=min_group_size,
            global_mean=global_mean,
        )

    elif strategy == "group_mode":
        if not group_by_field:
            raise ValueError("group_mode strategy requires group_by_field")
        global_mode_result = batch[working_field].mode()
        global_mode = (
            global_mode_result.iloc[0] if not global_mode_result.empty else None
        )
        batch[working_field] = apply_group_mode(
            df=batch,
            field_name=working_field,
            suppress_mask=batch_mask,
            group_by_field=group_by_field,
            min_group_size=min_group_size,
            global_mode=global_mode,
        )

    else:
        raise ValueError(f"Unsupported suppression strategy: {strategy}")

    # --- Restore dtype ---
    try:
        if pd.api.types.is_numeric_dtype(original_dtype):
            batch[working_field] = pd.to_numeric(batch[working_field], errors="coerce")
        else:
            batch[working_field] = batch[working_field].astype(
                original_dtype, errors="ignore"
            )
    except Exception:
        pass

    return batch


def _require_numeric(batch: pd.DataFrame, field_name: str) -> None:
    """
    Ensure that the field is numeric for strategies that require numeric values.
    """
    if not validate_numeric_field(batch, field_name, allow_null=True):
        raise FieldTypeError(
            field_name,
            expected_type="numeric",
            actual_type=str(batch[field_name].dtype),
        )
