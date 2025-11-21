"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Add Or Modify Fields Operation
Description: Operation for add/modify field based on lookups or conditions.
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides an operation for add/modify field based on lookups or
conditions. It supports both explicit lookup tables and conditional logic.

Key features:
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for distribution comparisons
- Chunked processing support for large datasets
- Memory-efficient operation with explicit cleanup for large datasets

Implementation follows the PAMOLA.CORE operation framework with standardized
interfaces for input/output, progress tracking, and result reporting.
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Iterable, Iterator, Union, Optional
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter, WriterResult
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.common.constants import Constants
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
from pamola_core.transformations.schemas.add_modify_fields_core_schema import (
    AddOrModifyFieldsOperationConfig,
)

# Configure module logger
logger = logging.getLogger(__name__)


@register(version="1.0.0")
class AddOrModifyFieldsOperation(TransformationOperation):
    """
    Operation for adding or modifying fields based on lookups or conditions.
    """

    def __init__(
        self,
        name: str = "add_modify_fields_operation",
        field_operations: Optional[Dict[str, Dict[str, Any]]] = None,
        lookup_tables: Optional[Dict[str, Union[Path, Dict[Any, Any]]]] = None,
        **kwargs,
    ):
        """
        Initialize operation.

        Parameters:
        -----------
        name : str
            Name of the operation (default: "add_modify_fields_operation")
        field_operations : dict, optional
            Fields operations
        lookup_tables : dict, optional
        **kwargs: dict
            Additional keyword arguments passed to TransformationOperation.
        """
        # Ensure default metadata
        kwargs.setdefault("name", name)
        kwargs.setdefault(
            "description",
            f"Add or modify fields based on lookups or conditions.",
        )

        # --- Build config object ---
        config = AddOrModifyFieldsOperationConfig(
            field_operations=field_operations or {},
            lookup_tables=lookup_tables or {},
            **kwargs,
        )

        # Inject config for parent class
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

            # Config logger task for operations
            self.logger = kwargs.get("logger", self.logger)

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            result = OperationResult(status=OperationStatus.PENDING)

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)
            output_dir = directories["output"]
            visualizations_dir = directories["visualizations"]
            metrics_dir = directories["metrics"]

            # Save configuration to task directory
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up progress tracking
            # Preparation, Data Loading, Validation, Checking Cache, Processing, Metrics, Finalization
            total_steps = 5 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"step": "Preparation"})

            # Step 1: Data Loading
            if progress_tracker:
                progress_tracker.update(1, {"step": "Data Loading"})

            # Get and validate data
            try:
                # Load data
                settings_operation = load_settings_operation(
                    data_source, dataset_name, **kwargs
                )
                df = load_data_operation(
                    data_source, dataset_name, **settings_operation
                )

                if df is None:
                    error_message = "Failed to load input data"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR, error_message=error_message
                    )
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 2: Validation
            if progress_tracker:
                progress_tracker.update(1, {"step": "Validation"})

            try:
                if reporter:
                    reporter.add_operation(
                        name=f"Add/modify fields",
                        details={
                            "operation_type": self.operation_name,
                            "field_operations": self.field_operations,
                            "lookup_tables": self.lookup_tables,
                        },
                    )

                # Validation
                # Get a copy of the original data for metrics calculation
                original_df = (
                    df.map_partitions(lambda partition: partition.copy())
                    if isinstance(df, dd.DataFrame)
                    else df.copy(deep=True)
                )
            except Exception as e:
                error_message = f"Validation error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 3: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Checking Cache"})

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(
                    df=df, task_dir=task_dir, reporter=reporter
                )

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(
                            total_steps - 3, {"step": "Complete (cached)"}
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Add/modify fields (from cache)", details={"cached": True}
                        )
                    return cache_result

            # Step 4: Processing
            if progress_tracker:
                progress_tracker.update(1, {"step": "Processing"})

            try:
                processed_df = self._process_dataframe(df, progress_tracker)
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 5: Metrics
            if progress_tracker:
                progress_tracker.update(1, {"step": "Metrics"})

            # Initialize metrics in scope
            metrics = {}
            metrics_result = DataWriter
            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            try:
                metrics = self._calculate_all_metrics(original_df, processed_df)

                metrics_result = self._save_metrics(
                    metrics=metrics,
                    task_dir=metrics_dir,
                    writer=writer,
                    result=result,
                    reporter=reporter,
                    progress_tracker=progress_tracker,
                    operation_timestamp=operation_timestamp,
                )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Finalization (Visualizations and Output Data)
            if progress_tracker:
                progress_tracker.update(1, {"step": "Finalization"})

            visualizations = {}
            # Generate visualizations if required
            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    }
                    visualizations = self._handle_visualizations(
                        original_df=original_df,
                        processed_df=processed_df,
                        metrics=metrics,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        progress_tracker=progress_tracker,
                        operation_timestamp=operation_timestamp,
                        **kwargs_encryption,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical

            output_result = DataWriter
            # Save output data if required
            if self.save_output:
                try:
                    output_result = self._save_output_data(
                        processed_df=processed_df,
                        task_dir=output_dir,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=progress_tracker,
                        operation_timestamp=operation_timestamp,
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
                        original_df=original_df,
                        processed_df=processed_df,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Report completion
            if reporter:
                # Create the details dictionary with checks for all values
                details = {
                    "execution_time_seconds": self.execution_time,
                    "records_processed": self.process_count,
                    "records_per_second": (
                        self.process_count / self.execution_time
                        if self.execution_time > 0
                        else 0
                    ),
                }

                # Only add generalization_ratio if metrics exists and has this key
                if metrics and isinstance(metrics, dict):
                    generalization_ratio = metrics.get("generalization_ratio")
                    if generalization_ratio is not None:
                        details["generalization_ratio"] = generalization_ratio

                # Add the operation to the reporter
                reporter.add_operation(f"Add/modify fields completed", details=details)

            # Cleanup memory
            self._cleanup_memory(original_df, processed_df)

            # Set success status
            result.status = OperationStatus.SUCCESS

            return result
        except Exception as e:
            # Handle unexpected errors
            error_message = f"Error in add/modify fields operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message, exception=e
            )

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data.

        Parameters:
        -----------
        batch : pd.DataFrame
            Batch to process

        Returns:
        --------
        pd.DataFrame
            Processed batch
        """
        for field_name, field_config in self.field_operations.items():
            operation_type = field_config.get("operation_type")

            if operation_type == "add_constant":
                constant_value = field_config.get("constant_value")
                if field_name and field_name not in batch.columns and constant_value:
                    batch[field_name] = constant_value

            if operation_type == "add_from_lookup":
                base_on_column = field_config.get("base_on_column")
                lookup_table_name = field_config.get("lookup_table_name")
                lookup_table = self.lookup_tables.get(lookup_table_name)
                if (
                    field_name
                    and field_name not in batch.columns
                    and base_on_column in batch.columns
                    and lookup_table_name
                    and lookup_table_name in self.lookup_tables
                    and lookup_table
                ):
                    if isinstance(lookup_table, Path):
                        with open(lookup_table, "r") as file:
                            lookup_table = json.load(file)

                    if lookup_table:
                        batch[field_name] = batch[base_on_column].map(lookup_table)

            if operation_type == "add_conditional":
                condition = field_config.get("condition")
                value_if_true = field_config.get("value_if_true")
                value_if_false = field_config.get("value_if_false")
                if (
                    field_name
                    and field_name not in batch.columns
                    and condition
                    and value_if_true
                ):
                    batch[field_name] = batch.apply(
                        lambda row: (
                            value_if_true if eval(condition) else value_if_false
                        ),
                        axis=1,
                    )

            if operation_type == "modify_constant":
                constant_value = field_config.get("constant_value")
                if field_name and field_name in batch.columns and constant_value:
                    # Config output field name
                    output_field_name = field_name
                    if self.mode == "ENRICH" and self.column_prefix:
                        output_field_name = f"{self.column_prefix}{field_name}"

                    batch[output_field_name] = constant_value

            if operation_type == "modify_from_lookup":
                base_on_column = field_config.get("base_on_column")
                lookup_table_name = field_config.get("lookup_table_name")
                lookup_table = self.lookup_tables.get(lookup_table_name)
                if (
                    field_name
                    and field_name in batch.columns
                    and base_on_column in batch.columns
                    and lookup_table_name
                    and lookup_table_name in self.lookup_tables
                    and lookup_table
                ):
                    if isinstance(lookup_table, Path):
                        with open(lookup_table, "r") as file:
                            lookup_table = json.load(file)

                    if lookup_table:
                        # Config output field name
                        output_field_name = field_name
                        if self.mode == "ENRICH" and self.column_prefix:
                            output_field_name = f"{self.column_prefix}{field_name}"

                        batch[output_field_name] = batch[base_on_column].map(
                            lookup_table
                        )

            if operation_type == "modify_conditional":
                condition = field_config.get("condition")
                value_if_true = field_config.get("value_if_true")
                value_if_false = field_config.get("value_if_false")
                if (
                    field_name
                    and field_name in batch.columns
                    and condition
                    and value_if_true
                ):
                    # Config output field name
                    output_field_name = field_name
                    if self.mode == "ENRICH" and self.column_prefix:
                        output_field_name = f"{self.column_prefix}{field_name}"

                    batch[output_field_name] = batch.apply(
                        lambda row: (
                            value_if_true if eval(condition) else value_if_false
                        ),
                        axis=1,
                    )

            if operation_type == "modify_expression":
                base_on_column = field_config.get("base_on_column")
                expression_character = field_config.get("expression_character")
                expression = field_config.get("expression")
                if (
                    field_name
                    and field_name in batch.columns
                    and base_on_column in batch.columns
                    and expression_character
                    and expression
                ):
                    # Config output field name
                    output_field_name = field_name
                    if self.mode == "ENRICH" and self.column_prefix:
                        output_field_name = f"{self.column_prefix}{field_name}"

                    batch[output_field_name] = batch[base_on_column].apply(
                        lambda x: eval(expression.replace(expression_character, str(x)))
                    )

        processed_batch = batch

        return processed_batch

    def process_value(self, value, **params):
        """
        Process a single value.

        Parameters:
        -----------
        value : Any
            Value to process
        **params : dict
            Additional parameters for processing

        Returns:
        --------
        Any
            Processed value
        """
        raise NotImplementedError("Not implement")

    def _check_cache(
        self, df: Union[pd.DataFrame, dd.DataFrame], task_dir: Path, reporter: Any
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame for the operation
        task_dir : Path
            Task directory
        reporter : Any
            The reporter to log artifacts to

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            operation_cache_dir = OperationCache(cache_dir=task_dir / "cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache_dir.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for name, value in metrics.items():
                        cached_result.add_metric(name, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        if isinstance(artifact, dict):
                            artifact_type = artifact.get("artifact_type", "")
                            path = artifact.get("path", "")
                            description = artifact.get("description", "")
                            category = artifact.get("category", "output")
                            cached_result.add_artifact(
                                artifact_type=artifact_type,
                                path=path,
                                description=description,
                                category=category,
                            )

                            if reporter:
                                reporter.add_operation(
                                    name=description,
                                    details={
                                        "artifact_type": artifact_type,
                                        "path": str(path),
                                    },
                                )

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Operation parameters
        """
        # Get basic operation parameters
        parameters = {
            "field_operations": self.field_operations,
            "lookup_tables": self.lookup_tables,
        }

        return parameters

    def _process_dataframe(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame for the operation
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        Union[pd.DataFrame, dd.DataFrame]
            Processed DataFrame
        """
        processed_df = self._process_dataframe_with_config(
            df=df,
            process_function=self.process_batch,
            chunk_size=self.chunk_size,
            use_dask=self.use_dask,
            npartitions=self.npartitions,
            meta=None,
            use_vectorization=self.use_vectorization,
            parallel_processes=self.parallel_processes,
            progress_tracker=progress_tracker,
            task_logger=self.logger,
        )

        return processed_df

    def _process_dataframe_with_config(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        process_function: Callable,
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: int = 1,
        meta: Optional[Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple]] = None,
        use_vectorization: bool = False,
        parallel_processes: int = 1,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
    ) -> Union[pd.DataFrame, dd.DataFrame, None, Any]:
        """
        Process a DataFrame to handle large datasets efficiently.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            The DataFrame to process.
        process_function : Callable
            Function to apply to each chunk, should take a DataFrame chunk as the first argument.
        chunk_size : int, optional
            Number of rows to process in each chunk (default: 10000).
        use_dask : bool, optional
            Whether to use Dask for processing (default: False).
        npartitions : int, optional
            Number of partitions use with Dask (default: 1).
        meta : Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple], optional
            Meta of output use with Dask.
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False).
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1).
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for monitoring the operation.
        task_logger : Optional[logging.Logger]
            Logger for tracking task progress and debugging.

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame, None, Any]
            The processed DataFrame.
        """
        df_len = (
            int(df.map_partitions(len).sum().compute())
            if isinstance(df, dd.DataFrame)
            else len(df)
        )
        if df_len == 0:
            task_logger.warning("Empty DataFrame provided! Returning as is")
            return df

        if df_len <= chunk_size:
            task_logger.warning("Small DataFrame! Process as usual")
            return process_function(
                df.compute() if isinstance(df, dd.DataFrame) else df
            )

        processed_df = None
        flag_processed = False

        task_logger.info("Process with config")

        if not flag_processed and use_dask:
            task_logger.info("Parallel Enabled")
            task_logger.info("Parallel Engine: Dask")
            task_logger.info(f"Parallel Workers: {npartitions}")
            task_logger.info(f"Using dask processing with chunk size {chunk_size}")
            if progress_tracker:
                progress_tracker.update(0, {"step": "Setting up dask processing"})

            task_logger.info("Process using Dask")

            processed_df, flag_processed = self._process_dataframe_using_dask(
                df=df,
                process_function=process_function,
                npartitions=npartitions,
                chunksize=chunk_size,
                meta=meta,
            )

            if flag_processed:
                task_logger.info("Completed using Dask")

        if not flag_processed and use_vectorization:
            task_logger.info("Parallel Enabled")
            task_logger.info("Parallel Engine: Joblib")
            task_logger.info(f"Parallel Workers: {parallel_processes}")
            task_logger.info(
                f"Using vectorized processing with chunk size {chunk_size}"
            )
            if progress_tracker:
                progress_tracker.update(0, {"step": "Setting up vectorized processing"})

            task_logger.info("Process using Joblib")

            processed_df, flag_processed = self._process_dataframe_using_joblib(
                df=df,
                process_function=process_function,
                n_jobs=parallel_processes,
                chunk_size=chunk_size,
            )

            if flag_processed:
                task_logger.info("Completed using Joblib")

        if not flag_processed and chunk_size > 1:
            # Regular chunk processing
            task_logger.info(f"Processing in chunks with chunk size {chunk_size}")
            total_chunks = (len(df) + chunk_size - 1) // chunk_size
            task_logger.info(f"Total chunks to process: {total_chunks}")
            if progress_tracker:
                progress_tracker.update(
                    0, {"step": "Processing in chunks", "total_chunks": total_chunks}
                )

            task_logger.info("Process using chunk")

            processed_df, flag_processed = self._process_dataframe_using_chunk(
                df=df, process_function=process_function, chunk_size=chunk_size
            )

            if flag_processed:
                task_logger.info("Completed using chunk")

        if not flag_processed:
            task_logger.info("Fallback process as usual")

            processed_df = process_function(
                df.compute() if isinstance(df, dd.DataFrame) else df
            )
            flag_processed = True

        return processed_df

    def _generate_dataframe_chunks(
        self, df: pd.DataFrame, chunk_size: int = 10000
    ) -> Iterator[tuple]:
        """
        Generate chunks of a DataFrame for efficient processing of large datasets.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to chunk.
        chunk_size : int, optional
            Number of rows in each chunk (default: 10000).

        Yields
        ------
        tuple
            Chunk of the original DataFrame with information.
        """
        if len(df) == 0:
            yield df, 0, 0, 0, 0
            return

        total_chunks = (len(df) + chunk_size - 1) // chunk_size

        for i in range(0, len(df), chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(df))
            chunk_num = i // chunk_size

            yield df.iloc[chunk_start:chunk_end].copy(
                deep=True
            ), chunk_num, chunk_start, chunk_end, total_chunks

    def _process_dataframe_using_dask(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        process_function: Callable,
        npartitions: int = 2,
        chunksize: int = 10000,
        meta: Optional[Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple]] = None,
    ) -> Tuple[Union[pd.DataFrame, dd.DataFrame], bool]:
        """
        Process DataFrame using Dask.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            The DataFrame to process.
        process_function : Callable
            Function to apply to each chunk, should take a DataFrame chunk as the first argument.
        npartitions : int, optional
            Number of partitions use with Dask (default: 2).
        chunksize : int, optional
            Number of rows to process in each chunk (default: 10000).
        meta : Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple], optional
            Meta of output use with Dask.

        Returns
        -------
        Tuple[Union[pd.DataFrame, dd.DataFrame], bool]
            The processed DataFrame.
        """
        import dask.dataframe as dd

        if npartitions <= 1 and chunksize <= 0:
            return df, False

        try:
            if not isinstance(df, dd.DataFrame):
                # Convert to Dask DataFrame
                total_rows = len(df)
                if npartitions is None or npartitions < 1:
                    nparts = (total_rows + chunksize - 1) // chunksize
                else:
                    nparts = npartitions

                df = dd.from_pandas(df, npartitions=nparts)

            # Define a function for processing that can be applied to Dask partitions
            def process_partition(partition):
                processed_partition = process_function(partition.copy(deep=True))
                return processed_partition

            # Apply to Dask DataFrame
            processed_partitions = df.map_partitions(process_partition)

            return processed_partitions, True
        except Exception as e:
            return df, False

    def _process_dataframe_using_joblib(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        process_function: Callable,
        n_jobs: int = -1,
        chunk_size: int = 10000,
    ) -> Tuple[Union[pd.DataFrame, dd.DataFrame], bool]:
        """
        Process DataFrame using Joblib.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            The DataFrame to process.
        process_function : Callable
            Function to apply to each chunk, should take a DataFrame chunk as the first argument.
        n_jobs : int, optional
            Number of jobs to run in parallel (-1 to use all processors) (default: -1).
        chunk_size : int, optional
            Number of rows to process in each chunk (default: 10000).

        Returns
        -------
        Tuple[Union[pd.DataFrame, dd.DataFrame], bool]
            The processed DataFrame.
        """
        from joblib import Parallel, delayed

        if n_jobs <= 0 and n_jobs != -1:
            return df, False

        try:
            df = df.compute() if isinstance(df, dd.DataFrame) else df

            # Update progress if tracker is provided
            total_rows = len(df)
            total_chunks = (total_rows + chunk_size - 1) // chunk_size

            # Function to process each chunk with error handling
            def process_with_progress(chunk, chunk_idx):
                try:
                    processed_chunk = process_function(chunk)
                    return processed_chunk
                except Exception as e:
                    return None

            # Directly use the generator to iterate through chunks
            processed_chunks = Parallel(n_jobs=n_jobs)(
                delayed(process_with_progress)(chunk, idx)
                for idx, (
                    chunk,
                    chunk_num,
                    chunk_start,
                    chunk_end,
                    total_chunks,
                ) in enumerate(
                    self._generate_dataframe_chunks(df, chunk_size=chunk_size)
                )
            )

            # Check if any processed chunks are None
            if any(chunk is None for chunk in processed_chunks):
                return df, False

            return pd.concat(processed_chunks, ignore_index=True), True
        except Exception as e:
            return df, False

    def _process_dataframe_using_chunk(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        process_function: Callable,
        chunk_size: int = 10000,
    ) -> Tuple[Union[pd.DataFrame, dd.DataFrame], bool]:
        """
        Process DataFrame using chunk.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            The DataFrame to process.
        process_function : Callable
            Function to apply to each chunk, should take a DataFrame chunk as the first argument.
        chunk_size : int, optional
            Number of rows to process in each chunk (default: 10000).

        Returns
        -------
        Tuple[Union[pd.DataFrame, dd.DataFrame], bool]
            The processed DataFrame.
        """
        if chunk_size <= 1:
            return df, False

        processed_chunks = []
        try:
            df = df.compute() if isinstance(df, dd.DataFrame) else df

            # Update progress if tracker is provided
            total_rows = len(df)
            total_chunks = (total_rows + chunk_size - 1) // chunk_size

            # Iterate through chunks using the generator function
            for (
                chunk,
                chunk_num,
                chunk_start,
                chunk_end,
                total_chunks,
            ) in self._generate_dataframe_chunks(df, chunk_size=chunk_size):
                try:
                    # Apply the processing function to the chunk
                    processed_chunk = process_function(chunk)

                    # Accumulate the results
                    processed_chunks.append(processed_chunk)
                except Exception as e:
                    # Log any error encountered while processing the chunk
                    processed_chunks.append(None)
                    continue  # Continue with the next chunk even if an error occurs

            # Check if any processed chunks are None
            if any(chunk is None for chunk in processed_chunks):
                return df, False

            return pd.concat(processed_chunks, ignore_index=True), True
        except Exception as e:
            return df, False

    def _calculate_all_metrics(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        processed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for operation.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original data
        processed_df : Union[pd.DataFrame, dd.DataFrame]
            The processed data

        Returns:
        --------
        Dict[str, Any]
            A dictionary of calculated metrics
        """
        # Calculate basic metrics
        metrics = self._collect_metrics(original_df, processed_df)

        # Add performance metrics
        metrics.update(
            {
                "execution_time_seconds": self.execution_time,
                "records_processed": self.process_count,
                "records_per_second": (
                    self.process_count / self.execution_time
                    if self.execution_time > 0
                    else 0
                ),
            }
        )

        return metrics

    def _collect_metrics(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        processed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Collect metrics for the operation.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original data
        processed_df : Union[pd.DataFrame, dd.DataFrame]
            The processed data

        Returns:
        --------
        Dict[str, Any]
            A dictionary of calculated metrics
        """
        import pandas.api.types as ptypes
        from pamola_core.transformations.commons.visualization_utils import (
            sample_large_dataset,
        )

        # Basic metrics
        metrics: Dict[str, Any] = {}

        # Specific metrics
        # Sample large datasets
        original_sample = sample_large_dataset(original_df, max_samples=10000)
        processed_sample = sample_large_dataset(processed_df, max_samples=10000)

        fields_added = [
            col
            for col in processed_sample.columns
            if col not in original_sample.columns
        ]
        fields_modified = [
            col
            for col in original_sample.columns
            if col in processed_sample.columns
            and not original_sample[col].equals(processed_sample[col])
        ]

        distribution_statistics = {}
        for processed_field in fields_modified + fields_added:
            distribution_statistics.update(
                {
                    processed_field: processed_sample[processed_field]
                    .describe()
                    .to_dict()
                }
            )

        correlations = {}
        for processed_field in fields_modified + fields_added:
            if processed_field.startswith(self.column_prefix):
                original_field = processed_field[len(self.column_prefix) :]
            else:
                original_field = processed_field

            if original_field in original_sample.columns:

                if ptypes.is_numeric_dtype(
                    original_sample[original_field]
                ) and ptypes.is_numeric_dtype(processed_sample[processed_field]):
                    x = pd.to_numeric(original_sample[original_field], errors="coerce")
                    y = pd.to_numeric(
                        processed_sample[processed_field], errors="coerce"
                    )
                    if x.std() == 0 or y.std() == 0 or x.isna().all() or y.isna().all():
                        correlation = np.nan
                    else:
                        correlation = x.corr(y, method="pearson")

                    correlations.update(
                        {
                            original_field: (
                                "NaN" if np.isnan(correlation) else correlation
                            )
                        }
                    )

        missing_values = {}
        for processed_field in fields_added:
            missing_values.update(
                {processed_field: int(processed_sample[processed_field].isnull().sum())}
            )

        metrics.update(
            {
                "fields_added_count": len(fields_added),
                "fields_modified_count": len(fields_modified),
                "distribution_statistics": distribution_statistics,
                "correlations": correlations,
                "missing_values": missing_values,
            }
        )

        return metrics

    def calculate_dataset_comparison(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        transformed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Calculate metrics comparing two datasets.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original DataFrame before transformation.
        transformed_df : Union[pd.DataFrame, dd.DataFrame]
            The transformed DataFrame after processing.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing comparison metrics.
        """
        if original_df is None or transformed_df is None:
            raise ValueError("Both DataFrames must be provided")

        start_time = time.time()

        try:
            result = {}

            # Row count comparison
            result["row_counts"] = self._compare_row_counts(original_df, transformed_df)

            # Column count comparison
            (
                result["column_counts"],
                result["common_columns"],
                result["added_columns"],
                result["removed_columns"],
            ) = self._compare_column_counts(original_df, transformed_df)

            # Value and null changes in common columns
            value_changes, null_changes = self._compare_values_and_nulls(
                original_df, transformed_df
            )
            result["value_changes"] = value_changes
            result["null_changes"] = null_changes

            # Memory usage comparison
            result["memory_usage"] = self._compare_memory_usage(
                original_df, transformed_df
            )

            elapsed_time = time.time() - start_time

            return result
        except Exception as e:
            raise

    def _compare_row_counts(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        transformed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Compare the row counts between the original and transformed DataFrames.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original DataFrame.
        transformed_df : Union[pd.DataFrame, dd.DataFrame]
            The transformed DataFrame.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing row count differences and percentage change.
        """
        original_rows = (
            int(original_df.map_partitions(len).sum().compute())
            if isinstance(original_df, dd.DataFrame)
            else len(original_df)
        )
        transformed_rows = (
            int(transformed_df.map_partitions(len).sum().compute())
            if isinstance(transformed_df, dd.DataFrame)
            else len(transformed_df)
        )

        row_diff = transformed_rows - original_rows
        row_pct_change = (
            (row_diff / original_rows * 100) if original_rows > 0 else float("inf")
        )

        return {
            "original": original_rows,
            "transformed": transformed_rows,
            "difference": row_diff,
            "percent_change": row_pct_change,
        }

    def _compare_column_counts(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        transformed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
        """
        Compare the column counts between the original and transformed DataFrames.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original DataFrame.
        transformed_df : pd.DataFrame
            The transformed DataFrame.

        Returns:
        --------
        Tuple containing:
            - Dict with column count differences and percentage change.
            - List of common columns.
            - List of added columns.
            - List of removed columns.
        """
        original_cols = set(original_df.columns)
        transformed_cols = set(transformed_df.columns)
        common_cols = original_cols.intersection(transformed_cols)
        added_cols = transformed_cols - original_cols
        removed_cols = original_cols - transformed_cols

        column_count = {
            "original": len(original_cols),
            "transformed": len(transformed_cols),
            "difference": len(transformed_cols) - len(original_cols),
            "percent_change": (
                (
                    (len(transformed_cols) - len(original_cols))
                    / len(original_cols)
                    * 100
                )
                if len(original_cols) > 0
                else float("inf")
            ),
        }

        return column_count, list(common_cols), list(added_cols), list(removed_cols)

    def _compare_values_and_nulls(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        transformed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compare values and nulls in common columns between the original and transformed DataFrames.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original DataFrame.
        transformed_df : Union[pd.DataFrame, dd.DataFrame]
            The transformed DataFrame.

        Returns:
        --------
        Tuple containing:
            - Dictionary of value changes for common columns.
            - Dictionary of null changes for common columns.
        """
        value_changes = {}
        null_changes = {}

        common_cols = original_df.columns.intersection(transformed_df.columns)

        # Sample data to speed up comparison for large datasets
        original_rows = (
            int(original_df.map_partitions(len).sum().compute())
            if isinstance(original_df, dd.DataFrame)
            else len(original_df)
        )
        transformed_rows = (
            int(transformed_df.map_partitions(len).sum().compute())
            if isinstance(transformed_df, dd.DataFrame)
            else len(transformed_df)
        )
        sample_size = min(10000, original_rows, transformed_rows)

        original_sample = (
            original_df.sample(sample_size)
            if original_rows > sample_size
            else original_df
        )
        original_sample = (
            original_sample.compute()
            if isinstance(original_df, dd.DataFrame)
            else original_sample
        )

        transformed_df_index = (
            transformed_df.index.compute()
            if isinstance(transformed_df, dd.DataFrame)
            else transformed_df.index
        )
        if set(original_sample.index).issubset(set(transformed_df_index)):
            if isinstance(transformed_df, dd.DataFrame):
                transformed_subset = transformed_df.map_partitions(
                    lambda df: df.loc[df.index.isin(original_sample.index)]
                )
                transformed_sample = transformed_subset.compute()
            else:
                transformed_sample = transformed_df.loc[transformed_df_index]

            for col in common_cols:
                # Compare values in common columns
                if not pd.api.types.is_dtype_equal(
                    original_sample[col].dtype, transformed_sample[col].dtype
                ):
                    value_changes[col] = {
                        "changed": "N/A - incompatible dtypes",
                        "original_dtype": str(original_sample[col].dtype),
                        "transformed_dtype": str(transformed_sample[col].dtype),
                    }
                    continue

                changes = self._count_value_changes(
                    original_sample[col], transformed_sample[col]
                )
                value_changes[col] = changes

                # Count null changes
                null_changes[col] = self._count_null_changes(
                    original_sample[col], transformed_sample[col]
                )

        return value_changes, null_changes

    def _count_value_changes(
        self, original_col: pd.Series, transformed_col: pd.Series
    ) -> Dict[str, Any]:
        """
        Count the value changes between the original and transformed columns.

        Parameters:
        -----------
        original_col : pd.Series
            The original column values.
        transformed_col : pd.Series
            The transformed column values.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the number of changes and the percentage change.
        """
        if pd.api.types.is_numeric_dtype(original_col):
            changes = (
                ~np.isclose(
                    original_col.fillna(np.nan),
                    transformed_col.fillna(np.nan),
                    equal_nan=True,
                )
            ).sum()
        else:
            changes = (original_col != transformed_col).sum()

        return {
            "changed": int(changes),
            "percent_changed": (
                (changes / len(original_col) * 100) if len(original_col) > 0 else 0
            ),
        }

    def _count_null_changes(
        self, original_col: pd.Series, transformed_col: pd.Series
    ) -> Dict[str, Any]:
        """
        Count the null value changes between the original and transformed columns.

        Parameters:
        -----------
        original_col : pd.Series
            The original column values.
        transformed_col : pd.Series
            The transformed column values.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the number of null changes and the percentage change.
        """
        original_nulls = original_col.isna().sum()
        transformed_nulls = transformed_col.isna().sum()

        return {
            "original_nulls": int(original_nulls),
            "transformed_nulls": int(transformed_nulls),
            "difference": int(transformed_nulls - original_nulls),
            "percent_change": (
                ((transformed_nulls - original_nulls) / len(original_col) * 100)
                if len(original_col) > 0
                else 0
            ),
        }

    def _compare_memory_usage(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        transformed_df: Union[pd.DataFrame, dd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Compare the memory usage between the original and transformed DataFrames.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original DataFrame.
        transformed_df : Union[pd.DataFrame, dd.DataFrame]
            The transformed DataFrame.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing memory usage comparison.
        """
        original_memory = 0
        if isinstance(original_df, dd.DataFrame):
            original_memory = original_df.map_partitions(
                lambda df: df.memory_usage(deep=True).sum()
            ).compute()
            original_memory = original_memory.sum()
        else:
            original_memory = original_df.memory_usage(deep=True).sum()
        original_memory = original_memory / (1024 * 1024)  # MB

        transformed_memory = 0
        if isinstance(transformed_df, dd.DataFrame):
            transformed_memory = transformed_df.map_partitions(
                lambda df: df.memory_usage(deep=True).sum()
            ).compute()
            transformed_memory = transformed_memory.sum()
        else:
            transformed_memory = transformed_df.memory_usage(deep=True).sum()
        transformed_memory = transformed_memory / (1024 * 1024)  # MB

        return {
            "original_mb": round(original_memory, 2),
            "transformed_mb": round(transformed_memory, 2),
            "difference_mb": round(transformed_memory - original_memory, 2),
            "percent_change": (
                round((transformed_memory - original_memory) / original_memory * 100, 2)
                if original_memory > 0
                else float("inf")
            ),
        }

    def _save_metrics(
        self,
        metrics: Dict[str, Any],
        task_dir: Path,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: Optional[str] = None,
    ) -> WriterResult:
        """
        Save metrics.

        Parameters:
        -----------
        metrics : dict
            The metrics of operation
        task_dir : Path
            The task directory
        writer : DataWriter
            The writer to use for saving data
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        operation_timestamp : str, optional
            Timestamp of the operation, if any  (default: None)

        Returns:
        --------
        WriterResult
            Result object with path and metadata
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving metrics"})

        # Use the DataWriter to save
        metrics_filename = (
            f"{self.operation_name.lower()}_metrics_{operation_timestamp}"
        )

        metrics_result = writer.write_metrics(
            metrics=metrics,
            name=metrics_filename,
            timestamp_in_name=False,  # Already included in the filename
            encryption_key=self.encryption_key if self.use_encryption else None,
        )

        # Add metrics to result
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                result.add_metric(key, value)

        # Register metrics artifact
        result.add_artifact(
            artifact_type="json",
            path=metrics_result.path,
            description=f"Add/modify fields",
            category=Constants.Artifact_Category_Metrics,
        )

        # Report artifact
        if reporter:
            reporter.add_artifact(
                artifact_type="json",
                path=str(metrics_result.path),
                description=f"Add/modify fields metrics",
            )

        return metrics_result

    def _handle_visualizations(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        processed_df: Union[pd.DataFrame, dd.DataFrame],
        metrics: Dict[str, Any],
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
        vis_strict: bool,
        vis_timeout: int,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original data
        processed_df : Union[pd.DataFrame, dd.DataFrame]
            The processed data
        metrics : dict
            The metrics of operation
        task_dir : Path
            The task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        operation_timestamp : str, optional
            Timestamp of the operation, if any  (default: None)

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

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
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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

                    # Generate visualizations
                    visualization_paths = self._generate_visualizations(
                        original_df=original_df,
                        processed_df=processed_df,
                        metrics=metrics,
                        task_dir=task_dir,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend,
                        vis_strict=vis_strict,
                        progress_tracker=viz_progress,
                        operation_timestamp=operation_timestamp,
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
                name=f"VizThread-",
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
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization",
                )

        return visualization_paths

    def _generate_visualizations(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        processed_df: Union[pd.DataFrame, dd.DataFrame],
        metrics: Dict[str, Any],
        task_dir: Path,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
        vis_strict: bool,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate visualizations for the operation.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            The original data before processing
        processed_df : Union[pd.DataFrame, dd.DataFrame]
            The anonymized data after processing
        metrics : dict
            The metrics of operation
        task_dir : Path
            Task directory for saving visualizations
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        operation_timestamp : str, optional
            Timestamp of the operation, if any  (default: None)

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        from pamola_core.utils.visualization import create_bar_plot
        from pamola_core.transformations.commons.visualization_utils import (
            sample_large_dataset,
        )

        visualization_paths = {}
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for filenames
        if operation_timestamp is None:
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if visualization should be skipped
        if vis_backend is None:
            self.logger.info(f"Skipping visualization (backend=None)")
            return visualization_paths

        self.logger.info(f"[VIZ] Starting visualization generation")
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(
                    n=1, postfix={"step": "Preparing visualization data"}
                )

            # Sample large datasets for visualization
            original_rows = (
                int(original_df.map_partitions(len).sum().compute())
                if isinstance(original_df, dd.DataFrame)
                else len(original_df)
            )
            max_samples = 10000
            if original_rows > max_samples:
                self.logger.info(
                    f"[VIZ] Sampling large dataset: {original_rows} -> {max_samples} samples"
                )
                original_for_viz = original_df.sample(n=max_samples, random_state=42)
                processed_for_viz = processed_df.sample(n=max_samples, random_state=42)

            else:
                original_for_viz = original_df
                processed_for_viz = processed_df

            if isinstance(original_df, dd.DataFrame):
                original_for_viz = original_for_viz.compute()

            if isinstance(processed_df, dd.DataFrame):
                processed_for_viz = processed_for_viz.compute()

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_for_viz)} samples"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(n=2, postfix={"step": "Creating visualization"})

            # Fields Count Comparison Before/After
            viz_data = {
                "1.Before": len(original_for_viz.columns),
                "2.After": len(processed_for_viz.columns),
                "3.Modified": metrics["fields_modified_count"],
                "4.Added": metrics["fields_added_count"],
            }
            viz_path = (
                viz_dir
                / f"{self.operation_name.lower()}_fields_count_comparison_{operation_timestamp}.png"
            )
            viz_result = create_bar_plot(
                data=viz_data,
                output_path=viz_path,
                title="Fields Count Comparison",
                x_label="Fields Count",
                y_label="Value",
                sort_by="key",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )

            if viz_result.startswith("Error"):
                self.logger.error(f"Failed to create visualization: {viz_result}")
            else:
                visualization_paths[f"fields_count_comparison"] = viz_path

            # Distribution statistics for new/modified fields
            for field, distribution_statistic in metrics[
                "distribution_statistics"
            ].items():
                viz_data = distribution_statistic
                viz_path = (
                    viz_dir
                    / f"{self.operation_name.lower()}_distribution_statistic_{field.lower()}_{operation_timestamp}.png"
                )
                viz_result = create_bar_plot(
                    data=viz_data,
                    output_path=viz_path,
                    title=f"Distribution Statistics For '{field.upper()}'",
                    x_label="Statistic",
                    y_label="Value",
                    sort_by="key",
                    backend=vis_backend,
                    theme=vis_theme,
                    strict=vis_strict,
                    **kwargs,
                )

                if viz_result.startswith("Error"):
                    self.logger.error(f"Failed to create visualization: {viz_result}")
                else:
                    visualization_paths[f"distribution_statistic_{field.lower()}"] = (
                        viz_path
                    )

            # Correlation between original and modified fields
            viz_data = metrics["correlations"]
            viz_path = (
                viz_dir
                / f"{self.operation_name.lower()}_correlations_{operation_timestamp}.png"
            )
            viz_result = create_bar_plot(
                data=viz_data,
                output_path=viz_path,
                title=f"Correlations",
                x_label="Field",
                y_label="Correlation",
                sort_by="key",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if viz_result.startswith("Error"):
                self.logger.error(f"Failed to create visualization: {viz_result}")
            else:
                visualization_paths[f"correlations"] = viz_path

            self.logger.info(
                f"[VIZ] Visualization generation completed. Created {len(visualization_paths)} visualizations"
            )

            return visualization_paths

        except Exception as e:
            self.logger.error(
                f"[VIZ] Error in visualization generation: {type(e).__name__}: {e}"
            )
            self.logger.debug(f"[VIZ] Stack trace:", exc_info=True)

        return visualization_paths

    def _save_output_data(
        self,
        processed_df: Union[pd.DataFrame, dd.DataFrame],
        task_dir: Path,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> WriterResult:
        """
        Save the processed output data.

        Parameters:
        -----------
        processed_df : Union[pd.DataFrame, dd.DataFrame]
            The processed dataframe to save
        task_dir : Path
            The task directory
        writer : DataWriter
            The writer to use for saving data
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        operation_timestamp : str, optional
            Timestamp of the operation, if any  (default: None)

        Returns:
        --------
        WriterResult
            Result object with path and metadata
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        # Use the DataWriter to save
        output_filename = f"{self.operation_name.lower()}_output_{operation_timestamp}"

        encryption_mode = get_encryption_mode(processed_df, **kwargs)
        output_result = writer.write_dataframe(
            df=processed_df,
            name=output_filename,
            format="csv",
            subdir="output",
            timestamp_in_name=False,  # Already included in the filename
            encryption_key=self.encryption_key if self.use_encryption else None,
            encryption_mode=encryption_mode,
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type="csv",
            path=output_result.path,
            description=f"Add/modify fields",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_artifact(
                artifact_type="csv",
                path=str(output_result.path),
                description=f"Add/modify fields",
            )

        return output_result

    def _save_to_cache(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        processed_df: Union[pd.DataFrame, dd.DataFrame],
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : Union[pd.DataFrame, dd.DataFrame]
            Original input data
        task_dir : Path
            Task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            # Generate operation cache
            operation_cache_dir = OperationCache(cache_dir=task_dir / "cache")

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            original_df_len = (
                int(original_df.map_partitions(len).sum().compute())
                if isinstance(original_df, dd.DataFrame)
                else len(original_df)
            )
            processed_df_len = (
                int(processed_df.map_partitions(len).sum().compute())
                if isinstance(processed_df, dd.DataFrame)
                else len(processed_df)
            )
            cache_data = {
                "metrics": result.metrics,
                "artifacts": [artifact.to_dict() for artifact in result.artifacts],
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "data_info": {
                    "original_df_length": original_df_len,
                    "processed_df_length": processed_df_len,
                },
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache_dir.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _cleanup_memory(
        self,
        original_df: Optional[Union[pd.DataFrame, dd.DataFrame]],
        processed_df: Optional[Union[pd.DataFrame, dd.DataFrame]],
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        temporary attributes and forcing garbage collection.

        Parameters:
        -----------
        original_df : pd.DataFrame, optional
            Original data before processing
        processed_df : pd.DataFrame, optional
            Anonymized data after processing
        """
        # Clear argument references
        if original_df is not None:
            del original_df

        if processed_df is not None:
            del processed_df

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Force garbage collection
        import gc

        gc.collect()


# Helper function to create the operation easily
def create_add_modify_fields_operation(**kwargs) -> AddOrModifyFieldsOperation:
    """
    Create add or modify fields operation with default settings.

    Parameters:
    -----------
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    AddOrModifyFieldsOperation
        Configured add or modify fields operation
    """
    return AddOrModifyFieldsOperation(**kwargs)
