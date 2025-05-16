"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Base Anonymization Operation
Description: Base class for all anonymization operations
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides the base class for all anonymization operations,
defining common functionality, interface, and behavior.

Key features:
- Standardized operation lifecycle with validation, execution, and result handling
- Support for both in-place (REPLACE) and new field creation (ENRICH) modes
- Configurable null value handling strategies (PRESERVE, EXCLUDE, ERROR)
- Memory-efficient chunk-based processing for large datasets
- Comprehensive metrics collection and visualization generation
- Robust caching mechanism for operation results
- Progress tracking and reporting throughout the operation
- Secure output generation with optional encryption

TODO:
- Add support for distributed processing using Dask for very large datasets
- Implement delta metrics calculation for cached vs. fresh execution
- Support custom visualization options beyond default charts
- Add configuration option for metrics verbosity levels
- Implement optimization for batch sizes based on data characteristics
- Support resumable operations for very large datasets
- Add integration with privacy model validation
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import create_histogram, create_bar_plot

logger = logging.getLogger(__name__)


class AnonymizationOperation(BaseOperation):
    """
    Base class for all anonymization operations.

    This class provides common functionality for all anonymization operations,
    including data source handling, result processing, and metric generation.
    """

    def __init__(self,
                 field_name: str,
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 column_prefix: str = "_",
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,
                 description: str = "",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
        """
        Initialize the anonymization operation.

        Parameters:
        -----------
        field_name : str
            Field to anonymize
        mode : str, optional
            "REPLACE" to modify the field in-place, or "ENRICH" to create a new field (default: "REPLACE")
        output_field_name : str, optional
            Name for the output field if mode is "ENRICH" (default: None)
        column_prefix : str, optional
            Prefix for new column if mode is "ENRICH" (default: "_")
        null_strategy : str, optional
            How to handle NULL values: "PRESERVE", "EXCLUDE", or "ERROR" (default: "PRESERVE")
        batch_size : int, optional
            Batch size for processing large datasets (default: 10000)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        description : str, optional
            Operation description (default: "")
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        """
        # Use a default description if none provided
        if not description:
            description = f"Anonymization operation for field '{field_name}'"

        # Initialize base class
        super().__init__(
            name=f"{field_name}_anonymization",
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        # Store parameters
        self.field_name = field_name
        self.mode = mode.upper()  # Ensure uppercase
        self.output_field_name = output_field_name
        self.column_prefix = column_prefix
        self.null_strategy = null_strategy.upper()  # Ensure uppercase
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.version = getattr(self, "version", "1.0.0")  # Default version

        # Set up performance tracking variables
        self.start_time = None
        self.end_time = None
        self.process_count = 0

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the anonymization operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[ProgressTracker]
            Progress tracker for the operation (default: None)
        **kwargs : dict
            Additional parameters for the operation
            - force_recalculation: bool - Force operation even if cached results exist
            - encrypt_output: bool - Override encryption setting for outputs
            - use_dask: bool - Use Dask for large dataset processing
            - generate_visualization: bool - Create visualizations
            - include_timestamp: bool - Include timestamp in filenames
            - save_output: bool - Save processed data to output directory

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Start timing
        self.start_time = time.time()
        self.process_count = 0

        # Initialize result object
        result = OperationResult(status=OperationStatus.PENDING)

        # Save operation configuration
        self.save_config(task_dir)

        # Create writer for consistent output handling
        writer = DataWriter(
            task_dir=task_dir,
            logger=self.logger,
            progress_tracker=progress_tracker
        )

        # Set up progress tracking
        total_steps = 6  # Preparation, Data Loading, Validation, Processing, Metrics, Finalization
        if progress_tracker:
            progress_tracker.total = total_steps
            progress_tracker.update(0, {"step": "Preparation", "field": self.field_name})

        # Decompose kwargs and introduce variables for clarity
        is_encryption_required = kwargs.get('encrypt_output', False) or self.use_encryption
        use_dask = kwargs.get('use_dask', False)
        generate_visualization = kwargs.get('generate_visualization', True)
        include_timestamp_in_filenames = kwargs.get('include_timestamp', True)
        save_output = kwargs.get('save_output', True)
        force_recalculation = kwargs.get('force_recalculation', False)

        try:
            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not force_recalculation:
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Checking cache", "field": self.field_name})

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(data_source, task_dir)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(total_steps, {"step": "Complete (cached)", "field": self.field_name})

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Anonymization of {self.field_name} (from cache)",
                            details={"cached": True}
                        )

                    return cache_result

            # Step 2: Data Loading
            if progress_tracker:
                progress_tracker.update(1 if self.use_cache else 2, {"step": "Data Loading", "field": self.field_name})

            # Validate and get dataframe
            try:
                df = self._validate_and_get_dataframe(data_source)
                if df is None:
                    error_message = "Failed to load DataFrame from data source"
                    self.logger.error(error_message)
                    return OperationResult(status=OperationStatus.ERROR, error_message=error_message)
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 3: Validation
            if progress_tracker:
                progress_tracker.update(3, {"step": "Validation", "field": self.field_name})

            try:
                output_field = self._prepare_output_field(df)
                self._report_operation_details(reporter, output_field)
                original_data = df[self.field_name].copy()
            except Exception as e:
                error_message = f"Validation error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 4: Processing
            if progress_tracker:
                progress_tracker.update(4, {"step": "Processing", "field": self.field_name})

            try:
                result_df = self._process_dataframe(df, use_dask, progress_tracker)
                anonymized_data = result_df[output_field]
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 5: Metrics Calculation
            if progress_tracker:
                progress_tracker.update(5, {"step": "Metrics Calculation", "field": self.field_name})

            # Collect final metrics before using them
            self.end_time = time.time()

            # Initialize metrics in scope
            metrics = {}

            try:
                metrics = self._calculate_all_metrics(original_data, anonymized_data)

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=f"{self.field_name}_anonymization_metrics",
                    timestamp_in_name=True,
                    encryption_key=self.encryption_key if is_encryption_required else None
                )

                # Add metrics to result
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        result.add_metric(key, value)

                # Register metrics artifact
                result.add_artifact(
                    artifact_type="json",
                    path=metrics_result.path,
                    description=f"{self.field_name} anonymization metrics",
                    category="metrics"
                )

                # Report artifact
                if reporter:
                    reporter.add_artifact(
                        "json",
                        str(metrics_result.path),
                        f"{self.field_name} anonymization metrics"
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Finalization (Visualizations and Output Data)
            if progress_tracker:
                progress_tracker.update(6, {"step": "Finalization", "field": self.field_name})

            # Generate visualizations if required
            if generate_visualization:
                try:
                    self._handle_visualizations(
                        original_data, anonymized_data, task_dir,
                        result, reporter, writer, progress_tracker
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical

            # Save output data if required
            if save_output:
                try:
                    self._save_output_data(
                        result_df, task_dir, include_timestamp_in_filenames,
                        is_encryption_required, writer, result, reporter,
                        progress_tracker, **kwargs
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(result_df, original_data, anonymized_data, task_dir)
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Report completion
            if reporter:
                # Create the details dictionary with checks for all values
                details = {
                    "records_processed": self.process_count,
                    "execution_time": self.end_time - self.start_time if self.end_time and self.start_time else None
                }

                # Only add generalization_ratio if metrics exists and has this key
                if metrics and isinstance(metrics, dict):
                    generalization_ratio = metrics.get("generalization_ratio")
                    if generalization_ratio is not None:
                        details["generalization_ratio"] = generalization_ratio

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Anonymization of {self.field_name} completed",
                    details=details
                )

            # Cleanup memory
            self._cleanup_memory(result_df)

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error in anonymization operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

    def _validate_and_get_dataframe(self, data_source: DataSource) -> pd.DataFrame:
        """
        Validate data source and retrieve the main dataframe.

        Parameters:
        -----------
        data_source : DataSource
            The data source to validate

        Returns:
        --------
        pd.DataFrame
            The validated dataframe

        Raises:
        -------
        ValueError
            If no valid dataframe is found or the field is missing
        """
        # Get DataFrame from the data source
        df, error_info = data_source.get_dataframe("main")

        if df is None:
            error_message = f"Failed to load input data: {error_info['message'] if error_info else 'Unknown error'}"
            self.logger.error(error_message)
            raise ValueError(error_message)

        if self.field_name not in df.columns:
            error_message = f"Field {self.field_name} not found in DataFrame"
            self.logger.error(error_message)
            raise ValueError(error_message)

        return df

    def _prepare_output_field(self, df: pd.DataFrame) -> str:
        """
        Validate and generate the output field name.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to check field names against

        Returns:
        --------
        str
            The validated output field name
        """
        # Determine output field name based on mode
        if self.mode == "REPLACE":
            return self.field_name
        else:  # ENRICH mode
            if self.output_field_name:
                output_field = self.output_field_name
            else:
                output_field = f"{self.column_prefix}{self.field_name}"

            # Check if output field already exists in DataFrame
            if output_field in df.columns:
                self.logger.warning(f"Output field '{output_field}' already exists and will be overwritten")

            return output_field

    def _report_operation_details(self, reporter: Any, output_field: str) -> None:
        """
        Report details of the operation to the reporter.

        Parameters:
        -----------
        reporter : Any
            The reporter to log details to
        output_field : str
            The name of the output field
        """
        if reporter:
            reporter.add_operation(f"Anonymizing field: {self.field_name}", details={
                "field_name": self.field_name,
                "output_field": output_field,
                "mode": self.mode,
                "null_strategy": self.null_strategy,
                "operation_type": self.__class__.__name__
            })

    def _process_dataframe(self, df: pd.DataFrame, use_dask: bool,
                           progress_tracker: Optional[ProgressTracker]) -> pd.DataFrame:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        use_dask : bool
            Whether to use Dask for distributed processing
        progress_tracker : Optional[ProgressTracker]
            Optional progress tracker

        Returns:
        --------
        pd.DataFrame
            The processed dataframe
        """
        from pamola_core.anonymization.commons.processing_utils import process_in_chunks, process_dataframe_parallel

        # Check if dataframe is empty
        if len(df) == 0:
            self.logger.warning("Empty DataFrame provided, returning as is")
            return df

        # For small dataframes, process directly
        if len(df) <= self.batch_size:
            if progress_tracker:
                progress_tracker.update(0, {"step": "Processing small dataframe"})
            return self.process_batch(df)

        # For larger dataframes, check if we should use parallel processing
        if use_dask:
            try:
                self.logger.info(f"Using parallel processing with batch size {self.batch_size}")
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Setting up parallel processing"})

                return process_dataframe_parallel(
                    df=df,
                    process_function=self.process_batch,
                    n_jobs=-1,  # Use all available cores
                    batch_size=self.batch_size,
                    progress_tracker=progress_tracker
                )
            except Exception as e:
                self.logger.warning(f"Error in parallel processing: {e}, falling back to chunk processing")

        # Regular chunk processing
        self.logger.info(f"Processing in chunks with batch size {self.batch_size}")
        if progress_tracker:
            total_chunks = (len(df) + self.batch_size - 1) // self.batch_size
            progress_tracker.update(0, {"step": "Processing in chunks", "total_chunks": total_chunks})

        return process_in_chunks(
            df=df,
            process_function=self.process_batch,
            batch_size=self.batch_size,
            progress_tracker=progress_tracker
        )

    def _calculate_all_metrics(self, original_data: pd.Series, anonymized_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate all metrics for the operation.

        Parameters:
        -----------
        original_data : pd.Series
            The original data before anonymization
        anonymized_data : pd.Series
            The anonymized data after processing

        Returns:
        --------
        Dict[str, Any]
            A dictionary of calculated metrics
        """
        # Calculate basic metrics
        metrics = self._collect_metrics(original_data, anonymized_data)

        # Add performance metrics
        metrics.update({
            "execution_time_seconds": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "records_processed": self.process_count,
            "records_per_second": self.process_count / (self.end_time - self.start_time)
            if self.end_time and self.start_time and (self.end_time - self.start_time) > 0 else 0
        })

        return metrics

    def _handle_visualizations(self, original_data: pd.Series, anonymized_data: pd.Series,
                               task_dir: Path, result: OperationResult, reporter: Any,
                               writer: DataWriter, progress_tracker: Optional[ProgressTracker]) -> None:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        original_data : pd.Series
            The original data before anonymization
        anonymized_data : pd.Series
            The anonymized data after processing
        task_dir : Path
            The task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        writer : DataWriter
            The writer to use for saving visualizations
        progress_tracker : Optional[ProgressTracker]
            Optional progress tracker
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        # Generate visualizations
        visualization_paths = self._generate_visualizations(
            original_data, anonymized_data, task_dir, result, reporter
        )

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{self.field_name} {viz_type} visualization",
                category="visualization"
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    "png",
                    str(path),
                    f"{self.field_name} {viz_type} visualization"
                )

    def _save_output_data(self, result_df: pd.DataFrame, task_dir: Path,
                          include_timestamp: bool, encrypt_output: bool,
                          writer: DataWriter, result: OperationResult,
                          reporter: Any, progress_tracker: Optional[ProgressTracker],
                          **kwargs) -> None:
        """
        Save the processed output data.

        Parameters:
        -----------
        result_df : pd.DataFrame
            The processed dataframe to save
        task_dir : Path
            The task directory
        include_timestamp : bool
            Whether to include a timestamp in the filename
        encrypt_output : bool
            Whether to encrypt the output
        writer : DataWriter
            The writer to use for saving data
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[ProgressTracker]
            Optional progress tracker
        **kwargs : dict
            Additional parameters for the operation
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        # Use the DataWriter to save the DataFrame
        output_result = writer.write_dataframe(
            df=result_df,
            name=f"{self.field_name}_anonymized",
            format="csv",
            subdir="output",
            timestamp_in_name=include_timestamp,
            encryption_key=self.encryption_key if encrypt_output else None,
            **kwargs
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type="csv",
            path=output_result.path,
            description=f"{self.field_name} anonymized data",
            category="output"
        )

        # Report to reporter
        if reporter:
            reporter.add_artifact(
                "csv",
                str(output_result.path),
                f"{self.field_name} anonymized data"
            )

    def _cleanup_memory(self, result_df: Optional[pd.DataFrame] = None) -> None:
        """
        Clean up memory after operation completes.

        Parameters:
        -----------
        result_df : pd.DataFrame, optional
            The dataframe to delete from memory
        """
        if result_df is not None:
            del result_df
        if hasattr(self, '_temp_data'):
            del self._temp_data

        # Force garbage collection
        import gc
        gc.collect()

    def _generate_data_hash(self, data: pd.Series) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters:
        -----------
        data : pd.Series
            Input data for the operation

        Returns:
        --------
        str
            Hash string representing the data
        """
        import hashlib
        import json

        try:
            # Create dictionary of key data characteristics
            characteristics = {
                "length": len(data),
                "null_count": int(data.isna().sum()),
                "unique_count": int(data.nunique())
            }

            # Add data type-specific characteristics
            if pd.api.types.is_numeric_dtype(data):
                # For numeric data, add statistical measures
                non_null = data.dropna()
                if len(non_null) > 0:
                    characteristics.update({
                        "min": float(non_null.min()),
                        "max": float(non_null.max()),
                        "mean": float(non_null.mean()),
                        "median": float(non_null.median()),
                        "std": float(non_null.std())
                    })
            elif isinstance(data.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data):
                # For categorical data, sample most common values
                top_values = data.value_counts().head(10)
                characteristics["top_values"] = {str(k): int(v) for k, v in top_values.items()}

            # Convert to JSON string and hash
            json_str = json.dumps(characteristics, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception as e:
            logger.warning(f"Error generating data hash: {str(e)}")
            # Fallback to a simple hash of the data length and type
            return hashlib.md5(f"{len(data)}_{data.dtype}".encode()).hexdigest()

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        This method should be overridden by subclasses to provide
        operation-specific parameters for caching.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Base implementation returns minimal parameters
        return {}

    def _generate_cache_key(self, data: pd.Series) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        data : pd.Series
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get basic operation parameters
        parameters = {
            "field_name": self.field_name,
            "mode": self.mode,
            "null_strategy": self.null_strategy,
            "version": self.version
        }

        # Add operation-specific parameters through method that subclasses can override
        parameters.update(self._get_cache_parameters())

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(data)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash
        )

    def _check_cache(self, data_source: DataSource, task_dir: Path) -> Optional[OperationResult]:
        """
        Check if a cached result exists for this operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        task_dir : Path
            Task directory

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Get DataFrame from data source
            df, error_info = data_source.get_dataframe("main")
            if df is None or self.field_name not in df.columns:
                if error_info:
                    logger.warning(f"Cannot check cache: {error_info.get('message', 'Unknown error')}")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df[self.field_name])

            # Check for cached result
            logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            if cached_data:
                logger.info(f"Cache hit for {self.field_name} anonymization operation")

                # Create result object from cached data
                result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                # Add cache information to result
                result.add_metric("cached", True)
                result.add_metric("cache_key", cache_key)
                result.add_metric("cache_timestamp", cached_data.get("timestamp", "unknown"))

                return result

            logger.debug(f"No cache found for key: {cache_key}")
            return None

        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(self, result_df: pd.DataFrame, original_data: pd.Series,
                       anonymized_data: pd.Series, task_dir: Path) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        result_df : pd.DataFrame
            Processed DataFrame
        original_data : pd.Series
            Original input data
        anonymized_data : pd.Series
            Anonymized output data
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_data)

            # Prepare metadata for cache
            operation_params = self._get_cache_parameters()
            operation_params.update({
                "field_name": self.field_name,
                "mode": self.mode,
                "null_strategy": self.null_strategy,
                "operation_class": self.__class__.__name__,
                "version": self.version
            })

            # Collect metrics
            metrics = self._collect_metrics(original_data, anonymized_data)

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "parameters": operation_params,
                "data_info": {
                    "original_length": len(original_data),
                    "anonymized_length": len(anonymized_data),
                    "original_null_count": int(original_data.isna().sum()),
                    "anonymized_null_count": int(anonymized_data.isna().sum())
                }
            }

            # Save to cache
            logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
            )

            if success:
                logger.info(f"Successfully saved {self.field_name} anonymization results to cache")
            else:
                logger.warning(f"Failed to save {self.field_name} anonymization results to cache")

            return success

        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement process_batch method")

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
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement process_value method")

    def _collect_metrics(self, original_data: pd.Series, anonymized_data: pd.Series) -> Dict[str, Any]:
        """
        Collect metrics for the operation.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing

        Returns:
        --------
        Dict[str, Any]
            Dictionary with metrics
        """
        # Basic metrics that apply to all anonymization operations
        metrics: Dict[str, Any] = {
            "operation_type": self.__class__.__name__,
            "field_name": self.field_name,
            "mode": self.mode,
            "null_strategy": self.null_strategy,
            "total_records": len(original_data),
            "null_count": int(original_data.isna().sum()),
            "unique_values_before": int(original_data.nunique()),
            "unique_values_after": int(anonymized_data.nunique())
        }

        # If the field is numeric, add numeric metrics
        if pd.api.types.is_numeric_dtype(original_data):
            try:
                # Filter out nulls for calculations
                orig_non_null = original_data.dropna()
                anon_non_null = anonymized_data.dropna()

                # Handle case where anonymized values might be strings (e.g., in binning)
                try:
                    anon_numeric = pd.to_numeric(anon_non_null, errors='coerce')
                    anon_is_numeric = not anon_numeric.isna().all()
                except Exception:
                    anon_is_numeric = False
                    anon_numeric = None  # Add a default definition

                # Add additional metrics if anonymized data is still numeric
                if anon_is_numeric and anon_numeric is not None:
                    try:
                        # Compare statistical properties
                        metrics.update({
                            "mean_original": float(orig_non_null.mean()),
                            "mean_anonymized": float(anon_numeric.mean()),
                            "std_original": float(orig_non_null.std()),
                            "std_anonymized": float(anon_numeric.std()),
                            "min_original": float(orig_non_null.min()),
                            "min_anonymized": float(anon_numeric.min()),
                            "max_original": float(orig_non_null.max()),
                            "max_anonymized": float(anon_numeric.max()),
                            "median_original": float(orig_non_null.median()),
                            "median_anonymized": float(anon_numeric.median())
                        })

                        # Calculate mean absolute difference if possible
                        if orig_non_null.index.equals(anon_numeric.index):
                            metrics["mean_absolute_difference"] = float(
                                (orig_non_null - anon_numeric).abs().mean()
                            )
                    except Exception as e:
                        self.logger.warning(f"Could not compute all numeric comparison metrics: {e}")
            except Exception as e:
                self.logger.warning(f"Error calculating numeric metrics: {e}")

        # Calculate generalization ratio (reduction in unique values)
        unique_values_before = metrics.get("unique_values_before", 0)
        if unique_values_before > 0:
            unique_values_after = metrics.get("unique_values_after", 0)
            metrics["generalization_ratio"] = 1.0 - (unique_values_after / unique_values_before)
        else:
            metrics["generalization_ratio"] = 0.0

        # Add operation-specific metrics (to be implemented by subclasses)
        specific_metrics = self._collect_specific_metrics(original_data, anonymized_data)
        if specific_metrics:
            metrics.update(specific_metrics)

        return metrics

    def _collect_specific_metrics(self, original_data: pd.Series, anonymized_data: pd.Series) -> Dict[str, Any]:
        """
        Collect operation-specific metrics.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing

        Returns:
        --------
        Dict[str, Any]
            Dictionary with operation-specific metrics
        """
        # This method should be overridden by subclasses to add specific metrics
        return {}

    def _generate_visualizations(self,
                                 original_data: pd.Series,
                                 anonymized_data: pd.Series,
                                 task_dir: Path,
                                 result: OperationResult,
                                 reporter: Any) -> Dict[str, Path]:
        """
        Generate visualizations using the pamola_core visualization utilities.

        This is a base implementation that provides a basic distribution comparison.
        Subclasses should override to provide operation-specific visualizations.

        Parameters:
        -----------
        original_data : pd.Series
            Original data before anonymization
        anonymized_data : pd.Series
            Anonymized data after processing
        task_dir : Path
            Task directory for saving visualizations
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter object for tracking artifacts

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        visualization_paths = {}

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine if data is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(original_data)

        try:
            # Create a basic distribution comparison
            if is_numeric:
                # Clean data (drop nulls) for visualization
                original_clean = original_data.dropna()

                # Try to convert anonymized data to numeric if possible
                try:
                    anonymized_numeric = pd.to_numeric(anonymized_data.dropna(), errors='coerce')
                    has_numeric_data = not anonymized_numeric.isna().all()
                except:
                    has_numeric_data = False
                    anonymized_numeric = None

                if has_numeric_data and anonymized_numeric is not None:
                    # Determining the adaptive number of bins
                    n_bins = min(20, max(5, int(np.sqrt(len(original_clean)))))

                    # Create histogram comparison
                    operation_name = self.__class__.__name__
                    strategy_info = getattr(self, 'strategy', '')
                    hist_filename = f"{self.field_name}_{operation_name}_{strategy_info}_histogram_{timestamp}.png"
                    hist_path = task_dir / hist_filename

                    # Create histogram comparison
                    create_histogram(
                        data={
                            "Original": original_clean.tolist(),
                            "Anonymized": anonymized_numeric.dropna().tolist()
                        },
                        output_path=str(hist_path),
                        title=f"Distribution Comparison for {self.field_name}",
                        x_label=self.field_name,
                        y_label="Frequency",
                        bins=n_bins,
                        kde=True
                    )

                    visualization_paths["distribution_comparison"] = hist_path
            else:
                # For categorical data, create a bar chart
                # Get value distribution (top N categories)
                max_categories = 10
                orig_counts = original_data.value_counts().head(max_categories)
                anon_counts = anonymized_data.value_counts().head(max_categories)

                # Create bar chart
                operation_name = self.__class__.__name__
                strategy_info = getattr(self, 'strategy', '')
                bar_filename = f"{self.field_name}_{operation_name}_{strategy_info}_categories_{timestamp}.png"
                bar_path = task_dir / bar_filename

                # Create bar plot
                create_bar_plot(
                    data={
                        "Original": orig_counts.to_dict(),
                        "Anonymized": anon_counts.to_dict()
                    },
                    output_path=str(bar_path),
                    title=f"Category Comparison for {self.field_name}",
                    x_label="Category",
                    y_label="Count",
                    orientation="v",
                    sort_by="value",
                    max_items=max_categories
                )

                visualization_paths["category_comparison"] = bar_path

        except Exception as e:
            self.logger.warning(f"Error creating basic visualization: {str(e)}")

        return visualization_paths

    def _is_visualizable(self, data: pd.Series) -> bool:
        """
        Check if the data can be visualized.

        Parameters:
        -----------
        data : pd.Series
            Data to check

        Returns:
        --------
        bool
            Whether the data can be visualized
        """
        # Check if series is empty
        if len(data) == 0:
            return False

        # Check if all values are null
        if data.isnull().all():
            return False

        # If data is numeric or categorical, it can be visualized
        column_dtype = data.dtype
        is_categorical = isinstance(column_dtype, pd.CategoricalDtype)
        is_string = pd.api.types.is_string_dtype(data)
        is_object = pd.api.types.is_object_dtype(data)
        is_numeric = pd.api.types.is_numeric_dtype(data)

        if is_numeric or is_categorical or is_string or is_object:
            return True

        return False

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare directories for artifacts.

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

        # Ensure task directory exists
        task_dir.mkdir(parents=True, exist_ok=True)
        directories["root"] = task_dir

        # Create output directory
        output_dir = task_dir / "output"
        output_dir.mkdir(exist_ok=True)
        directories["output"] = output_dir

        # Create cache directory
        cache_dir = task_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        directories["cache"] = cache_dir

        # Create logs directory (needed to separate logs)
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        directories["logs"] = logs_dir

        return directories