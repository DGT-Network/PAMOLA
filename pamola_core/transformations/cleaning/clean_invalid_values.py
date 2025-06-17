"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Clean Invalid Values Operation
Description: Operation for nullify values that violate defined constraints.
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides an operation for nullify values that violate defined
constraints. It supports different constraint types for different data types
and whitelist/blacklist validation from external files.

Key features:
- Comprehensive metrics collection for privacy impact assessment
- Visualization generation for distribution comparisons
- Chunked processing support for large datasets
- Memory-efficient operation with explicit cleanup for large datasets

Implementation follows the PAMOLA.CORE operation framework with standardized
interfaces for input/output, progress tracking, and result reporting.
"""

import logging
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Union, Optional, Any
import json

import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_string_dtype,
    is_object_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype,
)

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.common.constants import Constants
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.transformations.base_transformation_op import TransformationOperation

# Configure module logger
logger = logging.getLogger(__name__)

class CleanInvalidValuesConfig(OperationConfig):
    """Configuration for Clean Invalid Values Operation."""

    schema = {
        "type": "object",
        "properties": {
            "field_constraints": {"type": ["object", "null"]},
            "whitelist_path": {"type": ["object", "null"]},
            "blacklist_path": {"type": ["object", "null"]},
            "null_replacement": {"type": ["string", "object", "null"]},

            "output_format": {"type": "string", "enum": ["csv", "json", "parquet"]},

            "name": {"type": "string"},
            "description": {"type": "string"},

            "field_name": {"type": "string"},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "output_field_name": {"type": ["string", "null"]},
            "column_prefix": {"type": "string"},

            "chunk_size": {"type": "integer"},
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": "integer"},
            "meta": {"type": ["object", "null"]},
            "use_vectorization": {"type": "boolean"},
            "parallel_processes": {"type": "integer"},

            "batch_size": {"type": "integer", "minimum": 1},
            "use_cache": {"type": "boolean"},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]}
        }
    }

@register(version="1.0.0")
class CleanInvalidValuesOperation(TransformationOperation):
    """
    Operation for nullify values that violate defined constraints.
    """

    def __init__(
            self,
            field_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
            whitelist_path: Optional[Dict[str, Path]] = None,
            blacklist_path: Optional[Dict[str, Path]] = None,
            null_replacement: Optional[Union[str, Dict[str, Any]]] = None,

            output_format: str = "csv",

            name: str = "clean_invalid_values_operation",
            description: str = "Clean values violating constraints",

            field_name: str = "",
            mode: str = "REPLACE",
            output_field_name: Optional[str] = None,
            column_prefix: str = "_",

            chunk_size: int = 10000,
            use_dask: bool = False,
            npartitions: int = 2,
            meta: Optional[Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple]] = None,
            use_vectorization: bool = False,
            parallel_processes: int = 2,

            batch_size: int = 10000,
            use_cache: bool = True,
            use_encryption: bool = False,
            encryption_key: Optional[Union[str, Path]] = None
    ):
        """
        Initialize operation.

        Parameters:
        -----------
        field_constraints : dict, optional
            Fields constraints
        whitelist_path : dict, optional
            Whitelist
        blacklist_path : dict, optional
            Blacklist
        null_replacement : str or dict, optional
            null replacement

        output_format : str
            Output format: "csv" or "json" or "parquet" (default: "csv")

        name : str
            Name of the operation (default: "clean_invalid_values_operation")
        description : str
            Operation description (default: "Clean values violating constraints")

        field_name : str
            Field name to transform (default: "")
        mode : str
            "REPLACE" to modify the field in-place, or "ENRICH" to create a new field (default: "REPLACE")
        output_field_name : str, optional
            Name for the output field if mode is "ENRICH" (default: None)
        column_prefix : str, optional
            Prefix for new column if mode is "ENRICH" (default: "_")

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

        batch_size : int
            Batch size for processing large datasets (default: 10000)
        use_cache : bool
            Whether to use operation caching (default: True)
        use_encryption : bool
            Whether to encrypt output files (default: False)
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        """
        # Create configuration and validate parameters
        config = CleanInvalidValuesConfig(
            field_constraints=field_constraints,
            whitelist_path=whitelist_path,
            blacklist_path=blacklist_path,
            null_replacement=null_replacement,

            output_format=output_format,

            name=name,
            description=description,

            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,

            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            meta=meta,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,

            batch_size=batch_size,
            use_cache=use_cache,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        # Initialize base class
        super().__init__(
            output_format=output_format,

            name=name,
            description=description,

            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            column_prefix=column_prefix,

            batch_size=batch_size,
            use_cache=use_cache,
            use_dask=use_dask,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        self.chunk_size = config.get("chunk_size")
        self.use_dask = config.get("use_dask")
        self.npartitions = config.get("npartitions")
        self.meta = config.get("meta")
        self.use_vectorization = config.get("use_vectorization")
        self.parallel_processes = config.get("parallel_processes")

        # Store parameters from validated config
        self.field_constraints = config.get("field_constraints")
        self.whitelist_path = config.get("whitelist_path")
        self.blacklist_path = config.get("blacklist_path")
        self.null_replacement = config.get("null_replacement")
        self.version = "1.0.0"  # Semantic versioning

        self.execution_time = 0
        self.include_timestamp = True
        self.is_encryption_required = False

        # Temp storage for cleanup
        self._temp_data = None

    def execute(
            self,
            data_source: DataSource,
            task_dir: Path,
            reporter: Any,
            progress_tracker: Optional[HierarchicalProgressTracker] = None,
            **kwargs
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
            Progress tracker for the operation (default: None)
        **kwargs : dict
            Additional parameters for the operation including:
            - dataset_name: str - Name of dataset - main
            - force_recalculation: bool - Force operation even if cached results exist - False
            - use_dask: bool - Use Dask for large dataset processing - False
            - parallel_processes: int - Number of parallel processes to use - 1
            - generate_visualization: bool - Create visualizations - True
            - include_timestamp: bool - Include timestamp in filenames - True
            - save_output: bool - Save processed data to output directory - True
            - encrypt_output: bool - Override encryption setting for outputs - False

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

            # Create DataWriter for consistent file operations
            writer = DataWriter(task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Save configuration to task directory
            self.save_config(task_dir)

            # Decompose kwargs and introduce variables for clarity
            dataset_name = kwargs.get("dataset_name", "main")
            force_recalculation = kwargs.get("force_recalculation", False)
            use_dask = kwargs.get("use_dask", self.use_dask)
            parallel_processes = kwargs.get("parallel_processes", 1)
            generate_visualization = kwargs.get("generate_visualization", True)
            include_timestamp = kwargs.get("include_timestamp", True)
            save_output = kwargs.get("save_output", True)
            is_encryption_required = (kwargs.get("encrypt_output", False) or self.use_encryption)
            encryption_key = kwargs.get('encryption_key', None)

            self.use_dask = use_dask
            self.parallel_processes = parallel_processes
            self.include_timestamp = include_timestamp
            self.is_encryption_required = is_encryption_required

            # Set up progress tracking
            # Preparation, Checking Cache, Data Loading, Validation, Processing, Metrics, Finalization
            total_steps = 5 + (1 if self.use_cache and not force_recalculation else 0)
            current_steps = 0
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(current_steps, {"step": "Preparation"})

            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not force_recalculation:
                if progress_tracker:
                    current_steps += 1
                    progress_tracker.update(current_steps, {"step": "Checking Cache"})

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(data_source, **kwargs)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(total_steps,{"step": "Complete (cached)"})

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Clean invalid values (from cache)",
                            details={"cached": True}
                        )
                    return cache_result

            # Step 2: Data Loading
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Data Loading"})

            # Get and validate data
            try:
                # Load data
                settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
                df = load_data_operation(data_source, dataset_name, **settings_operation)
                if df is None:
                    error_message = 'Failed to load input data'
                    self.logger.error(error_message)
                    return OperationResult(status=OperationStatus.ERROR, error_message=error_message)
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 3: Validation
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Validation"})

            try:
                if reporter:
                    reporter.add_operation(
                        f"Clean invalid values",
                        details={
                            "field_constraints": self.field_constraints,
                            "whitelist_path": self.whitelist_path,
                            "blacklist_path": self.blacklist_path,
                            "null_replacement": self.null_replacement,
                            "operation_type": self.__class__.__name__
                        }
                    )

                # Get a copy of the original data for metrics calculation
                original_df = df.copy(deep=True)

                # Validation
            except Exception as e:
                error_message = f"Validation error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 4: Processing
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Processing"})

            try:
                processed_df = self._process_dataframe(df, progress_tracker)
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

            # Step 5: Metrics
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Metrics"})

            # Initialize metrics in scope
            metrics = {}
            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            try:
                metrics = self._calculate_all_metrics(original_df, processed_df)

                self._save_metrics(
                    metrics=metrics,
                    task_dir=task_dir,
                    writer=writer,
                    result=result,
                    reporter=reporter,
                    progress_tracker=progress_tracker
                )
            except Exception as e:
                    error_message = f"Error calculating metrics: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - metrics failure is not critical

            # Step 6: Finalization (Visualizations and Output Data)
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Finalization"})

            # Generate visualizations if required
            if generate_visualization:
                try:
                    kwargs_encryption = {
                        "use_encryption": kwargs.get('use_encryption', False),
                        "encryption_key": encryption_key
                    }
                    self._handle_visualizations(
                        original_df=original_df,
                        processed_df=processed_df,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        progress_tracker=progress_tracker,
                        **kwargs_encryption
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical

            # Save output data if required
            if save_output:
                try:
                    self._save_output_data(
                        processed_df=processed_df,
                        task_dir=task_dir,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=progress_tracker
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(status=OperationStatus.ERROR,error_message=error_message)

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_df=original_df,
                        processed_df=processed_df,
                        metrics=metrics,
                        task_dir=task_dir
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
                    "records_per_second": self.process_count / self.execution_time
                    if self.execution_time > 0 else 0
                }

                # Only add generalization_ratio if metrics exists and has this key
                if metrics and isinstance(metrics, dict):
                    generalization_ratio = metrics.get("generalization_ratio")
                    if generalization_ratio is not None:
                        details["generalization_ratio"] = generalization_ratio

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Clean invalid values completed",
                    details=details
                )

            # Cleanup memory
            self._cleanup_memory(original_df, processed_df)

            # Set success status
            result.status = OperationStatus.SUCCESS

            return result
        except Exception as e:
            # Handle unexpected errors
            error_message = f"Error in clean invalid values operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(status=OperationStatus.ERROR, error_message=error_message)

    def process_batch(
            self,
            batch: pd.DataFrame
    ) -> pd.DataFrame:
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
        batch_columns = batch.columns

        if self.field_constraints is None:
            self.field_constraints = {}
        for field_name, field_config in self.field_constraints.items():
            constraint_type = field_config.get("constraint_type")

            if field_name and field_name in batch_columns:
                # Config output field name
                output_field_name = field_name
                if self.mode == "ENRICH" and self.column_prefix:
                    output_field_name = f"{self.column_prefix}{field_name}"
                    if output_field_name not in batch.columns:
                        batch[output_field_name] = batch[field_name]

                if is_numeric_dtype(batch[output_field_name]):
                    min_value = field_config.get("min_value")
                    max_value = field_config.get("max_value")

                    # Clean with min_value
                    if constraint_type == "min_value":
                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] >= min_value
                        )

                    # Clean with max_value
                    if constraint_type == "max_value":
                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] <= max_value
                        )

                    # Clean with valid_range
                    if constraint_type == "valid_range":
                        batch[output_field_name] = batch[output_field_name].where(
                            (batch[output_field_name] >= min_value)
                            & (batch[output_field_name] <= max_value)
                        )

                    # Clean with custom_function
                    if constraint_type == "custom_function":
                        raise NotImplementedError("Not implement")

                if isinstance(batch[output_field_name].dtype, CategoricalDtype):
                    # Clean with allowed_values
                    if constraint_type == "allowed_values":
                        allowed_values = field_config.get("allowed_values")

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name].isin(allowed_values)
                        )

                    # Clean with disallowed_values
                    if constraint_type == "disallowed_values":
                        disallowed_values = field_config.get("disallowed_values")

                        batch[output_field_name] = batch[output_field_name].where(
                            ~batch[output_field_name].isin(disallowed_values)
                        )

                    # Clean with whitelist_file
                    if constraint_type == "whitelist_file":
                        whitelist_file = field_config.get("whitelist_file")
                        with open(whitelist_file) as f:
                            allowed_values = set(map(str.strip, f))

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name].isin(allowed_values)
                        )

                    # Clean with blacklist_file
                    if constraint_type == "blacklist_file":
                        blacklist_file = field_config.get("blacklist_file")
                        with open(blacklist_file) as f:
                            disallowed_values = set(map(str.strip, f))

                        batch[output_field_name] = batch[output_field_name].where(
                            ~batch[output_field_name].isin(disallowed_values)
                        )

                    # Clean with pattern
                    if constraint_type == "pattern":
                        pattern = field_config.get("pattern")

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name].astype(str).str.contains(pat=pattern,na=False,regex=True)
                        )

                if is_datetime64_any_dtype(batch[output_field_name]):
                    min_date = field_config.get("min_date")
                    max_date = field_config.get("max_date")

                    # Clean with min_date
                    if constraint_type == "min_date":
                        batch[output_field_name] = pd.to_datetime(batch[output_field_name], errors='coerce')

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] >= pd.to_datetime(min_date)
                        )

                    # Clean with max_date
                    if constraint_type == "max_date":
                        batch[output_field_name] = pd.to_datetime(batch[output_field_name], errors='coerce')

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] <= pd.to_datetime(max_date)
                        )

                    # Clean with date_range
                    if constraint_type == "date_range":
                        batch[output_field_name] = pd.to_datetime(batch[output_field_name], errors='coerce')

                        batch[output_field_name] = batch[output_field_name].where(
                            (batch[output_field_name] >= pd.to_datetime(min_date))
                            & (batch[output_field_name] <= pd.to_datetime(max_date))
                        )

                    # Clean with valid_format
                    if constraint_type == "valid_format":
                        valid_format = field_config.get("valid_format")

                        batch[output_field_name] = pd.to_datetime(
                            batch[output_field_name],
                            format=valid_format,
                            errors='coerce'
                        )

                if is_string_dtype(batch[output_field_name]) or is_object_dtype(batch[output_field_name]):
                    # Clean with min_length
                    if constraint_type == "min_length":
                        min_length = field_config.get("min_length")

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name].astype(str).str.len() >= min_length
                        )

                    # Clean with max_length
                    if constraint_type == "max_length":
                        max_length = field_config.get("max_length")

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name].astype(str).str.len() <= max_length
                        )

                    # Clean with valid_pattern
                    if constraint_type == "valid_pattern":
                        valid_pattern = field_config.get("valid_pattern")

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name].astype(str).str.contains(pat=valid_pattern,na=False,regex=True)
                        )

                    # Clean with valid_format
                    if constraint_type == "valid_format":
                        valid_format = field_config.get("valid_format")

                        try:
                            batch[output_field_name] = pd.to_datetime(
                                batch[output_field_name],
                                format=valid_format,
                                errors='coerce'
                            )
                        except Exception:
                            batch[output_field_name][:] = pd.NA  # fallback if format fails entirely

        if self.whitelist_path is None:
            self.whitelist_path = {}
        for field_name, whitelist_path in self.whitelist_path.items():
            if field_name and field_name in batch_columns:
                # Config output field name
                output_field_name = field_name
                if self.mode == "ENRICH" and self.column_prefix:
                    output_field_name = f"{self.column_prefix}{field_name}"
                    if output_field_name not in batch.columns:
                        batch[output_field_name] = batch[field_name]

                with open(whitelist_path) as f:
                    allowed_values = set(map(str.strip, f))

                batch[output_field_name] = batch[output_field_name].where(
                    batch[output_field_name].isin(allowed_values)
                )

        if self.blacklist_path is None:
            self.blacklist_path = {}
        for field_name, blacklist_path in self.blacklist_path.items():
            if field_name and field_name in batch_columns:
                # Config output field name
                output_field_name = field_name
                if self.mode == "ENRICH" and self.column_prefix:
                    output_field_name = f"{self.column_prefix}{field_name}"
                    if output_field_name not in batch.columns:
                        batch[output_field_name] = batch[field_name]

                with open(blacklist_path) as f:
                    disallowed_values = set(map(str.strip, f))

                batch[output_field_name] = batch[output_field_name].where(
                    ~batch[output_field_name].isin(disallowed_values)
                )

        if self.null_replacement is not None:
            for column in batch_columns:
                if isinstance(self.null_replacement, str):
                    strategy = self.null_replacement
                else:
                    strategy = self.null_replacement.get(column)

                if strategy is None:
                    continue

                # Config output field name
                output_column = column
                if self.mode == "ENRICH" and self.column_prefix:
                    output_column = f"{self.column_prefix}{column}"
                    if output_column not in batch.columns:
                        batch[output_column] = batch[column]

                if strategy == "mean" and is_numeric_dtype(batch[output_column]):
                    batch[output_column] = batch[output_column].fillna(
                        batch[output_column].mean()
                    )
                elif strategy == "median" and is_numeric_dtype(batch[output_column]):
                    batch[output_column] = batch[output_column].fillna(
                        batch[output_column].median()
                    )
                elif strategy == "mode":
                    batch[output_column] = batch[output_column].fillna(
                        batch[output_column].mode()[0]
                    )
                elif strategy == "random_sample":
                    choice_values = batch[output_column].dropna().values
                    num_missing = batch[output_column].isna().sum()

                    batch.loc[batch[output_column].isna(), output_column] = np.random.choice(
                        choice_values, size=num_missing
                    )
                elif isinstance(strategy, (str, int, float, pd.Timestamp)):
                    batch[output_column] = batch[output_column].fillna(strategy)

        processed_batch = batch

        return processed_batch

    def process_value(
            self,
            value,
            **params
    ):
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

    def _prepare_directories(
            self,
            task_dir: Path
    ) -> Dict[str, Path]:
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

        # Create standard directories
        directories["root"] = task_dir
        directories["output"] = task_dir / "output"
        directories["cache"] = task_dir / "cache"
        directories["logs"] = task_dir / "logs"
        directories["dictionaries"] = task_dir / "dictionaries"

        # Ensure all directories exist
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

    def _check_cache(
            self,
            data_source: DataSource,
            **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        task_dir : Path
            Task directory
        dataset_name: str
            Dataset name

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
            # Load data
            dataset_name = kwargs.get('dataset_name', "main")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                error_message = 'Failed to load input data'
                self.logger.warning(f"Cannot check cache: {error_message}")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            cached_result.add_metric(key, value)

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric("cache_timestamp", cached_data.get("timestamp", "unknown"))

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _generate_cache_key(
            self,
            df: pd.DataFrame
    ) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get operation parameters
        parameters = self._get_operation_parameters()

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(df)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash
        )

    def _get_operation_parameters(
            self
    ) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "field_constraints": self.field_constraints,
            "whitelist_path": self.whitelist_path,
            "blacklist_path": self.blacklist_path,
            "null_replacement": self.null_replacement,
            "version": self.version
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(
            self
    ) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

    def _generate_data_hash(
            self,
            df: pd.DataFrame
    ) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Hash string representing the data
        """
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format='iso')
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type         
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _process_dataframe(
            self,
            df: pd.DataFrame,
            progress_tracker: Optional[HierarchicalProgressTracker]
    ) -> pd.DataFrame:
        """
        Handle processing of the dataframe, including chunk-wise or full processing.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        pd.DataFrame
            The processed dataframe
        """
        from pamola_core.transformations.commons.processing_utils import (
            process_dataframe_with_config
        )

        processed_df = process_dataframe_with_config(
            df=df,
            process_function=self.process_batch,
            chunk_size = self.chunk_size,
            use_dask = self.use_dask,
            npartitions = self.npartitions,
            meta = self.meta,
            use_vectorization = self.use_vectorization,
            parallel_processes = self.parallel_processes,
            progress_tracker = progress_tracker
        )

        return processed_df

    def _calculate_all_metrics(
            self,
            original_df: pd.DataFrame,
            processed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for operation.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original data
        processed_df : pd.DataFrame
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
                "records_per_second": self.process_count / self.execution_time
                if self.execution_time > 0 else 0
            }
        )

        return metrics

    def _collect_metrics(
            self,
            original_df: pd.DataFrame,
            processed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Collect metrics for the operation.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original data
        processed_df : pd.DataFrame
            The processed data

        Returns:
        --------
        Dict[str, Any]
            A dictionary of calculated metrics
        """
        from pamola_core.transformations.commons.metric_utils import (
            calculate_dataset_comparison,
            calculate_transformation_impact
        )

        # Basic metrics
        metrics: Dict[str, Any] = {
            "operation_type": self.__class__.__name__,
            "field_constraints": self.field_constraints,
            "whitelist_path": self.whitelist_path,
            "blacklist_path": self.blacklist_path,
            "null_replacement": self.null_replacement
        }

        # Add metrics
        metrics.update(calculate_dataset_comparison(original_df,processed_df))
        metrics.update(calculate_transformation_impact(original_df, processed_df))

        return metrics

    def _save_metrics(
            self,
            metrics: Dict[str, Any],
            task_dir: Path,
            writer: DataWriter,
            result: OperationResult,
            reporter: Any,
            progress_tracker: Optional[HierarchicalProgressTracker]
    ) -> None:
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
        """
        from pamola_core.transformations.commons.visualization_utils import (
            generate_visualization_filename
        )

        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving metrics"})

        # Use the DataWriter to save
        metrics_filename = generate_visualization_filename(
            operation_name=f"{self.__class__.__name__}",
            visualization_type="metrics",
            extension="json",
            include_timestamp=self.include_timestamp
        )

        metrics_result = writer.write_metrics(
            metrics=metrics,
            name=metrics_filename.replace(".json", ""),  # writer appends .json
            timestamp_in_name=False,  # Already included in the filename
            encryption_key=self.encryption_key if self.is_encryption_required else None
        )

        # Add metrics to result
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                result.add_metric(key, value)

        # Register metrics artifact
        result.add_artifact(
            artifact_type="json",
            path=metrics_result.path,
            description=f"Clean invalid values",
            category=Constants.Artifact_Category_Metrics
        )

        # Report artifact
        if reporter:
            reporter.add_artifact(
                artifact_type="json",
                path=str(metrics_result.path),
                description=f"Clean invalid values metrics"
            )

    def _handle_visualizations(
            self,
            original_df: pd.DataFrame,
            processed_df: pd.DataFrame,
            task_dir: Path,
            result: OperationResult,
            reporter: Any,
            progress_tracker: Optional[HierarchicalProgressTracker],
            **kwargs
    ) -> None:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original data
        processed_df : pd.DataFrame
            The processed data
        task_dir : Path
            The task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        # Generate visualizations
        visualization_paths = self._generate_visualizations(
            original_df=original_df,
            processed_df=processed_df,
            task_dir=task_dir,
            **kwargs
        )

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization"
                )

    def _generate_visualizations(
            self,
            original_df: pd.DataFrame,
            processed_df: pd.DataFrame,
            task_dir: Path,
            **kwargs
    ) -> Dict[str, Path]:
        """
        Generate visualizations for the operation.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original data before processing
        processed_df : pd.DataFrame
            The anonymized data after processing
        task_dir : Path
            Task directory for saving visualizations

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        from pamola_core.transformations.commons.visualization_utils import (
            generate_field_count_comparison_vis
        )

        visualization_paths = {}

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        visualization_paths.update(
            generate_field_count_comparison_vis(
                original_df=original_df,
                transformed_df=processed_df,
                field_label="Clean invalid values",
                operation_name=f"{self.__class__.__name__}",
                task_dir=task_dir,
                timestamp=timestamp,
                visualization_paths=visualization_paths,
                **kwargs
            )
        )

        return visualization_paths

    def _save_output_data(
            self,
            processed_df: pd.DataFrame,
            task_dir: Path,
            writer: DataWriter,
            result: OperationResult,
            reporter: Any,
            progress_tracker: Optional[HierarchicalProgressTracker]
    ) -> None:
        """
        Save the processed output data.

        Parameters:
        -----------
        processed_df : pd.DataFrame
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
        """
        from pamola_core.transformations.commons.visualization_utils import (
            generate_visualization_filename
        )

        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        # Use the DataWriter to save
        output_filename = generate_visualization_filename(
            operation_name=f"{self.__class__.__name__}",
            visualization_type="generalized",
            extension="csv",
            include_timestamp=self.include_timestamp
        )

        output_result = writer.write_dataframe(
            df=processed_df,
            name=output_filename.replace(".csv", ""),  # writer appends .csv
            format="csv",
            subdir="output",
            timestamp_in_name=False,  # Already included in the filename
            encryption_key=self.encryption_key if self.is_encryption_required else None
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type="csv",
            path=output_result.path,
            description=f"Clean invalid values",
            category=Constants.Artifact_Category_Output
        )

        # Report to reporter
        if reporter:
            reporter.add_artifact(
                artifact_type="csv",
                path=str(output_result.path),
                description=f"Clean invalid values"
            )

    def _save_to_cache(
            self,
            original_df: pd.DataFrame,
            processed_df: pd.DataFrame,
            metrics: Dict[str, Any],
            task_dir: Path
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        processed_df : pd.DataFrame
            Processed DataFrame
        metrics : dict
            The metrics of operation
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
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "metrics": metrics,
                "data_info": {
                    "original_df_length": len(original_df),
                    "processed_df_length": len(processed_df)
                }
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
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
            original_df: Optional[pd.DataFrame],
            processed_df: Optional[pd.DataFrame]
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

        # Clear instance attribute references
        if hasattr(self, "_temp_data") and self._temp_data is not None:
            del self._temp_data
            self._temp_data = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Force garbage collection
        import gc
        gc.collect()

# Helper function to create the operation easily
def create_clean_invalid_values_operation(
        **kwargs
) -> CleanInvalidValuesOperation:
    """
    Create clean invalid values operation with default settings.

    Parameters:
    -----------
    **kwargs : dict
        Additional parameters to override defaults

    Returns:
    --------
    CleanInvalidValuesOperation
        Configured clean invalid values operation
    """
    return CleanInvalidValuesOperation(**kwargs)
