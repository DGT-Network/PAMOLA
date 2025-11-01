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
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, Optional, Any
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

from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter, WriterResult
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.common.constants import Constants
from pamola_core.utils.io import load_data_operation, load_settings_operation
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.transformations.schemas.clean_invalid_values_config import CleanInvalidValuesOperationConfig
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

# Configure module logger
logger = logging.getLogger(__name__)


@register(version="1.0.0")
class CleanInvalidValuesOperation(TransformationOperation):
    """
    Operation for nullify values that violate defined constraints.
    """

    def __init__(
        self,
        name: str = "clean_invalid_values_operation",
        field_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        whitelist_path: Optional[Dict[str, Path]] = None,
        blacklist_path: Optional[Dict[str, Path]] = None,
        null_replacement: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Initialize the CleanInvalidValuesOperation.

        Parameters:
        -----------
        name : str
            Name of the operation (default: "clean_invalid_values_operation")
        field_constraints : dict, optional
            Fields constraints
        whitelist_path : dict, optional
            Whitelist
        blacklist_path : dict, optional
            Blacklist
        null_replacement : str or dict, optional
            null replacement
        **kwargs: dict
            Additional keyword arguments passed to TransformationOperation.
        """
        # Ensure default metadata
        kwargs.setdefault("name", name)
        kwargs.setdefault(
            "description",
            f"Clean invalid values based on defined constraints.",
        )

        # --- Build config object ---
        config = CleanInvalidValuesOperationConfig(
            field_constraints=field_constraints,
            whitelist_path=whitelist_path,
            blacklist_path=blacklist_path,
            null_replacement=null_replacement,
            **kwargs,
        )

        # Inject config into kwargs
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

            # Config logger task for operation
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

            # Save configuration to task directory
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up progress tracking
            # Preparation, Checking Cache, Data Loading, Validation, Processing, Metrics, Finalization
            total_steps = 5 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            current_steps = 0
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(current_steps, {"step": "Preparation"})

            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    current_steps += 1
                    progress_tracker.update(current_steps, {"step": "Checking Cache"})

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(data_source, reporter, **kwargs)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(
                            total_steps, {"step": "Complete (cached)"}
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Clean invalid values (from cache)",
                            details={"cached": True},
                        )
                    return cache_result

            # Step 2: Data Loading
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Data Loading"})

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
                            "operation_type": self.operation_name,
                        },
                    )

                # Get a copy of the original data for metrics calculation
                original_df = df.copy(deep=True)

                # Validation
            except Exception as e:
                error_message = f"Validation error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 4: Processing
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Processing"})

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
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Metrics"})

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
                    task_dir=task_dir,
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
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Finalization"})

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
                        task_dir=task_dir,
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
                        metrics=metrics,
                        metrics_result=metrics_result,
                        output_result=output_result,
                        visualizations=visualizations,
                        task_dir=task_dir,
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
                reporter.add_operation(
                    f"Clean invalid values completed", details=details
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
        batch_columns = batch.columns

        if self.field_constraints is None:
            self.field_constraints = {}
        for field_name, field_config in self.field_constraints.items():
            data_type = field_config.get("data_type")
            constraint_type = field_config.get("constraint_type")

            if field_name and field_name in batch_columns:
                # Config output field name
                output_field_name = field_name
                if self.mode == "ENRICH" and self.column_prefix:
                    output_field_name = f"{self.column_prefix}{field_name}"
                    if output_field_name not in batch.columns:
                        batch[output_field_name] = batch[field_name]

                if is_numeric_dtype(batch[output_field_name]) or data_type == "numeric":
                    try:
                        min_value = float(field_config.get("min_value"))
                    except (ValueError, TypeError):
                        min_value = None

                    try:
                        max_value = float(field_config.get("max_value"))
                    except (ValueError, TypeError):
                        max_value = None

                    # Clean with min_value
                    if constraint_type == "min_value" and min_value is not None:
                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] >= min_value
                        )

                    # Clean with max_value
                    if constraint_type == "max_value" and max_value is not None:
                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] <= max_value
                        )

                    # Clean with valid_range
                    if (
                        constraint_type == "valid_range"
                        and min_value is not None
                        and max_value is not None
                    ):
                        batch[output_field_name] = batch[output_field_name].where(
                            (batch[output_field_name] >= min_value)
                            & (batch[output_field_name] <= max_value)
                        )

                    # Clean with custom_function
                    if constraint_type == "custom_function":
                        raise NotImplementedError("Not implement")

                if (
                    isinstance(batch[output_field_name].dtype, CategoricalDtype)
                    or data_type == "categorical"
                ):
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
                            batch[output_field_name]
                            .astype(str)
                            .str.contains(pat=pattern, na=False, regex=True)
                        )

                if (
                    is_datetime64_any_dtype(batch[output_field_name])
                    or data_type == "date"
                ):
                    min_date = field_config.get("min_date")
                    max_date = field_config.get("max_date")

                    # Clean with min_date
                    if constraint_type == "min_date":
                        batch[output_field_name] = pd.to_datetime(
                            batch[output_field_name], errors="coerce"
                        )

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] >= pd.to_datetime(min_date)
                        )

                    # Clean with max_date
                    if constraint_type == "max_date":
                        batch[output_field_name] = pd.to_datetime(
                            batch[output_field_name], errors="coerce"
                        )

                        batch[output_field_name] = batch[output_field_name].where(
                            batch[output_field_name] <= pd.to_datetime(max_date)
                        )

                    # Clean with date_range
                    if constraint_type == "date_range":
                        batch[output_field_name] = pd.to_datetime(
                            batch[output_field_name], errors="coerce"
                        )

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
                            errors="coerce",
                        )

                if (
                    is_string_dtype(batch[output_field_name])
                    or is_object_dtype(batch[output_field_name])
                    or data_type == "text"
                ):
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
                            batch[output_field_name]
                            .astype(str)
                            .str.contains(pat=valid_pattern, na=False, regex=True)
                        )

                    # Clean with valid_format
                    if constraint_type == "valid_format":
                        valid_format = field_config.get("valid_format")

                        try:
                            batch[output_field_name] = pd.to_datetime(
                                batch[output_field_name],
                                format=valid_format,
                                errors="coerce",
                            )
                        except Exception:
                            batch[output_field_name][
                                :
                            ] = pd.NA  # fallback if format fails entirely

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

                    batch.loc[batch[output_column].isna(), output_column] = (
                        np.random.choice(choice_values, size=num_missing)
                    )
                elif isinstance(strategy, (str, int, float, pd.Timestamp)):
                    try:
                        # Attempt to cast strategy to column dtype
                        column_dtype = batch[output_column].dtype
                        casted_strategy = column_dtype.type(strategy)
                        batch[output_column] = batch[output_column].fillna(
                            casted_strategy
                        )
                    except Exception:
                        pass  # Do nothing on failure

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
        self, data_source: DataSource, reporter: Any, **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        reporter : Any
            The reporter to log artifacts to
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
            dataset_name = kwargs.get("dataset_name", "main")
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                error_message = "Failed to load input data"
                self.logger.warning(f"Cannot check cache: {error_message}")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
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

                # Restore artifacts from cache
                artifacts_restored = 0

                # Add metrics artifact if exists
                metrics_result_path = cached_data.get("metrics_result_path")
                if metrics_result_path:
                    metrics_path = Path(metrics_result_path)
                    if metrics_path.exists():
                        cached_result.add_artifact(
                            artifact_type="json",
                            path=metrics_path,
                            description=f"Generalization metrics (cached)",
                            category=Constants.Artifact_Category_Metrics,
                        )
                        artifacts_restored += 1

                        if reporter:
                            reporter.add_operation(
                                f"Generalization metrics (cached)",
                                details={
                                    "artifact_type": "json",
                                    "path": str(metrics_path),
                                },
                            )

                # Add output artifact if file exists
                output_result_path = cached_data.get("output_result_path")
                if output_result_path:
                    output_path = Path(output_result_path)
                    if output_path.exists():
                        cached_result.add_artifact(
                            artifact_type=self.output_format,
                            path=output_path,
                            description=f"Generalized data (cached)",
                            category=Constants.Artifact_Category_Output,
                        )
                        artifacts_restored += 1

                        # Also report to reporter
                        if reporter:
                            reporter.add_operation(
                                f"Generalized data (cached)",
                                details={
                                    "artifact_type": self.output_format,
                                    "path": str(output_path),
                                },
                            )
                    else:
                        self.logger.warning(
                            f"Cached output file not found: {output_path}"
                        )

                # Add visualization artifacts
                visualizations = cached_data.get("visualizations", {})
                for viz_type, viz_path in visualizations.items():
                    path = Path(viz_path)
                    if path.exists():
                        cached_result.add_artifact(
                            artifact_type="png",
                            path=path,
                            description=f"{viz_type} visualization (cached)",
                            category=Constants.Artifact_Category_Visualization,
                        )
                        artifacts_restored += 1

                        if reporter:
                            reporter.add_operation(
                                f"{viz_type} visualization (cached)",
                                details={"artifact_type": "png", "path": str(path)},
                            )

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )
                cached_result.add_metric("artifacts_restored", artifacts_restored)

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
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
            operation_name=self.operation_name,
            parameters=parameters,
            data_hash=data_hash,
        )

    def _get_operation_parameters(self) -> Dict[str, Any]:
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
            "version": self.version,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
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
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _process_dataframe(
        self, df: pd.DataFrame, progress_tracker: Optional[HierarchicalProgressTracker]
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
            process_dataframe_with_config,
        )

        processed_df = process_dataframe_with_config(
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

    def _calculate_all_metrics(
        self, original_df: pd.DataFrame, processed_df: pd.DataFrame
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
                "records_per_second": (
                    self.process_count / self.execution_time
                    if self.execution_time > 0
                    else 0
                ),
            }
        )

        return metrics

    def _collect_metrics(
        self, original_df: pd.DataFrame, processed_df: pd.DataFrame
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
        # Basic metrics
        metrics: Dict[str, Any] = {}

        # Specific metrics
        fields_added = [
            col for col in processed_df.columns if col not in original_df.columns
        ]
        fields_modified = [
            col
            for col in original_df.columns
            if col in processed_df.columns
            and not original_df[col].equals(processed_df[col])
        ]

        invalid_values = {}
        constraint_type_violations = {}
        for processed_field in fields_modified + fields_added:
            if processed_field.startswith(self.column_prefix):
                original_field = processed_field[len(self.column_prefix) :]
            else:
                original_field = processed_field

            changes = (
                np.logical_not(
                    np.where(
                        original_df[original_field].isna()
                        | processed_df[processed_field].isna(),
                        original_df[original_field].isna()
                        & processed_df[processed_field].isna(),
                        original_df[original_field] == processed_df[processed_field],
                    )
                )
            ).sum()

            constraint_type = self.field_constraints.get(original_field, {}).get(
                "constraint_type", ""
            )

            invalid_values.update(
                {
                    original_field: {
                        "count": int(changes),
                        "percent": int(changes) / len(original_df),
                        "constraint_type": constraint_type,
                        "whitelist": original_field in self.whitelist_path,
                        "blacklist": original_field in self.blacklist_path,
                    }
                }
            )

            if constraint_type:
                if constraint_type not in constraint_type_violations:
                    constraint_type_violations[constraint_type] = 0

                constraint_type_violations[constraint_type] += int(changes)

        constraint_type_violations = dict(
            sorted(constraint_type_violations.items(), key=lambda x: x[1], reverse=True)
        )

        metrics.update(
            {
                "invalid_values": invalid_values,
                "constraint_type_violations": constraint_type_violations,
                "top_violations": list(constraint_type_violations.keys())[:3],
            }
        )

        return metrics

    def _save_metrics(
        self,
        metrics: Dict[str, Any],
        task_dir: Path,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: str,
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
        operation_timestamp : str
            Timestamp string for filenames

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
            description=f"Clean invalid values",
            category=Constants.Artifact_Category_Metrics,
        )

        # Report artifact
        if reporter:
            reporter.add_artifact(
                artifact_type="json",
                path=str(metrics_result.path),
                description=f"Clean invalid values metrics",
            )

        return metrics_result

    def _handle_visualizations(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        metrics: Dict[str, Any],
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
        vis_strict: bool,
        vis_timeout: int,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: str,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original data
        processed_df : pd.DataFrame
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
        operation_timestamp : str
            Timestamp string for filenames

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
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        metrics: Dict[str, Any],
        task_dir: Path,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
        vis_strict: bool,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: str,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate visualizations for the operation.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original data before processing
        processed_df : pd.DataFrame
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
        operation_timestamp : str
            Timestamp string for filenames

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        from pamola_core.utils.visualization import (
            create_bar_plot,
            create_heatmap,
            create_histogram,
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

            self.logger.debug(
                f"[VIZ] Data prepared for visualization: {len(original_df)} samples"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(n=2, postfix={"step": "Creating visualization"})

            # Number of invalid values per field
            viz_data = {
                field: statics["count"]
                for field, statics in metrics["invalid_values"].items()
            }
            viz_path = (
                viz_dir
                / f"{self.operation_name.lower()}_invalid_values_count_{operation_timestamp}.png"
            )
            viz_result = create_bar_plot(
                data=viz_data,
                output_path=viz_path,
                title="Invalid Values",
                x_label="Field",
                y_label="Count",
                sort_by="key",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )

            if viz_result.startswith("Error"):
                self.logger.error(f"Failed to create visualization: {viz_result}")
            else:
                visualization_paths[f"invalid_values_count"] = viz_path

            # Distribution of invalid values
            viz_data = [
                {
                    "field": field,
                    "constraint_type": statics["constraint_type"],
                    "count": statics["count"],
                }
                for field, statics in metrics["invalid_values"].items()
            ]
            viz_data = pd.DataFrame(viz_data)
            viz_data = viz_data.pivot(
                index="field", columns="constraint_type", values="count"
            ).fillna(0)
            viz_path = (
                viz_dir
                / f"{self.operation_name.lower()}_invalid_values_distribution_{operation_timestamp}.png"
            )
            viz_result = create_heatmap(
                data=viz_data,
                output_path=viz_path,
                title="Invalid Values Distribution",
                x_label="Constraint Type",
                y_label="Field",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )

            if viz_result.startswith("Error"):
                self.logger.error(f"Failed to create visualization: {viz_result}")
            else:
                visualization_paths[f"invalid_values_distribution"] = viz_path

            # Before/after fields
            fields_added = [
                col for col in processed_df.columns if col not in original_df.columns
            ]
            fields_modified = [
                col
                for col in original_df.columns
                if col in processed_df.columns
                and not original_df[col].equals(processed_df[col])
            ]

            for processed_field in fields_modified + fields_added:
                if processed_field.startswith(self.column_prefix):
                    original_field = processed_field[len(self.column_prefix) :]
                else:
                    original_field = processed_field

                if original_field in original_df.columns:
                    if is_numeric_dtype(original_df[original_field]):
                        viz_data = original_df[original_field].dropna()
                        viz_path = (
                            viz_dir
                            / f"{self.operation_name.lower()}_histograms_original_{original_field}_{operation_timestamp}.png"
                        )
                        viz_result = create_histogram(
                            data=viz_data,
                            output_path=viz_path,
                            title=f"Histograms original {original_field}",
                            x_label=f"{original_field}",
                            backend=vis_backend,
                            theme=vis_theme,
                            strict=vis_strict,
                            **kwargs,
                        )
                        if viz_result.startswith("Error"):
                            self.logger.error(
                                f"Failed to create visualization: {viz_result}"
                            )
                        else:
                            visualization_paths[
                                f"histograms_original_{original_field}"
                            ] = viz_path

                    if isinstance(original_df[original_field].dtype, CategoricalDtype):
                        viz_data = original_df[original_field].value_counts(
                            normalize=True
                        )
                        viz_path = (
                            viz_dir
                            / f"{self.operation_name.lower()}_bars_original_{original_field}_{operation_timestamp}.png"
                        )
                        viz_result = create_bar_plot(
                            data=viz_data,
                            output_path=viz_path,
                            title=f"Bars original {original_field}",
                            x_label="Categorical",
                            y_label="Frequency",
                            sort_by="key",
                            backend=vis_backend,
                            theme=vis_theme,
                            strict=vis_strict,
                            **kwargs,
                        )
                        if viz_result.startswith("Error"):
                            self.logger.error(
                                f"Failed to create visualization: {viz_result}"
                            )
                        else:
                            visualization_paths[f"bars_original_{original_field}"] = (
                                viz_path
                            )

                if processed_field in processed_df.columns:
                    if is_numeric_dtype(processed_df[processed_field]):
                        viz_data = processed_df[processed_field].dropna()
                        viz_path = (
                            viz_dir
                            / f"{self.operation_name.lower()}_histograms_processed_{processed_field}_{operation_timestamp}.png"
                        )
                        viz_result = create_histogram(
                            data=viz_data,
                            output_path=viz_path,
                            title=f"Histograms processed {processed_field}",
                            x_label=f"{processed_field}",
                            backend=vis_backend,
                            theme=vis_theme,
                            strict=vis_strict,
                            **kwargs,
                        )
                        if viz_result.startswith("Error"):
                            self.logger.error(
                                f"Failed to create visualization: {viz_result}"
                            )
                        else:
                            visualization_paths[
                                f"histograms_processed_{processed_field}"
                            ] = viz_path

                    if isinstance(
                        processed_df[processed_field].dtype, CategoricalDtype
                    ):
                        viz_data = processed_df[processed_field].value_counts(
                            normalize=True
                        )
                        viz_path = (
                            viz_dir
                            / f"{self.operation_name.lower()}_bars_processed_{processed_field}_{operation_timestamp}.png"
                        )
                        viz_result = create_bar_plot(
                            data=viz_data,
                            output_path=viz_path,
                            title=f"Bars processed {processed_field}",
                            x_label="Categorical",
                            y_label="Frequency",
                            sort_by="key",
                            backend=vis_backend,
                            theme=vis_theme,
                            strict=vis_strict,
                            **kwargs,
                        )
                        if viz_result.startswith("Error"):
                            self.logger.error(
                                f"Failed to create visualization: {viz_result}"
                            )
                        else:
                            visualization_paths[f"bars_processed_{processed_field}"] = (
                                viz_path
                            )

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
        processed_df: pd.DataFrame,
        task_dir: Path,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        operation_timestamp: str,
        **kwargs,
    ) -> WriterResult:
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
        operation_timestamp : str
            Timestamp string for filenames

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
            description=f"Clean invalid values",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_artifact(
                artifact_type="csv",
                path=str(output_result.path),
                description=f"Clean invalid values",
            )

        return output_result

    def _save_to_cache(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        metrics: Dict[str, Any],
        metrics_result: WriterResult,
        output_result: WriterResult,
        visualizations: Dict[str, Any],
        task_dir: Path,
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
        metrics_result : WriterResult
            The result of metrics
        output_result : WriterResult
            The result of output
        visualizations : dict
            The visualizations of operation
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
                "metrics_result_path": str(metrics_result.path),
                "output_result_path": str(output_result.path),
                "visualizations": visualizations,
                "data_info": {
                    "original_df_length": len(original_df),
                    "processed_df_length": len(processed_df),
                },
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
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
        self, original_df: Optional[pd.DataFrame], processed_df: Optional[pd.DataFrame]
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
def create_clean_invalid_values_operation(**kwargs) -> CleanInvalidValuesOperation:
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
