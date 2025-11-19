"""
PAMOLA.CORE - Privacy-Preserving AI Fake Data Generators
-------------------------------------------------------
Module:        Fake Data Generator Operation
Package:       pamola_core.fake_data
Version:       3.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-03
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
   Base classes and utilities for fake / synthetic data generation operations.
   Provides a standardized lifecycle, execution model, metrics collection,
   visualization support, memory-efficient processing and optional Dask/joblib
   execution paths for large-scale synthetic data workloads.

Purpose:
   Centralize generator and synthetic-data operation logic to ensure consistent
   behavior, deterministic mapping support, parameter validation, progress
   reporting and artifact management across all fake-data generators.

Key Features:
   - Standardized operation lifecycle (validation, execution, result handling)
   - Support for REPLACE and ENRICH modes for synthetic value injection
   - Configurable null handling strategies and conditional processing
   - Chunked processing for memory efficiency with Dask and joblib support
   - Comprehensive metrics collection and visualization generation
   - Deterministic mapping store support for repeatable generation
   - Result caching and artifact restoration for faster re-runs

Design Principles:
   - Type safety and explicit validation for inputs and parameters
   - Extensibility: subclasses override batch/value processing and cache params
   - Compatibility with PAMOLA operation framework and reporter interfaces
   - Performance: efficient batching, optional parallelism, and cached validators
   - Robustness: best-effort artifact generation and graceful degradation

Dependencies:
   - dask, pandas, numpy, joblib, psutil
   - pamola_core.* utilities (io, ops, fake_data.commons, utils.progress, etc.)
   - hashlib, json, datetime, pathlib, threading, contextvars
"""

import time
import dask.dataframe as dd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# Import anonymization-specific utilities
from pamola_core.anonymization.commons.data_utils import (
    process_nulls,
)
from pamola_core.common.constants import Constants
from pamola_core.fake_data.commons.base import BaseGenerator
from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.fake_data.commons.metrics import (
    create_metrics_collector,
    generate_metrics_report,
)
from pamola_core.fake_data.commons.processing_utils import (
    process_dataframe_using_chunk,
    process_dataframe_using_dask,
    process_dataframe_using_joblib,
)
from pamola_core.utils.io import (
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
from pamola_core.utils.ops.op_base import FieldOperation

# Import framework utilities
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_processing import (
    force_garbage_collection,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode


class GeneratorOperation(FieldOperation):
    """
    Base class for all generator-based generator or synthetic data operations.

    This class provides a consistent interface for operations that use
    generator-based approaches (e.g., fake names, fake emails, synthetic IDs),
    handling consistency, mapping stores, and metadata propagation.
    """

    def __init__(
        self,
        field_name: str,
        generator: BaseGenerator,
        generator_params: Optional[Dict[str, Any]] = None,
        consistency_mechanism: str = "prgn",
        id_field: Optional[str] = None,
        mapping_store_path: Optional[str] = None,
        save_mapping: bool = False,
        **kwargs,
    ):
        """
        Initialize a generator-based operation.

        Parameters
        ----------
        field_name : str
            Target field for generation/generator.
        generator : BaseGenerator
            Generator instance used to create synthetic data.
        generator_params : dict, optional
            Additional generator configuration parameters.
        consistency_mechanism : str, optional
            Consistency preservation mechanism (default "prgn").
        id_field : str, optional
            Unique identifier field for mapping consistency.
        mapping_store_path : str, optional
            Path to persistent mapping store (for deterministic reuse).
        mapping_store : MappingStore, optional
            Pre-loaded mapping store instance (used if provided).
        save_mapping : bool, optional
            Whether to persist mapping results after processing.
        **kwargs : dict
            Additional arguments forwarded to FieldOperation/BaseOperation.
        """

        # Default metadata
        kwargs.setdefault("name", f"{field_name}_generation")
        kwargs.setdefault(
            "description", f"Generator-based operation applied to '{field_name}'"
        )

        # Initialize parent FieldOperation
        super().__init__(field_name=field_name, **kwargs)

        # === Core Generator Configuration ===
        self.generator: BaseGenerator = generator
        self.generator_params = generator_params or {}

        # === Consistency and Mapping ===
        self.consistency_mechanism = consistency_mechanism.lower()
        self.mapping_store = MappingStore()
        self.mapping_store_path = mapping_store_path
        self.save_mapping = save_mapping

        # === id_field===
        self.id_field = id_field

        # === Descriptive Metadata ===
        self.name = f"{generator.__class__.__name__}_operation"
        self.description = (
            f"Generates synthetic '{field_name}' data using {generator.__class__.__name__} "
            f"(consistency='{self.consistency_mechanism}')"
        )

        # Create a metrics collector
        self._metrics_collector = create_metrics_collector()

        # === Optional Mapping Store Setup ===
        if mapping_store_path:
            self._initialize_mapping_store(mapping_store_path)

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the generator operation with enhanced features including Dask support.

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
            # Start timing
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(
                f"Starting {self.operation_name} operation at {self.start_time}"
            )

            # Initialize result object
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Create writer for consistent output handling
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Save operation configuration
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Load settings operation
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Data Loading & Validation, 3. Prepare output field, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
            TOTAL_MAIN_STEPS = 6 + (
                1 if self.use_cache and not self.force_recalculation else 0
            )
            main_progress = progress_tracker
            current_steps = 0
            if main_progress:
                self.logger.info(
                    f"Setting up progress tracker with {TOTAL_MAIN_STEPS} main steps"
                )
                try:
                    main_progress.total = TOTAL_MAIN_STEPS
                    main_progress.update(
                        current_steps,
                        {
                            "step": f"Starting {self.name}",
                            "field": self.field_name,
                        },
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # Step 1: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                try:
                    if main_progress:
                        current_steps += 1
                        main_progress.update(
                            current_steps,
                            {"step": "Checking cache", "field": self.field_name},
                        )
                    # Load data for cache check
                    self._original_df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
                    )

                    self.logger.info("Checking operation cache...")
                    cache_result = self._check_cache(self._original_df, reporter)

                    if cache_result:
                        self.logger.info(
                            f"Using cached result for {self.field_name} gereration operation"
                        )

                        # Update progress
                        if main_progress:
                            main_progress.update(
                                current_steps,
                                {"step": "Complete (cached)", "field": self.field_name},
                            )

                        # Report cache hit to reporter
                        if reporter:
                            reporter.add_operation(
                                f"Generator of {self.field_name} (from cache)",
                                details={"cached": True},
                            )

                        return cache_result
                except Exception as e:
                    error_message = f"Error checking cache: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

            # Step 2: Data Loading & Validation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Data Loading", "field": self.field_name}
                )

            # Validate and get dataframe
            try:
                if self._original_df is None:
                    self.logger.info(f"Loading data for field '{self.field_name}'")
                    self._original_df = self._validate_and_get_dataframe(
                        data_source, dataset_name, **settings_operation
                    )
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 3: Prepare output field
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Preparing output field", "field": self.field_name},
                )

            try:
                self.output_field_name = self._prepare_output_field(self._original_df)
                self.logger.info(f"Prepared output_field: '{self.output_field_name}'")
                self._report_operation_details(reporter, self.output_field_name)
            except Exception as e:
                error_message = f"Preparing output field error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 4: Processing
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", "field": self.field_name}
                )

            try:
                # Copy original data for processing
                original_data = self._original_df[self.field_name].copy(deep=True)

                # Create child progress tracker for Chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="Processing dataframe",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Process the filtered data
                processed_df = self._process_data_with_config(
                    df=self._original_df,
                    progress_tracker=data_tracker,
                    **kwargs,
                )

                # Get the generated data
                generated_data = processed_df[self.output_field_name]

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Record end time after processing metrics
            self.end_time = time.time()

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Metrics Calculation", "field": self.field_name},
                )

            # Initialize metrics in scope
            metrics = {}

            try:
                metrics = self._collect_metrics(processed_df)

                # Generate metrics file name (in self.name existed field_name)
                metrics_file_name = (
                    f"{self.field_name}_{self.name}_metrics_{operation_timestamp}"
                )

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if self.use_encryption else None
                    ),
                )

                # Add metrics to result
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        result.add_metric(key, value)

                # Register metrics artifact
                result.add_artifact(
                    artifact_type="json",
                    path=metrics_result.path,
                    description=f"{self.field_name} generator metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report artifact
                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} generator metrics",
                        details={
                            "artifact_type": "json",
                            "path": str(metrics_result.path),
                        },
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            try:
                report_dir = task_dir / "reports"
                generate_metrics_report(
                    metrics,
                    report_dir,
                    op_type=self.name,
                    field_name=self.field_name,
                    operation_timestamp=operation_timestamp,
                )

                if reporter:
                    reporter.add_operation(
                        f"Operation {self.operation_name}",
                        status="info",
                        details={
                            "message": "Generate and save metrics report successfully"
                        },
                    )
            except Exception as e:
                error_message = f"Operation: {self.operation_name}, Error generating metrics report: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Generating Visualizations", "field": self.field_name},
                )

            # Generate visualizations if required
            # Initialize visualization paths dictionary
            visualization_paths = {}
            if self.generate_visualization and self.visualization_backend is not None:
                try:
                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    }
                    visualization_paths = self._handle_visualizations(
                        df=processed_df,
                        metrics=metrics,
                        task_dir=task_dir,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        operation_timestamp=operation_timestamp,
                        **kwargs_encryption,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.warning(error_message)
                    # Continue execution - visualization failure is not critical
            else:
                self.logger.info(
                    "Skipping visualizations as generate_visualization is False or backend is not set"
                )

            # Step 7: Save Output Data
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Save Output Data", "field": self.field_name},
                )

            # Save output data if required
            output_result_path = None
            if self.save_output:
                try:
                    safe_kwargs = filter_used_kwargs(kwargs, self._save_output_data)
                    output_result_path = self._save_output_data(
                        result_df=processed_df,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        use_encryption=self.use_encryption,
                        **safe_kwargs,
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
                        original_data=original_data,
                        generated_data=generated_data,
                        metrics=metrics,
                        task_dir=task_dir,
                        visualization_paths=visualization_paths,
                        metrics_result_path=str(metrics_result.path),
                        output_result_path=output_result_path,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                processed_df=processed_df,
                original_data=original_data,
                generated_data=generated_data,
            )

            # Finalize timing
            self.end_time = time.time()

            # Report completion
            if reporter:
                reporter.add_operation(
                    f"Generator of {self.field_name} completed",
                    details={
                        "records_processed": self.process_count,
                        "execution_time": self.end_time - self.start_time,
                    },
                )

            # Set success status
            result.status = OperationStatus.SUCCESS
            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )
            return result

        except Exception as e:
            error_message = f"Error in generator operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message,
                exception=e,
            )

    def _initialize_mapping_store(self, path: Union[str, Path]) -> None:
        """
        Initialize the mapping store if needed.

        Args:
            path: Path to mapping store file
        """
        try:
            from pamola_core.fake_data.commons.mapping_store import MappingStore

            self.mapping_store = MappingStore()

            # Load existing mappings if the file exists
            path_obj = Path(path)
            if path_obj.exists():
                self.mapping_store.load(path_obj)
                self.logger.info(f"Loaded mapping store from {path_obj.name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize mapping store: {str(e)}")
            self.mapping_store = None

    def _validate_and_get_dataframe(
        self, data_source: DataSource, dataset_name: str, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Validate data source and retrieve the main dataframe.

        Parameters:
        -----------
        data_source : DataSource
            The data source to validate
        dataset_name : str
            The name of the dataset to retrieve
        **kwargs : Any
            Additional keyword arguments to pass to the data loading function

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
        df = load_data_operation(data_source, dataset_name, **kwargs)
        if df is None:
            error_message = f"Failed to load input data!"
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
            output_field = self.field_name
        else:  # ENRICH mode
            if self.output_field_name:
                output_field = self.output_field_name
            else:
                output_field = f"{self.column_prefix}{self.field_name}"

            # Check if output field already exists in DataFrame
            if output_field in df.columns:
                self.logger.warning(
                    f"Output field '{output_field}' already exists and will be overwritten"
                )
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
            reporter.add_operation(
                f"Generator field: {self.field_name}",
                details={
                    "field_name": self.field_name,
                    "output_field": output_field,
                    "mode": self.mode,
                    "null_strategy": self.null_strategy,
                    "operation_type": self.operation_name,
                },
            )

    def _process_data_with_config(
        self,
        df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
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
        # Check if dataframe is empty
        if len(df) == 0:
            self.logger.warning("Empty DataFrame provided, returning as is")
            return df

        # Handle null values based on strategy
        if self.null_strategy != "PRESERVE":
            df[self.field_name] = process_nulls(
                df[self.field_name], strategy=self.null_strategy.upper()
            )

        processed_df = None
        flag_processed = False
        # Backup and clear operation cache during processing
        cache_backup = self.operation_cache
        self.operation_cache = None
        self.logger.info("Process with config")

        # For larger dataframes, check if we should use parallel processing
        if not flag_processed and self.use_dask:
            try:
                self.logger.info("Parallel Enabled")
                self.logger.info("Parallel Engine: Dask")
                self.logger.info(f"Parallel Workers: {self.npartitions}")
                self.logger.info(
                    f"Using dask processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Setting up dask processing"})

                self.logger.info("Process using Dask")

                # Process with Dask - delegate to subclass method
                processed_df, flag_processed = process_dataframe_using_dask(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **kwargs,
                )

                if flag_processed:
                    self.logger.info("Completed using Dask")

            except Exception as e:
                self.logger.warning(
                    f"Error in dask processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and self.use_vectorization:
            try:
                self.logger.info("Parallel Enabled")
                self.logger.info("Parallel Engine: Joblib")
                self.logger.info(f"Parallel Workers: {self.parallel_processes}")
                self.logger.info(
                    f"Using vectorized processing with chunk size {self.chunk_size}"
                )
                if progress_tracker:
                    progress_tracker.update(
                        0, {"step": "Setting up vectorized processing"}
                    )

                self.logger.info("Process using Joblib")

                processed_df, flag_processed = process_dataframe_using_joblib(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **kwargs,
                )

                if flag_processed:
                    self.logger.info("Completed using Joblib")

            except Exception as e:
                self.logger.warning(
                    f"Error in vectorized processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and self.chunk_size > 1:
            try:
                # Regular chunk processing
                self.logger.info(
                    f"Processing in chunks with chunk size {self.chunk_size}"
                )
                total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
                self.logger.info(f"Total chunks to process: {total_chunks}")
                if progress_tracker:
                    progress_tracker.update(
                        0,
                        {
                            "step": "Processing in chunks",
                            "total_chunks": total_chunks,
                        },
                    )
                self.logger.info("Process using chunk")
                processed_df, flag_processed = process_dataframe_using_chunk(
                    df=df,
                    process_function=self.process_batch,
                    progress_tracker=progress_tracker,
                    task_logger=self.logger,
                    **kwargs,
                )
                if flag_processed:
                    self.logger.info("Completed using chunk")
            except Exception as e:
                self.logger.warning(f"Error in chunk processing: {e}")

        if not flag_processed:
            self.logger.info("Fallback process as usual")
            processed_df = self.process_batch(df, **kwargs)
            flag_processed = True

        # Update process count
        self.process_count += len(df)
        # Restore operation cache
        self.operation_cache = cache_backup
        return processed_df

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collects metrics for the operation (including generator-specific and performance metrics).

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame.

        Returns
        -------
        Dict[str, Any]
            Metrics for the operation.
        """
        import psutil
        import time

        # === 1. Basic metrics ===
        total_records = len(df)
        metrics = {
            "total_records": total_records,
            "non_null_records": df[self.field_name].notna().sum(),
            "execution_time": 0,  # kept for backward compatibility
            "performance": {},
        }

        # === 2. Performance metrics ===
        if getattr(self, "start_time", None) is not None:
            try:
                execution_time = time.time() - self.start_time
                records_per_second = (
                    int(total_records / execution_time) if execution_time > 0 else 0
                )

                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)

                metrics["performance"].update(
                    {
                        "generation_time": round(execution_time, 4),
                        "records_per_second": records_per_second,
                        "memory_usage_mb": round(memory_usage_mb, 2),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Error collecting performance metrics: {str(e)}")

        # === 3. Output field metrics (for ENRICH mode) ===
        if self.mode == "ENRICH" and self.output_field_name in df.columns:
            metrics["output_field"] = {
                "name": self.output_field_name,
                "non_null_records": df[self.output_field_name].notna().sum(),
            }

        # === 4. Metrics collector (distribution comparison, utility metrics) ===
        try:
            orig_series = (
                self._original_df[self.field_name]
                if self._original_df is not None
                else None
            )
            gen_series = None
            if self.mode == "REPLACE":
                gen_series = df[self.field_name]
            elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                gen_series = df[self.output_field_name]

            if orig_series is not None and gen_series is not None:
                collector_metrics = self._metrics_collector.collect_metrics(
                    orig_data=orig_series,
                    gen_data=gen_series,
                    field_name=self.field_name,
                    operation_params={"field_name": self.field_name},
                )

                for key, value in collector_metrics.items():
                    if key not in metrics:
                        metrics[key] = value
                    elif isinstance(metrics[key], dict) and isinstance(value, dict):
                        metrics[key].update(value)
        except Exception as e:
            self.logger.warning(f"Error collecting detailed metrics: {str(e)}")

        # === 5. Generator-specific extension (if exists) ===
        if hasattr(self, "generator"):
            metrics["generator"] = {
                "type": self.generator.__class__.__name__,
                "consistency_mechanism": getattr(self, "consistency_mechanism", None),
            }

            # Mapping mechanism details
            if getattr(self, "consistency_mechanism", None) == "mapping":
                try:
                    field_mappings = self.mapping_store.get_field_mappings(
                        self.field_name
                    )
                    metrics["mapping"] = {"total_mappings": len(field_mappings)}
                except Exception as e:
                    self.logger.warning(f"Error collecting mapping metrics: {str(e)}")

            # Dictionary info if available
            if hasattr(self.generator, "get_dictionary_info"):
                try:
                    dictionary_info = self.generator.get_dictionary_info()
                    if dictionary_info:
                        metrics.setdefault("dictionary_metrics", {}).update(
                            dictionary_info
                        )
                except Exception as e:
                    self.logger.warning(f"Error getting dictionary info: {str(e)}")

            # Transformation / distribution comparison
            if orig_series is not None and gen_series is not None:
                try:
                    compare_metrics = self._metrics_collector.compare_distributions(
                        orig_series, gen_series
                    )
                    metrics.setdefault("transformation_metrics", {}).update(
                        compare_metrics
                    )
                except Exception as e:
                    self.logger.warning(f"Error comparing distributions: {str(e)}")

        # === 6. Operation-specific extension hook ===
        if hasattr(self, "_collect_specific_metrics"):
            try:
                specific = self._collect_specific_metrics(df)
                if specific:
                    metrics.update(specific)
            except Exception as e:
                self.logger.warning(f"Error collecting specific metrics: {str(e)}")

        return metrics

    def _collect_specific_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect operation-specific metrics. Should be overridden by subclasses.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary with metrics
        """
        return {}

    def _handle_visualizations(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate and save visualizations with thread-safe context support.

        Parameters:
        -----------
        df : pd.DataFrame
            The processed dataframe
        metrics : Dict[str, Any]
            The collected metrics
        task_dir : Path
            The task directory
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        operation_timestamp : str, optional
            Timestamp for the operation (default: current time)
        **kwargs : dict
            Additional parameters for the operation
        """
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
                    f"[DIAG] Field: {self.field_name}, Strategy: {self.null_strategy}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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

                    # Generate visualizations with context parameters
                    visualization_paths = self._generate_visualizations(
                        df=df,
                        metrics=metrics,
                        task_dir=task_dir,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend or "plotly",
                        vis_strict=vis_strict,
                        progress_tracker=viz_progress,
                        timestamp=operation_timestamp,  # Pass the same timestamp
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
                name=f"VizThread-{self.field_name}",
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
                description=f"{self.field_name} {viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_operation(
                    f"{self.field_name} {viz_type} visualization",
                    details={"artifact_type": "png", "path": str(path)},
                )

        return visualization_paths

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate visualizations using the core visualization utilities with thread-safe context support.

        This is a base implementation that provides a basic distribution comparison.
        Subclasses should override to provide operation-specific visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            The processed dataframe
        metrics : Dict[str, Any]
            The collected metrics
        task_dir : Path
            Task directory for saving visualizations
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter object for tracking artifacts
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        visualization_paths = {}
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if visualization should be skipped
        if vis_backend is None:
            self.logger.info(
                f"Skipping visualization for {self.field_name} (backend=None)"
            )
            return visualization_paths

        self.logger.info(
            f"[VIZ] Starting visualization generation for {self.field_name} using {self.null_strategy} strategy"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            if self.mode == "REPLACE":
                gen_series = df[self.field_name]
            elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                gen_series = df[self.output_field_name]
            else:
                gen_series = None

            kwargs_visualization = {
                "use_encryption": self.use_encryption,
                "encryption_key": self.encryption_key,
                "backend": self.visualization_backend,
                "theme": self.visualization_theme,
                "strict": self.visualization_strict,
                "timestamp": timestamp,
            }

            if gen_series is not None:
                visualizations = self._metrics_collector.visualize_metrics(
                    metrics=metrics,
                    field_name=self.field_name,
                    output_dir=viz_dir,
                    op_type=self.name,
                    **kwargs_visualization,
                )
                visualization_paths = {
                    name: str(path) for name, path in visualizations.items()
                }

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")

        return visualization_paths

    def _save_output_data(
        self,
        result_df: pd.DataFrame,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        timestamp: Optional[str] = None,
        use_encryption: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """
        Save the processed output data.

        Parameters:
        -----------
        result_df : pd.DataFrame
            The processed dataframe to save
        use_encryption : bool
            Whether to encrypt the output
        writer : DataWriter
            The writer to use for saving data
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker
        timestamp : Optional[str]
            Optional timestamp for the operation
        **kwargs : dict
            Additional parameters for the operation
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        # Generate standardized output filename with timestamp
        field_name_output = (
            f"{self.field_name}_{self.operation_name}_output_{timestamp}"
        )

        # Use the DataWriter to save the DataFrame
        safe_kwargs = filter_used_kwargs(kwargs, writer.write_dataframe)
        safe_kwargs["encryption_mode"] = get_encryption_mode(result_df, **kwargs)
        output_result = writer.write_dataframe(
            df=result_df,
            name=field_name_output,
            format=self.output_format,
            subdir="output",
            timestamp_in_name=False,
            encryption_key=self.encryption_key if use_encryption else None,
            overwrite=True,
            **safe_kwargs,
        )

        # Register output artifact with the result
        result.add_artifact(
            artifact_type=self.output_format,
            path=output_result.path,
            description=f"{self.field_name} generator data",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_operation(
                f"{self.field_name} generator data",
                details={
                    "artifact_type": self.output_format,
                    "path": str(output_result.path),
                },
            )
        return str(output_result.path)

    def _cleanup_memory(
        self,
        processed_df: Optional[pd.DataFrame] = None,
        original_data: Optional[pd.Series] = None,
        generated_data: Optional[pd.Series] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        references and optionally calling garbage collection.

        Parameters:
        -----------
        processed_df : pd.DataFrame, optional
            Processed DataFrame to clear from memory
        original_data : pd.Series, optional
            Original data to clear from memory
        generated_data : pd.Series, optional
            Generated data to clear from memory
        """
        # Delete references
        if processed_df is not None:
            del processed_df
        if original_data is not None:
            del original_data
        if generated_data is not None:
            del generated_data

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Force garbage collection
        force_garbage_collection()

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data. Must be implemented by subclasses.

        Parameters:
        -----------
        batch : pd.DataFrame
            DataFrame batch to process
        kwargs : dict
            Additional keyword arguments for processing

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch
        """
        raise NotImplementedError("Subclasses must implement process_batch method")

    def _check_cache(
        self, df: pd.DataFrame, reporter: Any
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for this operation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame for the operation
        reporter : Any
            Reporter object for tracking progress and artifacts

        Returns
        -------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            if self.field_name not in df.columns:
                self.logger.warning(
                    f"Field '{self.field_name}' not found in DataFrame."
                )
                return None

            cache_key = self._generate_cache_key(df[self.field_name])
            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.operation_name
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(
                f"Using cached result for {self.field_name} generalization"
            )

            result = OperationResult(status=OperationStatus.SUCCESS)
            # Restore cached data
            self._add_cached_metrics(result, cached_result)
            artifacts_restored = self._restore_cached_artifacts(
                result, cached_result, reporter
            )

            # Add cache metadata
            result.add_metric("cached", True)
            result.add_metric("cache_key", cache_key)
            result.add_metric(
                "cache_timestamp", cached_result.get("timestamp", "unknown")
            )
            result.add_metric("artifacts_restored", artifacts_restored)

            if reporter:
                reporter.add_operation(
                    f"Generalization of {self.field_name} (cached)",
                    details={
                        "null_strategy": self.null_strategy,
                        "cached": True,
                        "artifacts_restored": artifacts_restored,
                    },
                )

            self.logger.info(
                f"Cache hit successful: restored {artifacts_restored} artifacts"
            )
            return result

        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _add_cached_metrics(self, result: OperationResult, cached: dict):
        """
        Add cached scalar metrics (int, float, str, bool) to the OperationResult.

        Parameters
        ----------
        result : OperationResult
            The result object to update.
        cached : dict
            Cached result dictionary from cache manager.
        """
        for key, value in cached.get("metrics", {}).items():
            if isinstance(value, (int, float, str, bool)):
                result.add_metric(key, value)

    def _restore_cached_artifacts(
        self, result: OperationResult, cached: dict, reporter: Optional[Any]
    ) -> int:
        """
        Restore artifacts (output, metrics, visualizations) from cached result if files exist.

        Parameters
        ----------
        result : OperationResult
            OperationResult object to update with restored artifacts.
        cached : dict
            Cached result dictionary from cache manager.
        reporter : Optional[Any]
            Optional reporter object for tracking operation-level artifacts.

        Returns
        -------
        int
            Number of artifacts successfully restored.
        """
        artifacts_restored = 0

        def restore_file_artifact(
            path: Union[str, Path], artifact_type: str, desc_suffix: str, category: str
        ):
            """
            Restore a single artifact from a file path if it exists.

            Parameters
            ----------
            path : Union[str, Path]
                Path to the artifact file.
            artifact_type : str
                Type of the artifact (e.g., 'json', 'csv', 'png').
            desc_suffix : str
                Description suffix (e.g., 'visualization', 'metrics').
            category : str
                Artifact category (e.g., output, metrics, visualization).
            """
            nonlocal artifacts_restored
            if not path:
                return

            artifact_path = Path(path)
            if artifact_path.exists():
                result.add_artifact(
                    artifact_type=artifact_type,
                    path=artifact_path,
                    description=f"{self.field_name} {desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{self.field_name} {desc_suffix} (cached)",
                        details={
                            "artifact_type": artifact_type,
                            "path": str(artifact_path),
                        },
                    )
            else:
                self.logger.warning(f"Cached file not found: {artifact_path}")

        # Restore main output and metrics and mapping artifacts
        restore_file_artifact(
            cached.get("output_file"),
            self.output_format,
            "generalized data",
            Constants.Artifact_Category_Output,
        )
        restore_file_artifact(
            cached.get("metrics_file"),
            "json",
            "generalization metrics",
            Constants.Artifact_Category_Metrics,
        )
        restore_file_artifact(
            cached.get("mapping_file"),
            "json",
            "generalized mapping",
            Constants.Artifact_Category_Mapping,
        )

        # Restore visualizations
        for viz_type, path_str in cached.get("visualizations", {}).items():
            restore_file_artifact(
                path_str,
                "png",
                f"{viz_type} visualization",
                Constants.Artifact_Category_Visualization,
            )

        return artifacts_restored

    def _save_to_cache(
        self,
        original_data: pd.Series,
        generated_data: pd.Series,
        metrics: Dict[str, Any],
        task_dir: Path,
        visualization_paths: Dict[str, Path] = {},
        metrics_result_path: Optional[str] = None,
        output_result_path: Optional[str] = None,
        mapping_result_path: Optional[str] = None,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_data : pd.Series
            Original input data
        generated_data : pd.Series
            Generated output data
        metrics : Dict[str, Any]
            Metrics collected during the operation
        visualization_paths : Dict[str, Path]
            Paths to generated visualizations
        task_dir : Path
            Task directory
        metrics_result_path : Optional[str]
            Path to the metrics result file
            If not provided, a default path will be used.
        output_result_path : Optional[str]
            Path to the output result file
            If not provided, a default path will be used.
        mapping_result_path : Optional[str]
            Path to the mapping result file
            If not provided, a default path will be used.

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(original_data)

            # Prepare metadata for cache
            operation_params = self._get_basic_parameters()
            operation_params.update(self._get_cache_parameters())
            self.logger.debug(f"Operation parameters for cache: {operation_params}")

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in metrics.items()
                },
                "parameters": operation_params,
                "data_info": {
                    "original_length": len(original_data),
                    "generated_length": len(generated_data),
                    "original_null_count": int(original_data.isna().sum()),
                    "generated_null_count": int(generated_data.isna().sum()),
                },
                "output_file": output_result_path,  # Path to main output file
                "metrics_file": metrics_result_path,  # Path to metrics file
                "mapping_file": mapping_result_path,  # Path to mapping file if applicable
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")

            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.operation_name,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(
                    f"Successfully saved {self.field_name} generator results to cache"
                )
            else:
                self.logger.warning(
                    f"Failed to save {self.field_name} generator results to cache"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

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
        # Get basic operation parameters
        parameters = self._get_basic_parameters()

        # Add operation-specific parameters through method that subclasses can override
        parameters.update(self._get_cache_parameters())

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(data)

        # Use the operation_cache utility to generate a consistent cache key
        return self.operation_cache.generate_cache_key(
            operation_name=self.operation_name,
            parameters=parameters,
            data_hash=data_hash,
        )

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generate a hash representing the key characteristics of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data for the operation

        Returns
        -------
        str
            Hash string representing the data
        """
        import hashlib
        import json

        try:
            # Generate summary statistics for all columns (numeric and non-numeric)
            characteristics = df.describe(include="all")

            # Convert to JSON string with consistent formatting (ISO for dates)
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback: use length and column data types
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _get_basic_parameters(self) -> Dict[str, str]:
        """Get the basic parameters for the cache key generation."""
        return {
            "mode": self.mode,
            "operation": self.operation_name,
            "version": self.version,
            "field_name": self.field_name,
            "output_field_name": self.output_field_name,
            "null_strategy": self.null_strategy,
            "chunk_size": self.chunk_size,
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
