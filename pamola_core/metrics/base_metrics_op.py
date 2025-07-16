"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Base Metrics Operation
Package:       pamola_core.metrics
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
    This module defines the base class for all metrics operations in PAMOLA.CORE,
    providing a unified interface and shared functionality for metrics calculation,
    result handling, and distributed processing. It supports both pandas and Dask
    backends for scalable, memory-efficient analytics on large datasets, and
    integrates with the operation framework for progress tracking, caching, and
    secure artifact management.

Key Features:
    - Standardized operation lifecycle: validation, execution, and result handling
    - In-place (REPLACE) and enrichment (ENRICH) processing modes
    - Configurable null value handling: PRESERVE, EXCLUDE, ERROR
    - Chunk-based and adaptive memory-efficient processing
    - Comprehensive metrics collection and visualization support
    - Robust caching for operation results and artifacts
    - Progress tracking and reporting throughout the operation
    - Secure output generation with optional encryption
    - Conditional processing based on field values and risk scores
    - Integration with k-anonymity and profiling results
    - Automatic memory optimization and adaptive chunk sizing
    - Dask integration for distributed, parallel processing

Framework:
    Follows the PAMOLA.CORE operation framework with standardized interfaces for
    input/output, progress tracking, result reporting, and extensibility for custom
    metrics operations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import dask.dataframe as dd

from pamola_core.common.constants import Constants
from pamola_core.utils.io import (
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.io_helpers.dask_utils import convert_to_dask
from pamola_core.utils.ops.op_base import BaseOperation

# Import framework utilities
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_processing import (
    optimize_dataframe_dtypes,
    get_memory_usage,
    force_garbage_collection,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker


class MetricsOperation(BaseOperation):
    """
    Base class for all metrics operation support.

    This class provides common functionality for all metrics operations,
    including data source handling, result processing, metric calculation,
    and automatic switching to Dask for large dataset processing.
    """

    def __init__(
        self,
        name: str = "base_metrics",
        mode: str = "comparison",  # comparison, standalone
        columns: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        normalize: bool = True,
        confidence_level: float = 0.95,
        description: str = "",
        # Memory optimization
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        # Specific parameters
        sample_size: Optional[int] = None,
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_cache: bool = True,
        use_encryption: bool = False,
        encryption_mode: Optional[str] = None,
        encryption_key: Optional[Union[str, Path]] = None,
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
    ):
        """
        Initialize the metrics operation.

        Parameters:
        -----------
        name : str
            Name of the operation (default: "base_metrics")
        mode : str, optional
            "comparison" for comparing original vs transformed data or "standalone" for independent analysis
        columns : List[str], optional
            List of columns to include in the operation
        column_mapping : Dict[str, str], optional
            Mapping of original columns to new names (if applicable)
        normalize : bool, optional
            Whether to normalize the data (default: True)
        confidence_level : float, optional
            Confidence level for statistical metrics (default: 0.95)
        sample_size : Optional[int], optional
            Size of the sample to use for metrics calculation (default: None, uses full dataset)
        description : str, optional
            Operation description
        optimize_memory : bool, optional
            Whether to optimize DataFrame memory usage
        adaptive_chunk_size : bool, optional
            Whether to adjust chunk size based on available memory
        chunk_size : int, optional
            Size of chunks for processing (default: 10000)
        use_dask : bool, optional
            Whether to use Dask for parallel processing (default: False)
        npartitions : Optional[int], optional
            Number of partitions to use with Dask (default: None)
        dask_partition_size : Optional[str], optional
            Size of Dask partitions (default: None, auto-determined)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        encryption_mode : Optional[str], optional
            The encryption mode to use (default: None)
        visualization_theme : Optional[str], optional
            Theme for visualizations (default: None)
        visualization_backend : Optional[str], optional
            Backend for visualizations ("plotly" or "matplotlib", default: None)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout for visualization generation in seconds (default: 120)
        """
        # Use a default description if none provided
        if not description:
            description = f"{name} metrics operation for field"

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
        )

        # Store basic parameters
        self.mode = mode.upper()
        self.columns = columns or []
        self.column_mapping = column_mapping or {}
        self.normalize = normalize
        self.confidence_level = confidence_level
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.dask_partition_size = dask_partition_size
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.encryption_mode = encryption_mode
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout
        self.version = getattr(self, "version", "4.0.0")

        # Memory optimization parameters
        self.optimize_memory = optimize_memory
        self.adaptive_chunk_size = adaptive_chunk_size
        self.original_chunk_size = chunk_size

        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.process_count = 0

        # Set up common variables
        self.force_recalculation = False  # Skip cache check
        self.generate_visualization = True  # Create visualizations
        self.process_kwargs = {}

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the metrics operation with enhanced features including Dask support.

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
            Additional parameters including profiling_results

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Start timing
            self.start_time = time.time()
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(f"Starting {self.name} operation at {self.start_time}")
            self.process_count = 0
            original_df = None
            transformed_df = None

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

            # Decompose kwargs and introduce variables for clarity
            self.generate_visualization = kwargs.get("generate_visualization", True)
            self.force_recalculation = kwargs.get("force_recalculation", False)
            self.original_dataset_name = kwargs.get("original_dataset_name", "main")
            self.transformed_dataset_name = kwargs.get(
                "transformed_dataset_name", "main"
            )

            # Extract visualization parameters
            self.visualization_theme = kwargs.get(
                "visualization_theme", self.visualization_theme
            )
            self.visualization_backend = kwargs.get(
                "visualization_backend", self.visualization_backend
            )
            self.visualization_strict = kwargs.get(
                "visualization_strict", self.visualization_strict
            )
            self.visualization_timeout = kwargs.get(
                "visualization_timeout", self.visualization_timeout
            )

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Data Loading & Validation, 3. Processing, 4. Metrics, 5. Visualization, 6. Save Cache
            TOTAL_MAIN_STEPS = 5 + (
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
                            "columns": self.columns,
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
                            {"step": "Checking cache", "name": self.name},
                        )

                    # Load original dataset for check cache
                    self.logger.info("Load original dataset for check cache...")
                    original_df = self._get_dataset_by_name(
                        data_source, self.original_dataset_name, **kwargs
                    )
                    self.logger.info(
                        f"Original dataset '{self.original_dataset_name}' loaded with {len(original_df)} records."
                    )

                    self.logger.info("Checking operation cache...")
                    cache_result = self._check_cache(original_df, reporter)

                    if cache_result:
                        self.logger.info(f"Using cached result for {self.name}")

                        # Update progress
                        if main_progress:
                            main_progress.update(
                                current_steps,
                                {"step": "Complete (cached)", "name": self.name},
                            )

                        # Report cache hit to reporter
                        if reporter:
                            reporter.add_operation(
                                f"Metric of {self.name} (from cache)",
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
                    current_steps, {"step": "Data Loading", "name": self.name}
                )

            # Validate and get dataframe
            try:
                if original_df is None:
                    self.logger.info(f"Loading original data for name '{self.name}'")
                    original_df = self._get_dataset_by_name(
                        data_source, self.original_dataset_name, **kwargs
                    )
                if transformed_df is None:
                    self.logger.info(f"Loading transformed data for name '{self.name}'")
                    transformed_df = self._get_dataset_by_name(
                        data_source, self.transformed_dataset_name, **kwargs
                    )

                # Validate loaded dataframes
                self._validate_inputs(original_df, transformed_df)
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 3: Processing
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps, {"step": "Processing", "name": self.name}
                )

            try:
                # Sample aligned data
                sampled_orig, sampled_trans = self._sample_aligned(
                    original_df=original_df,
                    transformed_df=transformed_df,
                    sample_size=self.sample_size,
                    columns=self.columns,
                    column_mapping=self.column_mapping,
                )

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
                processed_metrics = self._calculate_metrics_with_config(
                    original_df=sampled_orig,
                    transformed_df=sampled_trans,
                    progress_tracker=data_tracker,
                )

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

            # Step 4: Metrics Calculation
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Metrics Calculation", "name": self.name},
                )

            try:
                basic_metrics = self._collect_basic_metrics(
                    original_data=sampled_orig, transformed_data=sampled_trans
                )

                # Update processed metrics
                processed_metrics.update(basic_metrics)

                # Generate metrics file name (in self.name)
                metrics_file_name = f"{self.name}_{operation_timestamp}"

                # Save metrics using writer
                metrics_result = writer.write_metrics(
                    metrics=processed_metrics,
                    name=metrics_file_name,
                    timestamp_in_name=False,
                    encryption_key=(
                        self.encryption_key if self.use_encryption else None
                    ),
                )

                # Add metrics to result
                for key, value in processed_metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        result.add_metric(key, value)

                # Register metrics artifact
                result.add_artifact(
                    artifact_type="json",
                    path=metrics_result.path,
                    description=f"{self.name} metrics",
                    category=Constants.Artifact_Category_Metrics,
                )

                # Report artifact
                if reporter:
                    reporter.add_operation(
                        f"{self.name} metrics",
                        details={
                            "artifact_type": "json",
                            "path": str(metrics_result.path),
                        },
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 5: Visualizations
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Generating Visualizations", "name": self.name},
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
                        metrics=processed_metrics,
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

            # Step 6: Save Cache
            if main_progress:
                current_steps += 1
                main_progress.update(
                    current_steps,
                    {"step": "Saving Cache", "name": self.name},
                )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_df=original_df,
                        transformed_df=transformed_df,
                        metrics=processed_metrics,
                        task_dir=task_dir,
                        visualization_paths=visualization_paths,
                        metrics_result_path=str(metrics_result.path),
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Clean up memory AFTER all write operations are complete
            self.logger.info("Cleaning up memory after all file operations")
            self._cleanup_memory(
                sampled_orig=sampled_orig,
                sampled_trans=sampled_trans,
                original_df=original_df,
                transformed_df=transformed_df,
            )

            # Finalize timing
            self.end_time = time.time()

            # Report completion
            if reporter:
                reporter.add_operation(
                    f"Metrics of {self.name} completed",
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
            error_message = f"Error in transformation operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message,
                exception=e,
            )

    def _validate_inputs(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame
    ) -> None:
        """
        Validate the inputs and raise error if not compatible.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original DataFrame to validate.
        transformed_df : pd.DataFrame
            The transformed DataFrame to validate.
        """
        if not isinstance(original_df, pd.DataFrame) or not isinstance(
            transformed_df, pd.DataFrame
        ):
            raise ValueError(
                "Both original_df and transformed_df must be pandas DataFrames"
            )

        columns_to_check = self.columns or list(original_df.columns)

        for col in columns_to_check:
            if col not in original_df.columns:
                raise ValueError(f"Column '{col}' not found in original_df")

            mapped_col = self.column_mapping.get(col, col)
            if mapped_col not in transformed_df.columns:
                raise ValueError(
                    f"Mapped column '{mapped_col}' not found in transformed_df"
                )

    def _get_dataset_by_name(
        self, source: Any, dataset_name_or_path: Union[str, Path] = None, **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves a DataFrame from a given data source using a dataset name or path.

        Parameters:
        -----------
        source : Any
            The data source. Can be:
            - A dictionary-like pandas DataFrame with named datasets
            - A DataSource object
            - A file path (string or Path)
        dataset_name_or_path : str or None
            The name or path of the dataset to retrieve.
        Returns:
        --------
        pd.DataFrame or None
            The loaded DataFrame, or None if the dataset_name_or_path is None.
        """
        if dataset_name_or_path is None:
            return None
        if isinstance(source, pd.DataFrame):
            return source.get(dataset_name_or_path)

        # Load settings operation
        settings_operation = load_settings_operation(
            source, dataset_name_or_path, **kwargs
        )

        df = load_data_operation(source, dataset_name_or_path, **settings_operation)
        if df is None:
            error_message = f"Failed to load input data!"
            self.logger.error(error_message)
            raise ValueError(error_message)

        df = self._optimize_data(df)

        return df

    def _optimize_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Load data and optimize memory usage.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to optimize

        Returns:
        --------
        Optional[pd.DataFrame]
            Loaded and optimized DataFrame or None if error
        """
        # Optimize memory if enabled
        if self.optimize_memory and len(df) > 10000:
            self.logger.info("Optimizing DataFrame memory usage")
            initial_memory = get_memory_usage(df)

            df, optimization_info = optimize_dataframe_dtypes(
                df,
                categorical_threshold=0.5,
                downcast_integers=True,
                downcast_floats=True,
            )

            self.logger.info(
                f"Memory optimization: {initial_memory['total_mb']:.2f}MB -> "
                f"{optimization_info['memory_after_mb']:.2f}MB "
                f"(saved {optimization_info['memory_saved_percent']:.1f}%)"
            )

            # Adjust chunk size if adaptive
            if self.adaptive_chunk_size:
                self._adjust_chunk_size(df)

        return df

    def _adjust_chunk_size(self, df: pd.DataFrame) -> None:
        """
        Adjust chunk size based on available memory.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        """
        memory_info = get_memory_usage(df)
        row_memory = memory_info["per_row_bytes"]

        # Target to use at most 100MB per chunk
        target_memory_mb = 100
        target_rows = int((target_memory_mb * 1024 * 1024) / row_memory)

        # Adjust chunk size within reasonable bounds
        self.chunk_size = max(1000, min(target_rows, 100000))

        # Ensure chunk size doesn't exceed data size
        self.chunk_size = min(self.chunk_size, len(df))

        if self.chunk_size != self.original_chunk_size:
            self.logger.info(
                f"Adjusted chunk size from {self.original_chunk_size} "
                f"to {self.chunk_size} based on memory usage"
            )

    def _sample_aligned(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        sample_size: Optional[int],
        columns: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sample aligned rows from both original and transformed DataFrames based on shared indices,
        optionally using column mapping, with optional Dask support.

        Parameters
        ----------
        original_df : pd.DataFrame
            The original dataset.
        transformed_df : pd.DataFrame
            The transformed dataset.
        sample_size : Optional[int]
            Number of rows to sample. If None, return full datasets.
        columns : Optional[List[str]]
            List of original columns to keep. If None, use all columns from original_df.
        column_mapping : Optional[Dict[str, str]]
            Mapping from original columns to transformed columns (e.g., {"age": "age_enrich"}).
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Sampled original and transformed DataFrames with aligned indices and matched columns.
        """
        column_mapping = column_mapping or {}

        # Step 1: Select columns
        columns = columns or list(original_df.columns)
        original_df = original_df[columns].copy()
        mapped_columns = [column_mapping.get(col, col) for col in columns]

        # Step 2: Validate and map transformed columns
        missing_cols = [
            col for col in mapped_columns if col not in transformed_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Mapped columns {missing_cols} not found in transformed_df"
            )

        transformed_df = transformed_df[mapped_columns].copy()
        transformed_df.columns = columns  # Rename to match original

        # Step 3: Get common indices
        common_indices = original_df.index.intersection(transformed_df.index)
        if sample_size is not None and sample_size > len(common_indices):
            raise ValueError(
                f"sample_size={sample_size} exceeds number of common rows: {len(common_indices)}"
            )

        # Step 4: Sample indices
        if sample_size is not None:
            sampled_indices = pd.Series(list(common_indices)).sample(
                n=sample_size, random_state=random_state
            )
        else:
            sampled_indices = common_indices

        # Step 5: Helper for Dask or Pandas extraction
        def extract(df: pd.DataFrame) -> pd.DataFrame:
            if self.use_dask:
                ddf = convert_to_dask(
                    df.loc[common_indices],
                    npartitions=self.npartitions,
                    dask_partition_size=self.dask_partition_size or "100MB",
                    logger=self.logger,
                )
                return ddf.loc[sampled_indices].compute()
            return df.loc[sampled_indices]

        original_sample = extract(original_df)
        transformed_sample = extract(transformed_df)

        return original_sample.reset_index(drop=True), transformed_sample.reset_index(
            drop=True
        )

    def _calculate_metrics_with_config(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
    ) -> Dict[str, Any]:
        """
        Calculate metrics between the original and transformed DataFrames using either
        pandas or Dask, depending on the configuration.

        Parameters
        ----------
        original_df : pd.DataFrame
            Original (non-transformed) dataset
        transformed_df : pd.DataFrame
            Transformed version of the dataset
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker for tracking progress in UI or logs

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the calculated metrics
        """
        # Return early if either DataFrame is empty
        if original_df.empty or transformed_df.empty:
            self.logger.warning(
                "Empty original or transformed DataFrame provided. Skipping metric calculation."
            )
            return {}

        self.logger.info("Calculating metrics with config")
        if progress_tracker:
            progress_tracker.update(0, {"step": "Initializing metric calculation"})

        # Decide processing engine based on configuration
        if self.use_dask:
            return self._process_with_dask_or_fallback(
                original_df, transformed_df, progress_tracker
            )
        else:
            return self._process_with_pandas(
                original_df, transformed_df, progress_tracker
            )

    def _process_with_dask_or_fallback(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> Dict[str, Any]:
        """
        Try processing using Dask. If it fails, fall back to pandas processing.

        Parameters
        ----------
        original_df : pd.DataFrame
        transformed_df : pd.DataFrame
        progress_tracker : Optional[HierarchicalProgressTracker]

        Returns
        -------
        Dict[str, Any]
            Calculated metrics dictionary
        """
        try:
            # Log Dask processing details
            self.logger.info("Parallel Enabled: Using Dask")
            self.logger.info(f"Parallel Workers: {self.npartitions}")
            self.logger.info(
                f"Dask Partition Size: {self.dask_partition_size or '100MB'}"
            )

            # Convert both DataFrames to Dask for parallel processing
            original_ddf, org_npartitions = self._prepare_dask_dataframe(
                original_df, "original"
            )
            self.logger.info(
                f"Processing {len(original_ddf)} rows in {org_npartitions} chunks with Dask"
            )
            transformed_ddf, trans_npartitions = self._prepare_dask_dataframe(
                transformed_df, "transformed"
            )
            self.logger.info(
                f"Processing {len(transformed_ddf)} rows in {trans_npartitions} chunks with Dask"
            )
            # Delegate to subclass Dask-based metric calculator
            return self.calculate_metrics_with_dask(
                original_ddf=original_ddf,
                transformed_ddf=transformed_ddf,
                progress_tracker=progress_tracker,
                **self.process_kwargs,
            )

        except Exception as e:
            # If Dask fails, log and fall back to pandas
            self.logger.warning(f"Dask processing failed: {e}. Falling back to pandas.")
            return self._process_with_pandas(
                original_df, transformed_df, progress_tracker
            )

    def _prepare_dask_dataframe(
        self, df: pd.DataFrame, label: str
    ) -> Tuple[dd.DataFrame, int]:
        """
        Convert a pandas DataFrame to Dask DataFrame with configured partitioning.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        label : str
            Label used for logging ("original" or "transformed")

        Returns
        -------
        Tuple[dask.DataFrame, int]
            The Dask DataFrame and number of partitions created
        """
        self.logger.info(f"Converting {label} DataFrame with {len(df)} rows to Dask")

        ddf, npartitions = convert_to_dask(
            df=df,
            dask_npartitions=self.npartitions,
            dask_partition_size=self.dask_partition_size or "100MB",
            logger=self.logger,
        )

        self.logger.info(
            f"{label.title()} DataFrame partitioned into {npartitions} chunks"
        )
        return ddf, npartitions

    def _process_with_pandas(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> Dict[str, Any]:
        """
        Process metrics using standard pandas (non-parallel).

        Parameters
        ----------
        original_df : pd.DataFrame
        transformed_df : pd.DataFrame
        progress_tracker : Optional[HierarchicalProgressTracker]

        Returns
        -------
        Dict[str, Any]
            Calculated metrics dictionary
        """
        try:
            self.logger.info("Parallel Disabled: Using Pandas")
            self.logger.info("Parallel Workers: 0")

            # Call base metric calculation logic using pandas
            return self.calculate_metrics(
                original_df=original_df,
                transformed_df=transformed_df,
                progress_tracker=progress_tracker,
                **self.process_kwargs,
            )
        except Exception as e:
            self.logger.error(f"Pandas processing failed: {e}")
            raise RuntimeError(f"Failed to calculate metrics with pandas: {e}")

    def calculate_metrics(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Calculate the metric values - core evaluation logic.. Must be implemented by subclasses.

        Parameters:
        -----------
        original_df : pd.DataFrame
            DataFrame original to process
        transformed_df : pd.DataFrame
            DataFrame transformed to process
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for monitoring
        **kwargs : Any
            Additional parameters for processing

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch
        """
        raise NotImplementedError("Subclasses must implement calculate_metrics method")

    def calculate_metrics_with_dask(
        self,
        original_ddf: dd.DataFrame,
        transformed_ddf: dd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Calculate the metric values using Dask - must be implemented by subclasses.

        Parameters:
        -----------
        original_ddf : dd.DataFrame
            DataFrame original to process
        transformed_ddf : dd.DataFrame
            DataFrame transformed to process
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for monitoring
        **kwargs : Any
            Additional parameters for processing

        Returns:
        --------
        Dict[str, Any]
            - Metric result dictionary
        """
        raise NotImplementedError(
            "Subclasses must implement calculate_metrics_with_dask method"
        )

    def _collect_basic_metrics(
        self,
        original_data: pd.Series,
        transformed_data: pd.Series,
    ) -> Dict[str, Any]:
        """
        Collect comprehensive metrics.

        Parameters:
        -----------
        original_data : pd.Series
            Original data
        transformed_data : pd.Series
            Transformed data
        mask : pd.Series, optional
            Processing mask

        Returns:
        --------
        Dict[str, Any]
            Collected metrics
        """
        # Basic metrics
        metrics: Dict[str, Any] = {}

        # Add performance metrics
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            metrics.update(
                {
                    "duration_seconds": round(duration, 2),
                    "records_processed": self.process_count,
                    "records_per_second": (
                        round(self.process_count / duration, 2) if duration > 0 else 0
                    ),
                    "chunk_count": (
                        int((self.process_count - 1) / self.chunk_size + 1)
                        if self.process_count > 0
                        else 0
                    ),
                }
            )

        # Add filtering metrics
        metrics.update(
            {
                "total_original_records": len(original_data),
                "total_transformed_records": len(transformed_data),
            }
        )

        # Add memory metrics if optimization was performed
        if self.optimize_memory:
            metrics["chunk_size_used"] = self.chunk_size
            metrics["adaptive_chunk_size"] = self.adaptive_chunk_size

        return metrics

    def _handle_visualizations(
        self,
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
                    f"[DIAG] Name Operation: {self.name}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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
                name=f"VizThread-{self.name}",
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
                description=f"{self.name} {viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_operation(
                    f"{self.name} {viz_type} visualization",
                    details={"artifact_type": "png", "path": str(path)},
                )

        return visualization_paths

    def _generate_visualizations(
        self,
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
        metrics : Dict[str, Any]
            Collected metrics for visualization
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
        raise NotImplementedError(
            "Subclasses must implement _generate_visualizations method"
        )

    def _cleanup_memory(
        self,
        sampled_orig: Optional[pd.DataFrame] = None,
        sampled_trans: Optional[pd.DataFrame] = None,
        original_df: Optional[pd.DataFrame] = None,
        transformed_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        references and optionally calling garbage collection.

        Parameters:
        -----------
        sampled_orig : pd.DataFrame, optional
            Sampled original data to clear from memory
        sampled_trans : pd.DataFrame, optional
            Sampled transformed data to clear from memory
        original_df : pd.DataFrame, optional
            Original data to clear from memory
        transformed_df : pd.DataFrame, optional
            Transformed data to clear from memory
        """
        # Delete references
        if sampled_orig is not None:
            del sampled_orig
        if sampled_trans is not None:
            del sampled_trans
        if original_df is not None:
            del original_df
        if transformed_df is not None:
            del transformed_df

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Clear process kwargs
        if hasattr(self, "process_kwargs"):
            self.process_kwargs = {}

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Force garbage collection
        force_garbage_collection()

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
            cache_key = self._generate_cache_key(df)
            self.logger.debug(f"Checking cache for key: {cache_key}")

            cached_result = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(f"Using cached result for {self.name} generalization")

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
                    f"Metrics (cached)",
                    details={
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
                    description=f"{desc_suffix} (cached)",
                    category=category,
                )
                artifacts_restored += 1

                if reporter:
                    reporter.add_operation(
                        f"{desc_suffix} (cached)",
                        details={
                            "artifact_type": artifact_type,
                            "path": str(artifact_path),
                        },
                    )
            else:
                self.logger.warning(f"Cached file not found: {artifact_path}")

        # Restore main output and metrics and mapping artifacts
        restore_file_artifact(
            cached.get("metrics_file"),
            "json",
            "generalization metrics",
            Constants.Artifact_Category_Metrics,
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
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        metrics: Dict[str, Any],
        task_dir: Path,
        visualization_paths: Dict[str, Path] = {},
        metrics_result_path: Optional[str] = None,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        transformed_df : pd.DataFrame
            Transformed output data
        metrics : Dict[str, Any]
            Metrics collected during the operation
        visualization_paths : Dict[str, Path]
            Paths to generated visualizations
        task_dir : Path
            Task directory
        metrics_result_path : Optional[str]
            Path to the metrics result file
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
            cache_key = self._generate_cache_key(original_df)

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
                    "original_length": len(original_df),
                    "transformed_length": len(transformed_df),
                    "original_null_count": int(
                        original_df.isna().sum().sum()
                        if isinstance(original_df, pd.DataFrame)
                        else original_df.isna().sum()
                    ),
                    "transformed_null_count": (
                        int(transformed_df.isna().sum().sum())
                        if isinstance(transformed_df, pd.DataFrame)
                        else int(transformed_df.isna().sum())
                    ),
                },
                "metrics_file": metrics_result_path,  # Path to metrics file
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")

            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(f"Successfully saved metrics results to cache")
            else:
                self.logger.warning(f"Failed to save metrics results to cache")

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
            operation_name=self.__class__.__name__,
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
            "name": self.name,
            "description": self.description,
            "version": self.version,
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

    def _normalize_metric(
        self, value: float, min_value: float = 0.0, max_value: float = 1.0
    ) -> float:
        """
        Normalize metric value to [0,1] range if required.

        Parameters:
        -----------
        value : float
            The metric value to normalize.
        min_value : float, optional
            Minimum possible value (default: 0.0).
        max_value : float, optional
            Maximum possible value (default: 1.0).

        Returns:
        --------
        float
            Normalized metric value.
        """
        if not self.normalize or value is None:
            return value
        if max_value == min_value:
            return 0.0
        return (value - min_value) / (max_value - min_value)

    def _get_metric_metadata(self) -> Dict[str, Any]:
        """
        Return metric metadata (range, interpretation, etc.).

        Returns:
        --------
        Dict[str, Any]
            Metadata for each metric key.
        """
        # Example metadata; extend as needed for your metrics
        return {
            "duration_seconds": {
                "description": "Total duration of the operation in seconds",
                "type": "float",
                "range": [0, None],
                "interpretation": "Lower is better",
            },
            "records_processed": {
                "description": "Number of records processed",
                "type": "int",
                "range": [0, None],
                "interpretation": "Higher means more data processed",
            },
            "total_original_records": {
                "description": "Total number of records in the original data",
                "type": "int",
                "range": [0, None],
            },
            "total_transformed_records": {
                "description": "Total number of records in the transformed data",
                "type": "int",
                "range": [0, None],
            },
            # Add more metric metadata as needed
        }
