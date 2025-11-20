"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Multi-Valued Field Profiler Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides analyzers and operations for profiling multi-valued fields (MVF) in tabular datasets.
  It includes parsing, distribution analysis, value and combination dictionary creation, and visualization capabilities.
  The module supports chunked, parallel, and Dask-based processing for large datasets and integrates with the PAMOLA.CORE operation framework.

Key Features:
  - Parsing and detection of multi-valued field formats (string, JSON, CSV)
  - Frequency and combination analysis for MVF values
  - Value and combination dictionary generation with configurable thresholds
  - Visualization generation for value and combination distributions
  - Efficient chunked, parallel, and Dask-based processing for large datasets
  - Robust error handling, progress tracking, and operation logging
  - Caching and efficient repeated analysis
  - Integration with PAMOLA.CORE operation framework for standardized input/output
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from pamola_core.anonymization.commons.visualization_utils import (
    generate_visualization_filename,
)
from pamola_core.common.constants import Constants
from pamola_core.profiling.commons.mvf_utils import (
    aggregate_mvf_analysis,
    analyze_mvf_field_with_dask,
    analyze_mvf_field_with_parallel,
    analyze_mvf_in_chunks,
    detect_mvf_format,
    generate_analysis_distribution_vis,
    parse_mvf,
    create_value_dictionary,
    create_combinations_dictionary,
    analyze_value_count_distribution,
    estimate_resources,
    process_mvf_partition,
)
from pamola_core.profiling.schemas.mvf_core_schema import MVFAnalysisOperationConfig
from pamola_core.utils.io import (
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Configure logger
logger = logging.getLogger(__name__)


class MVFAnalyzer:
    """
    Analyzer for multi-valued fields.

    This analyzer provides methods for analyzing MVF fields, including
    parsing, frequency distributions, combinations analysis, and value count analysis.
    """

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        field_name: str,
        top_n: int = 20,
        parse_args: Dict[str, Any] = None,
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        task_logger: Optional[logging.Logger] = None,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a multi-valued field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top values to include in the results
        parse_args : Dict[str, Any] = {}
            Additional arguments for parsing the MVF
        chunk_size : int
            Number of records to process in each chunk
        use_dask : bool
            Whether to use Dask for parallel processing
        npartitions : int, optional
            Number of Dask partitions to use
        use_vectorization : bool
            Whether to use vectorized operations
        parallel_processes : int, optional
            Number of parallel processes to use
        task_logger : Optional[logging.Logger] = None,
            Logger for task-specific logging
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
            Progress tracker for monitoring the analysis progress

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        if task_logger:
            global logger
            logger = task_logger
            logger.info(f"Analyzing MVF field: {field_name}")

        results = None
        flag_processed = False
        logger.info("Process with config")

        if use_dask:
            try:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Dask")
                logger.info(f"Parallel Workers: {npartitions}")
                logger.info(f"Using dask processing with chunk size {chunk_size}")
                if progress_tracker:
                    progress_tracker.total = 3  # Setup, Processing, Finalization
                    progress_tracker.update(0, {"step": "Setting up dask processing"})

                logger.info("Process using Dask")

                results = analyze_mvf_field_with_dask(
                    df=df,
                    field_name=field_name,
                    top_n=top_n,
                    parse_args=parse_args,
                    chunk_size=chunk_size,
                    npartitions=npartitions,
                    progress_tracker=progress_tracker,
                    task_logger=logger,
                )

                if results is not None:
                    flag_processed = True
                    logger.info("Completed using Dask")

            except Exception as e:
                logger.warning(
                    f"Error in dask processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and use_vectorization:
            try:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Joblib")
                logger.info(f"Parallel Workers: {parallel_processes}")
                logger.info(f"Using vectorized processing with chunk size {chunk_size}")
                if progress_tracker:
                    progress_tracker.update(
                        0, {"step": "Setting up vectorized processing"}
                    )

                logger.info("Process using Joblib")
                results = analyze_mvf_field_with_parallel(
                    df=df,
                    field_name=field_name,
                    top_n=top_n,
                    parse_args=parse_args,
                    chunk_size=chunk_size,
                    n_jobs=parallel_processes,
                    progress_tracker=progress_tracker,
                    task_logger=logger,
                )

                if results is not None:
                    flag_processed = True
                    logger.info("Completed using Joblib")

            except Exception as e:
                logger.warning(
                    f"Error in vectorized processing: {e}, falling back to chunk processing"
                )

        if not flag_processed and chunk_size > 1:
            try:
                # Regular chunk processing
                logger.info(f"Processing in chunks with chunk size {chunk_size}")
                total_chunks = (len(df) + chunk_size - 1) // chunk_size
                logger.info(f"Total chunks to process: {total_chunks}")
                if progress_tracker:
                    progress_tracker.update(
                        0,
                        {
                            "step": "Processing in chunks",
                            "total_chunks": total_chunks,
                        },
                    )

                logger.info("Process using chunk")

                results = analyze_mvf_in_chunks(
                    df=df,
                    field_name=field_name,
                    top_n=top_n,
                    parse_args=parse_args,
                    chunk_size=chunk_size,
                    progress_tracker=progress_tracker,
                    task_logger=logger,
                )

                if results is not None:
                    flag_processed = True
                    logger.info("Completed using chunk")

            except Exception as e:
                logger.warning(
                    f"Error in chunk processing: {e}, falling back to chunk processing"
                )

        if not flag_processed:
            logger.info("Fallback process as usual")
            # Map partitions for processing
            parsed_df = process_mvf_partition(
                partition=df,
                field_name=field_name,
                parse_args=parse_args,
            )
            # Aggregate results
            total_records = len(df)
            null_count = df[field_name].isna().sum()
            results = aggregate_mvf_analysis(
                parsed_df, total_records, null_count, field_name, top_n=top_n
            )
            logger.info("Process using normal")
            flag_processed = True

        return results

    @staticmethod
    def parse_field(
        df: pd.DataFrame, field_name: str, format_type: Optional[str] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Parse an MVF field and add a new column with parsed values.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the field to parse
        format_type : str, optional
            Format type hint for parsing
        **kwargs : dict
            Additional parameters to pass to parse_mvf

        Returns:
        --------
        pd.DataFrame
            DataFrame with an additional column containing parsed values
        """
        if field_name not in df.columns:
            logger.error(f"Field {field_name} not found in DataFrame")
            return df

        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()

        # Create the new column name
        parsed_column = f"parsed_{field_name}"

        # Parse values and add to new column
        logger.info(f"Parsing MVF field: {field_name}")

        parse_args = kwargs.copy()
        parse_args["format_type"] = format_type

        # Apply parsing to each value
        result_df[parsed_column] = result_df[field_name].apply(
            lambda x: parse_mvf(x, **parse_args) if not pd.isna(x) else []
        )

        return result_df

    @staticmethod
    def create_value_dictionary(
        df: pd.DataFrame,
        field_name: str,
        min_frequency: int = 1,
        parse_args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Create a dictionary of values with frequencies for an MVF field.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the field
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        parse_args : Any
            Additional parameters for parsing

        Returns:
        --------
        pd.DataFrame
            DataFrame with values and frequencies
        """
        return create_value_dictionary(
            df=df,
            field_name=field_name,
            min_frequency=min_frequency,
            parse_args=parse_args,
        )

    @staticmethod
    def create_combinations_dictionary(
        df: pd.DataFrame,
        field_name: str,
        min_frequency: int = 1,
        parse_args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Create a dictionary of value combinations with frequencies for an MVF field.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the field
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        parse_args : dict
            Additional parameters for parsing

        Returns:
        --------
        pd.DataFrame
            DataFrame with combinations and frequencies
        """
        return create_combinations_dictionary(
            df=df,
            field_name=field_name,
            min_frequency=min_frequency,
            parse_args=parse_args,
        )

    @staticmethod
    def analyze_value_counts(
        df: pd.DataFrame, field_name: str, **kwargs
    ) -> Dict[str, int]:
        """
        Analyze the distribution of value counts per record in an MVF field.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the field
        **kwargs : dict
            Additional parameters for parsing

        Returns:
        --------
        Dict[str, int]
            Distribution of value counts
        """
        return analyze_value_count_distribution(
            df=df, field_name=field_name, parse_args=kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing an MVF field.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the field to analyze

        Returns:
        --------
        Dict[str, Any]
            Estimated resource requirements
        """
        return estimate_resources(df, field_name)


@register(version="1.0.0")
class MVFOperation(FieldOperation):
    """
    Operation for analyzing multi-valued fields.

    This operation wraps the MVFAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(
        self,
        field_name: str,
        top_n: int = 20,
        min_frequency: int = 1,
        format_type: Optional[str] = None,
        parse_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the MVF operation.

        Parameters
        ----------
        field_name : str
            The name of the field to analyze.
        top_n : int
            Number of top values to include in the results.
        min_frequency : int
            Minimum frequency for inclusion in the dictionary.
        format_type : str, optional
            Format type hint for parsing (default: None).
        parse_kwargs : dict, optional
            Additional parameters for parsing.
        **kwargs : dict
            Additional parameters passed to FieldOperation.
        """
        # --- Default description ---
        kwargs.setdefault(
            "description",
            f"Analysis of multi-valued field '{field_name}'",
        )

        # --- Build config ---
        config = MVFAnalysisOperationConfig(
            field_name=field_name,
            top_n=top_n,
            min_frequency=min_frequency,
            format_type=format_type,
            parse_kwargs=parse_kwargs or {},
            **kwargs,
        )

        # Inject config to parent kwargs
        kwargs["config"] = config

        # --- Initialize base FieldOperation ---
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # --- Apply config values to self ---
        for key, value in config.to_dict().items():
            setattr(self, key, value)

        # --- Operation metadata ---
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
        Execute the mvf analysis operation.

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
            self.logger = kwargs.get("logger", self.logger)
            self.logger.info(f"Starting {self.name} operation at {self.start_time}")
            df = None
            result = OperationResult(status=OperationStatus.PENDING)

            # Prepare directories for artifacts
            directories = self._prepare_directories(task_dir)

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Save configuration to task directory
            self.save_config(task_dir)

            # Create DataWriter for consistent file operations
            writer = DataWriter(
                task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker
            )

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up progress tracking with proper steps
            # Main steps: 1. Cache check, 2. Validation, 3. Data loading, 4. Processing, 5. Metrics, 6. Visualization, 7. Save output
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
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS,
                        current_steps,
                        {
                            "step": "Starting MVF analysis",
                        },
                        main_progress,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update progress tracker: {e}")

            # Get DataFrame from data source
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )

            # Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                # Step 1: Check if we have a cached result
                if main_progress:
                    current_steps += 1
                    self._update_progress_tracker(
                        TOTAL_MAIN_STEPS, current_steps, "Checking cache", main_progress
                    )

                # Load left dataset for check cache
                df = load_data_operation(
                    data_source, dataset_name, **settings_operation
                )
                if df is None:
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message="No valid DataFrame found in data source",
                    )

                self.logger.info(
                    f"Field: '{self.field_name}' loaded with {len(df)} records."
                )

                self.logger.info("Checking operation cache...")
                cache_result = self._check_cache(df=df, reporter=reporter)
                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if main_progress:
                        self._update_progress_tracker(
                            TOTAL_MAIN_STEPS,
                            current_steps,
                            "Complete (cached)",
                            main_progress,
                        )

                    return cache_result

            # Step 2: Data Loading
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Data Loading", main_progress
                )

            try:
                # Load DataFrame
                if df is None:
                    df = load_data_operation(
                        data_source, dataset_name, **settings_operation
                    )
                    if df is None:
                        return OperationResult(
                            status=OperationStatus.ERROR,
                            error_message="No valid DataFrame found in data source",
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
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Validation", main_progress
                )

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame",
                )

            # Add operation to reporter
            reporter.add_operation(
                f"Analyzing multi-valued field: {self.field_name}",
                details={
                    "field_name": self.field_name,
                    "top_n": self.top_n,
                    "min_frequency": self.min_frequency,
                    "operation_type": "mvf_analysis",
                },
            )

            # Step 4: Processing progress tracker
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS, current_steps, "Processing data", main_progress
                )

            # Detect format type if not set
            if self.format_type is None:
                self.format_type = detect_mvf_format(df[self.field_name])

            try:
                self.logger.info(f"Processing with field_name: {self.field_name}")

                # Create child progress tracker for chunk processing
                data_tracker = None
                if main_progress and hasattr(main_progress, "create_subtask"):
                    try:
                        data_tracker = main_progress.create_subtask(
                            total=3,
                            description="MVF analysis processing",
                            unit="steps",
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not create child progress tracker: {e}"
                        )

                # Combine parsing arguments
                parse_args = {**self.parse_kwargs, "format_type": self.format_type}

                # Execute the analyzer
                self.logger.info("Executing MVF analysis")
                analysis_results = MVFAnalyzer.analyze(
                    df=df,
                    field_name=self.field_name,
                    top_n=self.top_n,
                    parse_args=parse_args,
                    chunk_size=self.chunk_size,
                    use_dask=self.use_dask,
                    npartitions=self.npartitions,
                    use_vectorization=self.use_vectorization,
                    parallel_processes=self.parallel_processes,
                    task_logger=self.logger,
                    progress_tracker=data_tracker,
                )

                # Check for errors
                if "error" in analysis_results:
                    self.logger.error(analysis_results["error"])
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=analysis_results["error"],
                    )

                # Create and save value dictionary
                self.logger.info("Creating value dictionary")
                values_dict = MVFAnalyzer.create_value_dictionary(
                    df=df,
                    field_name=self.field_name,
                    min_frequency=self.min_frequency,
                    parse_args=parse_args,
                )

                # Create and save combinations dictionary
                self.logger.info("Creating combinations dictionary")
                combinations_dict = MVFAnalyzer.create_combinations_dictionary(
                    df=df,
                    field_name=self.field_name,
                    min_frequency=self.min_frequency,
                    parse_args=parse_args,
                )

                # Close child progress tracker
                if data_tracker:
                    try:
                        data_tracker.close()
                    except:
                        pass

                self.logger.info(
                    f"Processed data: {len(df)} records, dtype: {df.dtypes}"
                )

                # Log sample of processed data
                if len(df) > 0:
                    self.logger.debug(
                        f"Sample of processed data (first 5 rows): {df.head(5).to_dict(orient='records')}"
                    )
            except Exception as e:
                error_message = f"Processing error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )

            # Step 5: Metrics Calculation
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS,
                    current_steps,
                    "Metrics Calculation",
                    main_progress,
                )

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initialize metrics in scope
            metrics = {}

            try:
                # Save identifier statistics
                if analysis_results and "error" not in analysis_results:
                    # Generate metrics file name
                    statistics_filename = f"{self.field_name}_{self.name}_statistical_analysis_metrics_{operation_timestamp}"

                    # Write metrics to persistent storage/artifact repository
                    statistics_metrics_result = writer.write_metrics(
                        metrics=analysis_results,
                        name=statistics_filename,
                        timestamp_in_name=False,
                        encryption_key=(
                            self.encryption_key if self.use_encryption else None
                        ),
                    )

                    # Add simple metrics (int, float, str, bool) to the result object
                    for key, value in analysis_results.items():
                        if isinstance(value, (int, float, str, bool)):
                            result.add_metric(key, value)

                    result.add_artifact(
                        artifact_type="json",
                        path=statistics_metrics_result.path,
                        description=f"{self.name} profiling on {self.field_name} statistical analysis metrics",
                        category=Constants.Artifact_Category_Metrics,
                    )

                    # Add identifier stats to metrics dictionary
                    metrics["analysis_results"] = analysis_results
                    metrics["statistics_metrics_result_path"] = (
                        statistics_metrics_result.path
                    )
            except Exception as e:
                error_message = f"Error calculating metrics: {str(e)}"
                self.logger.warning(error_message)
                # Continue execution - metrics failure is not critical

            # Step 6: Visualizations
            if main_progress:
                current_steps += 1
                self._update_progress_tracker(
                    TOTAL_MAIN_STEPS,
                    current_steps,
                    "Generating Visualizations",
                    main_progress,
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
                        analysis_results=analysis_results,
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

            # Step 7: Save output data
            self.logger.info("Step 7: Saving output data")
            if main_progress:
                current_steps += 1
                main_progress.update(current_steps, {"step": "Saving output data"})

            # Save output data if required
            if self.save_output:
                try:
                    # Save values dictionary
                    values_str_path = self._save_output_data(
                        df=values_dict,
                        suffix="values_dictionary",
                        is_encryption_required=self.use_encryption,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        **kwargs,
                    )

                    # Save combinations dictionary
                    combinations_str_path = self._save_output_data(
                        df=combinations_dict,
                        suffix="combinations_dictionary",
                        is_encryption_required=self.use_encryption,
                        writer=writer,
                        result=result,
                        reporter=reporter,
                        progress_tracker=main_progress,
                        timestamp=operation_timestamp,
                        **kwargs,
                    )
                except Exception as e:
                    error_message = f"Error saving output data: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR, error_message=error_message
                    )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        original_data=df,
                        metrics=metrics,
                        visualization_paths=visualization_paths,
                        task_dir=task_dir,
                        values_str_path=values_str_path,
                        combinations_str_path=combinations_str_path,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            # Cleanup memory
            self._cleanup_memory(df, values_dict, combinations_dict)

            # Record end time
            self.end_time = time.time()

            # Report completion
            if reporter:
                # Create the details dictionary with checks for all values
                details = {
                    "records_processed": self.process_count,
                    "execution_time": (
                        self.end_time - self.start_time
                        if self.end_time and self.start_time
                        else None
                    ),
                }

                # Add the operation to the reporter
                reporter.add_operation(
                    f"Analyzing MVF {self.field_name} completed",
                    details={
                        "unique_values": analysis_results.get("unique_values", 0),
                        "unique_combinations": analysis_results.get(
                            "unique_combinations", 0
                        ),
                        "avg_values_per_record": analysis_results.get(
                            "avg_values_per_record", 0
                        ),
                        "null_percentage": analysis_results.get("null_percentage", 0),
                    },
                )

            self.logger.info(
                f"Processing completed {self.name} operation in {self.end_time - self.start_time:.2f} seconds"
            )

            # Set success status
            result.status = OperationStatus.SUCCESS
            return result

        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error in analyzing MVF operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing MVF field {self.field_name}: {str(e)}",
            )

    def _cleanup_memory(
        self,
        original_df: Optional[pd.DataFrame] = None,
        values_dict: Optional[Dict[str, Any]] = None,
        combinations_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Clean up memory after operation completes.

        For large datasets, explicitly free memory by deleting
        references and optionally calling garbage collection.

        Parameters:
        -----------
        original_df : pd.DataFrame, optional
            Original DataFrame to clear from memory
        values_dict : dict, optional
            Dictionary containing values to clear from memory
        combinations_dict : dict, optional
            Dictionary containing combinations to clear from memory
        """
        # Delete references
        if original_df is not None:
            del original_df
        if values_dict is not None:
            del values_dict
        if combinations_dict is not None:
            del combinations_dict

        # Clear operation cache
        if hasattr(self, "operation_cache"):
            self.operation_cache = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith("_temp_"):
                delattr(self, attr_name)

        # Optional: Force garbage collection for large datasets
        # Uncomment if memory pressure is an issue
        # import gc
        # gc.collect()

    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        # Get basic operation parameters
        parameters = self._get_basic_parameters()

        # Add operation-specific parameters (could be overridden by subclasses)
        parameters.update(self._get_cache_parameters())

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(data)

        # Use the operation_cache utility to generate a consistent cache key
        return self.operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash,
        )

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
        params = {
            "field_name": self.field_name,
            "top_n": self.top_n,
            "min_frequency": self.min_frequency,
            "format_type": self.format_type,
            "parse_kwargs": self.parse_kwargs,
            "chunk_size": self.chunk_size,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "use_cache": self.use_cache,
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
            "visualization_theme": self.visualization_theme,
            "visualization_backend": self.visualization_backend,
            "visualization_strict": self.visualization_strict,
            "visualization_timeout": self.visualization_timeout,
            "force_recalculation": self.force_recalculation,
            "generate_visualization": self.generate_visualization,
            "output_format": self.output_format,
        }

        return params

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

    def _save_to_cache(
        self,
        original_data: pd.DataFrame,
        metrics: Dict[str, Any],
        visualization_paths: Dict[str, Path],
        task_dir: Path,
        values_str_path: Optional[str] = None,
        combinations_str_path: Optional[str] = None,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_data : pd.DataFrame
            Original input data
        metrics : Dict[str, Any]
            Metrics collected during the operation
        visualization_paths : Dict[str, Path]
            Paths to generated visualizations
        task_dir : Path
            Task directory
        values_str_path : Optional[str] = None
        combinations_str_path : Optional[str] = None

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(original_data[self.field_name])

            # Prepare metadata for cache
            operation_params = self._get_basic_parameters()
            operation_params.update(self._get_cache_parameters())

            self.logger.debug(f"Operation parameters for cache: {operation_params}")

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "parameters": operation_params,
                "data_info": {
                    "original_length": len(original_data),
                    "original_null_count": int(original_data.isna().sum().sum()),
                },
                "visualizations": {
                    k: str(v) for k, v in visualization_paths.items()
                },  # Paths to visualizations
                "values_str_path": values_str_path,
                "combinations_str_path": combinations_str_path,
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
                self.logger.info(
                    f"Successfully saved {self.field_name} profiling results to cache"
                )
            else:
                self.logger.warning(
                    f"Failed to save {self.field_name} profiling results to cache"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

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
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if not cached_result:
                self.logger.info("No cached result found, proceeding with operation")
                return None

            self.logger.info(
                f"Using cached result for {self.field_name} of {self.name} profiling"
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
                    f"{self.name} profiling of {self.field_name} (cached)",
                    details={
                        "field_name": self.field_name,
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

        # Restore main output and metrics artifacts
        restore_file_artifact(
            cached.get("values_str_path"),
            self.output_format,
            "generalized data",
            Constants.Artifact_Category_Output,
        )

        # Restore main output and metrics artifacts
        restore_file_artifact(
            cached.get("combinations_str_path"),
            self.output_format,
            "generalized data",
            Constants.Artifact_Category_Output,
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

    def _save_output_data(
        self,
        df: pd.DataFrame,
        suffix: str,
        is_encryption_required: bool,
        writer: DataWriter,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Save the processed output data.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to save
        suffix : str
            The suffix to append to the output filename
        is_encryption_required : bool
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
        if df.empty:
            return None

        if progress_tracker:
            progress_tracker.update(0, {"step": "Saving output data"})

        custom_kwargs = {k: v for k, v in kwargs.items() if k != "encryption_key"}

        # Generate standardized output filename with timestamp
        filename = generate_visualization_filename(
            self.field_name,
            f"{self.name}_{suffix}",
            "output",
            timestamp=timestamp,
        )

        # Use the DataWriter to save the DataFrame
        output_result = writer.write_dataframe(
            df=df,
            name=filename,
            format=self.output_format,
            subdir="output",
            timestamp_in_name=False,
            encryption_key=self.encryption_key if is_encryption_required else None,
            **custom_kwargs,
        )

        # Get the output path
        path = output_result.path

        # Register output artifact with the result
        result.add_artifact(
            artifact_type=self.output_format,
            path=path,
            description=f"{self.field_name} generalized data",
            category=Constants.Artifact_Category_Output,
        )

        # Report to reporter
        if reporter:
            reporter.add_operation(
                f"{self.field_name} generalized data",
                details={"artifact_type": self.output_format, "path": str(path)},
            )

        return str(path)

    def _handle_visualizations(
        self,
        analysis_results: Dict[str, Any],
        task_dir: Path,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate and save visualizations with thread-safe context support.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            A dictionary containing various analysis results.
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
        **kwargs: Any
            Additional keyword arguments for visualization functions.
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
                    f"[DIAG] Field: {self.field_name}, Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
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
                        analysis_results=analysis_results,
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
        analysis_results: Dict[str, Any],
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Generate required visualizations for the merge operation using visualization utilities.

        Parameters
        ----------
        analysis_results : Dict[str, Any]
            A dictionary containing various analysis results.
        task_dir : Path
            The base directory where all task-related outputs (including visualizations) will be saved.
        vis_theme : Optional[str]
            The theme to use for visualizations.
        vis_backend : Optional[str]
            The backend to use for rendering visualizations.
        vis_strict : bool
            Whether to enforce strict visualization rules.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Tracker for monitoring progress of the visualization generation.
        timestamp : Optional[str]
            Timestamp to include in visualization filenames.
        **kwargs : Any
            Additional keyword arguments for visualization functions.

        Returns
        -------
        dict
            A dictionary mapping visualization types to their corresponding file paths.
        """
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        visualization_paths = {}

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
            f"[VIZ] Starting visualization generation for {self.field_name}"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        try:
            # Step 1: Prepare data
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparing visualization data"})

            self.logger.debug(
                f"[VIZ] Data prepared for visualization:analysis_results: {len(analysis_results)}"
            )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # Generate analysis results visualization
            if analysis_results and "error" not in analysis_results:
                visualization_paths.update(
                    generate_analysis_distribution_vis(
                        analysis_results=analysis_results,
                        field_label=self.field_name,
                        operation_name=self.name,
                        task_dir=viz_dir,
                        timestamp=timestamp,
                        top_n=self.top_n,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        visualization_paths=visualization_paths,
                        **kwargs,
                    )
                )

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(3, {"step": "Finalizing visualizations"})

        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")

        return visualization_paths

    def _update_progress_tracker(
        self,
        TOTAL_MAIN_STEPS: int,
        n: int,
        step_name: str,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> None:
        """
        Helper to update progress tracker for the step.
        """
        if progress_tracker:
            progress_tracker.total = TOTAL_MAIN_STEPS  # Ensure total steps is set
            progress_tracker.update(
                n,
                {
                    "step": step_name,
                    "operation": f"{self.name}",
                    "field_name": f"{self.field_name}",
                },
            )

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare directories for artifacts following PAMOLA.CORE conventions.

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

        # Create standard directories following PAMOLA.CORE conventions
        directories["root"] = task_dir
        directories["output"] = task_dir / "output"
        directories["dictionaries"] = task_dir / "dictionaries"
        directories["logs"] = task_dir / "logs"
        directories["cache"] = task_dir / "cache"

        # Ensure all directories exist
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories


def analyze_mvf_fields(
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    mvf_fields: List[str],
    **kwargs,
) -> Dict[str, OperationResult]:
    """
    Analyze multiple MVF fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    mvf_fields : List[str]
        List of MVF fields to analyze
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top values to include in results (default: 20)
        - min_frequency: int, minimum frequency for inclusion in dictionary (default: 1)
        - format_type: str, format type hint for parsing (default: None)
        - parse_kwargs: dict, additional parameters for MVF parsing

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    dataset_name = kwargs.get("dataset_name", "main")
    df = load_data_operation(data_source, dataset_name)
    if df is None:
        reporter.add_operation(
            "MVF fields analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"},
        )
        return {}

    # Extract operation parameters from kwargs
    top_n = kwargs.get("top_n", 20)
    min_frequency = kwargs.get("min_frequency", 1)
    format_type = kwargs.get("format_type", None)
    parse_kwargs = kwargs.get("parse_kwargs", {})

    # Report on fields to be analyzed
    reporter.add_operation(
        "MVF fields analysis",
        details={
            "fields_count": len(mvf_fields),
            "fields": mvf_fields,
            "top_n": top_n,
            "min_frequency": min_frequency,
            "parameters": {
                k: v
                for k, v in kwargs.items()
                if isinstance(v, (str, int, float, bool))
            },
        },
    )

    # Track progress if enabled
    track_progress = kwargs.get("track_progress", True)
    overall_tracker = None

    if track_progress and mvf_fields:
        from pamola_core.utils.progress import ProgressTracker

        overall_tracker = ProgressTracker(
            total=len(mvf_fields),
            description=f"Analyzing {len(mvf_fields)} MVF fields",
            unit="fields",
            track_memory=True,
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(mvf_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(
                        0, {"field": field, "progress": f"{i + 1}/{len(mvf_fields)}"}
                    )

                logger.info(f"Analyzing MVF field: {field}")

                # Create and execute operation
                operation = MVFOperation(
                    field, top_n=top_n, min_frequency=min_frequency
                )

                # Create kwargs for this field
                field_kwargs = kwargs.copy()
                field_kwargs["format_type"] = format_type
                field_kwargs["parse_kwargs"] = parse_kwargs

                result = operation.execute(
                    data_source, task_dir, reporter, **field_kwargs
                )

                # Store result
                results[field] = result

                # Update overall tracker after successful analysis
                if overall_tracker:
                    if result.status == OperationStatus.SUCCESS:
                        overall_tracker.update(
                            1, {"field": field, "status": "completed"}
                        )
                    else:
                        overall_tracker.update(
                            1,
                            {
                                "field": field,
                                "status": "error",
                                "error": result.error_message,
                            },
                        )

            except Exception as e:
                logger.error(f"Error analyzing MVF field {field}: {e}", exc_info=True)

                reporter.add_operation(
                    f"Analyzing {field} field",
                    status="error",
                    details={"error": str(e)},
                )

                # Update overall tracker in case of error
                if overall_tracker:
                    overall_tracker.update(1, {"field": field, "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(
        1 for r in results.values() if r.status == OperationStatus.SUCCESS
    )
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    reporter.add_operation(
        "MVF fields analysis completed",
        details={
            "fields_analyzed": len(results),
            "successful": success_count,
            "failed": error_count,
        },
    )

    return results
