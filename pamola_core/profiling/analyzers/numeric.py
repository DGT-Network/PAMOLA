"""
Numeric data analyzer for the project.

This module provides analyzers and operations for numeric fields,
including statistical analysis, distribution analysis, outlier detection,
and normality testing.
"""

import time
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd

from pamola_core.profiling.commons.numeric_utils import (
    prepare_numeric_data, handle_large_dataframe, analyze_numeric_chunk,
    calculate_extended_stats, calculate_percentiles, calculate_histogram,
    detect_outliers, test_normality, create_empty_stats,
    process_with_dask, process_with_joblib
)
from pamola_core.utils.io import write_json, get_timestamped_filename, load_data_operation, load_settings_operation
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.visualization import (
    create_histogram, create_boxplot, create_correlation_pair_plot
)
from pamola_core.utils.ops.op_registry import register
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
# Configure logger
logger = logging.getLogger(__name__)


class NumericAnalyzer:
    """
    Analyzer for numeric fields.

    This analyzer provides methods for analyzing numeric fields, including
    statistical analysis, distribution analysis, outlier detection, and normality testing.
    """

    def analyze(self,
                df: pd.DataFrame,
                field_name: str,
                bins: int = 10,
                should_detect_outliers: bool = True,
                should_test_normality: bool = True,
                chunk_size: int = 10000,
                use_dask: bool = False,
                npartitions: int = 2,
                use_vectorization: bool = False,
                parallel_processes: int = 2,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a numeric field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        bins : int
            Number of bins for histogram analysis
        should_detect_outliers : bool
            Whether to detect outliers
        should_test_normality : bool
            Whether to perform normality testing
        chunk_size : int
            Size of data chunks for processing large datasets
        use_dask : bool, optional
            Whether to use Dask for processing (default: False).
        npartitions : int, optional
            Number of partitions use with Dask (default: 1).
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False).
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1).
        **kwargs : dict
            Additional parameters for the analysis:
            - near_zero_threshold: threshold for detecting "near zero" values (default: 1e-10)
            - semantic_analysis: whether to perform semantic analysis on special fields (default: True)
            - track_progress: whether to track progress (default: True)
            - use_chunks: whether to process in chunks for large dataframes (default: True)
            - chunk_size: size of chunks for processing (default: 10000)
            - normality_test_method: method for normality testing (default: 'all')

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        logger.info(f"Analyzing numeric field: {field_name}")

        # Parameters
        near_zero_threshold = kwargs.get('near_zero_threshold', 1e-10)
        track_progress = kwargs.get('track_progress', True)
        use_chunks = kwargs.get('use_chunks', True)
        is_large_df = use_chunks and len(df) > chunk_size
        semantic_analysis = kwargs.get('semantic_analysis', True)
        normality_test_method = kwargs.get('normality_test_method', 'all')

        # Initialize progress tracker if enabled
        progress = None
        if track_progress:
            progress_steps = 6 if should_test_normality else 5  # Add step for normality testing
            progress = HierarchicalProgressTracker(
                total=progress_steps,
                description=f"Analyzing numeric field: {field_name}",
                unit="steps",
                track_memory=is_large_df
            )
            progress.update(0, {"step": "Preparation"})

        # Check if field exists
        if field_name not in df.columns:
            if progress:
                progress.close()
            return {'error': f"Field {field_name} not found in DataFrame"}

        total_rows = len(df)

        # Data preparation
        valid_data, null_count, non_null_count = prepare_numeric_data(df, field_name)
        valid_count = len(valid_data)

        if progress:
            progress.update(1, {"step": "Basic statistics", "valid_records": valid_count})

        # Calculate statistics
        if valid_count == 0:
            stats_dict = create_empty_stats()

            if should_detect_outliers:
                stats_dict['outliers'] = {
                    'iqr': None,
                    'lower_bound': None,
                    'upper_bound': None,
                    'count': 0,
                    'percentage': 0
                }
        else:
            # For large datasets, process in chunks
            stats_dict = create_empty_stats()
            flag_processed = False
            try:
                if not flag_processed and use_dask and npartitions > 1:
                    logger.info("Parallel Enabled")
                    logger.info("Parallel Engine: Dask")
                    logger.info(f"Parallel Workers: {npartitions}")
                    logger.info(f"Using dask processing with chunk size {chunk_size}")

                    stats_dict = process_with_dask(
                        df,
                        field_name,
                        analyze_numeric_chunk,
                        npartitions,
                        chunk_size,
                        near_zero_threshold=near_zero_threshold
                    )

                    flag_processed = True
            except Exception as e:
                logger.exception(f"Error in using dask processing: {e}")
                flag_processed = False

            try:
                if not flag_processed and use_vectorization and parallel_processes > 1:
                    logger.info("Parallel Enabled")
                    logger.info("Parallel Engine: Joblib")
                    logger.info(f"Parallel Workers: {parallel_processes}")
                    logger.info(f"Using vectorized processing with chunk size {chunk_size}")

                    stats_dict = process_with_joblib(
                        df,
                        field_name,
                        analyze_numeric_chunk,
                        parallel_processes,
                        chunk_size,
                        near_zero_threshold=near_zero_threshold
                    )

                    flag_processed = True
            except Exception as e:
                logger.exception(f"Error in using joblib processing: {e}")
                flag_processed = False

            try:
                if not flag_processed and is_large_df:
                    logger.info(f"Processing in chunks with chunk size {chunk_size}")
                    total_chunks = (len(df) + chunk_size - 1) // chunk_size
                    logger.info(f"Total chunks to process: {total_chunks}")

                    stats_dict = handle_large_dataframe(
                        df,
                        field_name,
                        analyze_numeric_chunk,
                        chunk_size,
                        near_zero_threshold=near_zero_threshold
                    )

                    flag_processed = True
            except Exception as e:
                logger.exception(f"Error in using chunks processing: {e}")
                flag_processed = False

            try:
                if not flag_processed:
                    logger.info("Fallback process as usual")

                    stats_dict = calculate_extended_stats(valid_data, near_zero_threshold)

                    flag_processed = True
            except Exception as e:
                logger.exception(f"Error in processing: {e}")
                flag_processed = False

            if not flag_processed:
                logger.exception(f"Error in processing")

            if progress:
                progress.update(1, {"step": "Descriptive statistics", "stats_calculated": True})

            # Calculate percentiles
            stats_dict['percentiles'] = calculate_percentiles(valid_data)

            # Calculate histogram data
            stats_dict['histogram'] = calculate_histogram(valid_data, bins)

            if progress:
                progress.update(1, {"step": "Histogram calculation", "bins": bins})

            # Semantic analysis for special fields
            if semantic_analysis:
                field_lower = field_name.lower()
                if any(term in field_lower for term in ['salary', 'price', 'cost', 'amount', 'budget']):
                    stats_dict['zero_analysis'] = {
                        'count': int(stats_dict['zero_count']),
                        'percentage': float(stats_dict['zero_percentage']),
                        'interpretation': 'Zero values in monetary fields may indicate unspecified amounts or placeholder values'
                    }

            # Detect outliers if requested
            if should_detect_outliers:
                outlier_results = detect_outliers(valid_data)
                stats_dict['outliers'] = outlier_results

                if progress:
                    progress.update(1, {
                        "step": "Outlier detection",
                        "outliers_found": outlier_results.get('count', 0)
                    })

                # Test normality if requested
                if should_test_normality and valid_count >= 8:  # Minimum sample size for meaningful normality tests
                    try:
                        normality_results = test_normality(valid_data, normality_test_method)
                        stats_dict['normality'] = normality_results

                        if progress:
                            progress.update(1, {
                                "step": "Normality testing",
                                "is_normal": normality_results.get('is_normal', False)
                            })
                    except Exception as e:
                        logger.warning(f"Error during normality testing for {field_name}: {e}")
                        stats_dict['normality'] = {
                            'error': str(e),
                            'is_normal': False
                        }
                else:
                    stats_dict['normality'] = {
                        'is_normal': False,
                        'message': 'Insufficient data for normality testing' if valid_count < 8 else 'Normality testing skipped'
                    }

                    if progress and should_test_normality:
                        progress.update(1, {
                            "step": "Normality testing",
                            "skipped": True,
                            "reason": "Insufficient data" if valid_count < 8 else "User choice"
                        })

        # Overall statistics
        results = {
            'total_rows': total_rows,
            'null_count': int(null_count),
            'non_null_count': int(non_null_count),
            'valid_count': int(valid_count),
            'null_percentage': float(round((null_count / total_rows) * 100, 2)) if total_rows > 0 else 0.0,
            'stats': stats_dict
        }

        if progress:
            progress.update(1, {"step": "Finalization", "complete": True})
            progress.close()

        return results

    def estimate_resources(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the numeric field.

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
        # Basic resource estimation based on DataFrame size
        row_count = len(df)

        # Memory estimation (rough approximation)
        if field_name in df.columns:
            # Estimate based on field type and non-null values
            non_null_count = df[field_name].notna().sum()
            bytes_per_value = 8  # 8 bytes for float64

            # Base memory for analysis
            base_memory_mb = 50

            # Memory for field data
            field_memory_mb = (non_null_count * bytes_per_value) / (1024 * 1024)

            # Memory for intermediate calculations
            calc_memory_mb = field_memory_mb * 3  # Multiplication factor for intermediate calculations

            # Total estimated memory
            estimated_memory_mb = base_memory_mb + field_memory_mb + calc_memory_mb

            # Estimated time (very rough approximation)
            if row_count < 10000:
                estimated_time_seconds = 1
            elif row_count < 100000:
                estimated_time_seconds = 5
            elif row_count < 1000000:
                estimated_time_seconds = 30
            else:
                estimated_time_seconds = 120

            return {
                'estimated_memory_mb': estimated_memory_mb,
                'estimated_time_seconds': estimated_time_seconds,
                'recommended_chunk_size': min(100000, max(10000, row_count // 10)),
                'use_chunks_recommended': row_count > 100000
            }
        else:
            # Field not found, return minimal estimates
            return {
                'estimated_memory_mb': 10,
                'estimated_time_seconds': 1,
                'error': f"Field {field_name} not found in DataFrame"
            }

@register(override=True)
class NumericOperation(FieldOperation):
    """
    Operation for analyzing numeric fields.

    This operation extends the FieldOperation base class and provides methods for
    executing numeric field analysis, including visualization generation and result reporting.
    """

    def __init__(self, field_name: str,
                 bins: int = 10,
                 detect_outliers: bool = True,
                 test_normality: bool = True,
                 near_zero_threshold: float = 1e-10,
                 generate_visualization: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = 'numeric',
                 visualization_theme: Optional[str] = None,
                 visualization_backend: Optional[str] = "plotly",
                 visualization_strict: bool = False,
                 visualization_timeout: int = 120,
                 chunk_size: int = 10000,
                 use_dask: bool = False,
                 npartitions: int = 2,
                 use_vectorization: bool = False,
                 parallel_processes: int = 2,
                 use_cache: bool = True,
                 description: str = "",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 encryption_mode: Optional[str] = None):
        """
        Initialize a numeric field operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to analyze
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        visualization_theme : str, optional
            Theme to use for visualizations (default: None - uses system default)
        visualization_backend : str, optional
            Backend to use for visualizations: "plotly" or "matplotlib" (default: None - uses system default)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout in seconds for visualization generation (default: 120)
        chunk_size : int
            Size of data chunks for processing large datasets
        use_dask : bool, optional
            Whether to use Dask for processing (default: False).
        npartitions : int, optional
            Number of partitions use with Dask (default: 1).
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False).
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1).
        use_cache : bool
            Whether to use caching for intermediate results
        description : str
            Description of the operation (optional)
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of numeric field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode
            )

        self.bins = bins
        self.detect_outliers = detect_outliers
        self.test_normality = test_normality
        self.analyzer = NumericAnalyzer()
        self.near_zero_threshold = near_zero_threshold
        self.generate_visualization = generate_visualization
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        self.use_cache = use_cache

        # Set up performance tracking variables
        self.is_encryption_required = False
        self.start_time = None
        self.end_time = None
        self.execution_time = 0
        self.process_count = 0

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the numeric field analysis operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation:
            - near_zero_threshold: float, threshold for near-zero detection
            - generate_visualization: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames
            - profile_type: str, type of profiling for organizing artifacts
            - dataset_name: str - Name of dataset - main
            - force_recalculation: bool - Force operation even if cached results exist - False
            - use_dask: bool - Use Dask for large dataset processing - False
            - parallel_processes: int - Number of parallel processes to use - 1
            - generate_visualization: bool - Create visualizations - True
            - save_output: bool - Save processed data to output directory - True
            - encrypt_output: bool - Override encryption setting for outputs - False
            - visualization_theme: str - Override theme for visualizations - None
            - visualization_backend: str - Override backend for visualizations - None
            - visualization_strict: bool - Override strict mode for visualizations - False
            - visualization_timeout: int - Override timeout for visualizations - 120

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        global logger
        if kwargs.get("logger"):
            logger = kwargs.get("logger")

        try:
            # Initialize timing and result
            self.start_time = time.time()
            self.process_count = 0
            result = OperationResult(status=OperationStatus.SUCCESS)

            # Decompose kwargs and introduce variables for clarity
            near_zero_threshold = kwargs.get('near_zero_threshold', self.near_zero_threshold)
            generate_visualization = kwargs.get('generate_visualization', self.generate_visualization)
            include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
            profile_type = kwargs.get('profile_type', self.profile_type)
            force_recalculation = kwargs.get("force_recalculation", False)
            use_dask = kwargs.get("use_dask", self.use_dask)
            parallel_processes = kwargs.get("parallel_processes", 1)
            save_output = kwargs.get("save_output", True)
            is_encryption_required = (kwargs.get("encrypt_output", False) or self.use_encryption)
            encryption_key = kwargs.get('encryption_key', None)

            self.use_dask = use_dask
            self.parallel_processes = parallel_processes
            self.include_timestamp = include_timestamp
            self.is_encryption_required = is_encryption_required

            # Extract visualization parameters
            self.visualization_theme = kwargs.get("visualization_theme", self.visualization_theme)
            self.visualization_backend = kwargs.get("visualization_backend", self.visualization_backend)
            self.visualization_strict = kwargs.get("visualization_strict", self.visualization_strict)
            self.visualization_timeout = kwargs.get("visualization_timeout", self.visualization_timeout)

            logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, "
                f"strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up directories
            dirs = self._prepare_directories(task_dir)
            output_dir = dirs["output"]
            visualizations_dir = dirs["visualizations"]
            dictionaries_dir = dirs["dictionaries"]
            cache_dir = dirs["cache"]

            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparation", "field": self.field_name})

            # Get DataFrame from data source
            dataset_name = kwargs.get('dataset_name', "main")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame"
                )

            # Add operation to reporter
            reporter.add_operation(f"Analyzing numeric field: {self.field_name}", details={
                "field_name": self.field_name,
                "bins": self.bins,
                "detect_outliers": self.detect_outliers,
                "test_normality": self.test_normality,
                "operation_type": "numeric_analysis"
            })

            # Check for cached results if caching is enabled
            if self.use_cache and not force_recalculation:
                cached_result = self._check_cache(df, reporter, task_dir, **kwargs)
                if cached_result:
                    logger.info(f"Using cached results for {self.field_name}")

                    # Update progress if tracker provided
                    if progress_tracker:
                        progress_tracker.update(6, {"step": "Loaded from cache", "field": self.field_name})

                    return cached_result

            # Adjust progress tracker total steps if provided
            total_steps = 4
            if generate_visualization:
                total_steps += 2

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = self.analyzer.analyze(
                df=df,
                field_name=self.field_name,
                bins=self.bins,
                should_detect_outliers=self.detect_outliers,
                should_test_normality=self.test_normality,
                chunk_size=self.chunk_size,
                use_dask=self.use_dask,
                npartitions=self.npartitions,
                use_vectorization=self.use_vectorization,
                parallel_processes=self.parallel_processes,
                near_zero_threshold=near_zero_threshold,
                track_progress=progress_tracker is not None
            )

            artifacts = []
            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analysis complete", "field": self.field_name})

            # Save analysis results to JSON
            stats_filename = get_timestamped_filename(f"{self.field_name}_stats", "json", include_timestamp)
            stats_path = output_dir / stats_filename

            encryption_mode = get_encryption_mode(analysis_results, **kwargs)
            write_json(analysis_results, stats_path, encryption_key=encryption_key, encryption_mode=encryption_mode)
            result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis", category=Constants.Artifact_Category_Output)

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), f"{self.field_name} statistical analysis")
            artifacts.append({
                "artifact_type": "json",
                "path": str(stats_path),
                "description": f"{self.field_name} statistical analysis",
                "category": Constants.Artifact_Category_Output
            })

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualizations if requested
            if generate_visualization and self.visualization_backend is not None:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualizations"})

                kwargs_encryption = {
                    "use_encryption": kwargs.get('use_encryption', False),
                    "encryption_key": encryption_key
                }
                visualization_paths = self._handle_visualizations(
                    df=df,
                    analysis_results=analysis_results,
                    include_timestamp=include_timestamp,
                    result=result,
                    reporter=reporter,
                    vis_dir=visualizations_dir,
                    vis_theme=self.visualization_theme,
                    vis_backend=self.visualization_backend,
                    vis_strict=self.visualization_strict,
                    vis_timeout=self.visualization_timeout,
                    progress_tracker=progress_tracker,
                    **kwargs_encryption
                )

                artifacts.extend(visualization_paths)

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Visualizations complete"})

            # Add metrics to the result
            stats_dict = analysis_results.get('stats', {})
            result.add_metric("total_rows", analysis_results.get('total_rows', 0))
            result.add_metric("null_count", analysis_results.get('null_count', 0))
            result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
            result.add_metric("min", stats_dict.get('min'))
            result.add_metric("max", stats_dict.get('max'))
            result.add_metric("mean", stats_dict.get('mean'))
            result.add_metric("median", stats_dict.get('median'))

            if 'outliers' in stats_dict:
                result.add_metric("outliers_count", stats_dict['outliers'].get('count', 0))
                result.add_metric("outliers_percentage", stats_dict['outliers'].get('percentage', 0))

            if 'normality' in stats_dict:
                result.add_metric("is_normal", stats_dict['normality'].get('is_normal', False))

            # Cache results if caching is enabled
            if self.use_cache:
                self._save_to_cache(
                    df=df,
                    analysis_results=analysis_results,
                    artifacts=artifacts,
                    task_dir=task_dir
                )

            # Update progress to completion
            if progress_tracker:
                progress_tracker.update(1, {"step": "Operation complete", "status": "success"})

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "valid_values": analysis_results.get('valid_count', 0),
                "null_percentage": analysis_results.get('null_percentage', 0),
                "min": stats_dict.get('min'),
                "max": stats_dict.get('max'),
                "mean": stats_dict.get('mean'),
                "outliers": stats_dict.get('outliers', {}).get('count', 0) if 'outliers' in stats_dict else 0,
                "is_normal": stats_dict.get('normality', {}).get('is_normal',
                                                                 False) if 'normality' in stats_dict else False
            })

            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            return result
        except Exception as e:
            logger.exception(f"Error in numeric operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing numeric field {self.field_name}: {str(e)}",
                exception=e,
            )

    def _generate_visualizations(self,
                                 df: pd.DataFrame,
                                 analysis_results: Dict[str, Any],
                                 vis_dir: Path,
                                 include_timestamp: bool,
                                 vis_theme: Optional[str],
                                 vis_backend: Optional[str],
                                 vis_strict: bool,
                                 **kwargs_encryption) -> List[Dict[str, str]]:
        """
        Generate visualizations for the numeric field analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors

        Returns:
        --------
        List[Dict[str, str]]
            Information of visualizations
        """
        visualization_paths = []

        stats_dict = analysis_results.get('stats', {})
        valid_data = pd.to_numeric(df[self.field_name], errors='coerce').dropna()

        # Generate histogram visualization if we have histogram data
        if 'histogram' in stats_dict:
            histogram_data = stats_dict['histogram']

            if histogram_data and histogram_data.get('bins') and histogram_data.get('counts'):
                hist_filename = get_timestamped_filename(f"{self.field_name}_distribution", "png", include_timestamp)
                hist_path = vis_dir / hist_filename

                # Create histogram using the visualization module
                min_value = stats_dict.get('min')
                max_value = stats_dict.get('max')

                title = f"Distribution of {self.field_name} histogram"
                if min_value is not None and max_value is not None:
                    title += f" (min: {min_value:.2f}, max: {max_value:.2f})"

                # Use the histogram data or regenerate from valid_data
                if len(valid_data) > 0:
                    hist_result = create_histogram(
                        data=valid_data,
                        output_path=str(hist_path),
                        title=title,
                        x_label=self.field_name,
                        y_label="Frequency",
                        bins=self.bins,
                        theme=vis_theme,
                        backend=vis_backend,
                        strict=vis_strict,
                        **kwargs_encryption
                    )

                    if not hist_result.startswith("Error"):
                        visualization_paths.append({
                            "artifact_type": "png",
                            "path": str(hist_path),
                            "description": title,
                            "category": Constants.Artifact_Category_Visualization
                        })

        # Generate boxplot visualization if we have enough data
        if len(valid_data) > 5:
            boxplot_filename = get_timestamped_filename(f"Boxplot of {self.field_name}", "png", include_timestamp)
            boxplot_path = vis_dir / boxplot_filename

            # Create boxplot using the visualization module
            boxplot_result = create_boxplot(
                data={self.field_name: valid_data.tolist()},
                output_path=str(boxplot_path),
                title=f"Boxplot of {self.field_name}",
                y_label=self.field_name,
                show_points=True,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs_encryption
            )

            if not boxplot_result.startswith("Error"):
                visualization_paths.append({
                    "artifact_type": "png",
                    "path": str(boxplot_path),
                    "description": f"Boxplot of {self.field_name}",
                    "category": Constants.Artifact_Category_Visualization
                })

        # Generate Q-Q plot for normality if requested and we have results
        if self.test_normality and 'normality' in stats_dict and len(valid_data) > 10:
            qq_filename = get_timestamped_filename(f"{self.field_name} Q-Q plot (normality test)", "png", include_timestamp)
            qq_path = vis_dir / qq_filename

            # Generate synthetic normal data for comparison
            np.random.seed(42)  # For reproducibility
            normal_data = np.random.normal(loc=valid_data.mean(), scale=valid_data.std(), size=len(valid_data))
            normal_data.sort()

            # Sort the actual data for Q-Q plot
            sorted_data = valid_data.sort_values()

            # Create scatter plot comparing theoretical quantiles to actual data
            normality_info = stats_dict['normality']
            is_normal = normality_info.get('is_normal', False)
            p_value = normality_info.get('shapiro', {}).get('p_value', None)

            title = f"{self.field_name} Q-Q plot (normality test)"
            if p_value is not None:
                title += f" (Shapiro p-value: {p_value:.4f})"

            qq_result = create_correlation_pair_plot(
                x_data=normal_data,
                y_data=sorted_data,
                output_path=str(qq_path),
                title=title,
                x_label="Theoretical Quantiles",
                y_label="Sample Quantiles",
                add_trendline=True,
                add_histogram=False,
                method="Q-Q Plot",
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs_encryption
            )

            if not qq_result.startswith("Error"):
                visualization_paths.append({
                    "artifact_type": "png",
                    "path": str(qq_path),
                    "description": f"{self.field_name} Q-Q plot (normality test)",
                    "category": Constants.Artifact_Category_Visualization
                })

        return visualization_paths

    def _check_cache(
            self,
            df: pd.DataFrame,
            reporter: Any,
            task_dir: Path,
            **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation
        reporter : Any
            The reporter to log artifacts to
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
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            operation_cache_dir = OperationCache(cache_dir=task_dir/"cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache_dir.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            if cached_data:
                logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Restore artifacts from cache
                artifacts_restored = 0

                # Add artifacts
                artifacts = cached_data.get("artifacts", [])
                for artifact in artifacts:
                    if artifact and isinstance(artifact, dict):
                        if Path(artifact.get("path")).exists():
                            artifacts_restored += 1
                            cached_result.add_artifact(
                                artifact.get("artifact_type"),
                                artifact.get("path"),
                                artifact.get("description"),
                                category=artifact.get("category")
                            )
                            reporter.add_artifact(
                                artifact.get("artifact_type"),
                                artifact.get("path"),
                                artifact.get("description")
                            )

                # Add cached metrics to result
                analysis_results = cached_data.get("analysis_results", {})
                stats_dict = analysis_results.get('stats', {})

                cached_result.add_metric("total_rows", analysis_results.get('total_rows', 0))
                cached_result.add_metric("null_count", analysis_results.get('null_count', 0))
                cached_result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
                cached_result.add_metric("min", stats_dict.get('min'))
                cached_result.add_metric("max", stats_dict.get('max'))
                cached_result.add_metric("mean", stats_dict.get('mean'))
                cached_result.add_metric("median", stats_dict.get('median'))

                if 'outliers' in stats_dict:
                    cached_result.add_metric("outliers_count", stats_dict['outliers'].get('count', 0))
                    cached_result.add_metric("outliers_percentage", stats_dict['outliers'].get('percentage', 0))

                if 'normality' in stats_dict:
                    cached_result.add_metric("is_normal", stats_dict['normality'].get('is_normal', False))

                # Add final operation status to reporter
                outliers = stats_dict.get('outliers', {})
                normality = stats_dict.get('normality', {})
                reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                    "valid_values": analysis_results.get('valid_count', 0),
                    "null_percentage": analysis_results.get('null_percentage', 0),
                    "min": stats_dict.get('min'),
                    "max": stats_dict.get('max'),
                    "mean": stats_dict.get('mean'),
                    "outliers": outliers.get('count', 0),
                    "is_normal": normality.get('is_normal', False)
                })

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric("cache_timestamp", cached_data.get("timestamp", "unknown"))
                cached_result.add_metric("artifacts_restored", artifacts_restored)

                return cached_result

            logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
            self,
            df: pd.DataFrame,
            analysis_results: Dict[str, Any],
            artifacts: List[Dict[str, str]],
            task_dir: Path
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation
        analysis_results : dict
            Analysis results to cache
        artifacts : list of dict
            Artifacts
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
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            # Generate operation cache
            operation_cache_dir = OperationCache(cache_dir=task_dir/"cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "analysis_results": analysis_results,
                "artifacts": artifacts,
                "data_info": {
                    "df_length": len(df)
                }
            }

            # Save to cache
            logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache_dir.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
            )

            if success:
                logger.info(f"Successfully saved results to cache")
            else:
                logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            return False

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
            "field_name": self.field_name,
            "bins": self.bins,
            "detect_outliers": self.detect_outliers,
            "test_normality": self.test_normality,
            "near_zero_threshold": self.near_zero_threshold,
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
        import json
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format='iso')
        except Exception as e:
            logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _handle_visualizations(
            self,
            df: pd.DataFrame,
            analysis_results: Dict[str, Any],
            vis_dir: Path,
            include_timestamp: bool,
            result: OperationResult,
            reporter: Any,
            vis_theme: Optional[str],
            vis_backend: Optional[str],
            vis_strict: bool,
            vis_timeout: int,
            progress_tracker: Optional[HierarchicalProgressTracker],
            **kwargs_encryption
    ) -> List[Dict[str, str]]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
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

        Returns:
        --------
        List[Dict[str, str]]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        logger.info(f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s")

        try:
            import threading
            import contextvars

            visualization_paths = []
            visualization_error = None

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                logger.info(f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}")
                logger.info(f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}")

                start_time = time.time()

                try:
                    # Log context variables
                    logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        logger.info(f"[DIAG] Context vars count: {len(list(current_context))}")
                    except Exception as ctx_e:
                        logger.warning(f"[DIAG] Could not inspect context: {ctx_e}")

                    # Generate visualizations with visualization context parameters
                    logger.info(f"[DIAG] Calling _generate_visualizations...")
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
                            logger.debug(f"Could not create child progress tracker: {e}")

                    # Generate visualizations
                    visualization_paths = self._generate_visualizations(
                        df=df,
                        analysis_results=analysis_results,
                        vis_dir=vis_dir,
                        include_timestamp=include_timestamp,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend,
                        vis_strict=vis_strict,
                        **kwargs_encryption
                    )

                    # Close visualization progress tracker
                    if viz_progress:
                        try:
                            viz_progress.close()
                        except:
                            pass

                    elapsed = time.time() - start_time
                    logger.info(
                        f"[DIAG] Visualization completed in {elapsed:.2f}s, generated {len(visualization_paths)} files"
                    )
                except Exception as e:
                    elapsed = time.time() - start_time
                    visualization_error = e
                    logger.error(f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}")
                    logger.error(f"[DIAG] Stack trace:", exc_info=True)

            # Copy context for the thread
            logger.info(f"[DIAG] Preparing to launch visualization thread...")
            ctx = contextvars.copy_context()

            # Create thread with context
            viz_thread = threading.Thread(
                target=ctx.run,
                args=(generate_viz_with_diagnostics,),
                name=f"VizThread-",
                daemon=False,  # Changed from True to ensure proper cleanup
            )

            logger.info(f"[DIAG] Starting visualization thread with timeout={vis_timeout}s")
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    logger.info(f"[DIAG] Visualization thread still running after {elapsed:.1f}s...")

            if viz_thread.is_alive():
                logger.error(f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout")
                logger.error(f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}")
                visualization_paths = []
            elif visualization_error:
                logger.error(f"[DIAG] Visualization failed with error: {visualization_error}")
                visualization_paths = []
            else:
                total_time = time.time() - thread_start_time
                logger.info(f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s")
                logger.info(f"[DIAG] Generated visualizations: {[v.get('description', v.get('path', str(v))) for v in visualization_paths]}")
        except Exception as e:
            logger.error(f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}")
            logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = []

        # Register visualization artifacts
        for vis_result  in visualization_paths:
            artifact_type = vis_result["artifact_type"]
            path = vis_result["path"]
            description = vis_result["description"]

            # Add to result
            result.add_artifact(
                artifact_type=artifact_type,
                path=path,
                description=description,
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type=artifact_type, path=path, description=description
                )

        return visualization_paths

    def _prepare_directories(
            self,
            task_dir: Path
    ) -> Dict[str, Path]:
        """
        Prepare standard directories for storing operation artifacts.

        Parameters:
        -----------
        task_dir : Path
            Base directory for the task

        Returns:
        --------
        Dict[str, Path]
            Dictionary with standard directory paths
        """
        from pamola_core.utils.io import ensure_directory

        # Create standard directories
        directories = {
            "output": task_dir / "output",
            "dictionaries": task_dir / "dictionaries",
            "visualizations": task_dir / "visualizations",
            "cache": task_dir / "cache",
        }

        # Ensure directories exist
        for dir_path in directories.values():
            ensure_directory(dir_path)

        return directories

def analyze_numeric_fields(data_source: DataSource,
                           task_dir: Path,
                           reporter: Any,
                           numeric_fields: List[str] = None,
                           **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple numeric fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    numeric_fields : List[str], optional
        List of numeric fields to analyze. If None, tries to find numeric fields automatically.
    **kwargs : dict
        Additional parameters for the operations

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    dataset_name = kwargs.get('dataset_name', "main")
    settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
    df = load_data_operation(data_source, dataset_name, **settings_operation)
    if df is None:
        reporter.add_operation("Numeric fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # If no numeric fields specified, try to find them
    if numeric_fields is None:
        numeric_fields = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_fields.append(col)

    # Report on fields to be analyzed
    reporter.add_operation("Numeric fields analysis", details={
        "fields_count": len(numeric_fields),
        "fields": numeric_fields,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and numeric_fields:
        overall_tracker = HierarchicalProgressTracker(
            total=len(numeric_fields),
            description=f"Analyzing {len(numeric_fields)} numeric fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(numeric_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(numeric_fields)}"})

                logger.info(f"Analyzing numeric field: {field}")

                # Create and execute operation
                operation = NumericOperation(field)
                result = operation.execute(data_source, task_dir, reporter, **kwargs)

                # Store result
                results[field] = result

                # Update overall tracker after successful analysis
                if overall_tracker:
                    if result.status == OperationStatus.SUCCESS:
                        overall_tracker.update(1, {"field": field, "status": "completed"})
                    else:
                        overall_tracker.update(1, {"field": field, "status": "error",
                                                   "error": result.error_message})

            except Exception as e:
                logger.error(f"Error analyzing numeric field {field}: {e}", exc_info=True)

                reporter.add_operation(f"Analyzing {field} field", status="error",
                                       details={"error": str(e)})

                # Update overall tracker in case of error
                if overall_tracker:
                    overall_tracker.update(1, {"field": field, "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(1 for r in results.values() if r.status == OperationStatus.SUCCESS)
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    reporter.add_operation("Numeric fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results