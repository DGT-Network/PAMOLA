"""
Numeric data analyzer for the project.

This module provides analyzers and operations for numeric fields,
including statistical analysis, distribution analysis, outlier detection,
and normality testing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd

from pamola_core.profiling.commons.numeric_utils import (
    prepare_numeric_data, handle_large_dataframe, analyze_numeric_chunk,
    calculate_extended_stats, calculate_percentiles, calculate_histogram,
    detect_outliers, test_normality, create_empty_stats
)
from pamola_core.utils.io import write_json, get_timestamped_filename, load_data_operation, load_settings_operation
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.visualization import (
    create_histogram, create_boxplot, create_correlation_pair
)
from pamola_core.utils.ops.op_registry import register
from pamola_core.common.constants import Constants
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
        chunk_size = kwargs.get('chunk_size', 10000)  
        is_large_df = use_chunks and len(df) > chunk_size
        semantic_analysis = kwargs.get('semantic_analysis', True)
        normality_test_method = kwargs.get('normality_test_method', 'all')

        # Initialize progress tracker if enabled
        progress = None
        if track_progress:
            progress_steps = 6 if should_test_normality else 5  # Add step for normality testing
            progress = ProgressTracker(
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
            if is_large_df:
                logger.info(f"Processing large field {field_name} with {valid_count} valid records using chunks")

                stats_dict = handle_large_dataframe(
                    df,
                    field_name,
                    analyze_numeric_chunk,
                    chunk_size,
                    near_zero_threshold=near_zero_threshold
                )
            else:
                # Standard processing for smaller datasets
                stats_dict = calculate_extended_stats(valid_data, near_zero_threshold)

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
                 near_zero_threshold: int = 1e-10,
                 generate_plots: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = 'numeric',
                 description: str = "",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
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
        description : str
            Description of the operation (optional)
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of numeric field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )
        
        self.bins = bins
        self.detect_outliers = detect_outliers
        self.test_normality = test_normality
        self.analyzer = NumericAnalyzer()
        self.near_zero_threshold = near_zero_threshold
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
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
        progress_tracker : ProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation:
            - near_zero_threshold: float, threshold for near-zero detection
            - generate_plots: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames
            - profile_type: str, type of profiling for organizing artifacts

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs
        near_zero_threshold = kwargs.get('near_zero_threshold', self.near_zero_threshold)
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        encryption_key = kwargs.get('encryption_key', None)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "field": self.field_name})

        try:
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

            # Adjust progress tracker total steps if provided
            total_steps = 4
            if generate_plots:
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
                near_zero_threshold=near_zero_threshold,
                track_progress=progress_tracker is not None
            )

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

            write_json(analysis_results, stats_path, encryption_key=encryption_key)
            result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis", category=Constants.Artifact_Category_Output)

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), f"{self.field_name} statistical analysis")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualizations if requested
            if generate_plots:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualizations"})

                kwargs_encryption = {
                    "use_encryption": kwargs.get('use_encryption', False),
                    "encryption_key": encryption_key
                }
                self._generate_visualizations(
                    df,
                    analysis_results,
                    visualizations_dir,
                    include_timestamp,
                    result,
                    reporter,
                    **kwargs_encryption
                )

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
                error_message=f"Error analyzing numeric field {self.field_name}: {str(e)}"
            )

    def _generate_visualizations(self,
                                 df: pd.DataFrame,
                                 analysis_results: Dict[str, Any],
                                 vis_dir: Path,
                                 include_timestamp: bool,
                                 result: OperationResult,
                                 reporter: Any,
                                 **kwargs):
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
        """
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

                title = f"Distribution of {self.field_name}"
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
                        **kwargs
                    )

                    if not hist_result.startswith("Error"):
                        result.add_artifact("png", hist_path, f"{self.field_name} distribution histogram", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(hist_path), f"{self.field_name} distribution histogram")

        # Generate boxplot visualization if we have enough data
        if len(valid_data) > 5:
            boxplot_filename = get_timestamped_filename(f"{self.field_name}_boxplot", "png", include_timestamp)
            boxplot_path = vis_dir / boxplot_filename

            # Create boxplot using the visualization module
            boxplot_result = create_boxplot(
                data={self.field_name: valid_data.tolist()},
                output_path=str(boxplot_path),
                title=f"Boxplot of {self.field_name}",
                y_label=self.field_name,
                show_points=True,
                **kwargs
            )

            if not boxplot_result.startswith("Error"):
                result.add_artifact("png", boxplot_path, f"{self.field_name} boxplot", category=Constants.Artifact_Category_Visualization)
                reporter.add_artifact("png", str(boxplot_path), f"{self.field_name} boxplot")

        # Generate Q-Q plot for normality if requested and we have results
        if self.test_normality and 'normality' in stats_dict and len(valid_data) > 10:
            qq_filename = get_timestamped_filename(f"{self.field_name}_qq_plot", "png", include_timestamp)
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

            title = f"Q-Q Plot for {self.field_name}"
            if p_value is not None:
                title += f" (Shapiro p-value: {p_value:.4f})"

            qq_result = create_correlation_pair(
                x_data=normal_data,
                y_data=sorted_data,
                output_path=str(qq_path),
                title=title,
                x_label="Theoretical Quantiles",
                y_label="Sample Quantiles",
                add_trendline=True,
                add_histogram=False,
                method="Q-Q Plot",
                **kwargs
            )

            if not qq_result.startswith("Error"):
                result.add_artifact("png", qq_path, f"{self.field_name} Q-Q plot (normality test)", category=Constants.Artifact_Category_Visualization)
                reporter.add_artifact("png", str(qq_path), f"{self.field_name} Q-Q plot (normality test)")


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
    df = load_data_operation(data_source, dataset_name)
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
        overall_tracker = ProgressTracker(
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