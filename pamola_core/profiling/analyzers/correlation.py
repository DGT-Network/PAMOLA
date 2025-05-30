"""
Correlation analysis module for the project.

This module provides analyzers and operations for calculating correlations between
fields, following the new operation architecture. It supports various correlation types:
- Pearson correlation for numeric-numeric fields
- Cramer's V for categorical-categorical fields
- Correlation ratio for numeric-categorical fields
- Point-biserial correlation for binary-numeric fields

It integrates with the new utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import pandas as pd

from pamola_core.profiling.commons.correlation_utils import (
    analyze_correlation,
    analyze_correlation_matrix,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation, load_settings_operation
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation, BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.visualization import (
    create_scatter_plot,
    create_boxplot,
    create_heatmap,
    create_correlation_matrix
)
from pamola_core.common.constants import Constants
# Configure logger
logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzer for correlations between fields.

    This analyzer provides methods for analyzing correlations between fields,
    supporting different correlation methods based on field types, and producing
    visualizations of the relationships.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field1: str,
                field2: str,
                method: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze correlation between two fields in a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        field1 : str
            Name of the first field
        field2 : str
            Name of the second field
        method : str, optional
            Correlation method to use. If None, automatically selected based on data types.
        **kwargs : dict
            Additional parameters for analysis

        Returns:
        --------
        Dict[str, Any]
            Results of the correlation analysis
        """
        return analyze_correlation(
            df=df,
            field1=field1,
            field2=field2,
            method=method,
            **kwargs
        )

    @staticmethod
    def analyze_matrix(df: pd.DataFrame,
                       fields: List[str],
                       **kwargs) -> Dict[str, Any]:
        """
        Create a correlation matrix for multiple fields.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        fields : List[str]
            List of field names to include in the correlation matrix
        **kwargs : dict
            Additional parameters for analysis

        Returns:
        --------
        Dict[str, Any]
            Dictionary with correlation matrix and supporting information
        """
        return analyze_correlation_matrix(
            df=df,
            fields=fields,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame,
                           field1: str,
                           field2: str) -> Dict[str, Any]:
        """
        Estimate resources needed for correlation analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        field1 : str
            Name of the first field
        field2 : str
            Name of the second field

        Returns:
        --------
        Dict[str, Any]
            Estimated resource requirements
        """
        return estimate_resources(df, field1, field2)


@register(override=True)
class CorrelationOperation(FieldOperation):
    """
    Operation for analyzing correlation between two fields.

    This operation wraps the CorrelationAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                 field1: str,
                 field2: str,
                 method: Optional[str] = None,
                 description: str = "",
                 generate_plots: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = "correlation",
                 null_handling: str = "drop",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
        """
        Initialize the correlation operation.

        Parameters:
        -----------
        field1 : str
            Name of the first field to analyze
        field2 : str
            Name of the second field to analyze
        method : str, optional
            Correlation method to use. If None, automatically selected based on data types.
        description : str
            Description of the operation (optional)
        generate_plots : bool
            Whether to generate visualizations (default: True)
        include_timestamp : bool
            Whether to include timestamps in filenames (default: True)
        profile_type : str
            Type of profiling for organizing artifacts (default: "correlation")
        null_handling : str
            Method for handling nulls ('drop', 'fill', 'pairwise')    
        """
        # Use field1 as the primary field for the parent class
        super().__init__(
            field_name=field1,
            description=description or f"Correlation analysis between '{field1}' and '{field2}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )
        self.field1 = field1
        self.field2 = field2
        self.method = method
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        self.null_handling = null_handling

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the correlation analysis operation.

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
            - generate_plots: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames
            - profile_type: str, type of profiling for organizing artifacts
            - null_handling: str, method for handling nulls ('drop', 'fill', 'pairwise')

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        null_handling = kwargs.get('null_handling', self.null_handling)
        encryption_key = kwargs.get('encryption_key', None)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "fields": f"{self.field1}, {self.field2}"})

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

            # Check if fields exist
            for field in [self.field1, self.field2]:
                if field not in df.columns:
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=f"Field {field} not found in DataFrame"
                    )

            # Add operation to reporter
            reporter.add_operation(f"Analyzing correlation between {self.field1} and {self.field2}", details={
                "field1": self.field1,
                "field2": self.field2,
                "method": self.method or "auto",
                "null_handling": null_handling,
                "operation_type": "correlation_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 3  # Preparation, analysis, saving results
            if generate_plots:
                total_steps += 1  # Add step for generating visualizations

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing correlation"})

            # Execute the analyzer
            analysis_results = CorrelationAnalyzer.analyze(
                df=df,
                field1=self.field1,
                field2=self.field2,
                method=self.method,
                null_handling=null_handling
            )

            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analysis complete", "fields": f"{self.field1}, {self.field2}"})

            # Save analysis results to JSON
            correlation_name = f"{self.field1}_{self.field2}_correlation"
            stats_filename = get_timestamped_filename(correlation_name, "json", include_timestamp)
            stats_path = output_dir / stats_filename

            write_json(analysis_results, stats_path, encryption_key=encryption_key)
            result.add_artifact("json", stats_path, f"Correlation analysis between {self.field1} and {self.field2}", category=Constants.Artifact_Category_Output)

            # Add to reporter
            reporter.add_artifact("json", str(stats_path),
                                  f"Correlation analysis between {self.field1} and {self.field2}")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualization if requested
            if generate_plots and 'plot_data' in analysis_results:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualization"})

                # Create visualization based on plot_data type
                plot_data = analysis_results['plot_data']
                plot_type = plot_data.get('type', 'unknown')
                viz_filename = get_timestamped_filename(correlation_name + "_plot", "png", include_timestamp)
                viz_path = visualizations_dir / viz_filename

                # Method details for plot title
                method_name = analysis_results.get('method', 'Unknown')
                method_display = method_name.replace('_', ' ').title()
                correlation_value = analysis_results.get('correlation_coefficient', 0)

                # Create appropriate visualization based on plot type
                viz_result = None

                if plot_type == "scatter":
                    # For numeric-numeric correlations: scatter plot
                    title = f"Correlation between {self.field1} and {self.field2}"
                    viz_result = create_scatter_plot(
                        x_data=plot_data['x_values'],
                        y_data=plot_data['y_values'],
                        output_path=str(viz_path),
                        title=title,
                        x_label=plot_data['x_label'],
                        y_label=plot_data['y_label'],
                        add_trendline=True,
                        correlation=correlation_value,
                        method=method_display,
                        **kwargs
                    )

                elif plot_type == "boxplot":
                    # For categorical-numeric correlations: boxplot
                    title = f"Relationship between {plot_data['x_label']} and {plot_data['y_label']}"
                    viz_result = create_boxplot(
                        data={cat: list(values) for cat, values in zip(
                            plot_data['categories'], plot_data['values']
                        ) if cat is not None},
                        output_path=str(viz_path),
                        title=title,
                        x_label=plot_data['x_label'],
                        y_label=plot_data['y_label'],
                        *kwargs
                    )

                elif plot_type == "heatmap":
                    # For categorical-categorical correlations: heatmap
                    title = f"Association between {plot_data['y_label']} and {plot_data['x_label']}"
                    viz_result = create_heatmap(
                        data=plot_data['matrix'],
                        output_path=str(viz_path),
                        title=title,
                        x_label=plot_data['x_label'],
                        y_label=plot_data['y_label'],
                        annotate=True,
                        **kwargs
                    )

                # Add visualization to result if successful
                if viz_result and not viz_result.startswith("Error"):
                    result.add_artifact("png", viz_path, f"Correlation plot for {self.field1} and {self.field2}", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(viz_path), f"Correlation plot for {self.field1} and {self.field2}")
                else:
                    logger.warning(f"Error creating visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualization"})

            # Add metrics to the result
            result.add_metric("correlation_method", analysis_results.get('method', 'unknown'))
            result.add_metric("correlation_coefficient", analysis_results.get('correlation_coefficient', 0))
            result.add_metric("sample_size", analysis_results.get('sample_size', 0))
            if 'p_value' in analysis_results and analysis_results['p_value'] is not None:
                result.add_metric("p_value", analysis_results['p_value'])
                result.add_metric("statistically_significant", analysis_results['p_value'] < 0.05)

            # Add final operation status to reporter
            reporter.add_operation(f"Correlation analysis between {self.field1} and {self.field2} completed", details={
                "correlation_coefficient": round(analysis_results.get('correlation_coefficient', 0), 4),
                "method": analysis_results.get('method', 'unknown'),
                "interpretation": analysis_results.get('interpretation', '')
            })

            return result

        except Exception as e:
            logger.exception(f"Error in correlation operation for {self.field1} and {self.field2}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing correlation between {self.field1} and {self.field2}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing correlation between {self.field1} and {self.field2}: {str(e)}"
            )

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare required directories for artifacts.

        Parameters:
        -----------
        task_dir : Path
            Base directory for the task

        Returns:
        --------
        Dict[str, Path]
            Dictionary of directory paths
        """
        # Create required directories
        output_dir = task_dir / 'output'
        visualizations_dir = task_dir / 'visualizations'

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir
        }


@register(override=True)
class CorrelationMatrixOperation(BaseOperation):
    """
    Operation for creating a correlation matrix for multiple fields.

    This operation analyzes correlations between all pairs of fields in a list
    and generates a correlation matrix visualization.
    """

    def __init__(self,
                 fields: List[str],
                 methods: Optional[Dict[str, str]] = None,
                 description: str = "",
                 generate_plots: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = "correlation",
                 min_threshold: float = 0.3,
                 null_handling: str = "drop",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
        """
        Initialize the correlation matrix operation.

        Parameters:
        -----------
        fields : List[str]
            List of fields to include in the matrix
        methods : Dict[str, str], optional
            Dictionary mapping field pairs to correlation methods
        description : str
            Description of the operation (optional)
        generate_plots : bool
            Whether to generate visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        profile_type : str
            Type of profiling for organizing artifacts
        min_threshold : float
            Minimum correlation threshold for significant correlations
        null_handling : str
            Method for handling nulls ('drop', 'fill', 'pairwise')
        """
        super().__init__(
            description or f"Correlation matrix analysis for {len(fields)} fields",
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )
        
        self.fields = fields
        self.methods = methods
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        self.min_threshold = min_threshold
        self.null_handling = null_handling

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the correlation matrix operation.

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
            - generate_plots: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames
            - profile_type: str, type of profiling for organizing artifacts
            - null_handling: str, method for handling nulls ('drop', 'fill', 'pairwise')
            - min_threshold: float, minimum correlation threshold for significant correlations

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        min_threshold = kwargs.get('min_threshold', self.min_threshold)
        null_handling = kwargs.get('null_handling', self.null_handling)
        encryption_key = kwargs.get('encryption_key', None)

        # Set up directories
        output_dir = task_dir / 'output'
        visualizations_dir = task_dir / 'visualizations'
        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "fields_count": len(self.fields)})

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

            # Check if fields exist
            missing_fields = [field for field in self.fields if field not in df.columns]
            if missing_fields:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Fields not found: {', '.join(missing_fields)}"
                )

            # Add operation to reporter
            reporter.add_operation(f"Creating correlation matrix for {len(self.fields)} fields", details={
                "fields": self.fields,
                "null_handling": null_handling,
                "min_threshold": min_threshold,
                "operation_type": "correlation_matrix"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 3  # Preparation, analysis, saving results
            if generate_plots:
                total_steps += 1  # Add step for generating visualizations

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Creating correlation matrix"})

            # Execute the analyzer
            analysis_results = CorrelationAnalyzer.analyze_matrix(
                df=df,
                fields=self.fields,
                methods=self.methods,
                null_handling=null_handling,
                min_threshold=min_threshold
            )

            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analysis complete", "fields_count": len(self.fields)})

            # Save analysis results to JSON
            stats_filename = get_timestamped_filename("correlation_matrix", "json", include_timestamp)
            stats_path = output_dir / stats_filename

            write_json(analysis_results, stats_path, encryption_key=encryption_key)
            result.add_artifact("json", stats_path, "Correlation matrix analysis", category=Constants.Artifact_Category_Output)

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), "Correlation matrix analysis")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualization if requested
            if generate_plots and 'correlation_matrix' in analysis_results:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualization"})

                # Convert dictionary to DataFrame for visualization
                matrix_dict = analysis_results['correlation_matrix']
                matrix_df = pd.DataFrame(matrix_dict)

                # Create visualization
                viz_filename = get_timestamped_filename("correlation_matrix_heatmap", "png", include_timestamp)
                viz_path = visualizations_dir / viz_filename

                viz_result = create_correlation_matrix(
                    data=matrix_df,
                    output_path=str(viz_path),
                    title="Correlation Matrix",
                    annotate=True,
                    annotation_format=".2f",
                    mask_diagonal=False,
                    mask_upper=False,
                    **kwargs
                )

                # Add visualization to result if successful
                if viz_result and not viz_result.startswith("Error"):
                    result.add_artifact("png", viz_path, "Correlation matrix visualization", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(viz_path), "Correlation matrix visualization")
                else:
                    logger.warning(f"Error creating visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualization"})

            # Add metrics to the result
            result.add_metric("fields_analyzed", len(self.fields))
            result.add_metric("significant_correlations",
                              len(analysis_results.get('significant_correlations', [])))
            result.add_metric("min_threshold", min_threshold)

            # Add final operation status to reporter
            significant_count = len(analysis_results.get('significant_correlations', []))
            reporter.add_operation(f"Correlation matrix analysis completed", details={
                "fields_analyzed": len(self.fields),
                "significant_correlations": significant_count,
                "min_threshold": min_threshold
            })

            return result

        except Exception as e:
            logger.exception(f"Error in correlation matrix operation: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error creating correlation matrix",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error creating correlation matrix: {str(e)}"
            )


def analyze_correlations(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        pairs: List[Tuple[str, str]],
        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze correlations between multiple pairs of fields.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    pairs : List[Tuple[str, str]]
        List of field pairs to analyze as tuples (field1, field2)
    **kwargs : dict
        Additional parameters for the operations:
        - methods: dict, mapping of field pairs to correlation methods
        - null_handling: str, method for handling nulls (default: 'drop')
        - generate_plots: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps (default: True)
        - profile_type: str, type of profiling (default: 'correlation')

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping pair names to their operation results
    """
    # Get DataFrame from data source to check fields
    dataset_name = kwargs.get('dataset_name', "main")
    df = load_data_operation(data_source, dataset_name)
    if df is None:
        reporter.add_operation("Correlation analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # Extract parameters from kwargs
    methods = kwargs.get('methods', {})
    null_handling = kwargs.get('null_handling', 'drop')
    generate_plots = kwargs.get('generate_plots', True)

    # Report on field pairs to be analyzed
    reporter.add_operation("Correlation analysis", details={
        "pairs_count": len(pairs),
        "pairs": [f"{field1}_{field2}" for field1, field2 in pairs],
        "null_handling": null_handling,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and pairs:
        overall_tracker = ProgressTracker(
            total=len(pairs),
            description=f"Analyzing {len(pairs)} field correlations",
            unit="pairs",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field pair
    for i, (field1, field2) in enumerate(pairs):
        # Validate fields existence
        if field1 not in df.columns or field2 not in df.columns:
            missing_fields = []
            if field1 not in df.columns:
                missing_fields.append(field1)
            if field2 not in df.columns:
                missing_fields.append(field2)

            error_msg = f"Fields not found: {', '.join(missing_fields)}"
            reporter.add_operation(
                f"Correlation Analysis: {field1} vs {field2}",
                status="error",
                details={"error": error_msg}
            )

            # Create an error result
            error_result = OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_msg
            )
            results[f"{field1}_{field2}"] = error_result

            # Update overall tracker if present
            if overall_tracker:
                overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "error"})

            continue

        try:
            # Update overall progress tracker
            if overall_tracker:
                overall_tracker.update(0, {"pair": f"{field1}_{field2}", "progress": f"{i + 1}/{len(pairs)}"})

            logger.info(f"Analyzing correlation between {field1} and {field2}")

            # Get method if specified
            method = methods.get(f"{field1}_{field2}")

            # Create and execute operation
            operation = CorrelationOperation(
                field1=field1,
                field2=field2,
                method=method
            )
            result = operation.execute(
                data_source,
                task_dir,
                reporter,
                null_handling=null_handling,
                generate_plots=generate_plots,
                **kwargs
            )

            # Store result
            results[f"{field1}_{field2}"] = result

            # Update overall tracker after successful analysis
            if overall_tracker:
                if result.status == OperationStatus.SUCCESS:
                    overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "completed"})
                else:
                    overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "error",
                                               "error": result.error_message})

        except Exception as e:
            logger.error(f"Error analyzing correlation between {field1} and {field2}: {e}", exc_info=True)

            reporter.add_operation(f"Analyzing correlation between {field1} and {field2}", status="error",
                                   details={"error": str(e)})

            # Create an error result
            error_result = OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
            results[f"{field1}_{field2}"] = error_result

            # Update overall tracker in case of error
            if overall_tracker:
                overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(1 for r in results.values() if r.status == OperationStatus.SUCCESS)
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    reporter.add_operation("Correlation analysis completed", details={
        "pairs_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results