"""
Categorical field analyzer module for the HHR project.

This module provides analyzers and operations for categorical fields, following the
new operation architecture. It includes distribution analysis, dictionary creation,
anomaly detection, and visualization capabilities.

It integrates with the new utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from pamola_core.profiling.commons.categorical_utils import (
    analyze_categorical_field,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.visualization import (
    plot_value_distribution
)

# Configure logger
logger = logging.getLogger(__name__)


class CategoricalAnalyzer:
    """
    Analyzer for categorical fields.

    This analyzer provides methods for analyzing categorical fields, including
    frequency distributions, cardinality metrics, and dictionary creation.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field_name: str,
                top_n: int = 15,
                min_frequency: int = 1,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a categorical field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top values to include in the results
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        return analyze_categorical_field(
            df=df,
            field_name=field_name,
            top_n=top_n,
            min_frequency=min_frequency,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the categorical field.

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


@register(override=True)
class CategoricalOperation(FieldOperation):
    """
    Operation for analyzing categorical fields.

    This operation wraps the CategoricalAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                 field_name: str,
                 top_n: int = 15,
                 min_frequency: int = 1,
                 description: str = ""):
        """
        Initialize the categorical operation.

        Parameters:
        -----------
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top values to include in the results
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        description : str
            Description of the operation (optional)
        """
        super().__init__(field_name, description or f"Analysis of categorical field '{field_name}'")
        self.top_n = top_n
        self.min_frequency = min_frequency

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the categorical analysis operation.

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
            - analyze_anomalies: bool, whether to analyze anomalies

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs
        generate_plots = kwargs.get('generate_plots', True)
        include_timestamp = kwargs.get('include_timestamp', True)
        profile_type = kwargs.get('profile_type', 'categorical')
        analyze_anomalies = kwargs.get('analyze_anomalies', True)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']
        output_dir = dirs['output']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "field": self.field_name})

        try:
            # Get DataFrame from data source
            df = data_source.get_dataframe("main")
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
            reporter.add_operation(f"Analyzing categorical field: {self.field_name}", details={
                "field_name": self.field_name,
                "top_n": self.top_n,
                "min_frequency": self.min_frequency,
                "operation_type": "categorical_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 4  # Preparation, analysis, saving results, and dictionary
            if generate_plots:
                total_steps += 1  # Add step for generating visualizations

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = CategoricalAnalyzer.analyze(
                df=df,
                field_name=self.field_name,
                top_n=self.top_n,
                min_frequency=self.min_frequency,
                detect_anomalies=analyze_anomalies
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

            # Save analysis results to JSON in the task root directory
            stats_filename = get_timestamped_filename(f"{self.field_name}_stats", "json", include_timestamp)
            stats_path = task_dir / stats_filename

            write_json(analysis_results, stats_path)
            result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis")

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), f"{self.field_name} statistical analysis")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Save dictionary to CSV if available
            if 'value_dictionary' in analysis_results and 'dictionary_data' in analysis_results['value_dictionary']:
                dict_filename = get_timestamped_filename(f"{self.field_name}_dictionary", "csv", include_timestamp)
                dict_path = dictionaries_dir / dict_filename

                # Convert dictionary data to DataFrame
                dict_records = analysis_results['value_dictionary']['dictionary_data']
                if isinstance(dict_records, list) and len(dict_records) > 0:
                    dict_df = pd.DataFrame(dict_records)
                    dict_df.to_csv(dict_path, index=False, encoding='utf-8')

                    result.add_artifact("csv", dict_path, f"{self.field_name} value dictionary")
                    reporter.add_artifact("csv", str(dict_path), f"{self.field_name} value dictionary")
                else:
                    logger.warning(f"Empty dictionary data for {self.field_name}")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved dictionary data"})

            # Generate visualization if requested
            if generate_plots and 'top_values' in analysis_results:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualization"})

                # Create visualization
                viz_filename = get_timestamped_filename(f"{self.field_name}_distribution", "png", include_timestamp)
                viz_path = visualizations_dir / viz_filename

                # Create visualization using the visualization module
                title = f"Distribution of {self.field_name}"
                viz_result = plot_value_distribution(
                    data=analysis_results['top_values'],
                    output_path=str(viz_path),
                    title=title,
                    max_items=self.top_n
                )

                if not viz_result.startswith("Error"):
                    result.add_artifact("png", viz_path, f"{self.field_name} distribution visualization")
                    reporter.add_artifact("png", str(viz_path), f"{self.field_name} distribution visualization")
                else:
                    logger.warning(f"Error creating visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualization"})

            # Save anomalies to CSV if detected
            if 'anomalies' in analysis_results and analysis_results['anomalies']:
                self._save_anomalies_to_csv(
                    analysis_results,
                    dictionaries_dir,
                    include_timestamp,
                    result,
                    reporter
                )

            # Add metrics to the result
            result.add_metric("total_records", analysis_results.get('total_records', 0))
            result.add_metric("null_count", analysis_results.get('null_values', 0))
            result.add_metric("null_percent", analysis_results.get('null_percent', 0))
            result.add_metric("unique_values", analysis_results.get('unique_values', 0))
            result.add_metric("entropy", analysis_results.get('entropy', 0))
            result.add_metric("cardinality_ratio", analysis_results.get('cardinality_ratio', 0))

            if 'distribution_type' in analysis_results:
                result.add_metric("distribution_type", analysis_results.get('distribution_type'))

            if 'anomalies' in analysis_results:
                result.add_metric("anomalies_count", len(analysis_results['anomalies']))

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "unique_values": analysis_results.get('unique_values', 0),
                "null_percent": analysis_results.get('null_percent', 0),
                "entropy": round(analysis_results.get('entropy', 0), 2),
                "anomalies_found": len(analysis_results.get('anomalies', {})) if 'anomalies' in analysis_results else 0
            })

            return result

        except Exception as e:
            logger.exception(f"Error in categorical operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing categorical field {self.field_name}: {str(e)}"
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
        dictionaries_dir = task_dir / 'dictionaries'

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)
        ensure_directory(dictionaries_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir,
            'dictionaries': dictionaries_dir
        }

    def _save_anomalies_to_csv(self,
                               analysis_results: Dict[str, Any],
                               dict_dir: Path,
                               include_timestamp: bool,
                               result: OperationResult,
                               reporter: Any):
        """
        Save anomalies to CSV file.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        dict_dir : Path
            Directory to save dictionaries
        include_timestamp : bool
            Whether to include timestamps in filenames
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        try:
            # Extract anomalies
            anomalies = analysis_results.get('anomalies', {})
            if not anomalies:
                return

            # Collect potential typos
            anomaly_records = []

            if 'potential_typos' in anomalies:
                for rare_value, typo_info in anomalies['potential_typos'].items():
                    anomaly_records.append({
                        'value': rare_value,
                        'frequency': typo_info['count'],
                        'anomaly_type': 'potential_typo',
                        'similar_to': typo_info['similar_to'],
                        'similar_count': typo_info['similar_count']
                    })

            # Collect single character values
            if 'single_char_values' in anomalies:
                for value, count in anomalies['single_char_values'].items():
                    anomaly_records.append({
                        'value': value,
                        'frequency': count,
                        'anomaly_type': 'single_char_value',
                        'similar_to': '',
                        'similar_count': 0
                    })

            # Collect numeric-like strings
            if 'numeric_like_strings' in anomalies:
                for value, count in anomalies['numeric_like_strings'].items():
                    anomaly_records.append({
                        'value': value,
                        'frequency': count,
                        'anomaly_type': 'numeric_like_string',
                        'similar_to': '',
                        'similar_count': 0
                    })

            # Create anomalies CSV filename
            if anomaly_records:
                anomalies_filename = get_timestamped_filename(f"{self.field_name}_anomalies", "csv", include_timestamp)
                anomalies_path = dict_dir / anomalies_filename

                # Create DataFrame and save to CSV
                anomalies_df = pd.DataFrame(anomaly_records)
                anomalies_df.to_csv(anomalies_path, index=False, encoding='utf-8')

                # Add artifact to result and reporter
                result.add_artifact("csv", anomalies_path, f"{self.field_name} anomalies")
                reporter.add_artifact("csv", str(anomalies_path), f"{self.field_name} anomalies")

        except Exception as e:
            logger.warning(f"Error saving anomalies for {self.field_name}: {e}")
            reporter.add_operation(f"Saving anomalies for {self.field_name}", status="warning",
                                   details={"warning": str(e)})


def analyze_categorical_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        cat_fields: List[str] = None,
        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple categorical fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    cat_fields : List[str], optional
        List of categorical fields to analyze. If None, tries to find categorical fields automatically.
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top values to include in results (default: 15)
        - min_frequency: int, minimum frequency for inclusion in dictionary (default: 1)
        - generate_plots: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps in filenames (default: True)
        - profile_type: str, type of profiling for organizing artifacts (default: 'categorical')

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    df = data_source.get_dataframe("main")
    # Use get_dataframe safely
    if df is None:
        reporter.add_operation("Categorical fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # Extract operation parameters from kwargs
    top_n = kwargs.get('top_n', 15)
    min_frequency = kwargs.get('min_frequency', 1)

    # If no categorical fields specified, try to detect them
    if cat_fields is None:
        cat_fields = []
        # Simple heuristic: select fields with string type or moderate number of unique values
        for col in df.columns:
            # Check if column is object type (usually string)
            if df[col].dtype == 'object':
                cat_fields.append(col)
            # Or check number of unique values relative to dataset size
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= min(100, int(len(df) * 0.1)):
                cat_fields.append(col)

    # Report on fields to be analyzed
    reporter.add_operation("Categorical fields analysis", details={
        "fields_count": len(cat_fields),
        "fields": cat_fields,
        "top_n": top_n,
        "min_frequency": min_frequency,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and cat_fields:
        overall_tracker = ProgressTracker(
            total=len(cat_fields),
            description=f"Analyzing {len(cat_fields)} categorical fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(cat_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(cat_fields)}"})

                logger.info(f"Analyzing categorical field: {field}")

                # Create and execute operation
                operation = CategoricalOperation(
                    field,
                    top_n=top_n,
                    min_frequency=min_frequency
                )
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
                logger.error(f"Error analyzing categorical field {field}: {e}", exc_info=True)

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

    reporter.add_operation("Categorical fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results