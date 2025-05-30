"""
Multi-valued field analyzer module for the project.

This module provides analyzers and operations for multi-valued fields (MVF),
following the operation architecture. It includes parsing, distribution analysis,
dictionary creation, and visualization capabilities.

It integrates with utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging

MVF fields contain multiple values per record, typically stored as:
- String representations of arrays: "['Value1', 'Value2']"
- JSON arrays: ["Value1", "Value2"]
- Comma-separated values: "Value1, Value2"
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from pamola_core.profiling.commons.mvf_utils import (
    parse_mvf,
    analyze_mvf_field,
    create_value_dictionary,
    create_combinations_dictionary,
    analyze_value_count_distribution,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation, write_dataframe_to_csv, load_settings_operation
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.visualization import (
    plot_value_distribution,
    create_bar_plot
)
from pamola_core.common.constants import Constants
# Configure logger
logger = logging.getLogger(__name__)


class MVFAnalyzer:
    """
    Analyzer for multi-valued fields.

    This analyzer provides methods for analyzing MVF fields, including
    parsing, frequency distributions, combinations analysis, and value count analysis.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field_name: str,
                top_n: int = 20,
                min_frequency: int = 1,
                **kwargs) -> Dict[str, Any]:
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
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        return analyze_mvf_field(
            df=df,
            field_name=field_name,
            top_n=top_n,
            min_frequency=min_frequency,
            **kwargs
        )

    @staticmethod
    def parse_field(df: pd.DataFrame,
                    field_name: str,
                    format_type: Optional[str] = None,
                    **kwargs) -> pd.DataFrame:
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
        parse_args['format_type'] = format_type

        # Apply parsing to each value
        result_df[parsed_column] = result_df[field_name].apply(
            lambda x: parse_mvf(x, **parse_args) if not pd.isna(x) else []
        )

        return result_df

    @staticmethod
    def create_value_dictionary(df: pd.DataFrame,
                                field_name: str,
                                min_frequency: int = 1,
                                **kwargs) -> pd.DataFrame:
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
        **kwargs : dict
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
            parse_args=kwargs
        )

    @staticmethod
    def create_combinations_dictionary(df: pd.DataFrame,
                                       field_name: str,
                                       min_frequency: int = 1,
                                       **kwargs) -> pd.DataFrame:
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
        **kwargs : dict
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
            parse_args=kwargs
        )

    @staticmethod
    def analyze_value_counts(df: pd.DataFrame,
                             field_name: str,
                             **kwargs) -> Dict[str, int]:
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
            df=df,
            field_name=field_name,
            parse_args=kwargs
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


@register(override=True)
class MVFOperation(FieldOperation):
    """
    Operation for analyzing multi-valued fields.

    This operation wraps the MVFAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                 field_name: str,
                 top_n: int = 20,
                 min_frequency: int = 1,
                 generate_plots: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = 'mvf',
                 format_type: Any = None,
                 parse_kwargs: Any = {},
                 description: str = "",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
        """
        Initialize the MVF operation.

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
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of multi-valued field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )
        
        self.top_n = top_n
        self.min_frequency = min_frequency
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        self.format_type = format_type
        self.parse_kwargs = parse_kwargs
        self.min_frequency = min_frequency

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the MVF analysis operation.

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
            - format_type: str, format type hint for parsing
            - parse_kwargs: dict, additional parameters for MVF parsing

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        format_type = kwargs.get('format_type', self.format_type)
        parse_kwargs = kwargs.get('parse_kwargs', self.parse_kwargs)
        encryption_key = kwargs.get('encryption_key', None)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

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
            reporter.add_operation(f"Analyzing multi-valued field: {self.field_name}", details={
                "field_name": self.field_name,
                "top_n": self.top_n,
                "min_frequency": self.min_frequency,
                "operation_type": "mvf_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 5  # Preparation, analysis, saving results, dictionaries, visualization
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = MVFAnalyzer.analyze(
                df=df,
                field_name=self.field_name,
                top_n=self.top_n,
                min_frequency=self.min_frequency,
                format_type=format_type,
                **parse_kwargs
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

            # Create and save value dictionary
            values_dict = MVFAnalyzer.create_value_dictionary(
                df=df,
                field_name=self.field_name,
                min_frequency=self.min_frequency,
                format_type=format_type,
                **parse_kwargs
            )

            if not values_dict.empty:
                values_dict_filename = get_timestamped_filename(f"{self.field_name}_values_dictionary", "csv",
                                                                include_timestamp)
                values_dict_path = dictionaries_dir / values_dict_filename
                write_dataframe_to_csv(df=values_dict, file_path=values_dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                result.add_artifact("csv", values_dict_path, f"{self.field_name} values dictionary", category=Constants.Artifact_Category_Dictionary)
                reporter.add_artifact("csv", str(values_dict_path), f"{self.field_name} values dictionary")

            # Create and save combinations dictionary
            combinations_dict = MVFAnalyzer.create_combinations_dictionary(
                df=df,
                field_name=self.field_name,
                min_frequency=self.min_frequency,
                format_type=format_type,
                **parse_kwargs
            )

            if not combinations_dict.empty:
                combinations_dict_filename = get_timestamped_filename(f"{self.field_name}_combinations_dictionary",
                                                                      "csv", include_timestamp)
                combinations_dict_path = dictionaries_dir / combinations_dict_filename
                write_dataframe_to_csv(df=combinations_dict, file_path=combinations_dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                result.add_artifact("csv", combinations_dict_path, f"{self.field_name} combinations dictionary", category=Constants.Artifact_Category_Dictionary)
                reporter.add_artifact("csv", str(combinations_dict_path), f"{self.field_name} combinations dictionary")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved dictionaries"})

            # Generate visualizations if requested
            if generate_plots:
                # Values distribution visualization
                if 'values_analysis' in analysis_results and analysis_results['values_analysis']:
                    values_viz_filename = get_timestamped_filename(f"{self.field_name}_values_distribution", "png",
                                                                   include_timestamp)
                    values_viz_path = visualizations_dir / values_viz_filename

                    # Create visualization using the visualization module
                    title = f"Distribution of values in {self.field_name}"
                    viz_result = plot_value_distribution(
                        data=analysis_results['values_analysis'],
                        output_path=str(values_viz_path),
                        title=title,
                        max_items=self.top_n,
                        **kwargs
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", values_viz_path, f"{self.field_name} values distribution", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(values_viz_path), f"{self.field_name} values distribution")
                    else:
                        logger.warning(f"Error creating values visualization: {viz_result}")

                # Combinations distribution visualization
                if 'combinations_analysis' in analysis_results and analysis_results['combinations_analysis']:
                    combos_viz_filename = get_timestamped_filename(f"{self.field_name}_combinations_distribution",
                                                                   "png", include_timestamp)
                    combos_viz_path = visualizations_dir / combos_viz_filename

                    # Create visualization using the visualization module
                    title = f"Distribution of combinations in {self.field_name}"
                    viz_result = plot_value_distribution(
                        data=analysis_results['combinations_analysis'],
                        output_path=str(combos_viz_path),
                        title=title,
                        max_items=min(10, len(analysis_results['combinations_analysis'])),
                        **kwargs
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", combos_viz_path, f"{self.field_name} combinations distribution", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(combos_viz_path),
                                              f"{self.field_name} combinations distribution")
                    else:
                        logger.warning(f"Error creating combinations visualization: {viz_result}")

                # Value counts distribution visualization
                if 'value_counts_distribution' in analysis_results and analysis_results['value_counts_distribution']:
                    counts_viz_filename = get_timestamped_filename(f"{self.field_name}_value_counts_distribution",
                                                                   "png", include_timestamp)
                    counts_viz_path = visualizations_dir / counts_viz_filename

                    # Create the data for bar plot - convert string keys to integers and then back to strings
                    # to match the expected Dict[str, Any] type
                    counts_data = {}
                    for k, v in analysis_results['value_counts_distribution'].items():
                        try:
                            # Convert key to int, then back to string for visualization
                            counts_data[str(int(k))] = v
                        except ValueError:
                            # If key can't be converted to int, keep as is
                            counts_data[k] = v

                    # Sort by numeric value of the key
                    sorted_keys = sorted(counts_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                    sorted_counts_data = {k: counts_data[k] for k in sorted_keys}

                    # Create visualization using the visualization module
                    title = f"Distribution of value counts in {self.field_name}"
                    viz_result = create_bar_plot(
                        data=sorted_counts_data,
                        output_path=str(counts_viz_path),
                        title=title,
                        orientation="v",
                        x_label="Number of values per record",
                        y_label="Frequency",
                        **kwargs
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", counts_viz_path, f"{self.field_name} value counts distribution", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(counts_viz_path),
                                              f"{self.field_name} value counts distribution")
                    else:
                        logger.warning(f"Error creating value counts visualization: {viz_result}")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Created visualizations"})

            # Add metrics to the result
            result.add_metric("total_records", analysis_results.get('total_records', 0))
            result.add_metric("null_count", analysis_results.get('null_count', 0))
            result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
            result.add_metric("empty_arrays_count", analysis_results.get('empty_arrays_count', 0))
            result.add_metric("unique_values", analysis_results.get('unique_values', 0))
            result.add_metric("unique_combinations", analysis_results.get('unique_combinations', 0))
            result.add_metric("avg_values_per_record", analysis_results.get('avg_values_per_record', 0))

            if 'error_count' in analysis_results:
                result.add_metric("error_count", analysis_results.get('error_count', 0))
                result.add_metric("error_percentage", analysis_results.get('error_percentage', 0))

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "unique_values": analysis_results.get('unique_values', 0),
                "unique_combinations": analysis_results.get('unique_combinations', 0),
                "avg_values_per_record": analysis_results.get('avg_values_per_record', 0),
                "null_percentage": analysis_results.get('null_percentage', 0)
            })

            return result

        except Exception as e:
            logger.exception(f"Error in MVF operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing MVF field {self.field_name}: {str(e)}"
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


def analyze_mvf_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        mvf_fields: List[str],
        **kwargs) -> Dict[str, OperationResult]:
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
        - generate_plots: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps in filenames (default: True)
        - profile_type: str, type of profiling for organizing artifacts (default: 'mvf')
        - format_type: str, format type hint for parsing (default: None)
        - parse_kwargs: dict, additional parameters for MVF parsing

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    dataset_name = kwargs.get('dataset_name', "main")
    df = load_data_operation(data_source, dataset_name)
    if df is None:
        reporter.add_operation("MVF fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # Extract operation parameters from kwargs
    top_n = kwargs.get('top_n', 20)
    min_frequency = kwargs.get('min_frequency', 1)
    format_type = kwargs.get('format_type', None)
    parse_kwargs = kwargs.get('parse_kwargs', {})

    # Report on fields to be analyzed
    reporter.add_operation("MVF fields analysis", details={
        "fields_count": len(mvf_fields),
        "fields": mvf_fields,
        "top_n": top_n,
        "min_frequency": min_frequency,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and mvf_fields:
        from pamola_core.utils.progress import ProgressTracker
        overall_tracker = ProgressTracker(
            total=len(mvf_fields),
            description=f"Analyzing {len(mvf_fields)} MVF fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(mvf_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(mvf_fields)}"})

                logger.info(f"Analyzing MVF field: {field}")

                # Create and execute operation
                operation = MVFOperation(
                    field,
                    top_n=top_n,
                    min_frequency=min_frequency
                )

                # Create kwargs for this field
                field_kwargs = kwargs.copy()
                field_kwargs['format_type'] = format_type
                field_kwargs['parse_kwargs'] = parse_kwargs

                result = operation.execute(data_source, task_dir, reporter, **field_kwargs)

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
                logger.error(f"Error analyzing MVF field {field}: {e}", exc_info=True)

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

    reporter.add_operation("MVF fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results