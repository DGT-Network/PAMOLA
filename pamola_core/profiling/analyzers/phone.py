"""
Phone data analyzer for the anonymization project.

This module provides analyzers and operations for phone number fields,
following the new operation architecture. It includes phone number validation,
component parsing, and messenger detection capabilities.

It integrates with utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging

Operations:
- PhoneAnalyzer: Static methods for phone analysis
- PhoneOperation: Main operation for phone field analysis
- analyze_phone_fields: Function for analyzing multiple phone fields
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from pamola_core.profiling.commons.phone_utils import (
    analyze_phone_field,
    create_country_code_dictionary,
    create_operator_code_dictionary,
    create_messenger_dictionary,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation, write_dataframe_to_csv, load_settings_operation
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.common.constants import Constants
# Configure logger
logger = logging.getLogger(__name__)


class PhoneAnalyzer:
    """
    Analyzer for phone number fields.

    This analyzer provides static methods for validating phone numbers, extracting components,
    and detecting messenger references in comments.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field_name: str,
                patterns_csv: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a phone field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        patterns_csv : str, optional
            Path to CSV with messenger patterns
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        return analyze_phone_field(
            df=df,
            field_name=field_name,
            patterns_csv=patterns_csv,
            **kwargs
        )

    @staticmethod
    def create_country_code_dictionary(df: pd.DataFrame,
                                       field_name: str,
                                       min_count: int = 1,
                                       **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for country codes.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the phone field
        min_count : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with country code frequency data and metadata
        """
        return create_country_code_dictionary(
            df=df,
            field_name=field_name,
            min_count=min_count,
            **kwargs
        )

    @staticmethod
    def create_operator_code_dictionary(df: pd.DataFrame,
                                        field_name: str,
                                        country_code: Optional[str] = None,
                                        min_count: int = 1,
                                        **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for operator codes.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the phone field
        country_code : str, optional
            The country code to filter by (if None, use all)
        min_count : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with operator code frequency data and metadata
        """
        return create_operator_code_dictionary(
            df=df,
            field_name=field_name,
            country_code=country_code,
            min_count=min_count,
            **kwargs
        )

    @staticmethod
    def create_messenger_dictionary(df: pd.DataFrame,
                                    field_name: str,
                                    min_count: int = 1,
                                    patterns_csv: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for messenger mentions in phone comments.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the phone field
        min_count : int
            Minimum frequency for inclusion in the dictionary
        patterns_csv : str, optional
            Path to CSV with messenger patterns
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with messenger frequency data and metadata
        """
        return create_messenger_dictionary(
            df=df,
            field_name=field_name,
            min_count=min_count,
            patterns_csv=patterns_csv,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the phone field.

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
class PhoneOperation(FieldOperation):
    """
    Operation for analyzing phone number fields.

    This operation wraps the PhoneAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                field_name: str,
                min_frequency: int = 1,
                patterns_csv: Optional[str] = None,
                generate_plots: bool = True,
                include_timestamp: bool = True,
                profile_type: str = 'phone',
                track_progress: bool = True,
                country_code: Any = None,
                description: str = "",
                use_encryption: bool = False,
                encryption_key: Optional[Union[str, Path]] = None):
        """
        Initialize the phone operation.

        Parameters:
        -----------
        field_name : str
            The name of the field to analyze
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        patterns_csv : str, optional
            Path to CSV file with messenger detection patterns
        description : str
            Description of the operation (optional)
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of phone field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )
        
        self.min_frequency = min_frequency
        self.patterns_csv = patterns_csv
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        self.track_progress = track_progress
        self.country_code = country_code

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the phone analysis operation.

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
            - country_code: str, specific country code to focus on for operator analysis

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        country_code = kwargs.get('country_code', self.country_code)
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
            reporter.add_operation(f"Analyzing phone field: {self.field_name}", details={
                "field_name": self.field_name,
                "min_frequency": self.min_frequency,
                "operation_type": "phone_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 4  # Preparation, analysis, saving results, dictionaries
            if generate_plots:
                total_steps += 1  # Add step for generating visualizations

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = PhoneAnalyzer.analyze(
                df=df,
                field_name=self.field_name,
                patterns_csv=self.patterns_csv,
                **kwargs
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

                # 1. Country code distribution visualization
                if 'country_codes' in analysis_results and analysis_results['country_codes']:
                    # Create visualization filename with extension "png"
                    viz_filename = get_timestamped_filename(
                        base_name=f"{self.field_name}_country_codes_distribution",
                        extension="png",
                        include_timestamp=include_timestamp
                    )
                    viz_path = visualizations_dir / viz_filename

                    # Create visualization using the visualization module
                    from pamola_core.utils.visualization import plot_value_distribution
                    title = f"Country Code Distribution in {self.field_name}"

                    viz_result = plot_value_distribution(
                        data=analysis_results['country_codes'],
                        output_path=str(viz_path),
                        title=title,
                        max_items=15,
                        **kwargs
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", viz_path, f"{self.field_name} country codes distribution", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(viz_path), f"{self.field_name} country codes distribution")
                    else:
                        logger.warning(f"Error creating country code visualization: {viz_result}")

                # 2. Operator code distribution visualization
                if 'operator_codes' in analysis_results and analysis_results['operator_codes']:
                    # Create visualization filename with extension "png"
                    viz_filename = get_timestamped_filename(
                        base_name=f"{self.field_name}_operator_codes_distribution",
                        extension="png",
                        include_timestamp=include_timestamp
                    )
                    viz_path = visualizations_dir / viz_filename

                    # Create visualization using the visualization module
                    from pamola_core.utils.visualization import plot_value_distribution
                    title = f"Operator Code Distribution in {self.field_name}"

                    viz_result = plot_value_distribution(
                        data=analysis_results['operator_codes'],
                        output_path=str(viz_path),
                        title=title,
                        max_items=15,
                        **kwargs
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", viz_path, f"{self.field_name} operator codes distribution", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(viz_path), f"{self.field_name} operator codes distribution")
                    else:
                        logger.warning(f"Error creating operator code visualization: {viz_result}")

                # 3. Messenger mentions visualization
                if 'messenger_mentions' in analysis_results and any(analysis_results['messenger_mentions'].values()):
                    # Create visualization filename with extension "png"
                    viz_filename = get_timestamped_filename(
                        base_name=f"{self.field_name}_messenger_mentions",
                        extension="png",
                        include_timestamp=include_timestamp
                    )
                    viz_path = visualizations_dir / viz_filename

                    # Create visualization using the visualization module
                    from pamola_core.utils.visualization import plot_value_distribution
                    title = f"Messenger Mentions in {self.field_name}"

                    viz_result = plot_value_distribution(
                        data=analysis_results['messenger_mentions'],
                        output_path=str(viz_path),
                        title=title,
                        max_items=10,
                        **kwargs
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", viz_path, f"{self.field_name} messenger mentions", category=Constants.Artifact_Category_Visualization)
                        reporter.add_artifact("png", str(viz_path), f"{self.field_name} messenger mentions")
                    else:
                        logger.warning(f"Error creating messenger mentions visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualizations"})

            # Create and save country code dictionary
            country_dict = PhoneAnalyzer.create_country_code_dictionary(
                df=df,
                field_name=self.field_name,
                min_count=self.min_frequency,
                **kwargs
            )

            if 'error' not in country_dict:
                # Save dictionary to CSV
                dict_filename = get_timestamped_filename(f"{self.field_name}_country_codes_dictionary", "csv",
                                                         include_timestamp)
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                import pandas as pd
                dict_df = pd.DataFrame(country_dict['country_codes'])
                write_dataframe_to_csv(df=dict_df, file_path=dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Save detailed dictionary as JSON
                json_dict_filename = get_timestamped_filename(f"{self.field_name}_country_codes_dictionary", "json",
                                                              include_timestamp)
                json_dict_path = output_dir / json_dict_filename
                write_json(country_dict, json_dict_path, encryption_key=encryption_key)

                result.add_artifact("csv", dict_path, f"{self.field_name} country codes dictionary (CSV)", category=Constants.Artifact_Category_Dictionary)
                result.add_artifact("json", json_dict_path, f"{self.field_name} country codes dictionary (JSON)", category=Constants.Artifact_Category_Output)

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} country codes dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path), f"{self.field_name} country codes dictionary (JSON)")

            # Create and save operator code dictionary (for specific country or all)
            operator_dict = PhoneAnalyzer.create_operator_code_dictionary(
                df=df,
                field_name=self.field_name,
                country_code=country_code,
                min_count=self.min_frequency,
                **kwargs
            )

            if 'error' not in operator_dict:
                # Save dictionary to CSV
                dict_filename = get_timestamped_filename(f"{self.field_name}_operator_codes_dictionary", "csv",
                                                         include_timestamp)
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                dict_df = pd.DataFrame(operator_dict['operator_codes'])
                write_dataframe_to_csv(df=dict_df, file_path=dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Save detailed dictionary as JSON
                json_dict_filename = get_timestamped_filename(f"{self.field_name}_operator_codes_dictionary", "json",
                                                              include_timestamp)
                json_dict_path = output_dir / json_dict_filename
                write_json(operator_dict, json_dict_path, encryption_key=encryption_key)

                result.add_artifact("csv", dict_path, f"{self.field_name} operator codes dictionary (CSV)", category=Constants.Artifact_Category_Dictionary)
                result.add_artifact("json", json_dict_path, f"{self.field_name} operator codes dictionary (JSON)", category=Constants.Artifact_Category_Output)

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} operator codes dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path),
                                      f"{self.field_name} operator codes dictionary (JSON)")

            # Create and save messenger dictionary
            messenger_dict = PhoneAnalyzer.create_messenger_dictionary(
                df=df,
                field_name=self.field_name,
                min_count=self.min_frequency,
                patterns_csv=self.patterns_csv,
                **kwargs
            )

            if 'error' not in messenger_dict:
                # Save dictionary to CSV
                dict_filename = get_timestamped_filename(f"{self.field_name}_messenger_dictionary", "csv",
                                                         include_timestamp)
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                dict_df = pd.DataFrame(messenger_dict['messengers'])
                write_dataframe_to_csv(df=dict_df, file_path=dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Save detailed dictionary as JSON
                json_dict_filename = get_timestamped_filename(f"{self.field_name}_messenger_dictionary", "json",
                                                              include_timestamp)
                json_dict_path = output_dir / json_dict_filename
                write_json(messenger_dict, json_dict_path, encryption_key=encryption_key)

                result.add_artifact("csv", dict_path, f"{self.field_name} messenger dictionary (CSV)", category=Constants.Artifact_Category_Dictionary)
                result.add_artifact("json", json_dict_path, f"{self.field_name} messenger dictionary (JSON)", category=Constants.Artifact_Category_Output)

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} messenger dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path), f"{self.field_name} messenger dictionary (JSON)")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Created dictionaries"})

            # Add metrics to the result
            result.add_metric("total_records", analysis_results.get('total_rows', 0))
            result.add_metric("null_count", analysis_results.get('null_count', 0))
            result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
            result.add_metric("valid_count", analysis_results.get('valid_count', 0))
            result.add_metric("valid_percentage", analysis_results.get('valid_percentage', 0))
            result.add_metric("format_error_count", analysis_results.get('format_error_count', 0))
            result.add_metric("has_comment_count", analysis_results.get('has_comment_count', 0))

            # Add normalization metrics if available
            if 'normalization_success_count' in analysis_results:
                result.add_metric("normalization_success_count", analysis_results.get('normalization_success_count', 0))
                result.add_metric("normalization_success_percentage",
                                  analysis_results.get('normalization_success_percentage', 0))

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "valid_phones": analysis_results.get('valid_count', 0),
                "format_errors": analysis_results.get('format_error_count', 0),
                "with_comments": analysis_results.get('has_comment_count', 0),
                "normalization_success": analysis_results.get('normalization_success_count', 0)
            })

            return result

        except Exception as e:
            logger.exception(f"Error in phone operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing phone field {self.field_name}: {str(e)}"
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


def analyze_phone_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        phone_fields: List[str] = None,
        patterns_csv: Optional[str] = None,
        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple phone fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    phone_fields : List[str], optional
        List of phone fields to analyze. If None, tries to find phone fields automatically.
    patterns_csv : str, optional
        Path to CSV file with messenger detection patterns
    **kwargs : dict
        Additional parameters for the operations:
        - min_frequency: int, minimum frequency for inclusion in dictionary (default: 1)
        - generate_plots: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps in filenames (default: True)
        - profile_type: str, type of profiling for organizing artifacts (default: 'phone')
        - country_code: str, specific country code to focus on for operator analysis

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    dataset_name = kwargs.get('dataset_name', "main")
    df = load_data_operation(data_source, dataset_name)
    if df is None:
        reporter.add_operation("Phone fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # Extract operation parameters from kwargs
    min_frequency = kwargs.get('min_frequency', 1)

    # If no phone fields specified, try to detect them
    if phone_fields is None:
        phone_fields = []
        for col in df.columns:
            if 'phone' in col.lower():
                phone_fields.append(col)

        if not phone_fields:
            phone_fields = ['home_phone', 'work_phone', 'cell_phone']  # Default field names

    # Report on fields to be analyzed
    reporter.add_operation("Phone fields analysis", details={
        "fields_count": len(phone_fields),
        "fields": phone_fields,
        "min_frequency": min_frequency,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and phone_fields:
        from pamola_core.utils.progress import ProgressTracker
        overall_tracker = ProgressTracker(
            total=len(phone_fields),
            description=f"Analyzing {len(phone_fields)} phone fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(phone_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(phone_fields)}"})

                logger.info(f"Analyzing phone field: {field}")

                # Create and execute operation
                operation = PhoneOperation(
                    field,
                    min_frequency=min_frequency,
                    patterns_csv=patterns_csv
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
                logger.error(f"Error analyzing phone field {field}: {e}", exc_info=True)

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

    reporter.add_operation("Phone fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results