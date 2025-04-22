"""
Email data analyzer for the HHR anonymization project.

This module provides analyzers and operations for email data fields,
following the new operation architecture. It includes email validation,
domain extraction, and pattern detection capabilities.

It integrates with utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging

Operations:
- EmailOperation: Main operation for email field analysis
- analyze_email_fields: Function for analyzing multiple email fields
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from pamola_core.profiling.commons.email_utils import (
    analyze_email_field,
    create_domain_dictionary,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Configure logger
logger = logging.getLogger(__name__)


class EmailAnalyzer:
    """
    Analyzer for email fields.

    This analyzer provides static methods for validating emails, extracting domains,
    and identifying patterns in email addresses.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field_name: str,
                top_n: int = 20,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze an email field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top domains to include in the results
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        return analyze_email_field(
            df=df,
            field_name=field_name,
            top_n=top_n,
            **kwargs
        )

    @staticmethod
    def create_domain_dictionary(df: pd.DataFrame,
                                 field_name: str,
                                 min_count: int = 1,
                                 **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for email domains.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the email field
        min_count : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with domain frequency data and metadata
        """
        return create_domain_dictionary(
            df=df,
            field_name=field_name,
            min_count=min_count,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the email field.

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
class EmailOperation(FieldOperation):
    """
    Operation for analyzing email fields.

    This operation wraps the EmailAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                 field_name: str,
                 top_n: int = 20,
                 min_frequency: int = 1,
                 description: str = ""):
        """
        Initialize the email operation.

        Parameters:
        -----------
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top domains to include in the results
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        description : str
            Description of the operation (optional)
        """
        super().__init__(field_name, description or f"Analysis of email field '{field_name}'")
        self.top_n = top_n
        self.min_frequency = min_frequency

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the email analysis operation.

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
            - analyze_privacy_risk: bool, whether to analyze privacy risks

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs
        generate_plots = kwargs.get('generate_plots', True)
        include_timestamp = kwargs.get('include_timestamp', True)
        profile_type = kwargs.get('profile_type', 'email')
        analyze_privacy_risk = kwargs.get('analyze_privacy_risk', True)

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
            reporter.add_operation(f"Analyzing email field: {self.field_name}", details={
                "field_name": self.field_name,
                "top_n": self.top_n,
                "min_frequency": self.min_frequency,
                "operation_type": "email_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 4  # Preparation, analysis, saving results, dictionary
            if generate_plots:
                total_steps += 1  # Add step for generating visualizations

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = EmailAnalyzer.analyze(
                df=df,
                field_name=self.field_name,
                top_n=self.top_n,
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

            write_json(analysis_results, stats_path)
            result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis")

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), f"{self.field_name} statistical analysis")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualization if requested
            if generate_plots and 'top_domains' in analysis_results and analysis_results['top_domains']:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualization"})

                # Create visualization filename with extension "png"
                viz_filename = get_timestamped_filename(
                    base_name=f"{self.field_name}_domains_distribution",
                    extension="png",
                    include_timestamp=include_timestamp
                )
                viz_path = visualizations_dir / viz_filename

                # Create visualization using the visualization module
                from pamola_core.utils.visualization import plot_email_domains
                title = f"Top Email Domains in {self.field_name}"

                viz_result = plot_email_domains(
                    domains=analysis_results['top_domains'],
                    output_path=str(viz_path),
                    title=title
                )

                if not viz_result.startswith("Error"):
                    result.add_artifact("png", viz_path, f"{self.field_name} domains distribution")
                    reporter.add_artifact("png", str(viz_path), f"{self.field_name} domains distribution")
                else:
                    logger.warning(f"Error creating visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualization"})

            # Create and save domain dictionary
            dict_result = EmailAnalyzer.create_domain_dictionary(
                df=df,
                field_name=self.field_name,
                min_count=self.min_frequency,
                **kwargs
            )

            if 'error' not in dict_result:
                # Save dictionary to CSV
                dict_filename = get_timestamped_filename(f"{self.field_name}_domains_dictionary", "csv",
                                                         include_timestamp)
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                dict_df = pd.DataFrame(dict_result['domains'])
                dict_df.to_csv(dict_path, index=False, encoding='utf-8')

                # Save detailed dictionary as JSON
                json_dict_filename = get_timestamped_filename(f"{self.field_name}_domains_dictionary", "json",
                                                              include_timestamp)
                json_dict_path = output_dir / json_dict_filename
                write_json(dict_result, json_dict_path)

                result.add_artifact("csv", dict_path, f"{self.field_name} domains dictionary (CSV)")
                result.add_artifact("json", json_dict_path, f"{self.field_name} domains dictionary (JSON)")

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} domains dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path), f"{self.field_name} domains dictionary (JSON)")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Created domain dictionary"})

            # Add privacy risk assessment if requested
            if analyze_privacy_risk:
                # Perform privacy risk assessment based on email uniqueness
                privacy_risk = self._assess_privacy_risk(df, self.field_name)

                if privacy_risk and len(privacy_risk) > 0:
                    # Save privacy risk assessment to JSON
                    privacy_filename = get_timestamped_filename(f"{self.field_name}_privacy_risk", "json",
                                                                include_timestamp)
                    privacy_path = output_dir / privacy_filename
                    write_json(privacy_risk, privacy_path)

                    result.add_artifact("json", privacy_path, f"{self.field_name} privacy risk assessment")
                    reporter.add_artifact("json", str(privacy_path), f"{self.field_name} privacy risk assessment")

            # Add metrics to the result
            result.add_metric("total_records", analysis_results.get('total_rows', 0))
            result.add_metric("null_count", analysis_results.get('null_count', 0))
            result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
            result.add_metric("valid_count", analysis_results.get('valid_count', 0))
            result.add_metric("valid_percentage", analysis_results.get('valid_percentage', 0))
            result.add_metric("unique_domains", analysis_results.get('unique_domains', 0))

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "valid_emails": analysis_results.get('valid_count', 0),
                "invalid_emails": analysis_results.get('invalid_count', 0),
                "unique_domains": analysis_results.get('unique_domains', 0),
                "null_percentage": analysis_results.get('null_percentage', 0)
            })

            return result

        except Exception as e:
            logger.exception(f"Error in email operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing email field {self.field_name}: {str(e)}"
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

    def _assess_privacy_risk(self, df: pd.DataFrame, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Assess privacy risk based on email uniqueness.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the email field

        Returns:
        --------
        Optional[Dict[str, Any]]
            Privacy risk assessment results or None if assessment cannot be performed
        """
        try:
            # Skip if field doesn't exist or is empty
            if field_name not in df.columns or df[field_name].isna().all():
                return {}

            # Count emails
            total_valid = df[field_name].apply(lambda x: not pd.isna(x)).sum()
            if total_valid == 0:
                return {}

            # Count unique emails
            unique_emails = df[field_name].dropna().nunique()

            # Calculate uniqueness ratio
            uniqueness_ratio = unique_emails / total_valid if total_valid > 0 else 0

            # Assess risk levels
            risk_level = "Low"
            if uniqueness_ratio > 0.9:
                risk_level = "Very High"
            elif uniqueness_ratio > 0.7:
                risk_level = "High"
            elif uniqueness_ratio > 0.5:
                risk_level = "Medium"

            # Find most frequent emails (for potential exclusion)
            value_counts = df[field_name].value_counts()
            most_frequent = value_counts.head(10).to_dict()

            # Calculate singleton count (emails appearing only once)
            singles = value_counts[value_counts == 1]
            singleton_count = len(singles)

            # Create risk assessment
            return {
                'field_name': field_name,
                'total_valid_emails': int(total_valid),
                'unique_emails': int(unique_emails),
                'uniqueness_ratio': round(uniqueness_ratio, 4),
                'risk_level': risk_level,
                'most_frequent_count': len([count for count in value_counts if count > 1]),
                'singleton_count': singleton_count,
                'singleton_percentage': round((singleton_count / total_valid) * 100, 2) if total_valid > 0 else 0,
                'most_frequent_examples': most_frequent
            }
        except Exception as e:
            logger.error(f"Error in privacy risk assessment for {field_name}: {e}", exc_info=True)
            return {}


def analyze_email_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        email_fields: List[str] = None,
        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple email fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    email_fields : List[str], optional
        List of email fields to analyze. If None, tries to find email fields automatically.
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top domains to include in results (default: 20)
        - min_frequency: int, minimum frequency for inclusion in dictionary (default: 1)
        - generate_plots: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps in filenames (default: True)
        - profile_type: str, type of profiling for organizing artifacts (default: 'email')

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    df = data_source.get_dataframe("main")
    if df is None:
        reporter.add_operation("Email fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # Extract operation parameters from kwargs
    top_n = kwargs.get('top_n', 20)
    min_frequency = kwargs.get('min_frequency', 1)

    # If no email fields specified, try to detect them
    if email_fields is None:
        email_fields = []
        for col in df.columns:
            if 'email' in col.lower():
                email_fields.append(col)

        if not email_fields:
            email_fields = ['email']  # Default field name

    # Report on fields to be analyzed
    reporter.add_operation("Email fields analysis", details={
        "fields_count": len(email_fields),
        "fields": email_fields,
        "top_n": top_n,
        "min_frequency": min_frequency,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and email_fields:
        overall_tracker = ProgressTracker(
            total=len(email_fields),
            description=f"Analyzing {len(email_fields)} email fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(email_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(email_fields)}"})

                logger.info(f"Analyzing email field: {field}")

                # Create and execute operation
                operation = EmailOperation(
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
                logger.error(f"Error analyzing email field {field}: {e}", exc_info=True)

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

    reporter.add_operation("Email fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results