"""
Date analyzer module for the project.

This module provides analyzers and operations for date fields, following the
new operation architecture. It includes validation, distribution analysis,
anomaly detection, and visualization capabilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from pamola_core.profiling.commons.date_utils import (
    analyze_date_field,
    estimate_resources
)
from pamola_core.utils.io import write_json, get_timestamped_filename, load_data_operation
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    plot_date_distribution,
    plot_value_distribution
)

# Configure logger
logger = logging.getLogger(__name__)


class DateAnalyzer:
    """
    Analyzer for date fields.

    This analyzer provides methods for analyzing date fields, including
    validation, distribution analysis, and anomaly detection.
    """

    def analyze(self,
                df: pd.DataFrame,
                field_name: str,
                min_year: int = 1940,
                max_year: int = 2005,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a date field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        min_year : int
            Minimum valid year for anomaly detection
        max_year : int
            Maximum valid year for anomaly detection
        **kwargs : dict
            Additional parameters for the analysis:
            - id_column: column to use for group analysis
            - uid_column: column to use for UID analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        # Extract additional parameters
        id_column = kwargs.get('id_column')
        uid_column = kwargs.get('uid_column')

        # Call the utility function for the actual analysis
        results = analyze_date_field(
            df=df,
            field_name=field_name,
            min_year=min_year,
            max_year=max_year,
            id_column=id_column,
            uid_column=uid_column,
            **kwargs
        )

        # For birth dates, calculate age distribution
        if kwargs.get('is_birth_date', False) and not results.get('error'):
            results.update(self._calculate_age_distribution(df, field_name))

        return results

    def estimate_resources(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the date field.

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

    def _calculate_age_distribution(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Calculate age distribution for birth dates.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing birth dates
        field_name : str
            Name of the birth date field

        Returns:
        --------
        Dict[str, Any]
            Age distribution statistics
        """
        today = datetime.now().date()
        valid_dates = pd.to_datetime(df[field_name], errors='coerce')
        valid_mask = ~valid_dates.isna()

        if valid_mask.sum() == 0:
            return {
                "age_distribution": {},
                "age_statistics": {
                    "min_age": None,
                    "max_age": None,
                    "mean_age": None,
                    "median_age": None
                }
            }

        # Calculate ages in years
        ages = []
        age_counts = {}

        for dt in valid_dates[valid_mask]:
            try:
                # Calculate age in years
                birth_date = dt.date()
                age = today.year - birth_date.year - (
                        (today.month, today.day) < (birth_date.month, birth_date.day)
                )

                if age >= 0:  # Only consider non-negative ages
                    ages.append(age)
                    age_counts[age] = age_counts.get(age, 0) + 1
            except (AttributeError, ValueError):
                continue

        if not ages:
            return {
                "age_distribution": {},
                "age_statistics": {
                    "min_age": None,
                    "max_age": None,
                    "mean_age": None,
                    "median_age": None
                }
            }

        # Calculate age groups (5-year intervals)
        age_groups = {}
        for age in ages:
            group = f"{5 * (age // 5)}-{5 * (age // 5) + 4}"
            age_groups[group] = age_groups.get(group, 0) + 1

        # Sort age groups
        sorted_age_groups = {k: age_groups[k] for k in sorted(age_groups.keys(),
                                                              key=lambda x: int(x.split('-')[0]))}

        # Calculate statistics
        age_statistics = {
            "min_age": min(ages),
            "max_age": max(ages),
            "mean_age": sum(ages) / len(ages),
            "median_age": sorted(ages)[len(ages) // 2]
        }

        return {
            "age_distribution": sorted_age_groups,
            "age_statistics": age_statistics
        }


@register(override=True)
class DateOperation(FieldOperation):
    """
    Operation for analyzing date fields.

    This operation wraps the DateAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                 field_name: str,
                 min_year: int = 1940,
                 max_year: int = 2005,
                 id_column: Optional[str] = None,
                 uid_column: Optional[str] = None,
                 description: str = "",
                 generate_plots: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = "date",
                 is_birth_date: Optional[bool] = None):
        """
        Initialize the date operation.

        Parameters:
        -----------
        field_name : str
            The name of the field to analyze
        min_year : int
            Minimum valid year for anomaly detection
        max_year : int
            Maximum valid year for anomaly detection
        id_column : str, optional
            The column to use for group analysis
        uid_column : str, optional
            The column to use for UID analysis
        description : str
            Description of the operation (optional)
        generate_plots : bool
            Whether to generate visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        profile_type : str
            Type of profiling for organizing artifacts
        is_birth_date : bool, optional
            Whether the field is a birth date field
        """
        super().__init__(field_name, description or f"Analysis of date field '{field_name}'")
        self.min_year = min_year
        self.max_year = max_year
        self.id_column = id_column
        self.uid_column = uid_column
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        
        # Set is_birth_date based on the provided value or field name
        if is_birth_date is None:
            self.is_birth_date = self.field_name.lower() in ['birth_day', 'birthdate', 'birth_date', 'dob']
        else:
            self.is_birth_date = is_birth_date

        self.analyzer = DateAnalyzer()

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the date analysis operation.

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
            - is_birth_date: bool, whether the field is a birth date field

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        is_birth_date = kwargs.get('is_birth_date', self.is_birth_date)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
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
            df = load_data_operation(data_source, dataset_name)
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
            reporter.add_operation(f"Analyzing date field: {self.field_name}", details={
                "field_name": self.field_name,
                "min_year": self.min_year,
                "max_year": self.max_year,
                "id_column": self.id_column,
                "uid_column": self.uid_column,
                "is_birth_date": is_birth_date,
                "operation_type": "date_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 5
            if generate_plots:
                total_steps += 3 if is_birth_date else 2

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = self.analyzer.analyze(
                df=df,
                field_name=self.field_name,
                min_year=self.min_year,
                max_year=self.max_year,
                id_column=self.id_column,
                uid_column=self.uid_column,
                is_birth_date=is_birth_date
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

            # Generate visualizations if requested
            if generate_plots:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualizations"})

                self._generate_visualizations(
                    analysis_results,
                    visualizations_dir,
                    include_timestamp,
                    is_birth_date,
                    result,
                    reporter
                )

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Visualizations complete"})

            # Save anomalies to CSV if any
            if 'anomalies' in analysis_results and sum(analysis_results['anomalies'].values()) > 0:
                self._save_anomalies_to_csv(
                    analysis_results,
                    dictionaries_dir,
                    include_timestamp,
                    result,
                    reporter
                )

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Saved anomalies data"})

            # Add metrics to the result
            result.add_metric("total_rows", analysis_results.get('total_records', 0))
            result.add_metric("null_count", analysis_results.get('null_count', 0))
            result.add_metric("valid_count", analysis_results.get('valid_count', 0))
            result.add_metric("invalid_count", analysis_results.get('invalid_count', 0))
            result.add_metric("fill_rate", analysis_results.get('fill_rate', 0))
            result.add_metric("valid_rate", analysis_results.get('valid_rate', 0))

            if 'min_date' in analysis_results:
                result.add_metric("min_date", analysis_results.get('min_date'))

            if 'max_date' in analysis_results:
                result.add_metric("max_date", analysis_results.get('max_date'))

            if 'anomalies' in analysis_results:
                result.add_metric("anomalies_count", sum(analysis_results['anomalies'].values()))

            # Add age metrics if it's a birth date
            if is_birth_date and 'age_statistics' in analysis_results:
                age_stats = analysis_results['age_statistics']
                result.add_metric("min_age", age_stats.get('min_age'))
                result.add_metric("max_age", age_stats.get('max_age'))
                result.add_metric("mean_age", age_stats.get('mean_age'))
                result.add_metric("median_age", age_stats.get('median_age'))

            # Update progress to completion
            if progress_tracker:
                progress_tracker.update(1, {"step": "Operation complete", "status": "success"})

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "valid_dates": analysis_results.get('valid_count', 0),
                "invalid_dates": analysis_results.get('invalid_count', 0),
                "date_range": f"{analysis_results.get('min_date', 'N/A')} to {analysis_results.get('max_date', 'N/A')}",
                "anomalies_found": sum(analysis_results.get('anomalies', {}).values()),
                "groups_with_changes": analysis_results.get('date_changes_within_group', {}).get('groups_with_changes',
                                                                                                 0) if 'date_changes_within_group' in analysis_results else 0
            })

            return result

        except Exception as e:
            logger.exception(f"Error in date operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing date field {self.field_name}: {str(e)}"
            )

    def _generate_visualizations(self,
                                 analysis_results: Dict[str, Any],
                                 vis_dir: Path,
                                 include_timestamp: bool,
                                 is_birth_date: bool,
                                 result: OperationResult,
                                 reporter: Any):
        """
        Generate visualizations for the date field analysis.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        is_birth_date : bool
            Whether the field is a birth date field
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        # Generate year distribution visualization if we have data
        if 'year_distribution' in analysis_results and analysis_results['year_distribution']:
            year_filename = get_timestamped_filename(f"{self.field_name}_year_distribution", "png", include_timestamp)
            year_path = vis_dir / year_filename

            # Create visualization using the visualization module
            title = f"Year Distribution of {self.field_name}"
            if is_birth_date:
                title = "Birth Year Distribution"

            # Use the year_distribution data
            year_result = plot_date_distribution(
                {'year_distribution': analysis_results['year_distribution']},
                str(year_path),
                title=title
            )

            if not year_result.startswith("Error"):
                result.add_artifact("png", year_path, f"{self.field_name} year distribution")
                reporter.add_artifact("png", str(year_path), f"{self.field_name} year distribution")

        # Generate month distribution visualization if we have data
        if 'month_distribution' in analysis_results and analysis_results['month_distribution']:
            month_filename = get_timestamped_filename(f"{self.field_name}_month_distribution", "png", include_timestamp)
            month_path = vis_dir / month_filename

            # Create visualization using the visualization module
            title = f"Month Distribution of {self.field_name}"
            if is_birth_date:
                title = "Birth Month Distribution"

            month_result = plot_value_distribution(
                analysis_results['month_distribution'],
                str(month_path),
                title=title
            )

            if not month_result.startswith("Error"):
                result.add_artifact("png", month_path, f"{self.field_name} month distribution")
                reporter.add_artifact("png", str(month_path), f"{self.field_name} month distribution")

        # Generate day of week distribution visualization if we have data
        if 'day_of_week_distribution' in analysis_results and analysis_results['day_of_week_distribution']:
            dow_filename = get_timestamped_filename(f"{self.field_name}_dow_distribution", "png", include_timestamp)
            dow_path = vis_dir / dow_filename

            # Create visualization using the visualization module
            title = f"Day of Week Distribution of {self.field_name}"
            if is_birth_date:
                title = "Birth Day of Week Distribution"

            dow_result = plot_value_distribution(
                analysis_results['day_of_week_distribution'],
                str(dow_path),
                title=title
            )

            if not dow_result.startswith("Error"):
                result.add_artifact("png", dow_path, f"{self.field_name} day of week distribution")
                reporter.add_artifact("png", str(dow_path), f"{self.field_name} day of week distribution")

        # Generate age distribution visualization if it's a birth date and we have data
        if is_birth_date and 'age_distribution' in analysis_results and analysis_results['age_distribution']:
            age_filename = get_timestamped_filename(f"{self.field_name}_age_distribution", "png", include_timestamp)
            age_path = vis_dir / age_filename

            # Create visualization using the visualization module
            title = "Age Distribution"

            age_result = plot_value_distribution(
                analysis_results['age_distribution'],
                str(age_path),
                title=title,
                x_label="Age Group",
                y_label="Count"
            )

            if not age_result.startswith("Error"):
                result.add_artifact("png", age_path, "Age distribution")
                reporter.add_artifact("png", str(age_path), "Age distribution")

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
            # Collect anomalies into a DataFrame
            anomalies_data = []

            for anomaly_type, count in analysis_results['anomalies'].items():
                if count > 0 and f'{anomaly_type}_examples' in analysis_results:
                    for example in analysis_results[f'{anomaly_type}_examples']:
                        anomaly_row = {
                            'index': example[0] if isinstance(example, tuple) and len(example) > 0 else None,
                            'value': example[1] if isinstance(example, tuple) and len(example) > 1 else example,
                            'anomaly_type': anomaly_type
                        }

                        if anomaly_type in ['too_old', 'future_dates', 'too_young'] and isinstance(example,
                                                                                                   tuple) and len(
                            example) > 2:
                            anomaly_row['year'] = example[2]

                        anomalies_data.append(anomaly_row)

            if anomalies_data:
                # Create anomalies CSV filename
                anomalies_filename = get_timestamped_filename(f"{self.field_name}_anomalies", "csv", include_timestamp)
                anomalies_path = dict_dir / anomalies_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                anomalies_df = pd.DataFrame(anomalies_data)
                anomalies_df.to_csv(anomalies_path, index=False, encoding='utf-8')

                # Add artifact to result and reporter
                result.add_artifact("csv", anomalies_path, f"{self.field_name} anomalies")
                reporter.add_artifact("csv", str(anomalies_path), f"{self.field_name} anomalies")

        except Exception as e:
            logger.warning(f"Error saving anomalies for {self.field_name}: {e}")
            reporter.add_operation(f"Saving anomalies for {self.field_name}", status="warning",
                                   details={"warning": str(e)})


def analyze_date_fields(data_source: DataSource,
                        task_dir: Path,
                        reporter: Any,
                        date_fields: List[str] = None,
                        id_column: str = 'resume_id',
                        uid_column: str = 'UID',
                        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple date fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    date_fields : List[str], optional
        List of date fields to analyze. If None, tries to find date fields automatically.
    id_column : str
        The column to use for group analysis
    uid_column : str
        The column to use for UID analysis
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
        reporter.add_operation("Date fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # If no date fields specified, try to find them
    if date_fields is None:
        date_fields = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['birth', 'day', 'date', 'time']):
                date_fields.append(col)

    # Check if id_column and uid_column exist
    actual_id_column = id_column if id_column in df.columns else None
    actual_uid_column = uid_column if uid_column in df.columns else None

    # Report on fields to be analyzed
    reporter.add_operation("Date fields analysis", details={
        "fields_count": len(date_fields),
        "fields": date_fields,
        "id_column": actual_id_column,
        "uid_column": actual_uid_column,
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and date_fields:
        overall_tracker = ProgressTracker(
            total=len(date_fields),
            description=f"Analyzing {len(date_fields)} date fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(date_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(date_fields)}"})

                logger.info(f"Analyzing date field: {field}")

                # Check if field is a birth date
                is_birth_date = any(keyword in field.lower() for keyword in ['birth', 'dob'])

                # Create and execute operation
                operation = DateOperation(
                    field,
                    id_column=actual_id_column,
                    uid_column=actual_uid_column,
                )
                result = operation.execute(data_source, task_dir, reporter, is_birth_date=is_birth_date, **kwargs)

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
                logger.error(f"Error analyzing date field {field}: {e}", exc_info=True)

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

    reporter.add_operation("Date fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results