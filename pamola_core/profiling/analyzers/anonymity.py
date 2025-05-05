"""
K-anonymity profiling operation for the HHR project.

This module provides operations for analyzing k-anonymity in data, identifying
quasi-identifiers that may compromise privacy, and generating visualizations
and reports about data anonymization risks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from pamola_core.profiling.commons.anonymity_utils import (
    get_field_combinations,
    create_ka_index_map,
    calculate_k_anonymity,
    find_vulnerable_records,
    prepare_metrics_for_spider_chart,
    prepare_field_uniqueness_data,
    save_ka_index_map,
    save_ka_metrics,
    save_vulnerable_records
)
from pamola_core.utils.io import ensure_directory, write_json, get_timestamped_filename, load_data_operation
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    create_bar_plot,
    create_line_plot,
    create_spider_chart,
    create_combined_chart
)

# Configure logger
logger = logging.getLogger(__name__)


@register()
class PreKAnonymityProfilingOperation(BaseOperation):
    """
    Operation for preliminary profiling of data for k-anonymity analysis.

    This operation analyzes combinations of fields to assess their potential
    as quasi-identifiers and calculates k-anonymity metrics for each combination.
    It generates visualizations and reports about anonymization risks.
    """

    def __init__(self,
                 name: str = "PreKAnonymityProfiling",
                 description: str = "Preliminary profiling for k-anonymity analysis",
                 min_combination_size: int = 2,
                 max_combination_size: int = 4,
                 treshold_k: int = 5,
                 fields_combinations: List = None,
                 excluded_combinations: List = None,
                 id_fields: List = [],
                 include_timestamp: bool = True):
        """
        Initialize the k-anonymity profiling operation.

        Parameters:
        -----------
        name : str
            Name of the operation
        description : str
            Description of the operation
        min_combination_size : int
            Minimum size of field combinations to analyze
        max_combination_size : int
            Maximum size of field combinations to analyze
        treshold_k : int
            Threshold for vulnerability (k < treshold_k is considered vulnerable)
        """
        super().__init__(name, description)
        self.min_combination_size = min_combination_size
        self.max_combination_size = max_combination_size
        self.treshold_k = treshold_k
        self.fields_combinations = fields_combinations
        self.excluded_combinations = excluded_combinations
        self.id_fields = id_fields
        self.include_timestamp = include_timestamp

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the k-anonymity profiling operation.

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
            Additional parameters:
            - fields_combinations: List of fields combinations to analyze
            - id_fields: List of ID fields for vulnerable records identification
            - excluded_combinations: List of combinations to exclude
            - treshold_k: Threshold for vulnerability (overrides constructor value)
            - include_timestamp: Whether to include timestamps in filenames

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        fields_combinations = kwargs.get('fields_combinations', self.fields_combinations)
        excluded_combinations = kwargs.get('excluded_combinations', self.excluded_combinations)
        id_fields = kwargs.get('id_fields', self.id_fields)
        treshold_k = kwargs.get('treshold_k', self.treshold_k)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(0, {"step": "Preparation", "operation": self.name})
            progress_tracker.total = 6  # Total steps

        try:
            # Get DataFrame from data source
            df = load_data_operation(data_source)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Log basic information
            logger.info(
                f"Starting k-anonymity profiling on DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Add operation to reporter
            reporter.add_operation(f"K-Anonymity Profiling", details={
                "records_count": len(df),
                "columns_count": len(df.columns),
                "treshold_k": treshold_k,
                "operation_type": "ka_profiling"
            })

            # If no specific field combinations provided, use all columns or detect quasi-identifiers
            if fields_combinations is None:
                # Use all columns except ID fields as potential quasi-identifiers
                all_fields = list(df.columns)

                # Exclude ID fields
                quasi_identifier_fields = [field for field in all_fields if field not in id_fields]

                # Generate all combinations within the size limits
                field_combinations = get_field_combinations(
                    quasi_identifier_fields,
                    min_size=self.min_combination_size,
                    max_size=self.max_combination_size,
                    excluded_combinations=excluded_combinations
                )
            else:
                # Use the provided field combinations
                field_combinations = fields_combinations

            if not field_combinations:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid field combinations to analyze"
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Generated field combinations",
                    "combinations_count": len(field_combinations)
                })

            # Create KA index map
            ka_index_map = create_ka_index_map(field_combinations)

            # Log the index map
            logger.info(f"Created {len(ka_index_map)} KA indices")

            # Save KA index map to CSV
            ka_index_map_path = dictionaries_dir / "ka_index_map.csv"
            save_ka_index_map(ka_index_map, str(ka_index_map_path))

            # Add to result and reporter
            result.add_artifact("csv", ka_index_map_path, "KA Index Map", category="dictionary")
            reporter.add_artifact("csv", str(ka_index_map_path), "KA Index Map")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Created KA index map"})

            # Calculate metrics for each KA index
            ka_metrics = {}
            vulnerable_records = {}

            # Set up progress reporting
            total_combinations = len(ka_index_map)
            combination_tracker = None

            if progress_tracker:
                combination_tracker = ProgressTracker(
                    total=total_combinations,
                    description="Analyzing field combinations",
                    unit="combinations"
                )

            # Process each combination
            for i, (ka_index, fields) in enumerate(ka_index_map.items()):
                logger.info(f"Analyzing combination {i + 1}/{total_combinations}: {ka_index} ({', '.join(fields)})")

                # Calculate k-anonymity metrics
                metrics = calculate_k_anonymity(df, fields, progress_tracker=combination_tracker)

                # Add to results
                if "error" not in metrics:
                    ka_metrics[ka_index] = metrics

                    # Find vulnerable records
                    vuln_records = find_vulnerable_records(
                        df,
                        fields,
                        k_threshold=treshold_k,
                        id_field=id_fields[0] if id_fields else None
                    )

                    vulnerable_records[ka_index] = {
                        **vuln_records,
                        "min_k": metrics.get("min_k", 0)
                    }
                else:
                    logger.warning(f"Error analyzing {ka_index}: {metrics.get('error')}")

                # Update main progress
                if combination_tracker:
                    combination_tracker.update(1, {
                        "combination": ka_index,
                        "fields": ", ".join(fields),
                        "progress": f"{i + 1}/{total_combinations}"
                    })

            # Close combination tracker
            if combination_tracker:
                combination_tracker.close()

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Calculated k-anonymity metrics",
                    "metrics_count": len(ka_metrics)
                })

            # Save metrics to CSV
            metrics_filename = get_timestamped_filename("ka_metrics", "csv", include_timestamp)
            metrics_path = output_dir / metrics_filename
            save_ka_metrics(ka_metrics, str(metrics_path), ka_index_map)

            # Add to result and reporter
            result.add_artifact("csv", metrics_path, "KA Metrics", category="metrics")
            reporter.add_artifact("csv", str(metrics_path), "KA Metrics")

            # Save vulnerable records to JSON
            vulnerable_filename = get_timestamped_filename("ka_vulnerable_records", "json", include_timestamp)
            vulnerable_path = output_dir / vulnerable_filename
            save_vulnerable_records(vulnerable_records, str(vulnerable_path))

            # Add to result and reporter
            result.add_artifact("json", vulnerable_path, "KA Vulnerable Records", category="metrics")
            reporter.add_artifact("json", str(vulnerable_path), "KA Vulnerable Records")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Generated metrics files"})

            # Create visualizations if metrics are available
            if ka_metrics:
                self._create_visualizations(
                    ka_metrics,
                    ka_index_map,
                    df,
                    field_combinations,
                    visualizations_dir,
                    include_timestamp,
                    treshold_k,
                    result,
                    reporter
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Created visualizations"})

            # Calculate individual field uniqueness
            all_fields = set()
            for fields in field_combinations:
                all_fields.update(fields)

            field_uniqueness = prepare_field_uniqueness_data(df, list(all_fields))

            # Save field uniqueness data
            uniqueness_filename = get_timestamped_filename("field_uniqueness", "json", include_timestamp)
            uniqueness_path = output_dir / uniqueness_filename
            write_json(field_uniqueness, str(uniqueness_path))

            # Add to result and reporter
            result.add_artifact("json", uniqueness_path, "Field Uniqueness Metrics", category="metrics")
            reporter.add_artifact("json", str(uniqueness_path), "Field Uniqueness Metrics")

            # Update progress to completion
            if progress_tracker:
                progress_tracker.update(1, {"step": "Operation complete", "status": "success"})

            # Add metrics to the result for easy access
            for ka_index, metrics in ka_metrics.items():
                result.add_nested_metric("ka_metrics", ka_index, {
                    "min_k": metrics.get("min_k", 0),
                    "mean_k": metrics.get("mean_k", 0),
                    "unique_percentage": metrics.get("unique_percentage", 0),
                    "vulnerable_percentage": 100 - metrics.get("threshold_metrics", {}).get("k≥5", 0)
                })

            # Add operation summary to reporter
            top_risks = sorted(
                [(ka, metrics.get("min_k", float('inf'))) for ka, metrics in ka_metrics.items()],
                key=lambda x: x[1]
            )[:3]

            reporter.add_operation("K-Anonymity Profiling Completed", details={
                "analyzed_combinations": len(ka_metrics),
                "top_risk_combinations": [f"{ka} (min_k={k})" for ka, k in top_risks],
                "threshold_k": treshold_k
            })

            return result

        except Exception as e:
            logger.exception(f"Error in k-anonymity profiling operation: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation("K-Anonymity Profiling",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in k-anonymity profiling: {str(e)}"
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

    @staticmethod
    def _create_visualizations(ka_metrics: Dict[str, Dict[str, Any]],
                               ka_index_map: Dict[str, List[str]],
                               df: pd.DataFrame,
                               field_combinations: List[List[str]],
                               vis_dir: Path,
                               include_timestamp: bool,
                               treshold_k: int,
                               result: OperationResult,
                               reporter: Any):
        """
        Create visualizations for k-anonymity metrics.

        Parameters:
        -----------
        ka_metrics : Dict[str, Dict[str, Any]]
            Dictionary mapping KA indices to their metrics
        ka_index_map : Dict[str, List[str]]
            Mapping from KA indices to field lists
        df : pd.DataFrame
            Original DataFrame
        field_combinations : List[List[str]]
            List of field combinations
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        treshold_k : int
            Threshold for vulnerability
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        if not ka_metrics:
            logger.warning("No metrics available for visualization")
            return

        try:
            # 1. Create k-range distribution visualization
            k_range_data = {}
            for ka_index, metrics in ka_metrics.items():
                if "k_range_distribution" in metrics:
                    k_range_data[ka_index] = metrics["k_range_distribution"]

            if k_range_data:
                # Convert to DataFrame for easier plotting
                df_k_range = pd.DataFrame(k_range_data)

                # Create bar plot
                k_range_filename = get_timestamped_filename("ka_k_distribution", "png", include_timestamp)
                k_range_path = vis_dir / k_range_filename

                # Convert DataFrame to dictionary format expected by create_bar_plot
                k_range_result = create_bar_plot(
                    data=df_k_range.to_dict(),  # Convert to dictionary format
                    output_path=str(k_range_path),
                    title="K-Anonymity Range Distribution",
                    orientation="h",
                    y_label="K Range",
                    x_label="Percentage of Records (%)"
                )

                if not k_range_result.startswith("Error"):
                    result.add_artifact("png", k_range_path, "K-anonymity range distribution", category="visualization")
                    reporter.add_artifact("png", str(k_range_path), "K-anonymity range distribution")

            # 2. Create threshold curve visualization
            threshold_data = {}
            for ka_index, metrics in ka_metrics.items():
                if "threshold_metrics" in metrics:
                    threshold_data[ka_index] = metrics["threshold_metrics"]

            if threshold_data:
                # Convert to DataFrame for easier plotting
                df_threshold = pd.DataFrame(threshold_data)

                # Create line plot
                threshold_filename = get_timestamped_filename("ka_vulnerable_curve", "png", include_timestamp)
                threshold_path = vis_dir / threshold_filename

                threshold_result = create_line_plot(
                    data=df_threshold.T,  # Transpose for better format
                    output_path=str(threshold_path),
                    title="Records Meeting K-Anonymity Thresholds",
                    x_label="K Threshold",
                    y_label="Percentage of Records (%)",
                    add_markers=True
                )

                if not threshold_result.startswith("Error"):
                    result.add_artifact("png", threshold_path, "K-anonymity threshold compliance",
                                        category="visualization")
                    reporter.add_artifact("png", str(threshold_path), "K-anonymity threshold compliance")

            # 3. Create spider chart for multi-metric comparison
            spider_data = prepare_metrics_for_spider_chart(ka_metrics)

            if spider_data:
                spider_filename = get_timestamped_filename("ka_comparison_spider", "png", include_timestamp)
                spider_path = vis_dir / spider_filename

                spider_result = create_spider_chart(
                    data=spider_data,
                    output_path=str(spider_path),
                    title="K-Anonymity Metrics Comparison",
                    normalize_values=False,  # Values are already normalized
                    fill_area=True
                )

                if not spider_result.startswith("Error"):
                    result.add_artifact("png", spider_path, "K-anonymity metrics comparison", category="visualization")
                    reporter.add_artifact("png", str(spider_path), "K-anonymity metrics comparison")

            # 4. Create field uniqueness visualization
            # Get all individual fields
            all_fields = set()
            for fields in field_combinations:
                all_fields.update(fields)

            # Prepare data for field uniqueness visualization
            field_uniqueness = prepare_field_uniqueness_data(df, list(all_fields))

            if field_uniqueness:
                # Prepare data for the combined chart
                fields = list(field_uniqueness.keys())
                unique_counts = [field_uniqueness[f].get("unique_values", 0) for f in fields]
                uniqueness_percent = [field_uniqueness[f].get("uniqueness_percentage", 0) for f in fields]

                # Create combined chart
                uniqueness_filename = get_timestamped_filename("ka_field_uniqueness", "png", include_timestamp)
                uniqueness_path = vis_dir / uniqueness_filename

                uniqueness_result = create_combined_chart(
                    primary_data=dict(zip(fields, unique_counts)),
                    secondary_data=dict(zip(fields, uniqueness_percent)),
                    output_path=str(uniqueness_path),
                    title="Quasi-Identifier Analysis",
                    primary_type="bar",
                    secondary_type="line",
                    x_label="Field",
                    primary_y_label="Unique Values Count",
                    secondary_y_label="Uniqueness (%)",
                    primary_color="steelblue",
                    secondary_color="crimson"
                )

                if not uniqueness_result.startswith("Error"):
                    result.add_artifact("png", uniqueness_path, "Field uniqueness analysis", category="visualization")
                    reporter.add_artifact("png", str(uniqueness_path), "Field uniqueness analysis")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}", exc_info=True)
            reporter.add_operation("Creating visualizations", status="warning",
                                   details={"warning": f"Error creating some visualizations: {str(e)}"})