"""
Attribute profiler operation for the HHR project.

This module provides operations for automatically profiling attributes of input datasets
to categorize each column by its role in anonymization and synthesis tasks. It supports
both pandas.DataFrame and CSV files (using io.py).
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np

from pamola_core.profiling.commons.attribute_utils import (
    analyze_dataset_attributes,
    load_attribute_dictionary
)
from pamola_core.utils.io import (
    ensure_directory,
    write_json,
    write_dataframe_to_csv,
    get_timestamped_filename, 
    load_data_operation
    
)
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    create_pie_chart,
    create_bar_plot,
    create_scatter_plot
)

# Configure logger
logger = logging.getLogger(__name__)


@register()
class DataAttributeProfilerOperation(BaseOperation):
    """
    Operation for automatically profiling attributes of input datasets.

    This operation analyzes datasets to categorize each column by its role in
    anonymization and synthesis tasks, generating metrics, visualizations, and
    recommendations for data handling.
    """

    def __init__(self,
                 name: str = "DataAttributeProfiler",
                 description: str = "Automatic profiling of dataset attributes",
                 dictionary_path: Optional[Union[str, Path]] = None,
                 language: str = "en",
                 sample_size: int = 10,
                 max_columns: Optional[int] = None,
                 id_column: Optional[str] = None,
                 include_timestamp: bool = True):
        """
        Initialize the attribute profiler operation.

        Parameters:
        -----------
        name : str
            Name of the operation
        description : str
            Description of the operation
        dictionary_path : str or Path, optional
            Path to the attribute dictionary file (if None, will look in
            DATA/external_dictionaries/attribute_roles_dictionary.json)
        language : str
            Language code for keyword matching
        sample_size : int
            Number of sample values to return per column
        max_columns : int, optional
            Maximum number of columns to analyze (for large datasets)
        id_column : str, optional
            Name of ID column for record-level analysis
        include_timestamp : bool
            Whether to include timestamps in filenames
        """
        super().__init__(name, description)
        self.dictionary_path = dictionary_path
        self.language = language
        self.sample_size = sample_size
        self.max_columns = max_columns
        self.id_column = id_column
        self.include_timestamp = include_timestamp

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute comprehensive attribute profiling for a dataset.

        This method performs end-to-end analysis of dataset attributes, including:
        - Loading custom or default attribute dictionary
        - Analyzing column characteristics
        - Generating multiple artifacts
        - Tracking progress
        - Reporting results

        Args:
            data_source (DataSource): Source of input data
            task_dir (Path): Directory for storing operation artifacts
            reporter (Any): Reporting mechanism for tracking progress and artifacts
            progress_tracker (ProgressTracker, optional): Tracks operation progress
            **kwargs (dict): Additional configuration parameters

        Returns:
            OperationResult: Detailed results of the attribute profiling operation
        """
        # Extract operation parameters with fallback to default values
        dictionary_path = kwargs.get('dictionary_path', self.dictionary_path)
        language = kwargs.get('language', self.language)
        sample_size = kwargs.get('sample_size', self.sample_size)
        max_columns = kwargs.get('max_columns', self.max_columns)
        id_column = kwargs.get('id_column', self.id_column)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)

        # Prepare output directories for artifacts
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

        # Initialize operation result with success status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Configure progress tracking if provided
        if progress_tracker:
            progress_tracker.update(0, {"step": "Preparation", "operation": self.name})
            progress_tracker.total = 5  # Define total steps for tracking

        try:
            # Retrieve DataFrame from data source
            df = load_data_operation(data_source)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Log initial dataset information
            logger.info(f"Starting attribute profiling on DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Report operation details
            reporter.add_operation(f"Attribute Profiling", details={
                "records_count": len(df),
                "columns_count": len(df.columns),
                "language": language,
                "operation_type": "attribute_profiling"
            })

            # Update progress tracker
            if progress_tracker:
                progress_tracker.update(1, {"step": "Loading dictionary and preparing analysis"})

            # Load attribute dictionary (custom or default)
            dictionary = load_attribute_dictionary(dictionary_path)

            # Log attribute analysis start
            logger.info("Analyzing dataset attributes")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analyzing dataset attributes"})

            # Perform comprehensive attribute analysis
            analysis_results = analyze_dataset_attributes(
                df=df,
                dictionary=dictionary,
                language=language,
                sample_size=sample_size,
                max_columns=max_columns,
                id_column=id_column
            )

            # Save attribute roles to JSON
            roles_filename = get_timestamped_filename("attribute_roles", "json", include_timestamp)
            roles_path = output_dir / roles_filename
            write_json(analysis_results, roles_path)

            # Register artifacts
            result.add_artifact("json", roles_path, "Attribute roles analysis", category="metrics")
            reporter.add_artifact("json", str(roles_path), "Attribute roles analysis")

            # Create and save entropy DataFrame
            entropy_filename = get_timestamped_filename("attribute_entropy", "csv", include_timestamp)
            entropy_path = output_dir / entropy_filename

            entropy_data = [
                {
                    "column_name": col_name,
                    "role": col_data["role"],
                    "entropy": col_data["statistics"].get("entropy", 0),
                    "normalized_entropy": col_data["statistics"].get("normalized_entropy", 0),
                    "uniqueness_ratio": col_data["statistics"].get("uniqueness_ratio", 0),
                    "missing_rate": col_data["statistics"].get("missing_rate", 0),
                    "inferred_type": col_data["statistics"].get("inferred_type", "unknown")
                }
                for col_name, col_data in analysis_results["columns"].items()
                if "statistics" in col_data
            ]

            if entropy_data:
                entropy_df = pd.DataFrame(entropy_data)
                write_dataframe_to_csv(entropy_df, entropy_path)
                result.add_artifact("csv", entropy_path, "Attribute entropy and uniqueness", category="metrics")
                reporter.add_artifact("csv", str(entropy_path), "Attribute entropy and uniqueness")

            # Save sample values
            sample_filename = get_timestamped_filename("attribute_sample", "json", include_timestamp)
            sample_path = output_dir / sample_filename

            sample_data = {
                col_name: {
                    "role": col_data["role"],
                    "inferred_type": col_data["statistics"].get("inferred_type", "unknown"),
                    "samples": col_data["statistics"]["samples"]
                }
                for col_name, col_data in analysis_results["columns"].items()
                if "statistics" in col_data and "samples" in col_data["statistics"]
            }

            write_json(sample_data, sample_path)
            result.add_artifact("json", sample_path, "Attribute sample values", category="dictionary")
            reporter.add_artifact("json", str(sample_path), "Attribute sample values")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saving analysis results"})

            # Generate visualizations
            self._create_visualizations(
                analysis_results,
                visualizations_dir,
                include_timestamp,
                result,
                reporter
            )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Creating visualizations"})

            # Process quasi-identifiers
            quasi_identifiers = analysis_results["column_groups"]["QUASI_IDENTIFIER"]

            if quasi_identifiers:
                result.add_metric("quasi_identifiers", quasi_identifiers)

                quasi_filename = get_timestamped_filename("quasi_identifiers", "json", include_timestamp)
                quasi_path = output_dir / quasi_filename

                write_json({"quasi_identifiers": quasi_identifiers}, quasi_path)
                result.add_artifact("json", quasi_path, "Quasi-identifiers list", category="metrics")
                reporter.add_artifact("json", str(quasi_path), "Quasi-identifiers list")

            # Add comprehensive metrics to result
            result.add_metric("total_columns", len(analysis_results["columns"]))
            result.add_metric("direct_identifiers_count", analysis_results["summary"]["DIRECT_IDENTIFIER"])
            result.add_metric("quasi_identifiers_count", analysis_results["summary"]["QUASI_IDENTIFIER"])
            result.add_metric("sensitive_attributes_count", analysis_results["summary"]["SENSITIVE_ATTRIBUTE"])
            result.add_metric("indirect_identifiers_count", analysis_results["summary"]["INDIRECT_IDENTIFIER"])
            result.add_metric("non_sensitive_count", analysis_results["summary"]["NON_SENSITIVE"])

            if "dataset_metrics" in analysis_results:
                result.add_metric("avg_entropy", analysis_results["dataset_metrics"]["avg_entropy"])
                result.add_metric("avg_uniqueness", analysis_results["dataset_metrics"]["avg_uniqueness"])

            # Add conflicts count if applicable
            if "conflicts" in analysis_results:
                result.add_metric("conflicts_count", len(analysis_results["conflicts"]))

            # Finalize progress tracking
            if progress_tracker:
                progress_tracker.update(1, {"step": "Operation complete", "status": "success"})

            # Report operation summary
            reporter.add_operation("Attribute Profiling Completed", details={
                "direct_identifiers": analysis_results["summary"]["DIRECT_IDENTIFIER"],
                "quasi_identifiers": analysis_results["summary"]["QUASI_IDENTIFIER"],
                "sensitive_attributes": analysis_results["summary"]["SENSITIVE_ATTRIBUTE"],
                "indirect_identifiers": analysis_results["summary"]["INDIRECT_IDENTIFIER"],
                "non_sensitive": analysis_results["summary"]["NON_SENSITIVE"],
                "conflicts": len(analysis_results.get("conflicts", []))
            })

            return result

        except Exception as e:
            # Comprehensive error handling
            logger.exception(f"Error in attribute profiling operation: {e}")

            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            reporter.add_operation("Attribute Profiling",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in attribute profiling: {str(e)}"
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

    def _create_visualizations(self,
                               analysis_results: Dict[str, Any],
                               vis_dir: Path,
                               include_timestamp: bool,
                               result: OperationResult,
                               reporter: Any):
        """
        Create visualizations for attribute profiling.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of attribute analysis
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        try:
            # 1. Create attribute role pie chart
            roles_summary = analysis_results["summary"]

            # Only create chart if we have data
            if sum(roles_summary.values()) > 0:
                pie_filename = get_timestamped_filename("attribute_type_pie", "png", include_timestamp)
                pie_path = vis_dir / pie_filename

                pie_result = create_pie_chart(
                    data=roles_summary,
                    output_path=str(pie_path),
                    title="Attribute Role Distribution",
                    show_percentages=True,
                    hole=0.3  # Create a donut chart
                )

                if not pie_result.startswith("Error"):
                    result.add_artifact("png", pie_path, "Attribute role distribution", category="visualization")
                    reporter.add_artifact("png", str(pie_path), "Attribute role distribution")

            # 2. Create entropy vs uniqueness scatter plot
            entropy_data = []
            for col_name, col_data in analysis_results["columns"].items():
                if "statistics" in col_data:
                    entropy_data.append({
                        "column_name": col_name,
                        "role": col_data["role"],
                        "entropy": col_data["statistics"].get("entropy", 0),
                        "uniqueness_ratio": col_data["statistics"].get("uniqueness_ratio", 0)
                    })

            if entropy_data:
                df_entropy = pd.DataFrame(entropy_data)

                # Prepare data for scatter plot
                entropy_filename = get_timestamped_filename("entropy_vs_uniqueness", "png", include_timestamp)
                entropy_path = vis_dir / entropy_filename

                # Use different colors by role
                color_map = {
                    "DIRECT_IDENTIFIER": "red",
                    "QUASI_IDENTIFIER": "orange",
                    "SENSITIVE_ATTRIBUTE": "purple",
                    "INDIRECT_IDENTIFIER": "blue",
                    "NON_SENSITIVE": "green"
                }

                # Create scatter plot
                scatter_result = create_scatter_plot(
                    x_data=df_entropy["entropy"],
                    y_data=df_entropy["uniqueness_ratio"],
                    output_path=str(entropy_path),
                    title="Entropy vs Uniqueness by Attribute Role",
                    x_label="Entropy",
                    y_label="Uniqueness Ratio",
                    #color=df_entropy["role"].map(color_map),
                    text=df_entropy["column_name"],
                    add_trendline=False
                )

                if not scatter_result.startswith("Error"):
                    result.add_artifact("png", entropy_path, "Entropy vs uniqueness analysis", category="visualization")
                    reporter.add_artifact("png", str(entropy_path), "Entropy vs uniqueness analysis")

            # 3. Create inferred type bar chart
            inferred_types = {}
            for col_data in analysis_results["columns"].values():
                if "statistics" in col_data and "inferred_type" in col_data["statistics"]:
                    inferred_type = col_data["statistics"]["inferred_type"]
                    inferred_types[inferred_type] = inferred_types.get(inferred_type, 0) + 1

            if inferred_types:
                types_filename = get_timestamped_filename("inferred_types", "png", include_timestamp)
                types_path = vis_dir / types_filename

                bar_result = create_bar_plot(
                    data=inferred_types,
                    output_path=str(types_path),
                    title="Inferred Data Type Distribution",
                    orientation="h",
                    x_label="Count",
                    y_label="Data Type"
                )

                if not bar_result.startswith("Error"):
                    result.add_artifact("png", types_path, "Inferred data type distribution", category="visualization")
                    reporter.add_artifact("png", str(types_path), "Inferred data type distribution")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}", exc_info=True)
            reporter.add_operation("Creating visualizations", status="warning",
                                   details={"warning": f"Error creating some visualizations: {str(e)}"})