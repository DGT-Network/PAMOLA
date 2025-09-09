"""
Date analyzer module for the project.

This module provides analyzers and operations for date fields, following the
new operation architecture. It includes validation, distribution analysis,
anomaly detection, and visualization capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Union

import pandas as pd

import dask
import dask.dataframe as dd

from pamola_core.profiling.commons.date_utils import (
    analyze_date_field,
    estimate_resources,
)
from pamola_core.utils.io import (
    write_json,
    load_data_operation,
    write_dataframe_to_csv,
    load_settings_operation,
)
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import (
    OperationArtifact,
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import (
    plot_date_distribution,
    plot_value_distribution,
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

# Configure logger
logger = logging.getLogger(__name__)


class DateAnalyzer:
    """
    Analyzer for date fields.

    This analyzer provides methods for analyzing date fields, including
    validation, distribution analysis, and anomaly detection.
    """

    def analyze(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        field_name: str,
        id_column: Optional[str],
        uid_column: Optional[str],
        min_year: int = 1940,
        max_year: int = 2005,
        chunk_size: int = 10000,
        is_birth_date: bool = False,
        use_dask: bool = False,
        use_vectorization: bool = False,
        npartitions: Optional[int] = 2,
        parallel_processes: Optional[int] = 1,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze a date field in the given DataFrame.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            The DataFrame to process.
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
        # Call the utility function for the actual analysis
        results = analyze_date_field(
            df=df,
            field_name=field_name,
            min_year=min_year,
            max_year=max_year,
            id_column=id_column,
            uid_column=uid_column,
            chunk_size=chunk_size,
            use_dask=use_dask,
            is_birth_date=is_birth_date,
            use_vectorization=use_vectorization,
            npartitions=npartitions,
            parallel_processes=parallel_processes,
            progress_tracker=progress_tracker,
            task_logger=task_logger,
        )

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


@register(override=True)
class DateOperation(FieldOperation):
    """
    Operation for analyzing date fields.

    This operation wraps the DateAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(
        self,
        field_name: str,
        min_year: int = 1940,
        max_year: int = 2005,
        id_column: Optional[str] = None,
        uid_column: Optional[str] = None,
        description: str = "",
        profile_type: str = "date",
        is_birth_date: Optional[bool] = None,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        use_dask: bool = False,
        use_cache: bool = True,
        use_vectorization: bool = False,
        chunk_size: int = 10000,
        npartitions: Optional[int] = 2,
        parallel_processes: Optional[int] = 2,
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        encryption_mode: Optional[str] = None,
    ):
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
        include_timestamp : bool
            Whether to include timestamps in filenames
        profile_type : str
            Type of profiling for organizing artifacts
        is_birth_date : bool, optional
            Whether the field is a birth date field
        use_dask : bool
            Whether to use Dask for processing (default: False)
        use_cache : bool
            Whether to use operation caching (default: True)
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False)
        chunk_size : int
            Batch size for processing large datasets (default: 10000)
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1)
        npartitions : int, optional
            Number of partitions for Dask processing (default: None)
        visualization_theme : str, optional
            Theme for visualizations (default: None, uses PAMOLA default)
        visualization_backend : str, optional
            Backend for visualizations (default: None, uses PAMOLA default)
        visualization_strict : bool, optional
            Whether to enforce strict visualization rules (default: False)
        visualization_timeout : int, optional
            Timeout for visualization generation in seconds (default: 120)
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of date field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
        )

        self.min_year = min_year
        self.max_year = max_year
        self.id_column = id_column
        self.uid_column = uid_column
        self.profile_type = profile_type

        self.use_dask = use_dask
        self.use_cache = use_cache
        self.use_vectorization = use_vectorization
        self.chunk_size = chunk_size
        self.npartitions = npartitions
        self.parallel_processes = parallel_processes
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout

        # Set is_birth_date based on the provided value or field name
        if is_birth_date is None:
            self.is_birth_date = self.field_name.lower() in [
                "birth_day",
                "birthdate",
                "birth_date",
                "dob",
            ]
        else:
            self.is_birth_date = is_birth_date

        self.analyzer = DateAnalyzer()

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
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
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation:
            - generate_visualization: bool, whether to generate visualizations
            - force_recalculation: bool - Skip cache check
            - visualization_theme: str - Theme for visualizations
            - visualization_backend: str - Backend for visualizations ("plotly" or "matplotlib")
            - visualization_strict: bool - If True, raise exceptions for visualization config errors
            - visualization_timeout: int - Timeout for visualization generation in seconds (default: 120)

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        if kwargs.get("logger"):
            self.logger = kwargs["logger"]

        # Extract parameters from kwargs, defaulting to instance variables
        generate_visualization = kwargs.get("generate_visualization", True)

        force_recalculation = kwargs.get("force_recalculation", False)
        dataset_name = kwargs.get("dataset_name", "main")

        # Extract visualization parameters
        self.visualization_theme = kwargs.get(
            "visualization_theme", self.visualization_theme
        )
        self.visualization_backend = kwargs.get(
            "visualization_backend", self.visualization_backend
        )
        self.visualization_strict = kwargs.get(
            "visualization_strict", self.visualization_strict
        )
        self.visualization_timeout = kwargs.get(
            "visualization_timeout", self.visualization_timeout
        )

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        visualizations_dir = dirs["visualizations"]
        dictionaries_dir = dirs["dictionaries"]
        output_dir = dirs["output"]

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Generate single timestamp for all artifacts
        operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up progress tracking
        # Preparation, Data Loading, Cache Check, Analysis, Saving results, Visualizations, Finalization
        total_steps = (
            6
            + (1 if self.use_cache and not force_recalculation else 0)
            + (1 if generate_visualization else 0)
        )

        # Update progress if tracker provided
        # Step 0: Preparation
        if progress_tracker:
            progress_tracker.update(
                0, {"step": "Preparation", "field": self.field_name}
            )

        # Step 1: Data Loading
        if progress_tracker:
            progress_tracker.update(1, {"step": "Data Loading"})

            # Get and validate data
        try:
            # Load data
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)

            if df is None:
                error_message = "Failed to load input data"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR, error_message=error_message
                )
        except Exception as e:
            error_message = f"Error loading data: {str(e)}"
            self.logger.error(error_message)
            return OperationResult(
                status=OperationStatus.ERROR, error_message=error_message, exception=e
            )

        # Step 2: Check Cache (if enabled and not forced to recalculate)
        if self.use_cache and not force_recalculation:
            if progress_tracker:
                progress_tracker.update(2, {"step": "Checking Cache"})

            logger.info("Checking operation cache...")
            cache_result = self._check_cache(df, task_dir, reporter, **kwargs)

            if cache_result:
                self.logger.info("Cache hit! Using cached results.")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(
                        total_steps - 4, {"step": "Complete (cached)"}
                    )

                # Report cache hit to reporter
                if reporter:
                    reporter.add_operation(
                        f"Date field analysis for '{self.field_name}' (from cache)",
                        details={"cached": True},
                    )
                return cache_result

        try:
            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame",
                )

            # Add operation to reporter
            reporter.add_operation(
                f"Analyzing date field: {self.field_name}",
                details={
                    "field_name": self.field_name,
                    "min_year": self.min_year,
                    "max_year": self.max_year,
                    "id_column": self.id_column,
                    "uid_column": self.uid_column,
                    "is_birth_date": self.is_birth_date,
                    "operation_type": "date_analysis",
                },
            )

            # Step 3: Analysis
            if progress_tracker:
                progress_tracker.update(3, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = self.analyzer.analyze(
                df=df,
                field_name=self.field_name,
                min_year=self.min_year,
                max_year=self.max_year,
                id_column=self.id_column,
                uid_column=self.uid_column,
                is_birth_date=self.is_birth_date,
                chunk_size=self.chunk_size,
                use_dask=self.use_dask,
                use_vectorization=self.use_vectorization,
                parallel_processes=self.parallel_processes,
                npartitions=self.npartitions,
                progress_tracker=progress_tracker,
                task_logger=self.logger,
            )

            # Check for errors
            if "error" in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results["error"],
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    3, {"step": "Analysis complete", "field": self.field_name}
                )

            # Save analysis results to JSON in the task root directory
            stats_filename = f"{self.field_name}_stats_{operation_timestamp}.json"
            stats_path = output_dir / stats_filename

            encryption_mode = get_encryption_mode(analysis_results, **kwargs)
            write_json(
                analysis_results,
                stats_path,
                encryption_key=self.encryption_key,
                encryption_mode=encryption_mode,
                use_encryption=self.use_encryption,
            )
            result.add_artifact(
                "json",
                stats_path,
                f"{self.field_name} statistical analysis",
                category=Constants.Artifact_Category_Output,
            )

            # Add to reporter
            reporter.add_artifact(
                "json", str(stats_path), f"{self.field_name} statistical analysis"
            )

            # Step 4: Saving results
            # Update progress
            if progress_tracker:
                progress_tracker.update(4, {"step": "Saved analysis results"})

            # Generate visualizations if requested
            if generate_visualization:
                # Update progress
                # Step 5: Visualizations
                if progress_tracker:
                    progress_tracker.update(5, {"step": "Generating visualizations"})

                self._handle_visualizations(
                    analysis_results=analysis_results,
                    vis_dir=visualizations_dir,
                    is_birth_date=self.is_birth_date,
                    result=result,
                    reporter=reporter,
                    vis_theme=self.visualization_theme,
                    vis_backend=self.visualization_backend,
                    vis_strict=self.visualization_strict,
                    vis_timeout=self.visualization_timeout,
                    operation_timestamp=operation_timestamp,
                    progress_tracker=progress_tracker,
                    use_encryption=self.use_encryption,
                    encryption_key=self.encryption_key,
                )

                # Update progress
                if progress_tracker:
                    progress_tracker.update(5, {"step": "Visualizations complete"})

            # Save anomalies to CSV if any
            if (
                "anomalies" in analysis_results
                and sum(analysis_results["anomalies"].values()) > 0
            ):
                self._save_anomalies_to_csv(
                    analysis_results,
                    dictionaries_dir,
                    operation_timestamp,
                    result,
                    reporter,
                    encryption_key=self.encryption_key,
                )

                # Update progress
                if progress_tracker:
                    progress_tracker.update(5, {"step": "Saved anomalies data"})

            # Add metrics to the result
            result.add_metric("total_rows", analysis_results.get("total_records", 0))
            result.add_metric("null_count", analysis_results.get("null_count", 0))
            result.add_metric("valid_count", analysis_results.get("valid_count", 0))
            result.add_metric("invalid_count", analysis_results.get("invalid_count", 0))
            result.add_metric("fill_rate", analysis_results.get("fill_rate", 0))
            result.add_metric("valid_rate", analysis_results.get("valid_rate", 0))

            if "min_date" in analysis_results:
                result.add_metric("min_date", analysis_results.get("min_date"))

            if "max_date" in analysis_results:
                result.add_metric("max_date", analysis_results.get("max_date"))

            if "anomalies" in analysis_results:
                result.add_metric(
                    "anomalies_count", sum(analysis_results["anomalies"].values())
                )

            # Add age metrics if it's a birth date
            if self.is_birth_date and "age_statistics" in analysis_results:
                age_stats = analysis_results["age_statistics"]
                result.add_metric("min_age", age_stats.get("min_age"))
                result.add_metric("max_age", age_stats.get("max_age"))
                result.add_metric("mean_age", age_stats.get("mean_age"))
                result.add_metric("median_age", age_stats.get("median_age"))

            # Update progress to completion
            # Step 6: Finalization
            if progress_tracker:
                progress_tracker.update(
                    6, {"step": "Operation complete", "status": "success"}
                )

            # Add final operation status to reporter
            reporter.add_operation(
                f"Analysis of {self.field_name} completed",
                details={
                    "valid_dates": analysis_results.get("valid_count", 0),
                    "invalid_dates": analysis_results.get("invalid_count", 0),
                    "date_range": f"{analysis_results.get('min_date', 'N/A')} to {analysis_results.get('max_date', 'N/A')}",
                    "anomalies_found": sum(
                        analysis_results.get("anomalies", {}).values()
                    ),
                    "groups_with_changes": (
                        analysis_results.get("date_changes_within_group", {}).get(
                            "groups_with_changes", 0
                        )
                        if "date_changes_within_group" in analysis_results
                        else 0
                    ),
                },
            )

            if self.use_cache:
                try:
                    self._save_to_cache(
                        artifacts=result.artifacts,
                        original_df=df,
                        metrics=result.metrics,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            return result

        except Exception as e:
            self.logger.exception(f"Error in date operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(
                f"Error analyzing {self.field_name}",
                status="error",
                details={"error": str(e)},
            )

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing date field {self.field_name}: {str(e)}",
                exception=e,
            )

    def _generate_visualizations(
        self,
        analysis_results: Dict[str, Any],
        vis_dir: Path,
        is_birth_date: bool,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        timestamp: Optional[str] = None,
        **kwargs,
    ):
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
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        timestamp : Optional[str]
            Timestamp for naming visualization files
        **kwargs : Any
            Additional keyword arguments for visualization functions.

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate year distribution visualization if we have data
        if (
            "year_distribution" in analysis_results
            and analysis_results["year_distribution"]
        ):

            year_filename = f"{self.field_name}_year_distribution_{timestamp}.png"
            year_path = vis_dir / year_filename

            # Create visualization using the visualization module
            title = f"Year Distribution of {self.field_name}"
            if is_birth_date:
                title = "Birth Year Distribution"

            # Use the year_distribution data
            year_result = plot_date_distribution(
                {"year_distribution": analysis_results["year_distribution"]},
                str(year_path),
                title=title,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if not year_result.startswith("Error"):
                result.add_artifact(
                    "png",
                    year_path,
                    f"{self.field_name} year distribution",
                    category=Constants.Artifact_Category_Visualization,
                )
                reporter.add_artifact(
                    "png", str(year_path), f"{self.field_name} year distribution"
                )
        # Generate month distribution visualization if we have data
        if (
            "month_distribution" in analysis_results
            and analysis_results["month_distribution"]
        ):
            month_filename = f"{self.field_name}_month_distribution_{timestamp}.png"
            month_path = vis_dir / month_filename

            # Create visualization using the visualization module
            title = f"Month Distribution of {self.field_name}"
            if is_birth_date:
                title = "Birth Month Distribution"

            month_result = plot_value_distribution(
                analysis_results["month_distribution"],
                str(month_path),
                title=title,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if not month_result.startswith("Error"):
                result.add_artifact(
                    "png",
                    month_path,
                    f"{self.field_name} month distribution",
                    category=Constants.Artifact_Category_Visualization,
                )
                reporter.add_artifact(
                    "png", str(month_path), f"{self.field_name} month distribution"
                )

        # Generate day of week distribution visualization if we have data
        if (
            "day_of_week_distribution" in analysis_results
            and analysis_results["day_of_week_distribution"]
        ):
            dow_filename = f"{self.field_name}_dow_distribution_{timestamp}.png"
            dow_path = vis_dir / dow_filename

            # Create visualization using the visualization module
            title = f"Day of Week Distribution of {self.field_name}"
            if is_birth_date:
                title = "Birth Day of Week Distribution"

            dow_result = plot_value_distribution(
                analysis_results["day_of_week_distribution"],
                str(dow_path),
                title=title,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if not dow_result.startswith("Error"):
                result.add_artifact(
                    "png",
                    dow_path,
                    f"{self.field_name} day of week distribution",
                    category=Constants.Artifact_Category_Visualization,
                )
                reporter.add_artifact(
                    "png", str(dow_path), f"{self.field_name} day of week distribution"
                )

        # Generate age distribution visualization if it's a birth date and we have data
        if (
            is_birth_date
            and "age_distribution" in analysis_results
            and analysis_results["age_distribution"]
        ):
            age_filename = f"{self.field_name}_age_distribution_{timestamp}.png"
            age_path = vis_dir / age_filename

            # Create visualization using the visualization module
            title = "Age Distribution"

            age_result = plot_value_distribution(
                analysis_results["age_distribution"],
                str(age_path),
                title=title,
                x_label="Age Group",
                y_label="Count",
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if not age_result.startswith("Error"):
                result.add_artifact(
                    "png",
                    age_path,
                    "Age distribution",
                    category=Constants.Artifact_Category_Visualization,
                )
                reporter.add_artifact("png", str(age_path), "Age distribution")

    def _save_anomalies_to_csv(
        self,
        analysis_results: Dict[str, Any],
        dict_dir: Path,
        timestamp: str,
        result: OperationResult,
        reporter: Any,
        encryption_key: Optional[str] = None,
    ):
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

            for anomaly_type, count in analysis_results["anomalies"].items():
                if count > 0 and f"{anomaly_type}_examples" in analysis_results:
                    for example in analysis_results[f"{anomaly_type}_examples"]:
                        anomaly_row = {
                            "index": (
                                example[0]
                                if isinstance(example, tuple) and len(example) > 0
                                else None
                            ),
                            "value": (
                                example[1]
                                if isinstance(example, tuple) and len(example) > 1
                                else example
                            ),
                            "anomaly_type": anomaly_type,
                        }

                        if (
                            anomaly_type in ["too_old", "future_dates", "too_young"]
                            and isinstance(example, tuple)
                            and len(example) > 2
                        ):
                            anomaly_row["year"] = example[2]

                        anomalies_data.append(anomaly_row)

            if anomalies_data:
                # Create anomalies CSV filename
                anomalies_filename = f"{self.field_name}_anomalies_{timestamp}.csv"
                anomalies_path = dict_dir / anomalies_filename

                # Create DataFrame and save to CSV
                import pandas as pd

                anomalies_df = pd.DataFrame(anomalies_data)
                write_dataframe_to_csv(
                    df=anomalies_df,
                    file_path=anomalies_path,
                    index=False,
                    encoding="utf-8",
                    encryption_key=encryption_key,
                )
                # Add artifact to result and reporter
                result.add_artifact(
                    "csv",
                    anomalies_path,
                    f"{self.field_name} anomalies",
                    category=Constants.Artifact_Category_Dictionary,
                )
                reporter.add_artifact(
                    "csv", str(anomalies_path), f"{self.field_name} anomalies"
                )

        except Exception as e:
            self.logger.warning(f"Error saving anomalies for {self.field_name}: {e}")
            reporter.add_operation(
                f"Saving anomalies for {self.field_name}",
                status="warning",
                details={"warning": str(e)},
            )

    def _check_cache(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        task_dir: Path,
        reporter: Any,
        **kwargs,
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame for the operation
        task_dir : Path
            Task directory
        reporter : Any
            The reporter to log artifacts to

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import OperationCache

            operation_cache_dir = OperationCache(cache_dir=task_dir / "cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache_dir.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        cached_result.add_metric(key, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        artifact_type = artifact.get("artifact_type", "")
                        artifact_path = artifact.get("path", "")
                        artifact_name = artifact.get("description", "")
                        artifact_category = artifact.get("category", "output")
                        cached_result.add_artifact(
                            artifact_type,
                            artifact_path,
                            artifact_name,
                            artifact_category,
                        )

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
        self,
        original_df: Union[pd.DataFrame, dd.DataFrame],
        artifacts: List[OperationArtifact],
        metrics: Dict[str, Any],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        processed_df : pd.DataFrame
            Processed DataFrame
        metrics : dict
            The metrics of operation
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache or (not artifacts and not metrics):
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            # Generate operation cache
            operation_cache_dir = OperationCache(cache_dir=task_dir / "cache")

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            artifacts_for_cache = [artifact.to_dict() for artifact in artifacts]

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "artifacts": artifacts_for_cache,
                "metrics": metrics,
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache_dir.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(self, df: Union[pd.DataFrame, dd.DataFrame]) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame for the operation

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
            data_hash=data_hash,
        )

    def _get_operation_parameters(self) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "min_year": self.min_year,
            "max_year": self.max_year,
            "id_column": self.id_column,
            "uid_column": self.uid_column,
            "profile_type": self.profile_type,
            "is_birth_date": self.is_birth_date,
            "use_encryption": self.use_encryption,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "chunk_size": self.chunk_size,
            "visualization_theme": self.visualization_theme,
            "visualization_backend": self.visualization_backend,
            "visualization_strict": self.visualization_strict,
            "visualization_timeout": self.visualization_timeout,
            "use_cache": self.use_cache,
            "use_encryption": self.use_encryption,
            "encryption_mode": self.encryption_mode,
            "encryption_key": self.encryption_key,
        }

        return parameters

    def _generate_data_hash(self, df: Union[pd.DataFrame, dd.DataFrame]) -> str:
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
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _handle_visualizations(
        self,
        analysis_results: Dict[str, Any],
        vis_dir: Path,
        is_birth_date: bool,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> Dict[str, Path]:
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
            Whether to include timestamps in output filenames
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
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        logger.info(
            f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s"
        )

        try:
            import threading
            import contextvars

            visualization_paths = {}
            visualization_error = None

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                logger.info(
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
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
                            logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations
                    self._generate_visualizations(
                        analysis_results,
                        vis_dir,
                        is_birth_date,
                        result,
                        reporter,
                        vis_theme,
                        vis_backend,
                        vis_strict,
                        operation_timestamp,
                        **kwargs,
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
                    logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
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

            logger.info(
                f"[DIAG] Starting visualization thread with timeout={vis_timeout}s"
            )
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = {}
            elif visualization_error:
                logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                logger.info(
                    f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}"
                )
        except Exception as e:
            logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = {}

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization",
                )

        return visualization_paths


def analyze_date_fields(
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    date_fields: List[str] = None,
    id_column: str = "resume_id",
    uid_column: str = "UID",
    **kwargs,
) -> Dict[str, OperationResult]:
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
    dataset_name = kwargs.get("dataset_name", "main")
    settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
    df = load_data_operation(data_source, dataset_name, **settings_operation)
    if df is None:
        reporter.add_operation(
            "Date fields analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"},
        )
        return {}

    # If no date fields specified, try to find them
    if date_fields is None:
        date_fields = []
        for col in df.columns:
            if any(
                keyword in col.lower() for keyword in ["birth", "day", "date", "time"]
            ):
                date_fields.append(col)

    # Check if id_column and uid_column exist
    actual_id_column = id_column if id_column in df.columns else None
    actual_uid_column = uid_column if uid_column in df.columns else None

    # Report on fields to be analyzed
    reporter.add_operation(
        "Date fields analysis",
        details={
            "fields_count": len(date_fields),
            "fields": date_fields,
            "id_column": actual_id_column,
            "uid_column": actual_uid_column,
            "parameters": {
                k: v
                for k, v in kwargs.items()
                if isinstance(v, (str, int, float, bool))
            },
        },
    )

    # Track progress if enabled
    track_progress = kwargs.get("track_progress", True)
    overall_tracker = None

    if track_progress and date_fields:
        overall_tracker = HierarchicalProgressTracker(
            total=len(date_fields),
            description=f"Analyzing {len(date_fields)} date fields",
            unit="fields",
            track_memory=True,
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(date_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(
                        0, {"field": field, "progress": f"{i + 1}/{len(date_fields)}"}
                    )

                logger.info(f"Analyzing date field: {field}")

                # Check if field is a birth date
                is_birth_date = any(
                    keyword in field.lower() for keyword in ["birth", "dob"]
                )

                # Create and execute operation
                operation = DateOperation(
                    field,
                    id_column=actual_id_column,
                    uid_column=actual_uid_column,
                )
                result = operation.execute(
                    data_source,
                    task_dir,
                    reporter,
                    is_birth_date=is_birth_date,
                    **kwargs,
                )

                # Store result
                results[field] = result

                # Update overall tracker after successful analysis
                if overall_tracker:
                    if result.status == OperationStatus.SUCCESS:
                        overall_tracker.update(
                            1, {"field": field, "status": "completed"}
                        )
                    else:
                        overall_tracker.update(
                            1,
                            {
                                "field": field,
                                "status": "error",
                                "error": result.error_message,
                            },
                        )

            except Exception as e:
                logger.error(f"Error analyzing date field {field}: {e}", exc_info=True)

                reporter.add_operation(
                    f"Analyzing {field} field",
                    status="error",
                    details={"error": str(e)},
                )

                # Update overall tracker in case of error
                if overall_tracker:
                    overall_tracker.update(1, {"field": field, "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(
        1 for r in results.values() if r.status == OperationStatus.SUCCESS
    )
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    reporter.add_operation(
        "Date fields analysis completed",
        details={
            "fields_analyzed": len(results),
            "successful": success_count,
            "failed": error_count,
        },
    )

    return results
