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

from datetime import datetime
import logging
from math import log
from pathlib import Path
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from click import Option
import pandas as pd
from pamola_core.profiling.commons.correlation_utils import (
    analyze_correlation,
    analyze_correlation_matrix,
    estimate_resources,
)
from pamola_core.utils.io import (
    write_json,
    ensure_directory,
    get_timestamped_filename,
    load_data_operation,
    load_settings_operation,
)
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.progress import ProgressTracker, HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation, BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
    OperationArtifact,
)
from pamola_core.utils.visualization import (
    create_scatter_plot,
    create_boxplot,
    create_heatmap,
    create_correlation_matrix,
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode


class CorrelationAnalyzer:
    """
    Analyzer for correlations between fields.

    This analyzer provides methods for analyzing correlations between fields,
    supporting different correlation methods based on field types, and producing
    visualizations of the relationships.
    """

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        field1: str,
        field2: str,
        method: Optional[str] = None,
        mvf_parser: Optional[str] = None,
        null_handling: str = "drop",
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze the correlation between two fields in a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the data.
        field1 : str
            Name of the first field to analyze.
        field2 : str
            Name of the second field to analyze.
        method : str, optional
            Correlation method to use (e.g., 'pearson', 'cramers_v', etc.). If None, selects automatically based on field types.
        mvf_parser : str, optional
            String lambda to parse multi-valued fields (MVF) if applicable.
        null_handling : str, optional
            Strategy for handling null values ('drop', 'fill', 'pairwise'). Default is 'drop'.
        **kwargs : dict
            Additional keyword arguments for correlation analysis.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the results of the correlation analysis, including method, coefficient, p-value, and plot data.
        """
        return analyze_correlation(
            df=df,
            field1=field1,
            field2=field2,
            method=method,
            mvf_parser=mvf_parser,
            null_handling=null_handling,
            task_logger=logger,
            **kwargs,
        )

    @staticmethod
    def analyze_matrix(df: pd.DataFrame, fields: List[str], **kwargs) -> Dict[str, Any]:
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
        return analyze_correlation_matrix(df=df, fields=fields, **kwargs)

    @staticmethod
    def estimate_resources(
        df: pd.DataFrame, field1: str, field2: str
    ) -> Dict[str, Any]:
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

    def __init__(
        self,
        field1: str,
        field2: str,
        method: Optional[str] = None,
        null_handling: Optional[str] = "drop",
        mvf_parser: Optional[str] = None,
        description: str = "",
        include_timestamp: bool = True,
        generate_visualization: bool = True,
        use_cache: bool = True,
        visualization_backend: Optional[str] = "plotly",
        visualization_theme: Optional[str] = None,
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        encryption_mode: Optional[str] = None,
    ):
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
        null_handling : str
            Method for handling nulls ('drop', 'fill', 'pairwise')
        mvf_parser : str, optional
            String lambda to parse multi-valued fields (MVF) if applicable.
            Example:
                mvf_parser = "lambda x: x.split(';') if isinstance(x, str) else x"
        description : str
            Description of the operation (optional)
        generate_visualization : bool
            Whether to generate visualizations (default: True)
        include_timestamp : bool
            Whether to include timestamps in filenames (default: True)
        use_cache : bool
            Whether to use caching for results (default: True)
        visualization_backend : str, optional
            Backend for visualizations (default: "plotly")
        visualization_theme : str, optional
            Theme for visualizations (default: None)
        visualization_strict : bool
            Whether to enforce strict mode for visualizations (default: False)
        visualization_timeout : int
            Timeout for visualizations in seconds (default: 120)
        use_encryption : bool
            Whether to use encryption for sensitive data (default: False)
        encryption_key : str or Path, optional
            Key for encryption if use_encryption is True (default: None)
        encryption_mode : str, optional
            Mode for encryption (default: None, uses default mode from utils)
        """
        # Use field1 as the primary field for the parent class
        super().__init__(
            field_name=field1,
            description=description
            or f"Correlation analysis between '{field1}' and '{field2}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
        )
        self.field1 = field1
        self.field2 = field2
        self.method = method
        self.null_handling = null_handling
        self.mvf_parser = mvf_parser

        self.include_timestamp = include_timestamp
        self.generate_visualization = generate_visualization

        self.use_cache = use_cache

        self.visualization_backend = visualization_backend
        self.visualization_theme = visualization_theme
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout

        self.operation_cache = None

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
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
            Optional overrides for instance attributes and execution parameters:
            - generate_visualization: bool, whether to generate visualizations
            - dataset_name: str - Name of dataset - main
            - force_recalculation: bool - Force operation even if cached results exist - False
            - visualization_theme: str - Override theme for visualizations - None
            - visualization_backend: str - Override backend for visualizations - None
            - visualization_strict: bool - Override strict mode for visualizations - False
            - visualization_timeout: int - Override timeout for visualizations - 120

        Returns:
        --------
        OperationResult
            Results of the operation
        """

        caller_operation = self.__class__.__name__
        self.logger = kwargs.get("logger", self.logger)

        generate_visualization = kwargs.get("generate_visualization", self.generate_visualization)

        self.visualization_backend = kwargs.get("visualization_backend", self.visualization_backend)
        self.visualization_theme = kwargs.get("visualization_theme", self.visualization_theme)
        self.visualization_strict = kwargs.get("visualization_strict", self.visualization_strict)
        self.visualization_timeout = kwargs.get("visualization_timeout", self.visualization_timeout)

        save_output = kwargs.get("save_output", True)
        force_recalculation = kwargs.get("force_recalculation", True)

        operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.operation_cache = OperationCache(cache_dir=task_dir / "cache")

        try:
            # Start operation
            self.logger.info(f"Operation: {caller_operation}, Start operation")
            if progress_tracker:
                progress_tracker.total = self._compute_total_steps(**kwargs)
                progress_tracker.update(
                    1,
                    {
                        "step": "Start operation - Preparation",
                        "operation": caller_operation,
                    },
                )

            # Set up directories
            dirs = self._prepare_directories(task_dir)
            visualizations_dir = dirs["visualizations"]
            output_dir = dirs["output"]

            if reporter:
                reporter.add_operation(
                    f"Operation {caller_operation}",
                    status="info",
                    details={
                        "step": "Preparation",
                        "message": "Preparation successfully",
                        "directories": {k: str(v) for k, v in dirs.items()},
                    },
                )

            # Load data and validate input parameters
            self.logger.info(
                f"Operation: {caller_operation}, Load data and validate input parameters"
            )
            if progress_tracker:
                progress_tracker.update(
                    1,
                    {
                        "step": "Load data and validate input parameters",
                        "operation": caller_operation,
                    },
                )

            df, is_valid = self._load_data_and_validate_input_parameters(
                data_source, **kwargs
            )

            if is_valid:
                if reporter:
                    reporter.add_operation(
                        f"Operation {caller_operation}",
                        status="info",
                        details={
                            "step": "Load data and validate input parameters",
                            "message": "Load data and validate input parameters successfully",
                            "shape": df.shape,
                        },
                    )
            else:
                if reporter:
                    reporter.add_operation(
                        f"Operation {caller_operation}",
                        status="info",
                        details={
                            "step": "Load data and validate input parameters",
                            "message": "Load data and validate input parameters failed",
                        },
                    )
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message="Load data and validate input parameters failed",
                    )

            # Handle cache if required
            if self.use_cache and not force_recalculation:
                self.logger.info(
                    f"Operation: {caller_operation}, Load result from cache"
                )
                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Load result from cache",
                            "operation": caller_operation,
                        },
                    )

                try:
                    cached_result = self._check_cache(df, reporter)
                except Exception as e:
                    error_message = f"Check cache error: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message=error_message,
                        exception=e,
                    )

                if cached_result is not None and isinstance(
                    cached_result, OperationResult
                ):
                    if reporter:
                        reporter.add_operation(
                            f"Operation {caller_operation}",
                            status="info",
                            details={
                                "step": "Load result from cache",
                                "message": "Load result from cache successfully",
                            },
                        )
                    return cached_result
                else:
                    self.logger.info(
                        f"Operation: {caller_operation}, Load result from cache failed — proceeding with execution."
                    )
                    if reporter:
                        reporter.add_operation(
                            f"Operation {caller_operation}",
                            status="info",
                            details={
                                "step": "Load result from cache",
                                "message": "Load result from cache failed - proceeding with execution",
                            },
                        )

            # Analyzing correlation
            self.logger.info(f"Operation: {caller_operation}, Analyzing correlation")
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Analyzing correlation", "operation": caller_operation}
                )

            analysis_results = CorrelationAnalyzer.analyze(
                df=df,
                field1=self.field1,
                field2=self.field2,
                method=self.method,
                mvf_parser=self.mvf_parser,
                null_handling=self.null_handling,
                logger=self.logger,
            )

            # Check analysis results
            if "error" in analysis_results and analysis_results["error"] is not None:
                if reporter:
                    reporter.add_operation(
                        f"Operation {caller_operation}",
                        status="info",
                        details={
                            "step": "Analyzing correlation",
                            "message": "Analyzing correlation failed",
                            "field1": self.field1,
                            "field2": self.field2,
                            "method": self.method or "auto",
                            "null_handling": self.null_handling,
                            "operation_type": "correlation_analysis",
                        },
                    )
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results["error"],
                )
            else:
                if reporter:
                    reporter.add_operation(
                        f"Operation: {caller_operation}",
                        status="info",
                        details={
                            "step": "Analyzing correlation",
                            "message": "Analyzing correlation successfully",
                            "field1": self.field1,
                            "field2": self.field2,
                            "method": self.method or "auto",
                            "null_handling": self.null_handling,
                            "operation_type": "correlation_analysis",
                        },
                    )

            # Collect metric
            self.logger.info(f"Operation: {caller_operation}, Collect metric")
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Collect metric", "operation": caller_operation}
                )

            result = OperationResult(status=OperationStatus.SUCCESS)

            self._collect_metrics(analysis_results, result)

            if reporter:
                reporter.add_operation(
                    f"Operation {caller_operation}",
                    status="info",
                    details={
                        "step": "Collect metric",
                        "message": "Collect metric successfully",
                        "method": analysis_results.get("method", "unknown"),
                        "correlation_coefficient": analysis_results.get(
                            "correlation_coefficient", 0
                        ),
                        "sample_size": analysis_results.get("sample_size", 0),
                        "p_value": analysis_results.get("p_value"),
                        "statistically_significant": (
                            analysis_results.get("p_value") is not None
                            and analysis_results["p_value"] < 0.05
                        ),
                    },
                )

            # Save output if required
            if save_output:
                self.logger.info(f"Operation: {caller_operation}, Save output")
                if progress_tracker:
                    progress_tracker.update(
                        1, {"step": "Save output", "operation": caller_operation}
                    )

                self._save_output(
                    analysis_results=analysis_results,
                    output_dir=output_dir,
                    result=result,
                    **kwargs,
                )

                if reporter:
                    reporter.add_operation(
                        f"Operation: {caller_operation}",
                        status="info",
                        details={
                            "step": "Save output",
                            "message": "Save output successfully",
                        },
                    )

            # Generate visualization if required
            if generate_visualization:
                self.logger.info(
                    f"Operation: {caller_operation}, Generate visualizations"
                )
                if progress_tracker:
                    progress_tracker.update(
                        1,
                        {
                            "step": "Generate visualizations",
                            "operation": caller_operation,
                        },
                    )

                try:
                     self._handle_visualizations(
                        analysis_results=analysis_results,
                        vis_dir=visualizations_dir,
                        result=result,
                        reporter=reporter,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        operation_timestamp=operation_timestamp,
                        progress_tracker=progress_tracker,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.error(error_message)

            # Save cache if required
            if self.use_cache:
                self.logger.info(f"Operation: {caller_operation}, Save cache")
                if progress_tracker:
                    progress_tracker.update(
                        1, {"step": "Save cache", "operation": caller_operation}
                    )

                try:
                    self._save_to_cache(
                        original_df=df,
                        artifacts=result.artifacts,
                        analysis_results=analysis_results,
                        metrics=result.metrics,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    error_message = f"Failed to cache results: {str(e)}"
                    self.logger.error(error_message)
                    # Continue execution - cache failure is not critical

                if reporter:
                    reporter.add_operation(
                        f"Operation {caller_operation}",
                        status="info",
                        details={
                            "step": "Save cache",
                            "message": "Save cache successfully",
                        },
                    )

            # Operation completed successfully
            self.logger.info(f"Operation: {caller_operation}, Completed successfully.")
            if reporter:
                reporter.add_operation(
                    f"Operation {caller_operation}",
                    status="info",
                    details={
                        "step": "Return result",
                        "message": "Operation completed successfully",
                    },
                )

            return result

        except Exception as e:
            self.logger.error(f"Operation: {caller_operation}, error occurred: {e}")

            if reporter:
                reporter.add_operation(
                    f"Operation {caller_operation}",
                    status="error",
                    details={
                        "step": "Exception",
                        "message": "Operation failed due to an exception",
                        "error": str(e),
                    },
                )

            return OperationResult(
                status=OperationStatus.ERROR, error_message=str(e), exception=e
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
        output_dir = task_dir / "output"
        visualizations_dir = task_dir / "visualizations"

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)

        return {"output": output_dir, "visualizations": visualizations_dir}

    def _collect_metrics(self, analysis_results: dict, result: OperationResult) -> None:
        """
        Collect and add analysis metrics to the result object.

        Parameters
        ----------
        analysis_results : dict
            Dictionary containing results from categorical analysis.
        result : OperationResult
            The result object where metrics will be added.
        """

        result.add_metric(
            "correlation_method", analysis_results.get("method", "unknown")
        )
        result.add_metric(
            "correlation_coefficient",
            analysis_results.get("correlation_coefficient", 0),
        )
        result.add_metric("sample_size", analysis_results.get("sample_size", 0))
        if "p_value" in analysis_results and analysis_results["p_value"] is not None:
            result.add_metric("p_value", analysis_results["p_value"])
            result.add_metric(
                "statistically_significant", analysis_results["p_value"] < 0.05
            )

        null_stats = analysis_results.get("null_stats", {})
        result.add_metric("total_rows", null_stats.get("total_rows", 0))
        result.add_metric("null_rows", null_stats.get("null_rows", 0))
        result.add_metric("null_percentage", null_stats.get("null_percentage", 0))
        result.add_metric("interpretation", analysis_results.get("interpretation", "unknown"))

    def _save_output(
        self,
        analysis_results: dict,
        output_dir: Path,
        result: OperationResult,
        **kwargs,
    ):
        """
        Save analysis results to JSON, dictionary to CSV, and anomalies (if any).
        """

        # Save analysis results to JSON
        correlation_name = f"{self.field1}_{self.field2}_correlation"
        stats_filename = get_timestamped_filename(
            correlation_name, "json", self.include_timestamp
        )
        stats_path = output_dir / stats_filename

        encryption_mode_analysis = get_encryption_mode(analysis_results, **kwargs)
        write_json(
            analysis_results,
            stats_path,
            encryption_key=self.encryption_key,
            encryption_mode=encryption_mode_analysis,
        )
        result.add_artifact(
            "json",
            stats_path,
            f"Correlation analysis between {self.field1} and {self.field2}",
            category=Constants.Artifact_Category_Output,
        )

    def _generate_visualizations(
        self,
        analysis_results: dict,
        visualizations_dir: Path,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate and save a visualization from top categorical values.

        Parameters
        ----------
        analysis_results : dict
            Dictionary containing the results of categorical analysis.
        visualizations_dir : Path
            Directory to save the visualization image.
        result : OperationResult
            Object to store generated artifacts.
        **kwargs : dict
            Visualization configuration options.

        Returns
        -------
        str
            The result string from the visualization function (can indicate success or error).
        """
        if "plot_data" not in analysis_results:
            warning_msg = f"Operation: {self.__class__.__name__}, No 'plot_data' found in analysis results for visualization."
            self.logger.warning(warning_msg)
            return []

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.logger.info(
            f"[VIZ] Starting visualization generation for {self.field_name}"
        )
        self.logger.debug(
            f"[VIZ] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
        )

        kwargs_visualization = {
            "use_encryption": self.use_encryption,
            "encryption_key": self.encryption_key,
        }

        # Create visualization based on plot_data type
        plot_data = analysis_results["plot_data"]
        plot_type = plot_data.get("type", "unknown")
        viz_filename = f"{self.field1}_{self.field2}_correlation_{timestamp}_plot.png"
        viz_path = visualizations_dir / viz_filename

        # Method details for plot title
        method_name = analysis_results.get("method", "Unknown")
        method_display = method_name.replace("_", " ").title()
        correlation_value = analysis_results.get("correlation_coefficient", 0)

        # Create appropriate visualization based on plot type
        viz_result = ""
        visualization_paths = []

        # Step 2: Create visualization
        if progress_tracker:
            progress_tracker.update(2, {"step": "Creating visualization"})

        if plot_type == "scatter":
            # For numeric-numeric correlations: scatter plot
            title = f"Correlation between {self.field1} and {self.field2}"
            viz_result = create_scatter_plot(
                x_data=plot_data["x_values"],
                y_data=plot_data["y_values"],
                output_path=str(viz_path),
                title=title,
                x_label=plot_data["x_label"],
                y_label=plot_data["y_label"],
                add_trendline=True,
                correlation=correlation_value,
                method=method_display,
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs_visualization,
            )
            if not viz_result.startswith("Error"):
                visualization_paths.append(
                    {
                        "artifact_type": "png",
                        "path": str(viz_path),
                        "description": title,
                        "category": Constants.Artifact_Category_Visualization,
                    }
                )
            else:
                self.logger.warning(
                    f"Failed to generate visualization image for '{title}': {viz_result}"
                )

        elif plot_type == "boxplot":
            # For categorical-numeric correlations: boxplot
            title = f"Relationship between {plot_data['x_label']} and {plot_data['y_label']}"
            viz_result = create_boxplot(
                data={
                    cat: values if isinstance(values, (list, tuple)) else [values]
                    for cat, values in zip(plot_data["categories"], plot_data["values"])
                    if cat is not None
                },
                output_path=str(viz_path),
                title=title,
                x_label=plot_data["x_label"],
                y_label=plot_data["y_label"],
                vis_backend=vis_backend,
                vis_theme=vis_theme,
                strict=vis_strict,
                **kwargs_visualization,
            )

            if not viz_result.startswith("Error"):
                visualization_paths.append(
                    {
                        "artifact_type": "png",
                        "path": str(viz_path),
                        "description": title,
                        "category": Constants.Artifact_Category_Visualization,
                    }
                )
            else:
                self.logger.warning(
                    f"Failed to generate visualization image for '{title}': {viz_result}"
                )

        elif plot_type == "heatmap":
            # For categorical-categorical correlations: heatmap
            title = (
                f"Association between {plot_data['y_label']} and {plot_data['x_label']}"
            )
            viz_result = create_heatmap(
                data=plot_data["matrix"],
                output_path=str(viz_path),
                title=title,
                x_label=plot_data["x_label"],
                y_label=plot_data["y_label"],
                annotate=True,
                vis_backend=vis_backend,
                vis_theme=vis_theme,
                strict=vis_strict,
                **kwargs_visualization,
            )
            if not viz_result.startswith("Error"):
                visualization_paths.append(
                    {
                        "artifact_type": "png",
                        "path": str(viz_path),
                        "description": title,
                        "category": Constants.Artifact_Category_Visualization,
                    }
                )
            else:
                self.logger.warning(
                    f"Failed to generate visualization image for '{title}': {viz_result}"
                )

        return visualization_paths

    def _handle_visualizations(
        self,
        analysis_results: Dict[str, Any],
        vis_dir: Path,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        operation_timestamp: Optional[str] = None,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
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
        operation_timestamp : Optional[str]
            Timestamp for naming visualization files
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        self.logger.info(
            f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s"
        )

        try:
            import threading
            import contextvars

            visualization_paths = []
            visualization_error = None
            if operation_timestamp is None:
                operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                self.logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                self.logger.info(
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    self.logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        self.logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
                    except Exception as ctx_e:
                        self.logger.warning(
                            f"[DIAG] Could not inspect context: {ctx_e}"
                        )

                    # Generate visualizations with visualization context parameters
                    self.logger.info(f"[DIAG] Calling _generate_visualizations...")
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
                            self.logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations
                    visualization_paths = self._generate_visualizations(
                        analysis_results,
                        vis_dir,
                        progress_tracker,
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
                    self.logger.info(
                        f"[DIAG] Visualization completed in {elapsed:.2f}s, generated {len(visualization_paths)} files"
                    )
                except Exception as e:
                    elapsed = time.time() - start_time
                    visualization_error = e
                    self.logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
                    self.logger.error(f"[DIAG] Stack trace:", exc_info=True)

            # Copy context for the thread
            self.logger.info(f"[DIAG] Preparing to launch visualization thread...")
            ctx = contextvars.copy_context()

            # Create thread with context
            viz_thread = threading.Thread(
                target=ctx.run,
                args=(generate_viz_with_diagnostics,),
                name=f"VizThread-",
                daemon=False,  # Changed from True to ensure proper cleanup
            )

            self.logger.info(
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
                    self.logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                self.logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                self.logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = []
            elif visualization_error:
                self.logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = []
            else:
                total_time = time.time() - thread_start_time
                self.logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                self.logger.info(
                    f"[DIAG] Generated visualizations: {[viz['path'] for viz in visualization_paths]}"
                )
        except Exception as e:
            self.logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            self.logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = []

        # Register visualization artifacts
        for viz in visualization_paths:
            artifact_type = viz["artifact_type"]
            path = viz["path"]
            description = viz["description"]

            # Add to result
            result.add_artifact(
                artifact_type=artifact_type,
                path=path,
                description=description,
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type=artifact_type, path=path, description=description
                )

        return visualization_paths

    def _save_to_cache(
        self,
        original_df: pd.DataFrame,
        artifacts: List[OperationArtifact],
        analysis_results: Dict[str, Any],
        metrics: Dict[str, Any],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        artifacts : List[OperationArtifact]
            List of artifacts to save
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
                "analysis_results": analysis_results
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = self.operation_cache.save_cache(
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

    def _check_cache(
        self,
        df: pd.DataFrame,
        reporter: Any,
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame for the operation
        task_dir : Path
            Task directory

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = self.operation_cache.get_cache(
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

                analysis_results = cached_data.get("analysis_results", {})

                if reporter:
                    reporter.add_operation(
                        f"Operation {self.__class__.__name__}",
                        status="info",
                        details={
                            "step": "Collect metric",
                            "message": "Collect metric successfully",
                            "method": analysis_results.get("method", "unknown"),
                            "correlation_coefficient": analysis_results.get(
                                "correlation_coefficient", 0
                            ),
                            "sample_size": analysis_results.get("sample_size", 0),
                            "p_value": analysis_results.get("p_value"),
                            "statistically_significant": (
                                analysis_results.get("p_value") is not None
                                and analysis_results["p_value"] < 0.05
                            ),
                        },
                    )

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

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : pd.DataFrame
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

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
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
        import json
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

    def _get_operation_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Get operation-specific parameters required for generating a cache key.

        These parameters define the behavior of the transformation and are used
        to determine cache uniqueness.

        Returns
        -------
        Dict[str, Any]
            Dictionary of relevant parameters to identify the operation configuration.
        """

        return {
            "operation": self.__class__.__name__,
            "version": self.version,
            "field1": self.field1,
            "field2": self.field2,
            "method": self.method,
            "description": self.description,
            "null_handling": self.null_handling,
            "mvf_parser": self.mvf_parser,
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

    def _validate_input_parameters(self, df: pd.DataFrame) -> bool:
        """
        Validate that all specified fields in field_groups exist in the DataFrame.
        Optionally check if the ID field exists.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset to validate.

        Returns:
        --------
        bool
            True if all fields are valid; False otherwise.
        """
        if self.field1 not in df.columns:
            self.logger.error(f"Column {self.field1} not existing in data frame")
            return False

        if self.field2 not in df.columns:
            self.logger.error(f"Column {self.field2} not existing in data frame")
            return False

        # All validations passed
        return True

    def _load_data_and_validate_input_parameters(
        self, data_source: DataSource, **kwargs
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        dataset_name = kwargs.get("dataset_name", "main")
        settings_operation = load_settings_operation(
            data_source, dataset_name, **kwargs
        )
        df = load_data_operation(data_source, dataset_name, **settings_operation)

        if df is None or df.empty:
            self.logger.error("Error data frame is None or empty")
            return None, False

        self._input_dataset = dataset_name
        self._original_df = df.copy(deep=True)

        return df, self._validate_input_parameters(df)

    def _compute_total_steps(self, **kwargs) -> int:
        force_recalculation = kwargs.get(
            "force_recalculation", self.force_recalculation
        )
        save_output = kwargs.get("save_output", self.save_output)
        generate_visualization = kwargs.get(
            "generate_visualization", self.generate_visualization
        )

        steps = 0

        steps += 1  # Step 1: Preparation
        steps += 1  # Step 2: Load data and validate input

        if self.use_cache and not force_recalculation:
            steps += 1  # Step 3: Try to load from cache

        steps += 1  # Step 4: Process data
        steps += 1  # Step 5: Collect metrics

        if save_output:
            steps += 1  # Step 6: Save output

        if generate_visualization:
            steps += 1  # Step 7: Generate visualizations

        if self.use_cache:
            steps += 1  # Step 8: Save cache

        return steps


@register(override=True)
class CorrelationMatrixOperation(BaseOperation):
    """
    Operation for creating a correlation matrix for multiple fields.

    This operation analyzes correlations between all pairs of fields in a list
    and generates a correlation matrix visualization.
    """

    def __init__(
        self,
        fields: List[str],
        methods: Optional[Dict[str, str]] = None,
        description: str = "",
        generate_visualization: bool = True,
        include_timestamp: bool = True,
        min_threshold: float = 0.3,
        null_handling: str = "drop",
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
    ):
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
        generate_visualization : bool
            Whether to generate visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        min_threshold : float
            Minimum correlation threshold for significant correlations
        null_handling : str
            Method for handling nulls ('drop', 'fill', 'pairwise')
        """
        super().__init__(
            description or f"Correlation matrix analysis for {len(fields)} fields",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
        )

        self.fields = fields
        self.methods = methods
        self.generate_visualization = generate_visualization
        self.include_timestamp = include_timestamp
        self.min_threshold = min_threshold
        self.null_handling = null_handling

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[ProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
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
            - generate_visualization: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames
            - null_handling: str, method for handling nulls ('drop', 'fill', 'pairwise')
            - min_threshold: float, minimum correlation threshold for significant correlations

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        generate_visualization = kwargs.get(
            "generate_visualization", self.generate_visualization
        )
        include_timestamp = kwargs.get("include_timestamp", self.include_timestamp)
        min_threshold = kwargs.get("min_threshold", self.min_threshold)
        null_handling = kwargs.get("null_handling", self.null_handling)
        encryption_key = kwargs.get("encryption_key", None)

        # Set up directories
        output_dir = task_dir / "output"
        visualizations_dir = task_dir / "visualizations"
        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(
                1, {"step": "Preparation", "fields_count": len(self.fields)}
            )

        try:
            # Get DataFrame from data source
            dataset_name = kwargs.get("dataset_name", "main")
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source",
                )

            # Check if fields exist
            missing_fields = [field for field in self.fields if field not in df.columns]
            if missing_fields:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Fields not found: {', '.join(missing_fields)}",
                )

            # Add operation to reporter
            if reporter:
                reporter.add_operation(
                    f"Creating correlation matrix for {len(self.fields)} fields",
                    details={
                        "fields": self.fields,
                        "null_handling": null_handling,
                        "min_threshold": min_threshold,
                        "operation_type": "correlation_matrix",
                    },
                )

            # Adjust progress tracker total steps if provided
            total_steps = 3  # Preparation, analysis, saving results
            if generate_visualization:
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
                min_threshold=min_threshold,
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
                    1, {"step": "Analysis complete", "fields_count": len(self.fields)}
                )

            # Save analysis results to JSON
            stats_filename = get_timestamped_filename(
                "correlation_matrix", "json", include_timestamp
            )
            stats_path = output_dir / stats_filename

            write_json(analysis_results, stats_path, encryption_key=encryption_key)
            result.add_artifact(
                "json",
                stats_path,
                "Correlation matrix analysis",
                category=Constants.Artifact_Category_Output,
            )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualization if requested
            if generate_visualization and "correlation_matrix" in analysis_results:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualization"})

                # Convert dictionary to DataFrame for visualization
                matrix_dict = analysis_results["correlation_matrix"]
                matrix_df = pd.DataFrame(matrix_dict)

                # Create visualization
                viz_filename = get_timestamped_filename(
                    "correlation_matrix_heatmap", "png", include_timestamp
                )
                viz_path = visualizations_dir / viz_filename

                viz_result = create_correlation_matrix(
                    data=matrix_df,
                    output_path=str(viz_path),
                    title="Correlation Matrix",
                    annotate=True,
                    annotation_format=".2f",
                    mask_diagonal=False,
                    mask_upper=False,
                    **kwargs,
                )

                # Add visualization to result if successful
                if viz_result and not viz_result.startswith("Error"):
                    result.add_artifact(
                        "png",
                        viz_path,
                        "Correlation matrix visualization",
                        category=Constants.Artifact_Category_Visualization,
                    )
                    if reporter:
                        reporter.add_artifact(
                            "png", str(viz_path), "Correlation matrix visualization"
                        )
                else:
                    self.logger.warning(f"Error creating visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualization"})

            # Add metrics to the result
            result.add_metric("fields_analyzed", len(self.fields))
            result.add_metric(
                "significant_correlations",
                len(analysis_results.get("significant_correlations", [])),
            )
            result.add_metric("min_threshold", min_threshold)

            # Add final operation status to reporter
            significant_count = len(
                analysis_results.get("significant_correlations", [])
            )
            if reporter:
                reporter.add_operation(
                    f"Correlation matrix analysis completed",
                    details={
                        "fields_analyzed": len(self.fields),
                        "significant_correlations": significant_count,
                        "min_threshold": min_threshold,
                    },
                )

            return result

        except Exception as e:
            self.logger.exception(f"Error in correlation matrix operation: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            if reporter:
                reporter.add_operation(
                    f"Error creating correlation matrix",
                    status="error",
                    details={"error": str(e)},
                )

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error creating correlation matrix: {str(e)}",
                exception=e,
            )


def analyze_correlations(
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    pairs: List[Tuple[str, str]],
    **kwargs,
) -> Dict[str, OperationResult]:
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
        - generate_visualization: bool, whether to generate visualization (default: True)
        - include_timestamp: bool, whether to include timestamps (default: True)

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping pair names to their operation results
    """
    # Get DataFrame from data source to check fields
    dataset_name = kwargs.get("dataset_name", "main")
    df = load_data_operation(data_source, dataset_name)
    if df is None:
        if reporter:
            reporter.add_operation(
                "Correlation analysis",
                status="error",
                details={"error": "No valid DataFrame found in data source"},
            )
        return {}

    # Extract parameters from kwargs
    methods = kwargs.get("methods", {})
    null_handling = kwargs.get("null_handling", "drop")
    generate_visualization = kwargs.get("generate_visualization", True)

    # Report on field pairs to be analyzed
    if reporter:
        reporter.add_operation(
            "Correlation analysis",
            details={
                "pairs_count": len(pairs),
                "pairs": [f"{field1}_{field2}" for field1, field2 in pairs],
                "null_handling": null_handling,
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

    if track_progress and pairs:
        overall_tracker = ProgressTracker(
            total=len(pairs),
            description=f"Analyzing {len(pairs)} field correlations",
            unit="pairs",
            track_memory=True,
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
            if reporter:
                reporter.add_operation(
                    f"Correlation Analysis: {field1} vs {field2}",
                    status="error",
                    details={"error": error_msg},
                )

            # Create an error result
            error_result = OperationResult(
                status=OperationStatus.ERROR, error_message=error_msg
            )
            results[f"{field1}_{field2}"] = error_result

            # Update overall tracker if present
            if overall_tracker:
                overall_tracker.update(
                    1, {"pair": f"{field1}_{field2}", "status": "error"}
                )

            continue

        try:
            # Update overall progress tracker
            if overall_tracker:
                overall_tracker.update(
                    0,
                    {"pair": f"{field1}_{field2}", "progress": f"{i + 1}/{len(pairs)}"},
                )

            print(f"Analyzing correlation between {field1} and {field2}")

            # Get method if specified
            method = methods.get(f"{field1}_{field2}")

            # Create and execute operation
            operation = CorrelationOperation(
                field1=field1, field2=field2, method=method
            )
            result = operation.execute(
                data_source,
                task_dir,
                reporter,
                null_handling=null_handling,
                generate_visualization=generate_visualization,
                **kwargs,
            )

            # Store result
            results[f"{field1}_{field2}"] = result

            # Update overall tracker after successful analysis
            if overall_tracker:
                if result.status == OperationStatus.SUCCESS:
                    overall_tracker.update(
                        1, {"pair": f"{field1}_{field2}", "status": "completed"}
                    )
                else:
                    overall_tracker.update(
                        1,
                        {
                            "pair": f"{field1}_{field2}",
                            "status": "error",
                            "error": result.error_message,
                        },
                    )

        except Exception as e:
            print(f"Error analyzing correlation between {field1} and {field2}: {e}")

            if reporter:
                reporter.add_operation(
                    f"Analyzing correlation between {field1} and {field2}",
                    status="error",
                    details={"error": str(e)},
                )

            # Create an error result
            error_result = OperationResult(
                status=OperationStatus.ERROR, error_message=str(e)
            )
            results[f"{field1}_{field2}"] = error_result

            # Update overall tracker in case of error
            if overall_tracker:
                overall_tracker.update(
                    1, {"pair": f"{field1}_{field2}", "status": "error"}
                )

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(
        1 for r in results.values() if r.status == OperationStatus.SUCCESS
    )
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    if reporter:
        reporter.add_operation(
            "Correlation analysis completed",
            details={
                "pairs_analyzed": len(results),
                "successful": success_count,
                "failed": error_count,
            },
        )

    return results
