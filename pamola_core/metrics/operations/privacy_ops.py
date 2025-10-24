"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Privacy Metric Operation
Package:       pamola_core.metrics
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
    This module provides the base class for all privacy metric operations,
    defining common functionality, interface, and behavior with enhanced
    support for conditional processing, profiling integration, memory-efficient
    operations, and Dask-based distributed processing for large datasets.

Key Features:
    - Standardized operation lifecycle with validation, execution, and result handling
    - Memory-efficient processing for large datasets
    - Comprehensive metrics collection and visualization generation
    - Robust caching mechanism for operation results
    - Progress tracking and reporting throughout the operation
    - Secure output generation with optional encryption
    - Conditional processing based on field values and risk scores
    - Integration with k-anonymity profiling results
    - Enhanced memory management with automatic optimization
    - Dask integration for distributed processing of large datasets

Framework:
    Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
    for input/output, progress tracking, and result reporting.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType
from pamola_core.metrics.base_metrics_op import MetricsOperation
from pamola_core.metrics.commons.safe_instantiate import safe_instantiate
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord
from pamola_core.metrics.privacy.identity import Uniqueness
from pamola_core.metrics.privacy.neighbor import NearestNeighborDistanceRatio
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import create_bar_plot, create_histogram

PRIVACY_METRIC_FACTORY = {
    PrivacyMetricsType.DCR.value: DistanceToClosestRecord,
    PrivacyMetricsType.NNDR.value: NearestNeighborDistanceRatio,
    PrivacyMetricsType.UNIQUENESS.value: Uniqueness,
}


class PrivacyMetricConfig(OperationConfig):
    """Configuration for PrivacyMetricOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common fields
            {
                "type": "object",
                "properties": {
                    "privacy_metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                PrivacyMetricsType.DCR.value,
                                PrivacyMetricsType.NNDR.value,
                                PrivacyMetricsType.UNIQUENESS.value,
                                PrivacyMetricsType.K_ANONYMITY.value,
                                PrivacyMetricsType.L_DIVERSITY.value,
                            ],
                        },
                        "default": [PrivacyMetricsType.DCR.value],
                    },
                    "metric_params": {"type": ["object", "null"]},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "column_mapping": {"type": ["object", "null"]},
                    "sample_size": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Size of dataset sample used for metric calculation.",
                    },
                },
                "required": ["privacy_metrics"],
            },
        ],
    }


@register(version="1.0.0")
class PrivacyMetricOperation(MetricsOperation):
    """
    Base class for all privacy operation support.

    This class provides common functionality for all privacy operations,
    including data source handling, result processing, metric calculation,
    and automatic switching to Dask for large dataset processing.
    """

    def __init__(
        self,
        name: str = "privacy_metrics",
        privacy_metrics: Optional[List[str]] = None,
        metric_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the PrivacyMetricOperation.

        Parameters
        ----------
        name : str, optional
            Name of the operation (default: "privacy_metrics")
        privacy_metrics : list[str], optional
            List of privacy metric identifiers to compute. Defaults to [DCR].
        metric_params : dict, optional
            Metric-specific parameters.
        **kwargs : dict
            Additional arguments forwarded to MetricsOperation / BaseOperation, such as:
            columns, sample_size, use_dask, npartitions, description, visualization options, etc.
        """

        # Fill sensible defaults for metadata
        kwargs.setdefault("name", name)
        kwargs.setdefault("description", "Privacy metrics operation")

        # Build/validate config
        config = PrivacyMetricConfig(
            privacy_metrics=privacy_metrics or [PrivacyMetricsType.DCR.value],
            metric_params=metric_params or {},
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize parent MetricsOperation
        super().__init__(**kwargs)

        # Save config attributes to self for easy access
        for k, v in config.to_dict().items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

        # Operation metadata
        self.operation_name = self.__class__.__name__

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute the metrics operation with enhanced features including Dask support.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters including profiling_results

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            # Call parent execute method
            result = super().execute(
                data_source, task_dir, reporter, progress_tracker, **kwargs
            )
            return result

        except Exception as e:
            error_message = f"Error in transformation operation: {str(e)}"
            self.logger.exception(error_message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_message,
                exception=e,
            )

    def calculate_metrics(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Calculate multiple privacy metrics between original and transformed DataFrames.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original dataset.

        transformed_df : pd.DataFrame
            The transformed/anonymized dataset.

        progress_tracker : Optional[HierarchicalProgressTracker]
            Tracker for reporting progress.

        **kwargs : Any
            - privacy_metrics: List[str]
                List of metric types to compute (e.g., ["dcr", "nndr", etc.])
            - metric_params: Dict[str, Dict]
                Mapping of metric type to its specific configuration params.

        Returns:
        --------
        Dict[str, Any]
            - result: Dict of metric_name → metric_result
        """
        privacy_metrics: List[str] = kwargs.get("privacy_metrics", [])
        metric_params: Dict[str, Dict] = kwargs.get("metric_params", {})

        if not privacy_metrics:
            raise ValueError(
                "No privacy metrics specified. 'privacy_metrics' list is empty."
            )

        results: Dict[str, Any] = {}

        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Calculate Privacy Metrics Setup",
                    "original_df_length": len(original_df),
                    "transformed_df_length": len(transformed_df),
                    "metrics": privacy_metrics,
                },
            )

        self.logger.info(
            f"Calculating privacy metrics: {privacy_metrics} with params: {metric_params}"
        )

        for metric_type in privacy_metrics:
            params = metric_params.get(metric_type, {})

            try:
                if progress_tracker:
                    progress_tracker.update(
                        2,
                        {
                            "step": f"Calculating {metric_type.upper()} metric",
                            "params": params,
                        },
                    )
                self.logger.info(
                    f"Calculating {metric_type.upper()} metric with params: {params}"
                )

                metric_class = PRIVACY_METRIC_FACTORY.get(metric_type)
                if not metric_class:
                    raise ValueError(f"Unsupported privacy metric: {metric_type}")

                # Instantiate the metric class with provided parameters
                metric = safe_instantiate(metric_class, params)

                # Calculate the metric
                if metric_type == PrivacyMetricsType.UNIQUENESS.value:
                    # Special handling for Uniqueness metric
                    result = metric.calculate_metric(transformed_df)
                else:
                    # For other metrics, filter numeric columns if necessary
                    result = metric.calculate_metric(original_df, transformed_df)

                if progress_tracker:
                    progress_tracker.update(
                        3,
                        {
                            "step": f"Calculated {metric_type.upper()} metric",
                            "params": params,
                        },
                    )
                self.logger.info(f"Calculated {metric_type.upper()} metric: {result}")
                results[metric_type] = result

            except Exception as e:
                self.logger.error(f"[{metric_type.upper()}] Metric failed: {e}")
                raise ValueError(
                    f"Failed to calculate {metric_type} metric: {str(e)}"
                ) from e

        return results

    def _generate_dcr_visualizations(
        self,
        metrics: Dict[str, Any],
        viz_dir: Path,
        vis_backend: Optional[str],
        vis_theme: Optional[str],
        vis_strict: bool,
        timestamp: str,
        **kwargs: Any,
    ) -> Dict[str, Path]:
        """
        Generate DCR metric visualizations and return their paths.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Dictionary containing DCR metric results.
        viz_dir : Path
            Directory to save visualization files.
        vis_backend : Optional[str]
            Visualization backend (e.g., 'plotly', 'matplotlib').
        vis_theme : Optional[str]
            Visualization theme.
        vis_strict : bool
            If True, raise exceptions for visualization config errors.
        timestamp : str
            Timestamp string for file naming.
        **kwargs : Any
            Additional keyword arguments for visualization.

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and their file paths.
        """
        visualization_paths = {}
        dcr_plot_path = viz_dir / f"{self.name}_dcr_visualization_{timestamp}.png"

        # DCR statistics
        dcr_statistics_dict = metrics["dcr"].get("dcr_statistics")
        if dcr_statistics_dict is not None:
            dcr_plot_result = create_bar_plot(
                dcr_statistics_dict,
                output_path=dcr_plot_path,
                title="DCR Statistics",
                x_label="Metric",
                y_label="Value",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not dcr_plot_result.startswith("Error"):
                visualization_paths["dcr_metric_bar"] = dcr_plot_result
            else:
                self.logger.error(
                    f"Failed to create DCR statistics visualization: {dcr_plot_result}"
                )

        # DCR risk assessment
        dcr_risk_assessment = metrics["dcr"].get("risk_assessment")
        if dcr_risk_assessment is not None:
            risk_plot_path = (
                viz_dir / f"{self.name}_dcr_risk_assessment_{timestamp}.png"
            )
            risk_plot_result = create_bar_plot(
                dcr_risk_assessment,
                output_path=risk_plot_path,
                title="DCR Risk Assessment",
                x_label="Metric",
                y_label="Value",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not risk_plot_result.startswith("Error"):
                visualization_paths["dcr_risk_assessment_bar"] = risk_plot_result
            else:
                self.logger.error(
                    f"Failed to create DCR risk assessment visualization: {risk_plot_result}"
                )

        return visualization_paths

    def _generate_nndr_visualizations(
        self,
        metrics: Dict[str, Any],
        viz_dir: Path,
        vis_backend: Optional[str],
        vis_theme: Optional[str],
        vis_strict: bool,
        timestamp: str,
        **kwargs: Any,
    ) -> Dict[str, Path]:
        """
        Generate visualizations for NNDR (Nearest Neighbor Distance Ratio) metrics.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Dictionary containing NNDR metric results.
        viz_dir : Path
            Directory to save visualization files.
        vis_backend : Optional[str]
            Visualization backend (e.g., 'plotly', 'matplotlib').
        vis_theme : Optional[str]
            Visualization theme.
        vis_strict : bool
            If True, raise exceptions for visualization config errors.
        timestamp : str
            Timestamp string for file naming.
        **kwargs : Any
            Additional keyword arguments for visualization.

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and their file paths.
        """
        visualization_paths = {}
        nndr_metrics = metrics.get("nndr", {})

        # NNDR Statistics Bar Plot
        nndr_statistics_data = nndr_metrics.get("nndr_statistics")
        if nndr_statistics_data is not None:
            plot_path = viz_dir / f"{self.name}_nndr_statistics_{timestamp}.png"
            result = create_bar_plot(
                nndr_statistics_data,
                output_path=plot_path,
                title="NNDR Statistics",
                x_label="Metric",
                y_label="Value",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not result.startswith("Error"):
                visualization_paths["nndr_statistics_bar"] = result
            else:
                self.logger.error(
                    f"Failed to create NNDR Statistics visualization: {result}"
                )

        # NNDR Values Histogram
        nndr_values_data = nndr_metrics.get("nndr_values")
        if nndr_values_data is not None:
            plot_path = viz_dir / f"{self.name}_nndr_values_{timestamp}.png"
            result = create_histogram(
                nndr_values_data,
                output_path=plot_path,
                title="NNDR Values",
                x_label="Metric",
                y_label="Value",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not result.startswith("Error"):
                visualization_paths["nndr_values_bar"] = result
            else:
                self.logger.error(
                    f"Failed to create NNDR Values visualization: {result}"
                )

        return visualization_paths

    def _generate_uniqueness_visualizations(
        self,
        metrics: Dict[str, Any],
        viz_dir: Path,
        vis_backend: Optional[str],
        vis_theme: Optional[str],
        vis_strict: bool,
        timestamp: str,
        **kwargs: Any,
    ) -> Dict[str, Path]:
        """
        Generate visualizations for Uniqueness metrics (k-anonymity, l-diversity, t-closeness).

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Dictionary containing uniqueness metric results.
        viz_dir : Path
            Directory to save visualization files.
        vis_backend : Optional[str]
            Visualization backend (e.g., 'plotly', 'matplotlib').
        vis_theme : Optional[str]
            Visualization theme.
        vis_strict : bool
            If True, raise exceptions for visualization config errors.
        timestamp : str
            Timestamp string for file naming.
        **kwargs : Any
            Additional keyword arguments for visualization.

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and their file paths.
        """
        # Initialize output dictionary for visualization paths
        visualization_paths = {}
        # Extract uniqueness metrics from input
        uniqueness_metrics = metrics.get("uniqueness", {})

        # --- K-Anonymity Visualization ---
        k_anonymity_metrics = uniqueness_metrics.get("k_anonymity")
        if k_anonymity_metrics:
            plot_path = viz_dir / f"{self.name}_uniqueness_k_anonymity_{timestamp}.png"
            # Prepare data for bar plot: k value vs percent violation
            k_anonymity_data = {
                f'k={item["k_value"]}': item["percent_violation"]
                for item in k_anonymity_metrics.get("k_anonymity_stats")
            }
            result = create_bar_plot(
                data=k_anonymity_data,
                output_path=plot_path,
                title="K-Anonymity Percent Violations",
                x_label="k Value",
                y_label="Percent Violation (%)",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not result.startswith("Error"):
                visualization_paths["uniqueness_metric_bar"] = result
            else:
                self.logger.error(
                    f"Failed to create uniqueness_metric_bar visualization: {result}"
                )

        # --- L-Diversity Visualization ---
        l_diversity_metrics = uniqueness_metrics.get("l_diversity", {})
        if l_diversity_metrics:
            # Prepare data for bar plot: min, max, avg l-diversity
            l_diversity_data = {
                "min_l_diversity": l_diversity_metrics.get("min_l_diversity"),
                "max_l_diversity": l_diversity_metrics.get("max_l_diversity"),
                "avg_l_diversity": l_diversity_metrics.get("avg_l_diversity"),
            }
            plot_path = viz_dir / f"{self.name}_uniqueness_l_diversity_{timestamp}.png"
            result = create_bar_plot(
                data=l_diversity_data,
                output_path=plot_path,
                title="L-Diversity Metrics",
                x_label="Metric",
                y_label="Value",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not result.startswith("Error"):
                visualization_paths["uniqueness_l_diversity_bar"] = result
            else:
                self.logger.error(
                    f"Failed to create l_diversity visualization: {result}"
                )

        # --- T-Closeness Visualization ---
        t_closeness_metrics = uniqueness_metrics.get("t_closeness", {})
        if t_closeness_metrics:
            # Prepare data for bar plot: only numeric metrics
            t_closeness_data = {
                k: v
                for k, v in t_closeness_metrics.items()
                if isinstance(v, (int, float))
            }
            plot_path = viz_dir / f"{self.name}_uniqueness_t_closeness_{timestamp}.png"
            result = create_bar_plot(
                data=t_closeness_data,
                output_path=plot_path,
                title="T-Closeness Metrics",
                x_label="Metric",
                y_label="Value",
                backend=vis_backend,
                theme=vis_theme,
                strict=vis_strict,
                **kwargs,
            )
            if not result.startswith("Error"):
                visualization_paths["uniqueness_t_closeness_bar"] = result
            else:
                self.logger.error(
                    f"Failed to create t_closeness visualization: {result}"
                )

        return visualization_paths

    def _generate_visualizations(
        self,
        metrics: Dict[str, Any],
        task_dir: Path,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate visualizations using the core visualization utilities with thread-safe context support.

        This implementation generates separate visualizations for each metric type if present.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Collected metrics for visualization (may include 'dcr', 'nndr', etc.)
        task_dir : Path
            Task directory for saving visualizations
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        visualization_paths = {}
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not vis_backend:
            self.logger.info("Visualization backend is None, skipping visualization.")
            return visualization_paths

        try:
            # Step 1: Prepare privacy metric visualizations
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Preparing privacy metric visualizations"}
                )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            if metrics.get("dcr") is not None:
                dcr_paths = self._generate_dcr_visualizations(
                    metrics,
                    viz_dir,
                    vis_backend,
                    vis_theme,
                    vis_strict,
                    timestamp,
                    **kwargs,
                )
                visualization_paths.update(dcr_paths)

            if metrics.get("nndr") is not None:
                nndr_paths = self._generate_nndr_visualizations(
                    metrics,
                    viz_dir,
                    vis_backend,
                    vis_theme,
                    vis_strict,
                    timestamp,
                    **kwargs,
                )
                visualization_paths.update(nndr_paths)
            if metrics.get("uniqueness") is not None:
                uniqueness_paths = self._generate_uniqueness_visualizations(
                    metrics,
                    viz_dir,
                    vis_backend,
                    vis_theme,
                    vis_strict,
                    timestamp,
                    **kwargs,
                )
                visualization_paths.update(uniqueness_paths)

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(
                    3, {"step": "Privacy metric visualizations complete"}
                )

        except Exception as e:
            self.logger.error(f"Failed to generate privacy metric visualizations: {e}")

        return visualization_paths

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        This method should be overridden by subclasses to provide
        operation-specific parameters for caching.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        params = dict(
            privacy_metrics=self.privacy_metrics,
            metric_params=self.metric_params,
            columns=self.columns,
            column_mapping=self.column_mapping,
            sample_size=self.sample_size,
            optimize_memory=self.optimize_memory,
            use_dask=self.use_dask,
            npartitions=self.npartitions,
            dask_partition_size=self.dask_partition_size,
            use_cache=self.use_cache,
            use_encryption=self.use_encryption,
            encryption_mode=self.encryption_mode,
            encryption_key=self.encryption_key,
            visualization_theme=self.visualization_theme,
            visualization_backend=self.visualization_backend,
            visualization_strict=self.visualization_strict,
            visualization_timeout=self.visualization_timeout,
            force_recalculation=self.force_recalculation,
            generate_visualization=self.generate_visualization,
        )

        return params
