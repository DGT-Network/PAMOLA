"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Utility Metric Operation
Package:       pamola_core.metrics
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
    This module provides the base class for all utility metric operations,
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

import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pamola_core.metrics.base_metrics_op import MetricsOperation
from pamola_core.metrics.schemas.utility_ops_config import UtilityMetricConfig
from pamola_core.metrics.utility.classification import ClassificationUtility
from pamola_core.metrics.utility.regression import RegressionUtility
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils import logging
from pamola_core.metrics.commons.validation import validate_dataframe

# Configure module logger
logger = logging.get_logger(__name__)

# Factory mapping for utility metrics
UTILITY_METRIC_FACTORY = {
    "classification": ClassificationUtility,
    "regression": RegressionUtility,
}


@register(version="1.0.0")
class UtilityMetricOperation(MetricsOperation):
    """
    Base class for all utility operation support.

    This class provides common functionality for all utility operations,
    including data source handling, result processing, metric calculation,
    and automatic switching to Dask for large dataset processing.
    """

    def __init__(
        self,
        name: str = "utility_metrics",
        utility_metrics: Optional[List[str]] = None,
        metric_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the UtilityMetricOperation.

        Parameters
        ----------
        name : str, optional
            Name of the operation (default: "utility_metrics")
        utility_metrics : list[str], optional
            List of utility metric identifiers to compute.
        metric_params : dict, optional
            Metric-specific parameters.
        **kwargs : dict
            Additional arguments forwarded to MetricsOperation / BaseOperation, such as:
            columns, normalize, confidence_level, sample_size, use_dask, npartitions,
            description, visualization options, encryption, etc.
        """

        # defaults for metadata if not provided
        kwargs.setdefault("name", name)
        kwargs.setdefault("description", "Utility metrics operation")

        # Build config and validate via OperationConfig (if implemented)
        config = UtilityMetricConfig(
            utility_metrics=utility_metrics or [],
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
                status=OperationStatus.ERROR, error_message=error_message, exception=e
            )

    def calculate_metrics(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate multiple utility metrics between original and transformed DataFrames.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original dataset.
        transformed_df : pd.DataFrame
            The transformed dataset.
        progress_tracker : Optional[HierarchicalProgressTracker]
            Tracker for reporting progress.
        **kwargs
            - utility_metrics: List[str]: List of metric types to compute (e.g., ["classification", "regression", etc.])
            - metric_params: Dict[str, Dict]: Mapping of metric type to its specific configuration params.

        Returns:
        --------
        Dict[str, Any]
            - result: Dict of metric_name → metric_result
        """
        utility_metrics: List[str] = kwargs.get("utility_metrics", [])
        metric_params: Dict[str, Dict] = kwargs.get("metric_params", {})

        # VALIDATE DATA Before calculation
        validate_dataframe(original_df)
        validate_dataframe(transformed_df)

        if not utility_metrics:
            raise ValueError(
                "No utility metrics specified. 'utility_metrics' list is empty."
            )

        results: Dict[str, Any] = {}

        if progress_tracker:
            progress_tracker.update(
                n=1,
                postfix={
                    "step": "Calculate Utility Metrics Setup",
                    "original_df_length": len(original_df),
                    "transformed_df_length": len(transformed_df),
                    "metrics": utility_metrics,
                },
            )

        self.logger.info(
            f"Calculating utility metrics: {utility_metrics} with params: {metric_params}"
        )

        for metric_type in utility_metrics:
            params = metric_params.get(metric_type, {})

            try:
                if progress_tracker:
                    progress_tracker.update(
                        n=2,
                        postfix={
                            "step": f"Calculating {metric_type.upper()} metric",
                            "params": params,
                        },
                    )
                self.logger.info(
                    f"Calculating {metric_type.upper()} metric with params: {params}"
                )

                metric_class = UTILITY_METRIC_FACTORY.get(metric_type)
                if not metric_class:
                    raise ValueError(f"Unsupported utility metric: {metric_type}")

                init_params = {
                    k: v
                    for k, v in params.items()
                    if k
                    in (
                        inspect.signature(metric_class.__init__).parameters.keys()
                        - {"self"}
                    )
                }
                calculate_params = {
                    k: v
                    for k, v in params.items()
                    if k
                    in (
                        inspect.signature(
                            metric_class.calculate_metric
                        ).parameters.keys()
                        - {"self"}
                    )
                }

                # Instantiate the metric class with provided parameters
                metric = metric_class(**init_params)

                # Calculate the metric
                metric_result = metric.calculate_metric(
                    original_df, transformed_df, **calculate_params
                )

                if progress_tracker:
                    progress_tracker.update(
                        n=3,
                        postfix={
                            "step": f"Calculated {metric_type.upper()} metric",
                            "params": params,
                        },
                    )
                self.logger.info(
                    f"Calculated {metric_type.upper()} metric: {metric_result}"
                )
                results[metric_type] = metric_result

            except Exception as e:
                self.logger.error(f"[{metric_type.upper()}] Metric failed: {e}")
                raise ValueError(
                    f"Failed to calculate {metric_type} metric: {str(e)}"
                ) from e

        return results

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
            Collected metrics for visualization (may include "classification", "regression", etc.)
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
        from pamola_core.utils.visualization import create_bar_plot, create_line_plot

        visualization_paths = {}
        viz_dir = task_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not vis_backend:
            self.logger.info("Visualization backend is None, skipping visualization.")
            return visualization_paths

        try:
            # Step 1: Prepare utility metric visualizations
            if progress_tracker:
                progress_tracker.update(
                    n=1, postfix={"step": "Preparing utility metric visualizations"}
                )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(n=2, postfix={"step": "Creating visualization"})

            viz_metrics = {
                key: value
                for key, value in metrics.items()
                if key in UTILITY_METRIC_FACTORY.keys()
            }
            for metric_type, metric_dict in viz_metrics.items():
                for metric_model, metric_values in metric_dict.items():
                    if metric_values:
                        viz_tradeoff = metric_values.pop(
                            "precision_recall_tradeoff", {}
                        )
                        viz_data = metric_values

                        viz_path = (
                            viz_dir / f"{metric_type}_{metric_model}_{timestamp}.png"
                        )
                        viz_result = create_bar_plot(
                            data=viz_data,
                            output_path=viz_path,
                            title=f"{metric_type.title()} {metric_model.title()} Metrics",
                            x_label="Metric",
                            y_label="Value",
                            sort_by="key",
                            backend=vis_backend,
                            theme=vis_theme,
                            strict=vis_strict,
                            **kwargs,
                        )

                        if viz_result.startswith("Error"):
                            self.logger.error(
                                f"Failed to create visualization: {viz_result}"
                            )
                        else:
                            visualization_paths[f"{metric_type}_{metric_model}"] = (
                                viz_path
                            )

                        if viz_tradeoff:
                            recall = viz_tradeoff.get("recall", {})
                            precision = viz_tradeoff.get("precision", {})

                            if len(recall) > 1:
                                max_length = max(
                                    max([len(lst) for lst in recall.values()]),
                                    max([len(lst) for lst in precision.values()]),
                                )
                                for lst in recall.values():
                                    lst += [0.0] * (max_length - len(lst))
                                for lst in precision.values():
                                    lst += [0.0] * (max_length - len(lst))

                                tradeoff_recall = list(recall.values())
                                tradeoff_precision = {
                                    f"Class {key}": value
                                    for key, value in precision.items()
                                }
                                multi_x_data = True
                            else:
                                tradeoff_recall = list(recall.values())[0]
                                tradeoff_precision = {
                                    f"Class {key}": value
                                    for key, value in precision.items()
                                }
                                multi_x_data = False

                            viz_path = (
                                viz_dir
                                / f"{metric_type}_{metric_model}_tradeoff_{timestamp}.png"
                            )
                            viz_result = create_line_plot(
                                data=tradeoff_precision,
                                output_path=viz_path,
                                title=f"Precision-Recall Curve",
                                x_data=tradeoff_recall,
                                x_label="Recall",
                                y_label="Precision",
                                add_markers=False,
                                backend=vis_backend,
                                theme=vis_theme,
                                strict=vis_strict,
                                multi_x_data=multi_x_data,
                                line_average=False,
                                **kwargs,
                            )

                            if viz_result.startswith("Error"):
                                self.logger.error(
                                    f"Failed to create visualization: {viz_result}"
                                )
                            else:
                                visualization_paths[
                                    f"{metric_type}_{metric_model}_tradeoff"
                                ] = viz_path

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(
                    n=3, postfix={"step": "Utility metric visualizations complete"}
                )

        except Exception as e:
            self.logger.error(f"Failed to generate utility metric visualizations: {e}")

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
            utility_metrics=self.utility_metrics,
            metric_params=self.metric_params,
            columns=self.columns,
            column_mapping=self.column_mapping,
            normalize=self.normalize,
            confidence_level=self.confidence_level,
            optimize_memory=self.optimize_memory,
            sample_size=self.sample_size,
            use_dask=self.use_dask,
            npartitions=self.npartitions,
            dask_partition_size=self.dask_partition_size,
            use_cache=self.use_cache,
            visualization_backend=self.visualization_backend,
            visualization_theme=self.visualization_theme,
            visualization_strict=self.visualization_strict,
            visualization_timeout=self.visualization_timeout,
            use_encryption=self.use_encryption,
            encryption_mode=self.encryption_mode,
            encryption_key=self.encryption_key,
            force_recalculation=self.force_recalculation,
            generate_visualization=self.generate_visualization,
        )

        return params
