"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fidelity Metric Operation
Package:       pamola_core.metrics
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
    This module provides the base class for all fidelity metric operations,
    defining common functionality, interface, and behavior with enhanced
    support for conditional processing, profiling integration, memory-efficient
    operations, and Dask-based distributed processing for large datasets.

Key Features:
    - Standardized operation lifecycle with validation, execution, and result handling
    - Support for both in-place (REPLACE) and new field creation (ENRICH) modes
    - Configurable null value handling strategies (PRESERVE, EXCLUDE, ERROR)
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
from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType
from pamola_core.metrics.base_metrics_op import MetricsOperation
from pamola_core.metrics.commons.safe_instantiate import safe_instantiate
from pamola_core.metrics.fidelity.distribution.kl_divergence import KLDivergence
from pamola_core.metrics.fidelity.distribution.ks_test import KolmogorovSmirnovTest
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import create_bar_plot

# Factory mapping for fidelity metrics
FIDELITY_METRIC_FACTORY = {
    FidelityMetricsType.KL.value: KLDivergence,
    FidelityMetricsType.KS.value: KolmogorovSmirnovTest,
}


class FidelityConfig(OperationConfig):
    """Configuration for FidelityOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common fields
            {
                "type": "object",
                "properties": {
                    "fidelity_metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                FidelityMetricsType.KS.value,
                                FidelityMetricsType.KL.value,
                            ],
                        },
                        "default": [
                            FidelityMetricsType.KS.value,
                            FidelityMetricsType.KL.value,
                        ],
                    },
                    "metric_params": {"type": ["object", "null"]},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "column_mapping": {"type": ["object", "null"]},
                    "normalize": {"type": "boolean", "default": True},
                    "confidence_level": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.95,
                    },
                    "sample_size": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Size of dataset sample used for metric calculation.",
                    },
                },
                "required": ["fidelity_metrics"],
            },
        ],
    }


@register(version="1.0.0")
class FidelityOperation(MetricsOperation):
    """
    Base class for all fidelity operation support.

    This class provides common functionality for all fidelity operations,
    including data source handling, result processing, metric calculation,
    and automatic switching to Dask for large dataset processing.
    """

    def __init__(
        self,
        name: str = "fidelity_metrics",
        fidelity_metrics: Optional[List[str]] = None,
        metric_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the Fidelity Operation.

        Parameters
        ----------
        name : str, optional
            Name of the operation (default: "fidelity_metrics")
        fidelity_metrics : List[str], optional
            List of fidelity metrics to calculate.
            Defaults to `[FidelityMetricsType.KS.value, FidelityMetricsType.KL.value]`.
        metric_params : Dict[str, Any], optional
            Additional parameters to customize metric computation.
        **kwargs : dict
            Additional arguments passed to :class:`MetricsOperation` or
            :class:`BaseOperation` (e.g., columns, normalize, confidence_level,
            description, use_dask, encryption_key, visualization options, etc.).
        """

        # Ensure default metadata
        kwargs.setdefault("name", name)
        kwargs.setdefault("description", "Fidelity metrics operation")

        # Build config object using FidelityConfig
        config = FidelityConfig(
            fidelity_metrics=fidelity_metrics
            or [FidelityMetricsType.KS.value, FidelityMetricsType.KL.value],
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
        Calculate multiple fidelity metrics between original and transformed DataFrames.

        Parameters:
        -----------
        original_df : pd.DataFrame
            The original dataset.

        transformed_df : pd.DataFrame
            The transformed/anonymized dataset.

        progress_tracker : Optional[HierarchicalProgressTracker]
            Tracker for reporting progress.

        **kwargs : Any
            - fidelity_metrics: List[str]
                List of metric types to compute (e.g., ["ks", "kl"])
            - metric_params: Dict[str, Dict]
                Mapping of metric type to its specific configuration params.

        Returns:
        --------
        Dict[str, Any]
            - result: Dict of metric_name → metric_result
        """
        fidelity_metrics: List[str] = kwargs.get("fidelity_metrics", [])
        metric_params: Dict[str, Dict] = kwargs.get("metric_params", {})

        if not fidelity_metrics:
            raise ValueError(
                "No fidelity metrics specified. 'fidelity_metrics' list is empty."
            )

        results: Dict[str, Any] = {}

        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Calculate Fidelity Metrics Setup",
                    "original_df_length": len(original_df),
                    "transformed_df_length": len(transformed_df),
                    "metrics": fidelity_metrics,
                },
            )

        self.logger.info(
            f"Calculating fidelity metrics: {fidelity_metrics} with params: {metric_params}"
        )

        for metric_type in fidelity_metrics:
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

                metric_class = FIDELITY_METRIC_FACTORY.get(metric_type)
                if not metric_class:
                    raise ValueError(f"Unsupported fidelity metric: {metric_type}")

                # Append shared init params if not in `params`
                params.setdefault("confidence_level", self.confidence_level)
                params.setdefault("normalize", self.normalize)

                # Instantiate the metric class with provided parameters
                metric = safe_instantiate(metric_class, params)

                # Calculate the metric
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

        This implementation generates separate visualizations for KS and KL metrics if present.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Collected metrics for visualization (may include 'ks', 'kl')
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
            # Step 1: Prepare fidelity metric visualizations
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Preparing fidelity metric visualizations"}
                )

            # Step 2: Create visualization
            if progress_tracker:
                progress_tracker.update(2, {"step": "Creating visualization"})

            # KS Visualization
            if "ks" in metrics:
                ks = metrics["ks"]
                ks_dict = {
                    "KS Statistic": ks.get("ks_statistic"),
                    "KS p-value": ks.get("p_value"),
                }

                ks_plot_path = viz_dir / f"{self.name}_ks_metric_{timestamp}.png"
                ks_plot_result = create_bar_plot(
                    ks_dict,
                    output_path=ks_plot_path,
                    title="Kolmogorov-Smirnov Test Metrics",
                    x_label="Metric",
                    y_label="Value",
                    backend=vis_backend,
                    theme=vis_theme,
                    strict=vis_strict,
                    **kwargs,
                )
                if not ks_plot_result.startswith("Error"):
                    visualization_paths["ks_metric_bar"] = ks_plot_path
                else:
                    self.logger.error(
                        f"Failed to create visualization: {ks_plot_result}"
                    )

            # KL Visualization
            if "kl" in metrics:
                kl = metrics["kl"]
                kl_dict = {
                    "KL Divergence (nats)": kl.get("kl_divergence"),
                    "KL Divergence (bits)": kl.get("kl_divergence_bits"),
                }

                kl_plot_path = viz_dir / f"{self.name}_kl_metric_{timestamp}.png"
                kl_plot_result = create_bar_plot(
                    kl_dict,
                    output_path=kl_plot_path,
                    title="KL Divergence Metrics",
                    x_label="Metric",
                    y_label="Value",
                    backend=vis_backend,
                    theme=vis_theme,
                    strict=vis_strict,
                    **kwargs,
                )
                if not kl_plot_result.startswith("Error"):
                    visualization_paths["kl_metric_bar"] = kl_plot_path
                else:
                    self.logger.error(
                        f"Failed to create visualization: {kl_plot_result}"
                    )

            # Step 3: Finalize visualizations
            if progress_tracker:
                progress_tracker.update(
                    3, {"step": "Fidelity metric visualizations complete"}
                )

        except Exception as e:
            self.logger.error(f"Failed to generate fidelity metric visualizations: {e}")

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
            fidelity_metrics=self.fidelity_metrics,
            metric_params=self.metric_params,
            columns=self.columns,
            column_mapping=self.column_mapping,
            normalize=self.normalize,
            confidence_level=self.confidence_level,
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
