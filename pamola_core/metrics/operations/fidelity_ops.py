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
    - Memory-efficient chunk-based processing for large datasets
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
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import dask.dataframe as dd
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics
from pamola_core.metrics.base_metrics_op import MetricsOperation
from pamola_core.metrics.fidelity.distribution.kl_divergence import KLDivergence
from pamola_core.metrics.fidelity.distribution.ks_test import KolmogorovSmirnovTest
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import create_bar_plot

# Default values
DEFAULT_SAMPLE_SIZE = 10000


class FidelityConfig(OperationConfig):
    """Configuration for FidelityOperation."""

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "default": "fidelity_metrics"},
            "fidelity_metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [FidelityMetrics.KS.value, FidelityMetrics.KL.value],
                },
                "default": [FidelityMetrics.KS.value, FidelityMetrics.KL.value],
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
            "description": {"type": "string", "default": ""},
            "optimize_memory": {"type": "boolean"},
            "adaptive_chunk_size": {"type": "boolean"},
            "chunk_size": {"type": "integer", "minimum": 1},
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": ["integer", "null"], "minimum": 1},
            "dask_partition_size": {"type": ["string", "null"], "default": "100MB"},
            "use_cache": {"type": "boolean"},
            "use_encryption": {"type": "boolean"},
            "encryption_key": {"type": ["string", "null"]},
            "encryption_mode": {
                "type": ["string", "null"],
                "enum": ["age", "simple", "none"],
                "default": "none",
            },
            "visualization_theme": {"type": ["string", "null"]},
            "visualization_backend": {
                "type": ["string", "null"],
                "enum": ["plotly", "matplotlib", None],
            },
            "visualization_strict": {"type": "boolean"},
            "visualization_timeout": {"type": "integer", "minimum": 1, "default": 120},
        },
        "required": ["fidelity_metrics"],
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
        fidelity_metrics: List[str] = [
            FidelityMetrics.KS.value,
            FidelityMetrics.KL.value,
        ],
        metric_params: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        normalize: bool = True,
        confidence_level: float = 0.95,
        description: str = "",
        # Memory optimization
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        # Specific parameters
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_cache: bool = True,
        use_encryption: bool = False,
        encryption_mode: Optional[str] = None,
        encryption_key: Optional[Union[str, Path]] = None,
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
    ):
        """
        Initialize the metrics operation.

        Parameters:
        -----------
        name : str, optional
            Name of the operation (default: "fidelity_metrics")
        fidelity_metrics : List[str], optional
            List of fidelity metrics to calculate (default: [FidelityMetric.KS.value, FidelityMetric.KL.value])
        metric_params : Optional[Dict[str, Any]], optional
            Additional parameters for metrics calculation (default: None)
        columns : List[str], optional
            List of columns to include in the operation
        column_mapping : Dict[str, str], optional
            Mapping of original columns to new names (if applicable)
        normalize : bool, optional
            Whether to normalize the data (default: True)
        confidence_level : float, optional
            Confidence level for statistical metrics (default: 0.95)
        sample_size : Optional[int], optional
            Size of the sample to use for metrics calculation (default: None, uses full dataset)
        description : str, optional
            Operation description
        optimize_memory : bool, optional
            Whether to optimize DataFrame memory usage
        adaptive_chunk_size : bool, optional
            Whether to adjust chunk size based on available memory
        chunk_size : int, optional
            Size of chunks for processing (default: 10000)
        use_dask : bool, optional
            Whether to use Dask for parallel processing (default: False)
        npartitions : Optional[int], optional
            Number of partitions to use with Dask (default: None)
        dask_partition_size : Optional[str], optional
            Size of Dask partitions (default: None, auto-determined)
        use_cache : bool, optional
            Whether to use operation caching (default: True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : str or Path, optional
            The encryption key or path to a key file (default: None)
        encryption_mode : Optional[str], optional
            The encryption mode to use (default: None)
        visualization_theme : Optional[str], optional
            Theme for visualizations (default: None)
        visualization_backend : Optional[str], optional
            Backend for visualizations ("plotly" or "matplotlib", default: None)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout for visualization generation in seconds (default: 120)
        """
        # Use a default description if none provided
        if not description:
            description = f"Fidelity metrics operation"

        # Group parameters into a config dict
        config_params = dict(
            name=name,
            fidelity_metrics=fidelity_metrics,
            metric_params=metric_params or {},
            columns=columns or [],
            column_mapping=column_mapping or {},
            normalize=normalize,
            confidence_level=confidence_level,
            sample_size=sample_size,
            description=description,
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_cache=use_cache,
            use_encryption=use_encryption,
            encryption_mode=encryption_mode,
            encryption_key=encryption_key,
            visualization_theme=visualization_theme,
            visualization_backend=visualization_backend,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
        )

        # Create configuration
        config = FidelityConfig(**config_params)

        # Initialize parent class
        super().__init__(
            **{
                k: config_params[k]
                for k in [
                    "name",
                    "columns",
                    "column_mapping",
                    "normalize",
                    "confidence_level",
                    "description",
                    "optimize_memory",
                    "adaptive_chunk_size",
                    "chunk_size",
                    "use_dask",
                    "npartitions",
                    "dask_partition_size",
                    "use_cache",
                    "use_encryption",
                    "encryption_mode",
                    "encryption_key",
                    "visualization_theme",
                    "visualization_backend",
                    "visualization_strict",
                    "visualization_timeout",
                ]
            }
        )

        # Save config attributes to self
        for k, v in config_params.items():
            setattr(self, k, v)
            self.process_kwargs[k] = v

        # Store the configuration
        self.config = config

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
                # Instantiate metric
                if metric_type == FidelityMetrics.KL.value:
                    metric = KLDivergence(
                        key_fields=params.get("key_fields"),
                        value_field=params.get("value_field"),
                        aggregation=params.get("aggregation", "count"),
                        epsilon=params.get("epsilon", 0.01),
                    )
                elif metric_type == FidelityMetrics.KS.value:
                    metric = KolmogorovSmirnovTest(
                        key_fields=params.get("key_fields"),
                        value_field=params.get("value_field"),
                        aggregation=params.get("aggregation", "sum"),
                    )
                else:
                    raise ValueError(f"Unsupported fidelity metric: {metric_type}")

                # Calculate and store result
                result = metric.calculate_metric(original_df, transformed_df)
                if progress_tracker:
                    progress_tracker.update(
                        3,
                        {
                            "step": f"Calculated  {metric_type.upper()} metric",
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

    def calculate_metrics_with_dask(
        self,
        original_ddf: dd.DataFrame,
        transformed_ddf: dd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Calculate the metric values using Dask - must be implemented by subclasses.

        Parameters:
        -----------
        original_ddf : dd.DataFrame
            DataFrame original to process
        transformed_ddf : dd.DataFrame
            DataFrame transformed to process
        progress_tracker : Optional[HierarchicalProgressTracker]
            Progress tracker for monitoring
        **kwargs : Any
            Additional parameters for processing

        Returns:
        --------
        Dict[str, Any]
            - Metric result dictionary
        """
        raise NotImplementedError("Not implement calculate_metrics_with_dask method")

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
                df_ks = pd.DataFrame(
                    [
                        {
                            "Metric": "KS Statistic",
                            "Value": ks.get("ks_statistic"),
                            "Interpretation": ks.get("interpretation", ""),
                        },
                        {
                            "Metric": "KS p-value",
                            "Value": ks.get("p_value"),
                            "Interpretation": ks.get("interpretation", ""),
                        },
                    ]
                )
                ks_plot_path = viz_dir / f"ks_metric_{timestamp}.png"
                create_bar_plot(
                    df_ks,
                    x="Metric",
                    y="Value",
                    output_path=ks_plot_path,
                    title="Kolmogorov-Smirnov Test Metrics",
                    backend=vis_backend,
                    theme=vis_theme,
                    strict=vis_strict,
                    **kwargs,
                )
                visualization_paths["ks_metric_bar"] = ks_plot_path

            # KL Visualization
            if "kl" in metrics:
                kl = metrics["kl"]
                df_kl = pd.DataFrame(
                    [
                        {
                            "Metric": "KL Divergence (nats)",
                            "Value": kl.get("kl_divergence"),
                            "Interpretation": kl.get("interpretation", ""),
                        },
                        {
                            "Metric": "KL Divergence (bits)",
                            "Value": kl.get("kl_divergence_bits"),
                            "Interpretation": kl.get("interpretation", ""),
                        },
                    ]
                )
                kl_plot_path = viz_dir / f"kl_metric_{timestamp}.png"
                create_bar_plot(
                    df_kl,
                    x="Metric",
                    y="Value",
                    output_path=kl_plot_path,
                    title="KL Divergence Metrics",
                    backend=vis_backend,
                    theme=vis_theme,
                    strict=vis_strict,
                    **kwargs,
                )
                visualization_paths["kl_metric_bar"] = kl_plot_path

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
            mode=self.mode,
            optimize_memory=self.optimize_memory,
            adaptive_chunk_size=self.adaptive_chunk_size,
            chunk_size=self.chunk_size,
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
