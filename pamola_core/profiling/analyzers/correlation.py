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
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
from pamola_core.profiling.commons.correlation_utils import (
    analyze_correlation,
    analyze_correlation_matrix,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation, load_settings_operation
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.progress import ProgressTracker, HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation, BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
from pamola_core.utils.visualization import (
    create_scatter_plot,
    create_boxplot,
    create_heatmap,
    create_correlation_matrix
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
    def analyze(df: pd.DataFrame,
                field1: str,
                field2: str,
                method: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze correlation between two fields in a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        field1 : str
            Name of the first field
        field2 : str
            Name of the second field
        method : str, optional
            Correlation method to use. If None, automatically selected based on data types.
        **kwargs : dict
            Additional parameters for analysis

        Returns:
        --------
        Dict[str, Any]
            Results of the correlation analysis
        """
        return analyze_correlation(
            df=df,
            field1=field1,
            field2=field2,
            method=method,
            **kwargs
        )

    @staticmethod
    def analyze_matrix(df: pd.DataFrame,
                       fields: List[str],
                       **kwargs) -> Dict[str, Any]:
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
        return analyze_correlation_matrix(
            df=df,
            fields=fields,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame,
                           field1: str,
                           field2: str) -> Dict[str, Any]:
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

    def __init__(self,
                 field1: str,
                 field2: str,
                 method: Optional[str] = None,
                 description: str = "",
                 profile_type: str = "correlation",
                 null_handling: str = "drop",
                 include_timestamp: bool = True,
                 save_output: bool = True,
                 generate_visualization: bool = True,
                 use_cache: bool = True,
                 force_recalculation: bool = False,
                 visualization_backend: Optional[str] = None,
                 visualization_theme: Optional[str] = None,
                 visualization_strict: bool = False,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 encryption_mode: Optional[str] = None):
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
        description : str
            Description of the operation (optional)
        generate_visualization : bool
            Whether to generate visualizations (default: True)
        include_timestamp : bool
            Whether to include timestamps in filenames (default: True)
        profile_type : str
            Type of profiling for organizing artifacts (default: "correlation")
        null_handling : str
            Method for handling nulls ('drop', 'fill', 'pairwise')    
        """
        # Use field1 as the primary field for the parent class
        super().__init__(
            field_name=field1,
            description=description or f"Correlation analysis between '{field1}' and '{field2}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode
            )
        self.field1 = field1
        self.field2 = field2
        self.method = method
        self.profile_type = profile_type
        self.null_handling = null_handling

        self.include_timestamp = include_timestamp
        self.save_output = save_output
        self.generate_visualization = generate_visualization

        self.use_cache = use_cache
        self.force_recalculation = force_recalculation

        self.visualization_backend = visualization_backend
        self.visualization_theme = visualization_theme
        self.visualization_strict = visualization_strict

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None,
                **kwargs) -> OperationResult:
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
            Additional parameters for the operation:
            - generate_visualization: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames
            - profile_type: str, type of profiling for organizing artifacts
            - null_handling: str, method for handling nulls ('drop', 'fill', 'pairwise')

        Returns:
        --------
        OperationResult
            Results of the operation
        """

        caller_operation = self.__class__.__name__
        self.logger = kwargs.get('logger', self.logger)

        try:
            # Start operation
            self.logger.info(f"Operation: {caller_operation}, Start operation")
            if progress_tracker:
                progress_tracker.total = self._compute_total_steps(**kwargs)
                progress_tracker.update(1, {"step": "Start operation - Preparation", "operation": caller_operation})

            # Set up directories
            dirs = self._prepare_directories(task_dir)
            visualizations_dir = dirs['visualizations']
            output_dir = dirs['output']

            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Preparation",
                                                "message": "Preparation successfully",
                                                "directories": {k: str(v) for k, v in dirs.items()}
                                                })

            # Load data and validate input parameters
            self.logger.info(f"Operation: {caller_operation}, Load data and validate input parameters")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Load data and validate input parameters",
                                            "operation": caller_operation})

            df, is_valid = self._load_data_and_validate_input_parameters(data_source, **kwargs)

            if is_valid:
                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Load data and validate input parameters",
                                                    "message": "Load data and validate input parameters successfully",
                                                    "shape": df.shape
                                                    })
            else:
                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Load data and validate input parameters",
                                                    "message": "Load data and validate input parameters failed"
                                                    })
                    return OperationResult(status=OperationStatus.ERROR,
                                           error_message="Load data and validate input parameters failed")

            # Handle cache if required
            if self.use_cache and not self.force_recalculation:
                self.logger.info(f"Operation: {caller_operation}, Load result from cache")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Load result from cache", "operation": caller_operation})

                cached_result = self._get_cache(df.copy(), **kwargs)  # _get_cache now returns OperationResult or None
                if cached_result is not None and isinstance(cached_result, OperationResult):
                    if reporter:
                        reporter.add_operation(f"Operation {caller_operation}", status="info",
                                               details={"step": "Load result from cache",
                                                        "message": "Load result from cache successfully"
                                                        })
                    return cached_result
                else:
                    self.logger.info(
                        f"Operation: {caller_operation}, Load result from cache failed — proceeding with execution.")
                    if reporter:
                        reporter.add_operation(f"Operation {caller_operation}", status="info",
                                               details={"step": "Load result from cache",
                                                        "message": "Load result from cache failed - proceeding with execution"
                                                        })

            # Analyzing correlation
            self.logger.info(f"Operation: {caller_operation}, Analyzing correlation")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analyzing correlation", "operation": caller_operation})

            analysis_results = CorrelationAnalyzer.analyze(
                df=df,
                field1=self.field1,
                field2=self.field2,
                method=self.method,
                null_handling=self.null_handling
            )

            # Check analysis results
            if 'error' in analysis_results:
                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Analyzing correlation",
                                                    "message": "Analyzing correlation failed",
                                                    "field1": self.field1,
                                                    "field2": self.field2,
                                                    "method": self.method or "auto",
                                                    "null_handling": self.null_handling,
                                                    "operation_type": "correlation_analysis"
                                           })
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )
            else:
                if reporter:
                    reporter.add_operation(f"Operation: {caller_operation}", status="info",
                                           details={"step": "Analyzing correlation",
                                                    "message": "Analyzing correlation successfully",
                                                    "field1": self.field1,
                                                    "field2": self.field2,
                                                    "method": self.method or "auto",
                                                    "null_handling": self.null_handling,
                                                    "operation_type": "correlation_analysis"
                                           })

            # Collect metric
            self.logger.info(f"Operation: {caller_operation}, Collect metric")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Collect metric", "operation": caller_operation})

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
                        "correlation_coefficient": analysis_results.get("correlation_coefficient", 0),
                        "sample_size": analysis_results.get("sample_size", 0),
                        "p_value": analysis_results.get("p_value"),
                        "statistically_significant": (
                                analysis_results.get("p_value") is not None and analysis_results["p_value"] < 0.05
                        )
                    }
                )

            # Save output if required
            if self.save_output:
                self.logger.info(f"Operation: {caller_operation}, Save output")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Save output", "operation": caller_operation})

                self._save_output(
                    analysis_results=analysis_results,
                    output_dir=output_dir,
                    result=result,
                    **kwargs
                )

                if reporter:
                    reporter.add_operation(f"Operation: {caller_operation}",
                                           status="info",
                                           details={"step": "Save output",
                                                    "message": "Save output successfully"
                                           })

            # Generate visualization if required
            if self.generate_visualization:
                self.logger.info(f"Operation: {caller_operation}, Generate visualizations")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Generate visualizations",
                                                "operation": caller_operation})

                viz_result = self._generate_visualizations(
                    analysis_results=analysis_results,
                    visualizations_dir=visualizations_dir,
                    result=result,
                    **kwargs
                )

                if not viz_result.startswith("Error"):
                    if reporter:
                        reporter.add_operation(f"Operation: {caller_operation}",
                                               status="info",
                                               details={"step": "Generate visualizations",
                                                        "message": "Generate visualizations successfully"
                                                        })
                else:
                    self.logger.warning(
                        f"Operation: {self.name}, Generate visualizations failed {viz_result}")
                    if reporter:
                        reporter.add_operation(f"Operation: {caller_operation}",
                                               status="info",
                                               details={"step": "Generate visualizations",
                                                        "message": "Generate visualizations failed",
                                                        "error": viz_result
                                                        })

            # Save cache if required
            if self.use_cache:
                self.logger.info(f"Operation: {caller_operation}, Save cache")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Save cache", "operation": caller_operation})

                self._save_cache(task_dir, result, **kwargs)

                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Save cache",
                                                    "message": "Save cache successfully"
                                                    })

            # Operation completed successfully
            self.logger.info(f"Operation: {caller_operation}, Completed successfully.")
            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Return result",
                                                "message": "Operation completed successfully"
                                                })

            return result


        except Exception as e:
            self.logger.error(f"Operation: {caller_operation}, error occurred: {e}")

            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="error",
                                       details={
                                           "step": "Exception",
                                           "message": "Operation failed due to an exception",
                                           "error": str(e)
                                       })

            return OperationResult(status=OperationStatus.ERROR, error_message=str(e))


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

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir
        }

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

        result.add_metric("correlation_method", analysis_results.get('method', 'unknown'))
        result.add_metric("correlation_coefficient", analysis_results.get('correlation_coefficient', 0))
        result.add_metric("sample_size", analysis_results.get('sample_size', 0))
        if 'p_value' in analysis_results and analysis_results['p_value'] is not None:
            result.add_metric("p_value", analysis_results['p_value'])
            result.add_metric("statistically_significant", analysis_results['p_value'] < 0.05)

    def _save_output(
            self,
            analysis_results: dict,
            output_dir: Path,
            result: OperationResult,
            **kwargs
    ):
        """
        Save analysis results to JSON, dictionary to CSV, and anomalies (if any).
        """

        # Save analysis results to JSON
        correlation_name = f"{self.field1}_{self.field2}_correlation"
        stats_filename = get_timestamped_filename(correlation_name, "json", self.include_timestamp)
        stats_path = output_dir / stats_filename

        encryption_mode_analysis = get_encryption_mode(analysis_results, **kwargs)
        write_json(analysis_results, stats_path, encryption_key=self.encryption_key, encryption_mode=encryption_mode_analysis)
        result.add_artifact("json", stats_path, f"Correlation analysis between {self.field1} and {self.field2}",
                            category=Constants.Artifact_Category_Output)

    def _generate_visualizations(
            self,
            analysis_results: dict,
            visualizations_dir: Path,
            result: OperationResult,
            **kwargs
    ) -> str:
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
            Visualization configuration options (theme, backend, strict, etc.).

        Returns
        -------
        str
            The result string from the visualization function (can indicate success or error).
        """
        if 'plot_data' not in analysis_results:
            warning_msg = f"Operation: {self.__class__.__name__}, No 'plot_data' found in analysis results for visualization."
            self.logger.warning(warning_msg)
            return f"Error: {warning_msg}"

        kwargs["backend"] = kwargs.pop("visualization_backend", self.visualization_backend)
        kwargs["theme"] = kwargs.pop("visualization_theme", self.visualization_theme)
        kwargs["strict"] = kwargs.pop("visualization_strict", self.visualization_strict)

        # Create visualization based on plot_data type
        plot_data = analysis_results['plot_data']
        plot_type = plot_data.get('type', 'unknown')
        correlation_name = f"{self.field1}_{self.field2}_correlation"
        viz_filename = get_timestamped_filename(correlation_name + "_plot", "png", self.include_timestamp)
        viz_path = visualizations_dir / viz_filename

        # Method details for plot title
        method_name = analysis_results.get('method', 'Unknown')
        method_display = method_name.replace('_', ' ').title()
        correlation_value = analysis_results.get('correlation_coefficient', 0)

        # Create appropriate visualization based on plot type
        viz_result = None

        if plot_type == "scatter":
            # For numeric-numeric correlations: scatter plot
            title = f"Correlation between {self.field1} and {self.field2}"
            viz_result = create_scatter_plot(
                x_data=plot_data['x_values'],
                y_data=plot_data['y_values'],
                output_path=str(viz_path),
                title=title,
                x_label=plot_data['x_label'],
                y_label=plot_data['y_label'],
                add_trendline=True,
                correlation=correlation_value,
                method=method_display,
                **kwargs
            )

        elif plot_type == "boxplot":
            # For categorical-numeric correlations: boxplot
            title = f"Relationship between {plot_data['x_label']} and {plot_data['y_label']}"
            viz_result = create_boxplot(
                data={cat: values if isinstance(values, (list, tuple)) else [values]
                      for cat, values in zip(plot_data['categories'], plot_data['values'])
                      if cat is not None},
                output_path=str(viz_path),
                title=title,
                x_label=plot_data['x_label'],
                y_label=plot_data['y_label'],
                **kwargs
            )

        elif plot_type == "heatmap":
            # For categorical-categorical correlations: heatmap
            title = f"Association between {plot_data['y_label']} and {plot_data['x_label']}"
            viz_result = create_heatmap(
                data=plot_data['matrix'],
                output_path=str(viz_path),
                title=title,
                x_label=plot_data['x_label'],
                y_label=plot_data['y_label'],
                annotate=True,
                **kwargs
            )

        if not viz_result.startswith("Error"):
            result.add_artifact("png", viz_result, f"{self.field_name} distribution visualization",
                                category=Constants.Artifact_Category_Visualization)

        return viz_result

    def _save_cache(self, task_dir: Path, result: OperationResult, **kwargs) -> None:
        """
        Save the operation result to cache.

        Parameters
        ----------
        task_dir : Path
            Root directory for the task.
        result : OperationResult
            The result object to be cached.
        """
        try:
            result_data = {
                "status": result.status.name if isinstance(result.status, OperationStatus) else str(result.status),
                "metrics": result.metrics,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "error_trace": result.error_trace,
                "artifacts": [artifact.to_dict() for artifact in result.artifacts]
            }

            cache_data = {
                "result": result_data,
                "parameters": self._get_cache_parameters(**kwargs),
            }

            cache_key = operation_cache.generate_cache_key(
                operation_name=self.__class__.__name__,
                parameters=self._get_cache_parameters(**kwargs),
                data_hash=self._generate_data_hash(self._original_df.copy())
            )

            operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
            )

            self.logger.info(f"Saved result to cache with key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _get_cache(self, df: pd.DataFrame, **kwargs) -> Optional[OperationResult]:
        """
        Retrieve cached result if available and valid.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame used to generate the cache key.

        Returns
        -------
        Optional[OperationResult]
            The cached OperationResult if available, otherwise None.
        """
        try:
            cache_key = operation_cache.generate_cache_key(
                operation_name=self.__class__.__name__,
                parameters=self._get_cache_parameters(**kwargs),
                data_hash=self._generate_data_hash(df)
            )

            cached = operation_cache.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            result_data = cached.get("result")
            if not isinstance(result_data, dict):
                return None

            # Parse enum safely
            status_str = result_data.get("status", OperationStatus.ERROR.name)
            status = OperationStatus[status_str] if isinstance(status_str,
                                                               str) and status_str in OperationStatus.__members__ else OperationStatus.ERROR

            # Rebuild artifacts
            artifacts = []
            for art_dict in result_data.get("artifacts", []):
                if isinstance(art_dict, dict):
                    try:
                        artifacts.append(OperationArtifact(
                            artifact_type=art_dict.get("type"),
                            path=art_dict.get("path"),
                            description=art_dict.get("description", ""),
                            category=art_dict.get("category", "output"),
                            tags=art_dict.get("tags", []),
                        ))
                    except Exception as e:
                        self.logger.warning(f"Failed to deserialize artifact: {e}")

            return OperationResult(
                status=status,
                artifacts=artifacts,
                metrics=result_data.get("metrics", {}),
                error_message=result_data.get("error_message"),
                execution_time=result_data.get("execution_time"),
                error_trace=result_data.get("error_trace"),
            )

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _get_cache_parameters(self, **kwargs) -> Dict[str, Any]:
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
            "field1": kwargs.get("field1"),
            "field2": kwargs.get("field2"),
            "method": kwargs.get("method", self.method),
            "include_timestamp": kwargs.get("include_timestamp", self.include_timestamp),
            "generate_visualization": kwargs.get("generate_visualization", self.generate_visualization),
            "profile_type": kwargs.get("profile_type", self.profile_type),
            "null_handling": kwargs.get("null_handling", self.null_handling),
            "use_cache": kwargs.get("use_cache", self.use_cache),
            "force_recalculation": kwargs.get("force_recalculation", self.force_recalculation),
            "visualization_backend": kwargs.get("visualization_backend", self.visualization_backend),
            "visualization_theme": kwargs.get("visualization_theme", self.visualization_theme),
            "visualization_strict": kwargs.get("visualization_strict", self.visualization_strict),
            "use_encryption": kwargs.get("use_encryption"),
            "encryption_key": str(kwargs.get("encryption_key")) if kwargs.get("encryption_key") else None
        }

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Generate a hash that represents key characteristics of the input DataFrame.

        The hash is based on structure and summary statistics to detect changes
        for caching purposes.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to generate a representative hash from.

        Returns
        -------
        str
            A hash string representing the structure and key properties of the data.
        """
        try:
            characteristics = {
                "columns": list(data.columns),
                "shape": data.shape,
                "summary": {}
            }

            for col in data.columns:
                col_data = data[col]
                col_info = {
                    "dtype": str(col_data.dtype),
                    "null_count": int(col_data.isna().sum()),
                    "unique_count": int(col_data.nunique())
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    non_null = col_data.dropna()
                    if not non_null.empty:
                        col_info.update({
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "mean": float(non_null.mean()),
                            "median": float(non_null.median()),
                            "std": float(non_null.std())
                        })
                elif pd.api.types.is_object_dtype(col_data) or isinstance(col_data.dtype, pd.CategoricalDtype):
                    top_values = col_data.value_counts(dropna=True).head(5)
                    col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

                characteristics["summary"][col] = col_info

            json_str = json.dumps(characteristics, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")
            fallback = f"{data.shape}_{list(data.dtypes)}"
            return hashlib.md5(fallback.encode()).hexdigest()

    def _set_input_parameters(self, **kwargs):
        """
        Set common configurable operation parameters from keyword arguments.
        """

        self.field1 = kwargs.get("field1", getattr(self, "field1", None))
        self.field2 = kwargs.get("field2", getattr(self, "field2", None))
        self.method = kwargs.get("method", getattr(self, "method", None))
        self.profile_type = kwargs.get("profile_type", getattr(self, "profile_type", "categorical"))
        self.null_handling = kwargs.get("null_handling", getattr(self, "null_handling", "drop"))
        self.generate_visualization = kwargs.get("generate_visualization", getattr(self, "generate_visualization", True))

        self.save_output = kwargs.get("save_output", getattr(self, "save_output", True))
        self.output_format = kwargs.get("output_format", getattr(self, "output_format", "csv"))
        self.include_timestamp = kwargs.get("include_timestamp", getattr(self, "include_timestamp", True))

        self.use_cache = kwargs.get("use_cache", getattr(self, "use_cache", True))
        self.force_recalculation = kwargs.get("force_recalculation", getattr(self, "force_recalculation", False))

        self.visualization_backend = kwargs.get("visualization_backend", getattr(self, "visualization_backend", False))
        self.visualization_theme = kwargs.get("visualization_theme", getattr(self, "visualization_theme", None))
        self.visualization_strict = kwargs.get("visualization_strict", getattr(self, "visualization_strict", None))

        self.use_encryption = kwargs.get("use_encryption", getattr(self, "use_encryption", False))
        self.encryption_key = kwargs.get("encryption_key",
                                         getattr(self, "encryption_key", None)) if self.use_encryption else None

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.include_timestamp else ""

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

    def _load_data_and_validate_input_parameters(self, data_source: DataSource, **kwargs) -> Tuple[Optional[pd.DataFrame], bool]:
        dataset_name = kwargs.get('dataset_name', "main")
        # settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
        df = load_data_operation(data_source, dataset_name)

        if df is None or df.empty:
            self.logger.error("Error data frame is None or empty")
            return None, False

        self._input_dataset = dataset_name
        self._original_df = df.copy(deep=True)

        return df, self._validate_input_parameters(df)

    def _compute_total_steps(self, **kwargs) -> int:
        use_cache = kwargs.get("use_cache", self.use_cache)
        force_recalculation = kwargs.get("force_recalculation", self.force_recalculation)
        save_output = kwargs.get("save_output", self.save_output)
        generate_visualization = kwargs.get("generate_visualization", self.generate_visualization)

        steps = 0

        steps += 1  # Step 1: Preparation
        steps += 1  # Step 2: Load data and validate input

        if use_cache and not force_recalculation:
            steps += 1  # Step 3: Try to load from cache

        steps += 1  # Step 4: Process data
        steps += 1  # Step 5: Collect metrics

        if save_output:
            steps += 1  # Step 6: Save output

        if generate_visualization:
            steps += 1  # Step 7: Generate visualizations

        if use_cache:
            steps += 1  # Step 8: Save cache

        return steps


@register(override=True)
class CorrelationMatrixOperation(BaseOperation):
    """
    Operation for creating a correlation matrix for multiple fields.

    This operation analyzes correlations between all pairs of fields in a list
    and generates a correlation matrix visualization.
    """

    def __init__(self,
                 fields: List[str],
                 methods: Optional[Dict[str, str]] = None,
                 description: str = "",
                 generate_visualization: bool = True,
                 include_timestamp: bool = True,
                 profile_type: str = "correlation",
                 min_threshold: float = 0.3,
                 null_handling: str = "drop",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
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
        profile_type : str
            Type of profiling for organizing artifacts
        min_threshold : float
            Minimum correlation threshold for significant correlations
        null_handling : str
            Method for handling nulls ('drop', 'fill', 'pairwise')
        """
        super().__init__(
            description or f"Correlation matrix analysis for {len(fields)} fields",
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )
        
        self.fields = fields
        self.methods = methods
        self.generate_visualization = generate_visualization
        self.include_timestamp = include_timestamp
        self.profile_type = profile_type
        self.min_threshold = min_threshold
        self.null_handling = null_handling

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
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
            - profile_type: str, type of profiling for organizing artifacts
            - null_handling: str, method for handling nulls ('drop', 'fill', 'pairwise')
            - min_threshold: float, minimum correlation threshold for significant correlations

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        generate_visualization = kwargs.get('generate_visualization', self.generate_visualization)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        profile_type = kwargs.get('profile_type', self.profile_type)
        min_threshold = kwargs.get('min_threshold', self.min_threshold)
        null_handling = kwargs.get('null_handling', self.null_handling)
        encryption_key = kwargs.get('encryption_key', None)

        # Set up directories
        output_dir = task_dir / 'output'
        visualizations_dir = task_dir / 'visualizations'
        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "fields_count": len(self.fields)})

        try:
            # Get DataFrame from data source
            dataset_name = kwargs.get('dataset_name', "main")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Check if fields exist
            missing_fields = [field for field in self.fields if field not in df.columns]
            if missing_fields:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Fields not found: {', '.join(missing_fields)}"
                )

            # Add operation to reporter
            if reporter:
                reporter.add_operation(f"Creating correlation matrix for {len(self.fields)} fields", details={
                    "fields": self.fields,
                    "null_handling": null_handling,
                    "min_threshold": min_threshold,
                    "operation_type": "correlation_matrix"
                })

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
                min_threshold=min_threshold
            )

            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analysis complete", "fields_count": len(self.fields)})

            # Save analysis results to JSON
            stats_filename = get_timestamped_filename("correlation_matrix", "json", include_timestamp)
            stats_path = output_dir / stats_filename

            write_json(analysis_results, stats_path, encryption_key=encryption_key)
            result.add_artifact("json", stats_path, "Correlation matrix analysis", category=Constants.Artifact_Category_Output)

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualization if requested
            if generate_visualization and 'correlation_matrix' in analysis_results:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualization"})

                # Convert dictionary to DataFrame for visualization
                matrix_dict = analysis_results['correlation_matrix']
                matrix_df = pd.DataFrame(matrix_dict)

                # Create visualization
                viz_filename = get_timestamped_filename("correlation_matrix_heatmap", "png", include_timestamp)
                viz_path = visualizations_dir / viz_filename

                viz_result = create_correlation_matrix(
                    data=matrix_df,
                    output_path=str(viz_path),
                    title="Correlation Matrix",
                    annotate=True,
                    annotation_format=".2f",
                    mask_diagonal=False,
                    mask_upper=False,
                    **kwargs
                )

                # Add visualization to result if successful
                if viz_result and not viz_result.startswith("Error"):
                    result.add_artifact("png", viz_path, "Correlation matrix visualization", category=Constants.Artifact_Category_Visualization)
                    if reporter:
                        reporter.add_artifact("png", str(viz_path), "Correlation matrix visualization")
                else:
                    self.logger.warning(f"Error creating visualization: {viz_result}")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualization"})

            # Add metrics to the result
            result.add_metric("fields_analyzed", len(self.fields))
            result.add_metric("significant_correlations",
                              len(analysis_results.get('significant_correlations', [])))
            result.add_metric("min_threshold", min_threshold)

            # Add final operation status to reporter
            significant_count = len(analysis_results.get('significant_correlations', []))
            if reporter:
                reporter.add_operation(f"Correlation matrix analysis completed", details={
                    "fields_analyzed": len(self.fields),
                    "significant_correlations": significant_count,
                    "min_threshold": min_threshold
                })

            return result

        except Exception as e:
            self.logger.exception(f"Error in correlation matrix operation: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            if reporter:
                reporter.add_operation(f"Error creating correlation matrix",
                                       status="error",
                                       details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error creating correlation matrix: {str(e)}"
            )


def analyze_correlations(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        pairs: List[Tuple[str, str]],
        **kwargs) -> Dict[str, OperationResult]:
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
        - generate_visualization: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps (default: True)
        - profile_type: str, type of profiling (default: 'correlation')

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping pair names to their operation results
    """
    # Get DataFrame from data source to check fields
    dataset_name = kwargs.get('dataset_name', "main")
    df = load_data_operation(data_source, dataset_name)
    if df is None:
        if reporter:
            reporter.add_operation("Correlation analysis", status="error",
                                   details={"error": "No valid DataFrame found in data source"})
        return {}

    # Extract parameters from kwargs
    methods = kwargs.get('methods', {})
    null_handling = kwargs.get('null_handling', 'drop')
    generate_visualization = kwargs.get('generate_visualization', True)

    # Report on field pairs to be analyzed
    if reporter:
        reporter.add_operation("Correlation analysis", details={
            "pairs_count": len(pairs),
            "pairs": [f"{field1}_{field2}" for field1, field2 in pairs],
            "null_handling": null_handling,
            "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
        })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and pairs:
        overall_tracker = ProgressTracker(
            total=len(pairs),
            description=f"Analyzing {len(pairs)} field correlations",
            unit="pairs",
            track_memory=True
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
                    details={"error": error_msg}
                )

            # Create an error result
            error_result = OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_msg
            )
            results[f"{field1}_{field2}"] = error_result

            # Update overall tracker if present
            if overall_tracker:
                overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "error"})

            continue

        try:
            # Update overall progress tracker
            if overall_tracker:
                overall_tracker.update(0, {"pair": f"{field1}_{field2}", "progress": f"{i + 1}/{len(pairs)}"})

            print(f"Analyzing correlation between {field1} and {field2}")

            # Get method if specified
            method = methods.get(f"{field1}_{field2}")

            # Create and execute operation
            operation = CorrelationOperation(
                field1=field1,
                field2=field2,
                method=method
            )
            result = operation.execute(
                data_source,
                task_dir,
                reporter,
                null_handling=null_handling,
                generate_visualization=generate_visualization,
                **kwargs
            )

            # Store result
            results[f"{field1}_{field2}"] = result

            # Update overall tracker after successful analysis
            if overall_tracker:
                if result.status == OperationStatus.SUCCESS:
                    overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "completed"})
                else:
                    overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "error",
                                               "error": result.error_message})

        except Exception as e:
            print(f"Error analyzing correlation between {field1} and {field2}: {e}", exc_info=True)

            if reporter:
                reporter.add_operation(f"Analyzing correlation between {field1} and {field2}", status="error",
                                       details={"error": str(e)})

            # Create an error result
            error_result = OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
            results[f"{field1}_{field2}"] = error_result

            # Update overall tracker in case of error
            if overall_tracker:
                overall_tracker.update(1, {"pair": f"{field1}_{field2}", "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(1 for r in results.values() if r.status == OperationStatus.SUCCESS)
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    if reporter:
        reporter.add_operation("Correlation analysis completed", details={
            "pairs_analyzed": len(results),
            "successful": success_count,
            "failed": error_count
        })

    return results