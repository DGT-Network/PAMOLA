"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Categorical Field Analyzer
Description: Operation for analyzing categorical fields and generating distribution statistics
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides analyzers and operations for categorical fields, following the
new operation architecture. It includes distribution analysis, dictionary creation,
anomaly detection, and visualization capabilities.

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
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from pamola_core.profiling.commons.categorical_utils import (
    analyze_categorical_field,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation, write_dataframe_to_csv, load_settings_operation
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
from pamola_core.utils.visualization import (
    plot_value_distribution
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

class CategoricalAnalyzer:
    """
    Analyzer for categorical fields.

    This analyzer provides methods for analyzing categorical fields, including
    frequency distributions, cardinality metrics, and dictionary creation.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field_name: str,
                top_n: int = 15,
                min_frequency: int = 1,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a categorical field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top values to include in the results
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        return analyze_categorical_field(
            df=df,
            field_name=field_name,
            top_n=top_n,
            min_frequency=min_frequency,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the categorical field.

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


# Use register_operation as a decorator by making a function
def register(override=False, dependencies=None, version=None):
    """
    Decorator to register an operation class.

    Parameters:
    -----------
    override : bool
        Whether to override an existing registration
    dependencies : List[Dict[str, str]], optional
        List of dependencies for the operation
    version : str, optional
        Version of the operation

    Returns:
    --------
    callable
        Decorator function
    """

    def decorator(cls):
        register_operation(cls, override=override, dependencies=dependencies, version=version)
        return cls

    return decorator


@register(override=True)
class CategoricalOperation(FieldOperation):
    """
    Operation for analyzing categorical fields.

    This operation wraps the CategoricalAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                 field_name: str,
                 top_n: int = 15,
                 min_frequency: int = 1,
                 profile_type: str = "categorical",
                 analyze_anomalies: bool = True,
                 description: str = "",
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
        Initialize the categorical operation.

        Parameters:
        -----------
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top values to include in the results
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        include_timestamp : bool
            Whether to include timestamps in filenames
        generate_visualization : bool
            Whether to generate visualizations
        profile_type : str
            Type of profiling for organizing artifacts
        analyze_anomalies : bool
            Whether to analyze anomalies
        description : str
            Description of the operation (optional)
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of categorical field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode
            )
        
        self.top_n = top_n
        self.min_frequency = min_frequency
        self.profile_type = profile_type
        self.analyze_anomalies = analyze_anomalies

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
        Execute the categorical analysis operation.

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
            - analyze_anomalies: bool, whether to analyze anomalies

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
            dictionaries_dir = dirs['dictionaries']
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

            # Analyzing categorical
            self.logger.info(f"Operation: {caller_operation}, Analyzing categorical")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analyzing categorical", "operation": caller_operation})

            analysis_results = CategoricalAnalyzer.analyze(
                df=df,
                field_name=self.field_name,
                top_n=self.top_n,
                min_frequency=self.min_frequency,
                detect_anomalies=self.analyze_anomalies
            )

            # Check analysis results
            if 'error' in analysis_results:
                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Analyzing categorical",
                                                    "message": "Analyzing categorical failed",
                                                    "field_name": self.field_name,
                                                    "top_n": self.top_n,
                                                    "min_frequency": self.min_frequency,
                                                    "operation_type": "categorical_analysis"
                                           })
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )
            else:
                if reporter:
                    reporter.add_operation(f"Operation: {caller_operation}", status="info",
                                           details={"step": "Analyzing categorical",
                                                    "message": "Analyzing categorical successfully",
                                                    "field_name": self.field_name,
                                                    "top_n": self.top_n,
                                                    "min_frequency": self.min_frequency,
                                                    "operation_type": "categorical_analysis"
                                           })

            # Collect metric
            self.logger.info(f"Operation: {caller_operation}, Collect metric")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Collect metric", "operation": caller_operation})

            result = OperationResult(status=OperationStatus.SUCCESS)

            self._collect_metrics(analysis_results, result)

            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Collect metric",
                                                "message": "Collect metric successfully",
                                                "unique_values": analysis_results.get('unique_values', 0),
                                                "null_percent": analysis_results.get('null_percent', 0),
                                                "entropy": round(analysis_results.get('entropy', 0), 2),
                                                "anomalies_found": len(analysis_results.get('anomalies',
                                                                                            {})) if 'anomalies' in analysis_results else 0
                                                })

            # Save output if required
            if self.save_output:
                self.logger.info(f"Operation: {caller_operation}, Save output")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Save output", "operation": caller_operation})

                self._save_output(
                    analysis_results=analysis_results,
                    output_dir=output_dir,
                    dictionaries_dir=dictionaries_dir,
                    result=result,
                    reporter=reporter,
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
                    progress_tracker.update(1, {"step": "Generate visualizations", "operation": caller_operation})

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
                    self.logger.warning(f"Operation: {self.name}, Generate visualizations failed {viz_result}")
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
        dictionaries_dir = task_dir / 'dictionaries'

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)
        ensure_directory(dictionaries_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir,
            'dictionaries': dictionaries_dir
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
        result.add_metric("total_records", analysis_results.get("total_records", 0))
        result.add_metric("null_count", analysis_results.get("null_values", 0))
        result.add_metric("null_percent", analysis_results.get("null_percent", 0))
        result.add_metric("unique_values", analysis_results.get("unique_values", 0))
        result.add_metric("entropy", analysis_results.get("entropy", 0))
        result.add_metric("cardinality_ratio", analysis_results.get("cardinality_ratio", 0))

        if "distribution_type" in analysis_results:
            result.add_metric("distribution_type", analysis_results["distribution_type"])

        if "anomalies" in analysis_results:
            result.add_metric("anomalies_count", len(analysis_results["anomalies"]))

    def _save_output(
            self,
            analysis_results: dict,
            output_dir: Path,
            dictionaries_dir: Path,
            result: OperationResult,
            reporter: Any,
            **kwargs
    ):
        """
        Save analysis results to JSON, dictionary to CSV, and anomalies (if any).
        """

        # Save analysis results to JSON
        stats_filename = get_timestamped_filename(f"{self.field_name}_stats", "json", self.include_timestamp)
        stats_path = output_dir / stats_filename

        encryption_mode_for_json = get_encryption_mode(analysis_results, **kwargs)
        write_json(analysis_results, stats_path, encryption_key=self.encryption_key, encryption_mode=encryption_mode_for_json)
        result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis",
                            category=Constants.Artifact_Category_Output)

        # Save dictionary to CSV
        if 'value_dictionary' in analysis_results and 'dictionary_data' in analysis_results['value_dictionary']:
            dict_filename = get_timestamped_filename(f"{self.field_name}_dictionary", "csv", self.include_timestamp)
            dict_path = dictionaries_dir / dict_filename

            dict_records = analysis_results['value_dictionary']['dictionary_data']
            if isinstance(dict_records, list) and len(dict_records) > 0:
                dict_df = pd.DataFrame(dict_records)
                write_dataframe_to_csv(
                    df=dict_df,
                    file_path=dict_path,
                    index=False,
                    encoding='utf-8',
                    encryption_key=self.encryption_key
                )
                result.add_artifact("csv", dict_path, f"{self.field_name} value dictionary",
                                    category=Constants.Artifact_Category_Dictionary)
            else:
                self.logger.info(f"Operation: {self.name}, Empty dictionary data for {self.field_name}")

        # Save anomalies to CSV if detected
        if 'anomalies' in analysis_results and analysis_results['anomalies']:
            self._save_anomalies_to_csv(
                analysis_results,
                dictionaries_dir,
                self.include_timestamp,
                result,
                reporter,
                encryption_key=self.encryption_key
            )

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
        if 'top_values' not in analysis_results:
            warning_msg = f"Operation: {self.name}, No 'top_values' found in analysis results for visualization."
            self.logger.warning(warning_msg)
            return f"Error: {warning_msg}"

        kwargs["backend"] = kwargs.pop("visualization_backend", self.visualization_backend)
        kwargs["theme"] = kwargs.pop("visualization_theme", self.visualization_theme)
        kwargs["strict"] = kwargs.pop("visualization_strict", self.visualization_strict)

        viz_filename = get_timestamped_filename(f"{self.field_name}_distribution", "png", self.include_timestamp)
        viz_path = visualizations_dir / viz_filename

        title = f"Distribution of {self.field_name}"
        viz_result = plot_value_distribution(
            data=analysis_results['top_values'],
            output_path=str(viz_path),
            title=title,
            max_items=self.top_n,
            **kwargs
        )

        if not viz_result.startswith("Error"):
            result.add_artifact("png", viz_result, f"{self.field_name} distribution visualization",
                                category=Constants.Artifact_Category_Visualization)

        return viz_result

    def _save_anomalies_to_csv(self,
                               analysis_results: Dict[str, Any],
                               dict_dir: Path,
                               include_timestamp: bool,
                               result: OperationResult,
                               reporter: Any,
                               encryption_key: Optional[str] = None):
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
            # Extract anomalies
            anomalies = analysis_results.get('anomalies', {})
            if not anomalies:
                return

            # Collect potential typos
            anomaly_records = []

            if 'potential_typos' in anomalies:
                for rare_value, typo_info in anomalies['potential_typos'].items():
                    anomaly_records.append({
                        'value': rare_value,
                        'frequency': typo_info['count'],
                        'anomaly_type': 'potential_typo',
                        'similar_to': typo_info['similar_to'],
                        'similar_count': typo_info['similar_count']
                    })

            # Collect single character values
            if 'single_char_values' in anomalies:
                for value, count in anomalies['single_char_values'].items():
                    anomaly_records.append({
                        'value': value,
                        'frequency': count,
                        'anomaly_type': 'single_char_value',
                        'similar_to': '',
                        'similar_count': 0
                    })

            # Collect numeric-like strings
            if 'numeric_like_strings' in anomalies:
                for value, count in anomalies['numeric_like_strings'].items():
                    anomaly_records.append({
                        'value': value,
                        'frequency': count,
                        'anomaly_type': 'numeric_like_string',
                        'similar_to': '',
                        'similar_count': 0
                    })

            # Create anomalies CSV filename
            if anomaly_records:
                anomalies_filename = get_timestamped_filename(f"{self.field_name}_anomalies", "csv", include_timestamp)
                anomalies_path = dict_dir / anomalies_filename

                # Create DataFrame and save to CSV
                anomalies_df = pd.DataFrame(anomaly_records)
                write_dataframe_to_csv(df=anomalies_df, file_path=anomalies_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Add artifact to result and reporter
                result.add_artifact("csv", anomalies_path, f"{self.field_name} anomalies", category=Constants.Artifact_Category_Dictionary)
                if reporter:
                    reporter.add_artifact("csv", str(anomalies_path), f"{self.field_name} anomalies")

        except Exception as e:
            self.logger.warning(f"Error saving anomalies for {self.field_name}: {e}")
            if reporter:
                reporter.add_operation(f"Saving anomalies for {self.field_name}", status="warning",
                                       details={"warning": str(e)})

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
            "field_name": kwargs.get("field_name"),
            "top_n": kwargs.get("top_n", self.top_n),
            "min_frequency": kwargs.get("min_frequency", self.min_frequency),
            "include_timestamp": kwargs.get("include_timestamp", self.include_timestamp),
            "generate_visualization": kwargs.get("generate_visualization", self.generate_visualization),
            "profile_type": kwargs.get("profile_type", self.profile_type),
            "analyze_anomalies": kwargs.get("analyze_anomalies", self.analyze_anomalies),
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

        self.field_name = kwargs.get("field_name", getattr(self, "field_name", None))
        self.top_n = kwargs.get("top_n", getattr(self, "top_n", None))
        self.min_frequency = kwargs.get("min_frequency", getattr(self, "min_frequency", True))
        self.profile_type = kwargs.get("profile_type", getattr(self, "profile_type", "categorical"))
        self.analyze_anomalies = kwargs.get("analyze_anomalies", getattr(self, "analyze_anomalies", True))
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
        if self.field_name not in df.columns:
            self.logger.error(f"Column {self.field_name} not existing in data frame")
            return False

        # All validations passed
        return True

    def _load_data_and_validate_input_parameters(self, data_source: DataSource, **kwargs) -> Tuple[Optional[pd.DataFrame], bool]:
        self._set_input_parameters(**kwargs)

        dataset_name = kwargs.get('dataset_name', "main")
        settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
        df = load_data_operation(data_source, dataset_name, **settings_operation)

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


def analyze_categorical_fields(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        cat_fields: List[str] = None,
        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple categorical fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    cat_fields : List[str], optional
        List of categorical fields to analyze. If None, tries to find categorical fields automatically.
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top values to include in results (default: 15)
        - min_frequency: int, minimum frequency for inclusion in dictionary (default: 1)
        - generate_visualization: bool, whether to generate plots (default: True)
        - include_timestamp: bool, whether to include timestamps in filenames (default: True)
        - profile_type: str, type of profiling for organizing artifacts (default: 'categorical')

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source safely
    df_result = data_source.get_dataframe("main")
    error_info = None  # Initialize error_info to avoid reference before assignment

    if isinstance(df_result, tuple) and len(df_result) >= 2:
        df, error_info = df_result
    else:
        df = df_result

    # Check if DataFrame is valid
    if df is None:
        error_message = "No valid DataFrame found in data source"
        if error_info is not None and isinstance(error_info, dict):
            error_message += f": {error_info.get('message', '')}"
        if reporter:
            reporter.add_operation("Categorical fields analysis", status="error",
                                   details={"error": error_message})
        return {}

    # Extract operation parameters from kwargs
    top_n = kwargs.get('top_n', 15)
    min_frequency = kwargs.get('min_frequency', 1)

    # If no categorical fields specified, try to detect them
    if cat_fields is None:
        cat_fields = []
        # Simple heuristic: select fields with string type or moderate number of unique values
        if hasattr(df, 'columns'):
            for col in df.columns:
                try:
                    # Check if column is object type (usually string)
                    if df[col].dtype == 'object':
                        cat_fields.append(col)
                    # Or check number of unique values relative to dataset size
                    elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= min(100, int(len(df) * 0.1)):
                        cat_fields.append(col)
                except Exception as e:
                    print(f"Error checking column {col}: {str(e)}")
        else:
            print("DataFrame does not have columns attribute")
            if reporter:
                reporter.add_operation("Categorical fields detection", status="error",
                                       details={"error": "DataFrame doesn't have expected structure"})

    # Report on fields to be analyzed
    if reporter:
        reporter.add_operation("Categorical fields analysis", details={
            "fields_count": len(cat_fields),
            "fields": cat_fields,
            "top_n": top_n,
            "min_frequency": min_frequency,
            "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
        })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and cat_fields:
        overall_tracker = HierarchicalProgressTracker(
            total=len(cat_fields),
            description=f"Analyzing {len(cat_fields)} categorical fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(cat_fields):
        # Check if field exists in DataFrame
        if hasattr(df, 'columns') and field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(cat_fields)}"})

                print(f"Analyzing categorical field: {field}")

                # Create and execute operation
                operation = CategoricalOperation(
                    field,
                    top_n=top_n,
                    min_frequency=min_frequency
                )
                result = operation.execute(data_source, task_dir, reporter, **kwargs)

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
                print(f"Error analyzing categorical field {field}: {e}", exc_info=True)
                if reporter:
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

    if reporter:
        reporter.add_operation("Categorical fields analysis completed", details={
            "fields_analyzed": len(results),
            "successful": success_count,
            "failed": error_count
        })

    return results


# Register the operation so it's discoverable
register_operation(CategoricalOperation)