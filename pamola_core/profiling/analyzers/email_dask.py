"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Field Profiler Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides analyzers and operations for profiling email fields in tabular datasets.
  It includes email validation, domain extraction, pattern detection, and privacy risk assessment,
  supporting both pandas and Dask DataFrames.

Key Features:
  - Email validation and domain extraction
  - Pattern and uniqueness analysis for privacy risk assessment
  - Domain frequency dictionary and top-N domain statistics
  - Visualization generation for domain distributions
  - Efficient chunked, parallel, and Dask-based processing for large datasets
  - Robust error handling, progress tracking, and operation logging
  - Integration with PAMOLA.CORE operation framework for standardized input/output
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import dask.dataframe as dd
from pamola_core.profiling.commons.email_utils_dask import (
    analyze_email_field,
    create_domain_dictionary,
    estimate_resources,
)
from pamola_core.utils.io import (
    write_json,
    ensure_directory,
    load_data_operation,
    write_dataframe_to_csv,
    load_settings_operation,
)
from pamola_core.utils.io_helpers.dask_utils import get_computed_df
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import (
    OperationArtifact,
    OperationResult,
    OperationStatus,
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

# Configure logger
logger = logging.getLogger(__name__)


class EmailAnalyzer:
    """
    Analyzer for email fields.

    This analyzer provides static methods for validating emails, extracting domains,
    and identifying patterns in email addresses.
    """

    @staticmethod
    def analyze(
        df: Union[pd.DataFrame, dd.DataFrame],
        field_name: str,
        top_n: int = 20,
        use_dask: bool = False,
        use_vectorization: bool = False,
        chunk_size: int = 1000,
        parallel_processes: Optional[int] = 1,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        task_logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze an email field in the given DataFrame.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        top_n : int
            Number of top domains to include in the results
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        return analyze_email_field(
            df=df,
            field_name=field_name,
            top_n=top_n,
            use_dask=use_dask,
            chunk_size=chunk_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            progress_tracker=progress_tracker,
            task_logger=task_logger,
        )

    @staticmethod
    def create_domain_dictionary(
        df: Union[pd.DataFrame, dd.DataFrame],
        field_name: str,
        min_count: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a frequency dictionary for email domains.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the email field
        min_count : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with domain frequency data and metadata
        """
        return create_domain_dictionary(
            df=df, field_name=field_name, min_count=min_count, **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the email field.

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


class EmailOperationConfig(OperationConfig):
    """Configuration for EmailOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common operation fields
            {
                "type": "object",
                "properties": {
                    # --- Email-specific parameters ---
                    "field_name": {"type": "string"},
                    "top_n": {"type": "integer", "minimum": 1, "default": 20},
                    "min_frequency": {"type": "integer", "minimum": 1, "default": 1},
                    "profile_type": {
                        "type": "string",
                        "enum": ["email"],
                        "default": "email",
                    },
                    "analyze_privacy_risk": {"type": "boolean", "default": True},
                },
                "required": ["field_name"],
            },
        ],
    }


@register(version="1.0.0")
class EmailOperation(FieldOperation):
    """
    Operation for analyzing email fields.

    This operation wraps the EmailAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(
        self,
        field_name: str,
        top_n: int = 20,
        min_frequency: int = 1,
        profile_type: str = "email",
        analyze_privacy_risk: bool = True,
        **kwargs,
    ):
        """
        Initialize the email operation.

        Parameters
        ----------
        field_name : str
            The name of the field to analyze.
        top_n : int
            Number of top domains to include in the results.
        min_frequency : int
            Minimum frequency for inclusion in the dictionary.
        profile_type : str
            Type of profiling for organizing artifacts.
        analyze_privacy_risk : bool
            Whether to analyze potential privacy risks from email patterns.
        **kwargs
            Additional keyword arguments passed to FieldOperation.
        """
        # Description fallback
        kwargs.setdefault("description", f"Analysis of email field '{field_name}'")

        # --- Build unified config ---
        config = EmailOperationConfig(
            field_name=field_name,
            top_n=top_n,
            min_frequency=min_frequency,
            profile_type=profile_type,
            analyze_privacy_risk=analyze_privacy_risk,
            **kwargs,
        )

        # Pass config into kwargs for parent constructor
        kwargs["config"] = config

        # Initialize base FieldOperation
        super().__init__(
            field_name=field_name,
            **kwargs,
        )

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)

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
        Execute the email analysis operation.

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
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        try:
            if kwargs.get("logger"):
                self.logger = kwargs["logger"]

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            # Set up directories
            dirs = self._prepare_directories(task_dir)
            output_dir = dirs["output"]
            visualizations_dir = dirs["visualizations"]
            dictionaries_dir = dirs["dictionaries"]

            # Create the main result object with initial status
            result = OperationResult(status=OperationStatus.SUCCESS)

            # Save configuration
            self.save_config(task_dir)

            # Set up progress tracking
            # Preparation, Data Loading, Cache Check, Analysis, Saving results, Visualizations, Dictionary, Finalization
            total_steps = (
                6
                + (1 if self.use_cache and not self.force_recalculation else 0)
                + (1 if self.generate_visualization else 0)
            )

            # Step 0: Preparation
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(
                    0, {"step": "Preparation", "field": self.field_name}
                )

            # Step 1: Data Loading
            if progress_tracker:
                progress_tracker.update(1, {"step": "Data Loading"})

            try:
                # Load data
                df, error_info = data_source.get_dataframe(
                    name=dataset_name,
                    use_dask=self.use_dask,
                    use_encryption=self.use_encryption,
                    encryption_mode=self.encryption_mode,
                    encryption_key=self.encryption_key,
                )

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
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            # Step 2: Check Cache (if enabled and not forced to recalculate)
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    progress_tracker.update(2, {"step": "Checking Cache"})

                logger.info("Checking operation cache...")
                cache_result = self._check_cache(df, task_dir, reporter, **kwargs)

                if cache_result:
                    self.logger.info("Cache hit! Using cached results.")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(
                            total_steps, {"step": "Complete (cached)"}
                        )

                    # Report cache hit to reporter
                    if reporter:
                        reporter.add_operation(
                            f"Clean invalid values (from cache)",
                            details={"cached": True},
                        )
                    return cache_result

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame",
                )

            # Add operation to reporter
            if reporter:
                reporter.add_operation(
                    f"Analyzing email field: {self.field_name}",
                    details={
                        "field_name": self.field_name,
                        "top_n": self.top_n,
                        "min_frequency": self.min_frequency,
                        "operation_type": "email_analysis",
                    },
                )

            # Step 3: Analyze the email field
            if progress_tracker:
                progress_tracker.update(3, {"status": "Analyzing field"})

            # Execute the analyzer
            analysis_results = EmailAnalyzer.analyze(
                df=df,
                field_name=self.field_name,
                top_n=self.top_n,
                use_dask=self.use_dask,
                use_vectorization=self.use_vectorization,
                chunk_size=self.chunk_size,
                parallel_processes=self.parallel_processes,
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

            # Save analysis results to JSON
            stats_filename = f"{self.field_name}_stats_{operation_timestamp}.json"
            stats_path = output_dir / stats_filename

            encryption_mode = get_encryption_mode(analysis_results, **kwargs)
            write_json(
                analysis_results,
                stats_path,
                encryption_key=self.encryption_key,
                encryption_mode=encryption_mode,
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

            # Generate visualization if requested
            if (
                self.generate_visualization
                and "top_domains" in analysis_results
                and analysis_results["top_domains"]
            ):
                # Step 5: Visualizations
                # Update progress
                if progress_tracker:
                    progress_tracker.update(5, {"step": "Generating visualization"})

                self._handle_visualizations(
                    analysis_results=analysis_results,
                    vis_dir=visualizations_dir,
                    operation_timestamp=operation_timestamp,
                    result=result,
                    reporter=reporter,
                    vis_theme=self.visualization_theme,
                    vis_backend=self.visualization_backend,
                    vis_strict=self.visualization_strict,
                    vis_timeout=self.visualization_timeout,
                    progress_tracker=progress_tracker,
                    use_encryption=self.use_encryption,
                    encryption_key=self.encryption_key,
                )

                # Update progress
                if progress_tracker:
                    progress_tracker.update(5, {"step": "Created visualization"})

            # Create and save domain dictionary
            dict_result = EmailAnalyzer.create_domain_dictionary(
                df=df,
                field_name=self.field_name,
                min_count=self.min_frequency,
                **kwargs,
            )

            if "error" not in dict_result:
                # Save dictionary to CSV
                dict_filename = (
                    f"{self.field_name}_domains_dictionary_{operation_timestamp}.csv"
                )
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                dict_df = pd.DataFrame(dict_result["domains"])
                write_dataframe_to_csv(
                    df=dict_df,
                    file_path=dict_path,
                    index=False,
                    encoding="utf-8",
                    encryption_key=self.encryption_key,
                )

                # Save detailed dictionary as JSON
                json_dict_filename = (
                    f"{self.field_name}_domains_dictionary_{operation_timestamp}.json"
                )
                json_dict_path = output_dir / json_dict_filename
                encryption_mode_dict_result = get_encryption_mode(dict_result, **kwargs)
                write_json(
                    dict_result,
                    json_dict_path,
                    encryption_key=self.encryption_key,
                    encryption_mode=encryption_mode_dict_result,
                )

                result.add_artifact(
                    "csv",
                    dict_path,
                    f"{self.field_name} domains dictionary (CSV)",
                    category=Constants.Artifact_Category_Dictionary,
                )
                result.add_artifact(
                    "json",
                    json_dict_path,
                    f"{self.field_name} domains dictionary (JSON)",
                    category=Constants.Artifact_Category_Output,
                )

                reporter.add_artifact(
                    "csv", str(dict_path), f"{self.field_name} domains dictionary (CSV)"
                )
                reporter.add_artifact(
                    "json",
                    str(json_dict_path),
                    f"{self.field_name} domains dictionary (JSON)",
                )

            # Update progress
            # Step 6: Dictionary
            if progress_tracker:
                progress_tracker.update(6, {"step": "Created domain dictionary"})

            # Add privacy risk assessment if requested
            if self.analyze_privacy_risk:
                # Perform privacy risk assessment based on email uniqueness
                privacy_risk = self._assess_privacy_risk(df, self.field_name)

                if privacy_risk and len(privacy_risk) > 0:
                    # Save privacy risk assessment to JSON
                    privacy_filename = (
                        f"{self.field_name}_privacy_risk_{operation_timestamp}.json"
                    )
                    privacy_path = output_dir / privacy_filename
                    encryption_mode_privacy_risk = get_encryption_mode(
                        privacy_risk, **kwargs
                    )
                    write_json(
                        privacy_risk,
                        privacy_path,
                        encryption_key=self.encryption_key,
                        encryption_mode=encryption_mode_privacy_risk,
                    )

                    result.add_artifact(
                        "json",
                        privacy_path,
                        f"{self.field_name} privacy risk assessment",
                        category=Constants.Artifact_Category_Output,
                    )
                    reporter.add_artifact(
                        "json",
                        str(privacy_path),
                        f"{self.field_name} privacy risk assessment",
                    )

            # Add metrics to the result
            result.add_metric("total_records", analysis_results.get("total_rows", 0))
            result.add_metric("null_count", analysis_results.get("null_count", 0))
            result.add_metric(
                "null_percentage", analysis_results.get("null_percentage", 0)
            )
            result.add_metric("valid_count", analysis_results.get("valid_count", 0))
            result.add_metric(
                "valid_percentage", analysis_results.get("valid_percentage", 0)
            )
            result.add_metric(
                "unique_domains", analysis_results.get("unique_domains", 0)
            )

            # Step 7: Finalization
            if progress_tracker:
                progress_tracker.update(
                    7, {"step": "Operation complete", "status": "success"}
                )

            # Add final operation status to reporter
            reporter.add_operation(
                f"Analysis of {self.field_name} completed",
                details={
                    "valid_emails": analysis_results.get("valid_count", 0),
                    "invalid_emails": analysis_results.get("invalid_count", 0),
                    "unique_domains": analysis_results.get("unique_domains", 0),
                    "null_percentage": analysis_results.get("null_percentage", 0),
                },
            )

            # Cache the result if caching is enabled
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
            self.logger.exception(
                f"Error in email operation for {self.field_name}: {e}"
            )

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
                error_message=f"Error analyzing email field {self.field_name}: {str(e)}",
                exception=e,
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
        dictionaries_dir = task_dir / "dictionaries"

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)
        ensure_directory(dictionaries_dir)

        return {
            "output": output_dir,
            "visualizations": visualizations_dir,
            "dictionaries": dictionaries_dir,
        }

    def _assess_privacy_risk(
        self, df: Union[dd.DataFrame, pd.DataFrame], field_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Assess privacy risk based on email uniqueness.

        Parameters:
        -----------
        df : Union[dd.DataFrame, pd.DataFrame]
            The DataFrame containing the data
        field_name : str
            The name of the email field

        Returns:
        --------
        Optional[Dict[str, Any]]
            Privacy risk assessment results or None if assessment cannot be performed
        """
        try:
            df = get_computed_df(df)

            # Skip if field doesn't exist or is empty
            if field_name not in df.columns or df[field_name].isna().all():
                return {}

            # Count emails
            if df[field_name].dtype.name == "category":
                total_valid = (
                    df[field_name].apply(lambda x: not pd.isna(x)).astype(bool).sum()
                )
            else:
                total_valid = df[field_name].apply(lambda x: not pd.isna(x)).sum()

            if total_valid == 0:
                return {}

            # Count unique emails
            unique_emails = df[field_name].dropna().nunique()

            # Calculate uniqueness ratio
            uniqueness_ratio = unique_emails / total_valid if total_valid > 0 else 0

            # Assess risk levels
            risk_level = "Low"
            if uniqueness_ratio > 0.9:
                risk_level = "Very High"
            elif uniqueness_ratio > 0.7:
                risk_level = "High"
            elif uniqueness_ratio > 0.5:
                risk_level = "Medium"

            # Find most frequent emails (for potential exclusion)
            value_counts = df[field_name].value_counts()
            most_frequent = value_counts.head(10).to_dict()

            # Calculate singleton count (emails appearing only once)
            singles = value_counts[value_counts == 1]
            singleton_count = len(singles)

            # Create risk assessment
            return {
                "field_name": field_name,
                "total_valid_emails": int(total_valid),
                "unique_emails": int(unique_emails),
                "uniqueness_ratio": round(uniqueness_ratio, 4),
                "risk_level": risk_level,
                "most_frequent_count": len(
                    [count for count in value_counts if count > 1]
                ),
                "singleton_count": singleton_count,
                "singleton_percentage": (
                    round((singleton_count / total_valid) * 100, 2)
                    if total_valid > 0
                    else 0
                ),
                "most_frequent_examples": most_frequent,
            }
        except Exception as e:
            self.logger.error(
                f"Error in privacy risk assessment for {field_name}: {e}", exc_info=True
            )
            return {}

    def _create_visualizations(
        self,
        analysis_results: Dict[str, Dict[str, Any]],
        vis_dir: Path,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        timestamp: Optional[str] = None,
        **kwargs,
    ):
        """
        Create visualizations for k-anonymity metrics.

        Parameters:
        -----------
        analysis_results : Dict[str, Dict[str, Any]]
            Analysis Results.
        vis_dir : Path
            Directory where visualization files will be saved.
        result : OperationResult
            The operation result object to which visualization artifacts will be added.
        reporter : Any
            Reporter object for logging and artifact registration.
        vis_theme : Optional[str], default=None
            Theme to use for visualizations.
        vis_backend : Optional[str], default=None
            Visualization backend to use (e.g., "matplotlib", "plotly").
        vis_strict : bool, default=False
            If True, enforce strict visualization rules and raise errors on failure.
        timestamp : Optional[str]
            Timestamp for naming visualization files
        **kwargs
            Additional keyword arguments for visualization functions.
        """
        try:
            # Use provided timestamp or generate new one
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create visualization filename with extension "png"
            viz_filename = f"{self.field_name}_domains_distribution_{timestamp}.png"
            viz_path = vis_dir / viz_filename

            # Create visualization using the visualization module
            from pamola_core.utils.visualization import plot_email_domains

            title = f"Top Email Domains in {self.field_name}"

            viz_result = plot_email_domains(
                domains=analysis_results["top_domains"],
                output_path=str(viz_path),
                title=title,
                theme=vis_theme,
                backend=vis_backend,
                strict=vis_strict,
                **kwargs,
            )

            if not viz_result.startswith("Error"):
                result.add_artifact(
                    "png",
                    viz_path,
                    f"{self.field_name} domains distribution",
                    category=Constants.Artifact_Category_Visualization,
                )
                reporter.add_artifact(
                    "png", str(viz_path), f"{self.field_name} domains distribution"
                )
            else:
                self.logger.warning(f"Error creating visualization: {viz_result}")

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}", exc_info=True)
            if reporter:
                reporter.add_operation(
                    "Creating visualizations",
                    status="warning",
                    details={
                        "warning": f"Error creating some visualizations: {str(e)}"
                    },
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
        original_df : Union[pd.DataFrame, dd.DataFrame]
            Original input data
        artifacts : List[OperationArtifact]
            Operation Artifact
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
            "top_n": self.top_n,
            "min_frequency": self.min_frequency,
            "profile_type": self.profile_type,
            "analyze_privacy_risk": self.analyze_privacy_risk,
            "encryption_key": self.encryption_key,
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
        df : Union[pd.DataFrame, dd.DataFrame]
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
        result: OperationResult,
        reporter: Any,
        operation_timestamp: Optional[str] = None,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
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
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        operation_timestamp : Optional[str]
            Timestamp for naming visualization files
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

        self.logger.info(
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
                    self._create_visualizations(
                        analysis_results,
                        vis_dir,
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
                visualization_paths = {}
            elif visualization_error:
                self.logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                self.logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                self.logger.info(
                    f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}"
                )
        except Exception as e:
            self.logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            self.logger.error(f"[DIAG] Stack trace:", exc_info=True)
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


def analyze_email_fields(
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    email_fields: List[str] = None,
    **kwargs,
) -> Dict[str, OperationResult]:
    """
    Analyze multiple email fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    email_fields : List[str], optional
        List of email fields to analyze. If None, tries to find email fields automatically.
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top domains to include in results (default: 20)
        - min_frequency: int, minimum frequency for inclusion in dictionary (default: 1)
        - generate_visualization: bool, whether to generate visualization (default: True)
        - profile_type: str, type of profiling for organizing artifacts (default: 'email')

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
            "Email fields analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"},
        )
        return {}

    # Extract operation parameters from kwargs
    top_n = kwargs.get("top_n", 20)
    min_frequency = kwargs.get("min_frequency", 1)

    # If no email fields specified, try to detect them
    if email_fields is None:
        email_fields = []
        for col in df.columns:
            if "email" in col.lower():
                email_fields.append(col)

        if not email_fields:
            email_fields = ["email"]  # Default field name

    # Report on fields to be analyzed
    reporter.add_operation(
        "Email fields analysis",
        details={
            "fields_count": len(email_fields),
            "fields": email_fields,
            "top_n": top_n,
            "min_frequency": min_frequency,
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

    if track_progress and email_fields:
        overall_tracker = HierarchicalProgressTracker(
            total=len(email_fields),
            description=f"Analyzing {len(email_fields)} email fields",
            unit="fields",
            track_memory=True,
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, field in enumerate(email_fields):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(
                        0, {"field": field, "progress": f"{i + 1}/{len(email_fields)}"}
                    )

                logger.info(f"Analyzing email field: {field}")

                # Create and execute operation
                operation = EmailOperation(
                    field, top_n=top_n, min_frequency=min_frequency
                )
                result = operation.execute(data_source, task_dir, reporter, **kwargs)

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
                logger.error(f"Error analyzing email field {field}: {e}", exc_info=True)

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
        "Email fields analysis completed",
        details={
            "fields_analyzed": len(results),
            "successful": success_count,
            "failed": error_count,
        },
    )

    return results
