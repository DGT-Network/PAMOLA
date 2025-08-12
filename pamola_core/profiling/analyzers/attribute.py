"""
Attribute profiler operation for the project.

This module provides operations for automatically profiling attributes of input datasets
to categorize each column by its role in anonymization and synthesis tasks. It supports
both pandas.DataFrame and CSV files (using io.py).
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
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
    load_data_operation,
    load_settings_operation
    
)
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationArtifact, OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import (
    create_pie_chart,
    create_bar_plot,
    create_scatter_plot
)
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
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
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 use_dask: bool = False,
                 use_cache: bool = True,
                 use_vectorization: bool = False,
                 chunk_size: int = 10000,
                 parallel_processes: int = 1,
                 npartitions: int = 1,
                 visualization_theme: Optional[str] = None,
                 visualization_backend: Optional[str] = "plotly",
                 visualization_strict: bool = False,
                 visualization_timeout: int = 120,
                 encryption_mode: Optional[str] = None):
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
        use_encryption : bool
            Whether to use encryption for output files
        encryption_key : str or Path, optional
            Encryption key for output files
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
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode
        )
        # Store configuration parameters
        self.dictionary_path = dictionary_path
        self.language = language
        self.sample_size = sample_size
        self.max_columns = max_columns
        self.id_column = id_column
        self.use_dask = use_dask
        self.use_cache = use_cache
        self.use_vectorization = use_vectorization
        self.chunk_size = chunk_size
        self.parallel_processes = parallel_processes
        self.npartitions = npartitions

        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None,
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
            progress_tracker (HierarchicalProgressTracker, optional): Tracks operation progress
            **kwargs (dict): Additional configuration parameters
            Additional parameters:
                - npartitions: Number of partitions for Dask
                - force_recalculation: bool - Skip cache check
                - include_timestamp : bool - Whether to include timestamps in output filenames

        Returns:
            OperationResult: Detailed results of the attribute profiling operation
        """
        if kwargs.get('logger'):
            self.logger = kwargs['logger']
            
        # Extract operation parameters with fallback to default values
        dataset_name = kwargs.get('dataset_name', "main")
        dictionary_path = kwargs.get('dictionary_path', self.dictionary_path)
        language = kwargs.get('language', self.language)
        sample_size = kwargs.get('sample_size', self.sample_size)
        max_columns = kwargs.get('max_columns', self.max_columns)
        id_column = kwargs.get('id_column', self.id_column)
        include_timestamp = kwargs.get('include_timestamp', True)
        encryption_key = kwargs.get('encryption_key', self.encryption_key)
        self.use_dask = kwargs.get('use_dask', self.use_dask)
        self.use_vectorization= kwargs.get('use_dask', self.use_vectorization)
        self.chunk_size = kwargs.get('chunk_size', self.chunk_size)
        self.parallel_processes = kwargs.get("parallel_processes", self.parallel_processes)
        npartitions = kwargs.get('npartitions', self.npartitions)
        force_recalculation = kwargs.get("force_recalculation", False)

        self.visualization_theme = kwargs.get("visualization_theme", self.visualization_theme)
        self.visualization_backend = kwargs.get("visualization_backend", self.visualization_backend)
        self.visualization_strict = kwargs.get("visualization_strict", self.visualization_strict)
        self.visualization_timeout = kwargs.get("visualization_timeout", self.visualization_timeout)

        # Prepare output directories for artifacts
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

        # Initialize operation result with success status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Set up progress tracking
        # Preparation, Cache Check, Data Loading, Analysis, Metrics, Visualizations, Finalization
        total_steps = 5 + (1 if self.use_cache and not force_recalculation else 0)
        current_steps = 0

        # Configure progress tracking if provided
        if progress_tracker:
            progress_tracker.update(current_steps, {"step": "Preparation", "operation": self.name})
            progress_tracker.total = total_steps  # Define total steps for tracking

        # Step 1: Check Cache (if enabled and not forced to recalculate)
        if self.use_cache and not force_recalculation:
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Checking Cache"})

            logger.info("Checking operation cache...")
            cache_result = self._check_cache(data_source=data_source, data_source_name=dataset_name, **kwargs)

            if cache_result:
                self.logger.info("Cache hit! Using cached results.")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(total_steps,{"step": "Complete (cached)"})

                # Report cache hit to reporter
                if reporter:
                    reporter.add_operation(
                        f"Clean invalid values (from cache)",
                        details={"cached": True}
                    )
                return cache_result
            
        try:
            # Retrieve DataFrame from data source
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Log initial dataset information
            self.logger.info(f"Starting attribute profiling on DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Report operation details
            if reporter:
                reporter.add_operation(f"Attribute Profiling", details={
                    "records_count": len(df),
                    "columns_count": len(df.columns),
                    "language": language,
                    "operation_type": "attribute_profiling"
                })

            # Step 2: Data Loading
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Loading dictionary and preparing analysis"})

            # Load attribute dictionary (custom or default)
            dictionary = load_attribute_dictionary(dictionary_path)

            # Log attribute analysis start
            self.logger.info("Analyzing dataset attributes")
            
            total_rows = len(df)
            is_large_df = total_rows > self.chunk_size
            # Analyzer processes entire dataset, so bypass parallel and chunked processing
            if self.use_dask and is_large_df:
                # TODO: Implement Dask-based processing
                self.logger.warning("Dask processing not yet implemented, falling back to normal processing")
                pass
            
            if self.use_vectorization and self.parallel_processes > 1:
                # TODO: Implement joblib
                self.logger.warning("Joblib vectorized processing not yet implemented, falling back to normal processing")
                pass
            elif self.use_vectorization and not self.use_dask:
                # TODO: Implement chunk
                self.logger.warning("Vectorized processing not yet implemented, falling back to normal processing")
                pass

            # Perform comprehensive attribute analysis
            analysis_results = analyze_dataset_attributes(
                df=df,
                dictionary=dictionary,
                language=language,
                sample_size=sample_size,
                max_columns=max_columns,
                id_column=id_column
            )

            # Step 3: Analysis
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Analyzing dataset attributes"})

            # Save attribute roles to JSON
            roles_filename = get_timestamped_filename("attribute_roles", "json", include_timestamp)
            roles_path = output_dir / roles_filename
            encryption_mode_analysis = get_encryption_mode(analysis_results, **kwargs)
            write_json(analysis_results, roles_path, encryption_key=encryption_key, encryption_mode=encryption_mode_analysis)

            # Register artifacts
            result.add_artifact("json", roles_path, "Attribute roles analysis", category=Constants.Artifact_Category_Output)
            
            if reporter:
                reporter.add_artifact("json", str(roles_path), "Attribute roles analysis")

            # Create and save entropy DataFrame
            entropy_filename = get_timestamped_filename("attribute_entropy", "csv", include_timestamp)
            entropy_path = output_dir / entropy_filename

            # Build entropy data for each column
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
                encryption_mode_entropy = get_encryption_mode(entropy_df, **kwargs)
                write_dataframe_to_csv(entropy_df, entropy_path, encryption_key=encryption_key, encryption_mode=encryption_mode_entropy)
                result.add_artifact("csv", entropy_path, "Attribute entropy and uniqueness", category=Constants.Artifact_Category_Output)
                
                if reporter:
                    reporter.add_artifact("csv", str(entropy_path), "Attribute entropy and uniqueness")

            # Save sample values for each column
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

            encryption_mode_sample_data = get_encryption_mode(sample_data, **kwargs)
            write_json(sample_data, sample_path, encryption_key=encryption_key, encryption_mode=encryption_mode_sample_data)
            result.add_artifact("json", sample_path, "Attribute sample values", category=Constants.Artifact_Category_Dictionary)
            if reporter:
                reporter.add_artifact("json", str(sample_path), "Attribute sample values")

            # Step 4: Saving Metrics
            if progress_tracker:
                progress_tracker.update(current_steps, {"step": "Saving analysis results"})

            # Generate visualizations
            kwargs_encryption = {
                    "use_encryption": kwargs.get('use_encryption', False),
                    "encryption_key": encryption_key
                }
            
            # Step 5: Creating Visualizations
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Creating Visualizations"})

            self._handle_visualizations(
                    analysis_results = analysis_results,
                    vis_dir = visualizations_dir,
                    include_timestamp = include_timestamp,
                    result = result,
                    reporter = reporter,
                    vis_theme = self.visualization_theme,
                    vis_backend = self.visualization_backend,
                    vis_strict = self.visualization_strict,
                    vis_timeout = self.visualization_timeout,
                    progress_tracker = progress_tracker,
                    **kwargs_encryption
                )

            # Process quasi-identifiers
            quasi_identifiers = analysis_results["column_groups"]["QUASI_IDENTIFIER"]

            if quasi_identifiers:
                result.add_metric("quasi_identifiers", quasi_identifiers)

                quasi_filename = get_timestamped_filename("quasi_identifiers", "json", include_timestamp)
                quasi_path = output_dir / quasi_filename
                
                encryption_mode_quasi_identifier = get_encryption_mode(quasi_identifiers, **kwargs)
                write_json({"quasi_identifiers": quasi_identifiers}, quasi_path, encryption_key=encryption_key, encryption_mode=encryption_mode_quasi_identifier)
                result.add_artifact("json", quasi_path, "Quasi-identifiers list", category=Constants.Artifact_Category_Metrics)
                if reporter:
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

            # Step 6: Finalize progress tracking
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Operation complete", "status": "success"})

            # Report operation summary
            reporter.add_operation("Attribute Profiling Completed", details={
                "direct_identifiers": analysis_results["summary"]["DIRECT_IDENTIFIER"],
                "quasi_identifiers": analysis_results["summary"]["QUASI_IDENTIFIER"],
                "sensitive_attributes": analysis_results["summary"]["SENSITIVE_ATTRIBUTE"],
                "indirect_identifiers": analysis_results["summary"]["INDIRECT_IDENTIFIER"],
                "non_sensitive": analysis_results["summary"]["NON_SENSITIVE"],
                "conflicts": len(analysis_results.get("conflicts", []))
            })

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        artifacts=result.artifacts,
                        original_df=df,
                        metrics=result.metrics,
                        task_dir=task_dir
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            return result

        except Exception as e:
            # Comprehensive error handling
            self.logger.exception(f"Error in attribute profiling operation: {e}")

            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})
            if reporter:
                reporter.add_operation("Attribute Profiling",
                                    status="error",
                                    details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in attribute profiling: {str(e)}",
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
        # Create required directories for output, visualizations, and dictionaries
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
                               reporter: Any,
                               visualization_theme: Optional[str] = None,
                               visualization_backend: Optional[str] = "plotly",
                               visualization_strict: bool = False,
                               **kwargs):
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
        visualization_theme : str, optional
            Theme to use for visualizations
        visualization_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        visualization_strict : bool, optional
            If True, raise exceptions for configuration errors
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
                    hole=0.3,  # Create a donut chart
                    theme=visualization_theme,
                    backend=visualization_backend,
                    strict=visualization_strict,
                    **kwargs
                )

                if not pie_result.startswith("Error"):
                    result.add_artifact("png", pie_path, "Attribute role distribution", category=Constants.Artifact_Category_Visualization)
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
                    add_trendline=False,
                    theme=self.visualization_theme,
                    backend=self.visualization_backend,
                    strict=self.visualization_strict,
                    **kwargs
                )

                if not scatter_result.startswith("Error"):
                    result.add_artifact("png", entropy_path, "Entropy vs uniqueness analysis", category=Constants.Artifact_Category_Visualization)
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
                    y_label="Data Type",
                    theme=self.visualization_theme,
                    backend=self.visualization_backend,
                    strict=self.visualization_strict,
                    **kwargs
                )

                if not bar_result.startswith("Error"):
                    result.add_artifact("png", types_path, "Inferred data type distribution", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(types_path), "Inferred data type distribution")

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}", exc_info=True)
            reporter.add_operation("Creating visualizations", status="warning",
                                   details={"warning": f"Error creating some visualizations: {str(e)}"})
        
    def _check_cache(
            self,
            data_source: DataSource,
            data_source_name: str = "main",
            **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        task_dir : Path
            Task directory
        data_source_name: str
            Dataset name

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Get DataFrame from data source
            settings_operation = load_settings_operation(data_source, data_source_name, **kwargs)
            df = load_data_operation(data_source, data_source_name, **settings_operation)
            if df is None:
                self.logger.warning("No valid DataFrame found in data source")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            cached_result.add_metric(key, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        if isinstance(artifact, dict):
                            artifact_type = artifact.get("artifact_type", "")
                            artifact_path = artifact.get("path", "")
                            artifact_name = artifact.get("description", "")
                            artifact_category = artifact.get("category", "output")
                            cached_result.add_artifact(artifact_type, artifact_path, artifact_name, artifact_category) 

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric("cache_timestamp", cached_data.get("timestamp", "unknown"))

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None
        
    def _save_to_cache(
            self,
            original_df: pd.DataFrame,
            artifacts: List[OperationArtifact],
            metrics: Dict[str, Any],
            task_dir: Path
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        artifacts : list
            List of artifacts generated by the operation
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
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "artifacts": artifacts,
                "metrics": metrics,
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False
        
    def _generate_cache_key(
            self,
            df: pd.DataFrame
    ) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

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
            data_hash=data_hash
        )

    def _get_operation_parameters(
            self
    ) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "id_column": self.id_column,
            "sample_size": self.sample_size,
            "max_columns": self.max_columns,
            "language": self.language,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters
    
    def _get_cache_parameters(
            self
    ) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}
        
    
    def _generate_data_hash(
            self,
            df: pd.DataFrame
    ) -> str:
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
            # Create data characteristics summary for hashing
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format='iso')
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type         
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    
    def _handle_visualizations(
            self,
            analysis_results: Dict[str, Any],
            vis_dir: Path,
            include_timestamp: bool,
            result: OperationResult,
            reporter: Any,
            vis_theme: Optional[str] = None,
            vis_backend: Optional[str] = None,
            vis_strict: bool = False,
            vis_timeout: int = 120,
            progress_tracker: Optional[HierarchicalProgressTracker] = None,
            **kwargs
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of attribute analysis
        vis_dir : Path
            The task directory
        include_timestamp : bool
            Whether to include a timestamp in output filenames.
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
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

        logger.info(f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s")

        try:
            import threading
            import contextvars

            visualization_paths = {}
            visualization_error = None

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                logger.info(f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}")
                logger.info(f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}")

                start_time = time.time()

                try:
                    # Log context variables
                    logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        logger.info(f"[DIAG] Context vars count: {len(list(current_context))}")
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
                            logger.debug(f"Could not create child progress tracker: {e}")

                    # Generate visualizations
                    self._create_visualizations(
                        analysis_results,
                        vis_dir,
                        include_timestamp,
                        result,
                        reporter,
                        vis_theme,
                        vis_backend,
                        vis_strict,
                        **kwargs
                    )
                    # self._create_visualizations(
                    #     ka_metrics,
                    #     field_uniqueness,
                    #     field_combinations,
                    #     vis_dir,
                    #     include_timestamp,
                    #     result,
                    #     reporter,
                    #     vis_theme,
                    #     vis_backend,
                    #     vis_strict,
                    #     **kwargs
                    # )

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
                    logger.error(f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}")
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

            logger.info(f"[DIAG] Starting visualization thread with timeout={vis_timeout}s")
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    logger.info(f"[DIAG] Visualization thread still running after {elapsed:.1f}s...")

            if viz_thread.is_alive():
                logger.error(f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout")
                logger.error(f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}")
                visualization_paths = {}
            elif visualization_error:
                logger.error(f"[DIAG] Visualization failed with error: {visualization_error}")
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                logger.info(f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s")
                logger.info(f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}")
        except Exception as e:
            logger.error(f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}")
            logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = {}

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization"
                )

        return visualization_paths
