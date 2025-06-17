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
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import (
    create_pie_chart,
    create_bar_plot,
    create_scatter_plot
)
from pamola_core.common.constants import Constants
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
                 include_timestamp: bool = True,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 chunk_size: int = 10000,
                 use_dask: bool = False,
                 use_cache: bool = True,
                 use_vectorization: bool = False):
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
        include_timestamp : bool
            Whether to include timestamps in filenames
        use_encryption : bool
            Whether to use encryption for output files
        encryption_key : str or Path, optional
            Encryption key for output files
        chunk_size : int
            Chunk size for processing large datasets
        use_dask : bool
            Whether to use Dask for large datasets
        use_cache : bool
            Whether to use cache for operation results
        use_vectorization : bool
            Whether to use vectorized operations
        """
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )
        # Store configuration parameters
        self.dictionary_path = dictionary_path
        self.language = language
        self.sample_size = sample_size
        self.max_columns = max_columns
        self.id_column = id_column
        self.include_timestamp = include_timestamp
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.include_timestamp = include_timestamp
        self.use_cache = use_cache
        self.use_vectorization = use_vectorization

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

        Returns:
            OperationResult: Detailed results of the attribute profiling operation
        """
        # Extract operation parameters with fallback to default values
        dataset_name = kwargs.get('dataset_name', "main")
        dictionary_path = kwargs.get('dictionary_path', self.dictionary_path)
        language = kwargs.get('language', self.language)
        sample_size = kwargs.get('sample_size', self.sample_size)
        max_columns = kwargs.get('max_columns', self.max_columns)
        id_column = kwargs.get('id_column', self.id_column)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        encryption_key = kwargs.get('encryption_key', self.encryption_key)
        use_dask = kwargs.get('use_dask', self.use_dask)
        use_vectorization= kwargs.get('use_dask', self.use_vectorization)
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        force_recalculation = kwargs.get("force_recalculation", False)
        parallel_processes = kwargs.get("parallel_processes", 1)
        npartitions = kwargs.get('npartitions', 1)

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

            self.logger.info("Checking operation cache...")
            cache_result = self._check_cache(data_source, dataset_name, **kwargs)

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
            logger.info(f"Starting attribute profiling on DataFrame with {len(df)} rows and {len(df.columns)} columns")

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
            logger.info("Analyzing dataset attributes")
            
            total_rows = len(df)
            is_large_df = total_rows > chunk_size
            # Analyzer processes entire dataset, so bypass parallel and chunked processing
            if self.use_dask and is_large_df:
                # TODO: Implement Dask-based processing for large datasets
                logger.warning("Dask processing not yet implemented, falling back to normal processing")
                pass
            # Chunked processing with vectorization and parallel processing
            if use_vectorization and parallel_processes > 1:
                # TODO: Implement parallel vectorized processing
                logger.warning("Parallel vectorized processing not yet implemented, falling back to normal processing")
                pass
            elif use_vectorization and not use_dask:
                # TODO: Implement vectorized processing without Dask
                logger.warning("Vectorized processing not yet implemented, falling back to normal processing")
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
            write_json(analysis_results, roles_path, encryption_key=encryption_key)

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
                write_dataframe_to_csv(entropy_df, entropy_path, encryption_key=encryption_key)
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

            write_json(sample_data, sample_path, encryption_key=encryption_key)
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

            self._create_visualizations(
                analysis_results,
                visualizations_dir,
                include_timestamp,
                result,
                reporter,
                **kwargs_encryption
            )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Creating visualizations"})

            # Process quasi-identifiers
            quasi_identifiers = analysis_results["column_groups"]["QUASI_IDENTIFIER"]

            if quasi_identifiers:
                result.add_metric("quasi_identifiers", quasi_identifiers)

                quasi_filename = get_timestamped_filename("quasi_identifiers", "json", include_timestamp)
                quasi_path = output_dir / quasi_filename

                write_json({"quasi_identifiers": quasi_identifiers}, quasi_path, encryption_key=encryption_key)
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
            logger.exception(f"Error in attribute profiling operation: {e}")

            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})
            if reporter:
                reporter.add_operation("Attribute Profiling",
                                    status="error",
                                    details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in attribute profiling: {str(e)}"
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
                    **kwargs
                )

                if not bar_result.startswith("Error"):
                    result.add_artifact("png", types_path, "Inferred data type distribution", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(types_path), "Inferred data type distribution")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}", exc_info=True)
            reporter.add_operation("Creating visualizations", status="warning",
                                   details={"warning": f"Error creating some visualizations: {str(e)}"})
        
    def _check_cache(
            self,
            data_source: DataSource,
            dataset_name: str = "main",
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
        dataset_name: str
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
            df = load_data_operation(data_source, dataset_name)
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
            artifacts: List[Any],
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
        if not self.use_cache:
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