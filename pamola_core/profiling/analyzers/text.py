"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Semantic Categorization Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides operations for analyzing and categorizing text fields in tabular datasets.
  It supports entity extraction, semantic categorization, clustering, and language analysis,
  with robust handling for large datasets and integration with the PAMOLA.CORE operation framework.

Key Features:
  - Entity extraction and semantic categorization for text fields
  - Configurable dictionary-based and NER-based categorization
  - Clustering of unresolved/unmatched text values
  - Language detection and text length statistics
  - Visualization generation for category, alias, and length distributions
  - Efficient chunked, parallel, and Dask-based processing for large datasets
  - Robust error handling, progress tracking, and operation logging
  - Caching and efficient repeated analysis
  - Integration with PAMOLA.CORE operation framework for standardized input/output
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Union, Tuple

import pandas as pd

from pamola_core.profiling.commons.text_utils import (
    analyze_language,
    analyze_null_and_empty,
    analyze_null_and_empty_in_chunks_joblib,
    analyze_null_and_empty_in_chunks_dask,
    calculate_length_stats,
    extract_text_and_ids,
    find_dictionary_file,
)
from pamola_core.profiling.schemas.text_schema import TextSemanticCategorizerOperationConfig
from pamola_core.utils.io import (
    ensure_directory,
    load_data_operation,
    load_settings_operation,
    write_json,
    write_dataframe_to_csv,
)
from pamola_core.utils.logging import get_logger
from pamola_core.utils.nlp.cache import get_cache
from pamola_core.utils.nlp.category_matching import (
    CategoryDictionary,
    analyze_hierarchy,
)
from pamola_core.utils.nlp.clustering import cluster_by_similarity
from pamola_core.utils.nlp.entity import create_entity_extractor
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_cache import OperationCache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import (
    create_pie_chart,
    create_bar_plot,
    plot_text_length_distribution,
)
from pamola_core.common.constants import Constants

# Configure logger
logger = get_logger(__name__)

# Get cache instances
file_cache = get_cache("file")
memory_cache = get_cache("memory")


@register(version="1.0.0")
class TextSemanticCategorizerOperation(FieldOperation):
    """
    Operation for categorizing text fields based on semantic content.

    This operation analyzes text fields and extracts semantic information,
    with support for multiple entity types, categorization, and clustering.
    """

    def __init__(
        self,
        field_name: str,
        id_field: Optional[str] = None,
        entity_type: str = "generic",
        dictionary_path: Optional[Union[str, Path]] = None,
        min_word_length: int = 3,
        clustering_threshold: float = 0.7,
        use_ner: bool = True,
        perform_categorization: bool = True,
        perform_clustering: bool = True,
        match_strategy: str = "specific_first",
        **kwargs,
    ):
        """
        Initialize the TextSemanticCategorizerOperation.

        Parameters
        ----------
        field_name : str
            Name of the field to analyze.
        id_field : str, optional
            Name of the ID field for record identification.
        entity_type : str
            Type of entities to extract ("job", "organization", "skill", "generic", etc.).
        dictionary_path : str or Path, optional
            Path to the semantic categories dictionary file.
        min_word_length : int
            Minimum length for words to include in token analysis.
        clustering_threshold : float
            Similarity threshold for clustering (0–1).
        use_ner : bool
            Whether to use Named Entity Recognition for uncategorized texts.
        perform_categorization : bool
            Whether to perform semantic categorization.
        perform_clustering : bool
            Whether to perform clustering for unmatched items.
        match_strategy : str
            Strategy for resolving category conflicts.
        **kwargs
            Additional arguments forwarded to FieldOperation.
        """
        # --- Default description ---
        kwargs.setdefault(
            "description",
            f"Semantic categorization of text field '{field_name}'",
        )

        # --- Build config object ---
        config = TextSemanticCategorizerOperationConfig(
            field_name=field_name,
            id_field=id_field,
            entity_type=entity_type,
            dictionary_path=dictionary_path,
            min_word_length=min_word_length,
            clustering_threshold=clustering_threshold,
            use_ner=use_ner,
            perform_categorization=perform_categorization,
            perform_clustering=perform_clustering,
            match_strategy=match_strategy,
            **kwargs,
        )

        # Inject config into kwargs
        kwargs["config"] = config

        # --- Initialize parent ---
        super().__init__(field_name=field_name, **kwargs)

        # --- Apply config attributes ---
        for key, value in config.to_dict().items():
            setattr(self, key, value)

        # --- Operation metadata ---
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
        Execute the text semantic categorization operation.

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
            # Initialize timing and result
            self.start_time = time.time()

            # Set logger if provided in kwargs
            if kwargs.get("logger"):
                self.logger = kwargs.get("logger")

            dirs = self._prepare_directories(task_dir)

            # Generate single timestamp for all artifacts
            operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Initialize operation cache
            self.operation_cache = OperationCache(
                cache_dir=task_dir / "cache",
            )

            # Save configuration
            self.save_config(task_dir)

            # Extract dataset name from kwargs (default to "main")
            dataset_name = kwargs.get("dataset_name", "main")

            logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, "
                f"strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Initialize result
            result = OperationResult(status=OperationStatus.SUCCESS)

            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(
                    1, {"step": "Initialization", "field": self.field_name}
                )

            # Get DataFrame from data source
            try:
                # Load data
                settings_operation = load_settings_operation(
                    data_source, dataset_name, **kwargs
                )
                df = load_data_operation(
                    data_source, dataset_name, **settings_operation
                )

                if df is None:
                    error_message = "Failed to load input data"
                    logger.error(error_message)
                    return OperationResult(
                        status=OperationStatus.ERROR, error_message=error_message
                    )
            except Exception as e:
                error_message = f"Error loading data: {str(e)}"
                logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    exception=e,
                )

            if progress_tracker:
                progress_tracker.update(2, {"step": "Data Loading"})

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame",
                )

            # Add operation to reporter
            reporter.add_operation(
                f"Semantic categorization of field: {self.field_name}",
                details={
                    "field_name": self.field_name,
                    "entity_type": self.entity_type,
                    "operation_type": "text_semantic_categorization",
                },
            )

            # Check for cached results if caching is enabled
            if self.use_cache and not self.force_recalculation:

                logger.info("Checking operation cache...")
                cached_result = self._check_cache(df, reporter, **kwargs)
                if cached_result:
                    logger.info(f"Using cached results for {self.field_name}")

                    # Update progress if tracker provided
                    if progress_tracker:
                        progress_tracker.update(
                            3, {"step": "Loaded from cache", "field": self.field_name}
                        )

                    return cached_result

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    3, {"step": "Basic text analysis", "field": self.field_name}
                )

            # Step 1: Perform basic text analysis (always executed)
            basic_analysis = self._perform_basic_analysis(
                df,
                self.field_name,
                self.chunk_size,
                self.use_dask,
                self.npartitions,
                self.use_vectorization,
                self.parallel_processes,
            )

            # Get text values and record IDs
            text_values, record_ids = extract_text_and_ids(
                df, self.field_name, self.id_field
            )

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    4, {"step": "Semantic categorization", "field": self.field_name}
                )

            # Initialize categorization results with defaults
            categorization_results = self._initialize_categorization_results(
                text_values
            )

            # Step 2: Perform categorization if requested
            if self.perform_categorization:
                # Load dictionary if categorization is needed
                if self.dictionary_path:
                    self.dictionary_path = find_dictionary_file(
                        self.dictionary_path,
                        self.entity_type,
                        task_dir,
                        logger,
                    )

                categorization_results = self._perform_semantic_categorization(
                    text_values,
                    record_ids,
                    self.dictionary_path,
                    basic_analysis["language_analysis"]["predominant_language"],
                    self.match_strategy,
                    self.use_ner,
                    self.perform_clustering,
                    self.clustering_threshold,
                    self.chunk_size,
                    self.use_dask,
                    self.npartitions,
                    self.use_vectorization,
                    self.parallel_processes,
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    5, {"step": "Creating artifacts", "field": self.field_name}
                )

            # Prepare complete analysis results
            analysis_results = self._compile_analysis_results(
                basic_analysis, categorization_results, self.field_name
            )

            # Save main artifacts
            analysis_result_path = self._save_main_artifacts(
                analysis_results, dirs, operation_timestamp, result, reporter
            )

            # Save categorization artifacts if categorization was performed
            categorization_result_paths = []
            if self.perform_categorization:
                categorization_result_paths = self._save_categorization_artifacts(
                    categorization_results,
                    record_ids,
                    text_values,
                    operation_timestamp,
                    dirs,
                    result,
                    reporter,
                    encryption_key=self.encryption_key,
                )

            # Generate visualizations
            visualization_paths = []
            if self.generate_visualization and self.visualization_backend is not None:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(5, {"step": "Generating visualizations"})

                try:
                    visualization_paths = self._handle_visualizations(
                        analysis_results=analysis_results,
                        visualizations_dir=dirs["visualizations"],
                        operation_timestamp=operation_timestamp,
                        result=result,
                        reporter=reporter,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        progress_tracker=progress_tracker,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    logger.error(error_message)
                    # Continue execution - visualization failure is not critical

            # Add metrics to result
            self._add_metrics_to_result(analysis_results, result)

            # Cache results if caching is enabled
            if self.use_cache:
                self._save_to_cache(
                    df=df,
                    analysis_results=analysis_results,
                    analysis_result_path=analysis_result_path,
                    categorization_result_paths=categorization_result_paths,
                    visualization_paths=visualization_paths,
                    task_dir=task_dir,
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    6, {"step": "Completed", "field": self.field_name}
                )

            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            return result
        except Exception as e:
            logger.exception(
                f"Error in text semantic categorization for {self.field_name}: {e}"
            )

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(1, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(
                f"Error categorizing {self.field_name}",
                status="error",
                details={"error": str(e)},
            )

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in semantic categorization of field {self.field_name}: {str(e)}",
                exception=e,
            )

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare standard directories for storing operation artifacts.

        Parameters:
        -----------
        task_dir : Path
            Base directory for the task

        Returns:
        --------
        Dict[str, Path]
            Dictionary with standard directory paths
        """
        # Create standard directories
        directories = {
            "output": task_dir / "output",
            "dictionaries": task_dir / "dictionaries",
            "visualizations": task_dir / "visualizations",
            "cache": task_dir / "cache",
        }

        # Ensure directories exist
        for dir_path in directories.values():
            ensure_directory(dir_path)

        return directories

    def _check_cache(
        self, df: pd.DataFrame, reporter: Any, **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation
        reporter : Any
            The reporter to log artifacts to
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
            logger.debug(f"Checking cache for key: {cache_key}")

            cached_data = self.operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if cached_data:
                logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Restore artifacts from cache
                artifacts_restored = 0

                # Add analysis result path
                analysis_result_path = cached_data.get("analysis_result_path")
                if analysis_result_path and isinstance(analysis_result_path, dict):
                    if Path(analysis_result_path.get("path")).exists():
                        artifacts_restored += 1
                        self._create_and_register_artifact(
                            artifact_type=analysis_result_path.get("artifact_type"),
                            path=analysis_result_path.get("path"),
                            description=analysis_result_path.get("description"),
                            result=cached_result,
                            reporter=reporter,
                            category=analysis_result_path.get("category"),
                        )

                # Add categorization result paths
                categorization_result_paths = cached_data.get(
                    "categorization_result_paths", []
                )
                for categorization_result_path in categorization_result_paths:
                    if categorization_result_path and isinstance(
                        categorization_result_path, dict
                    ):
                        if Path(categorization_result_path.get("path")).exists():
                            artifacts_restored += 1
                            self._create_and_register_artifact(
                                artifact_type=categorization_result_path.get(
                                    "artifact_type"
                                ),
                                path=categorization_result_path.get("path"),
                                description=categorization_result_path.get(
                                    "description"
                                ),
                                result=cached_result,
                                reporter=reporter,
                                category=categorization_result_path.get("category"),
                            )

                # Add categorization result paths
                visualization_paths = cached_data.get("visualization_paths", [])
                for visualization_path in visualization_paths:
                    if visualization_path and isinstance(visualization_path, dict):
                        if Path(
                            visualization_path.get("path")
                        ).exists() and visualization_path.get(
                            "is_create_and_register_artifact"
                        ):
                            artifacts_restored += 1
                            self._create_and_register_artifact(
                                artifact_type=visualization_path.get("artifact_type"),
                                path=visualization_path.get("path"),
                                description=visualization_path.get("description"),
                                result=cached_result,
                                reporter=reporter,
                                category=visualization_path.get("category"),
                            )

                # Add cached metrics to result
                analysis_results = cached_data.get("analysis_results")
                self._add_metrics_to_result(analysis_results, cached_result)

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )
                cached_result.add_metric("artifacts_restored", artifacts_restored)

                return cached_result

            logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        analysis_result_path: Dict[str, str],
        categorization_result_paths: List[Dict[str, str]],
        visualization_paths: List[Dict[str, Any]],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation
        analysis_results : dict
            Analysis results to cache
        analysis_result_path : dict
            Analysis result path
        categorization_result_paths : list of dict
            Categorization result paths
        visualization_paths : list of dict
            Visualizations paths
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
            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "analysis_results": analysis_results,
                "analysis_result_path": analysis_result_path,
                "categorization_result_paths": categorization_result_paths,
                "visualization_paths": visualization_paths,
                "data_info": {"df_length": len(df)},
            }

            # Save to cache
            logger.debug(f"Saving to cache with key: {cache_key}")
            success = self.operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                logger.info(f"Successfully saved results to cache")
            else:
                logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
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
        # Get operation parameters
        parameters = self._get_operation_parameters()

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(df)

        # Use the operation_cache utility to generate a consistent cache key
        return self.operation_cache.generate_cache_key(
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
            "field_name": self.field_name,
            "id_field": self.id_field,
            "entity_type": self.entity_type,
            "dictionary_path": self.dictionary_path,
            "min_word_length": self.min_word_length,
            "clustering_threshold": self.clustering_threshold,
            "use_ner": self.use_ner,
            "perform_categorization": self.perform_categorization,
            "perform_clustering": self.perform_clustering,
            "match_strategy": self.match_strategy,
            "version": self.version,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

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
            logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _perform_basic_analysis(
        self,
        df: pd.DataFrame,
        field_name: str,
        chunk_size: Optional[int] = 10000,
        use_dask: bool = False,
        npartitions: int = 2,
        use_vectorization: bool = False,
        parallel_processes: int = 2,
    ) -> Dict[str, Any]:
        """
        Perform basic analysis of text field.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the field
        field_name : str
            Name of the field to analyze
        chunk_size : int, optional
            Size of chunks for processing large datasets

        Returns:
        --------
        Dict[str, Any]
            Results of basic analysis
        """
        # Basic text analysis
        null_empty_analysis = {}
        flag_processed = False
        try:
            if not flag_processed and use_dask and npartitions > 1:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Dask")
                logger.info(f"Parallel Workers: {npartitions}")
                logger.info(f"Using dask processing with chunk size {chunk_size}")

                null_empty_analysis = analyze_null_and_empty_in_chunks_dask(
                    df, field_name, npartitions, chunk_size
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using dask processing: {e}")
            flag_processed = False

        try:
            if not flag_processed and use_vectorization and parallel_processes > 1:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Joblib")
                logger.info(f"Parallel Workers: {parallel_processes}")
                logger.info(f"Using vectorized processing with chunk size {chunk_size}")

                null_empty_analysis = analyze_null_and_empty_in_chunks_joblib(
                    df, field_name, parallel_processes, chunk_size
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using joblib processing: {e}")
            flag_processed = False

        try:
            if not flag_processed and len(df) > chunk_size:
                logger.info(f"Processing in chunks with chunk size {chunk_size}")
                total_chunks = (len(df) + chunk_size - 1) // chunk_size
                logger.info(f"Total chunks to process: {total_chunks}")

                null_empty_analysis = analyze_null_and_empty(df, field_name, chunk_size)

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using chunks processing: {e}")
            flag_processed = False

        try:
            if not flag_processed:
                logger.info("Fallback process as usual")

                null_empty_analysis = analyze_null_and_empty(df, field_name, chunk_size)

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in processing: {e}")
            flag_processed = False

        if not flag_processed:
            logger.exception(f"Error in processing")

        # Get text values
        text_values = df[field_name].astype("object").fillna("").astype(str).tolist()

        # Analyze language - sample to improve performance with large datasets
        max_texts_for_language = 1000  # Limit to avoid processing too many texts
        if len(text_values) > max_texts_for_language:
            import random

            random.seed(42)  # For reproducibility
            language_sample = random.sample(text_values, max_texts_for_language)
            language_analysis = analyze_language(language_sample)
        else:
            language_analysis = analyze_language(text_values)

        # Calculate length stats - can be compute-intensive for large datasets
        length_stats = calculate_length_stats(text_values, max_texts=chunk_size)

        return {
            "null_empty_analysis": null_empty_analysis,
            "language_analysis": language_analysis,
            "length_stats": length_stats,
        }

    def _initialize_categorization_results(
        self, text_values: List[str]
    ) -> Dict[str, Any]:
        """
        Initialize categorization results structure with defaults.

        Parameters:
        -----------
        text_values : List[str]
            List of text values

        Returns:
        --------
        Dict[str, Any]
            Default categorization results structure
        """
        return {
            "categorization": [],
            "summary": {
                "total_texts": len([t for t in text_values if t]),
                "num_matched": 0,
                "num_ner_matched": 0,
                "num_auto_clustered": 0,
                "num_unresolved": 0,
                "percentage_matched": 0,
                "top_categories": [],
                "top_aliases": [],
            },
            "category_distribution": {},
            "aliases_distribution": {},
            "hierarchy_analysis": {},
            "unresolved": [],
        }

    def _perform_semantic_categorization(
        self,
        text_values: List[str],
        record_ids: List[str],
        dictionary_path: Optional[Union[Path, str]],
        language: str,
        match_strategy: str,
        use_ner: bool,
        perform_clustering: bool,
        clustering_threshold: float,
        chunk_size: Optional[int] = None,
        use_dask: bool = False,
        npartitions: int = 2,
        use_vectorization: bool = False,
        parallel_processes: int = 2,
    ) -> Dict[str, Any]:
        """
        Perform semantic categorization using the entity extraction framework.

        Parameters:
        -----------
        text_values : List[str]
            List of text values to categorize
        record_ids : List[str]
            List of record IDs
        dictionary_path : Path, optional
            Path to the dictionary file
        language : str
            Language of the texts
        match_strategy : str
            Strategy for resolving conflicts
        use_ner : bool
            Whether to use NER for unmatched texts
        perform_clustering : bool
            Whether to perform clustering for unmatched texts
        clustering_threshold : float
            Similarity threshold for clustering

        Returns:
        --------
        Dict[str, Any]
            Categorization results
        """
        # Process in chunks if dataset is large
        if (
            (use_dask and npartitions > 1)
            or (use_vectorization and parallel_processes > 1)
            or (chunk_size and len(text_values) > chunk_size)
        ):
            return self._categorize_texts_in_chunks(
                text_values,
                record_ids,
                dictionary_path,
                language,
                match_strategy,
                use_ner,
                perform_clustering,
                clustering_threshold,
                chunk_size,
                use_dask,
                npartitions,
                use_vectorization,
                parallel_processes,
            )

        # Create entity extractor using the factory method
        extractor = create_entity_extractor(
            entity_type=self.entity_type,
            language=language,
            dictionary_path=str(dictionary_path) if dictionary_path else None,
            match_strategy=match_strategy,
            use_ner=use_ner,
        )

        # Extract entities
        extraction_results = extractor.extract_entities(
            texts=text_values, record_ids=record_ids
        )

        # Process extraction results
        categorization = []
        unresolved = []
        category_counts = {}
        alias_counts = {}

        # Track counts by match_method
        dictionary_matched = []
        ner_matched = []

        # Process matching results
        if "entities" in extraction_results:
            for match in extraction_results["entities"]:
                # Get matching match_method
                match_method = match.get("match_method", "unknown")
                record_id = match.get("record_id", "")

                # Track matched by match_method
                if match_method == "dictionary":
                    dictionary_matched.append(record_id)
                elif match_method == "ner":
                    ner_matched.append(record_id)

                # Update category and alias counts
                category = match.get("matched_category", "Unknown")
                alias = match.get("matched_alias", "unknown")

                category_counts[category] = category_counts.get(category, 0) + 1
                alias_counts[alias] = alias_counts.get(alias, 0) + 1

                # Add to categorization list
                categorization.append(match)

        # Get unresolved texts
        if "unresolved" in extraction_results:
            unmatched_texts = extraction_results["unresolved"]
            unmatched_ids = [item.get("record_id", "") for item in unmatched_texts]

            # Perform clustering on unmatched if requested
            if perform_clustering and unmatched_texts:
                # Extract just the text values
                unmatched_text_values = [
                    item.get("text", "") for item in unmatched_texts
                ]

                # Cluster unmatched texts
                clusters = cluster_by_similarity(
                    unmatched_text_values, clustering_threshold
                )

                # Track cluster matches
                cluster_matched = []

                # Process each cluster
                for cluster_label, cluster_indices in clusters.items():
                    # Create cluster category and alias
                    cluster_category = f"CLUSTER_{cluster_label}"
                    cluster_alias = f"cluster_{cluster_label.lower()}"

                    # Add each text in cluster to categorization
                    for i in cluster_indices:
                        if i < len(unmatched_ids):
                            record_id = unmatched_ids[i]
                            text = unmatched_text_values[i]

                            match_info = {
                                "record_id": record_id,
                                "original_text": text,
                                "normalized_text": text.lower().strip(),
                                "matched_category": cluster_category,
                                "matched_alias": cluster_alias,
                                "matched_domain": "Cluster",
                                "seniority": "Any",
                                "match_method": "cluster",
                                "score": 0.3,
                                "cluster_id": cluster_label,
                                "language": language,
                                "source_field": self.field_name,
                            }

                            # Add to categorization
                            categorization.append(match_info)
                            cluster_matched.append(record_id)

                            # Update category and alias counts
                            category_counts[cluster_category] = (
                                category_counts.get(cluster_category, 0) + 1
                            )
                            alias_counts[cluster_alias] = (
                                alias_counts.get(cluster_alias, 0) + 1
                            )

                # Add remaining unmatched to unresolved
                for i, item in enumerate(unmatched_texts):
                    record_id = item.get("record_id", "")
                    if record_id not in cluster_matched:
                        unresolved.append(item)
            else:
                # If not clustering, all unmatched go to unresolved
                unresolved = unmatched_texts

        # Get hierarchy analysis if available
        hierarchy_analysis = {}
        if dictionary_path:
            if isinstance(dictionary_path, str):
                dictionary_path = Path(dictionary_path)
            if dictionary_path.exists():
                try:
                    # Load dictionary to extract hierarchy
                    category_dict = CategoryDictionary.from_file(dictionary_path)
                    if category_dict.hierarchy:
                        hierarchy_analysis = analyze_hierarchy(category_dict.hierarchy)
                except Exception as e:
                    logger.warning(f"Error analyzing hierarchy: {e}")

        # Sort category and alias counts by frequency
        category_distribution = dict(
            sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        )

        aliases_distribution = dict(
            sorted(alias_counts.items(), key=lambda x: x[1], reverse=True)
        )

        # Prepare summary
        total_texts = len([t for t in text_values if t])
        num_matched = len(dictionary_matched)
        num_ner_matched = len(ner_matched)
        num_auto_clustered = len(
            set(
                [
                    m.get("record_id", "")
                    for m in categorization
                    if m.get("method", "") == "cluster"
                ]
            )
        )
        num_unresolved = len(unresolved)

        summary = {
            "total_texts": total_texts,
            "num_matched": num_matched,
            "num_ner_matched": num_ner_matched,
            "num_auto_clustered": num_auto_clustered,
            "num_unresolved": num_unresolved,
            "percentage_matched": (
                round((num_matched / total_texts) * 100, 2) if total_texts > 0 else 0
            ),
            "top_categories": (
                list(category_distribution.keys())[:5] if category_distribution else []
            ),
            "top_aliases": (
                list(aliases_distribution.keys())[:5] if aliases_distribution else []
            ),
        }

        return {
            "categorization": categorization,
            "unresolved": unresolved,
            "summary": summary,
            "category_distribution": category_distribution,
            "aliases_distribution": aliases_distribution,
            "hierarchy_analysis": hierarchy_analysis,
        }

    def _categorize_texts_in_chunks(
        self,
        text_values: List[str],
        record_ids: List[str],
        dictionary_path: Optional[Union[Path, str]],
        language: str,
        match_strategy: str,
        use_ner: bool,
        perform_clustering: bool,
        clustering_threshold: float,
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: int = 2,
        use_vectorization: bool = False,
        parallel_processes: int = 2,
    ) -> Dict[str, Any]:
        """
        Process large text datasets in chunks to improve memory efficiency.

        Parameters:
        -----------
        text_values : List[str]
            List of text values
        record_ids : List[str]
            List of record IDs
        dictionary_path : Path, optional
            Path to the dictionary file
        language : str
            Predominant language of the texts
        match_strategy : str
            Strategy for resolving category conflicts
        use_ner : bool
            Whether to use NER for unmatched texts
        perform_clustering : bool
            Whether to perform clustering for unmatched texts
        clustering_threshold : float
            Similarity threshold for clustering

        Returns:
        --------
        Dict[str, Any]
            Combined categorization results
        """
        import dask
        import joblib

        # Split texts and IDs into chunks
        chunks = []
        for i in range(0, len(text_values), self.chunk_size):
            end = min(i + self.chunk_size, len(text_values))
            chunks.append((text_values[i:end], record_ids[i:end]))

        logger.info(f"Processing {len(text_values)} texts in {len(chunks)} chunks")

        # For large datasets, process in chunks
        chunk_results = []
        flag_processed = False
        try:
            if not flag_processed and use_dask and npartitions > 1:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Dask")
                logger.info(f"Parallel Workers: {npartitions}")
                logger.info(f"Using dask processing with chunk size {chunk_size}")

                tasks = [
                    dask.delayed(self._perform_semantic_categorization)(
                        chunk_texts,
                        chunk_ids,
                        dictionary_path,
                        language,
                        match_strategy,
                        use_ner,
                        perform_clustering,
                        clustering_threshold,
                    )
                    for chunk_texts, chunk_ids in chunks
                ]

                chunk_results = dask.compute(*tasks, scheduler="processes")

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using dask processing: {e}")
            flag_processed = False

        try:
            if not flag_processed and use_vectorization and parallel_processes > 1:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Joblib")
                logger.info(f"Parallel Workers: {parallel_processes}")
                logger.info(f"Using vectorized processing with chunk size {chunk_size}")

                chunk_results = joblib.Parallel(n_jobs=-parallel_processes)(
                    joblib.delayed(self._perform_semantic_categorization)(
                        chunk_texts,
                        chunk_ids,
                        dictionary_path,
                        language,
                        match_strategy,
                        use_ner,
                        perform_clustering,
                        clustering_threshold,
                    )
                    for chunk_texts, chunk_ids in chunks
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using joblib processing: {e}")
            flag_processed = False

        try:
            if not flag_processed and len(text_values) > chunk_size:
                logger.info(f"Processing in chunks with chunk size {chunk_size}")
                total_chunks = (len(text_values) + chunk_size - 1) // chunk_size
                logger.info(f"Total chunks to process: {total_chunks}")

                for i, (chunk_texts, chunk_ids) in enumerate(chunks):
                    chunk_result = self._perform_semantic_categorization(
                        chunk_texts,
                        chunk_ids,
                        dictionary_path,
                        language,
                        match_strategy,
                        use_ner,
                        perform_clustering,
                        clustering_threshold,
                    )

                    chunk_results.append(chunk_result)

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using chunks processing: {e}")
            flag_processed = False

        try:
            if not flag_processed:
                logger.info("Fallback process as usual")

                chunk_result = self._perform_semantic_categorization(
                    text_values,
                    record_ids,
                    dictionary_path,
                    language,
                    match_strategy,
                    use_ner,
                    perform_clustering,
                    clustering_threshold,
                )
                chunk_results.append(chunk_result)

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in processing: {e}")
            flag_processed = False

        if not flag_processed:
            logger.exception(f"Error in processing")

        # Merge results
        merged_results = {
            "categorization": [],
            "unresolved": [],
            "summary": {
                "total_texts": 0,
                "num_matched": 0,
                "num_ner_matched": 0,
                "num_auto_clustered": 0,
                "num_unresolved": 0,
            },
            "category_distribution": {},
            "aliases_distribution": {},
            "hierarchy_analysis": (
                chunk_results[0]["hierarchy_analysis"] if chunk_results else {}
            ),
        }

        # Combine categorization and unresolved lists
        for result in chunk_results:
            merged_results["categorization"].extend(result["categorization"])
            merged_results["unresolved"].extend(result["unresolved"])

            # Combine summary metrics
            for key in [
                "total_texts",
                "num_matched",
                "num_ner_matched",
                "num_auto_clustered",
                "num_unresolved",
            ]:
                merged_results["summary"][key] += result["summary"][key]

            # Combine category and alias distributions
            for category, count in result["category_distribution"].items():
                merged_results["category_distribution"][category] = (
                    merged_results["category_distribution"].get(category, 0) + count
                )

            for alias, count in result["aliases_distribution"].items():
                merged_results["aliases_distribution"][alias] = (
                    merged_results["aliases_distribution"].get(alias, 0) + count
                )

        # Calculate percentage matched
        if merged_results["summary"]["total_texts"] > 0:
            merged_results["summary"]["percentage_matched"] = round(
                merged_results["summary"]["num_matched"]
                / merged_results["summary"]["total_texts"]
                * 100,
                2,
            )
        else:
            merged_results["summary"]["percentage_matched"] = 0

        # Sort category and alias distributions by frequency
        merged_results["category_distribution"] = dict(
            sorted(
                merged_results["category_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        merged_results["aliases_distribution"] = dict(
            sorted(
                merged_results["aliases_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Set top categories and aliases
        merged_results["summary"]["top_categories"] = list(
            merged_results["category_distribution"].keys()
        )[:5]
        merged_results["summary"]["top_aliases"] = list(
            merged_results["aliases_distribution"].keys()
        )[:5]

        return merged_results

    def _compile_analysis_results(
        self,
        basic_analysis: Dict[str, Any],
        categorization_results: Dict[str, Any],
        field_name: str,
    ) -> Dict[str, Any]:
        """
        Combine basic analysis and categorization results into a complete results object.

        Parameters:
        -----------
        basic_analysis : Dict[str, Any]
            Results of basic text analysis
        categorization_results : Dict[str, Any]
            Results of semantic categorization
        field_name : str
            Name of the analyzed field

        Returns:
        --------
        Dict[str, Any]
            Complete analysis results
        """
        # Create metrics dict for caching
        metrics = {
            "total_records": basic_analysis["null_empty_analysis"]["total_records"],
            "null_percentage": basic_analysis["null_empty_analysis"]["null_values"][
                "percentage"
            ],
            "empty_percentage": basic_analysis["null_empty_analysis"]["empty_strings"][
                "percentage"
            ],
            "avg_text_length": basic_analysis["length_stats"]["mean"],
            "max_text_length": basic_analysis["length_stats"]["max"],
            "predominant_language": basic_analysis["language_analysis"][
                "predominant_language"
            ],
        }

        # Add categorization metrics if available
        if "summary" in categorization_results:
            metrics.update(
                {
                    "num_matched": categorization_results["summary"]["num_matched"],
                    "num_ner_matched": categorization_results["summary"][
                        "num_ner_matched"
                    ],
                    "num_auto_clustered": categorization_results["summary"][
                        "num_auto_clustered"
                    ],
                    "num_unresolved": categorization_results["summary"][
                        "num_unresolved"
                    ],
                    "percentage_matched": categorization_results["summary"][
                        "percentage_matched"
                    ],
                }
            )

        # Combine all results
        analysis_results = {
            "field_name": field_name,
            "entity_type": self.entity_type,
            "null_empty_analysis": basic_analysis["null_empty_analysis"],
            "length_stats": basic_analysis["length_stats"],
            "language_analysis": basic_analysis["language_analysis"],
            "categorization": categorization_results.get("categorization", []),
            "match_summary": categorization_results.get("summary", {}),
            "category_distribution": categorization_results.get(
                "category_distribution", {}
            ),
            "aliases_distribution": categorization_results.get(
                "aliases_distribution", {}
            ),
            "hierarchy_analysis": categorization_results.get("hierarchy_analysis", {}),
            "unresolved_terms": categorization_results.get("unresolved", []),
            "metrics": metrics,  # Add metrics for caching
        }

        return analysis_results

    def _save_main_artifacts(
        self,
        analysis_results: Dict[str, Any],
        dirs: Dict[str, Path],
        operation_timestamp: Optional[str],
        result: OperationResult,
        reporter: Any,
        encryption_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save main analysis artifacts.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Analysis results to save
        dirs : Dict[str, Path]
            Directory paths for storing artifacts
        operation_timestamp : Optional[str]
            Timestamp for file naming
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to

        Returns:
        --------
        Dict[str, str]
            Information of artifact
        """
        # Save results as JSON
        results_filename = (
            f"{self.field_name}_text_semantic_analysis_{operation_timestamp}.json"
        )
        results_path = dirs["output"] / results_filename
        write_json(analysis_results, results_path, encryption_key=encryption_key)

        # Add artifact to result and reporter
        self._create_and_register_artifact(
            artifact_type="json",
            path=results_path,
            description=f"Semantic analysis of {self.field_name}",
            result=result,
            reporter=reporter,
            category=Constants.Artifact_Category_Output,
        )

        return {
            "artifact_type": "json",
            "path": results_path,
            "description": f"Semantic analysis of {self.field_name}",
            "category": Constants.Artifact_Category_Output,
        }

    def _save_categorization_artifacts(
        self,
        categorization_results: Dict[str, Any],
        record_ids: List[str],
        text_values: List[str],
        operation_timestamp: Optional[str],
        dirs: Dict[str, Path],
        result: OperationResult,
        reporter: Any,
        encryption_key: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Save artifacts specific to categorization.

        Parameters:
        -----------
        categorization_results : Dict[str, Any]
            Categorization results to save
        record_ids : List[str]
            List of record IDs
        text_values : List[str]
            List of text values
        operation_timestamp : Optional[str]
            Timestamp for file naming
        dirs : Dict[str, Path]
            Directory paths for storing artifacts
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to

        Returns:
        --------
        List[Dict[str, str]]
            Information of artifacts
        """
        categorization_result_paths = []

        # Save semantic roles mapping
        semantic_roles_filename = (
            f"{self.field_name}_semantic_roles_{operation_timestamp}.json"
        )
        semantic_roles_path = dirs["dictionaries"] / semantic_roles_filename
        write_json(
            categorization_results["categorization"],
            semantic_roles_path,
            encryption_key=encryption_key,
        )

        # Add semantic roles artifact
        self._create_and_register_artifact(
            artifact_type="json",
            path=semantic_roles_path,
            description=f"Semantic roles for {self.field_name}",
            result=result,
            reporter=reporter,
            category=Constants.Artifact_Category_Dictionary,
        )
        categorization_result_paths.append(
            {
                "artifact_type": "json",
                "path": semantic_roles_path,
                "description": f"Semantic roles for {self.field_name}",
                "category": Constants.Artifact_Category_Dictionary,
            }
        )

        # Save category mappings CSV
        category_mappings_filename = (
            f"{self.field_name}_category_mappings_{operation_timestamp}.csv"
        )
        category_mappings_path = dirs["dictionaries"] / category_mappings_filename

        # Create DataFrame for category mappings
        mapping_data = []
        for record_id, text_item in zip(record_ids, text_values):
            if not text_item:
                continue

            # Find categorization for this text
            match_info = next(
                (
                    item
                    for item in categorization_results["categorization"]
                    if item["record_id"] == record_id
                ),
                None,
            )

            if match_info:
                mapping_data.append(
                    {
                        "record_id": record_id,
                        "original_text": text_item,
                        "matched_alias": match_info.get("matched_alias", ""),
                        "matched_category": match_info.get("matched_category", ""),
                        "matched_domain": match_info.get("matched_domain", ""),
                        "match_method": match_info.get("method", ""),
                    }
                )

        # Convert to DataFrame and save
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            write_dataframe_to_csv(
                mapping_df, category_mappings_path, encryption_key=encryption_key
            )

            # Add category mappings artifact
            self._create_and_register_artifact(
                artifact_type="csv",
                path=category_mappings_path,
                description=f"Category mappings for {self.field_name}",
                result=result,
                reporter=reporter,
                category=Constants.Artifact_Category_Dictionary,
            )

            categorization_result_paths.append(
                {
                    "artifact_type": "csv",
                    "path": category_mappings_path,
                    "description": f"Category mappings for {self.field_name}",
                    "category": Constants.Artifact_Category_Dictionary,
                }
            )

        # Save unresolved terms CSV
        if categorization_results["unresolved"]:
            unresolved_filename = (
                f"{self.field_name}_unresolved_terms_{operation_timestamp}.csv"
            )
            unresolved_path = dirs["dictionaries"] / unresolved_filename

            # Convert to DataFrame and save
            unresolved_df = pd.DataFrame(categorization_results["unresolved"])
            write_dataframe_to_csv(
                unresolved_df, unresolved_path, encryption_key=encryption_key
            )

            # Add unresolved terms artifact
            self._create_and_register_artifact(
                artifact_type="csv",
                path=unresolved_path,
                description=f"Unresolved terms for {self.field_name}",
                result=result,
                reporter=reporter,
                category=Constants.Artifact_Category_Dictionary,
            )

            categorization_result_paths.append(
                {
                    "artifact_type": "csv",
                    "path": unresolved_path,
                    "description": f"Unresolved terms for {self.field_name}",
                    "category": Constants.Artifact_Category_Dictionary,
                }
            )

        return categorization_result_paths

    def _add_metrics_to_result(
        self, analysis_results: Dict[str, Any], result: OperationResult
    ) -> None:
        """
        Add metrics from analysis results to the operation result.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Analysis results containing metrics
        result : OperationResult
            Operation result to add metrics to
        """
        # Add from metrics if available (for cached results)
        if "metrics" in analysis_results:
            for key, value in analysis_results["metrics"].items():
                result.add_metric(key, value)
            return

        # Otherwise extract from analysis components
        result.add_metric(
            "total_records", analysis_results["null_empty_analysis"]["total_records"]
        )
        result.add_metric(
            "null_percentage",
            analysis_results["null_empty_analysis"]["null_values"]["percentage"],
        )
        result.add_metric(
            "empty_percentage",
            analysis_results["null_empty_analysis"]["empty_strings"]["percentage"],
        )
        result.add_metric("avg_text_length", analysis_results["length_stats"]["mean"])
        result.add_metric("max_text_length", analysis_results["length_stats"]["max"])
        result.add_metric(
            "predominant_language",
            analysis_results["language_analysis"]["predominant_language"],
        )

        # Add categorization metrics if available
        if "match_summary" in analysis_results:
            result.add_metric(
                "num_matched", analysis_results["match_summary"].get("num_matched", 0)
            )
            result.add_metric(
                "num_ner_matched",
                analysis_results["match_summary"].get("num_ner_matched", 0),
            )
            result.add_metric(
                "num_auto_clustered",
                analysis_results["match_summary"].get("num_auto_clustered", 0),
            )
            result.add_metric(
                "num_unresolved",
                analysis_results["match_summary"].get("num_unresolved", 0),
            )
            result.add_metric(
                "percentage_matched",
                analysis_results["match_summary"].get("percentage_matched", 0),
            )

    def _create_and_register_artifact(
        self,
        artifact_type: str,
        path: Path,
        description: str,
        result: OperationResult,
        reporter: Any,
        category: str = "",
    ) -> None:
        """
        Create and register an artifact in the result and reporter.

        Parameters:
        -----------
        artifact_type : str
            Type of artifact (e.g., "json", "csv", "png")
        path : Path
            Path to the artifact file
        description : str
            Description of the artifact
        result : OperationResult
            Operation result to add the artifact to
        reporter : Any
            Reporter to add the artifact to
        category : str, optional
            Category of the artifact
        """
        # Add to OperationResult
        result.add_artifact(
            artifact_type=artifact_type,
            path=path,
            description=description,
            category=category,
        )

        # Add to reporter
        reporter.add_artifact(artifact_type, str(path), description)

    def _handle_visualizations(
        self,
        analysis_results: Dict[str, Any],
        visualizations_dir: Path,
        operation_timestamp: Optional[str],
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
        vis_strict: bool,
        vis_timeout: int,
        progress_tracker: Optional[HierarchicalProgressTracker],
    ) -> List[Dict[str, Path]]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        visualizations_dir : Path
            Directory to save visualizations
        operation_timestamp : Optional[str]
            Timestamp for file naming
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
        List[Dict[str, Path]]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        logger.info(
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

                logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                logger.info(
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
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
                            logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations
                    visualization_paths = self._generate_visualizations(
                        analysis_results=analysis_results,
                        visualizations_dir=visualizations_dir,
                        operation_timestamp=operation_timestamp,
                        result=result,
                        reporter=reporter,
                        vis_theme=vis_theme,
                        vis_backend=vis_backend,
                        vis_strict=vis_strict,
                    )

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
                    logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
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

            logger.info(
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
                    logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = []
            elif visualization_error:
                logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = []
            else:
                total_time = time.time() - thread_start_time
                logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                for vis_result in visualization_paths:
                    logger.info(
                        f"[DIAG] Generated visualizations: {vis_result['path']}"
                    )
        except Exception as e:
            logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = []

        return visualization_paths

    def _generate_visualizations(
        self,
        analysis_results: Dict[str, Any],
        visualizations_dir: Path,
        operation_timestamp: Optional[str],
        result: OperationResult,
        reporter: Any,
        vis_strict: bool,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Generate visualizations for the text analysis.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        visualizations_dir : Path
            Directory to save visualizations
        operation_timestamp : Optional[str]
            Timestamp for file naming
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

        Returns:
        --------
        List[Dict[str, str]]
            Information of visualizations
        """
        visualization_paths = []

        # Generate category distribution pie chart
        if analysis_results.get("category_distribution"):
            visualization_path, is_create_and_register_artifact = (
                self._create_visualization(
                    data=analysis_results["category_distribution"],
                    vis_func=create_pie_chart,
                    filename=f"{self.field_name}_category_distribution",
                    title=f"Category Distribution: {self.field_name}",
                    output_dir=visualizations_dir,
                    operation_timestamp=operation_timestamp,
                    result=result,
                    reporter=reporter,
                    description=f"Category distribution for {self.field_name}",
                    vis_theme=vis_theme,
                    vis_backend=vis_backend,
                    vis_strict=vis_strict,
                    additional_params={
                        "show_percentages": True,
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    },
                )
            )
            visualization_paths.append(
                {
                    "artifact_type": "png",
                    "path": visualization_path,
                    "description": f"Category distribution for {self.field_name}",
                    "category": Constants.Artifact_Category_Visualization,
                    "is_create_and_register_artifact": is_create_and_register_artifact,
                }
            )

        # Generate alias distribution bar chart
        if analysis_results.get("aliases_distribution"):
            visualization_path, is_create_and_register_artifact = (
                self._create_visualization(
                    data=analysis_results["aliases_distribution"],
                    vis_func=create_bar_plot,
                    filename=f"{self.field_name}_alias_distribution",
                    title=f"Alias Distribution: {self.field_name}",
                    output_dir=visualizations_dir,
                    operation_timestamp=operation_timestamp,
                    result=result,
                    reporter=reporter,
                    description=f"Alias distribution for {self.field_name}",
                    vis_theme=vis_theme,
                    vis_backend=vis_backend,
                    vis_strict=vis_strict,
                    additional_params={
                        "orientation": "h",
                        "max_items": 15,
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    },
                )
            )
            visualization_paths.append(
                {
                    "artifact_type": "png",
                    "path": visualization_path,
                    "description": f"Alias distribution for {self.field_name}",
                    "category": Constants.Artifact_Category_Visualization,
                    "is_create_and_register_artifact": is_create_and_register_artifact,
                }
            )

        # Generate text length distribution
        if analysis_results.get("length_stats") and analysis_results[
            "length_stats"
        ].get("length_distribution"):
            visualization_path, is_create_and_register_artifact = (
                self._create_visualization(
                    data=analysis_results["length_stats"]["length_distribution"],
                    vis_func=plot_text_length_distribution,
                    filename=f"{self.field_name}_length_distribution",
                    title=f"Text Length Distribution: {self.field_name}",
                    output_dir=visualizations_dir,
                    operation_timestamp=operation_timestamp,
                    result=result,
                    reporter=reporter,
                    description=f"Length distribution for {self.field_name}",
                    vis_theme=vis_theme,
                    vis_backend=vis_backend,
                    vis_strict=vis_strict,
                    additional_params={
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    },
                )
            )
            visualization_paths.append(
                {
                    "artifact_type": "png",
                    "path": visualization_path,
                    "description": f"Length distribution for {self.field_name}",
                    "category": Constants.Artifact_Category_Visualization,
                    "is_create_and_register_artifact": is_create_and_register_artifact,
                }
            )

        return visualization_paths

    def _create_visualization(
        self,
        data: Dict[str, Any],
        vis_func: Callable,
        filename: str,
        title: str,
        output_dir: Path,
        operation_timestamp: Optional[str],
        result: OperationResult,
        reporter: Any,
        description: str,
        vis_strict: bool,
        vis_theme: Optional[str],
        vis_backend: Optional[str],
        additional_params: Dict[str, Any] = None,
    ) -> Tuple[str, bool]:
        """
        Create a visualization and register it as an artifact.

        Parameters:
        -----------
        data : Dict[str, Any]
            Data for the visualization
        vis_func : callable
            Visualization function to call
        filename : str
            Base filename for the visualization
        title : str
            Title for the visualization
        output_dir : Path
            Directory to save the visualization
        operation_timestamp : Optional[str]
            Timestamp for file naming
        result : OperationResult
            Operation result to add artifact to
        reporter : Any
            Reporter to add artifact to
        description : str
            Description of the visualization
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        additional_params : Dict[str, Any], optional
            Additional parameters for the visualization function

        Returns:
        --------
        Tuple[str, bool]
            Information of visualization
        """
        is_create_and_register_artifact = False

        output_filename = f"{filename}_{operation_timestamp}.png"
        output_path = output_dir / output_filename

        # Prepare parameters for the visualization function
        params = {
            "output_path": str(output_path),
            "title": title,
            "theme": vis_theme,
            "backend": vis_backend,
            "strict": vis_strict,
        }

        # Special handling for plot_text_length_distribution
        if vis_func.__name__ == "plot_text_length_distribution":
            params["length_data"] = data
        else:
            params["data"] = data

        # Add additional parameters if provided
        if additional_params:
            params.update(additional_params)

        # Create and save visualization
        vis_result = vis_func(**params)

        # Register artifact if visualization was successful
        if vis_result and (
            not isinstance(vis_result, str) or not vis_result.startswith("Error")
        ):
            is_create_and_register_artifact = True
            self._create_and_register_artifact(
                artifact_type="png",
                path=output_path,
                description=description,
                result=result,
                reporter=reporter,
                category=Constants.Artifact_Category_Visualization,
            )

        return str(output_path), is_create_and_register_artifact
