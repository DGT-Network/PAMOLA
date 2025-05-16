"""
Text semantic categorization operation for the project.

This module provides operations for analyzing and categorizing text fields
with support for entity extraction, semantic categorization, and clustering.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd

from pamola_core.profiling.commons.text_utils import (
    analyze_null_and_empty,
    calculate_length_stats
)
from pamola_core.utils.io import (
    ensure_directory,
    write_json,
    write_dataframe_to_csv,
    get_timestamped_filename,
    read_json,
    load_data_operation
)
from pamola_core.utils.logging import get_logger
from pamola_core.utils.nlp.cache import get_cache
from pamola_core.utils.nlp.category_matching import CategoryDictionary, analyze_hierarchy
from pamola_core.utils.nlp.clustering import cluster_by_similarity
from pamola_core.utils.nlp.entity import create_entity_extractor
from pamola_core.utils.nlp.language import detect_languages
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    create_pie_chart,
    create_bar_plot,
    plot_text_length_distribution
)

# Configure logger
logger = get_logger(__name__)

# Get cache instances
file_cache = get_cache('file')
memory_cache = get_cache('memory')


@register()
class TextSemanticCategorizerOperation(FieldOperation):
    """
    Operation for categorizing text fields based on semantic content.

    This operation analyzes text fields and extracts semantic information,
    with support for multiple entity types, categorization, and clustering.
    """

    def __init__(self,
                 field_name: str,
                 entity_type: str = "generic",
                 dictionary_path: Optional[Union[str, Path]] = None,
                 min_word_length: int = 3,
                 clustering_threshold: float = 0.7,
                 use_ner: bool = True,
                 perform_categorization: bool = True,
                 perform_clustering: bool = True,
                 match_strategy: str = "specific_first",
                 chunk_size: int = 10000,
                 use_cache: bool = True,
                 cache_dir: Optional[Path] = None,
                 include_timestamp: Any = None,
                 description: str = ""):
        """
        Initialize the text semantic categorizer operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to analyze
        entity_type : str
            Type of entities to extract ("job", "organization", "skill", "generic", etc.)
        dictionary_path : str or Path, optional
            Path to the semantic categories dictionary file
        min_word_length : int
            Minimum length for words to include in token analysis
        clustering_threshold : float
            Similarity threshold for clustering (0-1)
        use_ner : bool
            Whether to use Named Entity Recognition for uncategorized texts
        perform_categorization : bool
            Whether to perform semantic categorization
        perform_clustering : bool
            Whether to perform clustering for unmatched items
        match_strategy : str
            Strategy for resolving category conflicts:
            "specific_first" (default), "domain_prefer", "alias_only", "user_override"
        chunk_size : int
            Size of data chunks for processing large datasets
        use_cache : bool
            Whether to use caching for intermediate results
        cache_dir : Path, optional
            Directory to store cache files (defaults to task_dir/cache)
        description : str
            Description of the operation
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Semantic categorization of text field '{field_name}'"
        )
        self.entity_type = entity_type
        self.dictionary_path = dictionary_path
        self.min_word_length = min_word_length
        self.clustering_threshold = clustering_threshold
        self.use_ner = use_ner
        self.perform_categorization = perform_categorization
        self.perform_clustering = perform_clustering
        self.match_strategy = match_strategy
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.include_timestamp = include_timestamp

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
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
        progress_tracker : ProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for customizing the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Initialize result with success status
        global cache_key
        result = OperationResult(status=OperationStatus.SUCCESS)

        try:
            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(1, {"step": "Initialization", "field": self.field_name})

            # Prepare directories and execution parameters
            dirs = self._prepare_directories(task_dir)
            params = self._prepare_execution_parameters(kwargs, task_dir, dirs)

            # Get DataFrame from data source
            dataset_name = kwargs.get('dataset_name', "main")
            df = load_data_operation(data_source, dataset_name)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame"
                )

            # Add operation to reporter
            reporter.add_operation(f"Semantic categorization of field: {self.field_name}", details={
                "field_name": self.field_name,
                "entity_type": self.entity_type,
                "operation_type": "text_semantic_categorization"
            })

            # Check for cached results if caching is enabled
            if self.use_cache:
                cache_key = self._get_cache_key(data_source, task_dir)
                cached_result = self._load_cache(cache_key, self.cache_dir)
                if cached_result:
                    logger.info(f"Using cached results for {self.field_name}")

                    # Update progress if tracker provided
                    if progress_tracker:
                        progress_tracker.update(5, {"step": "Loaded from cache", "field": self.field_name})

                    # Generate visualizations from cached results
                    self._generate_visualizations(
                        cached_result,
                        dirs['visualizations'],
                        params["include_timestamp"],
                        result,
                        reporter
                    )

                    # Add metrics from cached result
                    for key, value in cached_result.get("metrics", {}).items():
                        result.add_metric(key, value)

                    return result

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Basic text analysis", "field": self.field_name})

            # Step 1: Perform basic text analysis (always executed)
            basic_analysis = self._perform_basic_analysis(df, self.field_name, params["chunk_size"])

            # Get text values and record IDs
            text_values, record_ids = self._extract_text_and_ids(df, self.field_name, params.get("id_field"))

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Semantic categorization", "field": self.field_name})

            # Initialize categorization results with defaults
            categorization_results = self._initialize_categorization_results(text_values)

            # Step 2: Perform categorization if requested
            if params["perform_categorization"]:
                # Load dictionary if categorization is needed
                dictionary_path = self._find_dictionary_file(self.entity_type, task_dir, dirs['dictionaries'])

                categorization_results = self._perform_semantic_categorization(
                    text_values,
                    record_ids,
                    dictionary_path,
                    basic_analysis["language_analysis"]["predominant_language"],
                    params["match_strategy"],
                    params["use_ner"],
                    params["perform_clustering"],
                    params["clustering_threshold"]
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Creating artifacts", "field": self.field_name})

            # Prepare complete analysis results
            analysis_results = self._compile_analysis_results(
                basic_analysis,
                categorization_results,
                self.field_name
            )

            # Cache results if caching is enabled
            if self.use_cache:
                self._save_cache(analysis_results, cache_key, self.cache_dir)

            # Save main artifacts
            self._save_main_artifacts(
                analysis_results,
                dirs,
                params["include_timestamp"],
                result,
                reporter
            )

            # Save categorization artifacts if categorization was performed
            if params["perform_categorization"]:
                self._save_categorization_artifacts(
                    categorization_results,
                    record_ids,
                    text_values,
                    dirs,
                    result,
                    reporter
                )

            # Generate visualizations
            self._generate_visualizations(
                analysis_results,
                dirs['visualizations'],
                params["include_timestamp"],
                result,
                reporter
            )

            # Add metrics to result
            self._add_metrics_to_result(analysis_results, result)

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Completed", "field": self.field_name})

            return result

        except Exception as e:
            logger.exception(f"Error in text semantic categorization for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(1, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error categorizing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in semantic categorization of field {self.field_name}: {str(e)}"
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

    def _prepare_execution_parameters(self, kwargs: Dict[str, Any], task_dir: Path, dirs: Dict[str, Path]) -> Dict[
        str, Any]:
        """
        Extract and validate parameters from kwargs with defaults from instance.

        Parameters:
        -----------
        kwargs : Dict[str, Any]
            Parameters passed to execute method
        task_dir : Path
            Task directory
        dirs : Dict[str, Path]
            Dictionary with standard directory paths

        Returns:
        --------
        Dict[str, Any]
            Dictionary with validated parameters
        """
        # Extract parameters from kwargs with defaults from instance
        params = {
            "dictionary_path": kwargs.get('dictionary_path', self.dictionary_path),
            "include_timestamp": kwargs.get('include_timestamp', self.include_timestamp),
            "min_word_length": kwargs.get('min_word_length', self.min_word_length),
            "clustering_threshold": kwargs.get('clustering_threshold', self.clustering_threshold),
            "use_ner": kwargs.get('use_ner', self.use_ner),
            "perform_categorization": kwargs.get('perform_categorization', self.perform_categorization),
            "perform_clustering": kwargs.get('perform_clustering', self.perform_clustering),
            "match_strategy": kwargs.get('match_strategy', self.match_strategy),
            "id_field": kwargs.get('id_field'),
            "chunk_size": kwargs.get('chunk_size', self.chunk_size),
            "use_cache": kwargs.get('use_cache', self.use_cache),
            "entity_type": kwargs.get('entity_type', self.entity_type)
        }

        # Set up cache directory if not provided
        if self.cache_dir is None and params["use_cache"]:
            params["cache_dir"] = dirs.get('cache', task_dir / "cache")
        else:
            params["cache_dir"] = self.cache_dir

        return params

    def _get_cache_key(self, data_source: DataSource, task_dir: Path) -> str:
        """
        Generate a unique cache key for the current operation and data.

        Parameters:
        -----------
        data_source : DataSource
            Data source used for the operation
        task_dir : Path
            Task directory

        Returns:
        --------
        str
            Unique cache key
        """
        import hashlib

        # Create a hash of the operation parameters and data source
        hasher = hashlib.md5()

        # Add operation parameters to hash
        params_str = (
            f"{self.field_name}:{self.entity_type}:{self.min_word_length}:{self.clustering_threshold}:"
            f"{self.use_ner}:{self.perform_categorization}:{self.perform_clustering}:"
            f"{self.match_strategy}:{self.dictionary_path}"
        )
        hasher.update(params_str.encode('utf-8'))

        # Add data source identifier to hash if available
        if hasattr(data_source, 'get_identifier'):
            source_id = data_source.get_identifier()
            if source_id:
                hasher.update(source_id.encode('utf-8'))

        # Generate a unique key for this operation and data
        return f"text_semantic_{self.field_name}_{self.entity_type}_{hasher.hexdigest()}"

    def _load_cache(self, cache_key: str, cache_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
        """
        Load cached results if available.

        Parameters:
        -----------
        cache_key : str
            Cache key for the results
        cache_dir : Path, optional
            Directory where cache files are stored

        Returns:
        --------
        Dict[str, Any] or None
            Cached results or None if not available
        """
        if not cache_dir or not cache_key:
            return None

        cache_file = cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            cached_data = read_json(cache_file)
            logger.info(f"Loaded cache from {cache_file}")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}")
            return None

    def _save_cache(self, results: Dict[str, Any], cache_key: str, cache_dir: Optional[Path]) -> bool:
        """
        Save results to cache.

        Parameters:
        -----------
        results : Dict[str, Any]
            Results to cache
        cache_key : str
            Cache key for the results
        cache_dir : Path, optional
            Directory where cache files should be stored

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not cache_dir or not cache_key:
            return False

        ensure_directory(cache_dir)
        cache_file = cache_dir / f"{cache_key}.json"

        try:
            write_json(results, cache_file)
            logger.info(f"Saved cache to {cache_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")
            return False

    def _extract_text_and_ids(self, df: pd.DataFrame, field_name: str, id_field: Optional[str]) -> Tuple[
        List[str], List[str]]:
        """
        Extract text values and record IDs from DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        field_name : str
            Name of the field containing text
        id_field : str, optional
            Name of the field containing record IDs

        Returns:
        --------
        Tuple[List[str], List[str]]
            Tuple of (text_values, record_ids)
        """
        # Get text values
        text_values = df[field_name].fillna("").astype(str).tolist()

        # Get record IDs (use specified ID field or fall back to index)
        if id_field and id_field in df.columns:
            record_ids = df[id_field].astype(str).tolist()
        else:
            record_ids = df.index.astype(str).tolist()

        return text_values, record_ids

    def _perform_basic_analysis(self, df: pd.DataFrame, field_name: str, chunk_size: Optional[int] = None) -> Dict[
        str, Any]:
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
        null_empty_analysis = analyze_null_and_empty(df, field_name, chunk_size)

        # Get text values
        text_values = df[field_name].fillna("").astype(str).tolist()

        # Analyze language - sample to improve performance with large datasets
        max_texts_for_language = 1000  # Limit to avoid processing too many texts
        if len(text_values) > max_texts_for_language:
            import random
            random.seed(42)  # For reproducibility
            language_sample = random.sample(text_values, max_texts_for_language)
            language_analysis = self._analyze_language(language_sample)
        else:
            language_analysis = self._analyze_language(text_values)

        # Calculate length stats - can be compute-intensive for large datasets
        length_stats = calculate_length_stats(text_values, max_texts=chunk_size)

        return {
            "null_empty_analysis": null_empty_analysis,
            "language_analysis": language_analysis,
            "length_stats": length_stats
        }

    def _analyze_language(self, text_values: List[str]) -> Dict[str, Any]:
        """
        Analyze language distribution in text values.

        Parameters:
        -----------
        text_values : List[str]
            List of text values

        Returns:
        --------
        Dict[str, Any]
            Language analysis results
        """
        # Use language detection from nlp module
        language_distribution = detect_languages(text_values)

        # Determine predominant language
        predominant_language = max(language_distribution.items(), key=lambda x: x[1])[
            0] if language_distribution else "unknown"

        return {
            "language_distribution": language_distribution,
            "predominant_language": predominant_language
        }

    def _find_dictionary_file(self, entity_type: str, task_dir: Path, dictionaries_dir: Path) -> Optional[Path]:
        """
        Find the appropriate dictionary file for the entity type.

        Search order:
        1. Explicitly provided dictionary path
        2. Task dictionaries directory
        3. Global data repository from config

        Parameters:
        -----------
        entity_type : str
            Type of entities to extract
        task_dir : Path
            Task directory
        dictionaries_dir : Path
            Dictionaries directory within task_dir

        Returns:
        --------
        Path or None
            Path to the dictionary file if found, None otherwise
        """
        # 1. Check explicitly provided path
        if self.dictionary_path:
            path = Path(self.dictionary_path)
            if path.exists():
                logger.info(f"Using provided dictionary: {path}")
                return path

        # 2. Check in task dictionaries directory
        task_dict_path = dictionaries_dir / f"{entity_type}.json"
        if task_dict_path.exists():
            logger.info(f"Using task dictionary: {task_dict_path}")
            return task_dict_path

        # 3. Check in global data repository from config
        try:
            # Try to find the PAMOLA.CORE config file
            config_paths = [
                Path("configs/prj_config.json"),  # Relative to working directory
                Path("D:/VK/_DEVEL/PAMOLA.CORE/configs/prj_config.json")  # Hardcoded path from example
            ]

            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break

            if config_file:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    data_repo = config.get("data_repository")
                    if data_repo:
                        repo_dict_path = Path(data_repo) / "external_dictionaries" / "ner" / f"{entity_type}.json"
                        if repo_dict_path.exists():
                            logger.info(f"Using repository dictionary: {repo_dict_path}")
                            return repo_dict_path
        except Exception as e:
            logger.warning(f"Error finding repository dictionary: {e}")

        logger.warning(f"No dictionary found for entity type '{entity_type}'. Using built-in fallback.")
        return None

    def _initialize_categorization_results(self, text_values: List[str]) -> Dict[str, Any]:
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
                "top_aliases": []
            },
            "category_distribution": {},
            "aliases_distribution": {},
            "hierarchy_analysis": {},
            "unresolved": []
        }

    def _perform_semantic_categorization(self,
                                         text_values: List[str],
                                         record_ids: List[str],
                                         dictionary_path: Optional[Path],
                                         language: str,
                                         match_strategy: str,
                                         use_ner: bool,
                                         perform_clustering: bool,
                                         clustering_threshold: float) -> Dict[str, Any]:
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
        if self.chunk_size and len(text_values) > self.chunk_size:
            return self._categorize_texts_in_chunks(
                text_values, record_ids, dictionary_path, language,
                match_strategy, use_ner, perform_clustering, clustering_threshold
            )

        # Create entity extractor using the factory method
        extractor = create_entity_extractor(
            entity_type=self.entity_type,
            language=language,
            dictionary_path=str(dictionary_path) if dictionary_path else None,
            match_strategy=match_strategy,
            use_ner=use_ner
        )

        # Extract entities
        extraction_results = extractor.extract_entities(
            texts=text_values,
            record_ids=record_ids
        )

        # Process extraction results
        categorization = []
        unresolved = []
        category_counts = {}
        alias_counts = {}

        # Track counts by method
        dictionary_matched = []
        ner_matched = []

        # Process matching results
        if "matches" in extraction_results:
            for match in extraction_results["matches"]:
                # Get matching method
                method = match.get("method", "unknown")
                record_id = match.get("record_id", "")

                # Track matched by method
                if method == "dictionary":
                    dictionary_matched.append(record_id)
                elif method == "ner":
                    ner_matched.append(record_id)

                # Update category and alias counts
                category = match.get("category", "Unknown")
                alias = match.get("alias", "unknown")

                category_counts[category] = category_counts.get(category, 0) + 1
                alias_counts[alias] = alias_counts.get(alias, 0) + 1

                # Add to categorization list
                categorization.append(match)

        # Get unresolved texts
        if "unmatched" in extraction_results:
            unmatched_texts = extraction_results["unmatched"]
            unmatched_ids = [item.get("record_id", "") for item in unmatched_texts]

            # Perform clustering on unmatched if requested
            if perform_clustering and unmatched_texts:
                # Extract just the text values
                unmatched_text_values = [item.get("text", "") for item in unmatched_texts]

                # Cluster unmatched texts
                clusters = cluster_by_similarity(unmatched_text_values, clustering_threshold)

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
                                "method": "cluster",
                                "score": 0.3,
                                "cluster_id": cluster_label,
                                "language": language,
                                "source_field": self.field_name
                            }

                            # Add to categorization
                            categorization.append(match_info)
                            cluster_matched.append(record_id)

                            # Update category and alias counts
                            category_counts[cluster_category] = category_counts.get(cluster_category, 0) + 1
                            alias_counts[cluster_alias] = alias_counts.get(cluster_alias, 0) + 1

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
        if dictionary_path and dictionary_path.exists():
            try:
                # Load dictionary to extract hierarchy
                category_dict = CategoryDictionary.from_file(dictionary_path)
                if category_dict.hierarchy:
                    hierarchy_analysis = analyze_hierarchy(category_dict.hierarchy)
            except Exception as e:
                logger.warning(f"Error analyzing hierarchy: {e}")

        # Sort category and alias counts by frequency
        category_distribution = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
        aliases_distribution = dict(sorted(alias_counts.items(), key=lambda x: x[1], reverse=True))

        # Prepare summary
        total_texts = len([t for t in text_values if t])
        num_matched = len(dictionary_matched)
        num_ner_matched = len(ner_matched)
        num_auto_clustered = len(set([m.get("record_id", "") for m in categorization
                                      if m.get("method", "") == "cluster"]))
        num_unresolved = len(unresolved)

        summary = {
            "total_texts": total_texts,
            "num_matched": num_matched,
            "num_ner_matched": num_ner_matched,
            "num_auto_clustered": num_auto_clustered,
            "num_unresolved": num_unresolved,
            "percentage_matched": round((num_matched / total_texts) * 100, 2) if total_texts > 0 else 0,
            "top_categories": list(category_distribution.keys())[:5] if category_distribution else [],
            "top_aliases": list(aliases_distribution.keys())[:5] if aliases_distribution else []
        }

        return {
            "categorization": categorization,
            "unresolved": unresolved,
            "summary": summary,
            "category_distribution": category_distribution,
            "aliases_distribution": aliases_distribution,
            "hierarchy_analysis": hierarchy_analysis
        }

    def _categorize_texts_in_chunks(self,
                                    text_values: List[str],
                                    record_ids: List[str],
                                    dictionary_path: Optional[Path],
                                    language: str,
                                    match_strategy: str,
                                    use_ner: bool,
                                    perform_clustering: bool,
                                    clustering_threshold: float) -> Dict[str, Any]:
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
        # Split texts and IDs into chunks
        chunks = []
        for i in range(0, len(text_values), self.chunk_size):
            end = min(i + self.chunk_size, len(text_values))
            chunks.append((
                text_values[i:end],
                record_ids[i:end]
            ))

        logger.info(f"Processing {len(text_values)} texts in {len(chunks)} chunks")

        # Process each chunk
        chunk_results = []
        for i, (chunk_texts, chunk_ids) in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk_texts)} texts)")

            # Process chunk
            chunk_result = self._perform_semantic_categorization(
                chunk_texts, chunk_ids, dictionary_path, language,
                match_strategy, use_ner, perform_clustering, clustering_threshold
            )

            chunk_results.append(chunk_result)

        # Merge results
        merged_results = {
            "categorization": [],
            "unresolved": [],
            "summary": {
                "total_texts": 0,
                "num_matched": 0,
                "num_ner_matched": 0,
                "num_auto_clustered": 0,
                "num_unresolved": 0
            },
            "category_distribution": {},
            "aliases_distribution": {},
            "hierarchy_analysis": chunk_results[0]["hierarchy_analysis"] if chunk_results else {}
        }

        # Combine categorization and unresolved lists
        for result in chunk_results:
            merged_results["categorization"].extend(result["categorization"])
            merged_results["unresolved"].extend(result["unresolved"])

            # Combine summary metrics
            for key in ["total_texts", "num_matched", "num_ner_matched", "num_auto_clustered", "num_unresolved"]:
                merged_results["summary"][key] += result["summary"][key]

            # Combine category and alias distributions
            for category, count in result["category_distribution"].items():
                merged_results["category_distribution"][category] = \
                    merged_results["category_distribution"].get(category, 0) + count

            for alias, count in result["aliases_distribution"].items():
                merged_results["aliases_distribution"][alias] = \
                    merged_results["aliases_distribution"].get(alias, 0) + count

        # Calculate percentage matched
        if merged_results["summary"]["total_texts"] > 0:
            merged_results["summary"]["percentage_matched"] = round(
                merged_results["summary"]["num_matched"] /
                merged_results["summary"]["total_texts"] * 100, 2
            )
        else:
            merged_results["summary"]["percentage_matched"] = 0

        # Sort category and alias distributions by frequency
        merged_results["category_distribution"] = dict(
            sorted(merged_results["category_distribution"].items(), key=lambda x: x[1], reverse=True)
        )
        merged_results["aliases_distribution"] = dict(
            sorted(merged_results["aliases_distribution"].items(), key=lambda x: x[1], reverse=True)
        )

        # Set top categories and aliases
        merged_results["summary"]["top_categories"] = list(merged_results["category_distribution"].keys())[:5]
        merged_results["summary"]["top_aliases"] = list(merged_results["aliases_distribution"].keys())[:5]

        return merged_results

    def _compile_analysis_results(self, basic_analysis: Dict[str, Any],
                                  categorization_results: Dict[str, Any],
                                  field_name: str) -> Dict[str, Any]:
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
            "null_percentage": basic_analysis["null_empty_analysis"]["null_values"]["percentage"],
            "empty_percentage": basic_analysis["null_empty_analysis"]["empty_strings"]["percentage"],
            "avg_text_length": basic_analysis["length_stats"]["mean"],
            "max_text_length": basic_analysis["length_stats"]["max"],
            "predominant_language": basic_analysis["language_analysis"]["predominant_language"]
        }

        # Add categorization metrics if available
        if "summary" in categorization_results:
            metrics.update({
                "num_matched": categorization_results["summary"]["num_matched"],
                "num_ner_matched": categorization_results["summary"]["num_ner_matched"],
                "num_auto_clustered": categorization_results["summary"]["num_auto_clustered"],
                "num_unresolved": categorization_results["summary"]["num_unresolved"],
                "percentage_matched": categorization_results["summary"]["percentage_matched"]
            })

        # Combine all results
        analysis_results = {
            "field_name": field_name,
            "entity_type": self.entity_type,
            "null_empty_analysis": basic_analysis["null_empty_analysis"],
            "length_stats": basic_analysis["length_stats"],
            "language_analysis": basic_analysis["language_analysis"],
            "categorization": categorization_results.get("categorization", []),
            "match_summary": categorization_results.get("summary", {}),
            "category_distribution": categorization_results.get("category_distribution", {}),
            "aliases_distribution": categorization_results.get("aliases_distribution", {}),
            "hierarchy_analysis": categorization_results.get("hierarchy_analysis", {}),
            "unresolved_terms": categorization_results.get("unresolved", []),
            "metrics": metrics  # Add metrics for caching
        }

        return analysis_results

    def _save_main_artifacts(self, analysis_results: Dict[str, Any],
                             dirs: Dict[str, Path],
                             include_timestamp: bool,
                             result: OperationResult,
                             reporter: Any) -> None:
        """
        Save main analysis artifacts.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Analysis results to save
        dirs : Dict[str, Path]
            Directory paths for storing artifacts
        include_timestamp : bool
            Whether to include timestamps in filenames
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        # Save results as JSON
        results_filename = get_timestamped_filename(
            f"{self.field_name}_text_semantic_analysis",
            extension="json",
            include_timestamp=include_timestamp
        )
        results_path = dirs["output"] / results_filename
        write_json(analysis_results, results_path)

        # Add artifact to result and reporter
        self._create_and_register_artifact(
            artifact_type="json",
            path=results_path,
            description=f"Semantic analysis of {self.field_name}",
            result=result,
            reporter=reporter
        )

    def _save_categorization_artifacts(self, categorization_results: Dict[str, Any],
                                       record_ids: List[str],
                                       text_values: List[str],
                                       dirs: Dict[str, Path],
                                       result: OperationResult,
                                       reporter: Any) -> None:
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
        dirs : Dict[str, Path]
            Directory paths for storing artifacts
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        # Save semantic roles mapping
        semantic_roles_filename = f"{self.field_name}_semantic_roles.json"
        semantic_roles_path = dirs["dictionaries"] / semantic_roles_filename
        write_json(categorization_results["categorization"], semantic_roles_path)

        # Add semantic roles artifact
        self._create_and_register_artifact(
            artifact_type="json",
            path=semantic_roles_path,
            description=f"Semantic roles for {self.field_name}",
            result=result,
            reporter=reporter,
            category="dictionary"
        )

        # Save category mappings CSV
        category_mappings_filename = f"{self.field_name}_category_mappings.csv"
        category_mappings_path = dirs["dictionaries"] / category_mappings_filename

        # Create DataFrame for category mappings
        mapping_data = []
        for record_id, text_item in zip(record_ids, text_values):
            if not text_item:
                continue

            # Find categorization for this text
            match_info = next((item for item in categorization_results["categorization"]
                               if item["record_id"] == record_id), None)

            if match_info:
                mapping_data.append({
                    "record_id": record_id,
                    "original_text": text_item,
                    "matched_alias": match_info.get("matched_alias", ""),
                    "matched_category": match_info.get("matched_category", ""),
                    "matched_domain": match_info.get("matched_domain", ""),
                    "match_method": match_info.get("method", "")
                })

        # Convert to DataFrame and save
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            write_dataframe_to_csv(mapping_df, category_mappings_path)

            # Add category mappings artifact
            self._create_and_register_artifact(
                artifact_type="csv",
                path=category_mappings_path,
                description=f"Category mappings for {self.field_name}",
                result=result,
                reporter=reporter,
                category="dictionary"
            )

        # Save unresolved terms CSV
        if categorization_results["unresolved"]:
            unresolved_filename = f"{self.field_name}_unresolved_terms.csv"
            unresolved_path = dirs["dictionaries"] / unresolved_filename

            # Convert to DataFrame and save
            unresolved_df = pd.DataFrame(categorization_results["unresolved"])
            write_dataframe_to_csv(unresolved_df, unresolved_path)

            # Add unresolved terms artifact
            self._create_and_register_artifact(
                artifact_type="csv",
                path=unresolved_path,
                description=f"Unresolved terms for {self.field_name}",
                result=result,
                reporter=reporter,
                category="dictionary"
            )

    def _add_metrics_to_result(self, analysis_results: Dict[str, Any], result: OperationResult) -> None:
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
        result.add_metric("total_records", analysis_results["null_empty_analysis"]["total_records"])
        result.add_metric("null_percentage", analysis_results["null_empty_analysis"]["null_values"]["percentage"])
        result.add_metric("empty_percentage", analysis_results["null_empty_analysis"]["empty_strings"]["percentage"])
        result.add_metric("avg_text_length", analysis_results["length_stats"]["mean"])
        result.add_metric("max_text_length", analysis_results["length_stats"]["max"])
        result.add_metric("predominant_language", analysis_results["language_analysis"]["predominant_language"])

        # Add categorization metrics if available
        if "match_summary" in analysis_results:
            result.add_metric("num_matched", analysis_results["match_summary"].get("num_matched", 0))
            result.add_metric("num_ner_matched", analysis_results["match_summary"].get("num_ner_matched", 0))
            result.add_metric("num_auto_clustered", analysis_results["match_summary"].get("num_auto_clustered", 0))
            result.add_metric("num_unresolved", analysis_results["match_summary"].get("num_unresolved", 0))
            result.add_metric("percentage_matched", analysis_results["match_summary"].get("percentage_matched", 0))

    def _create_and_register_artifact(self, artifact_type: str, path: Path, description: str,
                                      result: OperationResult, reporter: Any, category: str = "") -> None:
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
            category=category
        )

        # Add to reporter
        reporter.add_artifact(
            artifact_type,
            str(path),
            description
        )

    def _generate_visualizations(self,
                                 analysis_results: Dict[str, Any],
                                 visualizations_dir: Path,
                                 include_timestamp: bool,
                                 result: OperationResult,
                                 reporter: Any):
        """
        Generate visualizations for the text analysis.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        visualizations_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        # Generate category distribution pie chart
        if analysis_results.get("category_distribution"):
            self._create_visualization(
                data=analysis_results["category_distribution"],
                vis_func=create_pie_chart,
                filename=f"{self.field_name}_category_distribution",
                title=f"Category Distribution: {self.field_name}",
                output_dir=visualizations_dir,
                include_timestamp=include_timestamp,
                result=result,
                reporter=reporter,
                description=f"Category distribution for {self.field_name}",
                additional_params={"show_percentages": True}
            )

        # Generate alias distribution bar chart
        if analysis_results.get("aliases_distribution"):
            self._create_visualization(
                data=analysis_results["aliases_distribution"],
                vis_func=create_bar_plot,
                filename=f"{self.field_name}_alias_distribution",
                title=f"Alias Distribution: {self.field_name}",
                output_dir=visualizations_dir,
                include_timestamp=include_timestamp,
                result=result,
                reporter=reporter,
                description=f"Alias distribution for {self.field_name}",
                additional_params={"orientation": "h", "max_items": 15}
            )

        # Generate text length distribution
        if analysis_results.get("length_stats") and analysis_results["length_stats"].get("length_distribution"):
            self._create_visualization(
                data=analysis_results["length_stats"]["length_distribution"],
                vis_func=plot_text_length_distribution,
                filename=f"{self.field_name}_length_distribution",
                title=f"Text Length Distribution: {self.field_name}",
                output_dir=visualizations_dir,
                include_timestamp=include_timestamp,
                result=result,
                reporter=reporter,
                description=f"Length distribution for {self.field_name}"
            )

    def _create_visualization(self, data: Dict[str, Any], vis_func: callable,
                              filename: str, title: str, output_dir: Path,
                              include_timestamp: bool, result: OperationResult,
                              reporter: Any, description: str, additional_params: Dict[str, Any] = None):
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
        include_timestamp : bool
            Whether to include timestamp in the filename
        result : OperationResult
            Operation result to add artifact to
        reporter : Any
            Reporter to add artifact to
        description : str
            Description of the visualization
        additional_params : Dict[str, Any], optional
            Additional parameters for the visualization function
        """
        # Get timestamped filename
        output_filename = get_timestamped_filename(
            filename,
            extension="png",
            include_timestamp=include_timestamp
        )
        output_path = output_dir / output_filename

        # Prepare parameters for the visualization function
        params = {
            "data": data,
            "output_path": str(output_path),
            "title": title
        }

        # Add additional parameters if provided
        if additional_params:
            params.update(additional_params)

        # Create and save visualization
        vis_result = vis_func(**params)

        # Register artifact if visualization was successful
        if vis_result and (not isinstance(vis_result, str) or not vis_result.startswith("Error")):
            self._create_and_register_artifact(
                artifact_type="png",
                path=output_path,
                description=description,
                result=result,
                reporter=reporter,
                category="visualization"
            )