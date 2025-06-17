"""
Base classes for entity extraction.

This module provides abstract base classes and common utilities for
all entity extractors in the package.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from pamola_core.utils.nlp.base import normalize_language_code
from pamola_core.utils.nlp.cache import get_cache
from pamola_core.utils.nlp.category_matching import CategoryDictionary
from pamola_core.utils.nlp.language import detect_language
from pamola_core.utils.nlp.model_manager import NLPModelManager

# Configure logger
logger = logging.getLogger(__name__)

# Get cache instances
file_cache = get_cache('file')
memory_cache = get_cache('memory')
model_cache = get_cache('model')

# NLP model manager instance
nlp_model_manager = NLPModelManager()


def get_dictionaries_path() -> Path:
    """
    Get the path to the entity dictionaries directory.

    Tries to find the path in the following order:
    1. PAMOLA_ENTITIES_DIR environment variable
    2. Data repository from PAMOLA.CORE config under external_dictionaries/entities
    3. Default path in package resources

    Returns:
    --------
    Path
        Path to the entity dictionaries directory
    """
    # Check environment variable
    env_dir = os.environ.get('PAMOLA_ENTITIES_DIR')
    if env_dir and os.path.exists(env_dir):
        return Path(env_dir)

    # Check for PAMOLA.CORE config
    try:
        # Try to determine the project root
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        config_path = os.path.join(current_dir, 'configs', 'prj_config.json')

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if "data_repository" in config:
                data_repo = config["data_repository"]
                entities_dir = os.path.join(data_repo, 'external_dictionaries', 'entities')

                if os.path.exists(entities_dir):
                    return Path(entities_dir)

                # Create the directory if it doesn't exist
                os.makedirs(entities_dir, exist_ok=True)
                return Path(entities_dir)
    except Exception as e:
        logger.warning(f"Error reading PAMOLA.CORE config: {e}")

    # Default path in package resources
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_dir = os.path.join(package_dir, 'resources', 'entities')
    os.makedirs(default_dir, exist_ok=True)
    return Path(default_dir)


def find_dictionary_file(entity_type: str, language: Optional[str] = None) -> Optional[str]:
    """
    Find the dictionary file for a specific entity type.

    Parameters:
    -----------
    entity_type : str
        Type of entity (job, organization, skill, transaction)
    language : str, optional
        Language code to find a language-specific dictionary

    Returns:
    --------
    str or None
        Path to the dictionary file if found, None otherwise
    """
    # Get the dictionaries directory
    dictionaries_dir = get_dictionaries_path()

    # Build filename patterns
    patterns = []

    # Add language-specific patterns if language is provided
    if language:
        lang_code = normalize_language_code(language)
        patterns.extend([
            f"{entity_type}_{lang_code}.json",
            f"{entity_type}_{lang_code}_map.json",
            f"{lang_code}_{entity_type}_map.json",
            f"{lang_code}_{entity_type}.json"
        ])

    # Add generic patterns
    patterns.extend([
        f"{entity_type}_map.json",
        f"{entity_type}.json"
    ])

    # Check each pattern
    for pattern in patterns:
        file_path = dictionaries_dir / pattern
        if file_path.exists():
            return str(file_path)

    # Check for the common/default dictionary
    default_path = dictionaries_dir / "entities_map.json"
    if default_path.exists():
        return str(default_path)

    return None


class EntityMatchResult:
    """
    Class representing a single entity match result.
    """

    def __init__(
            self,
            original_text: str,
            normalized_text: str,
            category: Optional[str] = None,
            alias: Optional[str] = None,
            domain: Optional[str] = None,
            level: Optional[int] = None,
            seniority: Optional[str] = None,
            confidence: float = 0.0,
            method: str = "unknown",
            language: str = "unknown",
            conflicts: Optional[List[str]] = None,
            record_id: Optional[str] = None
    ):
        """
        Initialize an entity match result.

        Parameters:
        -----------
        original_text : str
            Original text that was matched
        normalized_text : str
            Normalized text (lowercase, no punctuation, etc.)
        category : str, optional
            Matched category name
        alias : str, optional
            Category alias (for replacement)
        domain : str, optional
            Domain of the category
        level : int, optional
            Hierarchy level of the category
        seniority : str, optional
            Seniority level if applicable
        confidence : float
            Confidence score of the match (0-1)
        method : str
            Method used for matching (dictionary, ner, cluster)
        language : str
            Detected language of the text
        conflicts : List[str], optional
            List of conflicting categories
        record_id : str, optional
            ID of the record if available
        """
        self.original_text = original_text
        self.normalized_text = normalized_text
        self.category = category
        self.alias = alias or (category.lower().replace(" ", "_") if category else None)
        self.domain = domain or "General"
        self.level = level or 0
        self.seniority = seniority or "Any"
        self.confidence = confidence
        self.method = method
        self.language = language
        self.conflicts = conflicts or []
        self.record_id = record_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the match result to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            Dictionary representation of the match result
        """
        result = {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "matched_category": self.category,
            "matched_alias": self.alias,
            "matched_domain": self.domain,
            "matched_level": self.level,
            "matched_seniority": self.seniority,
            "match_confidence": self.confidence,
            "match_method": self.method,
            "language_detected": self.language
        }

        if self.record_id:
            result["record_id"] = self.record_id

        if self.conflicts:
            result["conflict_candidates"] = self.conflicts

        return result


class BaseEntityExtractor(ABC):
    """
    Abstract base class for all entity extractors.
    """

    def __init__(
            self,
            language: str = "auto",
            dictionary_path: Optional[str] = None,
            match_strategy: str = "specific_first",
            use_ner: bool = True,
            min_confidence: float = 0.5,
            use_cache: bool = True,
            **kwargs
    ):
        """
        Initialize the entity extractor.

        Parameters:
        -----------
        language : str
            Language code or "auto" for detection
        dictionary_path : str, optional
            Path to the dictionary file
        match_strategy : str
            Strategy for resolving matches
        use_ner : bool
            Whether to use NER models if dictionary match fails
        min_confidence : float
            Minimum confidence threshold for entity recognition
        use_cache : bool
            Whether to use caching
        **kwargs
            Additional parameters for specific implementations
        """
        self.language = language
        self.dictionary_path = dictionary_path
        self.match_strategy = match_strategy
        self.use_ner = use_ner
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        self.category_dictionary = None
        self.hierarchy = None

        # Extra parameters
        self.extra_params = kwargs

        # Load dictionary if provided
        if dictionary_path:
            self.load_dictionary(dictionary_path)

    def load_dictionary(self, dictionary_path: str) -> bool:
        """
        Load the entity dictionary from the specified path.

        Parameters:
        -----------
        dictionary_path : str
            Path to the dictionary file

        Returns:
        --------
        bool
            True if dictionary was loaded successfully, False otherwise
        """
        try:
            # Try to load dictionary with cache
            cache_key = f"entity_dict:{dictionary_path}"

            if self.use_cache:
                dict_obj = file_cache.get(cache_key)
                if dict_obj is not None:
                    self.category_dictionary = dict_obj.dictionary
                    self.hierarchy = dict_obj.hierarchy
                    return True

            # Load from file if not in cache
            dict_obj = CategoryDictionary.from_file(dictionary_path)

            # Store in instance
            self.category_dictionary = dict_obj.dictionary
            self.hierarchy = dict_obj.hierarchy

            # Store in cache
            if self.use_cache:
                file_cache.set(cache_key, dict_obj, file_path=dictionary_path)

            logger.info(f"Loaded entity dictionary from {dictionary_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading dictionary from {dictionary_path}: {e}")
            return False

    def ensure_dictionary_loaded(self, entity_type: str) -> bool:
        """
        Ensure that a dictionary is loaded, trying to find a suitable one if needed.

        Parameters:
        -----------
        entity_type : str
            Type of entity for finding an appropriate dictionary

        Returns:
        --------
        bool
            True if a dictionary is loaded, False otherwise
        """
        # If dictionary is already loaded, return True
        if self.category_dictionary is not None:
            return True

        # Try to load from provided path
        if self.dictionary_path and os.path.exists(self.dictionary_path):
            return self.load_dictionary(self.dictionary_path)

        # Try to find a suitable dictionary
        dict_path = find_dictionary_file(entity_type, self.language)
        if dict_path:
            self.dictionary_path = dict_path
            return self.load_dictionary(dict_path)

        logger.warning(f"No dictionary found for entity type '{entity_type}' and language '{self.language}'")
        return False

    def extract_entities(self, texts: List[str], record_ids: Optional[List[str]] = None,
                         show_progress: bool = False) -> Dict[str, Any]:
        """
        Extract entities from a list of texts.

        Parameters:
        -----------
        texts : List[str]
            List of text strings to process
        record_ids : List[str], optional
            List of record IDs corresponding to the texts
        show_progress : bool
            Whether to show a progress bar

        Returns:
        --------
        Dict[str, Any]
            Extraction results containing entities, categories, and statistics
        """
        # Check for empty input
        if not texts:
            return self._empty_result()

        # Ensure record IDs are available
        if record_ids is None:
            record_ids = [str(i) for i in range(len(texts))]
        elif len(record_ids) != len(texts):
            logger.warning(f"Length mismatch: {len(texts)} texts but {len(record_ids)} record IDs")
            # Ensure lengths match by truncating or extending
            if len(record_ids) < len(texts):
                record_ids = record_ids + [str(i + len(record_ids)) for i in range(len(texts) - len(record_ids))]
            else:
                record_ids = record_ids[:len(texts)]

        # Detect language if auto
        if self.language == "auto" and texts:
            language = detect_language(texts[0])
        else:
            language = normalize_language_code(self.language)

        # Ensure dictionary is loaded
        self.ensure_dictionary_loaded(self._get_entity_type())

        # Initialize results
        entity_matches = []
        unresolved = []

        # Process each text
        iterator = tqdm(zip(texts, record_ids), total=len(texts), desc="Extracting entities") if show_progress else zip(
            texts, record_ids)

        for text, record_id in iterator:
            if not text:
                continue

            # Process the text
            match_result = self._process_text(text, record_id, language)

            # Store results
            if match_result and match_result.category:
                entity_matches.append(match_result)
            else:
                unresolved.append({
                    "record_id": record_id,
                    "text": text,
                    "normalized_text": text.lower().strip() if text else "",
                    "language": language
                })

        # Compile and return results
        return self._compile_results(entity_matches, unresolved, language)

    def _process_text(self, text: str, record_id: Optional[str] = None, language: str = "en") -> Optional[
        EntityMatchResult]:
        """
        Process a single text and extract entities.

        Parameters:
        -----------
        text : str
            Text to process
        record_id : str, optional
            ID of the record
        language : str
            Language of the text

        Returns:
        --------
        EntityMatchResult or None
            Match result if found, None otherwise
        """
        # Skip empty texts
        if not text:
            return None

        # Normalize text
        normalized_text = text.lower().strip()

        # Step 1: Try dictionary matching if dictionary is available
        if self.category_dictionary:
            cat_dict = CategoryDictionary(self.category_dictionary, self.hierarchy)
            category, score, conflicts = cat_dict.get_best_match(normalized_text, self.match_strategy)

            if category and score >= self.min_confidence:
                # Get category info
                category_info = cat_dict.get_category_info(category)

                # Create match result
                return EntityMatchResult(
                    original_text=text,
                    normalized_text=normalized_text,
                    category=category,
                    alias=category_info.get("alias"),
                    domain=category_info.get("domain"),
                    level=category_info.get("level"),
                    seniority=category_info.get("seniority"),
                    confidence=score,
                    method="dictionary",
                    language=language,
                    conflicts=conflicts,
                    record_id=record_id
                )

        # Step 2: Try NER if enabled and dictionary match failed
        if self.use_ner:
            ner_result = self._extract_with_ner(text, normalized_text, language)
            if ner_result:
                ner_result.record_id = record_id
                return ner_result

        # No match found
        return None

    @abstractmethod
    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        """
        Extract entities using NER models.

        Parameters:
        -----------
        text : str
            Original text
        normalized_text : str
            Normalized text
        language : str
            Language of the text

        Returns:
        --------
        EntityMatchResult or None
            Match result if found, None otherwise
        """
        pass

    @abstractmethod
    def _get_entity_type(self) -> str:
        """
        Get the entity type for this extractor.

        Returns:
        --------
        str
            Entity type string
        """
        pass

    def _empty_result(self) -> Dict[str, Any]:
        """
        Create an empty result structure.

        Returns:
        --------
        Dict[str, Any]
            Empty result dictionary
        """
        return {
            "entities": [],
            "unresolved": [],
            "summary": {
                "total_texts": 0,
                "matched_count": 0,
                "dictionary_matches": 0,
                "ner_matches": 0,
                "unresolved_count": 0,
                "match_percentage": 0.0
            },
            "category_distribution": {},
            "domain_distribution": {},
            "method_distribution": {}
        }

    @staticmethod
    def _compile_results(entity_matches: List[EntityMatchResult],
                         unresolved: List[Dict[str, Any]],
                         language: str) -> Dict[str, Any]:
        """
        Compile the final results from the matches and unresolved texts.

        Parameters:
        -----------
        entity_matches : List[EntityMatchResult]
            List of entity match results
        unresolved : List[Dict[str, Any]]
            List of unresolved texts
        language : str
            Language used for processing

        Returns:
        --------
        Dict[str, Any]
            Comprehensive results dictionary
        """
        # Count dictionary and NER matches
        dictionary_matches = sum(1 for match in entity_matches if match.method == "dictionary")
        ner_matches = sum(1 for match in entity_matches if match.method == "ner")

        # Get total text count (matched + unresolved)
        total_texts = len(entity_matches) + len(unresolved)

        # Calculate match percentage
        match_percentage = (len(entity_matches) / total_texts * 100) if total_texts > 0 else 0.0

        # Count categories and domains
        category_counts = {}
        domain_counts = {}
        method_counts = {}

        for match in entity_matches:
            # Count categories
            if match.category:
                category_counts[match.category] = category_counts.get(match.category, 0) + 1

            # Count domains
            if match.domain:
                domain_counts[match.domain] = domain_counts.get(match.domain, 0) + 1

            # Count methods
            if match.method:
                method_counts[match.method] = method_counts.get(match.method, 0) + 1

        # Convert results to dictionaries
        entities = [match.to_dict() for match in entity_matches]

        # Compile results
        return {
            "entities": entities,
            "unresolved": unresolved,
            "summary": {
                "total_texts": total_texts,
                "matched_count": len(entity_matches),
                "dictionary_matches": dictionary_matches,
                "ner_matches": ner_matches,
                "unresolved_count": len(unresolved),
                "match_percentage": round(match_percentage, 2)
            },
            "category_distribution": category_counts,
            "domain_distribution": domain_counts,
            "method_distribution": method_counts,
            "language": language
        }