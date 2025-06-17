"""
Category matching module for the project.

This module provides functionality for matching text to predefined categories,
with support for hierarchical dictionaries and conflict resolution strategies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from pamola_core.utils.nlp.base import ConfigurationError
from pamola_core.utils.nlp.cache import get_cache, cache_function

# Configure logger
logger = logging.getLogger(__name__)

# Get file cache for efficient dictionary loading
file_cache = get_cache('file')


class CategoryDictionary:
    """
    Class for managing category dictionaries with support for hierarchies.
    """

    def __init__(self, dictionary_data: Dict[str, Any] = None, hierarchy_data: Dict[str, Any] = None):
        """
        Initialize the category dictionary.

        Parameters:
        -----------
        dictionary_data : Dict[str, Any], optional
            Category dictionary data
        hierarchy_data : Dict[str, Any], optional
            Category hierarchy data
        """
        self.dictionary = dictionary_data or {}
        self.hierarchy = hierarchy_data or {}

    @classmethod
    @cache_function(ttl=3600, cache_type='file')
    def from_file(cls, file_path: Union[str, Path]) -> 'CategoryDictionary':
        """
        Load a category dictionary from a JSON file.

        Parameters:
        -----------
        file_path : str or Path
            Path to the dictionary JSON file

        Returns:
        --------
        CategoryDictionary
            Loaded category dictionary
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Dictionary file not found: {path}")
                return cls()

            # Use the file cache to efficiently load and monitor the file
            cache_key = f"category_dict:{path}"
            data = file_cache.get(cache_key)

            if data is None:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Store in cache (file_path enables mtime checking)
                file_cache.set(cache_key, data, file_path=str(path))

            # Handle different dictionary formats
            if "categories_hierarchy" in data:
                # New format with hierarchy
                dictionary_data, hierarchy_data = cls._prepare_data_from_hierarchy(data)
                return cls(dictionary_data, hierarchy_data)
            elif all(isinstance(v, dict) for v in data.values()):
                # Flat dictionary format
                return cls(data, {})
            else:
                logger.warning(f"Unknown dictionary format in file: {path}")
                return cls()

        except ConfigurationError:
            # Just re-raise configuration errors
            raise
        except Exception as e:
            logger.error(f"Error loading dictionary from {file_path}: {e}")
            return cls()

    @staticmethod
    def _prepare_data_from_hierarchy(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare dictionary and hierarchy data from a hierarchical dictionary.

        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary data with hierarchy information

        Returns:
        --------
        Tuple[Dict[str, Any], Dict[str, Any]]
            Tuple of (flattened dictionary data, hierarchy data)
        """
        # Extract hierarchy
        hierarchy = data.get("categories_hierarchy", {})

        # Create flattened dictionary for matching
        dictionary = {}

        # Process each category in the hierarchy
        for category_name, category_info in hierarchy.items():
            # Only include categories with keywords
            if "keywords" in category_info:
                dictionary[category_name] = category_info.copy()

        # Include fallback category if present
        if "Unclassified" in data:
            dictionary["Unclassified"] = data["Unclassified"]

        return dictionary, hierarchy

    def get_best_match(self, text: str, strategy: str = "specific_first") -> Tuple[Optional[str], float, List[str]]:
        """
        Find the best matching category for a text.

        Parameters:
        -----------
        text : str
            Text to match against the dictionary
        strategy : str
            Strategy for resolving conflicts:
            "specific_first", "domain_prefer", "alias_only", "user_override"

        Returns:
        --------
        Tuple[Optional[str], float, List[str]]
            Matched category, confidence score, and conflict candidates
        """
        if not text or not self.dictionary:
            return None, 0.0, []

        text_lower = text.lower().strip()

        # Find all matching categories
        matches = []

        for category, info in self.dictionary.items():
            # Skip categories without keywords
            if "keywords" not in info:
                continue

            keywords = info["keywords"]
            languages = info.get("language", ["en", "ru"])

            # Check each keyword
            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Check if keyword is in text
                if keyword_lower in text_lower:
                    # Calculate match score based on keyword length relative to text length
                    score = len(keyword_lower) / len(text_lower)

                    # Store match
                    matches.append({
                        "category": category,
                        "matched_keyword": keyword_lower,
                        "score": score,
                        "level": info.get("level", 0),
                        "domain": info.get("domain", "General"),
                        "alias": info.get("alias", category.lower().replace(" ", "_")),
                        "seniority": info.get("seniority", "Any")
                    })

        # If no matches, return None
        if not matches:
            return None, 0.0, []

        # Resolve conflicts based on strategy
        if strategy == "specific_first":
            # Sort by level (descending), then score (descending)
            matches.sort(key=lambda m: (-m["level"], -m["score"]))
        elif strategy == "domain_prefer":
            # Sort by domain specificity (non-General first), then score
            matches.sort(key=lambda m: (m["domain"] == "General", -m["score"]))
        elif strategy == "alias_only":
            # Sort just by score
            matches.sort(key=lambda m: -m["score"])
        elif strategy == "user_override":
            # Use override if present otherwise fall back to specific_first
            # This would typically use an external mapping
            matches.sort(key=lambda m: (-m["level"], -m["score"]))

        # Get best match
        best_match = matches[0]

        # Record conflict candidates
        conflict_candidates = []
        for match in matches[1:]:
            if match["category"] != best_match["category"]:
                conflict_candidates.append(match["category"])

        return best_match["category"], best_match["score"], conflict_candidates

    def get_category_info(self, category: str) -> Dict[str, Any]:
        """
        Get information about a category.

        Parameters:
        -----------
        category : str
            Category name

        Returns:
        --------
        Dict[str, Any]
            Category information or empty dict if not found
        """
        return self.dictionary.get(category, {})

    def analyze_hierarchy(self) -> Dict[str, Any]:
        """
        Analyze the category hierarchy to extract useful information.

        Returns:
        --------
        Dict[str, Any]
            Analysis results including level counts, domain distribution, etc.
        """
        if not self.hierarchy:
            return {}

        # Initialize results
        results = {
            "total_categories": len(self.hierarchy),
            "level_counts": {},
            "domain_counts": {},
            "has_children": {},
            "categories_by_level": {},
            "categories_by_domain": {}
        }

        # Analyze categories
        for category, info in self.hierarchy.items():
            # Extract level
            level = info.get("level", 0)
            results["level_counts"][level] = results["level_counts"].get(level, 0) + 1

            # Add to categories by level
            if level not in results["categories_by_level"]:
                results["categories_by_level"][level] = []
            results["categories_by_level"][level].append(category)

            # Extract domain
            domain = info.get("domain", "General")
            results["domain_counts"][domain] = results["domain_counts"].get(domain, 0) + 1

            # Add to categories by domain
            if domain not in results["categories_by_domain"]:
                results["categories_by_domain"][domain] = []
            results["categories_by_domain"][domain].append(category)

            # Check for children
            has_children = "children" in info and len(info["children"]) > 0
            results["has_children"][category] = has_children

        return results

    def get_fallback_category(self, confidence_threshold: float = 0.5) -> Optional[str]:
        """
        Get a fallback category for when matches don't meet confidence threshold.

        Parameters:
        -----------
        confidence_threshold : float
            Minimum confidence score for reliable matches

        Returns:
        --------
        Optional[str]
            Fallback category name or None if not defined
        """
        # Try to find the Unclassified category
        if "Unclassified" in self.dictionary:
            return "Unclassified"

        # Try to find a general category at level 0
        for category, info in self.dictionary.items():
            if info.get("level", 0) == 0 and info.get("domain", "") == "General":
                return category

        # If no suitable fallback found, return None
        return None


@cache_function(ttl=3600, cache_type='memory')
def get_best_match(text: str, dictionary: Dict[str, Dict[str, Any]],
                   match_strategy: str = "specific_first") -> Tuple[Optional[str], float, List[str]]:
    """
    Find the best matching category for a text based on the dictionary.

    Parameters:
    -----------
    text : str
        Text to categorize
    dictionary : Dict[str, Dict[str, Any]]
        Dictionary with category definitions
    match_strategy : str
        Strategy for resolving conflicts:
        "specific_first", "domain_prefer", "alias_only", "user_override"

    Returns:
    --------
    Tuple[Optional[str], float, List[str]]
        Tuple of (matched_category, confidence_score, conflict_candidates)
    """
    # Create a temporary CategoryDictionary
    category_dict = CategoryDictionary(dictionary)
    return category_dict.get_best_match(text, match_strategy)


@cache_function(ttl=3600, cache_type='memory')
def analyze_hierarchy(hierarchy: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a category hierarchy to extract useful information.

    Parameters:
    -----------
    hierarchy : Dict[str, Dict[str, Any]]
        Dictionary containing category hierarchy information

    Returns:
    --------
    Dict[str, Any]
        Analysis results including level counts, domain distribution, etc.
    """
    # Create a temporary CategoryDictionary
    category_dict = CategoryDictionary(hierarchy_data=hierarchy)
    return category_dict.analyze_hierarchy()


def batch_match_categories(texts: List[str], dictionary_path: Union[str, Path],
                           match_strategy: str = "specific_first",
                           processes: Optional[int] = None) -> List[Tuple[Optional[str], float, List[str]]]:
    """
    Match multiple texts to categories in parallel.

    Parameters:
    -----------
    texts : List[str]
        List of texts to categorize
    dictionary_path : str or Path
        Path to category dictionary file
    match_strategy : str
        Strategy for resolving conflicts
    processes : int, optional
        Number of processes for parallel execution

    Returns:
    --------
    List[Tuple[Optional[str], float, List[str]]]
        List of (category, score, conflicts) tuples for each text
    """
    # Load dictionary once to avoid loading it in each process
    category_dict = CategoryDictionary.from_file(dictionary_path)

    def match_text(text):
        return category_dict.get_best_match(text, match_strategy)

    # Use batch_process for parallel execution
    from pamola_core.utils.nlp.base import batch_process
    return batch_process(texts, match_text, processes=processes)