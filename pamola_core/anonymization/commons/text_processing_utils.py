"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Processing Utilities for Anonymization
Package:       pamola_core.anonymization.commons
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides text processing utilities specifically tailored for
  anonymization operations within the PAMOLA framework. It wraps and extends
  the general-purpose text utilities from pamola_core.utils.nlp.text_utils with
  anonymization-specific functionality and parameter defaults.

Key Features:
  - Privacy-aware text normalization with anonymization defaults
  - Category matching for hierarchical generalization
  - Composite value handling for multi-part fields
  - Safe string transformations that preserve privacy
  - Integration with external dictionaries for category matching
  - Configurable fuzzy matching for robust categorization

Design Principles:
  - Builds on top of pamola_core.utils.nlp.text_utils for consistency
  - Provides anonymization-specific defaults and behaviors
  - Ensures safe transformations that don't leak information
  - Optimized for batch processing of categorical data

Usage:
  Used by categorical generalization operations for text normalization,
  category matching, and hierarchy navigation within the anonymization package.

Dependencies:
  - pamola_core.utils.nlp.text_utils: Base text processing functions
  - re: Regular expressions for pattern matching
  - logging: Error and warning logging
  - typing: Type hints

TODO:
  - Add support for semantic similarity using embeddings
  - Implement multi-language category normalization
  - Add caching for frequently matched categories
  - Support for custom tokenization rules per field type
"""

import logging
import re
from typing import List, Optional, Tuple, Dict

# Import base text processing functions from NLP package
from pamola_core.utils.nlp.text_utils import (
    normalize_text as nlp_normalize_text,
    clean_category_name as nlp_clean_category_name,
    calculate_string_similarity,
    find_closest_match,
    find_closest_category as nlp_find_closest_category,
    split_composite_value as nlp_split_composite_value,
    extract_tokens as nlp_extract_tokens,
    truncate_text,
    is_valid_category_name,

)

# Configure module logger
logger = logging.getLogger(__name__)

# Anonymization-specific constants
ANONYMIZATION_SAFE_CHARS = r'[^a-zA-Z0-9\s\-_]'  # More restrictive for anonymization
DEFAULT_UNKNOWN_VALUE = "OTHER"
DEFAULT_SUPPRESSED_VALUE = "SUPPRESSED"
CATEGORY_SEPARATOR = "_"
MAX_CATEGORY_LENGTH = 40  # Shorter for anonymization fields


def normalize_text(text: str,
                   level: str = "basic",
                   preserve_case: bool = False) -> str:
    """
    Normalize text for anonymization purposes.

    This is a wrapper around the NLP normalize_text function with
    anonymization-specific defaults. It does not use stopword removal
    by default as this could affect category matching.

    Parameters:
    -----------
    text : str
        Text to normalize
    level : str, optional
        Normalization level (default: "basic")
        - "none": No normalization
        - "basic": Trim, lowercase, normalize whitespace
        - "advanced": Basic + remove special chars, normalize unicode
        - "aggressive": Advanced + alphanumeric only
    preserve_case : bool, optional
        Whether to preserve original case (default: False)

    Returns:
    --------
    str
        Normalized text suitable for anonymization

    Examples:
    ---------
    >>> normalize_text("  Sales & Marketing  ")
    'sales & marketing'
    >>> normalize_text("IT/Finance", level="advanced")
    'it/finance'
    """
    # For anonymization, we typically don't want stopword removal
    # as it could affect category matching accuracy
    return nlp_normalize_text(text, level, preserve_case, languages=None)


def clean_category_name(name: str,
                        max_length: int = MAX_CATEGORY_LENGTH,
                        invalid_chars: str = ANONYMIZATION_SAFE_CHARS,
                        separator: str = CATEGORY_SEPARATOR) -> str:
    """
    Clean category names for safe anonymization field naming.

    This function is more restrictive than the general clean_category_name
    to ensure generated field names are safe for databases and downstream
    processing systems.

    Parameters:
    -----------
    name : str
        Category name to clean
    max_length : int, optional
        Maximum length (default: 40, shorter for anonymization)
    invalid_chars : str, optional
        Regex pattern of invalid characters (more restrictive)
    separator : str, optional
        Character to use as separator (default: "_")

    Returns:
    --------
    str
        Cleaned category name safe for anonymization

    Examples:
    ---------
    >>> clean_category_name("Sales & Marketing")
    'Sales_Marketing'
    >>> clean_category_name("IT/Finance/HR")
    'IT_Finance_HR'
    """
    if not name:
        return DEFAULT_UNKNOWN_VALUE

    # First apply general cleaning
    clean = nlp_clean_category_name(name, max_length, invalid_chars)

    # Additional anonymization-specific cleaning
    # Replace common separators with our standard separator
    for sep in ['/', '\\', '|', '&', '+', '-', '.', ',']:
        clean = clean.replace(sep, separator)

    # Remove multiple separators
    clean = re.sub(f'{re.escape(separator)}+', separator, clean)

    # Remove leading/trailing separators
    clean = clean.strip(separator)

    # If empty after cleaning, return unknown
    if not clean:
        return DEFAULT_UNKNOWN_VALUE

    return clean


def find_closest_category(value: str,
                          categories: List[str],
                          threshold: float = 0.8,
                          method: str = "ratio",
                          normalize_value: bool = True,
                          fallback: str = DEFAULT_UNKNOWN_VALUE) -> str:
    """
    Find the best matching category with anonymization-specific handling.

    This function extends the basic find_closest_category with fallback
    handling and normalization options specific to anonymization needs.

    Parameters:
    -----------
    value : str
        Value to categorize
    categories : List[str]
        List of possible categories
    threshold : float, optional
        Minimum similarity threshold (default: 0.8)
    method : str, optional
        Similarity method (default: "ratio")
    normalize_value : bool, optional
        Whether to normalize the input value (default: True)
    fallback : str, optional
        Fallback category if no match found (default: "OTHER")

    Returns:
    --------
    str
        Best matching category or fallback value

    Examples:
    ---------
    >>> categories = ["Software Engineer", "Data Scientist"]
    >>> find_closest_category("Software Dev", categories)
    'Software Engineer'
    >>> find_closest_category("Unknown Role", categories)
    'OTHER'
    """
    if not value or not categories:
        return fallback

    # Normalize the input value if requested
    if normalize_value:
        value = normalize_text(value, "basic")

    # Use the NLP function to find matches
    result = nlp_find_closest_category(value, categories, threshold, method)

    # Return fallback if no match found
    return result if result is not None else fallback


def split_composite_value(value: str,
                          separators: Optional[List[str]] = None,
                          normalize: bool = True,
                          max_components: int = 10) -> List[str]:
    """
    Split composite values with anonymization constraints.

    This function limits the number of components to prevent data explosion
    and applies anonymization-friendly normalization.

    Parameters:
    -----------
    value : str
        Composite value to split
    separators : List[str], optional
        Separators (default: ["|", "/", ",", ";", "&"])
    normalize : bool, optional
        Whether to normalize components (default: True)
    max_components : int, optional
        Maximum number of components to return (default: 10)

    Returns:
    --------
    List[str]
        List of components (limited to max_components)

    Examples:
    ---------
    >>> split_composite_value("IT|Finance|HR")
    ['it', 'finance', 'hr']
    >>> split_composite_value("A/B/C/D/E/F/G/H/I/J/K", max_components=5)
    ['a', 'b', 'c', 'd', 'e']
    """
    # Default separators include common business separators
    if separators is None:
        separators = ["|", "/", ",", ";", "&"]

    # Use base function to split
    components = nlp_split_composite_value(value, separators, normalize)

    # Limit components for anonymization safety
    if len(components) > max_components:
        logger.warning(f"Composite value has {len(components)} components, "
                       f"limiting to {max_components}")
        components = components[:max_components]

    return components


def extract_tokens(text: str,
                   min_length: int = 3,  # Longer minimum for anonymization
                   pattern: Optional[str] = None,
                   lowercase: bool = True,
                   max_tokens: int = 50) -> List[str]:
    """
    Extract tokens with anonymization-specific constraints.

    This function uses a higher minimum length and limits the number of
    tokens to prevent information leakage through numerous small tokens.

    Parameters:
    -----------
    text : str
        Text to tokenize
    min_length : int, optional
        Minimum token length (default: 3, longer for anonymization)
    pattern : str, optional
        Token extraction pattern (default: alphanumeric)
    lowercase : bool, optional
        Whether to lowercase tokens (default: True)
    max_tokens : int, optional
        Maximum number of tokens to return (default: 50)

    Returns:
    --------
    List[str]
        List of tokens (limited to max_tokens)

    Examples:
    ---------
    >>> extract_tokens("Senior Software Engineer in NYC")
    ['senior', 'software', 'engineer', 'nyc']
    """
    # Use base function with anonymization parameters
    tokens = nlp_extract_tokens(text, min_length, pattern, lowercase)

    # Limit tokens for anonymization safety
    if len(tokens) > max_tokens:
        logger.warning(f"Text has {len(tokens)} tokens, limiting to {max_tokens}")
        tokens = tokens[:max_tokens]

    return tokens


def merge_composite_categories(categories: List[str],
                               strategy: str = "first",
                               separator: str = CATEGORY_SEPARATOR,
                               max_length: int = MAX_CATEGORY_LENGTH) -> str:
    """
    Merge multiple category assignments into a single category.

    This function is specific to anonymization for handling cases where
    multiple categories need to be combined into one.

    Parameters:
    -----------
    categories : List[str]
        List of categories to merge
    strategy : str, optional
        Merging strategy (default: "first"):
        - "first": Use the first category
        - "all": Concatenate all categories
        - "most_specific": Use the longest category
        - "shortest": Use the shortest category
    separator : str, optional
        Separator for "all" strategy (default: "_")
    max_length : int, optional
        Maximum length for merged category (default: 40)

    Returns:
    --------
    str
        Merged category name

    Examples:
    ---------
    >>> merge_composite_categories(["IT", "Finance"], strategy="all")
    'IT_Finance'
    >>> merge_composite_categories(["Junior", "Senior"], strategy="first")
    'Junior'
    """
    if not categories:
        return DEFAULT_UNKNOWN_VALUE

    # Remove empty categories
    categories = [cat for cat in categories if cat and cat.strip()]

    if not categories:
        return DEFAULT_UNKNOWN_VALUE

    if strategy == "first":
        result = categories[0]

    elif strategy == "all":
        # Join all categories
        result = separator.join(categories)

    elif strategy == "most_specific":
        # Assume longer names are more specific
        result = max(categories, key=len)

    elif strategy == "shortest":
        # Use the shortest (most general) category
        result = min(categories, key=len)

    else:
        logger.warning(f"Unknown merge strategy: {strategy}, using 'first'")
        result = categories[0]

    # Ensure result doesn't exceed max length
    if len(result) > max_length:
        result = truncate_text(result, max_length, suffix="", whole_words=False)

    return clean_category_name(result)


def prepare_value_for_matching(value: str,
                               remove_common_prefixes: bool = True,
                               remove_common_suffixes: bool = True) -> str:
    """
    Prepare a value for category matching by removing common prefixes/suffixes.

    This function is specific to anonymization for improving matching accuracy
    by removing common job title prefixes/suffixes, location indicators, etc.

    Parameters:
    -----------
    value : str
        Value to prepare
    remove_common_prefixes : bool, optional
        Whether to remove common prefixes (default: True)
    remove_common_suffixes : bool, optional
        Whether to remove common suffixes (default: True)

    Returns:
    --------
    str
        Prepared value for matching

    Examples:
    ---------
    >>> prepare_value_for_matching("Senior Software Engineer")
    'software engineer'
    >>> prepare_value_for_matching("Manager - Sales")
    'sales'
    """
    if not value:
        return ""

    # Normalize first
    prepared = normalize_text(value, "basic")

    if remove_common_prefixes:
        # Common job prefixes
        prefixes = [
            "senior", "junior", "lead", "principal", "chief", "head",
            "associate", "assistant", "deputy", "vice", "executive",
            "managing", "sr", "jr", "i", "ii", "iii", "iv", "v"
        ]

        for prefix in prefixes:
            pattern = r'^\s*' + re.escape(prefix) + r'\s+'
            prepared = re.sub(pattern, '', prepared, flags=re.IGNORECASE)

    if remove_common_suffixes:
        # Common job suffixes
        suffixes = [
            "manager", "specialist", "analyst", "coordinator",
            "administrator", "officer", "consultant", "advisor",
            "i", "ii", "iii", "iv", "v", "level", "grade"
        ]

        for suffix in suffixes:
            pattern = r'\s+' + re.escape(suffix) + r'\s*$'
            prepared = re.sub(pattern, '', prepared, flags=re.IGNORECASE)

    # Remove extra whitespace
    prepared = re.sub(r'\s+', ' ', prepared).strip()

    # If empty after cleaning, return original normalized value
    if not prepared:
        prepared = normalize_text(value, "basic")

    return prepared


def validate_hierarchy_value(value: str,
                             level: int,
                             max_length_by_level: Optional[Dict[int, int]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate a value for use in a hierarchy at a specific level.

    This function ensures values are appropriate for their hierarchy level,
    with more general categories at higher levels having stricter constraints.

    Parameters:
    -----------
    value : str
        Value to validate
    level : int
        Hierarchy level (0 = most general)
    max_length_by_level : Dict[int, int], optional
        Maximum lengths per level

    Returns:
    --------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)

    Examples:
    ---------
    >>> validate_hierarchy_value("Technology", level=0)
    (True, None)
    >>> validate_hierarchy_value("Senior Software Engineer", level=0)
    (False, 'Value too specific for level 0')
    """
    if not value:
        return False, "Value cannot be empty"

    # Default max lengths by level (more general = shorter)
    if max_length_by_level is None:
        max_length_by_level = {
            0: 20,  # Top level: very general
            1: 30,  # Mid level: somewhat specific
            2: 40,  # Lower level: more specific
            3: 50,  # Detail level: quite specific
        }

    # Get max length for level (default to 50 for undefined levels)
    max_length = max_length_by_level.get(level, 50)

    if len(value) > max_length:
        return False, f"Value too long for level {level} (max {max_length} chars)"

    # Check for overly specific values at high levels
    if level == 0:
        # Top level should not have many tokens
        tokens = extract_tokens(value, min_length=1)
        if len(tokens) > 3:
            return False, f"Value too specific for level {level}"

    # Validate as category name
    is_valid, error = is_valid_category_name(value, max_length=max_length)

    return is_valid, error


def create_safe_hierarchy_path(components: List[str],
                               separator: str = " > ",
                               max_length: int = 100) -> str:
    """
    Create a safe string representation of a hierarchy path.

    This function creates a human-readable hierarchy path while ensuring
    it's safe for storage and display in anonymized outputs.

    Parameters:
    -----------
    components : List[str]
        Hierarchy components from general to specific
    separator : str, optional
        Separator between levels (default: " > ")
    max_length : int, optional
        Maximum total length (default: 100)

    Returns:
    --------
    str
        Safe hierarchy path string

    Examples:
    ---------
    >>> create_safe_hierarchy_path(["Technology", "Software", "Engineering"])
    'Technology > Software > Engineering'
    """
    if not components:
        return DEFAULT_UNKNOWN_VALUE

    # Clean each component
    clean_components = []
    for comp in components:
        if comp and comp.strip():
            clean = clean_category_name(comp, max_length=30)
            if clean and clean != DEFAULT_UNKNOWN_VALUE:
                clean_components.append(clean)

    if not clean_components:
        return DEFAULT_UNKNOWN_VALUE

    # Join with separator
    path = separator.join(clean_components)

    # Ensure total length is within limit
    if len(path) > max_length:
        # Try shorter separator first
        if len(separator) > 1:
            path = ">".join(clean_components)

        # If still too long, truncate components
        if len(path) > max_length:
            while len(path) > max_length and len(clean_components) > 1:
                # Remove the most specific component
                clean_components.pop()
                path = separator.join(clean_components) + "..."

    return path


# Module metadata
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    # Text normalization (adapted)
    'normalize_text',
    'clean_category_name',

    # String similarity (re-exported)
    'calculate_string_similarity',
    'find_closest_match',
    'find_closest_category',

    # Text manipulation (adapted)
    'split_composite_value',
    'extract_tokens',
    'truncate_text',

    # Validation (re-exported)
    'is_valid_category_name',

    # Anonymization-specific functions
    'merge_composite_categories',
    'prepare_value_for_matching',
    'validate_hierarchy_value',
    'create_safe_hierarchy_path',

    # Constants
    'ANONYMIZATION_SAFE_CHARS',
    'DEFAULT_UNKNOWN_VALUE',
    'DEFAULT_SUPPRESSED_VALUE',
    'CATEGORY_SEPARATOR',
    'MAX_CATEGORY_LENGTH',
]