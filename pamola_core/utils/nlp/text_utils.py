"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        General Text Processing Utilities
Package:       pamola_core.utils.nlp
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides general-purpose text processing utilities for the PAMOLA
  framework. It offers text normalization, string similarity calculations, and
  text manipulation functions that are used across various NLP and anonymization
  operations.

Key Features:
  - Multi-level text normalization (basic, advanced, aggressive)
  - String similarity calculations using multiple algorithms
  - Fuzzy string matching with configurable thresholds
  - Composite value handling and token extraction
  - Integration with stopwords module for language-aware processing
  - Unicode-aware text processing with proper normalization

Design Principles:
  - Minimal external dependencies (uses built-in libraries where possible)
  - Efficient algorithms suitable for large-scale data processing
  - Consistent API across all text processing functions
  - Language-agnostic with optional language-specific features

Usage:
  Used by anonymization operations for text field processing, category matching,
  and general text manipulation tasks throughout the PAMOLA framework.

Dependencies:
  - difflib (built-in): For sequence matching and similarity calculations
  - re (built-in): For regular expression operations
  - unicodedata (built-in): For Unicode normalization
  - typing: For type hints
  - pamola_core.utils.nlp.stopwords: For stopword removal (optional)

TODO:
  - Add support for more similarity algorithms (Jaro-Winkler, N-gram)
  - Implement phonetic matching (Soundex, Metaphone) for name matching
  - Add language detection for automatic normalization settings
  - Optimize Levenshtein implementation with C extension if needed
  - Add support for custom normalization rules per language
  - Implement text tokenization with better compound word handling
"""

import difflib
import logging
import re
from typing import List, Optional, Tuple

import unicodedata

# Import stopwords functionality
from pamola_core.utils.nlp.stopwords import get_stopwords

# Configure module logger
logger = logging.getLogger(__name__)

# Constants for text processing
NORMALIZATION_LEVELS = ["none", "basic", "advanced", "aggressive"]
DEFAULT_INVALID_CHARS = r'[<>:"/\\|?*]'
MAX_STRING_LENGTH = 10000  # Maximum string length for similarity calculations

# Unicode categories to remove in advanced normalization
UNICODE_CONTROL_CATEGORIES = ["Cc", "Cf", "Cs", "Co", "Cn"]

# Smart quote replacements
SMART_QUOTES = {
    "\u2018": "'",  # Left single quotation mark
    "\u2019": "'",  # Right single quotation mark
    "\u201c": '"',  # Left double quotation mark
    "\u201d": '"',  # Right double quotation mark
    "\u2013": "-",  # En dash
    "\u2014": "-",  # Em dash
    "\u2026": "...",  # Horizontal ellipsis
    "\u00a0": " ",  # Non-breaking space
}


def normalize_text(
    text: str,
    level: str = "basic",
    preserve_case: bool = False,
    languages: Optional[List[str]] = None,
) -> str:
    """
    Normalize text according to specified level.

    This function provides multi-level text normalization suitable for various
    text processing tasks including anonymization and category matching.

    Parameters:
    -----------
    text : str
        Input text to normalize
    level : str, optional
        Normalization level (default: "basic"):
        - "none": No normalization
        - "basic": Trim, lowercase, normalize whitespace
        - "advanced": Basic + remove special chars, normalize unicode
        - "aggressive": Advanced + alphanumeric only, optional stopword removal
    preserve_case : bool, optional
        Whether to preserve original case (default: False)
    languages : List[str], optional
        Language codes for stopword removal (only used in aggressive mode)

    Returns:
    --------
    str
        Normalized text

    Examples:
    ---------
    >>> normalize_text("  Hello   World!  ", "basic")
    'hello world!'
    >>> normalize_text("Café—2025", "advanced")
    'café-2025'
    >>> normalize_text("Hello, World! @#$", "aggressive")
    'hello world'
    """
    # Handle None or non-string input
    if text is None:
        return ""

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    # Check for valid normalization level
    if level not in NORMALIZATION_LEVELS:
        logger.warning(f"Unknown normalization level: {level}. Using 'basic'.")
        level = "basic"

    # No normalization
    if level == "none":
        return text

    # Basic normalization
    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize internal whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text)

    # Lowercase if not preserving case
    if not preserve_case:
        text = text.lower()

    if level == "basic":
        return text

    # Advanced normalization
    if level in ["advanced", "aggressive"]:
        # Unicode normalization (NFKC - compatibility decomposition + canonical composition)
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters
        text = "".join(
            ch
            for ch in text
            if unicodedata.category(ch) not in UNICODE_CONTROL_CATEGORIES
        )

        # Replace smart quotes and special punctuation
        for old_char, new_char in SMART_QUOTES.items():
            text = text.replace(old_char, new_char)

        # Remove zero-width characters
        text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)

    # Aggressive normalization
    if level == "aggressive":
        # Keep only alphanumeric, spaces, and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s\-_.,]", "", text)

        # Normalize multiple punctuation
        text = re.sub(r"([.,])\1+", r"\1", text)

        # Remove stopwords if languages specified
        if languages:
            try:
                stopwords = get_stopwords(languages=languages)
                words = text.split()
                text = " ".join(w for w in words if w.lower() not in stopwords)
            except Exception as e:
                logger.warning(f"Error removing stopwords: {e}")

        # Final whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_category_name(
    name: str, max_length: int = 50, invalid_chars: str = DEFAULT_INVALID_CHARS
) -> str:
    """
    Clean category names for safe file/field naming.

    This function ensures category names are safe for use as file names
    or field names by removing invalid characters and enforcing length limits.

    Parameters:
    -----------
    name : str
        Category name to clean
    max_length : int, optional
        Maximum allowed length (default: 50)
    invalid_chars : str, optional
        Regex pattern of invalid characters to remove

    Returns:
    --------
    str
        Cleaned category name

    Examples:
    ---------
    >>> clean_category_name("Sales/Marketing", max_length=20)
    'Sales_Marketing'
    >>> clean_category_name("Very Long Category Name That Exceeds The Limit", max_length=30)
    'Very Long Category Name Tha...'
    """
    if not name:
        return ""

    # Basic normalization preserving case
    clean = normalize_text(name, "basic", preserve_case=True)

    # Replace invalid characters with underscore
    clean = re.sub(invalid_chars, "_", clean)

    # Replace multiple underscores with single
    clean = re.sub(r"_+", "_", clean)

    # Remove leading/trailing underscores
    clean = clean.strip("_")

    # Enforce maximum length
    if len(clean) > max_length:
        # Leave room for ellipsis
        clean = clean[: max_length - 3] + "..."

    return clean


def calculate_string_similarity(
    s1: str,
    s2: str,
    method: str = "ratio",
    normalize: bool = True,
    case_sensitive: bool = False,
) -> float:
    """
    Calculate similarity between two strings using various methods.

    Parameters:
    -----------
    s1 : str
        First string
    s2 : str
        Second string
    method : str, optional
        Similarity calculation method (default: "ratio"):
        - "ratio": difflib SequenceMatcher ratio (0-1)
        - "partial": Best partial match ratio
        - "token": Token-based Jaccard similarity
        - "levenshtein": Normalized Levenshtein distance
    normalize : bool, optional
        Whether to normalize strings before comparison (default: True)
    case_sensitive : bool, optional
        Whether comparison is case-sensitive (default: False)

    Returns:
    --------
    float
        Similarity score between 0 and 1

    Examples:
    ---------
    >>> calculate_string_similarity("hello world", "Hello World")
    1.0
    >>> calculate_string_similarity("python", "jython", method="levenshtein")
    0.8333...
    """
    # Apply normalization if requested
    if normalize:
        normalization_level = "basic" if not case_sensitive else "none"
        s1 = normalize_text(s1, normalization_level, preserve_case=case_sensitive)
        s2 = normalize_text(s2, normalization_level, preserve_case=case_sensitive)
    elif not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()

    # Handle empty strings
    if not s1 or not s2:
        return 1.0 if s1 == s2 else 0.0

    # Limit string length for performance
    if len(s1) > MAX_STRING_LENGTH or len(s2) > MAX_STRING_LENGTH:
        logger.warning(
            f"String length exceeds {MAX_STRING_LENGTH}, truncating for similarity calculation"
        )
        s1 = s1[:MAX_STRING_LENGTH]
        s2 = s2[:MAX_STRING_LENGTH]

    if method == "ratio":
        # Use difflib's SequenceMatcher for ratio calculation
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    elif method == "partial":
        # Find best partial match (substring matching)
        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)

        if len(shorter) == 0:
            return 0.0

        best_ratio = 0.0
        len_short = len(shorter)

        # Slide shorter string across longer string
        for i in range(len(longer) - len_short + 1):
            substring = longer[i : i + len_short]
            ratio = difflib.SequenceMatcher(None, shorter, substring).ratio()
            best_ratio = max(best_ratio, ratio)

            # Early exit if perfect match found
            if best_ratio == 1.0:
                break

        return best_ratio

    elif method == "token":
        # Token-based (Jaccard) similarity
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())

        if not tokens1 or not tokens2:
            return 1.0 if tokens1 == tokens2 else 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    elif method == "levenshtein":
        # Normalized Levenshtein distance
        distance = _levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    This is an optimized implementation using only O(min(m,n)) space.

    Parameters:
    -----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns:
    --------
    int
        Minimum number of edits (insertions, deletions, substitutions)
        required to transform s1 into s2
    """
    # Ensure s1 is the shorter string for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    # Handle edge cases
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Previous row of distances
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        # Current row starts with deletion distance
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def find_closest_match(
    target: str,
    candidates: List[str],
    threshold: float = 0.8,
    method: str = "ratio",
    top_n: int = 1,
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """
    Find closest matching strings from a list of candidates.

    Parameters:
    -----------
    target : str
        Target string to match
    candidates : List[str]
        List of candidate strings
    threshold : float, optional
        Minimum similarity threshold (default: 0.8)
    method : str, optional
        Similarity method to use (default: "ratio")
    top_n : int, optional
        Number of top matches to return (default: 1)
    normalize : bool, optional
        Whether to normalize strings (default: True)

    Returns:
    --------
    List[Tuple[str, float]]
        List of (candidate, similarity_score) tuples, sorted by score descending

    Examples:
    ---------
    >>> candidates = ["python", "jython", "cython", "java"]
    >>> find_closest_match("pyton", candidates, threshold=0.7)
    [('python', 0.9090909090909091)]
    """
    if not target or not candidates:
        return []

    # Calculate similarities for all candidates
    scores = []

    for candidate in candidates:
        if candidate:  # Skip empty candidates
            try:
                score = calculate_string_similarity(
                    target, candidate, method=method, normalize=normalize
                )
                if score >= threshold:
                    scores.append((candidate, score))
            except Exception as e:
                logger.warning(f"Error calculating similarity for '{candidate}': {e}")
                continue

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top N matches
    return scores[:top_n]


def find_closest_category(
    value: str, categories: List[str], threshold: float = 0.8, method: str = "ratio"
) -> Optional[str]:
    """
    Find the best matching category for a value using string similarity.

    This is a convenience function specifically for category matching,
    returning only the best match or None if no match meets the threshold.

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

    Returns:
    --------
    Optional[str]
        Best matching category or None if no match meets threshold

    Examples:
    ---------
    >>> categories = ["Software Engineer", "Data Scientist", "Product Manager"]
    >>> find_closest_category("Software Dev", categories)
    'Software Engineer'
    """
    matches = find_closest_match(value, categories, threshold, method, top_n=1)
    return matches[0][0] if matches else None


def split_composite_value(
    value: str, separators: Optional[List[str]] = None, normalize: bool = True
) -> List[str]:
    """
    Split composite values into components.

    Handles values that contain multiple items separated by delimiters,
    such as "IT|Finance" or "Sales/Marketing".

    Parameters:
    -----------
    value : str
        Composite value to split
    separators : List[str], optional
        List of separator characters (default: ["|", "/", ",", ";"])
    normalize : bool, optional
        Whether to normalize components (default: True)

    Returns:
    --------
    List[str]
        List of components

    Examples:
    ---------
    >>> split_composite_value("IT|Finance|HR")
    ['it', 'finance', 'hr']
    >>> split_composite_value("Sales / Marketing", normalize=False)
    ['Sales', 'Marketing']
    """
    if not value:
        return []

    if separators is None:
        separators = ["|", "/", ",", ";"]

    # Create regex pattern from separators
    # Escape special regex characters
    escaped_seps = [re.escape(sep) for sep in separators]
    pattern = f"[{''.join(escaped_seps)}]"

    # Split by any separator
    components = re.split(pattern, value)

    # Clean up components
    result = []
    for comp in components:
        comp = comp.strip()
        if comp:  # Skip empty components
            if normalize:
                comp = normalize_text(comp, "basic")
            result.append(comp)

    return result


def extract_tokens(
    text: str,
    min_length: int = 2,
    pattern: Optional[str] = None,
    lowercase: bool = True,
) -> List[str]:
    """
    Extract tokens from text using configurable rules.

    Parameters:
    -----------
    text : str
        Text to tokenize
    min_length : int, optional
        Minimum token length (default: 2)
    pattern : str, optional
        Regex pattern for token extraction (default: alphanumeric words)
    lowercase : bool, optional
        Whether to lowercase tokens (default: True)

    Returns:
    --------
    List[str]
        List of extracted tokens

    Examples:
    ---------
    >>> extract_tokens("Hello World! Test-123")
    ['hello', 'world', 'test', '123']
    >>> extract_tokens("user@example.com", pattern=r'[a-zA-Z0-9@.]+')
    ['user@example.com']
    """
    if not text:
        return []

    # Default pattern: alphanumeric sequences
    if pattern is None:
        pattern = r"\b\w+\b"

    # Extract tokens using pattern
    tokens = re.findall(pattern, text)

    # Filter and process tokens
    result = []
    for token in tokens:
        if len(token) >= min_length:
            if lowercase:
                token = token.lower()
            result.append(token)

    return result


def is_valid_category_name(
    name: str,
    max_length: int = 50,
    min_length: int = 1,
    invalid_chars: str = DEFAULT_INVALID_CHARS,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a category name is valid.

    Parameters:
    -----------
    name : str
        Category name to validate
    max_length : int, optional
        Maximum allowed length
    min_length : int, optional
        Minimum required length
    invalid_chars : str, optional
        Regex pattern of invalid characters

    Returns:
    --------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)

    Examples:
    ---------
    >>> is_valid_category_name("ValidCategory")
    (True, None)
    >>> is_valid_category_name("A/B")
    (False, 'Name contains invalid characters: /')
    """
    if not name:
        return False, "Name cannot be empty"

    if len(name) < min_length:
        return False, f"Name too short (minimum {min_length} characters)"

    if len(name) > max_length:
        return False, f"Name too long (maximum {max_length} characters)"

    # Check for invalid characters
    invalid_found = re.findall(invalid_chars, name)
    if invalid_found:
        unique_invalid = list(set(invalid_found))
        return False, f"Name contains invalid characters: {', '.join(unique_invalid)}"

    return True, None


def truncate_text(
    text: str, max_length: int, suffix: str = "...", whole_words: bool = True
) -> str:
    """
    Truncate text to maximum length with optional ellipsis.

    Parameters:
    -----------
    text : str
        Text to truncate
    max_length : int
        Maximum length including suffix
    suffix : str, optional
        Suffix to append to truncated text (default: "...")
    whole_words : bool, optional
        Whether to truncate at word boundaries (default: True)

    Returns:
    --------
    str
        Truncated text

    Examples:
    ---------
    >>> truncate_text("This is a long text", 10)
    'This...'
    >>> truncate_text("This is a long text", 15, whole_words=False)
    'This is a lo...'
    """
    if not text or len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length]

    # Calculate available length for text
    available_length = max_length - len(suffix)

    if not whole_words:
        return text[:available_length] + suffix

    # Truncate at word boundary
    truncated = text[:available_length]

    # Find last complete word
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + suffix


# Module metadata
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    # Text normalization
    "normalize_text",
    "clean_category_name",
    # String similarity
    "calculate_string_similarity",
    "find_closest_match",
    "find_closest_category",
    # Text manipulation
    "split_composite_value",
    "extract_tokens",
    "truncate_text",
    # Validation
    "is_valid_category_name",
    # Constants
    "NORMALIZATION_LEVELS",
    "DEFAULT_INVALID_CHARS",
    "MAX_STRING_LENGTH",
]
