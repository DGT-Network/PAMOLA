"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Diversity Metrics for Text Analysis
Package:       pamola_core.utils.nlp
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01
License:       BSD 3-Clause

Description:
  This module provides diversity metrics for text and categorical data analysis.
  It includes measures of semantic, lexical, and token diversity that are useful
  for assessing the variety and richness of textual data in various contexts
  including data profiling, quality assessment, and anonymization operations.

Key Features:
  - Semantic diversity calculation based on token overlap and similarity
  - Lexical diversity metrics (TTR, MTLD, vocd-D)
  - Token-based diversity measures
  - Support for various calculation methods and parameters
  - Efficient computation for large datasets
  - Configurable normalization and filtering options

Design Principles:
  - Performance: Optimized for large-scale text analysis
  - Flexibility: Multiple calculation methods for different use cases
  - Simplicity: Clear interfaces with sensible defaults
  - Compatibility: Works with pandas Series and list inputs

Dependencies:
  - numpy: Numerical computations
  - pandas: Data structure support
  - typing: Type hints
  - collections: Frequency analysis
  - math: Mathematical operations
  - logging: Debug and error logging

Changelog:
  1.0.0 - Initial implementation with three core diversity metrics
        - calculate_semantic_diversity() for category/token overlap
        - calculate_lexical_diversity() for vocabulary richness
        - calculate_token_diversity() for token distribution analysis

Example:
  >>> from pamola_core.utils.nlp.diversity_metrics import calculate_lexical_diversity
  >>> texts = ["The quick brown fox", "The lazy dog", "A quick cat"]
  >>> diversity = calculate_lexical_diversity(texts, method="ttr")
  >>> print(f"Lexical diversity: {diversity:.2f}")
"""

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MIN_TOKEN_LENGTH = 1
DEFAULT_WINDOW_SIZE = 100
EPSILON = 1e-10


def calculate_semantic_diversity(
    categories: Union[List[str], pd.Series],
    method: str = "token_overlap",
    normalize: bool = True,
    min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    tokenization_pattern: Optional[str] = r"\b\w+\b",
    case_sensitive: bool = False,
) -> float:
    """
    Calculate semantic diversity of categories based on various methods.

    Semantic diversity measures how different the categories are from each other
    based on their semantic content (tokens, characters, or structure).

    Parameters:
    -----------
    categories : Union[List[str], pd.Series]
        List or Series of category names to analyze
    method : str, optional
        Diversity calculation method (default: "token_overlap")
        - "token_overlap": Based on shared tokens between categories
        - "edit_distance": Based on average edit distance
        - "length_variance": Based on length distribution
        - "char_diversity": Based on character-level diversity
    normalize : bool, optional
        Whether to normalize the result to [0, 1] range (default: True)
    min_token_length : int, optional
        Minimum token length to consider (default: 1)
    tokenization_pattern : Optional[str]
        Regex pattern for tokenization (default: word boundaries)
    case_sensitive : bool, optional
        Whether to preserve case in analysis (default: False)

    Returns:
    --------
    float
        Diversity score. If normalized, returns value in [0, 1] where:
        - 0 = no diversity (all categories identical)
        - 1 = maximum diversity (no overlap between categories)

    Examples:
    --------
    >>> categories = ["Software Engineer", "Software Developer", "Data Scientist"]
    >>> diversity = calculate_semantic_diversity(categories)
    >>> print(f"Token overlap diversity: {diversity:.2f}")

    >>> # Using edit distance method
    >>> diversity = calculate_semantic_diversity(categories, method="edit_distance")
    >>> print(f"Edit distance diversity: {diversity:.2f}")
    """
    # Convert to list if pandas Series
    if isinstance(categories, pd.Series):
        categories = categories.dropna().astype(str).tolist()

    # Handle empty or single category
    if len(categories) <= 1:
        return 0.0

    # Ensure all are strings and handle case sensitivity
    if not case_sensitive:
        categories = [str(cat).lower() for cat in categories]
    else:
        categories = [str(cat) for cat in categories]

    # Remove duplicates for analysis
    unique_categories = list(set(categories))
    if len(unique_categories) == 1:
        return 0.0

    # Calculate diversity based on method
    if method == "token_overlap":
        diversity = _calculate_token_overlap_diversity(
            unique_categories, min_token_length, tokenization_pattern
        )
    elif method == "edit_distance":
        diversity = _calculate_edit_distance_diversity(unique_categories)
    elif method == "length_variance":
        diversity = _calculate_length_variance_diversity(unique_categories)
    elif method == "char_diversity":
        diversity = _calculate_character_diversity(unique_categories)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: "
            "token_overlap, edit_distance, length_variance, char_diversity"
        )

    # Normalize if requested
    if normalize and diversity > 0:
        if method == "token_overlap":
            # Already normalized
            pass
        elif method == "edit_distance":
            # Normalize by maximum possible edit distance
            max_length = max(len(cat) for cat in unique_categories)
            diversity = min(1.0, diversity / max_length)
        elif method == "length_variance":
            # Normalize by mean length
            mean_length = np.mean([len(cat) for cat in unique_categories])
            diversity = min(1.0, diversity / mean_length) if mean_length > 0 else 0.0
        elif method == "char_diversity":
            # Already normalized
            pass

    return float(diversity)


def calculate_lexical_diversity(
    texts: Union[List[str], pd.Series],
    method: str = "ttr",
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    tokenization_pattern: Optional[str] = r"\b\w+\b",
    case_sensitive: bool = False,
    remove_punctuation: bool = True,
) -> float:
    """
    Calculate lexical diversity (vocabulary richness) of text data.

    Lexical diversity measures the variety of unique words/tokens used in text.
    Higher values indicate more diverse vocabulary usage.

    Parameters:
    -----------
    texts : Union[List[str], pd.Series]
        Text data to analyze (can be single text or list of texts)
    method : str, optional
        Calculation method (default: "ttr")
        - "ttr": Type-Token Ratio (unique tokens / total tokens)
        - "root_ttr": Root TTR (unique tokens / sqrt(total tokens))
        - "log_ttr": Logarithmic TTR (log(unique) / log(total))
        - "mtld": Measure of Textual Lexical Diversity
        - "msttr": Mean Segmental TTR (moving window approach)
    window_size : int, optional
        Window size for MSTTR method (default: 100)
    min_token_length : int, optional
        Minimum token length to consider (default: 1)
    tokenization_pattern : Optional[str]
        Regex pattern for tokenization
    case_sensitive : bool, optional
        Whether to preserve case (default: False)
    remove_punctuation : bool, optional
        Whether to remove punctuation tokens (default: True)

    Returns:
    --------
    float
        Lexical diversity score. Range depends on method:
        - TTR: [0, 1] where 1 = all tokens unique
        - Root TTR: [0, sqrt(n)] where n = total tokens
        - Log TTR: [0, 1] normalized
        - MTLD: [0, ∞] where higher = more diverse
        - MSTTR: [0, 1] average TTR across windows

    Examples:
    --------
    >>> text = "The quick brown fox jumps over the lazy dog. The dog was lazy."
    >>> diversity = calculate_lexical_diversity([text], method="ttr")
    >>> print(f"TTR: {diversity:.3f}")

    >>> # Using MTLD for longer texts
    >>> diversity = calculate_lexical_diversity([text], method="mtld")
    >>> print(f"MTLD: {diversity:.2f}")
    """
    # Convert to list if needed
    if isinstance(texts, pd.Series):
        texts = texts.dropna().astype(str).tolist()
    elif isinstance(texts, str):
        texts = [texts]

    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = _tokenize_text(
            text,
            tokenization_pattern,
            case_sensitive,
            remove_punctuation,
            min_token_length,
        )
        all_tokens.extend(tokens)

    # Handle empty token list
    if not all_tokens:
        return 0.0

    # Calculate diversity based on method
    if method == "ttr":
        diversity = _calculate_ttr(all_tokens)
    elif method == "root_ttr":
        diversity = _calculate_root_ttr(all_tokens)
    elif method == "log_ttr":
        diversity = _calculate_log_ttr(all_tokens)
    elif method == "mtld":
        diversity = _calculate_mtld(all_tokens)
    elif method == "msttr":
        diversity = _calculate_msttr(all_tokens, window_size)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: "
            "ttr, root_ttr, log_ttr, mtld, msttr"
        )

    return float(diversity)


def calculate_token_diversity(
    tokens: Union[List[str], pd.Series],
    method: str = "entropy",
    normalize: bool = True,
    min_frequency: int = 1,
    top_n: Optional[int] = None,
    case_sensitive: bool = False,
) -> Dict[str, float]:
    """
    Calculate token-based diversity metrics for categorical or tokenized data.

    This function analyzes the distribution of tokens/categories to measure
    diversity from different perspectives.

    Parameters:
    -----------
    tokens : Union[List[str], pd.Series]
        List or Series of tokens/categories to analyze
    method : str, optional
        Calculation method (default: "entropy")
        - "entropy": Shannon entropy of token distribution
        - "simpson": Simpson's diversity index
        - "brillouin": Brillouin's diversity index
        - "all": Calculate all metrics
    normalize : bool, optional
        Whether to normalize results (default: True)
    min_frequency : int, optional
        Minimum frequency to include token (default: 1)
    top_n : Optional[int]
        Consider only top N most frequent tokens
    case_sensitive : bool, optional
        Whether to preserve case (default: False)

    Returns:
    --------
    Dict[str, float]
        Dictionary of diversity metrics:
        - "entropy": Shannon entropy (bits if normalized)
        - "simpson": Simpson's diversity index [0, 1]
        - "brillouin": Brillouin's diversity index
        - "unique_ratio": Ratio of unique tokens
        - "coverage_top_10": Coverage of top 10 tokens

    Examples:
    --------
    >>> tokens = ["cat", "dog", "cat", "bird", "dog", "cat", "fish"]
    >>> diversity = calculate_token_diversity(tokens, method="all")
    >>> print(f"Entropy: {diversity['entropy']:.3f}")
    >>> print(f"Simpson: {diversity['simpson']:.3f}")

    >>> # Focus on top tokens only
    >>> diversity = calculate_token_diversity(tokens, method="entropy", top_n=3)
    >>> print(f"Top-3 entropy: {diversity['entropy']:.3f}")
    """
    # Convert to list if needed
    if isinstance(tokens, pd.Series):
        tokens = tokens.dropna().astype(str).tolist()

    # Handle case sensitivity
    if not case_sensitive:
        tokens = [str(token).lower() for token in tokens]
    else:
        tokens = [str(token) for token in tokens]

    # Calculate frequency distribution
    token_counts = Counter(tokens)

    # Filter by minimum frequency
    if min_frequency > 1:
        token_counts = {k: v for k, v in token_counts.items() if v >= min_frequency}

    # Handle top_n
    if top_n and top_n < len(token_counts):
        token_counts = dict(token_counts.most_common(top_n))

    # Get frequencies
    frequencies = np.array(list(token_counts.values()))
    total_count = frequencies.sum()

    if total_count == 0:
        return {
            "entropy": 0.0,
            "simpson": 0.0,
            "brillouin": 0.0,
            "unique_ratio": 0.0,
            "coverage_top_10": 0.0,
        }

    # Calculate probabilities
    probabilities = frequencies / total_count

    # Initialize results
    results = {}

    # Calculate requested metrics
    if method in ["entropy", "all"]:
        entropy = _calculate_shannon_entropy(probabilities, normalize)
        results["entropy"] = float(entropy)

    if method in ["simpson", "all"]:
        simpson = _calculate_simpson_index(probabilities)
        results["simpson"] = float(simpson)

    if method in ["brillouin", "all"]:
        brillouin = _calculate_brillouin_index(frequencies, total_count, normalize)
        results["brillouin"] = float(brillouin)

    # Always include these basic metrics
    results["unique_ratio"] = float(len(token_counts) / len(tokens)) if tokens else 0.0

    # Coverage of top 10
    top_10_freq = sum(sorted(frequencies, reverse=True)[:10])
    results["coverage_top_10"] = (
        float(top_10_freq / total_count) if total_count > 0 else 0.0
    )

    return results


# ============================================================================
# Helper Functions
# ============================================================================


def _tokenize_text(
    text: str,
    pattern: Optional[str],
    case_sensitive: bool,
    remove_punctuation: bool,
    min_length: int,
) -> List[str]:
    """Tokenize text with given parameters."""
    import re

    if not case_sensitive:
        text = text.lower()

    if pattern:
        tokens = re.findall(pattern, text)
    else:
        # Simple split
        tokens = text.split()

    if remove_punctuation:
        tokens = [t for t in tokens if t.isalnum()]

    if min_length > 1:
        tokens = [t for t in tokens if len(t) >= min_length]

    return tokens


def _calculate_token_overlap_diversity(
    categories: List[str], min_token_length: int, pattern: Optional[str]
) -> float:
    """Calculate diversity based on token overlap between categories."""
    import re

    # Tokenize each category
    category_tokens = []
    all_tokens = set()

    for cat in categories:
        if pattern:
            tokens = set(re.findall(pattern, cat))
        else:
            tokens = set(cat.split())

        if min_token_length > 1:
            tokens = {t for t in tokens if len(t) >= min_token_length}

        category_tokens.append(tokens)
        all_tokens.update(tokens)

    if not all_tokens:
        return 0.0

    # Calculate pairwise overlap
    total_overlap = 0
    pair_count = 0

    for i in range(len(category_tokens)):
        for j in range(i + 1, len(category_tokens)):
            if category_tokens[i] and category_tokens[j]:
                intersection = len(category_tokens[i] & category_tokens[j])
                union = len(category_tokens[i] | category_tokens[j])
                if union > 0:
                    overlap = intersection / union
                    total_overlap += overlap
                    pair_count += 1

    if pair_count == 0:
        return 1.0  # No pairs means maximum diversity

    # Average overlap (invert for diversity)
    avg_overlap = total_overlap / pair_count
    diversity = 1.0 - avg_overlap

    return diversity


def _calculate_edit_distance_diversity(categories: List[str]) -> float:
    """Calculate diversity based on edit distances between categories."""

    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # Calculate average pairwise edit distance
    total_distance = 0
    pair_count = 0

    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            distance = levenshtein_distance(categories[i], categories[j])
            total_distance += distance
            pair_count += 1

    if pair_count == 0:
        return 0.0

    return total_distance / pair_count


def _calculate_length_variance_diversity(categories: List[str]) -> float:
    """Calculate diversity based on length variance."""
    lengths = [len(cat) for cat in categories]
    if len(lengths) <= 1:
        return 0.0

    return float(np.std(lengths))


def _calculate_character_diversity(categories: List[str]) -> float:
    """Calculate character-level diversity."""
    # Get all unique characters across categories
    all_chars = set()
    category_chars = []

    for cat in categories:
        chars = set(cat)
        category_chars.append(chars)
        all_chars.update(chars)

    if not all_chars:
        return 0.0

    # Calculate average character set diversity
    unique_char_ratios = []
    for chars in category_chars:
        if all_chars:
            ratio = len(chars) / len(all_chars)
            unique_char_ratios.append(ratio)

    if not unique_char_ratios:
        return 0.0

    # Return coefficient of variation
    mean_ratio = np.mean(unique_char_ratios)
    std_ratio = np.std(unique_char_ratios)

    if mean_ratio > 0:
        return min(1.0, std_ratio / mean_ratio)
    return 0.0


def _calculate_ttr(tokens: List[str]) -> float:
    """Calculate Type-Token Ratio."""
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    return unique_tokens / total_tokens if total_tokens > 0 else 0.0


def _calculate_root_ttr(tokens: List[str]) -> float:
    """Calculate Root Type-Token Ratio."""
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    if total_tokens > 0:
        return unique_tokens / math.sqrt(total_tokens)
    return 0.0


def _calculate_log_ttr(tokens: List[str]) -> float:
    """Calculate Logarithmic Type-Token Ratio."""
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    if total_tokens > 1 and unique_tokens > 1:
        return math.log(unique_tokens) / math.log(total_tokens)
    return 0.0


def _calculate_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Calculate Measure of Textual Lexical Diversity (MTLD).

    MTLD is calculated as the mean length of sequential word strings
    that maintain a TTR above the threshold.
    """
    if not tokens:
        return 0.0

    def mtld_forward(tokens: List[str], threshold: float) -> float:
        """Calculate MTLD in forward direction."""
        factor = 0.0
        factor_lengths = []
        start = 0

        for i in range(1, len(tokens) + 1):
            segment = tokens[start:i]
            ttr = _calculate_ttr(segment)

            if ttr <= threshold and i > start + 1:
                factor_lengths.append(i - start)
                factor += 1
                start = i

        # Handle remaining tokens
        if start < len(tokens):
            segment = tokens[start:]
            if len(segment) > 0:
                ttr = _calculate_ttr(segment)
                partial_factor = (1 - ttr) / (1 - threshold)
                factor += partial_factor
                factor_lengths.append(len(segment))

        if factor > 0:
            return len(tokens) / factor
        return float(len(set(tokens)))

    # Calculate both forward and backward
    mtld_fwd = mtld_forward(tokens, threshold)
    mtld_bwd = mtld_forward(tokens[::-1], threshold)

    # Return average
    return (mtld_fwd + mtld_bwd) / 2.0


def _calculate_msttr(tokens: List[str], window_size: int) -> float:
    """Calculate Mean Segmental Type-Token Ratio."""
    if len(tokens) < window_size:
        return _calculate_ttr(tokens)

    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        ttrs.append(_calculate_ttr(window))

    return np.mean(ttrs) if ttrs else 0.0


def _calculate_shannon_entropy(probabilities: np.ndarray, normalize: bool) -> float:
    """Calculate Shannon entropy."""
    # Remove zero probabilities
    probs = probabilities[probabilities > 0]

    if len(probs) == 0:
        return 0.0

    entropy = -np.sum(probs * np.log2(probs + EPSILON))

    if normalize and len(probs) > 1:
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probs))
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(entropy)


def _calculate_simpson_index(probabilities: np.ndarray) -> float:
    """Calculate Simpson's diversity index (1 - D)."""
    simpson_d = np.sum(probabilities**2)
    return float(1.0 - simpson_d)


def _calculate_brillouin_index(
    frequencies: np.ndarray, total: int, normalize: bool
) -> float:
    """Calculate Brillouin's diversity index."""
    if total <= 1:
        return 0.0

    # Calculate factorial term
    log_factorial_n = math.lgamma(total + 1)
    log_factorial_ni = sum(math.lgamma(freq + 1) for freq in frequencies)

    brillouin = (log_factorial_n - log_factorial_ni) / total

    if normalize and len(frequencies) > 1:
        # Maximum possible Brillouin index
        equal_freq = total // len(frequencies)
        remainder = total % len(frequencies)

        max_frequencies = np.array(
            [equal_freq + 1] * remainder + [equal_freq] * (len(frequencies) - remainder)
        )
        max_brillouin = _calculate_brillouin_index(max_frequencies, total, False)

        if max_brillouin > 0:
            brillouin = brillouin / max_brillouin

    return float(brillouin)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "calculate_semantic_diversity",
    "calculate_lexical_diversity",
    "calculate_token_diversity",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"
