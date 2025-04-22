"""
Utility functions for text analysis in the HHR project.

This module provides helper functions for text processing, analysis,
and frequency calculation with support for large datasets and caching.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from pamola_core.utils.logging import get_logger
from pamola_core.utils.io import ensure_directory, write_json, read_json

from pamola_core.utils.nlp.language import detect_language
from pamola_core.utils.nlp.tokenization import tokenize, calculate_word_frequencies as nlp_calculate_word_frequencies
from pamola_core.utils.nlp.tokenization import calculate_term_frequencies as nlp_calculate_term_frequencies
from pamola_core.utils.nlp.cache import get_cache, cache_function

# Configure logger
logger = get_logger(__name__)

# Get cache instances
file_cache = get_cache('file')
memory_cache = get_cache('memory')


def analyze_null_and_empty(df: pd.DataFrame, field_name: str, chunk_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze null and empty values in a text field.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the field
    field_name : str
        Name of the field to analyze
    chunk_size : int, optional
        Size of chunks for processing large DataFrames

    Returns:
    --------
    Dict[str, Any]
        Analysis results including:
        - total_records: Total number of records
        - null_values: Count and percentage of NULL values
        - empty_strings: Count and percentage of empty strings
        - whitespace_strings: Count and percentage of whitespace-only strings
        - actual_data: Count and percentage of records with meaningful data
    """
    # Total records
    total_records = len(df)

    # Process in chunks if dataframe is large and chunk_size is specified
    if chunk_size and total_records > chunk_size:
        return analyze_null_and_empty_in_chunks(df, field_name, chunk_size)

    # Count null values
    null_count = df[field_name].isna().sum()

    # Calculate empty strings (after filling NAs with empty string)
    empty_series = df[field_name].fillna("")
    empty_count = (empty_series == "").sum()

    # Calculate whitespace-only strings
    whitespace_count = 0
    if empty_count < total_records:
        whitespace_pattern = re.compile(r'^\s+$')
        whitespace_count = empty_series.str.match(whitespace_pattern).sum()

    # Calculate records with actual data
    actual_data_count = total_records - null_count - empty_count - whitespace_count

    return {
        "total_records": total_records,
        "null_values": {
            "count": int(null_count),
            "percentage": round((null_count / total_records) * 100, 2) if total_records > 0 else 0
        },
        "empty_strings": {
            "count": int(empty_count),
            "percentage": round((empty_count / total_records) * 100, 2) if total_records > 0 else 0
        },
        "whitespace_strings": {
            "count": int(whitespace_count),
            "percentage": round((whitespace_count / total_records) * 100, 2) if total_records > 0 else 0
        },
        "actual_data": {
            "count": int(actual_data_count),
            "percentage": round((actual_data_count / total_records) * 100, 2) if total_records > 0 else 0
        }
    }


def analyze_null_and_empty_in_chunks(df: pd.DataFrame, field_name: str, chunk_size: int) -> Dict[str, Any]:
    """
    Analyze null and empty values in a text field by processing the dataframe in chunks.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the field
    field_name : str
        Name of the field to analyze
    chunk_size : int
        Size of chunks to process

    Returns:
    --------
    Dict[str, Any]
        Aggregated analysis results
    """
    total_records = len(df)
    logger.info(f"Analyzing null and empty values for {field_name} in chunks (total records: {total_records})")

    # Initialize counters
    null_count = 0
    empty_count = 0
    whitespace_count = 0

    # Compile whitespace pattern once
    whitespace_pattern = re.compile(r'^\s+$')

    # Process in chunks
    for i in range(0, total_records, chunk_size):
        end_idx = min(i + chunk_size, total_records)
        chunk = df.iloc[i:end_idx]

        # Count nulls
        null_count += chunk[field_name].isna().sum()

        # Count empties
        empty_series = chunk[field_name].fillna("")
        empty_count += (empty_series == "").sum()

        # Count whitespace
        whitespace_count += empty_series.str.match(whitespace_pattern).sum()

    # Calculate actual data count
    actual_data_count = total_records - null_count - empty_count - whitespace_count

    return {
        "total_records": total_records,
        "null_values": {
            "count": int(null_count),
            "percentage": round((null_count / total_records) * 100, 2) if total_records > 0 else 0
        },
        "empty_strings": {
            "count": int(empty_count),
            "percentage": round((empty_count / total_records) * 100, 2) if total_records > 0 else 0
        },
        "whitespace_strings": {
            "count": int(whitespace_count),
            "percentage": round((whitespace_count / total_records) * 100, 2) if total_records > 0 else 0
        },
        "actual_data": {
            "count": int(actual_data_count),
            "percentage": round((actual_data_count / total_records) * 100, 2) if total_records > 0 else 0
        }
    }


@cache_function(ttl=3600, cache_type='memory')
def calculate_length_stats(texts: List[str], max_texts: Optional[int] = None,
                           use_cache: bool = False, cache_key: Optional[str] = None,
                           cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Calculate text length statistics with support for sampling and caching.

    Parameters:
    -----------
    texts : List[str]
        List of text values
    max_texts : int, optional
        Maximum number of texts to process for statistics
    use_cache : bool
        Whether to use caching
    cache_key : str, optional
        Key for caching results
    cache_dir : str or Path, optional
        Directory for cache files

    Returns:
    --------
    Dict[str, Any]
        Length statistics
    """
    # Check cache if enabled
    if use_cache and cache_key and cache_dir:
        cached_result = load_cached_result(cache_key, cache_dir, "length_stats")
        if cached_result:
            return cached_result

    # Sample if needed
    if max_texts and len(texts) > max_texts:
        import random
        random.seed(42)  # For reproducibility
        texts = random.sample(texts, max_texts)
        logger.info(f"Sampled {max_texts} texts for length statistics (from {len(texts)} total)")

    # Calculate lengths for non-empty strings
    lengths = [len(text) for text in texts if text]

    if not lengths:
        result = {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "length_distribution": {}
        }
        return result

    # Calculate basic statistics
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = sum(lengths) / len(lengths)
    median_length = sorted(lengths)[len(lengths) // 2]

    # Calculate standard deviation
    variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
    std_dev = variance ** 0.5

    # Create length distribution
    bins = [0, 50, 100, 150, 200, 250, float('inf')]
    bin_labels = ['<50', '50-100', '100-150', '150-200', '200-250', '>250']

    distribution = {label: 0 for label in bin_labels}

    for length in lengths:
        for i, upper_bound in enumerate(bins[1:], 0):
            if length < upper_bound:
                distribution[bin_labels[i]] += 1
                break

    # Convert counts to percentages
    total = len(lengths)
    distribution = {k: (v / total) * 100 for k, v in distribution.items()}

    result = {
        "min": min_length,
        "max": max_length,
        "mean": mean_length,
        "median": median_length,
        "std": std_dev,
        "length_distribution": distribution
    }

    # Cache result if enabled
    if use_cache and cache_key and cache_dir:
        save_cached_result(result, cache_key, cache_dir, "length_stats")

    return result


def chunk_texts(texts: List[str], chunk_size: int) -> List[List[str]]:
    """
    Split a list of texts into chunks for efficient processing.

    Parameters:
    -----------
    texts : List[str]
        List of text strings
    chunk_size : int
        Size of each chunk

    Returns:
    --------
    List[List[str]]
        List of text chunks
    """
    chunks = []
    for i in range(0, len(texts), chunk_size):
        end = min(i + chunk_size, len(texts))
        chunks.append(texts[i:end])
    return chunks


def process_texts_in_chunks(texts: List[str], process_func: callable,
                            chunk_size: int = 1000, **kwargs) -> List[Any]:
    """
    Process a large list of texts in chunks.

    Parameters:
    -----------
    texts : List[str]
        List of texts to process
    process_func : callable
        Function to apply to each chunk
    chunk_size : int
        Size of each chunk
    **kwargs
        Additional parameters to pass to process_func

    Returns:
    --------
    List[Any]
        Combined results from all chunks
    """
    # Use batch_process from the nlp.base module
    from core.utils.nlp.base import batch_process

    return batch_process(
        texts,
        process_func,
        chunk_size=chunk_size,
        **kwargs
    )


def merge_analysis_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge analysis results from multiple chunks.

    Parameters:
    -----------
    results_list : List[Dict[str, Any]]
        List of chunk results to merge

    Returns:
    --------
    Dict[str, Any]
        Merged analysis results
    """
    if not results_list:
        return {}

    # Start with the first result
    merged = results_list[0].copy()

    # If only one result, return it
    if len(results_list) == 1:
        return merged

    # Merge dictionaries with different strategies for different keys
    for result in results_list[1:]:
        # For numeric fields, aggregate
        for field in ["total_records", "null_count", "empty_count"]:
            if field in merged and field in result:
                merged[field] += result[field]

        # For distributions, combine counts
        for field in ["category_distribution", "aliases_distribution"]:
            if field in merged and field in result:
                for key, value in result[field].items():
                    merged[field][key] = merged[field].get(key, 0) + value

    # Recalculate percentages if needed
    if "total_records" in merged and merged["total_records"] > 0:
        for field in ["null_values", "empty_strings", "whitespace_strings", "actual_data"]:
            if field in merged:
                count = merged[field]["count"]
                merged[field]["percentage"] = round((count / merged["total_records"]) * 100, 2)

    return merged


def calculate_word_frequencies(texts: List[str], stop_words: Optional[Set[str]] = None,
                               min_word_length: int = 3, max_words: int = 100,
                               chunk_size: Optional[int] = None) -> Dict[str, int]:
    """
    Calculate word frequencies across multiple texts with support for large datasets.

    Parameters:
    -----------
    texts : List[str]
        List of text strings
    stop_words : Set[str], optional
        Set of stop words to exclude
    min_word_length : int
        Minimum word length to include
    max_words : int
        Maximum number of words to include in the result
    chunk_size : int, optional
        Size of chunks for processing large text lists

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping words to their frequencies
    """
    # Process in chunks if needed
    if chunk_size and len(texts) > chunk_size:
        # Use process_texts_in_chunks for parallel processing
        chunk_results = process_texts_in_chunks(
            texts,
            nlp_calculate_word_frequencies,
            chunk_size,
            stop_words=stop_words,
            min_word_length=min_word_length,
            max_words=None  # No limit for chunks
        )

        # Merge results
        word_counts = {}
        for result in chunk_results:
            for word, count in result.items():
                word_counts[word] = word_counts.get(word, 0) + count
    else:
        # Use the nlp module's function directly
        word_counts = nlp_calculate_word_frequencies(texts, stop_words, min_word_length)

    # Sort and limit to max_words
    sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    if max_words and len(sorted_counts) > max_words:
        sorted_counts = sorted_counts[:max_words]

    return dict(sorted_counts)


def calculate_term_frequencies(texts: List[str], language: str = "auto",
                               stop_words: Optional[Set[str]] = None,
                               min_word_length: int = 3, max_terms: int = 100,
                               chunk_size: Optional[int] = None) -> Dict[str, int]:
    """
    Calculate term frequencies with optional lemmatization and chunk processing.

    Parameters:
    -----------
    texts : List[str]
        List of text strings
    language : str
        Language code or "auto" for detection
    stop_words : Set[str], optional
        Set of stop words to exclude
    min_word_length : int
        Minimum word length to include
    max_terms : int
        Maximum number of terms to include in the result
    chunk_size : int, optional
        Size of chunks for processing large text lists

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping terms to their frequencies
    """
    # Process in chunks if needed
    if chunk_size and len(texts) > chunk_size:
        # Use process_texts_in_chunks for parallel processing
        chunk_results = process_texts_in_chunks(
            texts,
            nlp_calculate_term_frequencies,
            chunk_size,
            language=language,
            stop_words=stop_words,
            min_word_length=min_word_length,
            max_terms=None  # No limit for chunks
        )

        # Merge results
        term_counts = {}
        for result in chunk_results:
            for term, count in result.items():
                term_counts[term] = term_counts.get(term, 0) + count
    else:
        # Use the nlp module's function directly
        term_counts = nlp_calculate_term_frequencies(texts, language, stop_words, min_word_length)

    # Sort and limit to max_terms
    sorted_counts = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
    if max_terms and len(sorted_counts) > max_terms:
        sorted_counts = sorted_counts[:max_terms]

    return dict(sorted_counts)


def get_cache_key_for_texts(texts: List[str], operation: str, params: Dict[str, Any] = None) -> str:
    """
    Generate a cache key for text operations.

    Parameters:
    -----------
    texts : List[str]
        List of texts to process
    operation : str
        Name of the operation
    params : Dict[str, Any], optional
        Additional parameters that affect the result

    Returns:
    --------
    str
        Cache key
    """
    import hashlib

    # Create a hash of the text content
    hasher = hashlib.md5()

    # Add a sample of texts to the hash
    max_sample = 100  # Maximum number of texts to include in hash
    if len(texts) > max_sample:
        import random
        random.seed(42)
        sample = random.sample(texts, max_sample)
    else:
        sample = texts

    # Add texts to hash
    for text in sample:
        if text:
            hasher.update(text.encode('utf-8'))

    # Add operation and params
    hasher.update(operation.encode('utf-8'))

    if params:
        param_str = str(sorted(params.items()))
        hasher.update(param_str.encode('utf-8'))

    return f"text_utils_{operation}_{hasher.hexdigest()}"


def load_cached_result(cache_key: str, cache_dir: Union[str, Path], operation: str) -> Optional[Dict[str, Any]]:
    """
    Load cached result if available.

    Parameters:
    -----------
    cache_key : str
        Cache key for the results
    cache_dir : str or Path
        Directory where cache files are stored
    operation : str
        Name of the operation

    Returns:
    --------
    Dict[str, Any] or None
        Cached results or None if not available
    """
    # Ensure cache_dir is a Path
    cache_dir = Path(cache_dir)

    # Create full cache key with operation
    full_key = f"{cache_key}_{operation}"

    # Check if cache file exists
    cache_file = cache_dir / f"{full_key}.json"

    if not cache_file.exists():
        return None

    try:
        cached_data = read_json(cache_file)
        logger.debug(f"Loaded cache for {operation} from {cache_file}")
        return cached_data
    except Exception as e:
        logger.warning(f"Failed to load cache for {operation} from {cache_file}: {e}")
        return None


def save_cached_result(result: Dict[str, Any], cache_key: str,
                       cache_dir: Union[str, Path], operation: str) -> bool:
    """
    Save results to cache.

    Parameters:
    -----------
    result : Dict[str, Any]
        Results to cache
    cache_key : str
        Cache key for the results
    cache_dir : str or Path
        Directory where cache files should be stored
    operation : str
        Name of the operation

    Returns:
    --------
    bool
        True if successfully saved to cache, False otherwise
    """
    # Ensure cache_dir is a Path
    cache_dir = Path(cache_dir)

    # Create full cache key with operation
    full_key = f"{cache_key}_{operation}"

    # Ensure directory exists
    ensure_directory(cache_dir)

    # Save to cache
    cache_file = cache_dir / f"{full_key}.json"

    try:
        write_json(result, cache_file)
        logger.debug(f"Saved cache for {operation} to {cache_file}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save cache for {operation} to {cache_file}: {e}")
        return False


def detect_text_type(text: str) -> str:
    """
    Detect the potential type of content in a text field.

    This helps determine what kind of entity extractor is most appropriate.

    Parameters:
    -----------
    text : str
        Text to analyze

    Returns:
    --------
    str
        Detected type ("job", "organization", "transaction", "skill", "generic")
    """
    # Convert to lowercase for easier pattern matching
    text_lower = text.lower()

    # Patterns for different text types
    patterns = {
        "job": [
            r'\bengineer\b', r'\bdeveloper\b', r'\bmanager\b', r'\bdesigner\b',
            r'\banalyst\b', r'\bspecialist\b', r'\bdirector\b', r'\bconsultant\b',
            r'\bразработчик\b', r'\bменеджер\b', r'\bинженер\b', r'\bспециалист\b'
        ],
        "organization": [
            r'\binc\b', r'\bllc\b', r'\bcorp\b', r'\bcompany\b', r'\bgroup\b',
            r'\buniversity\b', r'\bcollege\b', r'\binstitute\b', r'\bschool\b',
            r'\bассоциация\b', r'\bуниверситет\b', r'\bинститут\b', r'\bкомпания\b'
        ],
        "transaction": [
            r'\bpayment\b', r'\btransfer\b', r'\bdeposit\b', r'\bwithdrawal\b',
            r'\binvoice\b', r'\bsalary\b', r'\brent\b', r'\butility\b',
            r'\bоплата\b', r'\bсчет\b', r'\bперевод\b', r'\bзарплата\b'
        ],
        "skill": [
            r'\bpython\b', r'\bjava\b', r'\bc\+\+\b', r'\bjavascript\b',
            r'\bexcel\b', r'\bword\b', r'\bpowerpoint\b', r'\bphotoshop\b',
            r'\bпрограммирование\b', r'\bанализ\b', r'\bпроектирование\b'
        ]
    }

    # Check each pattern type
    scores = {text_type: 0 for text_type in patterns}

    for text_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text_lower):
                scores[text_type] += 1

    # Get the type with highest score
    max_score = 0
    detected_type = "generic"

    for text_type, score in scores.items():
        if score > max_score:
            max_score = score
            detected_type = text_type

    return detected_type


def suggest_entity_type(texts: List[str], sample_size: int = 100) -> str:
    """
    Suggest the most appropriate entity type based on text content.

    Parameters:
    -----------
    texts : List[str]
        List of text values
    sample_size : int
        Number of texts to sample for analysis

    Returns:
    --------
    str
        Suggested entity type
    """
    if not texts:
        return "generic"

    # Sample texts if there are many
    if len(texts) > sample_size:
        import random
        random.seed(42)  # For reproducibility
        sampled_texts = random.sample([t for t in texts if t], min(sample_size, len([t for t in texts if t])))
    else:
        sampled_texts = [t for t in texts if t]

    if not sampled_texts:
        return "generic"

    # Count detected types
    type_counts = {}
    for text in sampled_texts:
        detected_type = detect_text_type(text)
        type_counts[detected_type] = type_counts.get(detected_type, 0) + 1

    # Get the most common type
    if not type_counts:
        return "generic"

    return max(type_counts.items(), key=lambda x: x[1])[0]