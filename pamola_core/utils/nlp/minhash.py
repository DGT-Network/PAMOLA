"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: MinHash Signature Generator
Description: Module for computing MinHash signatures for text fields to enable fast similarity comparison
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module implements MinHash signature generation for text data to enable efficient
similarity comparisons and diversity evaluation across records, typically for:
1. Long free-text fields (e.g., experience descriptions)
2. Multi-valued text fields (e.g., lists of skills)

Key features:
- Language-agnostic shingling with full Unicode support for mixed-language content
- Configurable signature parameters (permutation count, shingle size)
- Memory-efficient batch processing for large datasets
- Serialization/deserialization functionality for storage in CSV/databases
- Integration with datasketch library for efficient MinHash implementation
- Customizable text preprocessing pipeline

Implementation follows the PAMOLA.CORE operation framework with standardized interfaces
for input/output, processing, and result reporting.
"""

import csv
import hashlib
import logging
import os
import regex as re
from typing import Dict, List, Optional, Set

import pandas as pd
from datasketch import MinHash

from pamola_core.utils.nlp.stopwords import get_stopwords
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Configure module logger
logger = logging.getLogger(__name__)


def compute_minhash(text: str, num_perm: int = 128, shingle_size: int = 2) -> List[int]:
    """
    Compute a MinHash signature for the given text.

    Parameters:
    -----------
    text : str
        The input string to compute a signature for
    num_perm : int, optional
        Number of hash permutations (signature length); typical values: 64, 128, 256
    shingle_size : int, optional
        Size of n-grams (shingles); typical values: 2 (bigrams), 3 (trigrams)

    Returns:
    --------
    List[int]
        A list of integers representing the MinHash signature
    """
    # Create MinHash object with specified permutations
    minhash = MinHash(num_perm=num_perm)

    # Preprocess text
    cleaned_text = preprocess_text(text)

    # Generate shingles
    shingles = create_shingles(cleaned_text, shingle_size)

    # Update MinHash with shingles
    for shingle in shingles:
        minhash.update(shingle.encode('utf-8'))

    # Convert datasketch's hashvalues to standard Python integers
    # Avoiding direct iteration which can cause type errors with some versions
    result = []
    for i in range(num_perm):
        result.append(int(minhash.hashvalues[i]))

    return result


def preprocess_text(text: str, lowercase: bool = True,
                    remove_punctuation: bool = True,
                    remove_extra_spaces: bool = True,
                    remove_stopwords: bool = False,
                    languages: Optional[List[str]] = None) -> str:
    """
    Preprocess text for MinHash computation.

    Parameters:
    -----------
    text : str
        Input text to preprocess
    lowercase : bool, optional
        Whether to convert text to lowercase
    remove_punctuation : bool, optional
        Whether to remove punctuation
    remove_extra_spaces : bool, optional
        Whether to normalize whitespace
    remove_stopwords : bool, optional
        Whether to remove stopwords
    languages : List[str], optional
        Languages for stopword removal, defaults to ['en', 'ru'] if None

    Returns:
    --------
    str
        Preprocessed text
    """
    if text is None:
        return ""

    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    # Lowercase if requested
    if lowercase:
        text = text.lower()

    # Remove punctuation if requested
    if remove_punctuation:
        # Unicode-aware punctuation removal that preserves word boundaries
        text = re.sub(r'[\p{P}\p{S}]+', ' ', text, flags=re.UNICODE)

    # Remove extra whitespace if requested
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords if requested
    if remove_stopwords:
        if languages is None:
            languages = ['en', 'ru']  # Default languages

        # Get stopwords for the specified languages
        stop_words = get_stopwords(languages=languages)

        # Tokenize and filter out stopwords
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        text = ' '.join(filtered_words)

    return text


def create_shingles(text: str, shingle_size: int = 2,
                    method: str = 'character') -> Set[str]:
    """
    Create shingles (n-grams) from the input text.

    Parameters:
    -----------
    text : str
        Input text to create shingles from
    shingle_size : int, optional
        Size of n-grams (shingles)
    method : str, optional
        Shingling method: 'character', 'word', or 'unicode'

    Returns:
    --------
    Set[str]
        Set of shingles
    """
    if not text:
        return set()

    shingles = set()

    if method == 'word':
        # Word-based shingling
        words = text.split()
        for i in range(len(words) - shingle_size + 1):
            shingle = ' '.join(words[i:i + shingle_size])
            shingles.add(shingle)

    elif method == 'unicode':
        # Unicode-aware character shingling (considers grapheme clusters)
        # This is especially important for languages with complex characters
        import unicodedata

        # Normalize to composed form (NFC)
        text = unicodedata.normalize('NFC', text)

        # Now create character shingles
        for i in range(len(text) - shingle_size + 1):
            shingle = text[i:i + shingle_size]
            shingles.add(shingle)

    else:  # Default to character shingling
        for i in range(len(text) - shingle_size + 1):
            shingle = text[i:i + shingle_size]
            shingles.add(shingle)

    return shingles


def serialize_signature(signature: List[int], delimiter: str = ";") -> str:
    """
    Convert a signature vector into a string for storage.

    Parameters:
    -----------
    signature : List[int]
        MinHash signature as list of integers
    delimiter : str, optional
        Delimiter character for joining values

    Returns:
    --------
    str
        Serialized signature string
    """
    return delimiter.join(map(str, signature))


def deserialize_signature(signature_str: str, delimiter: str = ";") -> List[int]:
    """
    Convert a serialized signature string back into a list of integers.

    Parameters:
    -----------
    signature_str : str
        Serialized signature string
    delimiter : str, optional
        Delimiter character used in the serialized string

    Returns:
    --------
    List[int]
        MinHash signature as list of integers
    """
    if not signature_str:
        return []

    try:
        return [int(val) for val in signature_str.split(delimiter)]
    except ValueError as e:
        logger.error(f"Error deserializing signature: {e}")
        return []


def calculate_jaccard_similarity(signature1: List[int], signature2: List[int]) -> float:
    """
    Calculate Jaccard similarity between two MinHash signatures.

    Parameters:
    -----------
    signature1 : List[int]
        First MinHash signature
    signature2 : List[int]
        Second MinHash signature

    Returns:
    --------
    float
        Estimated Jaccard similarity (0-1 range)
    """
    if not signature1 or not signature2:
        return 0.0

    if len(signature1) != len(signature2):
        logger.warning("Signatures have different lengths, similarity may be inaccurate")
        min_len = min(len(signature1), len(signature2))
        signature1 = signature1[:min_len]
        signature2 = signature2[:min_len]

    # Count how many hash values are equal
    matches = sum(1 for i, j in zip(signature1, signature2) if i == j)

    # Return the estimated Jaccard similarity
    return matches / len(signature1)


def process_csv_file(input_path: str, output_path: str, field_name: str,
                     id_field: str, num_perm: int = 128, shingle_size: int = 2,
                     batch_size: int = 1000, preprocessing_params: Optional[Dict] = None) -> OperationResult:
    """
    Process a CSV file, computing MinHash signatures for a specified field.

    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to output CSV file
    field_name : str
        Name of the field to compute signatures for
    id_field : str
        Name of the ID field to include in output
    num_perm : int, optional
        Number of hash permutations
    shingle_size : int, optional
        Size of n-grams (shingles)
    batch_size : int, optional
        Number of records to process in each batch
    preprocessing_params : Dict, optional
        Parameters for text preprocessing

    Returns:
    --------
    OperationResult
        Result of the operation
    """
    result = OperationResult(status=OperationStatus.PENDING)

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Default preprocessing parameters if none provided
        if preprocessing_params is None:
            preprocessing_params = {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_extra_spaces': True,
                'remove_stopwords': False,
                'languages': ['en', 'ru']
            }

        # Initialize counters
        total_records = 0
        processed_records = 0
        error_records = 0

        # Open output file and write header
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([id_field, 'minhash_signature'])

            # Process input file in batches
            for i, chunk in enumerate(pd.read_csv(input_path, chunksize=batch_size)):
                logger.info(f"Processing batch {i + 1}")

                batch_results = []
                total_records += len(chunk)

                for _, row in chunk.iterrows():
                    try:
                        # Get the ID and text field
                        record_id = row[id_field]
                        text = row.get(field_name, "")

                        # Skip empty text
                        if pd.isna(text) or text == "":
                            logger.debug(f"Skipping empty text for ID {record_id}")
                            continue

                        # Compute MinHash signature
                        signature = compute_minhash(
                            text=text,
                            num_perm=num_perm,
                            shingle_size=shingle_size
                        )

                        # Serialize signature for storage
                        serialized_signature = serialize_signature(signature)

                        # Add to batch results
                        batch_results.append([record_id, serialized_signature])
                        processed_records += 1

                    except Exception as e:
                        logger.error(f"Error processing record {row.get(id_field, 'unknown')}: {e}")
                        error_records += 1

                # Write batch results to output file
                writer.writerows(batch_results)

                # Provide progress update
                logger.info(f"Processed {processed_records} records so far")

        # Set success status and add metrics
        result.status = OperationStatus.SUCCESS
        result.add_metric("total_records", total_records)
        result.add_metric("processed_records", processed_records)
        result.add_metric("error_records", error_records)
        result.add_metric("success_rate", processed_records / total_records if total_records > 0 else 0)

        logger.info(f"MinHash generation complete. Processed {processed_records} records.")
        return result

    except Exception as e:
        error_message = f"Error processing CSV file: {str(e)}"
        logger.exception(error_message)
        result.status = OperationStatus.ERROR
        result.error_message = error_message
        return result


def estimate_optimal_num_perm(desired_error: float = 0.05) -> int:
    """
    Estimate the optimal number of permutations for a desired error level.

    Parameters:
    -----------
    desired_error : float, optional
        Desired error level (e.g., 0.05 for 5% error)

    Returns:
    --------
    int
        Recommended number of permutations
    """
    # Based on the formula: error ≈ 1/sqrt(num_perm)
    num_perm = int(1 / (desired_error ** 2))

    # Round to nearest standard value (multiple of 32 for efficiency)
    standard_values = [32, 64, 128, 256, 512]

    for value in standard_values:
        if value >= num_perm:
            return value

    return standard_values[-1]  # Return largest standard value if none are sufficient


def get_cache_key(text: str, num_perm: int, shingle_size: int) -> str:
    """
    Generate a cache key for a MinHash computation.

    Parameters:
    -----------
    text : str
        Input text
    num_perm : int
        Number of permutations
    shingle_size : int
        Shingle size

    Returns:
    --------
    str
        Cache key string
    """
    # Create a hash of the text and parameters
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f"minhash_{text_hash}_{num_perm}_{shingle_size}"


# Function with caching support
def cached_compute_minhash(text: str, num_perm: int = 128, shingle_size: int = 2,
                           use_cache: bool = True) -> List[int]:
    """
    Compute MinHash with caching support.

    Parameters:
    -----------
    text : str
        Input text
    num_perm : int, optional
        Number of permutations
    shingle_size : int, optional
        Shingle size
    use_cache : bool, optional
        Whether to use caching

    Returns:
    --------
    List[int]
        MinHash signature
    """
    if not use_cache:
        return compute_minhash(text, num_perm, shingle_size)

    # Generate cache key
    cache_key = get_cache_key(text, num_perm, shingle_size)

    # Check cache
    cached_result = operation_cache.get_cache(
        cache_key=cache_key,
        operation_type="MinHash"
    )

    if cached_result:
        logger.debug("Using cached MinHash signature")
        return cached_result.get("signature", [])

    # Compute MinHash signature
    signature = compute_minhash(text, num_perm, shingle_size)

    # Cache the result
    operation_cache.save_cache(
        data={"signature": signature},
        cache_key=cache_key,
        operation_type="MinHash"
    )

    return signature


def batch_compute_minhash(texts: List[str], num_perm: int = 128,
                          shingle_size: int = 2) -> List[List[int]]:
    """
    Compute MinHash signatures for a batch of texts.

    Parameters:
    -----------
    texts : List[str]
        List of input strings
    num_perm : int, optional
        Number of permutations
    shingle_size : int, optional
        Shingle size

    Returns:
    --------
    List[List[int]]
        List of MinHash signatures
    """
    return [compute_minhash(text, num_perm, shingle_size) for text in texts]


def create_minhash_generator():
    """
    Create and return a function to generate MinHash signatures with reasonable defaults.

    Returns:
    --------
    Callable
        Function for generating MinHash signatures
    """

    def generate_minhash(text, num_perm=128, shingle_size=2):
        return compute_minhash(text, num_perm, shingle_size)

    return generate_minhash