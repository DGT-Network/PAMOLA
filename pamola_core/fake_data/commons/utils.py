"""
Utility functions for the fake_data module.

This module provides enhanced utility functions for common tasks in the fake_data module,
including string processing, dictionary loading, deterministic generation, etc.
The functions are designed for performance, error resilience, and seamless integration
with the core utils package.

Key functionality:
- String normalization and manipulation
- Consistent deterministic value generation
- Dictionary loading and caching
- Language and gender detection
- Format validation for various data types
"""

import hashlib
import json
import logging
import os
import random
import re
import string
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import pandas as pd
import numpy as np

# Import from core utils
from pamola_core.utils import io
from pamola_core.utils import progress

# Configure logger
logger = logging.getLogger(__name__)

# Try to import language module with graceful degradation
try:
    from pamola_core.utils.nlp import language

    LANGUAGE_MODULE_AVAILABLE = True
except ImportError:
    logger.warning("core.utils.nlp.language not available, using basic language detection")
    LANGUAGE_MODULE_AVAILABLE = False

# Dictionary cache
_dictionary_cache = {}


@lru_cache(maxsize=128)
def normalize_string(value: Optional[str] = None, keep_case: bool = False,
                     remove_punctuation: bool = False, remove_spaces: bool = False) -> str:
    """
    Normalizes a string by removing extra spaces, punctuation, etc.

    Parameters:
    -----------
    value : Optional[str]
        String to normalize (can be None)
    keep_case : bool
        Whether to preserve case (default: False)
    remove_punctuation : bool
        Whether to remove punctuation (default: False)
    remove_spaces : bool
        Whether to remove all spaces (default: False)

    Returns:
    --------
    str
        Normalized string
    """
    # Handle None or empty case
    if value is None or value == "":
        return ""

    # Convert to string if not already
    result = str(value).strip()

    # Replace multiple spaces with single space
    if not remove_spaces:
        result = re.sub(r'\s+', ' ', result)
    else:
        result = re.sub(r'\s+', '', result)

    # Remove punctuation if requested
    if remove_punctuation:
        result = re.sub(r'[^\w\s]', '', result)

    # Convert to lowercase if not keeping case
    if not keep_case:
        result = result.lower()

    return result


def hash_value(value: Any, salt: str = "", algorithm: str = "sha256") -> str:
    """
    Creates a hash of a value using specified algorithm.

    Parameters:
    -----------
    value : Any
        Value to hash
    salt : str
        Salt to add to the value before hashing
    algorithm : str
        Hashing algorithm to use ('sha256', 'md5', 'sha1', 'sha512')

    Returns:
    --------
    str
        Hexadecimal hash string

    Raises:
    -------
    ValueError
        If algorithm is not supported
    """
    if value is None:
        return ""

    try:
        # Convert value to string
        value_str = str(value)

        # Add salt
        salted_value = f"{value_str}{salt}"

        # Create hash based on selected algorithm
        if algorithm == "sha256":
            hash_obj = hashlib.sha256(salted_value.encode())
        elif algorithm == "md5":
            hash_obj = hashlib.md5(salted_value.encode())
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1(salted_value.encode())
        elif algorithm == "sha512":
            hash_obj = hashlib.sha512(salted_value.encode())
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}. Supported: sha256, md5, sha1, sha512")

        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing value: {e}")
        raise


def generate_deterministic_value(seed_value: Any, length: int = 10,
                                 chars: Optional[str] = None,
                                 seed_algorithm: str = "sha256") -> str:
    """
    Generates a deterministic value based on a seed.

    Parameters:
    -----------
    seed_value : Any
        Seed for the random generator
    length : int
        Length of the generated value
    chars : Optional[str]
        Character set to use (default: ascii_letters + digits)
    seed_algorithm : str
        Algorithm to use for hashing the seed value

    Returns:
    --------
    str
        Generated deterministic value
    """
    try:
        # Get a deterministic seed from the seed value
        seed_hash = hash_value(seed_value, algorithm=seed_algorithm)

        # Convert hexadecimal hash to integer for random seed
        seed_int = int(seed_hash, 16)

        # Create a predictable random generator
        rng = random.Random(seed_int)

        # Define character set if not provided
        if chars is None:
            chars = string.ascii_letters + string.digits

        # Generate random value
        return ''.join(rng.choice(chars) for _ in range(length))
    except Exception as e:
        logger.error(f"Error generating deterministic value: {e}")
        raise


def detect_language(text: Optional[str], default_language: str = "en") -> str:
    """
    Detects the language of a text.

    This function is a wrapper around the language detection functionality
    in core.utils.nlp.language. If that module is not available, it falls
    back to a basic implementation.

    Parameters:
    -----------
    text : Optional[str]
        Text to analyze (can be None)
    default_language : str
        Default language to return if detection fails

    Returns:
    --------
    str
        Detected language code ('en', 'ru', etc.) or default language if detection fails
    """
    if text is None or not text:
        return default_language

    try:
        # Use language detection from pamola_core.utils.nlp if available
        if LANGUAGE_MODULE_AVAILABLE:
            return language.detect_language(text, default_language)

        # Basic fallback implementation
        # Check for Cyrillic characters (likely Russian)
        if re.search(r'[а-яА-ЯёЁ]', text):
            return "ru"

        # Check for Latin characters (likely English or other Latin-based language)
        if re.search(r'[a-zA-Z]', text):
            return "en"

        # Default to specified default language
        return default_language
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return default_language


def detect_gender_by_dictionary(name: Optional[str],
                                gender_dict: Optional[Dict[str, str]] = None,
                                default_gender: Optional[str] = None) -> Optional[str]:
    """
    Determines gender from a name using a dictionary lookup.

    Parameters:
    -----------
    name : Optional[str]
        Name to analyze (can be None)
    gender_dict : Optional[Dict[str, str]]
        Dictionary mapping normalized names to genders ('M', 'F')
    default_gender : Optional[str]
        Default gender to return if detection fails

    Returns:
    --------
    Optional[str]
        Detected gender ('M', 'F') or default_gender if detection fails
    """
    if name is None or not name or not gender_dict:
        return default_gender

    # Normalize name for lookup
    normalized_name = normalize_string(name)

    # Look up in dictionary
    gender = gender_dict.get(normalized_name)

    return gender if gender is not None else default_gender


def load_dictionary(source: Union[str, Path, Dict],
                    dictionary_type: str = "general",
                    base_path: Optional[Path] = None,
                    encoding: str = "utf-8",
                    cache: bool = True) -> List[str]:
    """
    Loads a dictionary of values from various sources.

    Parameters:
    -----------
    source : Union[str, Path, Dict]
        Source for dictionary data. Can be:
        - Path to a file (as string or Path)
        - Dictionary name to be resolved relative to base_path
        - Directly provided dictionary data
    dictionary_type : str
        Type of dictionary for logging and caching
    base_path : Optional[Path]
        Base path for resolving dictionary names
    encoding : str
        File encoding for text files
    cache : bool
        Whether to cache the dictionary

    Returns:
    --------
    List[str]
        List of values in the dictionary

    Raises:
    -------
    ValueError
        If dictionary format is not supported or dictionary can't be found
    """
    global _dictionary_cache

    # Generate cache key based on source and type
    cache_key = f"{dictionary_type}:{source}"

    # Check cache first if enabled
    if cache and cache_key in _dictionary_cache:
        logger.debug(f"Using cached dictionary: {cache_key}")
        return _dictionary_cache[cache_key]

    try:
        result = []

        # Case 1: Source is a dictionary or list - use directly
        if isinstance(source, dict):
            result = list(source.values())
        elif isinstance(source, list):
            result = source

        # Case 2: Source is a file path (str or Path)
        elif isinstance(source, (str, Path)):
            file_path = None

            # If source is a string but not a path, try to resolve it relative to base_path
            if isinstance(source, str) and not os.path.exists(source):
                if base_path:
                    resolved_path = base_path / source
                    if os.path.exists(resolved_path):
                        file_path = resolved_path

                    # Try with common extensions if no extension in source
                    elif '.' not in source:
                        for ext in ['.txt', '.csv', '.json']:
                            ext_path = base_path / f"{source}{ext}"
                            if os.path.exists(ext_path):
                                file_path = ext_path
                                break

            # Source is an existing file path
            elif os.path.exists(source):
                file_path = source

            # If we found a file path, load it
            if file_path:
                file_path = Path(file_path)
                extension = file_path.suffix.lower()

                # Load based on file extension
                if extension in ['.txt', '.csv']:
                    # Read as a simple text file with one entry per line
                    with open(file_path, 'r', encoding=encoding) as f:
                        result = [line.strip() for line in f if line.strip()]
                elif extension == '.json':
                    # Read as JSON
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                result = data
                            elif isinstance(data, dict):
                                result = list(data.values())
                            else:
                                raise ValueError(f"Unsupported JSON structure in {file_path}")
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON from {file_path}")
                        raise
                else:
                    raise ValueError(f"Unsupported file extension: {extension}")
            else:
                raise ValueError(f"Dictionary source not found: {source}")
        else:
            raise ValueError(f"Unsupported dictionary source type: {type(source)}")

        # Cache the result if requested
        if cache and result:
            _dictionary_cache[cache_key] = result

        return result
    except Exception as e:
        logger.error(f"Error loading dictionary: {e}")
        raise


def find_dictionary_file(dictionary_name: str, base_dirs: List[Union[str, Path]] = None,
                         suffixes: List[str] = None) -> Optional[Path]:
    """
    Finds a dictionary file in standard locations.

    Parameters:
    -----------
    dictionary_name : str
        Name of the dictionary to find
    base_dirs : List[Union[str, Path]]
        List of base directories to search (if None, uses standard locations)
    suffixes : List[str]
        List of file suffixes to try (if None, uses standard suffixes)

    Returns:
    --------
    Optional[Path]
        Path to the dictionary file if found, None otherwise
    """
    # Default base directories
    if base_dirs is None:
        base_dirs = [
            Path("DATA/external_dictionaries/fake"),  # Standard project location
            Path(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))).parent / "DATA" / "external_dictionaries" / "fake",
            Path.cwd() / "DATA" / "external_dictionaries" / "fake",
            Path.home() / "DATA" / "external_dictionaries" / "fake"
        ]

    # Convert all to Path objects
    base_dirs = [Path(d) for d in base_dirs]

    # Default suffixes
    if suffixes is None:
        suffixes = ['.txt', '.csv', '.json', '']

    # Try each base directory
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        # Try each suffix
        for suffix in suffixes:
            # Create potential file path
            file_path = base_dir / f"{dictionary_name}{suffix}"

            if file_path.exists():
                logger.debug(f"Found dictionary file: {file_path}")
                return file_path

    logger.debug(f"Dictionary file not found: {dictionary_name}")
    return None


def validate_email(email: str) -> bool:
    """
    Validates an email address format.

    Parameters:
    -----------
    email : str
        Email address to validate

    Returns:
    --------
    bool
        True if the email address is valid, False otherwise
    """
    if not email:
        return False

    # Use a simple regex for basic validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def validate_phone(phone: str, region: str = "INTL") -> bool:
    """
    Validates a phone number format.

    Parameters:
    -----------
    phone : str
        Phone number to validate
    region : str
        Region code for validation rules

    Returns:
    --------
    bool
        True if the phone number is valid, False otherwise
    """
    if not phone:
        return False

    # Strip non-digit characters for validation
    digits_only = re.sub(r'\D', '', phone)

    # Must have at least some digits
    if not digits_only:
        return False

    # Different validation rules based on region
    if region == "US":
        # US phone numbers should have 10 or 11 digits (if including country code)
        return len(digits_only) in [10, 11]
    elif region == "RU":
        # Russian phone numbers typically have 11 digits
        return len(digits_only) == 11
    else:
        # International: at least 7 digits, not more than 15 (E.164 standard)
        return 7 <= len(digits_only) <= 15


def format_phone_number(digits: str, region: str = "INTL", formatting: bool = True) -> str:
    """
    Formats a phone number according to regional standards.

    Parameters:
    -----------
    digits : str
        Phone number digits (can include non-digit characters)
    region : str
        Region code for formatting rules
    formatting : bool
        Whether to apply formatting (if False, just returns digits)

    Returns:
    --------
    str
        Formatted phone number
    """
    if not digits:
        return ""

    # Strip non-digit characters
    digits_only = re.sub(r'\D', '', digits)

    if not formatting:
        return digits_only

    # Apply region-specific formatting
    if region == "US":
        if len(digits_only) == 10:
            return f"({digits_only[0:3]}) {digits_only[3:6]}-{digits_only[6:10]}"
        elif len(digits_only) == 11 and digits_only[0] == '1':
            return f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:11]}"
    elif region == "RU":
        if len(digits_only) == 11 and digits_only[0] in ['7', '8']:
            return f"+7 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:9]}-{digits_only[9:11]}"

    # Default international format with country code
    if len(digits_only) > 0:
        return f"+{digits_only}"

    # Return original if formatting fails
    return digits


def create_progress_bar(total: int, description: str, unit: str = "items") -> progress.ProgressBar:
    """
    Creates a standardized progress bar for fake data operations.

    Parameters:
    -----------
    total : int
        Total number of items to process
    description : str
        Description of the operation
    unit : str
        Unit of items being processed

    Returns:
    --------
    progress.ProgressBar
        Progress bar instance
    """
    return progress.ProgressBar(total=total, description=description, unit=unit)


def save_metrics(metrics: Dict[str, Any], path: Union[str, Path]) -> Path:
    """
    Saves operation metrics to a JSON file.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics data to save
    path : Union[str, Path]
        Path to save the metrics

    Returns:
    --------
    Path
        Path to the saved metrics file
    """
    path = Path(path)
    io.ensure_directory(path.parent)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return path


def clear_caches() -> None:
    """
    Clears all caches used by the utils module.
    """
    global _dictionary_cache
    _dictionary_cache = {}
    normalize_string.cache_clear()
    logger.debug("All utils caches cleared")