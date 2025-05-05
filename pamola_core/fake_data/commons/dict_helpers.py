"""
Dictionary handling utilities for fake data generation.

This module provides functions for loading, processing, and managing
dictionaries used in the fake data generation process.
"""

import os
import random
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, BinaryIO, TextIO

import pandas as pd

from pamola_core.utils import io, logging

# Import embedded dictionaries
from pamola_core.fake_data.dictionaries import names, domains, phones, addresses, organizations

# Configure logger
logger = logging.get_logger("pamola_core.fake_data.commons.dict_helpers")

# Dictionary cache for performance
_dictionary_cache = {}

# Dictionary validation patterns
_DICTIONARY_PATTERNS = {
    "default": r"^.+$",  # Any non-empty line
    "name": r"^[A-Za-zА-Яа-яЁё\s\-']+$",  # Names with spaces, hyphens, apostrophes
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",  # Email
    "domain": r"^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$",  # Domain
    "phone": r"^\+?[0-9\s\-\(\)]+$",  # Phone numbers
    "address": r"^[A-Za-zА-Яа-яЁё0-9\s\-\.,/]+$"  # Addresses
}


def find_dictionary(
        dictionary_path: Optional[Union[str, Path]] = None,
        language: str = "ru",
        gender: Optional[str] = None,
        name_type: str = "first_name",
        dict_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Finds an appropriate dictionary file based on parameters.

    Parameters:
    -----------
    dictionary_path : Optional[Union[str, Path]]
        Explicit path to dictionary file (takes precedence if provided)
    language : str
        Language code (default: "ru")
    gender : Optional[str]
        Gender code ("M" or "F", or None for gender-neutral)
    name_type : str
        Type of names ("first_name", "last_name", "middle_name", "full_name")
    dict_dir : Optional[Union[str, Path]]
        Base directory for dictionaries (defaults to DATA/external_dictionaries/fake)

    Returns:
    --------
    Optional[Path]
        Path to the dictionary file or None if not found
    """
    # If explicit path provided, use it directly
    if dictionary_path:
        path = Path(dictionary_path)
        if path.exists():
            return path
        logger.warning(f"Specified dictionary path not found: {path}")

    # Otherwise, try to find the dictionary based on conventions
    if not dict_dir:
        # Default location for dictionaries
        dict_dir = Path("DATA/external_dictionaries/fake")
    else:
        dict_dir = Path(dict_dir)

    # Normalize parameters
    language = language.lower()
    if gender:
        gender = gender.lower()

    # Map name_type to filename part
    name_type_map = {
        "first_name": "first_names",
        "last_name": "last_names",
        "middle_name": "middle_names",
        "full_name": "names"
    }

    name_part = name_type_map.get(name_type, name_type)

    # Try different filename patterns
    patterns = []

    # With gender
    if gender:
        patterns.append(f"{language}_{gender}_{name_part}.txt")
        patterns.append(f"{language}_{gender[0]}_{name_part}.txt")

    # Without gender
    patterns.append(f"{language}_{name_part}.txt")

    # Try each pattern
    for pattern in patterns:
        path = dict_dir / pattern
        if path.exists():
            return path

    # If still not found, try more general patterns
    if name_type != "full_name":
        # Try with full names
        path = find_dictionary(
            language=language,
            gender=gender,
            name_type="full_name",
            dict_dir=dict_dir
        )
        if path:
            return path

    logger.warning(f"Could not find dictionary for {language}, {gender}, {name_type}")
    return None


def load_dictionary_from_text(
        path: Union[str, Path],
        cache: bool = True,
        encoding: str = "utf-8",
        validate_pattern: Optional[str] = None,
        min_length: int = 2,
        max_length: int = 100
) -> List[str]:
    """
    Loads a simple text dictionary with one item per line, with enhanced validation.

    Parameters:
    -----------
    path : Union[str, Path]
        Path to the dictionary file
    cache : bool
        Whether to cache the dictionary
    encoding : str
        File encoding
    validate_pattern : Optional[str]
        Regex pattern for validating dictionary entries
    min_length : int
        Minimum allowed length for entries
    max_length : int
        Maximum allowed length for entries

    Returns:
    --------
    List[str]
        List of items from the dictionary
    """
    path = Path(path)
    cache_key = str(path)

    # Check cache first
    if cache and cache_key in _dictionary_cache:
        logger.debug(f"Using cached dictionary: {cache_key}")
        return _dictionary_cache[cache_key].copy()

    logger.debug(f"Loading dictionary from {path}")

    try:
        # Use io utility for file operations if possible
        try:
            # Try using project's IO utility
            content = io.read_text(path, encoding=encoding)
            lines = content.splitlines()
        except (AttributeError, ImportError):
            # Fallback to standard file operations
            with open(path, 'r', encoding=encoding) as f:
                lines = f.readlines()

        # Process lines
        items = []
        invalid_count = 0

        # Get validation pattern if needed
        if validate_pattern and validate_pattern in _DICTIONARY_PATTERNS:
            pattern = _DICTIONARY_PATTERNS[validate_pattern]
        elif validate_pattern:
            pattern = validate_pattern
        else:
            pattern = _DICTIONARY_PATTERNS["default"]

        # Process each line
        for line in lines:
            # Strip whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check length constraints
            if len(line) < min_length or len(line) > max_length:
                invalid_count += 1
                continue

            # Validate against pattern if provided
            if validate_pattern and not re.match(pattern, line):
                invalid_count += 1
                continue

            items.append(line)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid entries in {path}")

        # Cache the results
        if cache:
            logger.debug(f"Caching dictionary with {len(items)} items: {cache_key}")
            _dictionary_cache[cache_key] = items.copy()

        return items
    except Exception as e:
        logger.error(f"Error loading dictionary {path}: {e}")
        return []


def load_dictionary_with_stats(
        path: Union[str, Path],
        language: str = "ru",
        gender: Optional[str] = None,
        name_type: str = "first_name",
        cache: bool = True,
        validate_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads a dictionary and returns it with statistics.

    Parameters:
    -----------
    path : Union[str, Path]
        Path to the dictionary file
    language : str
        Language code
    gender : Optional[str]
        Gender code ("M" or "F")
    name_type : str
        Type of names
    cache : bool
        Whether to cache the dictionary
    validate_pattern : Optional[str]
        Regex pattern for validating dictionary entries

    Returns:
    --------
    Dict[str, Any]
        Dictionary with items and statistics
    """
    items = load_dictionary_from_text(
        path,
        cache=cache,
        validate_pattern=validate_pattern
    )

    if not items:
        return {
            "items": [],
            "count": 0,
            "language": language,
            "gender": gender,
            "name_type": name_type,
            "source": str(path),
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0
        }

    # Calculate additional statistics
    item_lengths = [len(item) for item in items]
    avg_length = sum(item_lengths) / len(item_lengths) if item_lengths else 0

    return {
        "items": items,
        "count": len(items),
        "language": language,
        "gender": gender,
        "name_type": name_type,
        "source": str(path),
        "avg_length": avg_length,
        "min_length": min(item_lengths) if item_lengths else 0,
        "max_length": max(item_lengths) if item_lengths else 0
    }


def clear_dictionary_cache():
    """
    Clears the dictionary cache.
    """
    global _dictionary_cache
    _dictionary_cache = {}
    logger.debug("Dictionary cache cleared")

    # Also clear caches in embedded dictionaries
    names.clear_cache()
    domains.clear_cache()
    phones.clear_cache()
    addresses.clear_cache()
    organizations.clear_cache()
    logger.debug("Embedded dictionary caches cleared")


def get_random_items(dictionary: List[str], count: int, seed: Optional[int] = None) -> List[str]:
    """
    Gets random items from a dictionary.

    Parameters:
    -----------
    dictionary : List[str]
        List of items to choose from
    count : int
        Number of items to get
    seed : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    List[str]
        List of randomly selected items
    """
    if not dictionary:
        return []

    if count <= 0:
        return []

    # Create a local random generator with the seed
    local_random = random.Random(seed)

    # If requesting more items than available, sample with replacement
    if count > len(dictionary):
        return [local_random.choice(dictionary) for _ in range(count)]

    # Otherwise sample without replacement
    return local_random.sample(dictionary, count)


def validate_dictionary(
        dictionary: List[str],
        dict_type: str = "default",
        min_length: int = 2,
        max_length: int = 100
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validates and filters dictionary entries.

    Parameters:
    -----------
    dictionary : List[str]
        List of dictionary entries to validate
    dict_type : str
        Type of dictionary for validation ("name", "email", "domain", "phone", "address")
    min_length : int
        Minimum allowed length for entries
    max_length : int
        Maximum allowed length for entries

    Returns:
    --------
    Tuple[List[str], Dict[str, Any]]
        Tuple containing (valid_entries, validation_stats)
    """
    pattern = _DICTIONARY_PATTERNS.get(dict_type, _DICTIONARY_PATTERNS["default"])
    valid_entries = []
    stats = {
        "original_count": len(dictionary),
        "valid_count": 0,
        "invalid_count": 0,
        "too_short_count": 0,
        "too_long_count": 0,
        "pattern_mismatch_count": 0
    }

    import re
    pattern_obj = re.compile(pattern)

    for entry in dictionary:
        # Check length constraints
        if len(entry) < min_length:
            stats["too_short_count"] += 1
            stats["invalid_count"] += 1
            continue

        if len(entry) > max_length:
            stats["too_long_count"] += 1
            stats["invalid_count"] += 1
            continue

        # Check pattern
        if not pattern_obj.match(entry):
            stats["pattern_mismatch_count"] += 1
            stats["invalid_count"] += 1
            continue

        valid_entries.append(entry)
        stats["valid_count"] += 1

    return valid_entries, stats


def load_multi_dictionary(
        dict_type: str,
        params: Dict[str, Any],
        fallback_to_embedded: bool = True
) -> List[str]:
    """
    Loads a dictionary based on parameters, with fallback to embedded dictionaries.

    Parameters:
    -----------
    dict_type : str
        Type of dictionary ("name", "email", "domain", "phone", "address", "organization")
    params : Dict[str, Any]
        Parameters for dictionary loading (language, gender, etc.)
    fallback_to_embedded : bool
        Whether to fall back to embedded dictionaries if external not found

    Returns:
    --------
    List[str]
        List of dictionary entries
    """
    result = []

    # Try to load from external dictionary first
    if 'path' in params:
        path = params['path']
        if path:
            try:
                result = load_dictionary_from_text(
                    path,
                    cache=params.get('cache', True),
                    validate_pattern=dict_type
                )
                if result:
                    logger.debug(f"Loaded {len(result)} items from external dictionary: {path}")
                    return result
            except Exception as e:
                logger.warning(f"Failed to load external dictionary {path}: {e}")

    # If external dictionary not found or empty, try to find dictionary by convention
    if not result and 'dict_dir' in params:
        dict_dir = params['dict_dir']
        language = params.get('language', 'en')
        gender = params.get('gender')
        name_type = params.get('name_type', 'first_name')

        dict_path = find_dictionary(
            language=language,
            gender=gender,
            name_type=name_type,
            dict_dir=dict_dir
        )

        if dict_path:
            try:
                result = load_dictionary_from_text(
                    dict_path,
                    cache=params.get('cache', True),
                    validate_pattern=dict_type
                )
                if result:
                    logger.debug(f"Loaded {len(result)} items from conventional dictionary: {dict_path}")
                    return result
            except Exception as e:
                logger.warning(f"Failed to load conventional dictionary {dict_path}: {e}")

    # Fall back to embedded dictionaries if allowed
    if fallback_to_embedded:
        logger.debug(f"Falling back to embedded dictionary for {dict_type}")

        language = params.get('language', 'en')
        gender = params.get('gender')
        name_type = params.get('name_type', 'first_name')

        if dict_type == "name":
            result = get_embedded_dictionary(name_type, gender, language)
        elif dict_type == "domain":
            result = domains.get_common_email_domains()
        elif dict_type == "phone":
            country_code = params.get('country', 'US')
            result = phones.get_area_codes(country_code)
        elif dict_type == "address":
            country_code = params.get('country', 'US')
            component = params.get('component', 'street')
            result = addresses.get_address_component(country_code, component)
        elif dict_type == "organization":
            country_code = params.get('country', 'US')
            org_type = params.get('org_type', 'business')
            industry = params.get('industry')
            result = organizations.get_organization_names(country_code, org_type, industry)

    return result


def is_multidictionary(path: Union[str, Path]) -> bool:
    """
    Checks if a dictionary file contains multiple columns.

    Parameters:
    -----------
    path : Union[str, Path]
        Path to the dictionary file

    Returns:
    --------
    bool
        True if the dictionary has multiple columns, False otherwise
    """
    try:
        # Try to read the first few lines to check format
        with open(path, 'r', encoding='utf-8') as f:
            # Read up to 10 lines
            lines = [f.readline().strip() for _ in range(10) if f.readline()]

        # Check if any line contains a common delimiter
        for line in lines:
            if ',' in line or ';' in line or '\t' in line or '|' in line:
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking dictionary format: {e}")
        return False


def parse_full_name(
        full_name: str,
        language: str = "en",
        name_format: Optional[str] = None
) -> Dict[str, str]:
    """
    Parses a full name into components based on language and format.

    Parameters:
    -----------
    full_name : str
        Full name to parse
    language : str
        Language code for name parsing rules
    name_format : Optional[str]
        Explicit name format (e.g., "first_last", "last_first", "first_middle_last")

    Returns:
    --------
    Dict[str, str]
        Dictionary with name components
    """
    components = full_name.strip().split()
    result = {
        "first_name": "",
        "middle_name": "",
        "last_name": ""
    }

    # Handle empty or single component
    if not components:
        return result
    elif len(components) == 1:
        result["first_name"] = components[0]
        return result

    # Use explicit format if provided
    if name_format:
        if name_format == "first_last" and len(components) >= 2:
            result["first_name"] = components[0]
            result["last_name"] = " ".join(components[1:])
        elif name_format == "last_first" and len(components) >= 2:
            result["last_name"] = components[0]
            result["first_name"] = " ".join(components[1:])
        elif name_format == "first_middle_last" and len(components) >= 3:
            result["first_name"] = components[0]
            result["middle_name"] = components[1]
            result["last_name"] = " ".join(components[2:])
        return result

    # Use language-specific rules
    if language.lower() in ["ru", "ru_ru"]:
        # Russian: Last First Middle
        if len(components) >= 3:
            result["last_name"] = components[0]
            result["first_name"] = components[1]
            result["middle_name"] = " ".join(components[2:])
        elif len(components) == 2:
            result["last_name"] = components[0]
            result["first_name"] = components[1]
    elif language.lower() in ["vn", "vi"]:
        # Vietnamese: Last Middle First (but Middle can be multiple words)
        if len(components) >= 3:
            result["last_name"] = components[0]
            result["first_name"] = components[-1]
            result["middle_name"] = " ".join(components[1:-1])
        elif len(components) == 2:
            result["last_name"] = components[0]
            result["first_name"] = components[1]
    else:
        # Default Western: First [Middle] Last
        if len(components) >= 3:
            result["first_name"] = components[0]
            result["last_name"] = components[-1]
            result["middle_name"] = " ".join(components[1:-1])
        elif len(components) == 2:
            result["first_name"] = components[0]
            result["last_name"] = components[1]

    return result


def get_embedded_dictionary(
        name_type: str = "first_name",
        gender: Optional[str] = None,
        language: str = "ru"
) -> List[str]:
    """
    Returns an embedded dictionary for cases when external dictionaries are not available.

    This is a wrapper around the new embedded dictionaries module.

    Parameters:
    -----------
    name_type : str
        Type of names ("first_name", "last_name", "middle_name", "full_name")
    gender : Optional[str]
        Gender code ("M" or "F")
    language : str
        Language code

    Returns:
    --------
    List[str]
        List of names from the embedded dictionary
    """
    # Normalize parameters
    language = language.lower()
    gender_normalized = gender.upper() if gender else None

    # Use the new embedded dictionaries module
    return names.get_names(language, gender_normalized, name_type)


def load_csv_dictionary(
        path: Union[str, Path],
        column_name: Optional[str] = None,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        cache: bool = True
) -> List[str]:
    """
    Loads a dictionary from a CSV file.

    Parameters:
    -----------
    path : Union[str, Path]
        Path to the CSV file
    column_name : Optional[str]
        Name of the column to extract (if None, uses the first column)
    delimiter : str
        CSV delimiter
    encoding : str
        File encoding
    cache : bool
        Whether to cache the dictionary

    Returns:
    --------
    List[str]
        List of items from the specified column
    """
    path = Path(path)
    cache_key = f"{str(path)}:{column_name}"

    # Check cache first
    if cache and cache_key in _dictionary_cache:
        logger.debug(f"Using cached CSV dictionary: {cache_key}")
        return _dictionary_cache[cache_key].copy()

    logger.debug(f"Loading CSV dictionary from {path}, column: {column_name}")

    try:
        # Try using project's IO utility first
        try:
            df = io.read_full_csv(path, delimiter=delimiter, encoding=encoding)
        except (AttributeError, ImportError):
            # Fallback to pandas directly
            df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)

        # Extract the specified column or first column
        if column_name and column_name in df.columns:
            values = df[column_name].dropna().astype(str).tolist()
        else:
            # Use first column if column_name not specified or not found
            first_col = df.columns[0]
            values = df[first_col].dropna().astype(str).tolist()
            if column_name and column_name != first_col:
                logger.warning(f"Column '{column_name}' not found in {path}, using '{first_col}' instead")

        # Cache the result
        if cache:
            logger.debug(f"Caching CSV dictionary with {len(values)} items: {cache_key}")
            _dictionary_cache[cache_key] = values.copy()

        return values
    except Exception as e:
        logger.error(f"Error loading CSV dictionary {path}: {e}")
        return []


def combine_dictionaries(
        dictionaries: List[List[str]],
        dedup: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
) -> List[str]:
    """
    Combines multiple dictionaries into one.

    Parameters:
    -----------
    dictionaries : List[List[str]]
        List of dictionaries to combine
    dedup : bool
        Whether to remove duplicates
    min_length : Optional[int]
        Minimum length for entries (if specified)
    max_length : Optional[int]
        Maximum length for entries (if specified)

    Returns:
    --------
    List[str]
        Combined dictionary
    """
    combined = []

    # Combine all dictionaries
    for dictionary in dictionaries:
        combined.extend(dictionary)

    # Apply length filtering if specified
    if min_length is not None or max_length is not None:
        filtered = []
        for item in combined:
            if min_length is not None and len(item) < min_length:
                continue
            if max_length is not None and len(item) > max_length:
                continue
            filtered.append(item)
        combined = filtered

    # Remove duplicates if requested
    if dedup:
        return list(dict.fromkeys(combined))  # Preserves order

    return combined