"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: JSON Utilities
Description: Tools for JSON validation, merging, transformation, and preparation for serialization
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Recursive conversion of NumPy types to standard Python types for safe JSON export
- Schema-aware validation of JSON structures
- In-memory merging of nested JSON objects with overwrite and recursion controls
- Preparation of consistent JSON writer options with prettification support
- JSON Schema validation using the jsonschema library (with graceful fallback)

"""

import json
from typing import Dict, Any, Optional, List, Tuple, Type

import numpy as np

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger("pamola_core.utils.io_helpers.json_utils")


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively converts NumPy data types to standard Python types for JSON serialization.

    Parameters:
    -----------
    obj : Any
        Object to convert

    Returns:
    --------
    Any
        Converted object with standard Python types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)


def validate_json_structure(data: Any,
                            expected_type: Optional[type] = None,
                            schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """
    Validates JSON structure.

    Parameters:
    -----------
    data : Any
        Data to validate
    expected_type : type, optional
        Expected type of the data
    schema : Dict[str, Any], optional
        Expected schema (for dict types)

    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check type if specified
    if expected_type and not isinstance(data, expected_type):
        errors.append(f"Expected {expected_type.__name__}, got {type(data).__name__}")
        # If type doesn't match, we can't validate schema
        return False, errors

    # Check schema if specified and data is a dict
    if schema and isinstance(data, dict):
        # Check required keys
        for key, value_info in schema.items():
            # Skip optional fields
            if value_info.get('optional', False) and key not in data:
                continue

            # Check if required key is present
            if key not in data:
                errors.append(f"Missing required key: {key}")
                continue

            # Check value type if specified
            if 'type' in value_info and not isinstance(data[key], value_info['type']):
                errors.append(
                    f"Key '{key}' has wrong type. "
                    f"Expected: {value_info['type'].__name__}, "
                    f"Got: {type(data[key]).__name__}"
                )

            # Check nested schema if specified
            if 'schema' in value_info and isinstance(data[key], dict):
                nested_valid, nested_errors = validate_json_structure(
                    data[key],
                    expected_type=dict,
                    schema=value_info['schema']
                )
                if not nested_valid:
                    # Prefix errors with key name
                    errors.extend([f"In '{key}': {err}" for err in nested_errors])

    return len(errors) == 0, errors


def validate_json_schema(data: Dict[str, Any],
                         schema: Dict[str, Any],
                         error_class: Optional[Type[Exception]] = None) -> None:
    """
    Validate data against a JSON schema with graceful fallback.

    Parameters:
    -----------
    data : Dict[str, Any]
        Data to validate
    schema : Dict[str, Any]
        JSON schema to validate against
    error_class : Type[Exception], optional
        Exception class to raise on validation failure (defaults to ValueError)

    Raises:
    -------
    error_class
        If validation fails

    Notes:
    ------
    If jsonschema is not installed, logs a warning and skips validation.
    This function supports the operation configuration validation requirements.
    """
    # Use provided error class or default to ValueError
    if error_class is None:
        error_class = ValueError

    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
    except ImportError:
        # jsonschema not installed, log warning and continue
        logger.warning("jsonschema package not installed. Schema validation skipped.")
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Schema validation failed: {e}")
        raise error_class(f"Schema validation failed: {e.message}") from e


def merge_json_objects_in_memory(base_obj: Dict[str, Any],
                                 new_obj: Dict[str, Any],
                                 overwrite_existing: bool = True,
                                 recursive_merge: bool = False) -> Dict[str, Any]:
    """
    Merges two JSON objects in memory.

    Parameters:
    -----------
    base_obj : Dict[str, Any]
        Base object to merge into
    new_obj : Dict[str, Any]
        New object to merge from
    overwrite_existing : bool
        Whether to overwrite existing keys (default: True)
    recursive_merge : bool
        Whether to recursively merge nested dictionaries (default: False)

    Returns:
    --------
    Dict[str, Any]
        Merged object
    """
    result = base_obj.copy()

    for key, value in new_obj.items():
        # Skip if key exists and we're not overwriting
        if key in result and not overwrite_existing:
            continue

        # Handle recursive merge for nested dictionaries
        if (recursive_merge and
                key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)):
            result[key] = merge_json_objects_in_memory(
                result[key],
                value,
                overwrite_existing,
                recursive_merge
            )
        else:
            result[key] = value

    return result


def prettify_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Converts a dictionary to a prettified JSON string.

    Parameters:
    -----------
    data : Dict[str, Any]
        Data to convert
    indent : int
        Number of spaces for indentation

    Returns:
    --------
    str
        Prettified JSON string
    """
    # Convert NumPy types
    converted_data = convert_numpy_types(data)

    return json.dumps(
        converted_data,
        indent=indent,
        ensure_ascii=False,
        sort_keys=False
    )


def detect_array_or_object(json_str: str) -> str:
    """
    Detects if a JSON string represents an array or object.

    Parameters:
    -----------
    json_str : str
        JSON string to analyze

    Returns:
    --------
    str
        'array', 'object', or 'invalid'
    """
    try:
        # Strip whitespace
        stripped = json_str.strip()

        # Check first character
        if stripped.startswith('['):
            return 'array'
        elif stripped.startswith('{'):
            return 'object'
        else:
            return 'invalid'
    except Exception:
        return 'invalid'


def extract_json_subset(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Extracts a subset of keys from a JSON object.

    Parameters:
    -----------
    data : Dict[str, Any]
        Original JSON object
    keys : List[str]
        List of keys to extract

    Returns:
    --------
    Dict[str, Any]
        JSON object with only the specified keys
    """
    result = {}

    for key in keys:
        if key in data:
            result[key] = data[key]

    return result


def prepare_json_writer_options(ensure_ascii: bool = False,
                                indent: int = 2,
                                **kwargs) -> Dict[str, Any]:
    """
    Prepares options for JSON writing.

    Parameters:
    -----------
    ensure_ascii : bool
        Whether to escape non-ASCII characters (default: False)
    indent : int
        Number of spaces for indentation (default: 2)
    **kwargs
        Additional json.dump options

    Returns:
    --------
    Dict[str, Any]
        Dictionary with JSON writer options
    """
    options = {
        'ensure_ascii': ensure_ascii,
        'indent': indent
    }

    # Add all other kwargs
    for key, value in kwargs.items():
        if key not in options:
            options[key] = value

    return options