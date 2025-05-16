"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Anonymization Validation Utilities
Description: Parameter validation utilities for anonymization operations
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides validation functions for parameters used in anonymization
operations, ensuring consistency and proper error handling.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_field_exists(df: pd.DataFrame, field_name: str, logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a field exists in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    field_name : str
        The name of the field to verify
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the field exists, False otherwise

    Logs:
    -----
    ERROR if the field does not exist
    """
    log = logger_instance or logger

    if field_name not in df.columns:
        error_message = f"Field '{field_name}' does not exist in the DataFrame"
        log.error(error_message)
        return False
    return True


def validate_numeric_field(df: pd.DataFrame, field_name: str, allow_null: bool = True,
                           logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a field is numeric.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to validate
    allow_null : bool, optional
        Whether to allow null values (default: True)
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the field is numeric and meets null criteria, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    # First check if field exists
    if not validate_field_exists(df, field_name, log):
        return False

    # Check if field is numeric
    if not pd.api.types.is_numeric_dtype(df[field_name]):
        error_message = f"Field '{field_name}' is not a numeric type"
        log.error(error_message)
        return False

    # Check for null values if not allowed
    if not allow_null and df[field_name].isnull().any():
        error_message = f"Field '{field_name}' contains null values"
        log.error(error_message)
        return False

    return True


def validate_categorical_field(df: pd.DataFrame, field_name: str, allow_null: bool = True,
                               logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a field is categorical or string type.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to validate
    allow_null : bool, optional
        Whether to allow null values (default: True)
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the field is categorical/string and meets null criteria, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    # First check if field exists
    if not validate_field_exists(df, field_name, log):
        return False

    # Check if field is categorical or string
    column_dtype = df[field_name].dtype
    is_categorical = isinstance(column_dtype, pd.CategoricalDtype)
    is_string = pd.api.types.is_string_dtype(df[field_name])
    is_object = pd.api.types.is_object_dtype(df[field_name])

    if not (is_categorical or is_string or is_object):
        error_message = f"Field '{field_name}' is not a categorical or string type"
        log.error(error_message)
        return False

    # Check for null values if not allowed
    if not allow_null and df[field_name].isnull().any():
        error_message = f"Field '{field_name}' contains null values"
        log.error(error_message)
        return False

    return True


def validate_datetime_field(df: pd.DataFrame, field_name: str, allow_null: bool = True,
                            logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a field is a datetime type.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to validate
    allow_null : bool, optional
        Whether to allow null values (default: True)
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the field is datetime and meets null criteria, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    # First check if field exists
    if not validate_field_exists(df, field_name, log):
        return False

    # Check if field is datetime
    is_datetime = pd.api.types.is_datetime64_dtype(df[field_name])

    if not is_datetime:
        # Try to convert to datetime to see if it's convertible
        try:
            pd.to_datetime(df[field_name])
            log.info(f"Field '{field_name}' can be converted to datetime")
            return True
        except:
            error_message = f"Field '{field_name}' is not a datetime type and cannot be converted"
            log.error(error_message)
            return False

    # Check for null values if not allowed
    if not allow_null and df[field_name].isnull().any():
        error_message = f"Field '{field_name}' contains null values"
        log.error(error_message)
        return False

    return True


def validate_generalization_strategy(strategy: str, valid_strategies: List[str],
                                     logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a generalization strategy is supported.

    Parameters:
    -----------
    strategy : str
        The strategy to validate
    valid_strategies : List[str]
        List of valid strategies
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the strategy is valid, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    if strategy not in valid_strategies:
        error_message = f"Strategy '{strategy}' is not supported. Valid strategies: {', '.join(valid_strategies)}"
        log.error(error_message)
        return False
    return True


def validate_bin_count(bin_count: int, logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a bin count is valid.

    Parameters:
    -----------
    bin_count : int
        The number of bins to validate
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the bin count is valid, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    if not isinstance(bin_count, int) or bin_count <= 0:
        error_message = f"Bin count must be a positive integer, got {bin_count}"
        log.error(error_message)
        return False
    return True


def validate_precision(precision: int, logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a precision value is valid.

    Parameters:
    -----------
    precision : int
        The precision value to validate
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the precision is valid, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    if not isinstance(precision, int):
        error_message = f"Precision must be an integer, got {type(precision).__name__}"
        log.error(error_message)
        return False
    return True


def validate_range_limits(range_limits: Tuple[float, float],
                          logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that range limits are valid.

    Parameters:
    -----------
    range_limits : Tuple[float, float]
        The (min, max) limits to validate
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the range limits are valid, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    if not isinstance(range_limits, tuple) or len(range_limits) != 2:
        error_message = f"Range limits must be a tuple of two values (min, max), got {range_limits}"
        log.error(error_message)
        return False

    min_val, max_val = range_limits

    try:
        min_val = float(min_val)
        max_val = float(max_val)
    except (ValueError, TypeError):
        error_message = f"Range limits must be numeric, got {min_val} and {max_val}"
        log.error(error_message)
        return False

    if min_val >= max_val:
        error_message = f"Minimum range limit ({min_val}) must be less than maximum ({max_val})"
        log.error(error_message)
        return False

    return True


def validate_output_field_name(df: pd.DataFrame, output_field_name: str, mode: str,
                               logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate output field name based on the mode.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    output_field_name : str
        Output field name to validate
    mode : str
        Mode of operation ("REPLACE" or "ENRICH")
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the output field name is valid, False otherwise

    Logs:
    -----
    ERROR if validation fails
    WARNING if output field already exists
    """
    log = logger_instance or logger

    if mode == "REPLACE":
        # For REPLACE mode, output_field_name is not required
        return True

    elif mode == "ENRICH":
        if not output_field_name:
            error_message = "Output field name must be specified when mode is 'ENRICH'"
            log.error(error_message)
            return False

        # Check if the output field name already exists
        if output_field_name in df.columns:
            log.warning(f"Output field '{output_field_name}' already exists and will be overwritten")

    else:
        error_message = f"Invalid mode: {mode}. Must be 'REPLACE' or 'ENRICH'"
        log.error(error_message)
        return False

    return True


def validate_null_strategy(strategy: str, valid_strategies: Optional[List[str]] = None,
                           logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a null handling strategy is supported.

    Parameters:
    -----------
    strategy : str
        The strategy to validate
    valid_strategies : List[str], optional
        List of valid strategies (default: None, will use ["PRESERVE", "EXCLUDE", "ERROR"])
    logger_instance : Optional[logging.Logger]
        Logger instance to use for logging (default: module logger)

    Returns:
    --------
    bool
        True if the strategy is valid, False otherwise

    Logs:
    -----
    ERROR if validation fails
    """
    log = logger_instance or logger

    # Use None as the default, and then set the list of strategies
    if valid_strategies is None:
        valid_strategies = ["PRESERVE", "EXCLUDE", "ERROR"]

    if strategy not in valid_strategies:
        error_message = f"Null strategy '{strategy}' is not supported. Valid strategies: {', '.join(valid_strategies)}"
        log.error(error_message)
        return False
    return True


def get_validation_error_result(error_message: str, field_name: str = None) -> Dict[str, Any]:
    """
    Create a standardized validation error result.

    Parameters:
    -----------
    error_message : str
        The error message
    field_name : str, optional
        The field name associated with the error

    Returns:
    --------
    Dict[str, Any]
        Validation error result with standardized structure
    """
    result = {
        "valid": False,
        "error": error_message,
        "error_type": "ValidationError"
    }

    if field_name:
        result["field"] = field_name

    return result