"""
Type checking helpers for pandas objects.

This module provides a stable API for checking pandas data types,
regardless of the pandas version being used. It includes fallback
implementations for functions that might not be available in some
pandas versions.
"""

import pandas as pd
import numpy as np


def is_numeric_dtype(series_or_dtype):
    """
    Check if series or dtype is numeric.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if numeric, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for numeric dtype
    try:
        return (
            isinstance(dtype, np.dtype) and
            np.issubdtype(dtype, np.number)
        )
    except (TypeError, AttributeError):
        # Fallback for string representation
        dtype_str = str(dtype).lower()
        return any(num_type in dtype_str for num_type in
                  ['int', 'float', 'double', 'number', 'numeric'])


def is_bool_dtype(series_or_dtype):
    """
    Check if series or dtype is boolean.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if boolean, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for bool dtype
    try:
        return (
            isinstance(dtype, np.dtype) and
            (dtype == np.bool_ or dtype == bool)
        )
    except (TypeError, AttributeError):
        # Fallback for string representation
        dtype_str = str(dtype).lower()
        return 'bool' in dtype_str


def is_object_dtype(series_or_dtype):
    """
    Check if series or dtype is object.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if object, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for object dtype
    try:
        return dtype == np.dtype('O')
    except (TypeError, AttributeError):
        # Fallback for string representation
        dtype_str = str(dtype).lower()
        return 'object' in dtype_str


def is_string_dtype(series_or_dtype):
    """
    Check if series or dtype is string.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if string, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype

        # In older versions, strings are stored as object dtype
        if dtype == np.dtype('O'):
            # If we have a series, check if all non-NA values are strings
            if hasattr(series_or_dtype, 'dropna'):
                sample = series_or_dtype.dropna()[:100]  # Check up to 100 values
                return all(isinstance(x, str) for x in sample) if len(sample) > 0 else False
    else:
        dtype = series_or_dtype

    # Check for string dtype
    dtype_str = str(dtype).lower()
    return 'string' in dtype_str or dtype_str == 'str'


def is_datetime64_dtype(series_or_dtype):
    """
    Check if series or dtype is datetime64.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if datetime64, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for datetime64 dtype
    try:
        return (
            isinstance(dtype, np.dtype) and
            np.issubdtype(dtype, np.datetime64)
        )
    except (TypeError, AttributeError):
        # Fallback for string representation
        dtype_str = str(dtype).lower()
        return 'datetime' in dtype_str or 'timestamp' in dtype_str


def is_categorical_dtype(series_or_dtype):
    """
    Check if series or dtype is categorical.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if categorical, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if it's a Series with 'cat' accessor
    if hasattr(series_or_dtype, 'cat'):
        return True

    # Check dtype name
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for categorical dtype by name
    dtype_str = str(dtype).lower()
    return dtype_str == 'category' or 'categorical' in dtype_str


def is_integer_dtype(series_or_dtype):
    """
    Check if series or dtype is integer.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if integer, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for integer dtype
    try:
        return (
            isinstance(dtype, np.dtype) and
            np.issubdtype(dtype, np.integer)
        )
    except (TypeError, AttributeError):
        # Fallback for string representation
        dtype_str = str(dtype).lower()
        return any(int_type in dtype_str for int_type in ['int', 'int8', 'int16', 'int32', 'int64', 'uint'])


def is_float_dtype(series_or_dtype):
    """
    Check if series or dtype is float.

    Parameters:
    -----------
    series_or_dtype : pd.Series, np.dtype, or type
        Series or dtype to check

    Returns:
    --------
    bool
        True if float, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    # Check if has dtype attribute
    if hasattr(series_or_dtype, 'dtype'):
        dtype = series_or_dtype.dtype
    else:
        dtype = series_or_dtype

    # Check for float dtype
    try:
        return (
            isinstance(dtype, np.dtype) and
            np.issubdtype(dtype, np.floating)
        )
    except (TypeError, AttributeError):
        # Fallback for string representation
        dtype_str = str(dtype).lower()
        return any(float_type in dtype_str for float_type in ['float', 'double'])


def is_list_like(obj):
    """
    Check if object is list-like (an iterable but not a string/bytes).

    Parameters:
    -----------
    obj : any
        Object to check

    Returns:
    --------
    bool
        True if list-like, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    return (
        hasattr(obj, '__iter__') and
        not isinstance(obj, (str, bytes))
    )


def is_dict_like(obj):
    """
    Check if object is dict-like (has keys and __getitem__).

    Parameters:
    -----------
    obj : any
        Object to check

    Returns:
    --------
    bool
        True if dict-like, False otherwise
    """
    # Direct implementation without relying on pd.api.types
    return (
        hasattr(obj, 'keys') and
        hasattr(obj, '__getitem__')
    )