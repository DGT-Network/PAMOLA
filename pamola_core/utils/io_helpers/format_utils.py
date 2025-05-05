"""
Utilities for handling different file formats and conversions.

This module provides helper functions for working with various file formats
like parquet, excel, and others. It isolates format-specific code and
optional dependencies from the main I/O module.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import pandas as pd

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger("hhr.utils.io_helpers.format_utils")


def detect_format_from_extension(file_path: Union[str, Path]) -> str:
    """
    Detect file format from file extension.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    str
        Detected format: "csv", "json", "parquet", "excel", "pickle"
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower().lstrip('.')

    if ext in ["csv", "txt"]:
        return "csv"
    elif ext in ["json"]:
        return "json"
    elif ext in ["parquet", "pq"]:
        return "parquet"
    elif ext in ["xlsx", "xls"]:
        return "excel"
    elif ext in ["pkl", "pickle"]:
        return "pickle"
    elif ext in ["png", "jpg", "jpeg", "svg", "pdf"]:
        return "image"
    else:
        raise ValueError(f"Unknown file extension: {ext}")


def is_format_supported(format: str, for_reading: bool = True) -> bool:
    """
    Check if a format is supported with the current dependencies.

    Parameters:
    -----------
    format : str
        Format to check: "csv", "json", "parquet", "excel", "pickle"
    for_reading : bool
        Whether to check for reading (True) or writing (False)

    Returns:
    --------
    bool
        True if format is supported
    """
    # CSV, JSON, and pickle are always supported
    if format.lower() in ["csv", "json", "pickle"]:
        return True

    # Check for parquet support
    if format.lower() == "parquet":
        try:
            import pyarrow
            return True
        except ImportError:
            return False

    # Check for Excel support
    if format.lower() == "excel":
        try:
            import openpyxl
            return True
        except ImportError:
            return False

    # Check for image formats
    if format.lower() == "image":
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    return False


def get_format_extension(format: str) -> str:
    """
    Get the standard file extension for a format.

    Parameters:
    -----------
    format : str
        Format name

    Returns:
    --------
    str
        Standard file extension (without dot)
    """
    format_extensions = {
        "csv": "csv",
        "json": "json",
        "parquet": "parquet",
        "excel": "xlsx",
        "pickle": "pkl",
        "png": "png",
        "jpg": "jpg",
        "jpeg": "jpg",
        "svg": "svg",
        "pdf": "pdf"
    }

    format_lower = format.lower()
    if format_lower in format_extensions:
        return format_extensions[format_lower]
    else:
        raise ValueError(f"Unknown format: {format}")


def check_pyarrow_available():
    """
    Check if pyarrow is available and raise an informative error if not.

    Raises:
    -------
    ImportError
        If pyarrow is not available
    """
    try:
        import pyarrow
    except ImportError:
        logger.error("pyarrow is required for Parquet operations")
        raise ImportError("pyarrow is required for Parquet operations. Please install it with 'pip install pyarrow'.")


def check_openpyxl_available():
    """
    Check if openpyxl is available and raise an informative error if not.

    Raises:
    -------
    ImportError
        If openpyxl is not available
    """
    try:
        import openpyxl
    except ImportError:
        logger.error("openpyxl is required for Excel operations")
        raise ImportError("openpyxl is required for Excel operations. Please install it with 'pip install openpyxl'.")


def check_matplotlib_available():
    """
    Check if matplotlib is available and raise an informative error if not.

    Raises:
    -------
    ImportError
        If matplotlib is not available
    """
    try:
        import matplotlib
    except ImportError:
        logger.error("matplotlib is required for image operations")
        raise ImportError(
            "matplotlib is required for image operations. Please install it with 'pip install matplotlib'.")


def convert_dataframe_to_json(df: pd.DataFrame, orient: str = "records") -> Dict[str, Any]:
    """
    Convert a DataFrame to a JSON-serializable object.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert
    orient : str
        Orientation for the conversion (default: "records")

    Returns:
    --------
    Dict[str, Any]
        JSON-serializable object
    """
    # Convert to JSON string and then parse back to ensure proper conversion
    json_str = df.to_json(orient=orient)
    return json.loads(json_str)


def get_pandas_dtypes_info(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get detailed information about column data types.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping column names to detailed data type information
    """
    dtypes_info = {}

    for col in df.columns:
        # Get basic dtype
        dtype_str = str(df[col].dtype)

        # Add more specific info for object columns
        if df[col].dtype == 'object':
            # Sample non-null values
            sample = df[col].dropna()
            if len(sample) > 0:
                sample_type = type(sample.iloc[0]).__name__
                dtype_str = f"object ({sample_type})"

        dtypes_info[col] = dtype_str

    return dtypes_info


def get_dataframe_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic statistics about a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns:
    --------
    Dict[str, Any]
        Dictionary with statistics about the DataFrame
    """
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage_bytes": df.memory_usage(deep=True).sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "dtypes": get_pandas_dtypes_info(df),
        "null_counts": df.isna().sum().to_dict()
    }

    return stats