"""
CSV file format specific utilities.

This module provides utilities for working with CSV files, including
size estimation, option preparation, and data validation.
"""

import io
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple

import pandas as pd
import psutil

from pamola_core.utils import logging
from pamola_core.utils import progress

# Configure module logger
logger = logging.get_logger("hhr.utils.io_helpers.csv_utils")


def estimate_csv_size(df: pd.DataFrame,
                      delimiter: str = ",",
                      quotechar: str = '"',
                      encoding: str = "utf-16") -> float:
    """
    Estimates the size of a DataFrame when saved as CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to estimate
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    encoding : str
        File encoding (default: "utf-16")

    Returns:
    --------
    float
        Estimated size in MB
    """
    # Sample a subset of rows for estimation
    sample_size = min(1000, len(df))
    if sample_size < 10:
        return 0.0

    sample = df.sample(sample_size) if len(df) > sample_size else df

    # Write sample to a StringIO object to measure size
    buffer = io.StringIO()
    sample.to_csv(buffer, encoding=encoding, sep=delimiter, quotechar=quotechar)

    # Get size of content and scale to full DataFrame
    buffer_size = len(buffer.getvalue())
    estimated_size = buffer_size * (len(df) / sample_size) / (1024 * 1024)  # Convert to MB

    return estimated_size


def prepare_csv_reader_options(encoding: str = "utf-16",
                               delimiter: str = ",",
                               quotechar: str = '"',
                               **kwargs) -> Dict[str, Any]:
    """
    Prepares options for CSV reading.

    Parameters:
    -----------
    encoding : str
        File encoding (default: "utf-16")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    **kwargs
        Additional pandas read_csv options

    Returns:
    --------
    Dict[str, Any]
        Dictionary with CSV reader options
    """
    options = {
        'encoding': encoding,
        'sep': delimiter,
        'quotechar': quotechar,
        'low_memory': kwargs.get('low_memory', False)
    }

    # Add all other kwargs
    for key, value in kwargs.items():
        if key not in options:
            options[key] = value

    return options


def prepare_csv_writer_options(encoding: str = "utf-16",
                               delimiter: str = ",",
                               quotechar: str = '"',
                               index: bool = False,
                               **kwargs) -> Dict[str, Any]:
    """
    Prepares options for CSV writing.

    Parameters:
    -----------
    encoding : str
        File encoding (default: "utf-16")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    index : bool
        Whether to write row indices (default: False)
    **kwargs
        Additional pandas to_csv options

    Returns:
    --------
    Dict[str, Any]
        Dictionary with CSV writer options
    """
    options = {
        'encoding': encoding,
        'sep': delimiter,
        'quotechar': quotechar,
        'index': index
    }

    # Add all other kwargs
    for key, value in kwargs.items():
        if key not in options:
            options[key] = value

    return options


def count_csv_lines(file_path: Union[str, Path], encoding: str = "utf-16") -> int:
    """
    Counts the number of lines in a CSV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    encoding : str
        File encoding (default: "utf-16")

    Returns:
    --------
    int
        Number of lines in the file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # First try to count lines with the specified encoding
        with open(file_path, 'r', encoding=encoding) as f:
            line_count = sum(1 for _ in f)
        return line_count
    except UnicodeDecodeError:
        # Fall back to binary mode which is faster but might not count lines correctly
        # for some encodings
        logger.warning(
            f"Could not count lines using encoding {encoding}. "
            f"Falling back to binary mode which may be less accurate."
        )
        with open(file_path, 'rb') as f:
            line_count = sum(1 for _ in f)
        return line_count


def validate_csv_structure(df: pd.DataFrame,
                           required_columns: Optional[List[str]] = None,
                           column_types: Optional[Dict[str, type]] = None) -> Tuple[bool, List[str]]:
    """
    Validates the structure of a CSV DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str], optional
        List of required column names
    column_types : Dict[str, type], optional
        Dictionary mapping column names to expected types

    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    # Check column types
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                # Get actual type
                actual_type = df[col].dtype

                # Check if type matches
                type_match = False

                # Special handling for pandas dtypes
                if expected_type == str and pd.api.types.is_string_dtype(actual_type):
                    type_match = True
                elif expected_type == int and pd.api.types.is_integer_dtype(actual_type):
                    type_match = True
                elif expected_type == float and pd.api.types.is_float_dtype(actual_type):
                    type_match = True
                elif expected_type == bool and pd.api.types.is_bool_dtype(actual_type):
                    type_match = True

                if not type_match:
                    errors.append(f"Column '{col}' has wrong type. Expected: {expected_type}, Got: {actual_type}")

    return len(errors) == 0, errors


def monitor_csv_operation(total: Optional[int] = None,
                          description: str = "CSV Operation",
                          unit: str = "rows") -> progress.ProgressBar:
    """
    Creates a progress bar for CSV operations.

    Parameters:
    -----------
    total : int, optional
        Total number of items for the progress bar
    description : str
        Description of the operation
    unit : str
        Unit of measurement

    Returns:
    --------
    progress.ProgressBar
        Progress bar object
    """
    return progress.ProgressBar(
        total=total,
        description=description,
        unit=unit
    )


def report_memory_usage() -> Dict[str, Any]:
    """
    Reports current memory usage.

    Returns:
    --------
    Dict[str, Any]
        Dictionary with memory usage information
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        'rss_bytes': memory_info.rss,  # Resident Set Size
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_bytes': memory_info.vms,  # Virtual Memory Size
        'vms_mb': memory_info.vms / (1024 * 1024),
        'percent': process.memory_percent()
    }