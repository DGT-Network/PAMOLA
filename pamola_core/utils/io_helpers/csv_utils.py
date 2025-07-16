"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: CSV Utilities
Version: 1.1.0+hotfix.2025.06.03
Description: Specialized helpers for efficient and safe CSV processing
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Optimized CSV read/write option preparation
- Memory-efficient chunked processing with progress tracking
- Validation and filtering of CSV columns and schema
- Dialect detection and adaptive chunk size calculation
- CSV quoting control for proper handling of multiline content

Changelog:
1.1.0 (2025-06-03): CSV quoting support
    - Added 'quoting' parameter to prepare_csv_writer_options() as keyword-only argument
    - Added explicit type declarations to resolve type checker warnings
    - Maintains full backward compatibility with existing code
1.0.0 (2025-01-01): Initial release
"""

import csv
import io
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple, Iterator

import pandas as pd

from pamola_core.utils import logging
from pamola_core.utils import progress
from pamola_core.utils.io_helpers import error_utils
from pamola_core.utils.io_helpers import memory_utils

# Configure module logger
logger = logging.get_logger("pamola_core.utils.io_helpers.csv_utils")


def estimate_csv_size(
    df: pd.DataFrame,
    delimiter: str = ",",
    quotechar: str = '"',
    encoding: str = "utf-8",
) -> float:
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
        File encoding (default: "utf-8")

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
    estimated_size = (
        buffer_size * (len(df) / sample_size) / (1024 * 1024)
    )  # Convert to MB

    return estimated_size


def prepare_csv_reader_options(
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"',
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Prepares options for CSV reading with enhanced parameter support.

    Parameters:
    -----------
    encoding : str
        File encoding (default: "utf-8")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    columns : List[str], optional
        Specific columns to include (maps to usecols in pandas)
    nrows : int, optional
        Maximum number of rows to read
    skiprows : Union[int, List[int]], optional
        Rows to skip (int = skip first N rows, list = specific row indices)
    **kwargs
        Additional pandas read_csv options

    Returns:
    --------
    Dict[str, Any]
        Dictionary with CSV reader options
    """
    # Explicitly declare dictionary type
    options: Dict[str, Any] = {
        "encoding": encoding,
        "sep": delimiter,
        "quotechar": quotechar,
        "low_memory": kwargs.get("low_memory", False),
    }

    # Add column filtering if specified
    if columns is not None:
        options["usecols"] = columns

    # Add row selection parameters
    if nrows is not None:
        options["nrows"] = nrows

    if skiprows is not None:
        options["skiprows"] = skiprows

    # Add all other kwargs
    for key, value in kwargs.items():
        if key not in options:
            options[key] = value

    return options


def prepare_csv_writer_options(
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"',
    index: bool = False,
    quoting: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Prepares options for CSV writing.

    Parameters:
    -----------
    encoding : str
        File encoding (default: "utf-8")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    index : bool
        Whether to write row indices (default: False)
    quoting : int, optional
        Control field quoting behavior per csv.QUOTE_* constants:
        - None: Use pandas default (csv.QUOTE_MINIMAL)
        - csv.QUOTE_ALL (1): Quote all fields
        - csv.QUOTE_MINIMAL (0): Quote only when necessary
        - csv.QUOTE_NONNUMERIC (2): Quote all non-numeric fields
        - csv.QUOTE_NONE (3): Never quote (use escapechar)
    **kwargs
        Additional pandas to_csv options

    Returns:
    --------
    Dict[str, Any]
        Dictionary with CSV writer options
    """
    # Explicitly declare dictionary type
    options: Dict[str, Any] = {
        "encoding": encoding,
        "sep": delimiter,
        "quotechar": quotechar,
        "index": index,
    }

    # Add quoting parameter if specified
    if quoting is not None:
        options["quoting"] = quoting

    # Add all other kwargs
    for key, value in kwargs.items():
        if key not in options:
            options[key] = value

    return options


def count_csv_lines(file_path: Union[str, Path], encoding: str = "utf-8") -> int:
    """
    Counts the number of lines in a CSV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    encoding : str
        File encoding (default: "utf-8")

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
        with open(file_path, "r", encoding=encoding) as f:
            line_count = sum(1 for _ in f)
        return line_count
    except UnicodeDecodeError:
        # Fall back to binary mode which is faster but might not count lines correctly
        # for some encodings
        logger.warning(
            f"Could not count lines using encoding {encoding}. "
            f"Falling back to binary mode which may be less accurate."
        )
        with open(file_path, "rb") as f:
            line_count = sum(1 for _ in f)
        return line_count


def validate_csv_structure(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, type]] = None,
) -> Tuple[bool, List[str]]:
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
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.warning(error_msg)
            errors.append(error_msg)

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
                elif expected_type == int and pd.api.types.is_integer_dtype(
                    actual_type
                ):
                    type_match = True
                elif expected_type == float and pd.api.types.is_float_dtype(
                    actual_type
                ):
                    type_match = True
                elif expected_type == bool and pd.api.types.is_bool_dtype(actual_type):
                    type_match = True

                if not type_match:
                    error_msg = f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
                    logger.warning(error_msg)
                    errors.append(error_msg)

    return len(errors) == 0, errors


def monitor_csv_operation(
    total: Optional[int] = None, description: str = "CSV Operation", unit: str = "rows"
) -> progress.ProgressBar:
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
    return progress.ProgressBar(total=total, description=description, unit=unit)


def report_memory_usage() -> Dict[str, Any]:
    """
    Reports current memory usage.

    Returns:
    --------
    Dict[str, Any]
        Dictionary with memory usage information
    """
    return memory_utils.get_process_memory_usage()


def filter_csv_columns(
    df: pd.DataFrame, columns: List[str], strict: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Filters DataFrame columns with detailed error handling.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to filter
    columns : List[str]
        List of columns to include
    strict : bool
        If True, errors if any column is missing

    Returns:
    --------
    Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]
        (filtered_df, error_info) - One will be None
    """
    if not columns:
        return df, None

    # Find which requested columns exist
    existing_columns = set(df.columns)
    valid_columns = [col for col in columns if col in existing_columns]
    missing_columns = [col for col in columns if col not in existing_columns]

    # Check if we have any valid columns
    if not valid_columns:
        error_info = error_utils.create_error_info(
            "NoValidColumnsError",
            f"None of the requested columns exist in the DataFrame",
            "Check column names or DataFrame structure",
            details={
                "requested_columns": columns,
                "available_columns": list(existing_columns),
            },
        )
        return None, error_info

    # Check for missing columns in strict mode
    if strict and missing_columns:
        error_info = error_utils.create_error_info(
            "MissingColumnsError",
            f"Missing columns: {', '.join(missing_columns)}",
            "Check column names or set strict=False to ignore missing columns",
            details={
                "missing_columns": missing_columns,
                "available_columns": list(existing_columns),
            },
        )
        return None, error_info

    # Log warning for missing columns in non-strict mode
    if missing_columns:
        logger.warning(f"Missing columns: {', '.join(missing_columns)}")

    # Filter columns
    return df[valid_columns], None


def detect_csv_dialect(
    file_path: Union[str, Path], max_lines: int = 5, encoding: str = "utf-8"
) -> Dict[str, Any]:
    """
    Detects the dialect of a CSV file (delimiter, quotechar, etc.).

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    max_lines : int
        Number of rows to sample for detection
    encoding : str
        File encoding to try first

    Returns:
    --------
    Dict[str, Any]
        Dictionary with detected dialect information
    """
    import csv

    file_path = Path(file_path)

    if not file_path.exists():
        return error_utils.create_error_info(
            "FileNotFoundError", f"File not found: {file_path}", "Check the file path"
        )

    # Common delimiters to restrict sniffer from guessing random characters
    common_delimiters = [",", ";", "\t", "|"]

    def read_lines_with_encoding(encoding):
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            return "".join([f.readline() for _ in range(max_lines)])

    try:
        # Try to read a sample using the specified encoding
        sample = read_lines_with_encoding(encoding)
        if len(sample.strip()) < 10:
            raise csv.Error("Sample too small to sniff")

        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=common_delimiters)
        has_header = sniffer.has_header(sample)

        return {
            "delimiter": dialect.delimiter,
            "quotechar": dialect.quotechar,
            "has_header": has_header,
            "encoding": encoding,
            "confidence": "high",
        }
    except UnicodeDecodeError:
        # Try common alternative encodings
        alternative_encodings = ["utf-8", "latin-1", "cp1252"]
        for alt_encoding in alternative_encodings:
            try:
                sample = read_lines_with_encoding(alt_encoding)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=common_delimiters)
                has_header = sniffer.has_header(sample)

                return {
                    "delimiter": dialect.delimiter,
                    "quotechar": dialect.quotechar,
                    "has_header": has_header,
                    "encoding": alt_encoding,
                    "confidence": "medium",
                    "note": f"Used alternative encoding: {alt_encoding}",
                }
            except:
                continue
    except Exception as e:
        # Fall back to common defaults
        with open(file_path, "rb") as f:
            lines = []
            for _ in range(max_lines):
                line = f.readline().decode("latin-1", errors="ignore")
                if not line:
                    break
                lines.append(line)
        sample = "".join(lines)
        # Check for common delimiters
        delimiter_counts = {d: sample.count(d) for d in common_delimiters}
        most_common = max(delimiter_counts.items(), key=lambda x: x[1])

        return {
            "delimiter": most_common[0],
            "quotechar": '"',
            "has_header": True,  # Assume header as default
            "encoding": encoding,
            "confidence": "low",
            "error": str(e),
            "note": "Detection failed, using best guess",
        }


def get_optimal_csv_chunk_size(
    file_path: Union[str, Path],
    available_memory_mb: Optional[int] = None,
    safety_factor: float = 0.5,
) -> int:
    """
    Calculate optimal chunk size for reading a CSV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    available_memory_mb : int, optional
        Available memory in MB, auto-detected if None
    safety_factor : float
        Memory safety factor (0.0 to 1.0)

    Returns:
    --------
    int
        Optimal chunk size in rows
    """
    return memory_utils.get_optimal_chunk_size(
        file_path=file_path,
        available_memory_mb=available_memory_mb,
        memory_factor=safety_factor,
    )


def read_csv_in_efficient_chunks(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"',
    columns: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    show_progress: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Read a CSV file in memory-efficient chunks.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    encoding : str
        File encoding (default: "utf-8")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    columns : List[str], optional
        Specific columns to include
    chunk_size : int, optional
        Size of each chunk (auto-calculated if None)
    show_progress : bool
        Whether to show a progress bar

    Yields:
    -------
    pd.DataFrame
        Chunks of the CSV file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Auto-determine chunk size if not specified
    if chunk_size is None:
        chunk_size = get_optimal_csv_chunk_size(file_path)
        logger.info(f"Auto-determined chunk size: {chunk_size} rows")

    # Prepare reader options
    reader_options = prepare_csv_reader_options(
        encoding=encoding,
        delimiter=delimiter,
        quotechar=quotechar,
        columns=columns,
        chunksize=chunk_size,
    )

    # Count total rows for progress tracking
    total_rows = None
    if show_progress:
        try:
            total_rows = count_csv_lines(file_path, encoding) - 1  # Account for header
        except:
            logger.warning("Could not count lines for progress tracking")

    # Create progress bar
    progress_bar = None
    current_row = 0

    if show_progress:
        if total_rows is not None:
            progress_bar = monitor_csv_operation(
                total=total_rows, description=f"Reading {file_path.name}", unit="rows"
            )
        else:
            progress_bar = monitor_csv_operation(
                description=f"Reading {file_path.name}", unit="chunks"
            )

    # Read in chunks
    try:
        for chunk in pd.read_csv(file_path, **reader_options):
            if progress_bar:
                if total_rows is not None:
                    current_row += len(chunk)
                    progress_bar.update(len(chunk))
                else:
                    progress_bar.update(1)

            yield chunk

    finally:
        if progress_bar:
            progress_bar.close()


def optimize_csv_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting data types.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to optimize

    Returns:
    --------
    pd.DataFrame
        Optimized DataFrame
    """
    optimized_df, _ = memory_utils.optimize_dataframe_memory(df)
    return optimized_df


def validate_csv_file(
    file_path: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
    max_validation_rows: int = 1000,
) -> Dict[str, Any]:
    """
    Validate a CSV file against a schema.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    schema : Dict[str, Any], optional
        Schema to validate against (required columns, types, etc.)
    max_validation_rows : int
        Maximum number of rows to validate

    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return error_utils.create_error_info(
            "FileNotFoundError", f"File not found: {file_path}", "Check the file path"
        )

    try:
        # Detect dialect first
        dialect = detect_csv_dialect(file_path)

        # Read sample rows
        df_sample = pd.read_csv(
            file_path,
            encoding=dialect["encoding"],
            sep=dialect["delimiter"],
            quotechar=dialect["quotechar"],
            nrows=max_validation_rows,
        )

        # Basic validation results
        validation_results = {
            "file_path": str(file_path),
            "file_size_bytes": file_path.stat().st_size,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "dialect": dialect,
            "columns": list(df_sample.columns),
            "column_count": len(df_sample.columns),
            "sample_row_count": len(df_sample),
            "null_counts": {
                col: int(df_sample[col].isnull().sum()) for col in df_sample.columns
            },
            "dtypes": {col: str(df_sample[col].dtype) for col in df_sample.columns},
            "is_valid": True,
            "validation_errors": [],
        }

        # Schema validation if provided
        if schema:
            # Check required columns
            if "required_columns" in schema:
                required_cols = schema["required_columns"]
                missing_cols = [
                    col for col in required_cols if col not in df_sample.columns
                ]

                if missing_cols:
                    validation_results["is_valid"] = False
                    validation_results["validation_errors"].append(
                        f"Missing required columns: {', '.join(missing_cols)}"
                    )

            # Check column types if specified
            if "column_types" in schema:
                for col, expected_type in schema["column_types"].items():
                    # Skip columns that don't exist
                    if col not in df_sample.columns:
                        continue

                    # Check if types are compatible
                    actual_type = df_sample[col].dtype
                    is_compatible = (
                        (
                            expected_type == "string"
                            and pd.api.types.is_string_dtype(actual_type)
                        )
                        or (
                            expected_type == "int"
                            and pd.api.types.is_integer_dtype(actual_type)
                        )
                        or (
                            expected_type == "float"
                            and pd.api.types.is_float_dtype(actual_type)
                        )
                        or (
                            expected_type == "bool"
                            and pd.api.types.is_bool_dtype(actual_type)
                        )
                        or (
                            expected_type == "date"
                            and pd.api.types.is_datetime64_dtype(actual_type)
                        )
                    )

                    if not is_compatible:
                        validation_results["is_valid"] = False
                        validation_results["validation_errors"].append(
                            f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
                        )

        return validation_results

    except Exception as e:
        return error_utils.create_error_info(
            "ValidationError",
            f"Error validating CSV file: {str(e)}",
            "Check file format and encoding",
            details={"file_path": str(file_path)},
        )
