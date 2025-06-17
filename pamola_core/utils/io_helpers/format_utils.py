"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Format Utilities
Description: Tools for detecting, validating, and handling different file formats and conversions
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Enhanced format detection based on file content and signatures
- Validation of format-specific dependencies (Parquet, Excel, etc.)
- Extraction of format metadata and diagnostics
- Integration with encrypted format detection
"""

import json
from pathlib import Path
from typing import Dict, Any, Literal, Optional, Union, List, Tuple

import pandas as pd

from pamola_core.utils import logging
from pamola_core.utils.io_helpers.file_utils import get_file_metadata  # noqa: F401

# Configure module logger
logger = logging.get_logger("pamola_core.utils.io_helpers.format_utils")

# Format identification constants
FORMAT_SIGNATURES = {
    # CSV signatures
    'csv': [b',', b';', b'\t'],

    # Excel signatures
    'xlsx': [b'PK\x03\x04', b'PK\x05\x06'],
    'xls': [b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'],

    # Parquet signatures
    'parquet': [b'PAR1'],

    # JSON signatures
    'json': [b'{', b'['],

    # Zip signatures
    'zip': [b'PK\x03\x04'],

    # PDF signatures
    'pdf': [b'%PDF'],

    # Image signatures
    'png': [b'\x89PNG\r\n\x1a\n'],
    'jpg': [b'\xff\xd8\xff'],
    'gif': [b'GIF87a', b'GIF89a'],

    # Encrypted file signatures (from crypto_utils)
    'age': [b'age-encryption.org/'],
    'encrypted_json': [b'{"algorithm"', b'{"mode"'],
}



# Define pandas-compatible orient type
PandasOrient = Literal["split", "records", "index", "table", "columns", "values"]


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

    if ext in ["csv", "txt", "tsv"]:
        return "csv"
    elif ext in ["json"]:
        return "json"
    elif ext in ["parquet", "pq"]:
        return "parquet"
    elif ext in ["xlsx", "xls"]:
        return "excel"
    elif ext in ["pkl", "pickle"]:
        return "pickle"
    elif ext in ["png", "jpg", "jpeg", "svg", "pdf", "gif", "bmp", "tiff", "tif"]:
        return "image"
    elif ext in ["age"]:
        return "encrypted"
    else:
        logger.warning(f"Unknown file extension: {ext}, defaulting to 'unknown'")
        return "unknown"


def detect_format_from_content(file_path: Union[str, Path], sample_size: int = 8192) -> str:
    """
    Detect file format by examining file content.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file
    sample_size : int
        Number of bytes to examine

    Returns:
    --------
    str
        Detected format based on content
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read a sample of the file
    with open(file_path, 'rb') as f:
        sample = f.read(sample_size)

    # Check for binary vs text
    is_binary = False
    for byte in sample[:1000]:  # Check first 1000 bytes
        if byte == 0:  # Null bytes indicate binary
            is_binary = True
            break

    # Check for format signatures
    for format_name, signatures in FORMAT_SIGNATURES.items():
        for signature in signatures:
            if sample.startswith(signature) or signature in sample[:100]:
                logger.debug(f"Detected {format_name} format based on signature")

                # Convert to standard format name
                if format_name in ['xlsx', 'xls']:
                    return 'excel'
                elif format_name in ['age', 'encrypted_json']:
                    return 'encrypted'
                elif format_name in ['png', 'jpg', 'gif']:
                    return 'image'
                else:
                    return format_name

    # Check for JSON content if no signature match
    try:
        if sample.startswith(b'{') or sample.startswith(b'['):
            json.loads(sample.decode('utf-8', errors='ignore'))
            return 'json'
    except:
        pass

    # Check for CSV content
    if not is_binary:
        # Try to decode as text
        text_sample = sample.decode('utf-8', errors='ignore')

        # Check for CSV indicators - multiple lines with consistent delimiters
        lines = text_sample.splitlines()
        if len(lines) >= 2:
            # Check common delimiters
            for delimiter in [',', ';', '\t', '|']:
                if all(delimiter in line for line in lines[:10]):
                    # Check if all lines have same number of delimiters
                    delimiter_counts = [line.count(delimiter) for line in lines[:10]]
                    if max(delimiter_counts) > 0 and max(delimiter_counts) - min(delimiter_counts) <= 1:
                        logger.debug(f"Detected CSV format with delimiter '{delimiter}'")
                        return 'csv'

    # If no format detected, fall back to extension
    ext_format = detect_format_from_extension(file_path)
    if ext_format != "unknown":
        return ext_format

    # Last resort - guess based on binary/text
    if is_binary:
        return "binary"
    else:
        return "text"


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
    if format.lower() in ["csv", "json", "pickle", "text", "txt"]:
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

    # Check for encrypted format support
    if format.lower() == "encrypted":
        try:
            from pamola_core.utils.io_helpers import crypto_utils
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
        "pdf": "pdf",
        "text": "txt",
        "txt": "txt",
        "encrypted": "enc",
        "age": "age",
        "binary": "bin",
        "unknown": "dat"
    }

    format_lower = format.lower()
    if format_lower in format_extensions:
        return format_extensions[format_lower]
    else:
        logger.warning(f"Unknown format: {format}, using 'dat' extension")
        return "dat"


def check_dependencies(format: str) -> Tuple[bool, str]:
    """
    Check if all dependencies for a format are available.

    Parameters:
    -----------
    format : str
        Format to check dependencies for

    Returns:
    --------
    Tuple[bool, str]
        (dependencies_available, error_message)
    """
    if format.lower() == "parquet":
        try:
            import pyarrow
            return True, ""
        except ImportError:
            return False, "pyarrow is required for Parquet operations. Please install it with 'pip install pyarrow'."

    elif format.lower() == "excel":
        try:
            import openpyxl
            return True, ""
        except ImportError:
            return False, "openpyxl is required for Excel operations. Please install it with 'pip install openpyxl'."

    elif format.lower() == "image":
        try:
            import matplotlib
            return True, ""
        except ImportError:
            return False, "matplotlib is required for image operations. Please install it with 'pip install matplotlib'."

    elif format.lower() == "encrypted":
        try:
            from pamola_core.utils.io_helpers import crypto_utils
            return True, ""
        except ImportError:
            return False, "crypto_utils is required for encryption operations."

    # Default for always-supported formats
    return True, ""


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


def convert_dataframe_to_json(df: pd.DataFrame,
                             orient: PandasOrient = "records") -> Dict[str, Any]:
    """
    Convert a DataFrame to a JSON-serializable object.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert
    orient : str
        Orientation for the conversion (default: "records")
        Allowed values: "split", "records", "index", "columns", "values", "table"

    Returns:
    --------
    Dict[str, Any]
        JSON-serializable object
    """
    # Convert to JSON string and then parse back to ensure proper conversion
    # This handles proper type conversion and ensures JSON compatibility
    try:
        json_str = df.to_json(orient=orient)
        return json.loads(json_str)
    except ValueError as e:
        logger.warning(f"Invalid orient value: {orient}. Using 'records' instead. Error: {e}")
        json_str = df.to_json(orient="records")
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


def is_encrypted_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is encrypted.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    bool
        True if the file appears to be encrypted
    """
    try:
        from pamola_core.utils.io_helpers import crypto_utils
        return crypto_utils.is_encrypted(file_path)
    except ImportError:
        # If crypto_utils is not available, try basic detection
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        # Check extension
        if file_path.suffix.lower() in ['.enc', '.age']:
            return True

        # Check content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(100)  # Read first 100 bytes

                # Check for known encrypted format signatures
                for signatures in FORMAT_SIGNATURES['age']:
                    if signatures in header:
                        return True

                # Check for encrypted JSON format
                for signatures in FORMAT_SIGNATURES['encrypted_json']:
                    if signatures in header:
                        return True

            return False
        except:
            return False


def detect_encoding(file_path: Union[str, Path],
                    default_encoding: str = "utf-8",
                    fallback_encodings: Optional[List[str]] = None) -> str:
    """
    Detect the encoding of a text file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file
    default_encoding : str
        Default encoding to try first
    fallback_encodings : List[str], optional
        List of encodings to try if default fails
        (default: ["utf-8", "latin-1", "cp1252"])

    Returns:
    --------
    str
        Detected encoding
    """
    # Initialize fallback encodings with default value if None
    if fallback_encodings is None:
        fallback_encodings = ["utf-8", "latin-1", "cp1252"]

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Try the default encoding first
    try:
        with open(file_path, 'r', encoding=default_encoding) as f:
            f.read(1024)  # Try to read a chunk
        return default_encoding
    except UnicodeDecodeError:
        pass

    # Try fallback encodings
    for encoding in fallback_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Try to read a chunk
            return encoding
        except UnicodeDecodeError:
            continue

    # If all fail, return the most permissive encoding
    return "latin-1"


def validate_file_format(file_path: Union[str, Path], expected_format: str = None) -> Dict[str, Any]:
    """
    Validate that a file has the expected format.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file
    expected_format : str, optional
        Expected format (if None, inferred from extension)

    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    # Get full metadata
    metadata = get_file_metadata(file_path)

    # If metadata is an error, return it
    if "error_type" in metadata:
        return metadata

    # Determine expected format
    if expected_format is None:
        expected_format = metadata["format_from_extension"]

    # Validate format
    validation_results = {
        "file_path": str(file_path),
        "expected_format": expected_format,
        "detected_format": metadata["detected_format"],
        "is_valid": expected_format.lower() == metadata["detected_format"].lower(),
        "metadata": metadata
    }

    # Add validation message
    if validation_results["is_valid"]:
        validation_results["message"] = f"File is a valid {expected_format} file"
    else:
        validation_results["message"] = (
            f"File format mismatch: expected {expected_format}, "
            f"detected {metadata['detected_format']}"
        )

    return validation_results