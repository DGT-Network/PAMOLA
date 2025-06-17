"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Error Handling Utilities
Description: Centralized error information and structured exception management for I/O operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Standardized error information dictionaries with details and resolution hints
- Decorator-based error wrapping for I/O operations
- Consistent logging of recoverable and fatal errors across subsystems
"""


import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Configure module logger
logger = logging.getLogger("pamola_core.utils.io_helpers.error_utils")

# Type variable for decorated functions
T = TypeVar('T')


def create_error_info(
        error_type: str,
        message: str,
        resolution: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error information dictionary.

    Parameters:
    -----------
    error_type : str
        Type of error (e.g., "FileNotFoundError", "DecryptionError")
    message : str
        Human-readable error message
    resolution : str, optional
        Suggested actions to resolve the error
    file_path : str or Path, optional
        Path to the file associated with the error
    details : Dict[str, Any], optional
        Additional error details or context

    Returns:
    --------
    Dict[str, Any]
        Standardized error information dictionary
    """
    error_info = {
        "error_type": error_type,
        "message": message
    }

    # Add optional fields if provided
    if resolution is not None:
        error_info["resolution"] = resolution

    if file_path is not None:
        error_info["file_path"] = str(file_path)

    if details is not None:
        # Don't overwrite base fields with details
        for key, value in details.items():
            if key not in error_info:
                error_info[key] = value

    return error_info


def handle_io_errors(
        func: Callable[..., T]
) -> Callable[..., Union[T, Dict[str, Any]]]:
    """
    Decorator to standardize error handling for IO operations.

    This decorator catches exceptions, logs them appropriately, and returns
    standardized error information dictionaries.

    Parameters:
    -----------
    func : Callable
        Function to decorate

    Returns:
    --------
    Callable
        Decorated function with standardized error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Union[T, Dict[str, Any]]:
        # Extract file path from args or kwargs for error reporting
        file_path = None
        for i, arg in enumerate(args):
            if isinstance(arg, (str, Path)):
                file_path = arg
                break

        if file_path is None:
            for param in ['file_path', 'path', 'source_path', 'destination_path']:
                if param in kwargs and isinstance(kwargs[param], (str, Path)):
                    file_path = kwargs[param]
                    break

        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            error_info = create_error_info(
                "FileNotFoundError",
                str(e),
                "Verify the file path and permissions",
                file_path
            )
            logger.error(f"File not found: {error_info['message']}")
            return error_info
        except PermissionError as e:
            error_info = create_error_info(
                "PermissionError",
                str(e),
                "Check file access permissions",
                file_path
            )
            logger.error(f"Permission error: {error_info['message']}")
            return error_info
        except IsADirectoryError as e:
            error_info = create_error_info(
                "IsADirectoryError",
                str(e),
                "Specify a file path, not a directory",
                file_path
            )
            logger.error(f"Path is a directory: {error_info['message']}")
            return error_info
        except MemoryError as e:
            error_info = create_error_info(
                "MemoryError",
                "Not enough memory to complete operation",
                "Try processing in smaller chunks or increasing available memory",
                file_path
            )
            logger.error(f"Memory error: {str(e)}")
            return error_info
        except UnicodeDecodeError as e:
            error_info = create_error_info(
                "UnicodeDecodeError",
                f"Encoding error: {str(e)}",
                "Try specifying a different encoding parameter",
                file_path
            )
            logger.error(f"Unicode decode error: {error_info['message']}")
            return error_info
        except Exception as e:
            # Get the specific exception type
            error_type = type(e).__name__

            # Create detailed error information
            error_info = create_error_info(
                error_type,
                str(e),
                "See error details for more information",
                file_path,
                {"traceback": traceback.format_exc()}
            )

            logger.error(f"Unexpected error ({error_type}): {str(e)}")
            return error_info

    return wrapper


def extract_error_message(error_info: Dict[str, Any]) -> str:
    """
    Extract a user-friendly error message from error information.

    Parameters:
    -----------
    error_info : Dict[str, Any]
        Error information dictionary

    Returns:
    --------
    str
        User-friendly error message
    """
    if not isinstance(error_info, dict):
        return "Unknown error"

    message_parts = []

    # Add error type and message
    if "error_type" in error_info and "message" in error_info:
        message_parts.append(f"{error_info['error_type']}: {error_info['message']}")
    elif "message" in error_info:
        message_parts.append(error_info['message'])

    # Add file path if available
    if "file_path" in error_info:
        message_parts.append(f"File: {error_info['file_path']}")

    # Add resolution if available
    if "resolution" in error_info:
        message_parts.append(f"Resolution: {error_info['resolution']}")

    return "\n".join(message_parts)


def is_error_info(result: Any) -> bool:
    """
    Check if a result is an error information dictionary.

    Parameters:
    -----------
    result : Any
        Result to check

    Returns:
    --------
    bool
        True if the result is an error information dictionary, False otherwise
    """
    return (
            isinstance(result, dict) and
            "error_type" in result and
            "message" in result
    )


def is_recoverable_error(error_info: Dict[str, Any]) -> bool:
    """
    Check if an error is potentially recoverable.

    Parameters:
    -----------
    error_info : Dict[str, Any]
        Error information dictionary

    Returns:
    --------
    bool
        True if the error is potentially recoverable, False otherwise
    """
    if not is_error_info(error_info):
        return False

    # Define recoverable error types
    recoverable_types = {
        "FileNotFoundError",  # File might be created later
        "PermissionError",  # Permissions might be fixed
        "UnicodeDecodeError",  # Different encoding might work
        "EncodingError",  # Different encoding might work
        "ConnectionError",  # Network might recover
        "TimeoutError",  # Operation might succeed with longer timeout
        "IOError",  # Generic IO error might be temporary
        "TempFileError",  # Temporary file issues might resolve
    }

    # Check if error type is in the recoverable list
    return error_info["error_type"] in recoverable_types


def combine_error_infos(error_infos: List[Dict[str, Any]],
                        operation_name: str) -> Dict[str, Any]:
    """
    Combine multiple error information dictionaries into a single summary.

    This is useful for operations that process multiple files and encounter
    different errors for each file.

    Parameters:
    -----------
    error_infos : List[Dict[str, Any]]
        List of error information dictionaries
    operation_name : str
        Name of the operation that generated the errors

    Returns:
    --------
    Dict[str, Any]
        Combined error information dictionary with aggregated details
    """
    if not error_infos:
        return create_error_info(
            "NoErrors",
            "No errors to combine",
            "This is likely a programming error"
        )

    # Count error types
    error_counts = {}
    for error in error_infos:
        error_type = error.get("error_type", "UnknownError")
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    # Create a summary message
    total_errors = len(error_infos)
    error_types = len(error_counts)

    summary_message = f"{total_errors} errors occurred during {operation_name}"
    if error_types > 1:
        summary_message += f" ({error_types} different error types)"

    # Create detailed error information
    return create_error_info(
        "MultipleErrors",
        summary_message,
        "Check error_details for individual errors",
        details={
            "error_counts": error_counts,
            "error_details": error_infos
        }
    )


def raise_if_error(result: Any) -> Any:
    """
    Raise an exception if the result is an error information dictionary.

    Parameters:
    -----------
    result : Any
        Result to check

    Returns:
    --------
    Any
        The result if it's not an error information dictionary

    Raises:
    -------
    Exception
        If the result is an error information dictionary
    """
    if not is_error_info(result):
        return result

    # Create an exception message
    message = extract_error_message(result)

    # Get the error type
    error_type = result.get("error_type", "UnknownError")

    # Raise an exception with the appropriate type if possible
    exception_class = globals().get(error_type)
    if exception_class and issubclass(exception_class, Exception):
        raise exception_class(message)
    else:
        # Fallback to generic exception
        raise Exception(f"{error_type}: {message}")