"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        File Utilities
Package:       pamola_core.utils.io_helpers.file_utils
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
Helper functions for file operations, metadata extraction, path handling,
and conflict resolution. Provides utilities for safe file operations,
checksum calculation, and handling locked/unwritable files.

Key Features:
- Comprehensive file metadata extraction
- Multiple checksum algorithms (SHA256, MD5, SHA1)
- Safe file removal with optional secure deletion
- File type validation and age calculation
- Automatic path conflict resolution for locked files
- Timestamp-based alternative naming for unwritable paths
- Cross-platform file operation handling
- Improved thread safety and error handling

Framework:
Part of PAMOLA.CORE infrastructure, providing low-level file operations
used throughout the system for data processing and storage management.

Changelog:
1.2.0 - Improved thread safety and race condition handling
      - Enhanced secure deletion for large files
      - Better temporary file cleanup
      - Added input validation for timestamp formats
      - Improved error handling specificity
1.1.0 - Added resolve_writable_path for handling locked files
      - Enhanced error handling and logging
      - Added support for custom timestamp formats
      - Improved cross-platform compatibility
1.0.0 - Initial implementation with basic file operations
      - Added metadata extraction and checksum calculation
      - Implemented safe file removal with secure deletion option

Dependencies:
- Standard library only for maximum compatibility
- pathlib for modern path handling
- hashlib for checksum calculation
- tempfile for safe temporary file creation

TODO:
- Add support for file locking mechanisms
- Implement atomic file operations
- Add file compression utilities
- Support for cloud storage paths (S3, GCS, etc.)
"""


import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union, Optional

from pamola_core.common.type_aliases import PathLike
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)

# Constants
CHUNK_SIZE = 8 * (2**10)  # 8KB chunks for file operations
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.1  # seconds


def get_file_metadata(
        file_path: PathLike
) -> Dict[str, Any]:
    """
    Get comprehensive metadata about a file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file

    Returns
    -------
    Dict[str, Any]
        Dictionary with file metadata including:
        - exists: Whether the file exists
        - extension: File extension
        - path: Absolute path to the file
        - size_bytes: File size in bytes
        - created_at: Creation timestamp
        - modified_at: Last modification timestamp
        - accessed_at: Last access timestamp
        - is_file: Whether it's a file (not a directory)
    """
    file_path = Path(file_path)

    metadata = {
        "exists": file_path.exists(),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "path": str(file_path.absolute()),
        "directory": str(file_path.parent)
    }

    if metadata["exists"]:
        try:
            stats = file_path.stat()
            metadata.update(
                {
                    "size_bytes": stats.st_size,
                    "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "accessed_at": datetime.fromtimestamp(stats.st_atime).isoformat(),
                    "is_file": file_path.is_file(),
                    "is_dir": file_path.is_dir()
                }
            )
        except OSError as e:
            logger.error(f"Error getting file stats for {file_path}: {e}")
            metadata.update(
                {
                    "size_bytes": None,
                    "created_at": None,
                    "modified_at": None,
                    "accessed_at": None,
                    "is_file": None,
                    "is_dir": None,
                    "error": str(e)
                }
            )

    return metadata


def get_file_size(
        file_path: PathLike
) -> Optional[int]:
    """
    Get the size of a file in bytes.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file

    Returns
    -------
    Optional[int]
        File size in bytes, or None if file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found when getting size: {file_path}")
        return None

    try:
        return file_path.stat().st_size
    except (IOError, OSError) as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return None


def get_file_age(
        file_path: PathLike
) -> Optional[float]:
    """
    Get the age of a file in seconds since last modification.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file

    Returns
    -------
    Optional[float]
        Age of the file in seconds, or None if the file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found when getting age: {file_path}")
        return None

    try:
        mtime = file_path.stat().st_mtime
        current_time = time.time() # Use time.time() for consistency
        return current_time - mtime
    except (IOError, OSError) as e:
        logger.error(f"Error getting file age for {file_path}: {e}")
        return None


def file_exists(
        file_path: PathLike
) -> bool:
    """
    Check if a file exists.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file

    Returns
    -------
    bool
        True if the file exists, False otherwise
    """
    file_path = Path(file_path)
    return file_path.exists() and file_path.is_file()


def is_file_locked(
        file_path: PathLike
) -> bool:
    """
    Check if a file is locked (not writable).

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to check

    Returns
    -------
    bool
        True if file is locked/not writable, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found when check file is locked: {file_path}")
        return False

    try:
        # Try to open file in append mode
        with open(file_path, "a"):
            pass
        return False
    except (PermissionError, OSError):
        return True


def validate_file_type(
        file_path: PathLike,
        expected_extension: str
) -> bool:
    """
    Validate that a file has the expected extension.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file
    expected_extension : str
        Expected file extension (without dot)

    Returns
    -------
    bool
        True if the file has the expected extension, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found when validating type: {file_path}")
        return False

    # Normalize extensions
    actual_ext = file_path.suffix.lower().lstrip(".")
    expected_ext = expected_extension.lower().lstrip(".")

    return actual_ext == expected_ext


def calculate_checksum(
        file_path: PathLike,
        algorithm: str = "sha256"
) -> Optional[str]:
    """
    Calculate a checksum for a file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file
    algorithm : str
        Hash algorithm to use ("md5", "sha1", "sha256")

    Returns
    -------
    Optional[str]
        Checksum as a hexadecimal string, or None if file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found when calculating checksum: {file_path}")
        return None

    try:
        algorithm = algorithm.lower()

        # Validate algorithm
        if algorithm not in ("md5", "sha1", "sha256"):
            logger.warning(
                f"Unsupported hash algorithm: {algorithm}, using sha256 instead"
            )
            algorithm = "sha256"

        # Create hash object
        hash_func = getattr(hashlib, algorithm)()

        # Read and hash file in chunks for memory efficiency
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()
    except (IOError, OSError) as e:
        logger.error(f"Error calculating checksum for {file_path}: {e}")
        return None


def safe_remove_file(
        file_path: PathLike,
        secure: bool = False
) -> Optional[bool]:
    """
    Safely remove a file with optional secure deletion.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file to remove
    secure : bool
        Whether to use secure deletion (overwrite with zeros before deletion)

    Returns
    -------
    bool
        True if the file was successfully removed, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found when trying to remove: {file_path}")
        return False

    try:
        if secure and file_path.is_file():
            # Secure deletion: overwrite with zeros in chunks
            file_size = file_path.stat().st_size
            with open(file_path, "r+b") as f:
                # Overwrite in chunks to avoid memory issues
                remaining = file_size
                while remaining > 0:
                    chunk_size = min(CHUNK_SIZE, remaining)
                    f.write(b"\x00" * chunk_size)
                    remaining -= chunk_size
                f.flush()
                os.fsync(f.fileno())

        # Remove the file with retry logic
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                file_path.unlink()
                return True
            except (OSError, PermissionError) as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise e

    except (IOError, OSError, PermissionError) as e:
        logger.error(f"Error removing file {file_path}: {e}")
        return False


def resolve_writable_path(
        file_path: PathLike,
        add_timestamp: bool = True,
        timestamp_format: str = "%Y%m%d_%H%M%S",
        max_attempts: int = 10
) -> Path:
    """
    Return a writable path. If the original file is locked/unwritable,
    generate an alternative filename.

    Parameters
    ----------
    file_path : Union[str, Path]
        Original file path
    add_timestamp : bool
        Whether to add timestamp to alternative names
    timestamp_format : str
        Format string for timestamp (must be valid strftime format)
    max_attempts : int
        Maximum number of alternative paths to try

    Returns
    -------
    Path
        Writable path (original or alternative)

    Raises
    ------
    ValueError
        If max_attempts < 1 or timestamp_format is invalid
    OSError
        If no writable path can be found
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    # Validate timestamp format
    if add_timestamp:
        try:
            datetime.now().strftime(timestamp_format)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timestamp format: {e}")

    original_path = Path(file_path)

    # Ensure parent directory exists
    try:
        original_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create parent directory: {e}")

    # Check if original path is writable
    if _is_path_writable(file_path=original_path):
        return original_path

    # Generate alternative path
    alternative = _find_alternative_path(
        original_path=original_path,
        add_timestamp=add_timestamp,
        timestamp_format=timestamp_format,
        max_attempts=max_attempts
    )
    if alternative is None:
        raise OSError(f"No writable path found after {max_attempts} attempts")

    return alternative


def _is_path_writable(
        file_path: Path
) -> bool:
    """
    Check if a path is writable by testing write operations.

    Returns
    -------
    bool
        True if writable, False otherwise (never None)
    """
    # Initialize result flag
    can_write = False

    # Check parent directory first
    if not file_path.parent.exists():
        logger.debug(f"Parent directory does not exist: {file_path.parent}")
        return False

    if not os.access(file_path.parent, os.W_OK):
        logger.debug(f"No write access to parent directory: {file_path.parent}")
        return False

    # If file exists, try to open it
    if file_path.exists():
        try:
            with open(file_path, "a"):
                pass
            can_write = True
        except Exception as e:
            logger.debug(f"Cannot write to existing file {str(file_path)}: {e}") # noqa
            can_write = False
    else:
        # File doesn't exist, try to create a test file
        test_file = (
            file_path.parent / f".test_{os.getpid()}_{datetime.now().timestamp()}.tmp"
        )
        try:
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
            can_write = True
        except Exception as e:
            logger.debug(f"Cannot create test file in {file_path.parent}: {e}")
            can_write = False
            # Try to clean up if file was created
            try:
                if test_file.exists():
                    test_file.unlink()
            except:
                pass

    # Single return point
    return can_write


def _find_alternative_path(
        original_path: Path,
        add_timestamp: bool,
        timestamp_format: str,
        max_attempts: int
) -> Optional[Path]:
    """
    Find an alternative path when original is not writable.

    Returns
    -------
    Optional[Path]
        Alternative writable path, or None if not found
    """
    base = original_path.stem
    ext = original_path.suffix
    parent = original_path.parent

    # Generate timestamp once if needed
    timestamp = ""
    if add_timestamp:
        timestamp = datetime.now().strftime(timestamp_format)

    # Try timestamp-only first
    if timestamp:
        candidate = parent / f"{base}_{timestamp}{ext}"
        if _is_path_writable(file_path=candidate):
            logger.info(f"Using alternative path with timestamp: {candidate}")
            return candidate

    # Try with counters
    for i in range(1, max_attempts + 1):
        if timestamp:
            candidate = parent / f"{base}_{timestamp}_{i}{ext}"
        else:
            candidate = parent / f"{base}_{i}{ext}"

        if _is_path_writable(file_path=candidate):
            logger.info(f"Using alternative path: {candidate}")
            return candidate

    # All attempts exhausted
    logger.error(f"Could not find writable path after {max_attempts} attempts")
    return None
