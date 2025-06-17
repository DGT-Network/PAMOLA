"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: File Utilities
Description: Helper functions for file operations, metadata extraction, and path handling
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides utility functions for working with files, including:
- Getting file metadata (size, checksum, modification time)
- Calculating checksums with different algorithms
- Safe file removal and path validation
- File existence and format validation

These utilities are used by the data models and operations in PAMOLA Core
to ensure consistent handling of file operations and metadata extraction.
"""

import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive metadata about a file.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file

    Returns:
    --------
    Dict[str, Any]
        Dictionary with file metadata including:
        - size_bytes: File size in bytes
        - created_at: Creation timestamp
        - modified_at: Last modification timestamp
        - accessed_at: Last access timestamp
        - exists: Whether the file exists
        - is_file: Whether it's a file (not a directory)
        - extension: File extension
        - path: Absolute path to the file
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    metadata = {
        "exists": file_path.exists(),
        "path": str(file_path.absolute()),
        "extension": file_path.suffix.lower(),
        "filename": file_path.name,
        "directory": str(file_path.parent)
    }

    if metadata["exists"]:
        stats = file_path.stat()
        metadata.update({
            "size_bytes": stats.st_size,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "accessed_at": datetime.fromtimestamp(stats.st_atime).isoformat(),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir()
        })

    return metadata


def calculate_checksum(file_path: Union[str, Path], algorithm: str = 'sha256') -> Optional[str]:
    """
    Calculate a checksum for a file.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file
    algorithm : str
        Hash algorithm to use ('sha256', 'md5', 'sha1')

    Returns:
    --------
    Optional[str]
        Checksum as a hexadecimal string, or None if file doesn't exist
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        logger.warning(f"File not found when calculating checksum: {file_path}")
        return None

    try:
        if algorithm not in ('sha256', 'md5', 'sha1'):
            logger.warning(f"Unsupported hash algorithm: {algorithm}, using sha256 instead")
            algorithm = 'sha256'

        hash_func = getattr(hashlib, algorithm)()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating checksum for {file_path}: {str(e)}")
        return None


def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    Get the size of a file in bytes.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file

    Returns:
    --------
    Optional[int]
        File size in bytes, or None if file doesn't exist
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        logger.warning(f"File not found when getting size: {file_path}")
        return None

    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return None


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file

    Returns:
    --------
    bool
        True if the file exists, False otherwise
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    return file_path.exists() and file_path.is_file()


def safe_remove_file(file_path: Union[str, Path], secure: bool = False) -> bool:
    """
    Safely remove a file with optional secure deletion.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file to remove
    secure : bool
        Whether to use secure deletion (overwrite with zeros before deletion)

    Returns:
    --------
    bool
        True if the file was successfully removed, False otherwise
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        logger.warning(f"File not found when trying to remove: {file_path}")
        return False

    try:
        if secure and file_path.is_file():
            # Secure deletion: overwrite with zeros
            file_size = file_path.stat().st_size
            with open(file_path, 'wb') as f:
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())

        # Remove the file
        file_path.unlink()
        return True
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {str(e)}")
        return False


def validate_file_type(file_path: Union[str, Path], expected_extension: str) -> bool:
    """
    Validate that a file has the expected extension.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file
    expected_extension : str
        Expected file extension (without dot)

    Returns:
    --------
    bool
        True if the file has the expected extension, False otherwise
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        logger.warning(f"File not found when validating type: {file_path}")
        return False

    # Normalize extensions
    actual_ext = file_path.suffix.lower().lstrip('.')
    expected_ext = expected_extension.lower().lstrip('.')

    return actual_ext == expected_ext


def get_file_age(file_path: Union[str, Path]) -> Optional[float]:
    """
    Get the age of a file in seconds since last modification.

    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the file

    Returns:
    --------
    Optional[float]
        Age of the file in seconds, or None if the file doesn't exist
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        logger.warning(f"File not found when getting age: {file_path}")
        return None

    try:
        mtime = file_path.stat().st_mtime
        return datetime.now().timestamp() - mtime
    except Exception as e:
        logger.error(f"Error getting file age for {file_path}: {str(e)}")
        return None