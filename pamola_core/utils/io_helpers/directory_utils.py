"""
Directory and path management utilities.

This module provides functions for working with directories, paths,
and file timestamps. It also includes utilities for directory cleanup,
file statistics, and directory content listing.
"""

import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger("hhr.utils.io_helpers.directory_utils")


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensures the specified directory exists, creating it if necessary.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory to ensure exists

    Returns:
    --------
    Path
        Pathlib object for the ensured directory
    """
    directory = Path(directory)

    if not directory.exists():
        logger.info(f"Creating directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_timestamped_filename(base_name: str, extension: str = "csv",
                             include_timestamp: bool = True) -> str:
    """
    Creates a timestamped filename.

    Parameters:
    -----------
    base_name : str
        Base name for the file
    extension : str
        File extension (default: "csv")
    include_timestamp : bool
        Whether to include a timestamp in the filename (default: True)

    Returns:
    --------
    str
        Timestamped filename
    """
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"

    return f"{base_name}.{extension}"


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Gets statistics about a file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    Dict[str, Any]
        Dictionary with file statistics (size, creation time, modification time, etc.)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat_result = file_path.stat()

    return {
        "name": file_path.name,
        "path": str(file_path),
        "size_bytes": stat_result.st_size,
        "size_mb": stat_result.st_size / (1024 * 1024),
        "creation_time": datetime.fromtimestamp(stat_result.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modification_time": datetime.fromtimestamp(stat_result.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "extension": file_path.suffix,
        "is_directory": file_path.is_dir(),
        "is_file": file_path.is_file(),
        "is_symlink": file_path.is_symlink(),
        "permissions": stat_result.st_mode & 0o777,  # Extract permission bits
        "owner_readable": bool(stat_result.st_mode & stat.S_IRUSR),
        "owner_writable": bool(stat_result.st_mode & stat.S_IWUSR),
        "owner_executable": bool(stat_result.st_mode & stat.S_IXUSR),
    }


def list_directory_contents(directory: Union[str, Path],
                            pattern: str = "*",
                            recursive: bool = False) -> List[Path]:
    """
    Lists the contents of a directory.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory
    pattern : str
        Glob pattern for filtering files (default: "*")
    recursive : bool
        Whether to search recursively (default: False)

    Returns:
    --------
    List[Path]
        List of paths to the files matching the pattern
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    if recursive:
        return list(directory.glob(f"**/{pattern}"))
    else:
        return list(directory.glob(pattern))


def clear_directory(directory: Union[str, Path],
                    ignore_patterns: Optional[List[str]] = None,
                    confirm: bool = True) -> int:
    """
    Clears all files and subdirectories in the specified directory.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory to clear
    ignore_patterns : List[str], optional
        List of glob patterns to ignore
    confirm : bool
        Whether to ask for confirmation before clearing (default: True)

    Returns:
    --------
    int
        Number of items removed

    Raises:
    -------
    FileNotFoundError
        If directory doesn't exist
    ValueError
        If path is not a directory
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Prepare ignore patterns
    if ignore_patterns is None:
        ignore_patterns = []

    # Get all items in directory
    all_items = list(directory.glob("*"))

    # Filter out ignored items
    items_to_remove = []
    for item in all_items:
        should_ignore = False
        for pattern in ignore_patterns:
            if item.match(pattern):
                should_ignore = True
                break

        if not should_ignore:
            items_to_remove.append(item)

    if not items_to_remove:
        logger.info(f"No items to remove from {directory}")
        return 0

    # Ask for confirmation if needed
    if confirm:
        logger.warning(f"About to remove {len(items_to_remove)} items from {directory}")

    # Perform removal
    removed_count = 0
    for item in items_to_remove:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed_count += 1
            logger.debug(f"Removed: {item}")
        except Exception as e:
            logger.error(f"Error removing {item}: {e}")

    logger.info(f"Successfully removed {removed_count} items from {directory}")
    return removed_count


def make_unique_path(base_path: Union[str, Path],
                     create: bool = False) -> Path:
    """
    Creates a unique path by appending a counter if the path already exists.

    Parameters:
    -----------
    base_path : str or Path
        Base path to make unique
    create : bool
        Whether to create the directory if it doesn't exist (default: False)

    Returns:
    --------
    Path
        Unique path
    """
    base_path = Path(base_path)

    # If it doesn't exist, we can use it as is
    if not base_path.exists():
        if create and str(base_path).endswith(os.sep):  # It's intended to be a directory
            ensure_directory(base_path)
        return base_path

    # Handle file paths
    if '.' in base_path.name:
        name_parts = base_path.stem.split('.')
        extension = base_path.suffix
        counter = 1

        while True:
            new_path = base_path.parent / f"{name_parts[0]}_{counter}{extension}"
            if not new_path.exists():
                if create and str(new_path).endswith(os.sep):
                    ensure_directory(new_path)
                return new_path
            counter += 1

    # Handle directory paths
    else:
        counter = 1
        while True:
            new_path = Path(f"{base_path}_{counter}")
            if not new_path.exists():
                if create and str(new_path).endswith(os.sep):
                    ensure_directory(new_path)
                return new_path
            counter += 1