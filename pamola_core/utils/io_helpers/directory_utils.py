"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Directory and Path Utilities
Description: Tools for secure directory, file, and temporary resource management
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Secure creation and cleanup of temporary files and directories
- Path normalization and security checks to prevent traversal vulnerabilities
- Automatic cleanup of registered resources using atexit hooks
- Utilities for unique filename generation, permission management, and secure deletion

"""


import atexit
import os
import shutil
import stat
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Callable

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger("pamola_core.utils.io_helpers.directory_utils")

# Dictionary to track temporary resources for cleanup
_temp_resources = {
    "directories": set(),
    "files": set(),
}


# Register cleanup handler
def _cleanup_temp_resources():
    """Clean up all registered temporary resources on exit."""
    # Clean up temp files first
    for file_path in _temp_resources["files"]:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

    # Clean up temp directories next
    for dir_path in _temp_resources["directories"]:
        try:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {dir_path}: {e}")


# Register atexit handler for cleanup
atexit.register(_cleanup_temp_resources)


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


def create_secure_temp_directory(prefix: str = "pamola_",
                                 parent_dir: Optional[Union[str, Path]] = None,
                                 register_for_cleanup: bool = True) -> Path:
    """
    Creates a secure temporary directory.

    Parameters:
    -----------
    prefix : str
        Prefix for the directory name
    parent_dir : str or Path, optional
        Parent directory (uses system temp dir if None)
    register_for_cleanup : bool
        Whether to register for automatic cleanup

    Returns:
    --------
    Path
        Path to the created temporary directory
    """
    # Use system temp directory if parent_dir not specified
    if parent_dir is None:
        parent_dir = tempfile.gettempdir()
    else:
        parent_dir = Path(parent_dir)
        ensure_directory(parent_dir)

    # Create a unique directory name with UUID
    dir_name = f"{prefix}{uuid.uuid4()}"
    temp_dir = Path(parent_dir) / dir_name

    # Create the directory with secure permissions (mode 0o700)
    temp_dir.mkdir(mode=0o700, exist_ok=False)

    # Register for cleanup if requested
    if register_for_cleanup:
        _temp_resources["directories"].add(str(temp_dir))

    logger.debug(f"Created secure temporary directory: {temp_dir}")
    return temp_dir


def create_secure_temp_file(prefix: str = "pamola_",
                            suffix: str = ".tmp",
                            directory: Optional[Union[str, Path]] = None,
                            register_for_cleanup: bool = True,
                            text: bool = False) -> Path:
    """
    Creates a secure temporary file.

    Parameters:
    -----------
    prefix : str
        Prefix for the file name
    suffix : str
        Suffix for the file name
    directory : str or Path, optional
        Directory for the file (uses system temp dir if None)
    register_for_cleanup : bool
        Whether to register for automatic cleanup
    text : bool
        Whether the file should be opened in text mode

    Returns:
    --------
    Path
        Path to the created temporary file
    """
    if directory is not None:
        directory = ensure_directory(directory)

    # Create a temporary file
    fd, temp_file_path = tempfile.mkstemp(
        prefix=prefix, suffix=suffix, dir=directory, text=text
    )

    # Close the file descriptor
    os.close(fd)

    # Set secure permissions (mode 0o600)
    os.chmod(temp_file_path, 0o600)

    # Register for cleanup if requested
    if register_for_cleanup:
        _temp_resources["files"].add(temp_file_path)

    logger.debug(f"Created secure temporary file: {temp_file_path}")
    return Path(temp_file_path)


def safe_remove_temp_file(file_path: Union[str, Path], logger_obj=None) -> bool:
    """
    Safely removes a temporary file with error handling.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file to remove
    logger_obj : Logger, optional
        Logger object to use (uses default if None)

    Returns:
    --------
    bool
        True if removal was successful, False otherwise
    """
    if logger_obj is None:
        logger_obj = logger

    if file_path is None:
        return True

    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger_obj.debug(f"Removed temporary file: {path}")

            # Remove from tracking if registered
            if str(path) in _temp_resources["files"]:
                _temp_resources["files"].remove(str(path))

            return True
        return True  # File doesn't exist, consider success
    except Exception as e:
        logger_obj.warning(f"Error removing temporary file {file_path}: {str(e)}")
        return False


def normalize_path(path: Union[str, Path],
                   make_absolute: bool = False,
                   resolve_symlinks: bool = False) -> Path:
    """
    Normalizes a path for consistent handling.

    Parameters:
    -----------
    path : str or Path
        Path to normalize
    make_absolute : bool
        Whether to convert to an absolute path
    resolve_symlinks : bool
        Whether to resolve symlinks

    Returns:
    --------
    Path
        Normalized path
    """
    normalized_path = Path(path)

    if make_absolute:
        normalized_path = normalized_path.absolute()

    if resolve_symlinks:
        normalized_path = normalized_path.resolve()

    return normalized_path


def is_path_in_directory(path: Union[str, Path],
                         directory: Union[str, Path],
                         include_subdirs: bool = True) -> bool:
    """
    Checks if a path is inside a directory.

    Parameters:
    -----------
    path : str or Path
        Path to check
    directory : str or Path
        Directory to check against
    include_subdirs : bool
        Whether to include subdirectories in the check

    Returns:
    --------
    bool
        True if the path is in the directory, False otherwise
    """
    # Normalize paths
    path = normalize_path(path, make_absolute=True)
    directory = normalize_path(directory, make_absolute=True)

    # Check if path is the directory itself
    if path == directory:
        return True

    # Check if path is in the directory or its subdirectories
    if include_subdirs:
        try:
            # Use relative_to to check if path is a descendant
            path.relative_to(directory)
            return True
        except ValueError:
            return False
    else:
        # Only check if path is a direct child of directory
        return path.parent == directory


def create_secure_path(base_dir: Union[str, Path],
                       subpath: str,
                       create: bool = False) -> Path:
    """
    Creates a secure path by ensuring it doesn't escape the base directory.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory for the path
    subpath : str
        Subpath (can include subdirectories)
    create : bool
        Whether to create directories if they don't exist

    Returns:
    --------
    Path
        Secure path

    Raises:
    -------
    ValueError
        If the subpath would escape the base directory
    """
    # Normalize base directory
    base_dir = normalize_path(base_dir, make_absolute=True)

    # Create base directory if needed
    if create:
        ensure_directory(base_dir)

    # Remove leading slashes and directory traversal
    safe_subpath = Path(subpath.lstrip('/').lstrip('\\'))

    # Replace parent directory references (..) with underscores
    parts = []
    for part in safe_subpath.parts:
        if part == '..':
            parts.append('__')
        else:
            parts.append(part)

    # Recreate the subpath
    safe_subpath = Path(*parts)

    # Combine with base directory
    secure_path = base_dir / safe_subpath

    # Verify the path is still within the base directory
    if not is_path_in_directory(secure_path, base_dir):
        raise ValueError(f"Subpath '{subpath}' would escape the base directory")

    # Create directories if requested
    if create and secure_path.suffix == '':  # It's a directory
        ensure_directory(secure_path)
    elif create:  # It's a file, create parent directories
        ensure_directory(secure_path.parent)

    return secure_path


def with_secure_temp_directory(func: Callable) -> Callable:
    """
    Decorator that provides a secure temporary directory to a function.

    Parameters:
    -----------
    func : Callable
        Function to decorate

    Returns:
    --------
    Callable
        Decorated function
    """

    def wrapper(*args, **kwargs):
        # Create a secure temporary directory
        temp_dir = create_secure_temp_directory()

        try:
            # Add temp_dir to the kwargs
            kwargs['temp_dir'] = temp_dir
            return func(*args, **kwargs)
        finally:
            # Clean up the temporary directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)

                # Remove from tracking
                if str(temp_dir) in _temp_resources["directories"]:
                    _temp_resources["directories"].remove(str(temp_dir))
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory {temp_dir}: {e}")

    return wrapper


def is_path_writable(path: Union[str, Path]) -> bool:
    """
    Checks if a path is writable.

    Parameters:
    -----------
    path : str or Path
        Path to check

    Returns:
    --------
    bool
        True if the path is writable, False otherwise
    """
    path = Path(path)

    if path.exists():
        # Check if the file/directory is writable
        return os.access(path, os.W_OK)
    else:
        # Check if the parent directory is writable
        return os.access(path.parent, os.W_OK)


def protect_path(path: Union[str, Path], readonly: bool = True) -> None:
    """
    Sets or removes write protection for a file.

    Parameters:
    -----------
    path : str or Path
        Path to protect
    readonly : bool
        Whether to make the file read-only (True) or writable (False)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if readonly:
        # Make read-only
        if os.name == 'nt':  # Windows
            os.chmod(path, stat.S_IREAD)
        else:  # Unix/Linux/Mac
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

        logger.debug(f"Set read-only protection for {path}")
    else:
        # Make writable
        if os.name == 'nt':  # Windows
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        else:  # Unix/Linux/Mac
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | stat.S_IWUSR)

        logger.debug(f"Removed read-only protection for {path}")


def secure_cleanup(paths: List[Union[str, Path]],
                   secure_delete: bool = False,
                   ignore_errors: bool = False) -> List[str]:
    """
    Securely cleans up files and directories.

    Parameters:
    -----------
    paths : List[Union[str, Path]]
        Paths to clean up
    secure_delete : bool
        Whether to use secure deletion (overwrite before deletion)
    ignore_errors : bool
        Whether to ignore errors during cleanup

    Returns:
    --------
    List[str]
        List of paths that couldn't be cleaned up (empty if all successful)
    """
    failed_paths = []

    for path_item in paths:
        path = Path(path_item)

        try:
            if not path.exists():
                continue

            if path.is_file():
                if secure_delete:
                    # Securely delete by overwriting with zeros
                    try:
                        file_size = path.stat().st_size
                        with open(path, 'wb') as f:
                            f.write(b'\x00' * min(file_size, 1024 * 1024))  # Overwrite with zeros (max 1MB)
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception as e:
                        logger.warning(f"Error during secure overwrite of {path}: {e}")

                # Remove the file
                path.unlink()
                logger.debug(f"Removed file: {path}")

                # Remove from tracking if registered
                if str(path) in _temp_resources["files"]:
                    _temp_resources["files"].remove(str(path))

            elif path.is_dir():
                if secure_delete:
                    # For directories, secure delete each file
                    for file_path in path.glob('**/*'):
                        if file_path.is_file():
                            try:
                                secure_cleanup([file_path], secure_delete=True, ignore_errors=True)
                            except Exception as e:
                                if not ignore_errors:
                                    logger.warning(f"Error during secure deletion of {file_path}: {e}")

                # Remove the directory
                shutil.rmtree(path, ignore_errors=ignore_errors)
                logger.debug(f"Removed directory: {path}")

                # Remove from tracking if registered
                if str(path) in _temp_resources["directories"]:
                    _temp_resources["directories"].remove(str(path))

        except Exception as e:
            error_msg = f"Error cleaning up {path}: {e}"
            if ignore_errors:
                logger.warning(error_msg)
                failed_paths.append(str(path))
            else:
                logger.error(error_msg)
                failed_paths.append(str(path))
                raise

    return failed_paths


def get_temp_file_for_decryption(original_file: Union[str, Path],
                                 suffix: str = ".dec",
                                 create: bool = True) -> Path:
    """
    Creates a temporary file path for decryption operations.

    Parameters:
    -----------
    original_file : str or Path
        Path to the original encrypted file
    suffix : str
        Suffix for the temporary file
    create : bool
        Whether to create the file (empty)

    Returns:
    --------
    Path
        Path to the temporary file
    """
    original_path = Path(original_file)

    # Create a temporary file with a secure name derived from the original
    file_id = uuid.uuid4().hex[:8]
    temp_filename = f"{original_path.stem}_dec_{file_id}{suffix}"

    # Use same directory as original file if possible
    try:
        if original_path.parent.exists() and os.access(original_path.parent, os.W_OK):
            temp_path = original_path.parent / temp_filename
        else:
            # Fall back to system temp directory
            temp_path = Path(tempfile.gettempdir()) / temp_filename
    except:
        # Fall back to system temp directory
        temp_path = Path(tempfile.gettempdir()) / temp_filename

    # Create empty file if requested
    if create:
        # Create with secure permissions
        with open(temp_path, 'wb') as f:
            pass
        os.chmod(temp_path, 0o600)  # Read/write for owner only

    # Register for cleanup
    _temp_resources["files"].add(str(temp_path))

    logger.debug(f"Created temporary decryption file: {temp_path}")
    return temp_path


def get_temp_file_for_encryption(original_file: Union[str, Path],
                                 suffix: str = ".enc",
                                 create: bool = True) -> Path:
    """
    Creates a temporary file path for encryption operations.

    Parameters:
    -----------
    original_file : str or Path
        Path to the original unencrypted file
    suffix : str
        Suffix for the temporary file
    create : bool
        Whether to create the file (empty)

    Returns:
    --------
    Path
        Path to the temporary file
    """
    # Use the same logic as decryption, but with encryption suffix
    return get_temp_file_for_decryption(original_file, suffix=suffix, create=create)


def ensure_parent_directory(file_path: Union[str, Path]) -> bool:
    """
    Ensures the parent directory of a file exists.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    bool
        True if parent directory exists or was created, False otherwise
    """
    file_path = Path(file_path)

    try:
        ensure_directory(file_path.parent)
        return True
    except Exception as e:
        logger.error(f"Error creating parent directory for {file_path}: {e}")
        return False


def get_unique_filename(directory: Union[str, Path],
                        base_name: str,
                        extension: str,
                        include_timestamp: bool = True) -> Path:
    """
    Generates a unique filename in a directory.

    Parameters:
    -----------
    directory : str or Path
        Directory for the file
    base_name : str
        Base name for the file
    extension : str
        File extension
    include_timestamp : bool
        Whether to include a timestamp

    Returns:
    --------
    Path
        Unique file path
    """
    directory = ensure_directory(directory)

    # Clean extension
    extension = extension.lstrip('.')

    # Create timestamped filename
    filename = get_timestamped_filename(base_name, extension, include_timestamp)

    # Create full path
    file_path = directory / filename

    # If file exists, make it unique
    if file_path.exists():
        file_path = make_unique_path(file_path)

    return file_path


def create_secure_directory_structure(base_dir: Union[str, Path],
                                      subdirs: List[str],
                                      permissions: int = 0o700) -> Dict[str, Path]:
    """
    Creates a secure directory structure with controlled permissions.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory
    subdirs : List[str]
        List of subdirectory names to create
    permissions : int
        Unix-style permissions (default: 0o700 = rwx for owner only)

    Returns:
    --------
    Dict[str, Path]
        Dictionary mapping subdirectory names to paths
    """
    # Ensure base directory exists with secure permissions
    base_dir = Path(base_dir)
    ensure_directory(base_dir)

    try:
        os.chmod(base_dir, permissions)
    except Exception as e:
        logger.warning(f"Could not set permissions on {base_dir}: {e}")

    # Create subdirectories
    result = {'base': base_dir}

    for subdir in subdirs:
        # Sanitize subdir name to prevent directory traversal
        safe_subdir = subdir.replace('..', '__').lstrip('/')

        # Create full path
        subdir_path = base_dir / safe_subdir
        ensure_directory(subdir_path)

        # Set permissions
        try:
            os.chmod(subdir_path, permissions)
        except Exception as e:
            logger.warning(f"Could not set permissions on {subdir_path}: {e}")

        # Add to result
        result[safe_subdir] = subdir_path

    return result