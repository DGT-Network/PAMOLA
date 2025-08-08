"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Path Security
Description: Security validation for file paths
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides utilities for validating path security,
preventing directory traversal attacks, access to system directories,
and other path-based security vulnerabilities.
"""

import logging
import platform
from pathlib import Path
from typing import Union, List, Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """Exception raised for path security violations."""
    pass


def validate_path_security(
        path: Union[str, Path],
        allowed_paths: Optional[List[Union[str, Path]]] = None,
        allow_external: bool = False,
        strict_mode: bool = True
) -> bool:
    """
    Validate that a path is safe to use.

    This function checks that a path doesn't contain potentially
    dangerous components like path traversal sequences.

    Args:
        path: Path to validate
        allowed_paths: List of allowed external paths (absolute)
        allow_external: Whether to allow external paths outside data repository
        strict_mode: If True, raises PathSecurityError for unsafe paths

    Returns:
        True if the path is safe, False otherwise

    Raises:
        PathSecurityError: If the path is unsafe and strict_mode is True
    """
    path_obj = Path(path) if isinstance(path, str) else path
    path_str = str(path_obj).replace("\\", "/")

    # Check for path traversal patterns
    dangerous_patterns = [
        "..",  # Parent directory traversal
        "~",  # Home directory
        "|",  # Command chaining (Windows)
        ";",  # Command chaining (Unix)
        "&",  # Command chaining
        "$",  # Variable substitution
        "`",  # Command substitution
        "\\x",  # Hex escape
        "\\u"  # Unicode escape
    ]

    # Check for path traversal patterns
    for pattern in dangerous_patterns:
        if pattern in path_str:
            error_msg = f"Potentially unsafe path detected: {path_str} (contains '{pattern}')"
            logger.warning(error_msg)
            if strict_mode:
                raise PathSecurityError(error_msg)
            return False

    # Check absolute path safety
    if path_obj.is_absolute():
        # Get system-specific dangerous paths
        system_paths = get_system_specific_dangerous_paths()

        # Check against system paths
        for sys_path in system_paths:
            if path_str.startswith(sys_path):
                error_msg = f"Potentially unsafe path detected: {path_str} (system directory)"
                logger.warning(error_msg)
                if strict_mode:
                    raise PathSecurityError(error_msg)
                return False

        # If external paths are not allowed, check if in allowed_paths
        if not allow_external and allowed_paths:
            if not is_within_allowed_paths(path_obj, allowed_paths):
                error_msg = f"External path not allowed: {path_str}"
                logger.warning(error_msg)
                if strict_mode:
                    raise PathSecurityError(error_msg)
                return False

    # Check if path contains symbolic links that might lead outside allowed paths
    if path_obj.exists():
        try:
            # Get the real path with symbolic links resolved
            real_path = path_obj.resolve()

            # If the real path is different and not permitted, raise an error
            if real_path != path_obj and not allow_external and allowed_paths:
                if not is_within_allowed_paths(real_path, allowed_paths):
                    error_msg = f"Path contains symbolic link to external location: {real_path}"
                    logger.warning(error_msg)
                    if strict_mode:
                        raise PathSecurityError(error_msg)
                    return False
        except (OSError, RuntimeError) as e:
            # Handle errors in resolving paths (e.g., recursive symlinks)
            error_msg = f"Error resolving path {path_str}: {str(e)}"
            logger.warning(error_msg)
            if strict_mode:
                raise PathSecurityError(error_msg)
            return False

    return True


def is_within_allowed_paths(
        path: Path,
        allowed_paths: List[Union[str, Path]]
) -> bool:
    """
    Check if a path is within any of the allowed paths.

    Args:
        path: Path to check
        allowed_paths: List of allowed parent paths

    Returns:
        True if path is within any allowed path, False otherwise
    """
    # Normalize the path
    try:
        normalized_path = str(path.resolve())
    except (OSError, RuntimeError):
        # If resolution fails (e.g., due to broken symlinks), use the original path
        normalized_path = str(path)

    # Check each allowed path
    for allowed_path in allowed_paths:
        try:
            # Convert to Path if string
            allowed_path_obj = Path(allowed_path) if isinstance(allowed_path, str) else allowed_path

            # Normalize the allowed path
            normalized_allowed = str(allowed_path_obj.resolve())

            # Check if the path starts with the allowed path
            if normalized_path.startswith(normalized_allowed):
                return True
        except (OSError, RuntimeError):
            # If resolution fails, try string comparison as fallback
            allowed_str = str(allowed_path)
            if normalized_path.startswith(allowed_str):
                return True

    return False


def get_system_specific_dangerous_paths() -> List[str]:
    """
    Get a list of system-specific paths that should be protected.

    Returns:
        List of dangerous system paths
    """
    system = platform.system().lower()

    if system == "windows":
        return [
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            "C:\\Users\\Default",
            "C:\\ProgramData",
            "C:\\System Volume Information",
            "C:\\$Recycle.Bin",
            # Add more Windows system paths as needed
        ]
    elif system in ("linux", "darwin"):  # Linux or macOS
        return [
            "/bin",
            "/sbin",
            "/etc",
            "/dev",
            "/sys",
            "/proc",
            "/boot",
            "/lib",
            "/lib64",
            "/usr/bin",
            "/usr/sbin",
            "/usr/lib",
            "/var/run",
            "/var/lock",
            # Add more Unix-like system paths as needed
        ]
    else:
        # For other systems, provide a basic set
        logger.warning(f"Unknown system: {system}, using default protected paths")
        return [
            "/bin", "/sbin", "/etc", "/dev", "/sys", "/proc",  # Unix-like
            "C:\\Windows", "C:\\Program Files"  # Windows
        ]


def validate_paths(
        paths: List[Union[str, Path]],
        allowed_paths: Optional[List[Union[str, Path]]] = None,
        allow_external: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate multiple paths at once.

    Args:
        paths: List of paths to validate
        allowed_paths: List of allowed external paths
        allow_external: Whether to allow external paths

    Returns:
        Tuple containing:
            - Boolean indicating if all paths are valid
            - List of error messages for invalid paths
    """
    errors = []

    for path in paths:
        try:
            if not validate_path_security(
                    path,
                    allowed_paths=allowed_paths,
                    allow_external=allow_external,
                    strict_mode=False
            ):
                errors.append(f"Invalid path: {path}")
        except Exception as e:
            errors.append(f"Error validating path {path}: {str(e)}")

    return len(errors) == 0, errors


def is_potentially_dangerous_path(path: Union[str, Path]) -> bool:
    """
    Check if a path might be potentially dangerous without raising exceptions.

    This is a convenience method for quick checks without stopping execution.

    Args:
        path: Path to check

    Returns:
        True if path might be dangerous, False if likely safe
    """
    try:
        return not validate_path_security(path, strict_mode=False)
    except Exception:
        # If any exception occurs during validation, consider it potentially dangerous
        return True


def normalize_and_validate_path(
        path: Union[str, Path],
        base_dir: Optional[Path] = None,
        allowed_paths: Optional[List[Union[str, Path]]] = None,
        allow_external: bool = False
) -> Path:
    """
    Normalize a path and validate its security.

    If the path is relative, it will be resolved against the base_dir.

    Args:
        path: Path to normalize and validate
        base_dir: Base directory for resolving relative paths
        allowed_paths: List of allowed external paths
        allow_external: Whether to allow external paths

    Returns:
        Normalized path object

    Raises:
        PathSecurityError: If the path fails security validation
    """
    path_obj = Path(path) if isinstance(path, str) else path

    # If path is relative and base_dir is provided, resolve against base_dir
    if not path_obj.is_absolute() and base_dir is not None:
        path_obj = base_dir / path_obj

    # Validate the path
    if not validate_path_security(
            path_obj,
            allowed_paths=allowed_paths,
            allow_external=allow_external
    ):
        raise PathSecurityError(f"Path failed security validation: {path_obj}")

    return path_obj