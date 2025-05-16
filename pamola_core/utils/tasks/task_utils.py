"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Utilities
Description: Utility functions for working with tasks
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides utility functions for working with tasks,
including directory management, file naming, and data source preparation.

Key features:
- Directory creation and management
- Path resolution and naming utilities
- Data source preparation from file paths
- Execution time formatting
- Previous task artifact discovery
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from pamola_core.utils.io import ensure_directory, get_timestamped_filename
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.progress import SimpleProgressBar
from pamola_core.utils.tasks.task_config import validate_path_security

# Set up logger
logger = logging.getLogger(__name__)


def create_task_directories(task_dir: Path) -> Dict[str, Path]:
    """
    Create standard directories for a task.

    This function creates the standard directory structure used by tasks,
    including directories for outputs, dictionaries, visualizations, and logs.

    Args:
        task_dir: Base directory for the task

    Returns:
        Dictionary with paths to standard directories
    """
    # Validate task_dir for security
    if not validate_path_security(task_dir):
        logger.error(f"Insecure task directory path: {task_dir}")
        raise ValueError(f"Insecure task directory path: {task_dir}")

    # Define standard directories
    directories = {
        "output": task_dir / "output",
        "dictionaries": task_dir / "dictionaries",
        "visualizations": task_dir / "visualizations",
        "metrics": task_dir / "metrics",
        "logs": task_dir.parent.parent / "logs"
    }

    # Create directories
    for name, dir_path in directories.items():
        ensure_directory(dir_path)
        logger.debug(f"Created directory: {dir_path}")

    return directories


def prepare_data_source_from_paths(file_paths: Dict[str, str],
                                   show_progress: bool = True) -> DataSource:
    """
    Prepare a data source from file paths.

    This function creates a DataSource with the provided file paths,
    which can then be used with operations.

    Args:
        file_paths: Dictionary mapping dataset names to file paths
        show_progress: Whether to show a progress bar during loading

    Returns:
        DataSource with file paths added
    """
    data_source = DataSource()

    # Create progress tracker if requested
    progress = None
    if show_progress and len(file_paths) > 1:
        progress = SimpleProgressBar(total=len(file_paths), description="Adding files to data source")

    try:
        for i, (name, path) in enumerate(file_paths.items()):
            # Validate path security
            if not validate_path_security(path):
                logger.error(f"Insecure input path: {path}")
                raise ValueError(f"Insecure input path: {path}")

            # Convert path to Path object if it's a string
            path_obj = Path(path) if isinstance(path, str) else path

            # Add to data source
            data_source.add_file_path(name, path_obj)
            logger.debug(f"Added file path '{name}': {path_obj}")

            # Update progress
            if progress:
                progress.update(i + 1)
    finally:
        # Clean up progress bar
        if progress:
            progress.close()

    return data_source


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in seconds to a human-readable string.

    Args:
        seconds: Execution time in seconds

    Returns:
        Formatted execution time string
    """
    if seconds < 0.1:
        return f"{seconds * 1000:.2f} milliseconds"
    elif seconds < 1:
        return f"{seconds * 1000:.0f} milliseconds"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        seconds_remainder = seconds % 60
        return f"{int(minutes)} minutes, {seconds_remainder:.0f} seconds"
    else:
        hours = seconds / 3600
        minutes_remainder = (seconds % 3600) / 60
        return f"{int(hours)} hours, {int(minutes_remainder)} minutes"


def get_artifact_path(task_dir: Path,
                      artifact_name: str,
                      artifact_type: str = "json",
                      sub_dir: str = "output",
                      include_timestamp: bool = True) -> Path:
    """
    Get a standardized path for a task artifact.

    Args:
        task_dir: Base directory for the task
        artifact_name: Name of the artifact
        artifact_type: Type/extension of the artifact
        sub_dir: Subdirectory for the artifact
        include_timestamp: Whether to include a timestamp in the filename

    Returns:
        Path to the artifact
    """
    # Validate task_dir and artifact_name for security
    if not validate_path_security(task_dir):
        logger.error(f"Insecure task directory path: {task_dir}")
        raise ValueError(f"Insecure task directory path: {task_dir}")

    if not validate_path_security(artifact_name):
        logger.error(f"Insecure artifact name: {artifact_name}")
        raise ValueError(f"Insecure artifact name: {artifact_name}")

    # Get the appropriate subdirectory
    artifact_dir = task_dir / sub_dir
    ensure_directory(artifact_dir)

    # Generate timestamped filename if requested
    if include_timestamp:
        filename = get_timestamped_filename(artifact_name, artifact_type)
    else:
        # Ensure extension format
        ext = artifact_type if artifact_type.startswith('.') else f'.{artifact_type}'
        filename = f"{artifact_name}{ext}"

    return artifact_dir / filename


def find_previous_output(task_id: str,
                         data_repository: Optional[Path] = None,
                         project_root: Optional[Path] = None,
                         file_pattern: Optional[str] = None) -> List[Path]:
    """
    Find output files from a previous task.

    This function searches for output files from a previous task,
    either in the standardized location or by pattern matching.

    Args:
        task_id: ID of the previous task
        data_repository: Path to the data repository (optional)
        project_root: Path to the project root (optional)
        file_pattern: Glob pattern to match specific files (optional)

    Returns:
        List of paths to output files
    """
    # Validate task_id and file_pattern for security
    if not validate_path_security(task_id):
        logger.error(f"Insecure task ID: {task_id}")
        raise ValueError(f"Insecure task ID: {task_id}")

    if file_pattern and not validate_path_security(file_pattern):
        logger.error(f"Insecure file pattern: {file_pattern}")
        raise ValueError(f"Insecure file pattern: {file_pattern}")

    # Find data repository if not provided
    if data_repository is None:
        if project_root is None:
            # Try to find project root
            from pamola_core.utils.tasks.task_config import find_project_root
            project_root = find_project_root()

        # Use default data repository location
        data_repository = project_root / "DATA"

    # Check if data repository exists
    if not data_repository.exists():
        logger.warning(f"Data repository not found: {data_repository}")
        return []

    # Build path to previous task output directory
    previous_task_dir = data_repository / "processed" / task_id
    previous_output_dir = previous_task_dir / "output"

    # Check if output directory exists
    if not previous_output_dir.exists():
        logger.warning(f"Previous task output directory not found: {previous_output_dir}")
        return []

    # Find files matching pattern or all files in output directory
    if file_pattern:
        output_files = list(previous_output_dir.glob(file_pattern))
    else:
        output_files = list(previous_output_dir.iterdir())

    # Filter out directories
    output_files = [f for f in output_files if f.is_file()]

    # Sort by modification time (newest first)
    output_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    return output_files


def find_task_report(task_id: str,
                     data_repository: Optional[Path] = None,
                     project_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the report file from a previous task.

    Args:
        task_id: ID of the previous task
        data_repository: Path to the data repository (optional)
        project_root: Path to the project root (optional)

    Returns:
        Path to the report file or None if not found
    """
    # Validate task_id for security
    if not validate_path_security(task_id):
        logger.error(f"Insecure task ID: {task_id}")
        raise ValueError(f"Insecure task ID: {task_id}")

    # Find data repository if not provided
    if data_repository is None:
        if project_root is None:
            # Try to find project root
            from pamola_core.utils.tasks.task_config import find_project_root
            project_root = find_project_root()

        # Use default data repository location
        data_repository = project_root / "DATA"

    # Check if data repository exists
    if not data_repository.exists():
        logger.warning(f"Data repository not found: {data_repository}")
        return None

    # Build path to reports directory
    reports_dir = data_repository / "reports"

    # Check if reports directory exists
    if not reports_dir.exists():
        logger.warning(f"Reports directory not found: {reports_dir}")
        return None

    # Check for report file
    report_path = reports_dir / f"{task_id}_report.json"
    if report_path.exists():
        return report_path

    # Look for alternative formats
    alternative_paths = [
        reports_dir / f"{task_id}.json",
        reports_dir / task_id / "report.json"
    ]

    for path in alternative_paths:
        if path.exists():
            return path

    return None


def get_temp_dir(task_dir: Path) -> Path:
    """
    Get a temporary directory for the task.

    Args:
        task_dir: Base directory for the task

    Returns:
        Path to the temporary directory
    """
    # Validate task_dir for security
    if not validate_path_security(task_dir):
        logger.error(f"Insecure task directory path: {task_dir}")
        raise ValueError(f"Insecure task directory path: {task_dir}")

    temp_dir = task_dir / "temp"
    ensure_directory(temp_dir)
    return temp_dir


def clean_temp_dir(task_dir: Path) -> bool:
    """
    Clean the temporary directory for the task.

    Args:
        task_dir: Base directory for the task

    Returns:
        True if cleaning was successful, False otherwise
    """
    import shutil

    # Validate task_dir for security
    if not validate_path_security(task_dir):
        logger.error(f"Insecure task directory path: {task_dir}")
        raise ValueError(f"Insecure task directory path: {task_dir}")

    temp_dir = task_dir / "temp"
    if temp_dir.exists():
        try:
            # Remove all files in temp directory
            for item in temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            return True
        except Exception as e:
            logger.error(f"Error cleaning temporary directory: {str(e)}")
            return False
    return True


def format_error_for_report(error: Exception) -> Dict[str, Any]:
    """
    Format an exception for inclusion in a task report.

    Args:
        error: Exception to format

    Returns:
        Dictionary with formatted error information
    """
    import traceback

    return {
        "type": error.__class__.__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }


def ensure_secure_directory(path: Union[str, Path]) -> Path:
    """
    Create a directory with secure permissions.

    This function creates a directory with permissions that ensure
    only the current user can access it.

    Args:
        path: Path to the directory

    Returns:
        Path to the created directory
    """
    # Validate path for security
    if not validate_path_security(path):
        logger.error(f"Insecure directory path: {path}")
        raise ValueError(f"Insecure directory path: {path}")

    # Create the directory
    dir_path = ensure_directory(Path(path))

    try:
        # Set secure permissions
        if os.name == 'posix':  # Unix-like systems
            import stat
            os.chmod(dir_path, stat.S_IRWXU)  # 0o700 - User read/write/execute only
            logger.debug(f"Set secure permissions (0o700) for directory: {dir_path}")
    except Exception as e:
        logger.warning(f"Could not set secure permissions for directory {dir_path}: {str(e)}")

    return dir_path


def is_master_key_exposed() -> bool:
    """
    Check if the master encryption key has insecure permissions.

    Returns:
        True if the master key has insecure permissions, False otherwise
    """
    try:
        from pamola_core.utils.crypto_helpers.key_store import is_master_key_exposed
        return is_master_key_exposed()
    except ImportError:
        logger.debug("Crypto key store module not available, cannot check master key exposure")
        return False


def extract_previous_output_info(task_id: str,
                                 data_repository: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extract information about outputs from a previous task.

    This function extracts metadata about outputs from a previous task
    by reading its report file.

    Args:
        task_id: ID of the previous task
        data_repository: Path to the data repository (optional)

    Returns:
        Dictionary with information about previous outputs
    """
    # Validate task_id for security
    if not validate_path_security(task_id):
        logger.error(f"Insecure task ID: {task_id}")
        raise ValueError(f"Insecure task ID: {task_id}")

    # Find the report file
    report_path = find_task_report(task_id, data_repository)
    if not report_path:
        logger.warning(f"Could not find report for task {task_id}")
        return {}

    try:
        # Read the report file
        import json
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # Extract artifacts information
        artifacts = report.get("artifacts", [])
        result = {
            "task_id": task_id,
            "report_path": str(report_path),
            "execution_time": report.get("execution_time_seconds"),
            "status": report.get("status"),
            "artifacts": {}
        }

        # Organize artifacts by type
        for artifact in artifacts:
            artifact_type = artifact.get("type")
            if artifact_type not in result["artifacts"]:
                result["artifacts"][artifact_type] = []

            # Validate artifact path for security
            artifact_path = artifact.get("path")
            if not validate_path_security(artifact_path):
                logger.warning(f"Skipping insecure artifact path: {artifact_path}")
                continue

            result["artifacts"][artifact_type].append({
                "path": artifact_path,
                "description": artifact.get("description"),
                "filename": artifact.get("filename", Path(artifact_path).name)
            })

        return result
    except Exception as e:
        logger.error(f"Error extracting information from previous task report: {str(e)}")
        return {}