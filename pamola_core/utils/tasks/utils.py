"""
Task Utilities Module for HHR project.

This module provides utility functions for working with tasks,
including directory management, file naming, and data source preparation.
"""

from pathlib import Path
from typing import Dict

from pamola_core.utils.io import ensure_directory, get_timestamped_filename
from pamola_core.utils.ops.op_data_source import DataSource


def create_task_directories(task_dir: Path) -> Dict[str, Path]:
    """
    Create standard directories for a task.

    Parameters:
    -----------
    task_dir : Path
        Base directory for the task

    Returns:
    --------
    Dict[str, Path]
        Dictionary with paths to standard directories
    """
    # Define standard directories
    directories = {
        "output": task_dir / "output",
        "dictionaries": task_dir / "dictionaries",
        "visualizations": task_dir / "visualizations",
        "logs": task_dir.parent.parent / "logs" / task_dir.parent.name / task_dir.name
    }

    # Create directories
    for name, dir_path in directories.items():
        ensure_directory(dir_path)

    return directories


def prepare_data_source_from_paths(file_paths: Dict[str, str]) -> DataSource:
    """
    Prepare a data source from file paths.

    Parameters:
    -----------
    file_paths : Dict[str, str]
        Dictionary mapping dataset names to file paths

    Returns:
    --------
    DataSource
        Data source with file paths added
    """
    data_source = DataSource()

    for name, path in file_paths.items():
        data_source.add_file_path(name, path)

    return data_source


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in seconds to a human-readable string.

    Parameters:
    -----------
    seconds : float
        Execution time in seconds

    Returns:
    --------
    str
        Formatted execution time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def get_artifact_path(task_dir: Path,
                      artifact_name: str,
                      artifact_type: str = "json",
                      sub_dir: str = "output",
                      include_timestamp: bool = True) -> Path:
    """
    Get a standardized path for a task artifact.

    Parameters:
    -----------
    task_dir : Path
        Base directory for the task
    artifact_name : str
        Name of the artifact
    artifact_type : str
        Type/extension of the artifact
    sub_dir : str
        Subdirectory for the artifact
    include_timestamp : bool{
  "tasks": {},
  "executions": []
}
        Whether to include a timestamp in the filename

    Returns:
    --------
    Path
        Path to the artifact
    """
    # Get the appropriate subdirectory
    artifact_dir = task_dir / sub_dir
    ensure_directory(artifact_dir)

    # Generate timestamped filename
    filename = get_timestamped_filename(artifact_name, artifact_type, include_timestamp)

    return artifact_dir / filename