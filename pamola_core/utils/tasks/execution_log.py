"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Execution Log
Description: Manage persistent task execution history
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for managing persistent task execution
history at the project level, enabling tracking of task dependencies,
data flow, and execution status.

Key features:
- Execution log initialization and management
- Task execution recording and metadata tracking
- Data flow tracking (input/output dependencies)
- Execution history querying
- Task dependency validation

The module is designed to work with the progress_manager.py module
for coordinated progress tracking and logging. When a progress_manager
is provided, operations like recording task execution or updating the
execution log will be tracked with appropriate progress bars.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Protocol

import filelock

from pamola_core.utils.io import read_json, write_json, ensure_directory
from pamola_core.utils.tasks.task_config import find_project_root, validate_path_security

# Set up logger
logger = logging.getLogger(__name__)

# Default execution log path
DEFAULT_EXECUTION_LOG_PATH = "configs/execution_log.json"
# Default lock timeout (10 seconds)
DEFAULT_LOCK_TIMEOUT = 10


class ProgressManagerProtocol(Protocol):
    """Protocol defining the interface for progress managers."""

    def create_operation_context(self, name: str, total: int, description: Optional[str] = None,
                                 unit: str = "items", leave: bool = False) -> Any:
        """Create a context manager for tracking an operation's progress."""
        ...

    def log_info(self, message: str) -> None:
        """Log an info message without breaking progress bars."""
        ...

    def log_warning(self, message: str) -> None:
        """Log a warning message without breaking progress bars."""
        ...

    def log_error(self, message: str) -> None:
        """Log an error message without breaking progress bars."""
        ...


class ExecutionLogError(Exception):
    """Exception raised for execution log errors."""
    pass


def _get_execution_log_path() -> Path:
    """
    Get the path to the execution log file.

    Returns:
        Path to the execution log file
    """
    # Find project root
    project_root = find_project_root()

    # Get path from environment variable if set
    env_path = os.environ.get("PAMOLA_EXECUTION_LOG_PATH")
    if env_path:
        log_path = Path(env_path)
        if not log_path.is_absolute():
            log_path = project_root / log_path
    else:
        # Use default path
        log_path = project_root / DEFAULT_EXECUTION_LOG_PATH

    return log_path


def initialize_execution_log(project_path: Optional[Path] = None,
                             progress_manager: Optional[ProgressManagerProtocol] = None) -> Path:
    """
    Initialize the execution log file.

    Creates or initializes the execution log file with empty data.

    Args:
        project_path: Path to the project root (optional, auto-detected if None)
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Path to the execution log file

    Raises:
        ExecutionLogError: If initialization fails
    """
    # Get project path if not provided
    if project_path is None:
        project_path = find_project_root()

    # Get execution log path
    log_path = _get_execution_log_path()

    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="initialize_execution_log",
                total=2,
                description="Initializing execution log"
        ) as progress:
            # Create parent directory if it doesn't exist
            ensure_directory(log_path.parent)
            progress.update(1, {"status": "directory_created"})

            if not log_path.exists():
                try:
                    # Create lock file path
                    lock_path = f"{log_path}.lock"

                    # Use file lock to prevent race conditions
                    with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
                        # Check again to prevent race condition
                        if not log_path.exists():
                            # Create empty registry
                            registry_data = {
                                "tasks": {},  # Task registry
                                "executions": [],  # Execution history
                                "timestamp": datetime.now().isoformat(),
                                "version": "1.0.0"  # Schema version for future compatibility
                            }

                            # Save registry
                            write_json(registry_data, log_path)
                            progress_manager.log_info(f"Initialized execution log at {log_path}")
                except filelock.Timeout:
                    progress_manager.log_warning(f"Timeout waiting for lock on execution log: {log_path}")
                    # Continue as another process may have initialized the log
                except Exception as e:
                    progress_manager.log_error(f"Error initializing execution log: {e}")
                    raise ExecutionLogError(f"Failed to initialize execution log: {e}")

            progress.update(1, {"status": "log_initialized"})
            return log_path
    else:
        # Create parent directory if it doesn't exist
        ensure_directory(log_path.parent)

        if not log_path.exists():
            try:
                # Create lock file path
                lock_path = f"{log_path}.lock"

                # Use file lock to prevent race conditions
                with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
                    # Check again to prevent race condition
                    if not log_path.exists():
                        # Create empty registry
                        registry_data = {
                            "tasks": {},  # Task registry
                            "executions": [],  # Execution history
                            "timestamp": datetime.now().isoformat(),
                            "version": "1.0.0"  # Schema version for future compatibility
                        }

                        # Save registry
                        write_json(registry_data, log_path)
                        logger.info(f"Initialized execution log at {log_path}")
            except filelock.Timeout:
                logger.warning(f"Timeout waiting for lock on execution log: {log_path}")
                # Continue as another process may have initialized the log
            except Exception as e:
                logger.error(f"Error initializing execution log: {e}")
                raise ExecutionLogError(f"Failed to initialize execution log: {e}")

        return log_path


def _load_execution_log(progress_manager: Optional[ProgressManagerProtocol] = None) -> Dict[str, Any]:
    """
    Load the execution log.

    Args:
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Execution log data

    Raises:
        ExecutionLogError: If log cannot be loaded
    """
    log_path = _get_execution_log_path()

    # Initialize if it doesn't exist
    if not log_path.exists():
        initialize_execution_log(progress_manager=progress_manager)

    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="load_execution_log",
                total=1,
                description="Loading execution log"
        ) as progress:
            try:
                # Create lock file path
                lock_path = f"{log_path}.lock"

                # Use file lock to prevent race conditions
                with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
                    # Load execution log
                    data = read_json(log_path)

                    # Basic validation
                    if not isinstance(data, dict) or "tasks" not in data or "executions" not in data:
                        error_msg = f"Invalid execution log format at {log_path}"
                        progress_manager.log_error(error_msg)
                        raise ExecutionLogError(error_msg)

                    progress.update(1, {"status": "success", "entries": len(data.get("executions", []))})
                    return data
            except filelock.Timeout:
                error_msg = f"Timeout waiting for lock on execution log: {log_path}"
                progress_manager.log_warning(error_msg)
                raise ExecutionLogError(error_msg)
            except Exception as e:
                error_msg = f"Error loading execution log: {e}"
                progress_manager.log_error(error_msg)
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Create lock file path
            lock_path = f"{log_path}.lock"

            # Use file lock to prevent race conditions
            with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
                # Load execution log
                data = read_json(log_path)

                # Basic validation
                if not isinstance(data, dict) or "tasks" not in data or "executions" not in data:
                    logger.error(f"Invalid execution log format at {log_path}")
                    raise ExecutionLogError(f"Invalid execution log format at {log_path}")

                return data
        except filelock.Timeout:
            logger.warning(f"Timeout waiting for lock on execution log: {log_path}")
            raise ExecutionLogError(f"Timeout waiting for lock on execution log: {log_path}")
        except Exception as e:
            logger.error(f"Error loading execution log: {e}")
            raise ExecutionLogError(f"Failed to load execution log: {e}")


def _save_execution_log(data: Dict[str, Any],
                        progress_manager: Optional[ProgressManagerProtocol] = None) -> None:
    """
    Save the execution log.

    Args:
        data: Execution log data
        progress_manager: Progress manager for tracking (optional)

    Raises:
        ExecutionLogError: If log cannot be saved
    """
    log_path = _get_execution_log_path()

    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="save_execution_log",
                total=1,
                description="Saving execution log"
        ) as progress:
            try:
                # Create lock file path
                lock_path = f"{log_path}.lock"

                # Use file lock to prevent race conditions
                with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
                    # Update last modified time
                    data["last_modified"] = datetime.now().isoformat()

                    # Save execution log
                    write_json(data, log_path)
                    progress.update(1, {"status": "success"})
            except filelock.Timeout:
                error_msg = f"Timeout waiting for lock on execution log: {log_path}"
                progress_manager.log_warning(error_msg)
                progress.update(1, {"status": "timeout"})
                raise ExecutionLogError(error_msg)
            except Exception as e:
                error_msg = f"Error saving execution log: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Create lock file path
            lock_path = f"{log_path}.lock"

            # Use file lock to prevent race conditions
            with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
                # Update last modified time
                data["last_modified"] = datetime.now().isoformat()

                # Save execution log
                write_json(data, log_path)
        except filelock.Timeout:
            logger.warning(f"Timeout waiting for lock on execution log: {log_path}")
            raise ExecutionLogError(f"Timeout waiting for lock on execution log: {log_path}")
        except Exception as e:
            logger.error(f"Error saving execution log: {e}")
            raise ExecutionLogError(f"Failed to save execution log: {e}")


def record_task_execution(task_id: str,
                          task_type: str,
                          success: bool,
                          execution_time: float,
                          report_path: Path,
                          input_datasets: Optional[Dict[str, str]] = None,
                          output_artifacts: Optional[List[Any]] = None,
                          progress_manager: Optional[ProgressManagerProtocol] = None) -> Optional[str]:
    """
    Record a task execution in the execution log.

    Args:
        task_id: ID of the task
        task_type: Type of the task
        success: Whether the task executed successfully
        execution_time: Task execution time in seconds
        report_path: Path to the task report
        input_datasets: Dictionary of input datasets (optional)
        output_artifacts: List of output artifacts (optional)
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Task run UUID or None if recording fails

    Raises:
        ExecutionLogError: If recording fails
    """
    # Generate a unique ID for this execution
    task_run_id = str(uuid.uuid4())

    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="record_task_execution",
                total=4,
                description="Recording task execution in log"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)
                progress.update(1, {"status": "log_loaded"})

                # Process input datasets
                input_files = []
                if input_datasets:
                    for name, path in input_datasets.items():
                        # Validate path security
                        if not validate_path_security(path):
                            progress_manager.log_warning(f"Skipping insecure input path in execution log: {path}")
                            continue

                        input_files.append({
                            "name": name,
                            "path": str(path)
                        })
                progress.update(1, {"status": "inputs_processed", "count": len(input_files)})

                # Process output artifacts
                output_files = []
                if output_artifacts:
                    for artifact in output_artifacts:
                        try:
                            # Handle different artifact types
                            if hasattr(artifact, "path") and hasattr(artifact, "artifact_type"):
                                # Validate path security
                                if not validate_path_security(artifact.path):
                                    progress_manager.log_warning(
                                        f"Skipping insecure artifact path in execution log: {artifact.path}")
                                    continue

                                # OperationArtifact or similar
                                output_files.append({
                                    "path": str(artifact.path),
                                    "type": artifact.artifact_type,
                                    "description": getattr(artifact, "description", "")
                                })
                            elif isinstance(artifact, dict) and "path" in artifact:
                                # Validate path security
                                if not validate_path_security(artifact["path"]):
                                    progress_manager.log_warning(
                                        f"Skipping insecure artifact path in execution log: {artifact['path']}")
                                    continue

                                # Dictionary with path
                                output_files.append({
                                    "path": str(artifact["path"]),
                                    "type": artifact.get("type", "unknown"),
                                    "description": artifact.get("description", "")
                                })
                            elif isinstance(artifact, (str, Path)):
                                # Validate path security
                                if not validate_path_security(artifact):
                                    progress_manager.log_warning(
                                        f"Skipping insecure artifact path in execution log: {artifact}")
                                    continue

                                # Simple path
                                path_obj = Path(artifact)
                                output_files.append({
                                    "path": str(path_obj),
                                    "type": path_obj.suffix.lstrip("."),
                                    "description": ""
                                })
                        except Exception as e:
                            progress_manager.log_warning(f"Error processing artifact for execution log: {str(e)}")
                            continue
                progress.update(1, {"status": "outputs_processed", "count": len(output_files)})

                # Create execution record
                execution = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "task_run_id": task_run_id,
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "execution_time": execution_time,
                    "report_path": str(report_path),
                    "input_files": input_files,
                    "output_files": output_files,
                    "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown"
                }

                # Update task in tasks registry
                registry_data["tasks"][task_id] = {
                    "task_type": task_type,
                    "last_execution": int(time.time()),
                    "last_status": "success" if success else "failed",
                    "last_report_path": str(report_path),
                    "last_task_run_id": task_run_id
                }

                # Add execution to executions list
                registry_data["executions"].append(execution)

                # Save updated registry
                _save_execution_log(registry_data, progress_manager)
                progress.update(1, {"status": "log_saved"})

                progress_manager.log_info(f"Recorded execution of task {task_id} in execution log")
                return task_run_id

            except Exception as e:
                progress_manager.log_error(f"Failed to record task execution: {e}")
                progress.update(1, {"status": "error", "error": str(e)})
                # Return None to indicate failure
                return None
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Process input datasets
            input_files = []
            if input_datasets:
                for name, path in input_datasets.items():
                    # Validate path security
                    if not validate_path_security(path):
                        logger.warning(f"Skipping insecure input path in execution log: {path}")
                        continue

                    input_files.append({
                        "name": name,
                        "path": str(path)
                    })

            # Process output artifacts
            output_files = []
            if output_artifacts:
                for artifact in output_artifacts:
                    try:
                        # Handle different artifact types
                        if hasattr(artifact, "path") and hasattr(artifact, "artifact_type"):
                            # Validate path security
                            if not validate_path_security(artifact.path):
                                logger.warning(f"Skipping insecure artifact path in execution log: {artifact.path}")
                                continue

                            # OperationArtifact or similar
                            output_files.append({
                                "path": str(artifact.path),
                                "type": artifact.artifact_type,
                                "description": getattr(artifact, "description", "")
                            })
                        elif isinstance(artifact, dict) and "path" in artifact:
                            # Validate path security
                            if not validate_path_security(artifact["path"]):
                                logger.warning(f"Skipping insecure artifact path in execution log: {artifact['path']}")
                                continue

                            # Dictionary with path
                            output_files.append({
                                "path": str(artifact["path"]),
                                "type": artifact.get("type", "unknown"),
                                "description": artifact.get("description", "")
                            })
                        elif isinstance(artifact, (str, Path)):
                            # Validate path security
                            if not validate_path_security(artifact):
                                logger.warning(f"Skipping insecure artifact path in execution log: {artifact}")
                                continue

                            # Simple path
                            path_obj = Path(artifact)
                            output_files.append({
                                "path": str(path_obj),
                                "type": path_obj.suffix.lstrip("."),
                                "description": ""
                            })
                    except Exception as e:
                        logger.warning(f"Error processing artifact for execution log: {str(e)}")
                        continue

            # Create execution record
            execution = {
                "task_id": task_id,
                "task_type": task_type,
                "task_run_id": task_run_id,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "execution_time": execution_time,
                "report_path": str(report_path),
                "input_files": input_files,
                "output_files": output_files,
                "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown"
            }

            # Update task in tasks registry
            registry_data["tasks"][task_id] = {
                "task_type": task_type,
                "last_execution": int(time.time()),
                "last_status": "success" if success else "failed",
                "last_report_path": str(report_path),
                "last_task_run_id": task_run_id
            }

            # Add execution to executions list
            registry_data["executions"].append(execution)

            # Save updated registry
            _save_execution_log(registry_data)
            logger.info(f"Recorded execution of task {task_id} in execution log")

            return task_run_id

        except Exception as e:
            logger.error(f"Failed to record task execution: {e}")
            # Return None to indicate failure
            return None


def get_task_execution_history(task_id: Optional[str] = None,
                               limit: int = 10,
                               success_only: bool = False,
                               progress_manager: Optional[ProgressManagerProtocol] = None) -> List[Dict[str, Any]]:
    """
    Get execution history for a specific task or all tasks.

    Args:
        task_id: ID of the task (None for all tasks)
        limit: Maximum number of executions to return
        success_only: Whether to include only successful executions
        progress_manager: Progress manager for tracking (optional)

    Returns:
        List of execution records

    Raises:
        ExecutionLogError: If history cannot be retrieved
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="get_execution_history",
                total=1,
                description=f"Getting execution history for {'task ' + task_id if task_id else 'all tasks'}"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)

                # Get executions
                executions = registry_data["executions"]

                # Filter by task_id if provided
                if task_id:
                    executions = [e for e in executions if e.get("task_id") == task_id]

                # Filter by success if requested
                if success_only:
                    executions = [e for e in executions if e.get("success")]

                # Sort by timestamp (newest first)
                executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

                # Limit number of results
                results = executions[:limit]
                progress.update(1, {"status": "success", "count": len(results)})
                return results

            except Exception as e:
                error_msg = f"Error getting task execution history: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Get executions
            executions = registry_data["executions"]

            # Filter by task_id if provided
            if task_id:
                executions = [e for e in executions if e.get("task_id") == task_id]

            # Filter by success if requested
            if success_only:
                executions = [e for e in executions if e.get("success")]

            # Sort by timestamp (newest first)
            executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

            # Limit number of results
            return executions[:limit]

        except Exception as e:
            logger.error(f"Error getting task execution history: {e}")
            raise ExecutionLogError(f"Failed to get task execution history: {e}")


def find_latest_execution(task_id: str, success_only: bool = True,
                          progress_manager: Optional[ProgressManagerProtocol] = None) -> Optional[Dict[str, Any]]:
    """
    Find the most recent execution of a task.

    Args:
        task_id: ID of the task
        success_only: Whether to include only successful executions
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Execution record or None if not found

    Raises:
        ExecutionLogError: If execution cannot be found
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="find_latest_execution",
                total=1,
                description=f"Finding latest execution for task {task_id}"
        ) as progress:
            try:
                # Get execution history for the task
                history = get_task_execution_history(task_id, limit=1, success_only=success_only,
                                                     progress_manager=progress_manager)

                # Return the first (most recent) execution if available
                result = history[0] if history else None
                progress.update(1, {"status": "success", "found": result is not None})
                return result

            except Exception as e:
                error_msg = f"Error finding latest execution: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Get execution history for the task
            history = get_task_execution_history(task_id, limit=1, success_only=success_only)

            # Return the first (most recent) execution if available
            return history[0] if history else None

        except Exception as e:
            logger.error(f"Error finding latest execution: {e}")
            raise ExecutionLogError(f"Failed to find latest execution: {e}")


def find_task_by_output(file_path: Union[str, Path],
                        progress_manager: Optional[ProgressManagerProtocol] = None) -> Optional[Dict[str, Any]]:
    """
    Find the task that produced a specific output file.

    Args:
        file_path: Path to the output file
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Task execution record or None if not found

    Raises:
        ExecutionLogError: If task cannot be found
    """
    # Normalize file path
    file_path_str = str(Path(file_path).resolve())

    # Validate path security
    if not validate_path_security(file_path_str):
        error_msg = f"Insecure file path detected: {file_path_str}"
        if progress_manager:
            progress_manager.log_error(error_msg)
        else:
            logger.error(error_msg)
        raise ExecutionLogError(f"Insecure file path: {file_path_str}")

    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="find_task_by_output",
                total=1,
                description=f"Finding task for output file"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)

                # Search all executions for the output file
                for execution in registry_data["executions"]:
                    for output_file in execution.get("output_files", []):
                        output_path = output_file.get("path", "")
                        if output_path == file_path_str:
                            progress.update(1, {"status": "success", "found": True})
                            return execution

                # File not found in any execution
                progress.update(1, {"status": "success", "found": False})
                return None

            except Exception as e:
                error_msg = f"Error finding task by output: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Search all executions for the output file
            for execution in registry_data["executions"]:
                for output_file in execution.get("output_files", []):
                    output_path = output_file.get("path", "")
                    if output_path == file_path_str:
                        return execution

            # File not found in any execution
            return None

        except Exception as e:
            logger.error(f"Error finding task by output: {e}")
            raise ExecutionLogError(f"Failed to find task by output: {e}")


def track_input_files(task_id: str, file_paths: List[Union[str, Path]],
                      progress_manager: Optional[ProgressManagerProtocol] = None) -> bool:
    """
    Register input files for a task.

    This function updates the task's input files in the execution log,
    which can be used to trace data flow.

    Args:
        task_id: ID of the task
        file_paths: List of input file paths
        progress_manager: Progress manager for tracking (optional)

    Returns:
        True if successful, False otherwise

    Raises:
        ExecutionLogError: If tracking fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="track_input_files",
                total=1,
                description=f"Tracking input files for task {task_id}"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)

                # Find the latest execution of the task
                executions = [e for e in registry_data["executions"] if e.get("task_id") == task_id]
                if not executions:
                    progress_manager.log_warning(f"No execution found for task {task_id}")
                    progress.update(1, {"status": "no_execution"})
                    return False

                # Sort by timestamp (newest first)
                executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
                latest_execution = executions[0]

                # Update input files
                input_files = latest_execution.get("input_files", [])
                added_count = 0
                for path in file_paths:
                    # Validate path security
                    if not validate_path_security(path):
                        progress_manager.log_warning(f"Skipping insecure input path in execution log: {path}")
                        continue

                    path_str = str(path)
                    # Check if already present
                    if not any(f.get("path") == path_str for f in input_files):
                        input_files.append({
                            "path": path_str,
                            "added_at": datetime.now().isoformat()
                        })
                        added_count += 1

                latest_execution["input_files"] = input_files

                # Save updated registry
                _save_execution_log(registry_data, progress_manager)
                progress.update(1, {"status": "success", "added": added_count})
                progress_manager.log_info(f"Updated input files for task {task_id}")

                return True

            except Exception as e:
                error_msg = f"Error tracking input files: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Find the latest execution of the task
            executions = [e for e in registry_data["executions"] if e.get("task_id") == task_id]
            if not executions:
                logger.warning(f"No execution found for task {task_id}")
                return False

            # Sort by timestamp (newest first)
            executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
            latest_execution = executions[0]

            # Update input files
            input_files = latest_execution.get("input_files", [])
            for path in file_paths:
                # Validate path security
                if not validate_path_security(path):
                    logger.warning(f"Skipping insecure input path in execution log: {path}")
                    continue

                path_str = str(path)
                # Check if already present
                if not any(f.get("path") == path_str for f in input_files):
                    input_files.append({
                        "path": path_str,
                        "added_at": datetime.now().isoformat()
                    })

            latest_execution["input_files"] = input_files

            # Save updated registry
            _save_execution_log(registry_data)
            logger.info(f"Updated input files for task {task_id}")

            return True

        except Exception as e:
            logger.error(f"Error tracking input files: {e}")
            raise ExecutionLogError(f"Failed to track input files: {e}")


def track_output_files(task_id: str, file_paths: List[Union[str, Path]],
                       progress_manager: Optional[ProgressManagerProtocol] = None) -> bool:
    """
    Register output files from a task.

    This function updates the task's output files in the execution log,
    which can be used to trace data flow.

    Args:
        task_id: ID of the task
        file_paths: List of output file paths
        progress_manager: Progress manager for tracking (optional)

    Returns:
        True if successful, False otherwise

    Raises:
        ExecutionLogError: If tracking fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="track_output_files",
                total=1,
                description=f"Tracking output files for task {task_id}"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)

                # Find the latest execution of the task
                executions = [e for e in registry_data["executions"] if e.get("task_id") == task_id]
                if not executions:
                    progress_manager.log_warning(f"No execution found for task {task_id}")
                    progress.update(1, {"status": "no_execution"})
                    return False

                # Sort by timestamp (newest first)
                executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
                latest_execution = executions[0]

                # Update output files
                output_files = latest_execution.get("output_files", [])
                added_count = 0
                for path in file_paths:
                    # Validate path security
                    if not validate_path_security(path):
                        progress_manager.log_warning(f"Skipping insecure output path in execution log: {path}")
                        continue

                    path_obj = Path(path)
                    path_str = str(path_obj)
                    # Check if already present
                    if not any(f.get("path") == path_str for f in output_files):
                        output_files.append({
                            "path": path_str,
                            "type": path_obj.suffix.lstrip("."),
                            "added_at": datetime.now().isoformat()
                        })
                        added_count += 1

                latest_execution["output_files"] = output_files

                # Save updated registry
                _save_execution_log(registry_data, progress_manager)
                progress.update(1, {"status": "success", "added": added_count})
                progress_manager.log_info(f"Updated output files for task {task_id}")

                return True

            except Exception as e:
                error_msg = f"Error tracking output files: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Find the latest execution of the task
            executions = [e for e in registry_data["executions"] if e.get("task_id") == task_id]
            if not executions:
                logger.warning(f"No execution found for task {task_id}")
                return False

            # Sort by timestamp (newest first)
            executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
            latest_execution = executions[0]

            # Update output files
            output_files = latest_execution.get("output_files", [])
            for path in file_paths:
                # Validate path security
                if not validate_path_security(path):
                    logger.warning(f"Skipping insecure output path in execution log: {path}")
                    continue

                path_obj = Path(path)
                path_str = str(path_obj)
                # Check if already present
                if not any(f.get("path") == path_str for f in output_files):
                    output_files.append({
                        "path": path_str,
                        "type": path_obj.suffix.lstrip("."),
                        "added_at": datetime.now().isoformat()
                    })

            latest_execution["output_files"] = output_files

            # Save updated registry
            _save_execution_log(registry_data)
            logger.info(f"Updated output files for task {task_id}")

            return True

        except Exception as e:
            logger.error(f"Error tracking output files: {e}")
            raise ExecutionLogError(f"Failed to track output files: {e}")


def update_execution_record(task_run_id: str, updates: Dict[str, Any],
                            progress_manager: Optional[ProgressManagerProtocol] = None) -> bool:
    """
    Update an existing execution record.

    Args:
        task_run_id: Unique ID for the task execution
        updates: Dictionary of updates to apply
        progress_manager: Progress manager for tracking (optional)

    Returns:
        True if successful, False otherwise

    Raises:
        ExecutionLogError: If update fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="update_execution_record",
                total=1,
                description=f"Updating execution record"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)

                # Find the execution record
                found = False
                for execution in registry_data["executions"]:
                    if execution.get("task_run_id") == task_run_id:
                        # Apply updates
                        for key, value in updates.items():
                            # Skip updates to sensitive fields unless properly validated
                            if key in ["input_files", "output_files"]:
                                if isinstance(value, list):
                                    # For file lists, validate each path
                                    for item in value:
                                        if isinstance(item, dict) and "path" in item:
                                            if not validate_path_security(item["path"]):
                                                progress_manager.log_warning(
                                                    f"Skipping insecure path in update: {item['path']}")
                                                continue
                            execution[key] = value

                        found = True
                        break

                if not found:
                    # Execution not found
                    progress_manager.log_warning(f"Execution record {task_run_id} not found")
                    progress.update(1, {"status": "not_found"})
                    return False

                # Save updated registry
                _save_execution_log(registry_data, progress_manager)
                progress.update(1, {"status": "success"})
                progress_manager.log_info(f"Updated execution record {task_run_id}")

                return True

            except Exception as e:
                error_msg = f"Error updating execution record: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Find the execution record
            for execution in registry_data["executions"]:
                if execution.get("task_run_id") == task_run_id:
                    # Apply updates
                    for key, value in updates.items():
                        # Skip updates to sensitive fields unless properly validated
                        if key in ["input_files", "output_files"]:
                            if isinstance(value, list):
                                # For file lists, validate each path
                                for item in value:
                                    if isinstance(item, dict) and "path" in item:
                                        if not validate_path_security(item["path"]):
                                            logger.warning(f"Skipping insecure path in update: {item['path']}")
                                            continue
                        execution[key] = value

                    # Save updated registry
                    _save_execution_log(registry_data)
                    logger.info(f"Updated execution record {task_run_id}")

                    return True

            # Execution not found
            logger.warning(f"Execution record {task_run_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error updating execution record: {e}")
            raise ExecutionLogError(f"Failed to update execution record: {e}")


def remove_execution_record(task_run_id: str,
                            progress_manager: Optional[ProgressManagerProtocol] = None) -> bool:
    """
    Remove an execution record from the log.

    Args:
        task_run_id: Unique ID for the task execution
        progress_manager: Progress manager for tracking (optional)

    Returns:
        True if successful, False otherwise

    Raises:
        ExecutionLogError: If removal fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="remove_execution_record",
                total=1,
                description=f"Removing execution record"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)

                # Find the execution record
                found = False
                for i, execution in enumerate(registry_data["executions"]):
                    if execution.get("task_run_id") == task_run_id:
                        # Remove the execution
                        removed = registry_data["executions"].pop(i)
                        found = True
                        break

                if not found:
                    # Execution not found
                    progress_manager.log_warning(f"Execution record {task_run_id} not found")
                    progress.update(1, {"status": "not_found"})
                    return False

                # Save updated registry
                _save_execution_log(registry_data, progress_manager)
                progress.update(1, {"status": "success"})
                progress_manager.log_info(f"Removed execution record {task_run_id}")

                return True

            except Exception as e:
                error_msg = f"Error removing execution record: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Find the execution record
            for i, execution in enumerate(registry_data["executions"]):
                if execution.get("task_run_id") == task_run_id:
                    # Remove the execution
                    registry_data["executions"].pop(i)

                    # Save updated registry
                    _save_execution_log(registry_data)
                    logger.info(f"Removed execution record {task_run_id}")

                    return True

            # Execution not found
            logger.warning(f"Execution record {task_run_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error removing execution record: {e}")
            raise ExecutionLogError(f"Failed to remove execution record: {e}")


def cleanup_old_executions(max_age_days: int = 30,
                           max_per_task: int = 10,
                           dry_run: bool = False,
                           progress_manager: Optional[ProgressManagerProtocol] = None) -> Tuple[int, List[str]]:
    """
    Clean up old execution records.

    Args:
        max_age_days: Maximum age of execution records to keep (in days)
        max_per_task: Maximum number of executions to keep per task
        dry_run: Whether to perform a dry run (don't actually delete)
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Tuple containing:
            - Number of records removed
            - List of removed task run IDs

    Raises:
        ExecutionLogError: If cleanup fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="cleanup_old_executions",
                total=3,
                description=f"Cleaning up old execution records"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)
                progress.update(1, {"status": "log_loaded"})

                # Calculate cutoff date
                cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

                # Group executions by task_id
                task_executions = {}
                for execution in registry_data["executions"]:
                    task_id = execution.get("task_id")
                    if task_id not in task_executions:
                        task_executions[task_id] = []
                    task_executions[task_id].append(execution)

                progress.update(1, {"status": "executions_grouped", "task_count": len(task_executions)})

                # Identify executions to remove
                removed_count = 0
                removed_ids = []
                new_executions = []

                for task_id, executions in task_executions.items():
                    # Sort by timestamp (newest first)
                    executions.sort(key=lambda e: datetime.fromisoformat(e.get("timestamp", "1970-01-01")).timestamp(),
                                    reverse=True)

                    # Keep recent executions up to max_per_task
                    keep_executions = executions[:max_per_task]

                    # Keep executions newer than cutoff date
                    for execution in executions[max_per_task:]:
                        timestamp = datetime.fromisoformat(execution.get("timestamp", "1970-01-01")).timestamp()
                        if timestamp > cutoff_date:
                            keep_executions.append(execution)
                        else:
                            removed_count += 1
                            removed_ids.append(execution.get("task_run_id"))

                    # Add to new executions list
                    new_executions.extend(keep_executions)

                # Update registry if not a dry run
                if not dry_run and removed_count > 0:
                    registry_data["executions"] = new_executions
                    _save_execution_log(registry_data, progress_manager)
                    progress_manager.log_info(f"Removed {removed_count} old execution records")

                progress.update(1, {
                    "status": "success",
                    "removed_count": removed_count,
                    "dry_run": dry_run
                })

                return removed_count, removed_ids

            except Exception as e:
                error_msg = f"Error cleaning up old executions: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Calculate cutoff date
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            # Group executions by task_id
            task_executions = {}
            for execution in registry_data["executions"]:
                task_id = execution.get("task_id")
                if task_id not in task_executions:
                    task_executions[task_id] = []
                task_executions[task_id].append(execution)

            # Identify executions to remove
            removed_count = 0
            removed_ids = []
            new_executions = []

            for task_id, executions in task_executions.items():
                # Sort by timestamp (newest first)
                executions.sort(key=lambda e: datetime.fromisoformat(e.get("timestamp", "1970-01-01")).timestamp(),
                                reverse=True)

                # Keep recent executions up to max_per_task
                keep_executions = executions[:max_per_task]

                # Keep executions newer than cutoff date
                for execution in executions[max_per_task:]:
                    timestamp = datetime.fromisoformat(execution.get("timestamp", "1970-01-01")).timestamp()
                    if timestamp > cutoff_date:
                        keep_executions.append(execution)
                    else:
                        removed_count += 1
                        removed_ids.append(execution.get("task_run_id"))

                # Add to new executions list
                new_executions.extend(keep_executions)

            # Update registry if not a dry run
            if not dry_run and removed_count > 0:
                registry_data["executions"] = new_executions
                _save_execution_log(registry_data)
                logger.info(f"Removed {removed_count} old execution records")

            return removed_count, removed_ids

        except Exception as e:
            logger.error(f"Error cleaning up old executions: {e}")
            raise ExecutionLogError(f"Failed to clean up old executions: {e}")


def validate_execution_log(progress_manager: Optional[ProgressManagerProtocol] = None) -> Tuple[bool, List[str]]:
    """
    Validate the execution log.

    Checks for inconsistencies, missing fields, and other issues.

    Args:
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Tuple containing:
            - Whether the log is valid
            - List of validation errors

    Raises:
        ExecutionLogError: If validation fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="validate_execution_log",
                total=3,
                description="Validating execution log"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)
                progress.update(1, {"status": "log_loaded"})

                errors = []

                # Check required top-level fields
                required_fields = ["tasks", "executions"]
                for field in required_fields:
                    if field not in registry_data:
                        errors.append(f"Missing required field: {field}")

                progress.update(1, {"status": "structure_checked"})

                # Check execution records
                for i, execution in enumerate(registry_data.get("executions", [])):
                    # Check required fields in execution records
                    execution_required_fields = ["task_id", "task_type", "task_run_id", "timestamp"]
                    for field in execution_required_fields:
                        if field not in execution:
                            errors.append(f"Execution {i}: Missing required field: {field}")

                    # Check task_run_id uniqueness
                    if "task_run_id" in execution:
                        task_run_id = execution["task_run_id"]
                        matching_executions = [e for e in registry_data.get("executions", [])
                                               if e.get("task_run_id") == task_run_id]
                        if len(matching_executions) > 1:
                            errors.append(f"Duplicate task_run_id: {task_run_id}")

                # Check task registry
                for task_id, task_info in registry_data.get("tasks", {}).items():
                    # Check required fields in task registry
                    task_required_fields = ["task_type", "last_execution"]
                    for field in task_required_fields:
                        if field not in task_info:
                            errors.append(f"Task {task_id}: Missing required field: {field}")

                progress.update(1, {
                    "status": "validation_complete",
                    "valid": len(errors) == 0,
                    "error_count": len(errors)
                })

                return len(errors) == 0, errors

            except Exception as e:
                error_msg = f"Error validating execution log: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            errors = []

            # Check required top-level fields
            required_fields = ["tasks", "executions"]
            for field in required_fields:
                if field not in registry_data:
                    errors.append(f"Missing required field: {field}")

            # Check execution records
            for i, execution in enumerate(registry_data.get("executions", [])):
                # Check required fields in execution records
                execution_required_fields = ["task_id", "task_type", "task_run_id", "timestamp"]
                for field in execution_required_fields:
                    if field not in execution:
                        errors.append(f"Execution {i}: Missing required field: {field}")

                # Check task_run_id uniqueness
                if "task_run_id" in execution:
                    task_run_id = execution["task_run_id"]
                    matching_executions = [e for e in registry_data.get("executions", [])
                                           if e.get("task_run_id") == task_run_id]
                    if len(matching_executions) > 1:
                        errors.append(f"Duplicate task_run_id: {task_run_id}")

            # Check task registry
            for task_id, task_info in registry_data.get("tasks", {}).items():
                # Check required fields in task registry
                task_required_fields = ["task_type", "last_execution"]
                for field in task_required_fields:
                    if field not in task_info:
                        errors.append(f"Task {task_id}: Missing required field: {field}")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating execution log: {e}")
            raise ExecutionLogError(f"Failed to validate execution log: {e}")


def export_execution_log(output_path: Optional[Path] = None,
                         format: str = "json",
                         progress_manager: Optional[ProgressManagerProtocol] = None) -> Path:
    """
    Export the execution log to a file.

    Args:
        output_path: Path to save the export (optional)
        format: Export format ("json" or "csv")
        progress_manager: Progress manager for tracking (optional)

    Returns:
        Path to the exported file

    Raises:
        ExecutionLogError: If export fails
    """
    # Use progress tracking if progress manager is provided
    if progress_manager:
        with progress_manager.create_operation_context(
                name="export_execution_log",
                total=3,
                description="Exporting execution log"
        ) as progress:
            try:
                # Load execution log
                registry_data = _load_execution_log(progress_manager)
                progress.update(1, {"status": "log_loaded"})

                # Generate default output path if not provided
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(f"execution_log_export_{timestamp}.{format}")

                # Validate path security
                if not validate_path_security(output_path):
                    error_msg = f"Insecure output path: {output_path}"
                    progress_manager.log_error(error_msg)
                    progress.update(1, {"status": "error", "error": error_msg})
                    raise ExecutionLogError(error_msg)

                # Ensure parent directory exists
                ensure_directory(output_path.parent)
                progress.update(1, {"status": "directory_created"})

                # Export based on format
                if format.lower() == "json":
                    # Export as JSON
                    write_json(registry_data, output_path)
                elif format.lower() == "csv":
                    # Export as CSV
                    import csv

                    # Flatten execution records for CSV export
                    flattened_executions = []
                    for execution in registry_data.get("executions", []):
                        flat_record = {
                            "task_id": execution.get("task_id", ""),
                            "task_type": execution.get("task_type", ""),
                            "task_run_id": execution.get("task_run_id", ""),
                            "timestamp": execution.get("timestamp", ""),
                            "success": execution.get("success", False),
                            "execution_time": execution.get("execution_time", 0),
                            "report_path": execution.get("report_path", ""),
                            "input_files_count": len(execution.get("input_files", [])),
                            "output_files_count": len(execution.get("output_files", [])),
                            "hostname": execution.get("hostname", "")
                        }
                        flattened_executions.append(flat_record)

                    # Write CSV file
                    with open(output_path, "w", newline="", encoding="utf-8") as f:
                        if flattened_executions:
                            fieldnames = flattened_executions[0].keys()
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(flattened_executions)
                else:
                    raise ExecutionLogError(f"Unsupported export format: {format}")

                progress.update(1, {"status": "success", "format": format})
                progress_manager.log_info(f"Exported execution log to {output_path}")
                return output_path

            except Exception as e:
                error_msg = f"Error exporting execution log: {e}"
                progress_manager.log_error(error_msg)
                progress.update(1, {"status": "error", "error": str(e)})
                raise ExecutionLogError(error_msg)
    else:
        try:
            # Load execution log
            registry_data = _load_execution_log()

            # Generate default output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"execution_log_export_{timestamp}.{format}")

            # Validate path security
            if not validate_path_security(output_path):
                logger.error(f"Insecure output path: {output_path}")
                raise ExecutionLogError(f"Insecure output path: {output_path}")

            # Ensure parent directory exists
            ensure_directory(output_path.parent)

            # Export based on format
            if format.lower() == "json":
                # Export as JSON
                write_json(registry_data, output_path)
            elif format.lower() == "csv":
                # Export as CSV
                import csv

                # Flatten execution records for CSV export
                flattened_executions = []
                for execution in registry_data.get("executions", []):
                    flat_record = {
                        "task_id": execution.get("task_id", ""),
                        "task_type": execution.get("task_type", ""),
                        "task_run_id": execution.get("task_run_id", ""),
                        "timestamp": execution.get("timestamp", ""),
                        "success": execution.get("success", False),
                        "execution_time": execution.get("execution_time", 0),
                        "report_path": execution.get("report_path", ""),
                        "input_files_count": len(execution.get("input_files", [])),
                        "output_files_count": len(execution.get("output_files", [])),
                        "hostname": execution.get("hostname", "")
                    }
                    flattened_executions.append(flat_record)

                # Write CSV file
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    if flattened_executions:
                        fieldnames = flattened_executions[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flattened_executions)
            else:
                raise ExecutionLogError(f"Unsupported export format: {format}")

            logger.info(f"Exported execution log to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting execution log: {e}")
            raise ExecutionLogError(f"Failed to export execution log: {e}")


def import_execution_log(input_path: Path, merge: bool = False,
                       progress_manager: Optional[ProgressManagerProtocol] = None) -> bool:
   """
   Import an execution log from a file.

   Args:
       input_path: Path to the file to import
       merge: Whether to merge with existing log (True) or replace (False)
       progress_manager: Progress manager for tracking (optional)

   Returns:
       True if successful, False otherwise

   Raises:
       ExecutionLogError: If import fails
   """
   # Use progress tracking if progress manager is provided
   if progress_manager:
       with progress_manager.create_operation_context(
           name="import_execution_log",
           total=3,
           description="Importing execution log"
       ) as progress:
           try:
               # Validate path security
               if not validate_path_security(input_path):
                   error_msg = f"Insecure input path: {input_path}"
                   progress_manager.log_error(error_msg)
                   progress.update(1, {"status": "error", "error": error_msg})
                   raise ExecutionLogError(error_msg)

               # Load import file
               import_data = read_json(input_path)
               progress.update(1, {"status": "file_loaded"})

               # Basic validation
               if not isinstance(import_data, dict) or "tasks" not in import_data or "executions" not in import_data:
                   error_msg = f"Invalid execution log format in import file: {input_path}"
                   progress_manager.log_error(error_msg)
                   progress.update(1, {"status": "error", "error": error_msg})
                   raise ExecutionLogError(error_msg)

               if merge:
                   # Load existing execution log
                   registry_data = _load_execution_log(progress_manager)

                   # Merge tasks
                   for task_id, task_info in import_data["tasks"].items():
                       registry_data["tasks"][task_id] = task_info

                   # Merge executions, avoiding duplicates
                   existing_task_run_ids = set(e.get("task_run_id") for e in registry_data["executions"])
                   imported_count = 0
                   for execution in import_data["executions"]:
                       if execution.get("task_run_id") not in existing_task_run_ids:
                           registry_data["executions"].append(execution)
                           imported_count += 1

                   # Save merged registry
                   _save_execution_log(registry_data, progress_manager)
                   progress.update(1, {
                       "status": "merged",
                       "imported_count": imported_count,
                       "mode": "merge"
                   })
                   progress_manager.log_info(f"Merged execution log from {input_path}, imported {imported_count} new records")
               else:
                   # Replace existing execution log
                   _save_execution_log(import_data, progress_manager)
                   progress.update(1, {
                       "status": "imported",
                       "execution_count": len(import_data.get("executions", [])),
                       "mode": "replace"
                   })
                   progress_manager.log_info(f"Imported execution log from {input_path}, replacing existing log")

               return True

           except Exception as e:
               error_msg = f"Error importing execution log: {e}"
               progress_manager.log_error(error_msg)
               progress.update(1, {"status": "error", "error": str(e)})
               raise ExecutionLogError(error_msg)
   else:
       try:
           # Validate path security
           if not validate_path_security(input_path):
               logger.error(f"Insecure input path: {input_path}")
               raise ExecutionLogError(f"Insecure input path: {input_path}")

           # Load import file
           import_data = read_json(input_path)

           # Basic validation
           if not isinstance(import_data, dict) or "tasks" not in import_data or "executions" not in import_data:
               logger.error(f"Invalid execution log format in import file: {input_path}")
               raise ExecutionLogError(f"Invalid execution log format in import file: {input_path}")

           if merge:
               # Load existing execution log
               registry_data = _load_execution_log()

               # Merge tasks
               for task_id, task_info in import_data["tasks"].items():
                   registry_data["tasks"][task_id] = task_info

               # Merge executions, avoiding duplicates
               existing_task_run_ids = set(e.get("task_run_id") for e in registry_data["executions"])
               for execution in import_data["executions"]:
                   if execution.get("task_run_id") not in existing_task_run_ids:
                       registry_data["executions"].append(execution)

               # Save merged registry
               _save_execution_log(registry_data)
               logger.info(f"Merged execution log from {input_path}")
           else:
               # Replace existing execution log
               _save_execution_log(import_data)
               logger.info(f"Imported execution log from {input_path}")

           return True

       except Exception as e:
           logger.error(f"Error importing execution log: {e}")
           raise ExecutionLogError(f"Failed to import execution log: {e}")