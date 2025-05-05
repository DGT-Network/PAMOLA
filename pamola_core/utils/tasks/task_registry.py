"""
Task Registry Module for HHR project.

This module provides functionality for registering task executions
and checking task dependencies.
"""

import logging
import time  # Add this import
from pathlib import Path
from typing import List

from pamola_core.utils.io import read_json, write_json, ensure_directory

# Set up logger
logger = logging.getLogger(__name__)

# Pipeline registry file path
PIPELINE_REGISTRY_PATH = Path("configs/pipeline.json")


def initialize_registry():
    """
    Initialize the pipeline registry if it doesn't exist.

    Creates the pipeline registry file with empty data.
    """
    if not PIPELINE_REGISTRY_PATH.exists():
        ensure_directory(PIPELINE_REGISTRY_PATH.parent)

        # Create empty registry
        registry_data = {
            "tasks": {},
            "executions": []
        }

        # Save registry
        write_json(registry_data, PIPELINE_REGISTRY_PATH)
        logger.info(f"Initialized pipeline registry at {PIPELINE_REGISTRY_PATH}")


def register_task_execution(task_id: str, task_type: str, success: bool,
                            execution_time: float, report_path: Path):
    """
    Register a task execution in the pipeline registry.

    Parameters:
    -----------
    task_id : str
        ID of the task
    task_type : str
        Type of the task
    success : bool
        Whether the task executed successfully
    execution_time : float
        Task execution time in seconds
    report_path : Path
        Path to the task report
    """
    # Initialize registry if needed
    initialize_registry()

    try:
        # Load current registry
        registry_data = read_json(PIPELINE_REGISTRY_PATH)

        # Create execution record
        execution = {
            "task_id": task_id,
            "task_type": task_type,
            "timestamp": int(time.time()),
            "success": success,
            "execution_time": execution_time,
            "report_path": str(report_path)
        }

        # Update task in tasks registry
        registry_data["tasks"][task_id] = {
            "task_type": task_type,
            "last_execution": execution["timestamp"],
            "last_status": "success" if success else "failed",
            "last_report_path": str(report_path)
        }

        # Add execution to executions list
        registry_data["executions"].append(execution)

        # Save updated registry
        write_json(registry_data, PIPELINE_REGISTRY_PATH)
        logger.info(f"Registered execution of task {task_id} in pipeline registry")

    except Exception as e:
        logger.error(f"Failed to register task execution: {e}")


def check_task_dependencies(task_id: str, task_type: str, dependencies: List[str]) -> bool:
    """
    Check if dependencies for a task are satisfied.

    Parameters:
    -----------
    task_id : str
        ID of the task
    task_type : str
        Type of the task
    dependencies : List[str]
        List of task IDs that this task depends on

    Returns:
    --------
    bool
        True if all dependencies are satisfied, False otherwise
    """
    # If no dependencies, return True
    if not dependencies:
        return True

    # Initialize registry if needed
    initialize_registry()

    try:
        # Load current registry
        registry_data = read_json(PIPELINE_REGISTRY_PATH)

        # Check each dependency
        for dep_task_id in dependencies:
            # Check if dependency exists in registry
            if dep_task_id not in registry_data["tasks"]:
                logger.error(f"Dependency task {dep_task_id} not found in registry")
                return False

            # Check if dependency was successful
            dep_task = registry_data["tasks"][dep_task_id]
            if dep_task.get("last_status") != "success":
                logger.error(f"Dependency task {dep_task_id} has not completed successfully")
                return False

        # All dependencies satisfied
        return True

    except Exception as e:
        logger.error(f"Failed to check task dependencies: {e}")
        return False