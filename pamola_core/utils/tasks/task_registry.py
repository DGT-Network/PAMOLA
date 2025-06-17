"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Registry
Description: In-memory task type registration and discovery
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for registering task classes,
discovering available tasks, and checking task dependencies.

Key features:
- Task class registration and discovery
- Task dependency validation
- Task instantiation by type
- Task metadata extraction and validation
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Any, Optional, List, Type

from pamola_core.utils.tasks.execution_log import find_latest_execution

# Import BaseTask indirectly to avoid circular imports
# Tasks will inherit from this, but we don't need the actual class here
# We'll use string type annotations and runtime type checking

# Set up logger
logger = logging.getLogger(__name__)

# Global registry of task classes
_task_classes: Dict[str, Type] = {}


class TaskRegistryError(Exception):
    """Exception raised for task registry errors."""
    pass


def register_task_class(task_id: str, task_class: Type) -> bool:
    """
    Register a task class by ID.

    Args:
        task_id: ID of the task
        task_class: Task class to register

    Returns:
        True if registration was successful, False otherwise

    Raises:
        TaskRegistryError: If registration fails
    """
    try:
        # Check if class is a valid task class (has required attributes)
        if not _validate_task_class(task_class):
            logger.warning(f"Invalid task class for {task_id}: missing required attributes")
            return False

        # Register task class
        _task_classes[task_id] = task_class
        logger.debug(f"Registered task class for {task_id}: {task_class.__name__}")

        return True
    except Exception as e:
        logger.error(f"Error registering task class: {str(e)}")
        raise TaskRegistryError(f"Failed to register task class: {str(e)}")


def get_task_class(task_id: str) -> Optional[Type]:
    """
    Get a task class by ID.

    Args:
        task_id: ID of the task

    Returns:
        Task class or None if not found

    Raises:
        TaskRegistryError: If lookup fails
    """
    try:
        # Return task class from registry
        return _task_classes.get(task_id)
    except Exception as e:
        logger.error(f"Error getting task class: {str(e)}")
        raise TaskRegistryError(f"Failed to get task class: {str(e)}")


def list_registered_tasks() -> Dict[str, Dict[str, Any]]:
    """
    List all registered task types with their metadata.

    Returns:
        Dictionary mapping task IDs to task metadata

    Raises:
        TaskRegistryError: If listing fails
    """
    try:
        # Build dictionary of task metadata
        result = {}
        for task_id, task_class in _task_classes.items():
            result[task_id] = get_task_metadata(task_class)
        return result
    except Exception as e:
        logger.error(f"Error listing registered tasks: {str(e)}")
        raise TaskRegistryError(f"Failed to list registered tasks: {str(e)}")


def create_task_instance(task_id: str, **kwargs) -> Optional[Any]:
    """
    Create a task instance by ID.

    Args:
        task_id: ID of the task
        **kwargs: Arguments to pass to the task constructor

    Returns:
        Task instance or None if task class not found

    Raises:
        TaskRegistryError: If instantiation fails
    """
    try:
        # Get task class from registry
        task_class = get_task_class(task_id)
        if task_class is None:
            logger.warning(f"Task class not found for {task_id}")
            return None

        # Create task instance
        task_instance = task_class(**kwargs)
        logger.debug(f"Created task instance for {task_id}")

        return task_instance
    except Exception as e:
        logger.error(f"Error creating task instance: {str(e)}")
        raise TaskRegistryError(f"Failed to create task instance: {str(e)}")


def discover_task_classes(package_paths: Optional[List[str]] = None,
                          recursive: bool = True) -> Dict[str, Type]:
    """
    Discover task classes in specified packages.

    This function scans Python packages for classes that inherit from BaseTask
    and have a task_id attribute.

    Args:
        package_paths: List of package paths to scan (e.g., ["mypackage.tasks"])
        recursive: Whether to scan subpackages recursively

    Returns:
        Dictionary mapping task IDs to task classes

    Raises:
        TaskRegistryError: If discovery fails
    """
    try:
        # Default to scanning scripts directory if not specified
        if package_paths is None:
            package_paths = ["scripts"]

        discovered_tasks = {}

        # Process each package path
        for package_path in package_paths:
            try:
                # Import package
                package = importlib.import_module(package_path)

                # Get package directory
                if hasattr(package, "__path__"):
                    package_dir = package.__path__
                else:
                    logger.warning(f"Package {package_path} does not have a __path__ attribute")
                    continue

                # Walk through package
                for _, module_name, is_pkg in pkgutil.walk_packages(package_dir, package.__name__ + "."):
                    # Skip packages if not recursive
                    if is_pkg and not recursive:
                        continue

                    try:
                        # Import module
                        module = importlib.import_module(module_name)

                        # Scan module for task classes
                        for name, obj in module.__dict__.items():
                            if (inspect.isclass(obj) and
                                    obj.__module__ == module.__name__ and
                                    _is_task_class(obj)):

                                # Get task ID
                                task_id = _get_task_id(obj)
                                if task_id:
                                    discovered_tasks[task_id] = obj
                                    logger.debug(f"Discovered task class: {task_id} ({obj.__name__})")

                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Error importing module {module_name}: {str(e)}")

            except ImportError as e:
                logger.warning(f"Error importing package {package_path}: {str(e)}")

        return discovered_tasks

    except Exception as e:
        logger.error(f"Error discovering task classes: {str(e)}")
        raise TaskRegistryError(f"Failed to discover task classes: {str(e)}")


def register_discovered_tasks(package_paths: Optional[List[str]] = None,
                              recursive: bool = True) -> int:
    """
    Discover and register task classes in specified packages.

    Args:
        package_paths: List of package paths to scan
        recursive: Whether to scan subpackages recursively

    Returns:
        Number of tasks registered

    Raises:
        TaskRegistryError: If registration fails
    """
    try:
        # Discover task classes
        discovered_tasks = discover_task_classes(package_paths, recursive)

        # Register discovered tasks
        count = 0
        for task_id, task_class in discovered_tasks.items():
            if register_task_class(task_id, task_class):
                count += 1

        logger.info(f"Registered {count} discovered tasks")
        return count

    except Exception as e:
        logger.error(f"Error registering discovered tasks: {str(e)}")
        raise TaskRegistryError(f"Failed to register discovered tasks: {str(e)}")


def get_task_metadata(task_class: Type) -> Dict[str, Any]:
    """
    Extract metadata from a task class.

    Extracts metadata without creating an instance of the class to avoid
    expensive initialization.

    Args:
        task_class: Task class to extract metadata from

    Returns:
        Dictionary with task metadata

    Raises:
        TaskRegistryError: If metadata extraction fails
    """
    try:
        # Get task ID without creating an instance
        task_id = _get_task_id(task_class)

        # Extract metadata from class attributes and docstring
        metadata = {
            "task_id": task_id,
            "task_type": getattr(task_class, "task_type", "unknown"),
            "description": getattr(task_class, "description", ""),
            "version": getattr(task_class, "version", "1.0.0"),
            "class_name": task_class.__name__,
            "module": task_class.__module__,
            "dependencies": getattr(task_class, "dependencies", []),
            "author": getattr(task_class, "author", "unknown")
        }

        # Extract additional metadata from docstring if available
        if task_class.__doc__:
            doc_lines = task_class.__doc__.strip().split('\n')

            # Look for metadata in docstring (e.g., @author, @version, etc.)
            for line in doc_lines:
                line = line.strip()
                if line.startswith('@'):
                    try:
                        tag, value = line[1:].split(':', 1)
                        metadata[tag.strip().lower()] = value.strip()
                    except ValueError:
                        # Skip malformatted tags
                        pass

        return metadata

    except Exception as e:
        logger.error(f"Error extracting task metadata: {str(e)}")
        raise TaskRegistryError(f"Failed to extract task metadata: {str(e)}")


def check_task_dependencies(task_id: str, task_type: str, dependencies: List[str]) -> bool:
    """
    Check if dependencies for a task are satisfied.

    This function checks the execution log to see if all dependency tasks
    have been executed successfully.

    Args:
        task_id: ID of the task
        task_type: Type of the task
        dependencies: List of task IDs that this task depends on

    Returns:
        True if all dependencies are satisfied, False otherwise

    Raises:
        TaskRegistryError: If dependency check fails
    """
    # If no dependencies, return True
    if not dependencies:
        return True

    try:
        # Check each dependency
        for dep_task_id in dependencies:
            # Get latest execution of dependency
            latest_execution = find_latest_execution(dep_task_id, success_only=True)

            # Check if dependency has been executed successfully
            if not latest_execution:
                logger.warning(f"Dependency task {dep_task_id} has not been executed successfully")
                return False

        # All dependencies satisfied
        return True

    except Exception as e:
        logger.error(f"Error checking task dependencies: {str(e)}")
        # Return False for safety, but don't re-raise to allow task to handle dependency issues
        return False


def _is_task_class(cls: Type) -> bool:
    """
    Check if a class is a task class.

    A task class must:
    1. Have a 'task_id' attribute or method to be instantiated with a task_id
    2. Inherit from a class named 'BaseTask'

    Args:
        cls: Class to check

    Returns:
        True if the class is a task class, False otherwise
    """
    # Check for BaseTask in class hierarchy
    for base in cls.__mro__:
        if base.__name__ == "BaseTask":
            return True

    # Check constructor signature (for adapting interfaces)
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # Check for required parameters
        if ("task_id" in params and
                "task_type" in params and
                "description" in params):
            return True
    except (ValueError, TypeError):
        pass

    return False


def _get_task_id(cls: Type) -> Optional[str]:
    """
    Get the task ID from a task class.

    Args:
        cls: Task class

    Returns:
        Task ID or None if not found
    """
    # Check for task_id class attribute
    if hasattr(cls, "task_id") and isinstance(cls.task_id, str):
        return cls.task_id

    # Try to extract from class name (e.g., MyTask -> my_task)
    import re
    if cls.__name__.endswith("Task"):
        # Extract task name and convert to snake_case
        task_name = cls.__name__[:-4]  # Remove "Task" suffix

        # Convert CamelCase to snake_case
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', task_name).lower()

        return snake_case

    # Check for Task_X pattern in class name
    match = re.match(r't_\w+', cls.__name__.lower())
    if match:
        return match.group(0)

    return None


def _validate_task_class(cls: Type) -> bool:
    """
    Validate that a class meets the requirements for a task class.

    Args:
        cls: Class to validate

    Returns:
        True if the class is valid, False otherwise
    """
    # Check that it's a task class
    if not _is_task_class(cls):
        return False

    # Check that it has required constructor parameters
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # Check for required parameters
        required_params = ["task_id", "task_type", "description"]
        for param in required_params:
            if param not in params:
                return False
    except (ValueError, TypeError):
        return False

    # Check for required methods more efficiently
    required_methods = ["configure_operations", "run"]
    for method_name in required_methods:
        if not hasattr(cls, method_name) or not callable(getattr(cls, method_name)):
            return False

    # Check if the class actually inherits from BaseTask instead of just having a similar interface
    for base in cls.__mro__:
        if base.__name__ == "BaseTask":
            return True

    # If it just has a compatible interface but doesn't actually inherit from BaseTask
    # Check for more required methods that are part of the task lifecycle
    additional_methods = ["initialize", "execute", "finalize"]
    for method_name in additional_methods:
        if not hasattr(cls, method_name) or not callable(getattr(cls, method_name)):
            return False

    return True