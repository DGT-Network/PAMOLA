"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Framework
Description: Support for task creation and orchestration
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This package provides the foundation for creating and executing tasks
in the PAMOLA ecosystem, handling configurations, reporting, and
integration with pamola core operations.

Key features:
- Task lifecycle management (initialization, execution, finalization)
- Configuration loading and management with priority cascade
- Task registration and discovery
- Standardized task reporting
- Execution history tracking
- Integration with operation framework
- Path resolution and directory management
"""

# Import and expose key classes and functions
from pamola_core.utils.tasks.base_task import BaseTask, TaskError, TaskInitializationError, TaskExecutionError, TaskFinalizationError, TaskDependencyError
from pamola_core.utils.tasks.task_config import TaskConfig, load_task_config, EncryptionMode, validate_path_security
from pamola_core.utils.tasks.task_registry import register_task_class, get_task_class, list_registered_tasks, create_task_instance, discover_task_classes, register_discovered_tasks
from pamola_core.utils.tasks.task_reporting import TaskReporter, ArtifactGroup, ReportingError
from pamola_core.utils.tasks.execution_log import initialize_execution_log, record_task_execution, get_task_execution_history, find_latest_execution, find_task_by_output
from pamola_core.utils.tasks.task_utils import (
    create_task_directories,
    prepare_data_source_from_paths,
    format_execution_time,
    get_artifact_path,
    find_previous_output,
    find_task_report,
    format_error_for_report,
    ensure_secure_directory,
    extract_previous_output_info
)

# Package version
__version__ = "1.0.0"

# Public exports
__all__ = [
    # Base classes
    'BaseTask',
    'TaskError',
    'TaskInitializationError',
    'TaskExecutionError',
    'TaskFinalizationError',
    'TaskDependencyError',

    # Configuration
    'TaskConfig',
    'load_task_config',
    'EncryptionMode',
    'validate_path_security',

    # Task registry
    'register_task_class',
    'get_task_class',
    'list_registered_tasks',
    'create_task_instance',
    'discover_task_classes',
    'register_discovered_tasks',

    # Reporting
    'TaskReporter',
    'ArtifactGroup',
    'ReportingError',

    # Execution logging
    'initialize_execution_log',
    'record_task_execution',
    'get_task_execution_history',
    'find_latest_execution',
    'find_task_by_output',

    # Utilities
    'create_task_directories',
    'prepare_data_source_from_paths',
    'format_execution_time',
    'get_artifact_path',
    'find_previous_output',
    'find_task_report',
    'format_error_for_report',
    'ensure_secure_directory',
    'extract_previous_output_info'
]