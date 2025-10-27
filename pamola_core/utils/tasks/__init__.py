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
from pamola_core.utils.tasks.task_config import TaskConfig, load_task_config, EncryptionMode, validate_path_security
from pamola_core.utils.tasks.task_reporting import TaskReporter, ArtifactGroup, ReportingError


# Package version
__version__ = "1.0.0"

# Public exports
__all__ = [
    # Configuration
    'TaskConfig',
    'load_task_config',
    'EncryptionMode',
    'validate_path_security',

    # Reporting
    'TaskReporter',
    'ArtifactGroup',
    'ReportingError',
]