"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Base Task
Description: Foundation class for all task implementations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides the base class that all tasks in the PAMOLA ecosystem
should inherit from. It defines the standard task lifecycle, configuration
handling, and integration with the operation framework.

Key features:
- Standardized initialization, execution, and finalization phases
- Configuration management with priority cascade
- Operation orchestration and result handling
- Progress tracking
- Directory and resource management
- Encryption support with multiple modes (none, simple, age)
- Error handling with configurable strategies
- Dual logging to both project-level and task-specific log files
- Improved operation parameter handling with efficient filtering
- Legacy path support for backward compatibility

This class follows the Facade pattern, delegating specialized responsibilities
to dedicated components while maintaining a simple, unified interface for task developers.

Tasks follow a standard lifecycle:
1. Initialization (load config, create directories, setup logging)
2. Operation configuration (define what operations to execute)
3. Execution (run operations in sequence, handle errors)
4. Finalization (generate reports, cleanup resources)
"""

import inspect
import logging
import sys
import time
from typing import Dict, Any, Optional, List, Union, Tuple, Type, TypeVar

from pamola_core.utils.ops import op_registry  # Import registry module for operations
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.tasks.context_manager import create_task_context_manager
from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager, DependencyMissingError
# Import component managers
from pamola_core.utils.tasks.directory_manager import create_directory_manager
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager
from pamola_core.utils.tasks.execution_log import record_task_execution
from pamola_core.utils.tasks.operation_executor import create_operation_executor
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager
from pamola_core.utils.tasks.task_config import EncryptionMode
from pamola_core.utils.tasks.task_config import load_task_config
from pamola_core.utils.tasks.task_reporting import TaskReporter

# Reserved parameter names that are handled directly by the framework
# These should not be included in operation configuration
RESERVED_OPERATION_PARAMS = {
    "data_source",  # Data source for operation input
    "task_dir",  # Directory for operation artifacts
    "reporter",  # Reporter for logging operation progress
    "progress_tracker",  # Tracker for operation progress

    # Performance parameters
    "parallel_processes",
    "use_vectorization",
    "use_dask",
    "npartitions",
    "chunk_size"
}

# Type variables for better type hinting
T = TypeVar('T', bound='BaseTask')
OpType = TypeVar('OpType', bound=BaseOperation)


class TaskError(Exception):
    """Base exception for task-related errors."""
    pass


class TaskInitializationError(TaskError):
    """
    Exception raised when task initialization fails.

    This can occur due to:
    - Missing or invalid configuration
    - Failed directory creation
    - Logging setup failure
    - Data source initialization problems
    - Path security violations
    """
    pass


class TaskDependencyError(TaskError):
    """
    Exception raised when task dependencies are not satisfied.

    This can occur when previous tasks that this task depends on:
    - Have not been executed
    - Failed during execution
    - Did not produce required outputs
    """
    pass


class TaskExecutionError(TaskError):
    """Base exception for task-related errors."""
    pass


class TaskFinalizationError(TaskError):
    """Base exception for task-related errors."""
    pass


class BaseTask:
    """
    Base class for all tasks in the PAMOLA ecosystem.

    Defines the interface and common functionality for all tasks,
    including lifecycle management, configuration handling, and operation interaction.

    Tasks follow a standard lifecycle:
    1. Initialization (load config, create directories, setup logging)
    2. Operation configuration (define what operations to execute)
    3. Execution (run operations in sequence, handle errors)
    4. Finalization (generate reports, cleanup resources)

    Tasks are responsible for:
    - Managing their configuration
    - Creating and maintaining directory structure
    - Orchestrating operations
    - Collecting and aggregating results
    - Generating reports
    - Error handling
    """

    def __init__(self,
                 task_id: str,
                 task_type: str,
                 description: str,
                 input_datasets: Optional[Dict[str, str]] = None,
                 auxiliary_datasets: Optional[Dict[str, str]] = None,
                 version: str = "1.0.0",
                 encryption_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the task with basic information and defaults.

        Args:
            task_id: Unique identifier for the task. Used to locate configuration and create directories.
            task_type: Type of task (e.g., profiling, anonymization). Used for categorization and logging.
            description: Human-readable description of the task's purpose.
            input_datasets: Dictionary mapping dataset names to file paths. These are the primary inputs.
            auxiliary_datasets: Dictionary mapping auxiliary dataset names to file paths.
                           These are secondary inputs like dictionaries or lookup tables.
            version: Version of the task implementation. Used for tracking and compatibility.
        """
        # Pamola Core task metadata
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.version = version
        self.input_datasets = input_datasets or {}
        self.encryption_keys = encryption_keys or {}
        self.auxiliary_datasets = auxiliary_datasets or {}

        # Basic initialization
        self.start_time = None
        self.execution_time = None
        self.status = "pending"
        self.error_info = None

        # These will be initialized in initialize()
        self.config = None
        self.logger = None
        self.reporter = None

        # Component managers (initialized in initialize())
        self.directory_manager = None
        self.context_manager = None
        self.encryption_manager = None
        self.dependency_manager = None
        self.operation_executor = None
        self.progress_manager = None

        # Data management components
        self.data_source = None
        self.data_writer = None

        # Operation and result tracking
        self.operations = []
        self.results = {}
        self.artifacts = []
        self.metrics = {}

        # Task directory reference (for backward compatibility)
        self.task_dir = None
        self.directories = None

        # Encryption settings (for backward compatibility)
        self.encryption_mode = EncryptionMode.NONE
        self.use_encryption = False

        # Checkpoint settings
        self.enable_checkpoints = False  # Default to False for safety

        # Performance settings (for backward compatibility)
        self.use_vectorization = False
        self.parallel_processes = 1
        self.use_dask = False
        self.npartitions = 1
        self.chunk_size = 100000

        # Checkpoint restoration state
        self._resuming_from_checkpoint = False
        self._restored_checkpoint_name = None
        self._restored_state = None

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values for this task.

        This method should be overridden by subclasses to provide default
        configuration values when no task-specific configuration file exists.

        The default implementation provides basic values based on task metadata.
        Subclasses should extend this to include task-specific default settings.

        Returns:
            Dictionary containing default configuration values.
        """
        return {
            "description": self.description,
            "task_type": self.task_type,
            "version": self.version,
            "dependencies": self.dependencies.copy() if hasattr(self, 'dependencies') else [],
            "continue_on_error": False,
            "use_encryption": self.use_encryption,
            "encryption_mode": "simple",
            "encryption_key_path": "pamola_datasets/configs/keys.db",
            # Default logging configuration
            "log_level": "INFO",
            "log_file": "",
            "task_log_file": "",
            # Default report path (will be overridden by task_config.py)
            "report_path": "",
            # Default for checkpointing
            "enable_checkpoints": False
        }

    def initialize(self, args: Optional[Dict[str, Any]] = None, force_restart: bool = False,
                   enable_checkpoints: Optional[bool] = None) -> bool:
        """
        Initialize the task by loading configuration, creating directories,
        setting up logging, and checking dependencies.

        This is the first phase of the task lifecycle. It prepares the environment
        for task execution.

        Args:
            args: Command line arguments to override configuration.
                 These take highest priority in the configuration cascade.
            force_restart: Whether to ignore existing checkpoints and start fresh.
                          Defaults to False to respect the configuration value.
            enable_checkpoints: Whether to enable checkpoint restoration.
                              If None, uses the value from configuration.

        Returns:
            True if initialization is successful, False otherwise.

        Raises:
            TaskDependencyError: If task dependencies are not satisfied and continue_on_error is False.
        """
        try:
            # Record start time
            self.start_time = time.time()

            # Get default configuration from task
            default_config = self.get_default_config()

            # 1. Load configuration using the task_config module with defaults
            self.config = load_task_config(
                task_id=self.task_id,
                task_type=self.task_type,
                args=args,
                default_config=default_config
            )

            # 2. Set up logging with both project-level and task-specific logs
            # This must happen before any logging calls
            self._setup_logging()

            # Now it's safe to use the logger
            self.logger.debug(
                f"Loaded config: continue_on_error={getattr(self.config, 'continue_on_error', False)}, dependencies={getattr(self.config, 'dependencies', [])}")
            self.logger.info(f"Initializing task: {self.task_id} ({self.task_type}) - {self.description}")

            # 3. Initialize directory manager and create directories
            self.directory_manager = create_directory_manager(
                task_config=self.config,
                logger=self.logger
            )

            # Ensure directories are created
            self.directories = self.directory_manager.ensure_directories()
            self.task_dir = self.directory_manager.get_directory("task")

            # 4. Create reporter with the report path from configuration
            self.reporter = TaskReporter(
                self.task_id,
                self.task_type,
                self.description,
                self.config.report_path
            )

            # 5. Initialize dependency manager using factory method
            self.dependency_manager = self.create_dependency_manager()

            # Check dependencies using centralized method
            self._check_dependencies()

            # 6. Initialize encryption manager
            self.encryption_manager = TaskEncryptionManager(
                task_config=self.config,
                logger=self.logger
            )

            # Initialize encryption
            self.encryption_manager.initialize()

            # Update encryption settings for backward compatibility
            encryption_info = self.encryption_manager.get_encryption_info()
            self.use_encryption = encryption_info["enabled"]
            self.encryption_mode = EncryptionMode.from_string(encryption_info["mode"])
            self.logger.debug(f"Encryption initialized: {self.use_encryption}, mode: {self.encryption_mode.value}")

            # 7. Initialize context manager for task state
            # Use log_directory for centralized checkpoint storage
            self.context_manager = create_task_context_manager(
                task_id=self.task_id,
                task_dir=self.task_dir,
                log_directory=self.config.log_directory if hasattr(self.config, "log_directory") else None
            )

            # Use configuration value if enable_checkpoints is None
            if enable_checkpoints is None:
                enable_checkpoints = getattr(self.config, "enable_checkpoints", False)
                self.logger.debug(f"Using enable_checkpoints={enable_checkpoints} from configuration")

            # Store the enable_checkpoints value for future reference
            self.enable_checkpoints = enable_checkpoints

            # Clear checkpoints if force_restart is enabled or checkpoints are disabled
            if force_restart or not enable_checkpoints:
                if self.context_manager:
                    self.logger.info(
                        f"Clearing checkpoints: force_restart={force_restart}, enable_checkpoints={enable_checkpoints}")
                    self.context_manager.clear_checkpoints()

                    # Explicitly reset checkpoint state
                    self._resuming_from_checkpoint = False
                    self._restored_checkpoint_name = None
                    self._restored_state = None

            # Only check for checkpoints if explicitly enabled and not forcing restart
            elif enable_checkpoints and not force_restart:
                # Check if task can be resumed from checkpoint
                self._resuming_from_checkpoint, self._restored_checkpoint_name = self.context_manager.can_resume_execution()
                if self._resuming_from_checkpoint:
                    self.logger.info(f"Found checkpoint for task {self.task_id}: {self._restored_checkpoint_name}")

                    # Restore execution state
                    try:
                        self._restored_state = self.context_manager.restore_execution_state(
                            self._restored_checkpoint_name)
                        self.logger.info(f"Restored execution state from checkpoint: {self._restored_checkpoint_name}")
                    except Exception as e:
                        self.logger.warning(f"Could not restore from checkpoint: {e}, will perform full execution")
                        self._resuming_from_checkpoint = False
                        self._restored_checkpoint_name = None
                        self._restored_state = None
                else:
                    self.logger.info(f"No valid checkpoints found for task {self.task_id}, starting from beginning")

            # 8. Initialize progress manager
            # Start with 0 operations, will be updated after configure_operations()
            self.progress_manager = create_task_progress_manager(
                task_id=self.task_id,
                task_type=self.task_type,
                logger=self.logger,
                reporter=self.reporter,
                total_operations=0
            )

            # 9. Initialize operation executor
            self.operation_executor = create_operation_executor(
                task_config=self.config,
                logger=self.logger,
                reporter=self.reporter
            )

            # 10. Create data source with input and auxiliary datasets
            self._initialize_data_source()

            # 11. Create data writer for output handling
            self._initialize_data_writer()

            # Update performance settings from configuration for backward compatibility
            self.use_vectorization = getattr(self.config, "use_vectorization", False)
            self.parallel_processes = getattr(self.config, "parallel_processes", 1)
            self.use_dask = getattr(self.config, "use_dask", False)
            self.npartitions = getattr(self.config, "npartitions", 1)
            self.chunk_size = getattr(self.config, "chunk_size", 100000)

            # Add initialization details to reporter
            self.reporter.add_operation(
                name="Initialization",
                status="success",
                details={
                    "task_id": self.task_id,
                    "task_type": self.task_type,
                    "start_time": self.start_time,
                    "directories": {name: str(path) for name, path in self.directories.items()},
                    "encryption": {
                        "enabled": self.use_encryption,
                        "mode": self.encryption_mode.value
                    },
                    "checkpoints": {
                        "enabled": enable_checkpoints,
                        "force_restart": force_restart,
                        "resuming": self._resuming_from_checkpoint
                    }
                }
            )

            self.logger.info(f"Task initialization complete: {self.task_id}")
            return True

        except TaskDependencyError as e:
            # Handle dependency errors specifically
            if self.logger:
                self.logger.error(f"Dependency error in task {self.task_id}: {str(e)}")
            self.error_info = {
                "type": "dependency_error",
                "message": str(e),
                "dependencies": getattr(self.config, "dependencies", [])
            }

            # Check if we should continue despite dependency errors
            if hasattr(self.config, "continue_on_error") and self.config.continue_on_error:
                self.logger.warning(
                    f"Task dependencies not satisfied, but continue_on_error=True - continuing execution")
                # Reset status to pending to reflect that task is continuing despite errors
                self.status = "pending"
                return True
            else:
                self.status = "dependency_error"
                return False
        except Exception as e:
            # Handle other initialization errors
            error_msg = f"Error initializing task {self.task_id}: {str(e)}"
            if self.logger:
                self.logger.exception(error_msg)
            else:
                logging.exception(error_msg)
            self.error_info = {
                "type": "initialization_error",
                "message": str(e)
            }
            self.status = "initialization_error"
            return False

    def create_dependency_manager(self) -> TaskDependencyManager:
        """
        Factory method for the dependency manager.

        This extension point allows subclasses to supply a custom dependency manager
        implementation while retaining the standard interface.

        Returns:
            TaskDependencyManager: A dependency manager instance
        """
        return TaskDependencyManager(task_config=self.config, logger=self.logger)

    def _check_dependencies(self) -> bool:
        """
        Check task dependencies using the dependency manager.

        This method centralizes dependency checking logic and handles
        continue_on_error behavior consistently.

        Returns:
            bool: True if dependencies are satisfied or continue_on_error is enabled,
                  False otherwise (should never return False, as it raises an exception instead)

        Raises:
            TaskDependencyError: If dependencies are not satisfied and continue_on_error is False
        """
        try:
            # No dependencies case - nothing to check
            if not hasattr(self.config, "dependencies") or not self.config.dependencies:
                self.logger.info("No dependencies specified for this task")
                return True

            self.dependency_manager.assert_dependencies_completed()
            self.logger.info("All dependencies satisfied")
            return True
        except (DependencyMissingError, TaskDependencyError) as e:
            # If continue_on_error is enabled, log warning and proceed
            if getattr(self.config, "continue_on_error", False):
                self.logger.warning(f"Dependency error: {e} (continuing due to continue_on_error=True)")
                # Don't update status here - let initialize() handle it for consistency
                return True
            else:
                # Otherwise, propagate the error to fail initialization
                raise TaskDependencyError(f"Task dependencies not satisfied for {self.task_id}: {e}") from e

    def _setup_logging(self) -> None:
        """
        Set up dual logging to both project-level and task-specific logs.

        This method configures logging to write to both the main project log file
        and a task-specific log file in the task's logs directory.
        """
        # Get log paths from configuration, with fallbacks for empty values
        project_log_file = getattr(self.config, "log_file", None)
        task_log_file = getattr(self.config, "task_log_file", None)
        log_level_str = getattr(self.config, "log_level", "INFO")
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)

        # Configure logger
        self.logger = logging.getLogger(f"task.{self.task_id}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers

        # Add console handler (use stderr to avoid conflict with progress bars)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handlers for both project and task logs
        if project_log_file:
            try:
                # Ensure log directory exists
                if hasattr(project_log_file, 'parent'):
                    project_log_file.parent.mkdir(parents=True, exist_ok=True)

                project_handler = logging.FileHandler(project_log_file, encoding='utf-8', mode='w')
                project_handler.setLevel(log_level)
                project_handler.setFormatter(formatter)
                self.logger.addHandler(project_handler)
            except Exception as e:
                # Use standard logging since self.logger might not be fully set up
                logging.warning(f"Failed to create project log file handler: {e}")

        if task_log_file and str(task_log_file) != str(project_log_file):
            try:
                # Ensure log directory exists
                if hasattr(task_log_file, 'parent'):
                    task_log_file.parent.mkdir(parents=True, exist_ok=True)

                task_handler = logging.FileHandler(task_log_file, encoding='utf-8', mode='w')
                task_handler.setLevel(logging.DEBUG)  # Task log gets more detail
                task_handler.setFormatter(formatter)
                self.logger.addHandler(task_handler)
            except Exception as e:
                # Use standard logging since self.logger might not be fully set up
                logging.warning(f"Failed to create task log file handler: {e}")

        # Log setup completion
        logging.debug(f"Logging initialized for task {self.task_id} with level {log_level_str}")

    def _initialize_data_source(self) -> None:
        """
        Initialize the data source with input and auxiliary datasets.

        This method prepares the data source that will be used by operations by:
        1. Creating a new DataSource instance
        2. Adding all input datasets
        3. Adding all auxiliary datasets
        """
        self.data_source = DataSource()

        # Check if we have any input datasets to process
        if not self.input_datasets:
            self.logger.warning("No input datasets defined for task. DataSource initialization may be incomplete.")
            # Don't create a progress bar for empty datasets to avoid hanging
            return

        # Process input datasets with progress tracking
        with self.progress_manager.create_operation_context(
                name="initialize_input_data",
                total=len(self.input_datasets),
                description="Initializing input datasets",
                unit="datasets"
        ) as progress:
            for name, path in self.input_datasets.items():
                try:
                    # Use directory manager to resolve and validate path
                    path_obj = self.directory_manager.normalize_and_validate_path(path)

                    # Add to data source
                    self.data_source.add_file_path(name, path_obj)
                    self.logger.debug(f"Added input dataset: {name} from {path_obj}")

                    # Update progress
                    progress.update(1)
                except Exception as e:
                    self.logger.error(f"Error adding input dataset '{name}': {str(e)}")
                    progress.update(1, {"status": "error"})
                    raise TaskInitializationError(f"Failed to add input dataset '{name}': {str(e)}")

        # Check if we have any auxiliary datasets to process
        if not self.auxiliary_datasets:
            return  # Skip creating empty progress bar for auxiliary datasets

        # Process auxiliary datasets
        with self.progress_manager.create_operation_context(
                name="initialize_auxiliary_data",
                total=len(self.auxiliary_datasets),
                description="Initializing auxiliary datasets",
                unit="datasets"
        ) as progress:
            for name, path in self.auxiliary_datasets.items():
                try:
                    # Use directory manager to resolve and validate path
                    path_obj = self.directory_manager.normalize_and_validate_path(path)

                    # Add to data source
                    self.data_source.add_file_path(name, path_obj)
                    self.logger.debug(f"Added auxiliary dataset: {name} from {path_obj}")

                    # Update progress
                    progress.update(1)
                except Exception as e:
                    self.logger.error(f"Error adding auxiliary dataset '{name}': {str(e)}")
                    progress.update(1, {"status": "error"})
                    raise TaskInitializationError(f"Failed to add auxiliary dataset '{name}': {str(e)}")

        # Check encryption status if enabled
        if self.use_encryption:
            self.encryption_manager.check_dataset_encryption(self.data_source)

    def _initialize_data_writer(self) -> None:
        """
        Initialize the data writer for task outputs.

        The data writer provides a consistent interface for writing operation outputs
        to the appropriate locations within the task directory structure, with
        proper encryption handling.
        """
        # Create data writer instance with encryption configuration
        self.data_writer = DataWriter(
            task_dir=self.task_dir,
            logger=self.logger,
            use_encryption=self.use_encryption,
            encryption_key=self.encryption_manager.get_encryption_context(),
            encryption_mode=self.encryption_mode.value
        )

        self.logger.debug(
            f"Initialized DataWriter for task_dir: {self.task_dir} with encryption: {self.use_encryption}"
        )

    def configure_operations(self) -> None:
        """
        Configure operations to be executed by the task.

        This method should be overridden in subclasses to define the specific
        operations that the task will execute. It should populate the
        self.operations list with operation instances.

        Example implementation:
        ```python
        def configure_operations(self) -> None:
            # Add profiling operation
            self.add_operation(
                "ProfileOperation",
                dataset_name="customers",
                output_prefix="profile_results"
            )

            # Add anonymization operation
            self.add_operation(
                "AnonymizeOperation",
                input_dataset="customers",
                method="k-anonymity",
                k=5,
                quasi_identifiers=["age", "zipcode", "gender"]
            )
        ```

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("Subclasses must implement configure_operations()")

    def _run_operations(self, start_idx: int = 0) -> bool:
        """
        Run operations starting from the specified index.

        This method handles the actual execution of operations, including progress tracking,
        checkpointing, and error handling.

        Args:
            start_idx: Index of the first operation to execute

        Returns:
            True if all operations executed successfully, False otherwise
        """
        # Execute operations using the operation executor
        for i in range(start_idx, len(self.operations)):
            operation = self.operations[i]
            operation_name = operation.name if hasattr(operation, 'name') else f"Operation {i + 1}"

            # Prepare operation parameters
            operation_params = self._prepare_operation_parameters(operation)

            # Use operation executor to run the operation
            try:
                # Execute operation
                result = self.operation_executor.execute_with_retry(
                    operation=operation,
                    params=operation_params
                )

                # Store the result
                self.results[operation_name] = result

                # Register artifacts from the operation
                if hasattr(result, 'artifacts') and result.artifacts:
                    self.artifacts.extend(result.artifacts)

                # Collect metrics from the operation
                if hasattr(result, 'metrics') and result.metrics:
                    self.metrics[operation_name] = result.metrics

                # Check result status
                if result.status == OperationStatus.ERROR:
                    self.logger.error(f"Operation {operation_name} failed: {result.error_message}")

                    # Check if we should continue on error
                    if not self.config.continue_on_error:
                        self.logger.error("Aborting task due to operation failure")
                        self.error_info = {
                            "type": "operation_error",
                            "operation": operation_name,
                            "message": result.error_message
                        }
                        self.status = "operation_error"
                        return False

                # Save checkpoint after each operation
                if self.context_manager:
                    try:
                        self.context_manager.create_automatic_checkpoint(
                            operation_index=i,
                            metrics=self.metrics
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not create checkpoint: {e}")

            except KeyboardInterrupt:
                # Allow keyboard interrupts to propagate up the call stack
                self.logger.info("Keyboard interrupt detected, stopping task execution")
                self.error_info = {
                    "type": "keyboard_interrupt",
                    "operation": operation_name,
                    "message": "Task execution interrupted by user"
                }
                self.status = "interrupted"
                raise  # Re-raise KeyboardInterrupt to ensure it's properly handled

            except Exception as e:
                # Check for KeyboardInterrupt before general exception handling
                if isinstance(e, KeyboardInterrupt):
                    self.logger.info("Keyboard interrupt detected, stopping task execution")
                    self.error_info = {
                        "type": "keyboard_interrupt",
                        "operation": operation_name,
                        "message": "Task execution interrupted by user"
                    }
                    self.status = "interrupted"
                    raise  # Re-raise KeyboardInterrupt

                self.logger.exception(f"Error executing operation {operation_name}: {str(e)}")

                # Check if we should continue on error - centralized error handling
                if not self.config.continue_on_error:
                    self.logger.error("Aborting task due to operation failure")
                    self.error_info = {
                        "type": "exception",
                        "operation": operation_name,
                        "message": str(e)
                    }
                    self.status = "exception"
                    return False

        # If we get here, all operations completed
        self.status = "success"
        return True

    def execute(self) -> bool:
        """
        Execute the task by running operations sequentially, collecting results,
        and generating metrics.

        This is the main phase of the task lifecycle where the actual data processing
        occurs through configured operations.

        Returns:
            True if execution is successful, False otherwise.
        """
        try:
            self.logger.info(f"Executing task: {self.task_id}")

            # Check if we have operations configured
            if not self.operations:
                self.logger.error("No operations configured for this task")
                self.error_info = {
                    "type": "configuration_error",
                    "message": "No operations configured for this task"
                }
                self.status = "configuration_error"
                return False

            # Update total operations in progress manager
            if self.progress_manager:
                self.progress_manager.set_total_operations(len(self.operations))

            # Determine start index for operations based on checkpoint restoration
            start_idx = 0
            if self._resuming_from_checkpoint and self._restored_state is not None:
                # Get the last completed operation index from the restored state
                last_completed_index = self._restored_state.get("operation_index", -1)

                # Skip operations that were already completed
                if last_completed_index >= 0:
                    # Log skipped operations
                    for i in range(last_completed_index + 1):
                        if i < len(self.operations):
                            operation = self.operations[i]
                            operation_name = operation.name if hasattr(operation, 'name') else f"Operation {i + 1}"
                            self.logger.info(f"Skipping already completed operation: {operation_name}")

                            # Update progress manager
                            if self.progress_manager:
                                # Handle different progress manager interfaces
                                if hasattr(self.progress_manager, 'start_operation'):
                                    self.progress_manager.start_operation(operation_name)
                                elif hasattr(self.progress_manager, 'operations_completed'):
                                    self.progress_manager.operations_completed += 1
                                elif hasattr(self.progress_manager, 'complete_operation'):
                                    self.progress_manager.complete_operation(operation_name, success=True)
                                else:
                                    self.logger.debug(
                                        f"Progress manager does not support operation completion tracking for {operation_name}")

                    # Start from the next operation
                    start_idx = last_completed_index + 1
                    self.logger.info(f"Resuming execution from operation index {start_idx}")

            # Run operations starting from the determined index
            return self._run_operations(start_idx)

        except KeyboardInterrupt:
            # Handle keyboard interrupts specifically
            self.logger.info("Keyboard interrupt detected, stopping task execution")
            self.error_info = {
                "type": "keyboard_interrupt",
                "message": "Task execution interrupted by user"
            }
            self.status = "interrupted"
            raise  # Re-raise KeyboardInterrupt to ensure proper shutdown

        except Exception as e:
            # Check for KeyboardInterrupt specifically
            if isinstance(e, KeyboardInterrupt):
                self.logger.info("Keyboard interrupt detected, stopping task execution")
                self.error_info = {
                    "type": "keyboard_interrupt",
                    "message": "Task execution interrupted by user"
                }
                self.status = "interrupted"
                raise  # Re-raise KeyboardInterrupt

            # Handle unexpected execution errors
            error_msg = f"Unhandled error in task execution {self.task_id}: {str(e)}"
            self.logger.exception(error_msg)
            self.error_info = {
                "type": "unhandled_exception",
                "message": str(e)
            }
            self.status = "unhandled_exception"
            return False

    def _prepare_operation_parameters(self, operation: BaseOperation) -> Dict[str, Any]:
        """
        Prepare parameters for an operation, including system parameters and
        operation-specific parameters.

        Args:
            operation: The operation to prepare parameters for

        Returns:
            Dictionary of parameters for the operation
        """
        # Start with empty parameters dictionary
        operation_params = {}

        # Add operation-specific parameters from config, skipping reserved keys
        if hasattr(operation, 'config') and operation.config:
            for key, value in operation.config.to_dict().items():
                if key not in RESERVED_OPERATION_PARAMS:
                    operation_params[key] = value

        # Explicitly add system parameters
        operation_params["data_source"] = self.data_source
        operation_params["task_dir"] = self.task_dir
        operation_params["reporter"] = self.reporter

        # Progress tracking parameters will be added by the operation executor

        # Check if operation supports encryption
        supports_encryption = False
        supports_vectorization = False

        # First check if operation has explicit support flags
        if hasattr(operation, 'supports_encryption'):
            supports_encryption = operation.supports_encryption
        if hasattr(operation, 'supports_vectorization'):
            supports_vectorization = operation.supports_vectorization

        # If not explicitly defined, use parameter inspection
        if not supports_encryption or not supports_vectorization:
            supported_params = self._get_operation_supported_parameters(operation)

            # Check for encryption support via parameters
            if not supports_encryption and "use_encryption" in supported_params:
                supports_encryption = True

            # Check for vectorization support via parameters
            if not supports_vectorization and "use_vectorization" in supported_params:
                supports_vectorization = True

        # Add encryption parameters if supported
        if supports_encryption:
            operation_params["use_encryption"] = self.use_encryption
            # Always provide encryption_context (may be None) for consistent interface
            operation_params["encryption_context"] = (
                self.encryption_manager.get_encryption_context() if self.use_encryption else None
            )
            operation_params["encryption_mode"] = self.encryption_mode.value

        # Add vectorization parameters if supported
        if supports_vectorization:
            operation_params["use_vectorization"] = self.use_vectorization
            operation_params["parallel_processes"] = (
                self.parallel_processes if self.use_vectorization else 1
            )

        return operation_params

    def _get_operation_supported_parameters(self, operation: Union[BaseOperation, Type[BaseOperation]]) -> set:
        """
        Get the set of parameters that an operation's constructor accepts.

        This method uses efficient caching to avoid repeated inspections
        of the same operation classes.

        Args:
            operation: Operation instance or class

        Returns:
            Set of parameter names that the operation accepts
        """
        # Handle both instances and classes
        if isinstance(operation, type):
            cls = operation
        else:
            cls = operation.__class__

        # Use local cache for parameters
        if not hasattr(self, '_operation_parameter_cache'):
            self._operation_parameter_cache = {}

        cache_key = f"{cls.__module__}.{cls.__name__}"

        # Return from cache if available
        if cache_key in self._operation_parameter_cache:
            return self._operation_parameter_cache[cache_key]

        # Create parameter set from constructor signature
        try:
            params = set(inspect.signature(cls.__init__).parameters.keys())
            params.discard('self')  # Remove 'self' parameter
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not inspect parameters for {cls.__name__}: {e}. Using empty set.")
            params = set()

        # Cache and return
        self._operation_parameter_cache[cache_key] = params
        return params

    def finalize(self, success: bool) -> bool:
        """
        Finalize the task by releasing resources, closing files, and registering the execution result.

        This is the last phase of the task lifecycle, responsible for:
        - Recording final status
        - Generating execution report
        - Registering execution in the execution log
        - Releasing resources

        Args:
            success: Whether the task executed successfully.

        Returns:
            True if finalization is successful, False otherwise.
        """
        try:
            # Calculate execution time
            self.execution_time = time.time() - self.start_time

            self.logger.info(f"Finalizing task: {self.task_id} (success: {success})")
            self.logger.info(f"Execution time: {self.execution_time:.2f} seconds")

            # Add final status to reporter
            self.reporter.add_task_summary(
                success=success,
                execution_time=self.execution_time,
                metrics=self.metrics,
                error_info=self.error_info,
                encryption={
                    "enabled": self.use_encryption,
                    "mode": self.encryption_mode.value
                }
            )

            # Summarize artifacts
            for artifact in self.artifacts:
                # Register with reporter if not already tracked
                if artifact not in self.reporter.artifacts:
                    # Check if artifact is encrypted
                    encrypted = False
                    encryption_mode = EncryptionMode.NONE.value

                    if self.use_encryption and hasattr(artifact, 'path'):
                        # Use encryption manager to check if artifact is encrypted
                        try:
                            encrypted = self.encryption_manager.is_file_encrypted(artifact.path)
                            if encrypted:
                                encryption_mode = self.encryption_mode.value
                        except Exception as e:
                            self.logger.debug(f"Error checking encryption status of artifact: {str(e)}")

                    self.reporter.add_artifact(
                        artifact_type=artifact.artifact_type,
                        path=artifact.path,
                        description=artifact.description,
                        category=artifact.category,
                        tags=artifact.tags,
                        metadata={
                            "encrypted": encrypted,
                            "encryption_mode": encryption_mode
                        }
                    )

            # Generate and save report
            try:
                report_path = self.reporter.save_report()
                self.logger.info(f"Task report saved to: {report_path}")
            except Exception as e:
                self.logger.error(f"Error saving task report: {str(e)}")
                self.error_info = {
                    "type": "report_error",
                    "message": str(e)
                }
                self.status = "report_error"
                return False

            # Register task execution - continue even if this fails
            try:
                record_task_execution(
                    task_id=self.task_id,
                    task_type=self.task_type,
                    success=success,
                    execution_time=self.execution_time,
                    report_path=report_path,
                    input_datasets=self.input_datasets,
                    output_artifacts=self.artifacts
                )
            except Exception as e:
                self.logger.error(f"Error recording task execution: {str(e)}")
                self.error_info = {
                    "type": "log_error",
                    "message": str(e)
                }
                self.status = "log_error"
                # Continue with cleanup despite the error

            # Clean up resources using the component managers
            try:
                # Clean temporary resources
                if self.directory_manager:
                    self.directory_manager.clean_temp_directory()

                # Close progress manager
                if self.progress_manager:
                    self.progress_manager.close()

                # Close other managers if they have cleanup methods
                for manager in [self.context_manager, self.encryption_manager]:
                    if manager and hasattr(manager, 'cleanup'):
                        manager.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during resource cleanup: {str(e)}")
                # Don't fail the task for cleanup issues

            return True
        except Exception as e:
            # Handle finalization errors
            error_msg = f"Error finalizing task {self.task_id}: {str(e)}"
            self.logger.exception(error_msg)
            self.error_info = {
                "type": "finalization_error",
                "message": str(e)
            }
            self.status = "finalization_error"
            return False

    def run(self, args: Optional[Dict[str, Any]] = None, force_restart: bool = False,
            enable_checkpoints: Optional[bool] = None) -> bool:
        """
        Run the complete task lifecycle: initialize, configure, execute, finalize.

        This is the main method that orchestrates the entire task, handling
        initialization, operation configuration, execution, and finalization.

        It's typically the only method that needs to be called externally
        to run a task.

        Args:
            args: Command line arguments to override configuration.
            force_restart: If True, ignores existing checkpoints and starts from beginning.
                          Defaults to False to respect the configuration value.
            enable_checkpoints: Whether to enable checkpoint restoration.
                              If None, uses the value from configuration.

        Returns:
            True if the task executed successfully, False otherwise.
        """
        try:
            # Initialize with checkpoint settings
            if not self.initialize(args, force_restart=force_restart, enable_checkpoints=enable_checkpoints):
                self.logger.error(f"Failed to initialize task: {self.task_id}")
                self.finalize(False)
                return False

            # Configure operations
            try:
                # This method should populate self.operations list
                self.configure_operations()

                # Validate that operations were properly configured
                if not self.operations:
                    self.logger.warning("No operations were configured - task may not perform any work")
                else:
                    self.logger.info(f"Configured {len(self.operations)} operations")

                # Update progress manager with total operations
                if self.progress_manager:
                    total_ops = len(self.operations)
                    self.logger.debug(f"Updating progress manager with {total_ops} total operations")
                    self.progress_manager.set_total_operations(total_ops)
            except Exception as e:
                self.logger.exception(f"Error configuring operations: {str(e)}")
                self.error_info = {
                    "type": "configuration_error",
                    "message": str(e)
                }
                self.status = "configuration_error"
                self.finalize(False)
                return False

            # Execute task
            success = self.execute()

            # Finalize
            finalize_result = self.finalize(success)

            # Return combined result (both execution and finalization must succeed)
            return success and finalize_result

        except Exception as e:
            # Handle any unhandled exceptions in the task execution
            error_msg = f"Unhandled error in task {self.task_id}: {str(e)}"
            if self.logger:
                self.logger.exception(error_msg)
            else:
                logging.exception(error_msg)

            # Try to finalize with error status
            try:
                self.finalize(False)
            except Exception:
                # If finalization itself fails, just log and continue
                if self.logger:
                    self.logger.exception("Error during finalization after task failure")
                else:
                    logging.exception("Error during finalization after task failure")

            return False

    def add_operation(self, operation_class: Union[str, Type[BaseOperation]], **kwargs) -> bool:
        """
        Add an operation to the task's execution queue.

        This method creates an operation instance and adds it to the
        list of operations to be executed. Operations are executed
        in the order they are added.

        The operation can be specified either by class name (string) or
        by the actual class. Parameters for the operation are passed
        as keyword arguments.

        Args:
            operation_class: Name of the operation class or the class itself.
                        If a string is provided, the operation is loaded from the registry.
            **kwargs: Parameters for the operation constructor.

        Returns:
            True if operation was added successfully, False otherwise.
        """
        try:
            # 1. Filter out Reserved Keys
            filtered_kwargs = {k: v for k, v in kwargs.items()
                               if k not in RESERVED_OPERATION_PARAMS}

            # 2. Get the operation class
            if isinstance(operation_class, str):
                op_cls = op_registry.get_operation_class(operation_class)
            else:
                op_cls = operation_class

            # 3. Add infrastructure flags only if supported by the operation
            if op_cls is not None:
                # Get the supported parameters for this operation
                supported_params = self._get_operation_supported_parameters(op_cls)

                # Infrastructure-level flags that can be overridden via kwargs
                infra_flags = {
                    'use_encryption': getattr(self, 'use_encryption', None),
                    'encryption_mode': getattr(self, 'encryption_mode', None),
                    'use_vectorization': getattr(self, 'use_vectorization', None),
                    'parallel_processes': getattr(self, 'parallel_processes', None),
                    'use_dask': getattr(self, 'use_dask', None),
                    'npartitions': getattr(self, 'npartitions', None),
                    'chunk_size': getattr(self, 'chunk_size', None),
                }

                for key, default_value in infra_flags.items():
                    if key in supported_params:
                        # Prefer the value from kwargs if provided; fall back to default from self
                        value = kwargs.get(key, default_value)
                        # If the value is an Enum, extract its actual value
                        if hasattr(value, "value"):
                            value = value.value
                        if value is not None:
                            filtered_kwargs[key] = value 
            elif isinstance(operation_class, str):
                # If the class is not found, log it, but give the factory a chance (it has lazy-load)
                self.logger.warning(f"Operation class {operation_class} not found in registry, attempting lazy load")

            # 4. Create Instance
            operation = (
                op_registry.create_operation_instance(operation_class, **filtered_kwargs)
                if isinstance(operation_class, str)
                else op_cls(**filtered_kwargs)
            )

            if not operation:
                self.logger.error(f"Failed to create operation: {operation_class}")
                return False

            # 5. Make Registration
            self.operations.append(operation)
            self.logger.info(f"Added operation: {operation.__class__.__name__}")

            # 6. Update progress manager
            if self.progress_manager:
                self.progress_manager.increment_total_operations()

            return True

        except Exception as exc:
            self.logger.exception(f"Error adding operation {operation_class}: {exc}")
            return False

    def get_results(self) -> Dict[str, OperationResult]:
        """
        Get the results of all operations executed by the task.

        This method returns the results of all operations that
        were executed by the task, mapped by operation name.

        Returns:
            Dictionary mapping operation names to their results.
        """
        return self.results

    def get_artifacts(self) -> List[Any]:
        """
        Get all artifacts produced by the task.

        Artifacts are files or other resources produced by the task
        and its operations. They can include output datasets, visualizations,
        dictionaries, and other files.

        Returns:
            List of artifact objects with metadata.
        """
        return self.artifacts

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics collected by the task.

        Metrics include statistical information, performance measurements,
        and other quantitative data produced by the task and its operations.

        Returns:
            Dictionary containing metrics organized by category.
        """
        return self.metrics

    def get_execution_status(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Get the execution status and error information.

        This method returns the current status of the task execution
        and detailed error information if an error occurred.

        Returns:
            Tuple containing:
                - Status string (e.g., "pending", "success", "error")
                - Error information dictionary (or None if no error)
        """
        return self.status, self.error_info

    def get_encryption_info(self) -> Dict[str, Any]:
        """
        Get information about the task's encryption settings.

        Returns:
            Dictionary with encryption information including mode and status.
        """
        if self.encryption_manager:
            return self.encryption_manager.get_encryption_info()
        else:
            return {
                "enabled": self.use_encryption,
                "mode": self.encryption_mode.value,
                "key_available": False
            }

    def get_checkpoint_status(self) -> Dict[str, Any]:
        """
        Get information about the checkpoint status of the task.

        Returns:
            Dictionary with checkpoint information including restoration status.
        """
        return {
            "resuming_from_checkpoint": self._resuming_from_checkpoint,
            "checkpoint_name": self._restored_checkpoint_name,
            "has_restored_state": self._restored_state is not None,
            "operation_index": self._restored_state.get("operation_index", -1) if self._restored_state else -1
        }

    def __enter__(self) -> 'BaseTask':
        """
        Enter the context manager.

        This method allows the task to be used as a context manager
        with the 'with' statement, ensuring proper finalization even
        if an exception occurs.

        Returns:
            The task instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context manager.

        This method is called when exiting the 'with' statement context.
        It ensures the task is properly finalized even if an exception occurs.

        Args:
            exc_type: The type of the exception if one occurred, or None
            exc_val: The exception instance if one occurred, or None
            exc_tb: The traceback if an exception occurred, or None

        Returns:
            False to allow exception propagation, True to suppress exceptions.
        """
        if exc_type is not None:
            # An exception occurred during the with block
            self.logger.error(f"Exception during task execution: {exc_type.__name__}: {exc_val}")
            self.error_info = {
                "type": "context_exception",
                "message": str(exc_val),
                "exception_type": exc_type.__name__
            }
            self.status = "context_exception"
            self.finalize(False)
            # Return False to propagate the exception
            return False

        # No exception occurred, but make sure we finalize if not already done
        if self.status == "pending":
            self.finalize(True)

        return False  # False means don't suppress exceptions